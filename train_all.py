import os
import torch
from transformers import (
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)
from datasets import load_dataset, concatenate_datasets
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType,
    PeftModel
)

from transformers import AutoModelForCausalLM
from prune import apply_compression_to_model

model_name = "/home/tgx/data/models/Llama-3-8B-Instruct"
output_dir = "./lora_all_mixed" 
block_size = 512 


#resume_from_checkpoint_path = '/home/tgx/data/projects/gpt2_sim/tcd_llama/tcd-llama-ptb-wiki-lora-packed-qkv-v3/final_checkpoint'
#resume_from_checkpoint_path = '/home/tgx/data/projects/gpt2_sim/tcd_llama/tcd-llama-5-datasets-mixed-lora-128rank/final_checkpoint'
resume_from_checkpoint_path = None
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager" 
)

model = apply_compression_to_model(base_model) 
DISTILLED_DIR = "./compressed_llama_distilled_topk_wo_sink"

small_state = torch.load(f"{DISTILLED_DIR}/small_attn_weights.pt", map_location="cpu")
base_state = base_model.state_dict()

for name, param in small_state.items():
    if name in base_state and base_state[name].shape == param.shape:
        base_state[name] = param

model.load_state_dict(base_state)

for name, param in model.named_parameters():
    if "small" in name:
        param.requires_grad = False
        print(f"freezed: {name}")


if resume_from_checkpoint_path is None:
    lora_config = LoraConfig(
        r=128,
        lora_alpha=256,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"] 
    )
    model = get_peft_model(model, lora_config)
else:
    print(f" from {resume_from_checkpoint_path} loading.")
    model = PeftModel.from_pretrained(model, resume_from_checkpoint_path, is_trainable=True)
print("Trainable parameters:")
model.print_trainable_parameters()



print("loading Tulu-3 subsets...")
tulu_subset = load_dataset(
    "allenai/tulu-3-sft-mixture", 
    data_files='data/train-00000-of-00006.parquet', 
    split='train',
    verification_mode="no_checks"
).select(range(15000))

print("loading Orca-AgentInstruct subsets...")
orca_subset = load_dataset(
    "mlabonne/orca-agentinstruct-1M-v1-cleaned", 
    data_files='data/train-00000-of-00010.parquet', 
    split='train',
    verification_mode="no_checks"
).select(range(15000))

print("loading c4 subsets...")
c4_files = {
    "train": "en/c4-train.00000-of-01024.json.gz",
    "validation": "en/c4-validation.00000-of-00008.json.gz"
}
c4_subset = load_dataset("allenai/c4", data_files=c4_files)
c4_subset["train"] = c4_subset["train"].select(range(15000))

print("loading PTB...")
ptb_dataset = load_dataset('ptb_text_only', 'penn_treebank')
ptb_dataset = ptb_dataset.rename_column("sentence", "text")

print("loading WikiText-2...")
wiki_dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

def format_instruction_as_text(example):
    text = ""
    for message in example['messages']:
        role = message['role']
        content = message['content']
        text += f"{role}\n{content}\n"
    return {"text": text}

tulu_formatted = tulu_subset.map(format_instruction_as_text, remove_columns=tulu_subset.column_names)
orca_formatted = orca_subset.map(format_instruction_as_text, remove_columns=orca_subset.column_names)

print("concating...")
train_dataset = concatenate_datasets([
    tulu_formatted, 
    orca_formatted,
    c4_subset["train"],
    ptb_dataset["train"], 
    wiki_dataset["train"],
])
eval_dataset = concatenate_datasets([
    c4_subset["validation"],
    ptb_dataset["validation"], 
    wiki_dataset["validation"]
])

train_dataset = train_dataset.shuffle(seed=42)
print(f"train sample: {len(train_dataset)}")
print(f"eval sample: {len(eval_dataset)}")


def tokenize_function(examples):
    text_with_eos = [s + tokenizer.eos_token for s in examples['text'] if len(s) > 0]
    return tokenizer(text_with_eos)

def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

print("Tokenizing datasets...")

tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names, num_proc=os.cpu_count()//2)
tokenized_eval = eval_dataset.map(tokenize_function, batched=True, remove_columns=eval_dataset.column_names, num_proc=os.cpu_count()//2)


print("Grouping and packing datasets...")
lm_train_dataset = tokenized_train.map(group_texts, batched=True, num_proc=os.cpu_count()//2)
lm_eval_dataset = tokenized_eval.map(group_texts, batched=True, num_proc=os.cpu_count()//2)

print(f"after concat, training sample: {len(lm_train_dataset)}")
print(f"after concat, eval sample: {len(lm_eval_dataset)}")

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    learning_rate=2e-4,
    num_train_epochs=2,
    
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    
    bf16=True,
    
    logging_steps=10,
    save_steps=100,
    save_total_limit=3,
    
    #eval_strategy="steps",
    #eval_steps=10,
    
    report_to="tensorboard",
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_train_dataset,
    #eval_dataset=lm_eval_dataset,
    data_collator=data_collator,
)

print("="*30)
print("     beinging LoRA fintune (on 5 Mixed Datasets - Packed Mode)     ")
print("="*30)
trainer.train()

final_model_dir = os.path.join(output_dir, "final_checkpoint")
print(f"trained finished in: {final_model_dir}")
model.save_pretrained(final_model_dir)
tokenizer.save_pretrained(final_model_dir)

print("\nfinetuned finished!")
print(f"model saved in: {final_model_dir}")