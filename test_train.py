
# filepath: /home/tgx/data/projects/pruned_attention/test_train.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from prune import apply_compression_to_model
from peft import PeftModel

BASE_MODEL_ID = "/home/tgx/data/models/Llama-3-8B-Instruct"
DISTILLED_DIR = "./compressed_llama_distilled_topk"
adapter_model_path = "./lora_all_mixed/final_checkpoint"


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager"
)
model = apply_compression_to_model(base_model)

small_state = torch.load(f"./compressed_llama_distilled_topk_wo_sink/small_attn_weights.pt", map_location="cpu")
base_state = model.state_dict()

for name, param in small_state.items():
    name = 'model.' + name
    if name in base_state and base_state[name].shape == param.shape:
        base_state[name] = param
        
model.load_state_dict(base_state)

lora_model = PeftModel.from_pretrained(model, adapter_model_path)
lora_model.eval()
with open("prompt1.txt", "r", encoding="utf-8") as f:
    prompt = f.read()

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")


output = model.generate(
    **inputs,
    max_new_tokens=256,
    use_cache=False,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.2,
)

input_len = inputs["input_ids"].shape[1]          
generated_ids = output[0][input_len:]            
generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

print(generated_text)