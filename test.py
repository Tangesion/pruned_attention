from transformers import AutoTokenizer, LlamaForCausalLM
from prune import apply_compression_to_model

model_name = "/home/tgx/data/models/Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
#apply_compression_to_model(model)


with open("prompt1.txt", "r", encoding="utf-8") as f:
    prompt = f.read()

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")


output = model.generate(
    **inputs,
    max_new_tokens=256,
    use_cache=True,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.2,
)


input_len = inputs["input_ids"].shape[1]         
generated_ids = output[0][input_len:]            
generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

print(generated_text)