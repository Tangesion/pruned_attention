
# filepath: /home/tgx/data/projects/pruned_attention/test_train.py
import torch
import os
import sys
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList, LlamaForCausalLM
from peft import PeftModel

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#from src.pruned_attention.modeling_llama import LlamaForCausalLM
from src.pruned_attention.calibration import apply_compression_to_model
from src.pruned_attention.attention import KVWithSmallKCache

BASE_MODEL_ID = "/home/tgx/data/models/Llama-3-8B-Instruct"
DISTILLED_DIR = "./compressed_llama_distilled_topk"
adapter_model_path = "./lora_all_mixed/final_checkpoint"


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
base_model = LlamaForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager"
)
config = base_model.config
model = apply_compression_to_model(base_model)

small_state = torch.load(
    "./compressed_llama_distilled_topk_wo_sink/small_attn_weights.pt",
    map_location="cpu"
)
base_state = model.state_dict()

for name, param in small_state.items():
    name = "model." + name
    if name in base_state and base_state[name].shape == param.shape:
        base_state[name] = param

model.load_state_dict(base_state)
lora_model = model
#lora_model = PeftModel.from_pretrained(model, adapter_model_path)
lora_model.eval()

with open("prompt1.txt", "r", encoding="utf-8") as f:
    prompt = f.read()

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
input_len = inputs["input_ids"].shape[1]


class StreamTokens(StoppingCriteria):
    def __init__(self, tokenizer, input_len):
        super().__init__()
        self.tokenizer = tokenizer
        self.input_len = input_len
        self.last_printed = input_len  

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # input_ids: [batch, seq]
        seq = input_ids[0]
        cur_len = seq.shape[0]

        if cur_len > self.last_printed:
            new_tokens = seq[self.last_printed:cur_len]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            print(text, end="", flush=True)
            self.last_printed = cur_len

        return False


streamer = StreamTokens(tokenizer, input_len)
kv_cache = KVWithSmallKCache(config=config)
output = lora_model.generate(
    **inputs,
    max_new_tokens=256,
    use_cache=True,
    do_sample=True,
    past_key_values=kv_cache,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.2,
    stopping_criteria=StoppingCriteriaList([streamer]),
)

print()