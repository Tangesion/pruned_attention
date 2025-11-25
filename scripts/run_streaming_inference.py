import torch
import os
import sys
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pruned_attention.modeling_llama import LlamaForCausalLM
from src.pruned_attention.calibration import apply_compression_to_model
model_name = "/home/tgx/data/models/Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto", attn_implementation="eager")
#apply_compression_to_model(model)


with open("prompt1.txt", "r", encoding="utf-8") as f:
    prompt = f.read()

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")


input_len = inputs["input_ids"].shape[1]


# --- 定义 streaming stopping criteria ---
class StreamTokens(StoppingCriteria):
    def __init__(self, tokenizer, input_len):
        super().__init__()
        self.tokenizer = tokenizer
        self.input_len = input_len
        self.last_printed = input_len  # 已经打印到的 token 位置

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # input_ids: [batch, seq]
        seq = input_ids[0]
        # 当前已经生成到的位置
        cur_len = seq.shape[0]

        # 有新 token，就增量 decode & 打印
        if cur_len > self.last_printed:
            new_tokens = seq[self.last_printed:cur_len]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            print(text, end="", flush=True)
            self.last_printed = cur_len

        # 返回 False 表示不要提前停止，让 generate 自己根据 max_new_tokens 停
        return False


streamer = StreamTokens(tokenizer, input_len)

output = model.generate(
    **inputs,
    max_new_tokens=256,
    use_cache=True,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.2,
    stopping_criteria=StoppingCriteriaList([streamer]),
)

# 最后换行一下
print()