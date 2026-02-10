# PPL测试
## 1. 目标
验证压缩后的PPL不显著上升，且优于其他方法。
## 2. 对比方法
* Full attention
* h2o attention
* slide attention
* ours(int4)
* ours(bf16)
## 3. 数据集
选择pg-19数据集
## 4. 作图要求
折线图 x=压缩比例，y=ppl，legend为不同的方法，topk ratio比例从10%，增长到80%，区间为10%。 文本长度为16k
## 5. 参考代码
* ppl测试naive实验目前如下
```python
import torch
import os
import sys
import argparse
from transformers import AutoTokenizer, LlamaForCausalLM
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from peft import PeftModel
import numpy as np

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pruned_attention.calibration import apply_compression_to_model
from src.h2o_attention.attention import convert_kvcache_llama_heavy_recent
from src.pruned_attention.attention import KVWithSmallKCache


torch.no_grad()
def ppl_eval_parallel(model, tokenizer, args=None):
    """
    Parallel PPL (shifted logits/labels), equivalent to token-by-token CE in expectation
    for a causal LM under the same context.
    """
    device = args.device if args else "cuda"
    data = load_dataset("emozilla/pg19-test", split="test")

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    nlls = []

    # Make sure we're in eval mode (no dropout)
    model.eval()

    # Evaluate only first sample (same as your original code)
    for text in data["text"][:1]:
        encodings = tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.to(device)  # [1, T]

        print(input_ids[:, :10])
        seq_len = input_ids.size(1)
        print(f"seq_len: {seq_len}")

        # To match your token loop which runs idx=0..(seq_len-2),
        # we compute loss on positions 1..(T-1) predicted by logits 0..(T-2).
        # Also respect args.num_eval_tokens (number of prediction steps).
        if args is not None and args.num_eval_tokens is not None:
            # num_eval_tokens refers to number of next-token predictions
            # so we need input length = num_eval_tokens + 1
            max_len = min(seq_len, args.num_eval_tokens + 1)
            input_ids = input_ids[:, :max_len]

        # Single forward, no cache
        outputs = model(input_ids=input_ids, use_cache=False)
        logits = outputs.logits  # [1, T, vocab]

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()  # [1, T-1, vocab]
        shift_labels = input_ids[:, 1:].contiguous()   # [1, T-1]

        # Per-token NLL
        token_nll = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )  # [(T-1)]

        nlls.append(token_nll)

    mean_nll = torch.cat(nlls, dim=0).mean()
    ppl = torch.exp(mean_nll).item()
    return ppl

@torch.no_grad()
def ppl_eval(model, tokenizer, args=None):
    device = args.device if args else "cuda"
    data = load_dataset("emozilla/pg19-test", split="test")
    nlls = []
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    past_key_values=KVWithSmallKCache(config=model.config) if args.eval_compressed else None
    current_num_eval_tokens = 0
    for text in data["text"][:1]:
        encodings = tokenizer(text, return_tensors="pt")

        print(encodings.input_ids[:, :10])

        seq_len = encodings.input_ids.size(1)
        print(f"seq_len: {seq_len}")
        pbar = tqdm(range(0, seq_len - 1))

        for idx in pbar:
            input_ids = encodings.input_ids[:, idx : idx + 1].to(device)
            with torch.no_grad():
                outputs = model(
                    input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                logits = outputs.logits.view(-1, model.config.vocab_size)
                past_key_values = outputs.past_key_values
                label = encodings.input_ids[:, idx + 1 : idx + 2].to(logits.device).view(-1)
                neg_log_likelihood = loss_fn(logits, label)

            nlls.append(neg_log_likelihood)
            pbar.set_description(
                f"nll: {neg_log_likelihood.item():.2f}, ppl: {torch.exp(neg_log_likelihood).item():.2f}"
            )
            current_num_eval_tokens += 1
            if args.num_eval_tokens is not None and current_num_eval_tokens >= args.num_eval_tokens:
                break
        if args.num_eval_tokens is not None and current_num_eval_tokens >= args.num_eval_tokens:
            break
    ppl = torch.exp(torch.stack(nlls).mean()).item()
    return ppl

# -----------------------------
# CLI / main
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Perplexity test for original/compressed model, with optional H2O token-by-token eval.")

    # Model paths
    p.add_argument("--base-model", type=str, required=True, help="Base model path or HF repo id.")
    p.add_argument("--adapter", type=str, default=None, help="Optional LoRA adapter path. If set, will load via PeftModel.")
    p.add_argument("--attn-impl", type=str, default="eager", choices=["eager", "sdpa", "flash_attention_2"], help="HF attention implementation.")

    p.add_argument("--device", type=str, default="cuda")

    # Compression knobs
    p.add_argument("--quant-mode", type=str, default="int", help="Passed to apply_compression_to_model (e.g., int/none/...).")
    p.add_argument("--topk-ratio", type=float, default=0.1)
    p.add_argument("--sink-size", type=int, default=4)
    p.add_argument("--local-window", type=int, default=16)
    p.add_argument("--escaped-layers", type=int, nargs="*", default=[], help="List of layer indices to skip compression.")
    # What to run
    p.add_argument("--eval-original", action="store_true", help="Evaluate base/original model.")
    p.add_argument("--eval-compressed", action="store_true", help="Evaluate compressed model (requires --compress).")
    p.add_argument("--eval-h2o", action="store_true", help="Also run token-by-token cache eval (slow).")
    
    p.add_argument("--num-eval-tokens", type=int, default=1000, help="Number of tokens to eval for PPL (to limit runtime).")

    return p.parse_args()


def load_small_state(model, model_name):
    DISTILLED_DIR = f"./data/{model_name}"
    small_state = torch.load(f"{DISTILLED_DIR}/small_attn_weights.pt", map_location="cpu")
    base_state = model.state_dict()
    for name, param in small_state.items():
        if name in base_state and base_state[name].shape == param.shape:
            base_state[name] = param
    model.load_state_dict(base_state)
    return model


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    original_model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_impl,
    )
    
    model_name = os.path.basename(args.base_model)

    # Build test loader once

    if args.eval_original:
        ppl_original = ppl_eval_parallel(original_model, tokenizer, args)
        print(f"Original Model PPL on pg19-test: {ppl_original}")

    if args.eval_compressed:
 
        model = apply_compression_to_model(
            original_model,
            model_name,
            quant_mode=args.quant_mode,
            topk_ratio=args.topk_ratio,
            sink_size=args.sink_size,
            local_window=args.local_window,
            escaped_layers=args.escaped_layers
        )

        model = load_small_state(model, model_name)


        model.eval()

        ppl_compressed = ppl_eval_parallel(model, tokenizer, args)
        print(f"Compressed Model PPL on pg19-test: {ppl_compressed}")

    if args.eval_h2o:
        original_model.config.heavy_ratio = args.topk_ratio
        original_model.config.sink_size = args.sink_size
        original_model.config.local_window = args.local_window
        model = convert_kvcache_llama_heavy_recent(original_model, original_model.config)
        ppl_compressed_h2o = ppl_eval(model, tokenizer, args)
        print(f"Compressed Model PPL (H2O/token) on pg16-test: {ppl_compressed_h2o}")


if __name__ == "__main__":
    main()
    
    
```
