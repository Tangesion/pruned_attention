import torch
import os
import sys
import argparse
import json
from transformers import AutoTokenizer, LlamaForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pruned_attention.calibration import apply_compression_to_model
from src.h2o_attention.attention import convert_kvcache_llama_heavy_recent
from src.slide_attention.attention import convert_kvcache_llama_sliding_window
from src.pruned_attention.attention import KVWithSmallKCache


import os
import json
import torch
from datasets import load_dataset

def _dataset_rows(dataset_name: str):
    n = dataset_name.lower()
    if n in ["wikitext", "wikitext2"]:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        return ds["text"]
    if n == "wikitext103":
        ds = load_dataset("yehzw/wikitext-103", "clean", split="test")
        rows = []
        for x in ds["text"]:
            if isinstance(x, list):
                rows.extend([s for s in x if isinstance(s, str)])
            elif isinstance(x, str):
                rows.append(x)
        return rows
    if n == "ptb":
        ds = load_dataset("ptb_text_only", "penn_treebank", split="test")
        return ds["sentence"]
    if n == "lambada":
        ds = load_dataset("lambada", split="test")
        return ds["text"]
    if n in ["pg19", "pg19-test"]:
        ds = load_dataset("emozilla/pg19-test", split="test")
        return ds["text"]
    if n == "c4":
        # 小切片，避免拉全量
        ds = load_dataset(
            "json",
            data_files="https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-validation.00000-of-00008.json.gz",
            split="train[:200]",
        )
        return ds["text"]
    raise ValueError(f"Unsupported dataset: {dataset_name}")

@torch.no_grad()
def ppl_eval_parallel(model, tokenizer, args=None):
    device = args.device if args else "cuda"
    dataset_name = getattr(args, "dataset", "pg19-test")
    num_eval_tokens = getattr(args, "num_eval_tokens", 1024)
    log_interval = max(1, int(getattr(args, "log_interval", 64)))
    output_file = getattr(args, "output_file", None)

    target = int(num_eval_tokens) if num_eval_tokens is not None else None
    rows = _dataset_rows(dataset_name)
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    nll_chunks = []
    used = 0
    
    is_ptb = dataset_name.lower() == "ptb"
    carry_text = ""
    min_chars_for_ptb = 2048 

    for t in rows:
        if not isinstance(t, str) or len(t.strip()) == 0:
            continue
        
        text_to_eval = t.strip()

        if is_ptb:
            # 累积到足够长度再评估，避免单句过短
            carry_text = (carry_text + " " + text_to_eval).strip()
            if len(carry_text) < min_chars_for_ptb:
                continue
            text_to_eval = carry_text
            carry_text = ""

        ids = tokenizer(text_to_eval, return_tensors="pt", truncation=False).input_ids.to(device)  # [1, T]
        
        #ids = tokenizer(t, return_tensors="pt", truncation=False).input_ids.to(device)  # [1, T]
        if ids.size(1) < 2:
            continue

        # 截断到剩余预算（token 预测步数 = length-1）
        if target is not None:
            remain = target - used
            if remain <= 0:
                break
            max_len = min(ids.size(1), remain + 1)
            ids = ids[:, :max_len]
            if ids.size(1) < 2:
                continue

        out = model(input_ids=ids, use_cache=False)
        logits = out.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = ids[:, 1:].contiguous()

        token_nll = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        if token_nll.numel() == 0:
            continue

        nll_chunks.append(token_nll.detach().cpu())
        used += token_nll.numel()

        if target is not None and used >= target:
            break

    if len(nll_chunks) == 0:
        raise ValueError(f"No valid token_nll for dataset={dataset_name}")

    all_nll = torch.cat(nll_chunks, dim=0)
    ppl = float(torch.exp(all_nll.mean()).item())

    if output_file:
        cum_loss = torch.cumsum(all_nll, dim=0)
        cnt = torch.arange(1, all_nll.size(0) + 1)
        cum_ppl = torch.exp(cum_loss / cnt)

        idx = torch.arange(log_interval - 1, all_nll.size(0), log_interval)
        if idx.numel() == 0 or idx[-1].item() != all_nll.size(0) - 1:
            idx = torch.cat([idx, torch.tensor([all_nll.size(0) - 1])])

        curve = [{"step": int(i.item() + 1), "ppl": float(cum_ppl[i].item())} for i in idx]
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(curve, f, indent=2)

    return ppl

@torch.no_grad()
def ppl_eval(model, tokenizer, args, use_kv_cache=False):
    """
    Token-by-token PPL evaluation with optional KV cache.
    """
    device = args.device if args else "cuda"
    data = load_dataset("emozilla/pg19-test", split="test")
    nlls = []
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    past_key_values = KVWithSmallKCache(config=model.config) if use_kv_cache else None
    current_num_eval_tokens = 0

    for text in data["text"][:1]:
        encodings = tokenizer(text, return_tensors="pt")
        seq_len = encodings.input_ids.size(1)
        print(f"seq_len: {seq_len}")
        pbar = tqdm(range(0, seq_len - 1))

        for idx in pbar:
            input_ids = encodings.input_ids[:, idx : idx + 1].to(device)
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
    p.add_argument("--attn-impl", type=str, default="eager", choices=["eager", "sdpa", "flash_attention_2"], help="HF attention implementation.")

    p.add_argument("--device", type=str, default="cuda")

    # Compression knobs
    p.add_argument("--topk-ratios", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                   help="List of topk ratios to evaluate (for batch experiments).")
    p.add_argument("--sink-size", type=int, default=4)
    p.add_argument("--local-window", type=int, default=16)
    p.add_argument("--escaped-layers", type=int, nargs="*", default=[], help="List of layer indices to skip compression.")

    # What to run (methods to compare)
    p.add_argument("--eval-original", action="store_true", help="Evaluate base/original model (Full Attention).")
    p.add_argument("--eval-ours-int4", action="store_true", help="Evaluate our method with INT4 quantization.")
    p.add_argument("--eval-ours-bf16", action="store_true", help="Evaluate our method with BF16 (no quantization).")
    p.add_argument("--eval-h2o", action="store_true", help="Evaluate H2O attention method.")
    p.add_argument("--eval-slide", action="store_true", help="Evaluate sliding window attention method.")
    p.add_argument("--eval-all", action="store_true", help="Run all evaluation methods.")

    p.add_argument("--num-eval-tokens", type=int, default=16384, help="Number of tokens to eval for PPL (default 16k).")
    p.add_argument("--output-dir", type=str, default="./experiment_ppl_results", help="Directory to save PPL results.")

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

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_name = os.path.basename(args.base_model)

    # If --eval-all, enable all methods
    if args.eval_all:
        args.eval_original = True
        args.eval_ours_int4 = True
        args.eval_ours_bf16 = True
        args.eval_h2o = True
        args.eval_slide = True

    # Results storage: {method: {topk_ratio: ppl}}
    results = {}

    # 1. Full Attention (Original) - only one value, no topk dependency
    if args.eval_original:
        print("\n" + "="*50)
        print("Evaluating: Full Attention (Original)")
        print("="*50)
        model = LlamaForCausalLM.from_pretrained(
            args.base_model, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation=args.attn_impl
        )
        ppl = ppl_eval(model, tokenizer, args, use_kv_cache=False)
        results["full_attention"] = {ratio: ppl for ratio in args.topk_ratios}
        print(f"Full Attention PPL: {ppl:.4f}")
        del model
        torch.cuda.empty_cache()

    # 2. Ours (INT4)
    if args.eval_ours_int4:
        print("\n" + "="*50)
        print("Evaluating: Ours (INT4)")
        print("="*50)
        results["ours_int4"] = {}
        for ratio in args.topk_ratios:
            print(f"\n--- topk_ratio: {ratio} ---")
            # Reload model for each ratio
            model = LlamaForCausalLM.from_pretrained(
                args.base_model, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation=args.attn_impl
            )
            model = apply_compression_to_model(
                model, model_name, quant_mode="int", topk_ratio=ratio,
                sink_size=args.sink_size, local_window=args.local_window, escaped_layers=args.escaped_layers
            )
            model = load_small_state(model, model_name)
            model.eval()
            ppl = ppl_eval(model, tokenizer, args, use_kv_cache=True)
            results["ours_int4"][ratio] = ppl
            print(f"Ours (INT4) topk={ratio}: PPL={ppl:.4f}")
            del model
            torch.cuda.empty_cache()

    # 3. Ours (BF16)
    if args.eval_ours_bf16:
        print("\n" + "="*50)
        print("Evaluating: Ours (BF16)")
        print("="*50)
        results["ours_bf16"] = {}
        for ratio in args.topk_ratios:
            print(f"\n--- topk_ratio: {ratio} ---")
            model = LlamaForCausalLM.from_pretrained(
                args.base_model, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation=args.attn_impl
            )
            model = apply_compression_to_model(
                model, model_name, quant_mode=None, topk_ratio=ratio,
                sink_size=args.sink_size, local_window=args.local_window, escaped_layers=args.escaped_layers
            )
            model = load_small_state(model, model_name)
            model.eval()
            ppl = ppl_eval(model, tokenizer, args, use_kv_cache=True)
            results["ours_bf16"][ratio] = ppl
            print(f"Ours (BF16) topk={ratio}: PPL={ppl:.4f}")
            del model
            torch.cuda.empty_cache()

    # 4. H2O
    if args.eval_h2o:
        print("\n" + "="*50)
        print("Evaluating: H2O Attention")
        print("="*50)
        results["h2o"] = {}
        for ratio in args.topk_ratios:
            print(f"\n--- topk_ratio: {ratio} ---")
            model = LlamaForCausalLM.from_pretrained(
                args.base_model, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation=args.attn_impl
            )
            model.config.heavy_ratio = ratio
            model.config.sink_size = args.sink_size
            model.config.local_window = args.local_window
            model.config.escaped_layers = args.escaped_layers
            model = convert_kvcache_llama_heavy_recent(model, model.config)
            ppl = ppl_eval(model, tokenizer, args, use_kv_cache=False)
            results["h2o"][ratio] = ppl
            print(f"H2O topk={ratio}: PPL={ppl:.4f}")
            del model
            torch.cuda.empty_cache()

    # 5. Sliding Window
    if args.eval_slide:
        print("\n" + "="*50)
        print("Evaluating: Sliding Window")
        print("="*50)
        results["slide"] = {}
        for ratio in args.topk_ratios:
            print(f"\n--- window_ratio: {ratio} ---")
            model = LlamaForCausalLM.from_pretrained(
                args.base_model, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation=args.attn_impl
            )
            model.config.sink_size = args.sink_size
            model.config.local_window_ratio = ratio
            model.config.escaped_layers = args.escaped_layers
            model = convert_kvcache_llama_sliding_window(model, model.config)
            ppl = ppl_eval(model, tokenizer, args, use_kv_cache=False)
            results["slide"][ratio] = ppl
            print(f"Slide window_ratio={ratio}: PPL={ppl:.4f}")
            del model
            torch.cuda.empty_cache()

    # Save results
    output_file = os.path.join(args.output_dir, "ppl_results.json")
    with open(output_file, "w") as f:
        json.dump({
            "config": {
                "model": args.base_model,
                "num_eval_tokens": args.num_eval_tokens,
                "sink_size": args.sink_size,
                "local_window": args.local_window,
                "escaped_layers": args.escaped_layers,
                "topk_ratios": args.topk_ratios,
            },
            "results": results
        }, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Print summary table
    print("\n" + "="*70)
    print("Summary: PPL vs TopK Ratio")
    print("="*70)
    header = "Method".ljust(20) + "".join([f"{r:.1f}".center(10) for r in args.topk_ratios])
    print(header)
    print("-"*70)
    for method, ppls in results.items():
        row = method.ljust(20) + "".join([f"{ppls.get(r, '-'):.2f}".center(10) if isinstance(ppls.get(r), float) else "-".center(10) for r in args.topk_ratios])
        print(row)


if __name__ == "__main__":
    main()
    
"""
python scripts/run_pg16_ppl_test.py \                                                                                              ─╯
      --base-model /home/tgx/models/Llama-3.2-1B \
      --eval-all \
      --topk-ratios 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 \
      --num-eval-tokens 1024 \
      --sink-size 4 --local-window 16 \
      --escaped-layers 0 1 14 15 \
      --output-dir ./experiment_ppl_results

"""