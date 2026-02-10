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

        seq_len = input_ids.size(1)

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

        if args.output_file:
            # Calculate cumulative PPL
            cum_loss = torch.cumsum(token_nll, dim=0)
            count = torch.arange(1, token_nll.size(0) + 1, device=token_nll.device)
            cum_ppl = torch.exp(cum_loss / count)

            # Subsample for logging
            indices = torch.arange(args.log_interval - 1, token_nll.size(0), args.log_interval)
            if indices.numel() == 0 or indices[-1] != token_nll.size(0) - 1:
                # Ensure last point is included if possible/desired, or just rely on interval
                pass

            logged_steps = (indices + 1).tolist()
            logged_ppls = cum_ppl[indices].tolist()

            rows = [{"step": s, "ppl": p} for s, p in zip(logged_steps, logged_ppls)]

            with open(args.output_file, 'w') as f:
                json.dump(rows, f, indent=4)
            print(f"Saved PPL results to {args.output_file}")

        nlls.append(token_nll)

    mean_nll = torch.cat(nlls, dim=0).mean()
    ppl = torch.exp(mean_nll).item()
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