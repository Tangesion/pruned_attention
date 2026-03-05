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

# -----------------------------
# Dataset helpers
# -----------------------------
class IndexDataset(Dataset):
    def __init__(self, tensors):
        self.tensors = tensors

    def __getitem__(self, index):
        return self.tensors[index]

    def __len__(self):
        return len(self.tensors)


def get_test_data(name, tokenizer, seq_len=2048, batch_size=4):
    name = name.lower()

    def chunk_token_ids(token_ids, seq_len):
        n = token_ids.numel() // seq_len
        if n == 0:
            raise ValueError(f"Not enough tokens for seq_len={seq_len}")
        blocks = [token_ids[i * seq_len:(i + 1) * seq_len] for i in range(n)]
        return torch.stack(blocks, dim=0)

    def process_rows_text(rows):
        # 避免一次性 join 全量文本导致超长 tokenize
        all_blocks = []
        for t in rows:
            if not isinstance(t, str) or len(t) == 0:
                continue
            ids = tokenizer(t, return_tensors="pt", truncation=False).input_ids[0]
            if ids.numel() < seq_len:
                continue
            all_blocks.append(chunk_token_ids(ids, seq_len))
        if len(all_blocks) == 0:
            raise ValueError(f"No valid text blocks for dataset={name}")
        stacked = torch.cat(all_blocks, dim=0)
        return IndexDataset(stacked)

    if "wikitext2" in name or name == "wikitext":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        dataset = process_rows_text(ds["text"])
    elif "wikitext103" in name:
        ds = load_dataset("yehzw/wikitext-103", "clean", split="test")
        # 兼容 list[str] / str
        rows = []
        for x in ds["text"]:
            if isinstance(x, list):
                rows.extend([s for s in x if isinstance(s, str)])
            elif isinstance(x, str):
                rows.append(x)
        dataset = process_rows_text(rows)
    elif "ptb" in name:
        ds = load_dataset("allenai/ptb_text_only", "penn_treebank", split="test")
        dataset = process_rows_text(ds["sentence"])
    elif "lambada" in name:
        ds = load_dataset("lambada", split="test")
        dataset = process_rows_text(ds["text"])
    elif "pg19" in name:
        ds = load_dataset("emozilla/pg19-test", split="test")
        dataset = process_rows_text(ds["text"])
    elif "c4" in name:
        # C4 容易网络失败；只取少量样本并做兜底
        ds = load_dataset("allenai/c4", "en", split="validation[:1%]")
        dataset = process_rows_text(ds["text"])
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    return DataLoader(dataset, batch_size=batch_size, shuffle=False), ds


# -----------------------------
# PPL eval
# -----------------------------
@torch.no_grad()
def ppl_eval(model, tokenizer, test_loader=None, dataset='wikitext2', model_seq_len=2048, batch_size=32, device="cuda"):
    model.eval()
    if test_loader is None:
        test_loader, _ = get_test_data(dataset, tokenizer, seq_len=model_seq_len, batch_size=batch_size)

    nlls = []
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    for batch in tqdm(test_loader, desc="Evaluating PPL (no-cache)"):
        batch = batch.to(device)
        output = model(batch, use_cache=False)
        lm_logits = output.logits

        # Skip NaN/Inf batches (optional safety)
        if not torch.isfinite(lm_logits).all():
            continue

        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1))
        nlls.append(loss)

    ppl = float(np.exp(torch.cat(nlls, dim=-1).mean().item()))
    return ppl


@torch.no_grad()
def ppl_eval_h2o_slow(model, tokenizer, test_loader=None, dataset='wikitext2', model_seq_len=2048, device="cuda"):
    """
    Token-by-token evaluation with use_cache=True.
    Useful for H2O-like / eviction / streaming attention implementations.
    """
    model.eval()
    if test_loader is None:
        test_loader = get_test_data(dataset, tokenizer, seq_len=model_seq_len, batch_size=1)

    nlls = []
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    print("Start H2O PPL Evaluation (token-by-token, use_cache=True). This will be slow.")

    for batch in tqdm(test_loader, desc="Evaluating PPL (h2o/token)"):
        batch = batch.to(device)  # [1, T]
        seq_len = batch.size(1)

        # Reset per-sequence internal states if model has them
        for module in model.modules():
            if hasattr(module, "_reset_masks") and callable(getattr(module, "_reset_masks")):
                module._reset_masks()

        past_key_values = None

        for i in range(seq_len - 1):
            input_ids = batch[:, i:i + 1]    # [1,1]
            target_id = batch[:, i + 1:i + 2]  # [1,1]

            outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values

            logits = outputs.logits[:, -1, :]  # [1, vocab]
            loss = loss_fct(logits, target_id.view(-1))
            nlls.append(loss)

    ppl = torch.exp(torch.stack(nlls).mean()).item()
    return ppl

@torch.no_grad()
def ppl_eval_h2o_fast(model, tokenizer, dataset='wikitext2', model_seq_len=2048, device="cuda", num_eval_tokens=None):
    _, data = get_test_data(dataset, tokenizer, seq_len=model_seq_len, batch_size=1)
    #data = load_dataset("emozilla/pg19-test", split="test")
    nlls = []
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    past_key_values = None
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
            if num_eval_tokens is not None and current_num_eval_tokens >= num_eval_tokens:
                break
        if num_eval_tokens is not None and current_num_eval_tokens >= num_eval_tokens:
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

    # Data / eval
    p.add_argument("--dataset", type=str, default="wikitext2", help="wikitext2|wikitext103|ptb|c4|lambada|pg19-test")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda")

    # Compression knobs
    p.add_argument("--quant-mode", type=str, default="int", help="Passed to apply_compression_to_model (e.g., int/none/...).")
    p.add_argument("--topk-ratio", type=float, default=0.1)
    p.add_argument("--sink-size", type=int, default=4)
    p.add_argument("--local-window", type=int, default=16)

    # Optional state injection (your small_attn_weights)
    p.add_argument("--small-state", type=str, default=None, help="Path to small_attn_weights.pt to load into (compressed) model.")
    p.add_argument("--small-state-prefix", type=str, default="model.", help="Prefix to add to keys when loading small state.")

    # What to run
    p.add_argument("--eval-original", action="store_true", help="Evaluate base/original model.")
    p.add_argument("--eval-compressed", action="store_true", help="Evaluate compressed model (requires --compress).")
    p.add_argument("--eval-h2o", action="store_true", help="Also run token-by-token cache eval (slow).")

    return p.parse_args()


def maybe_load_small_state(model, small_state_path: str, prefix: str = "model."):
    if not small_state_path:
        return model
    small_state = torch.load(small_state_path, map_location="cpu")
    base_state = model.state_dict()

    updated = 0
    for name, param in small_state.items():
        full_name = prefix + name if prefix is not None else name
        if full_name in base_state and base_state[full_name].shape == param.shape:
            base_state[full_name] = param
            updated += 1

    model.load_state_dict(base_state)
    print(f"Loaded small-state tensors: {updated}/{len(small_state)} matched.")
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

    # Build test loader once
    test_loader = get_test_data(args.dataset, tokenizer, seq_len=args.seq_len, batch_size=args.batch_size)

    if args.eval_original:
        ppl_original = ppl_eval(original_model, tokenizer, test_loader=test_loader, dataset=args.dataset, model_seq_len=args.seq_len, batch_size=args.batch_size, device=args.device)
        print(f"Original Model PPL on {args.dataset}: {ppl_original}")

    if args.eval_compressed:
 
        model = apply_compression_to_model(
            original_model,
            quant_mode=args.quant_mode,
            topk_ratio=args.topk_ratio,
            sink_size=args.sink_size,
            local_window=args.local_window,
        )

        model = maybe_load_small_state(model, args.small_state, prefix=args.small_state_prefix)

        if args.adapter:
            model = PeftModel.from_pretrained(model, args.adapter)

        model.eval()

        #ppl_compressed = ppl_eval(model, tokenizer, test_loader=test_loader, dataset=args.dataset, model_seq_len=args.seq_len, batch_size=args.batch_size, device=args.device)
        ppl_compressed = ppl_eval_h2o_fast(model, tokenizer, dataset=args.dataset, model_seq_len=args.seq_len, device=args.device, num_eval_tokens=1000)
        print(f"Compressed Model PPL on {args.dataset}: {ppl_compressed}")

    if args.eval_h2o:
        original_model.config.heavy_ratio = args.topk_ratio
        original_model.config.sink_size = args.sink_size
        original_model.config.local_window = args.local_window
        model = convert_kvcache_llama_heavy_recent(original_model, original_model.config)
        h2o_loader = get_test_data(args.dataset, tokenizer, seq_len=args.seq_len, batch_size=1)
        #ppl_compressed_h2o = ppl_eval_h2o_slow(model, tokenizer, test_loader=h2o_loader, dataset=args.dataset, model_seq_len=args.seq_len, device=args.device)
        ppl_compressed_h2o = ppl_eval_h2o_fast(model, tokenizer, dataset=args.dataset, model_seq_len=args.seq_len, device=args.device, num_eval_tokens=1000)
        print(f"Compressed Model PPL (H2O/token) on {args.dataset}: {ppl_compressed_h2o}")


if __name__ == "__main__":
    main()
    
    
"""
python scripts/run_perplexity_test.py \
  --base-model /home/tgx/data/models/Llama-3-8B-Instruct \
  --dataset wikitext2 --seq-len 2048 --batch-size 2 \
  --eval-original
  
python scripts/run_perplexity_test.py \
  --base-model /home/tgx/data/models/Llama-3-8B-Instruct \
  --eval-compressed \
  --quant-mode int --topk-ratio 0.1 --sink-size 4 --local-window 16 \
  --small-state ./compressed_llama_distilled_topk_wo_sink/small_attn_weights.pt
  
python scripts/run_perplexity_test.py \
  --base-model /home/tgx/data/models/Llama-3-8B-Instruct \
  --eval-h2o \
  --topk-ratio 0.1 --sink-size 4 --local-window 16 \
  --seq-len 2048


"""