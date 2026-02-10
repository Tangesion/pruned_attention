# 蒸馏实验

## 实验目标
我们需要去测试before distillation， after distillation(排序蒸馏)，MSE Distillation下的Recall，PPL以及loss训练曲线

## 现有的关键函数代码
```python
#scrips/run_distillation_experiment.py
import torch
import torch.nn as nn
import argparse
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pruned_attention.calibration import get_c4_simple, get_compression_indices
from src.pruned_attention.attention import CompressedLlamaAttention, repeat_kv
from src.pruned_attention.distillation import CompressedLlamaAttentionTopKDistillWrapper, _get_layer_inputs

# --- 1. Define MSE Distillation Wrapper ---
class CompressedLlamaAttentionMSEDistillWrapper(nn.Module):
    def __init__(self, compressed_layer, original_layer_config):
        super().__init__()
        self.layer = compressed_layer
        self.config = original_layer_config
        self.head_dim = self.layer.head_dim
        self.scaling = self.head_dim**-0.5
        self.num_key_value_groups = self.layer.num_key_value_groups

        # Freeze all except small projections
        for p in self.layer.parameters():
            p.requires_grad = False

        # Make small projections trainable
        self.layer.q_proj_small.weight = nn.Parameter(self.layer.q_proj_small.weight)
        self.layer.k_proj_small.weight = nn.Parameter(self.layer.k_proj_small.weight)
        self.layer.q_proj_small.weight.requires_grad = True
        self.layer.k_proj_small.weight.requires_grad = True

        self.loss_fct = nn.MSELoss()

    def forward(self, hidden_states, position_embeddings, attention_mask=None):
        bsz, q_len, _ = hidden_states.shape

        # Teacher (Full Attention)
        with torch.no_grad():
            q_full = self.layer.q_proj(hidden_states).view(bsz, q_len, self.layer.num_q_heads, self.head_dim).transpose(1, 2)
            k_full = self.layer.k_proj(hidden_states).view(bsz, q_len, self.layer.num_kv_heads, self.head_dim).transpose(1, 2)

            cos, sin = position_embeddings
            # Expand cos/sin to match batch size if needed (fix from Exp 4)
            # Actually apply_rotary_pos_emb in HF handles broadcasting usually,
            # but let's ensure we pass correct shapes if wrapper expects it.
            # In distillation.py, it expects tuple.
            q_full, k_full = apply_rotary_pos_emb(q_full, k_full, cos, sin)

            k_full = repeat_kv(k_full, self.num_key_value_groups)
            teacher_scores = torch.matmul(q_full, k_full.transpose(2, 3)) * self.scaling
            if attention_mask is not None:
                teacher_scores = teacher_scores + attention_mask

        # Student (Compressed Attention)
        q_small = self.layer.q_proj_small(hidden_states).view(bsz, q_len, self.layer.num_q_heads, self.layer.compressed_dim).transpose(1, 2)
        k_small = self.layer.k_proj_small(hidden_states).view(bsz, q_len, self.layer.num_kv_heads, self.layer.compressed_dim).transpose(1, 2)

        # Mixed RoPE
        q_small, k_small = self.layer.apply_mixed_index_rope(q_small, k_small, position_embeddings)
        k_small = repeat_kv(k_small, self.num_key_value_groups)

        student_scores = torch.matmul(q_small, k_small.transpose(2, 3)) * (self.layer.compressed_dim**-0.5)
        if attention_mask is not None:
            student_scores = student_scores + attention_mask

        # MSE Loss
        # Mask out padding/masked positions to avoid learning from -inf
        if attention_mask is not None:
             # Typically attention_mask is 0 for valid, large negative for masked
             # We only want to compute MSE on valid tokens
             valid_mask = (attention_mask > -1000)
             return self.loss_fct(student_scores[valid_mask], teacher_scores[valid_mask]), student_scores, teacher_scores
        else:
            return self.loss_fct(student_scores, teacher_scores), student_scores, teacher_scores
        
def cal_recall(student_scores, teacher_scores, topk_ratio=0.1):
    """
    Calculates Top-K Recall between student and teacher attention scores.
    """
    with torch.no_grad():
        bsz, num_heads, q_len, kv_len = teacher_scores.shape
        k_val = max(1, int(kv_len * topk_ratio))

        # Ground Truth Indices (from Teacher)
        _, indices_true = torch.topk(teacher_scores, k=k_val, dim=-1)

        # Predicted Indices (from Student)
        _, indices_pred = torch.topk(student_scores, k=k_val, dim=-1)

        # Calculate intersection
        # [B, H, Q, K, 1] == [B, H, Q, 1, K] -> [B, H, Q, K, K] -> any match in last dim
        matches = (indices_true.unsqueeze(-1) == indices_pred.unsqueeze(-2)).any(dim=-1).sum()

        total_k = indices_true.numel()
        recall = matches.float() / total_k
        return recall.item()

def collect_layer_wise_distillation(
    model, 
    loss_func:str,
    tokenizer, 
    indices_path, 
    train_steps_per_layer=100,
    batch_size=4,
    lr=1e-3,
    seq_len=128,
    num_calibration_samples=128 
):
    """
    Performs layer-wise distillation to fine-tune the compressed attention projections.
    """
    device = model.device
    
    loss_history = {[] for _ in range(model.config.num_hidden_layers)}
    recall_history = {}
    
    print("Preparing data...")
    c4_data = get_c4_simple(tokenizer, num_calibration_samples, seq_len)
    dataset = TensorDataset(c4_data)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    hidden_states, attention_mask, position_embeddings = _get_layer_inputs(model, dataloader, device, num_samples=num_calibration_samples)

    with open(indices_path, 'r') as f:
        indices_data = json.load(f)

    model.config.topk_ratio = 0.2
    model.config.sink_size = 0
    model.config.local_window = 0
    model.config.escaped_layers = []

    for i in range(model.config.num_hidden_layers):
        print(f"\n=== Processing Layer {i}/{model.config.num_hidden_layers} ===")
        layer = model.model.layers[i]
        
        keep_indices = torch.tensor(indices_data[str(i)], dtype=torch.long).to(device)
        
        
        # This replaces the original self_attn with the compressed version for distillation
        compressed_attn = CompressedLlamaAttention(
            model.config, 
            layer.self_attn, 
            group_keep_indices=keep_indices
        ).to(device)
        
        if loss_func == "mse":
            distill_wrapper = CompressedLlamaAttentionMSEDistillWrapper(compressed_attn, model.config).to(device)
        elif loss_func == "topk":
            distill_wrapper = CompressedLlamaAttentionTopKDistillWrapper(compressed_attn, model.config).to(device)

        print(f"  Training small projections...")
        optimizer = torch.optim.AdamW(distill_wrapper.parameters(), lr=lr)
        
        layer_dataset = TensorDataset(hidden_states)
        layer_loader = DataLoader(layer_dataset, batch_size=batch_size, shuffle=True)
        
        pbar = tqdm(total=train_steps_per_layer, desc=f"Layer {i} Distill")
        step = 0
        while step < train_steps_per_layer:
            for (batch_hidden,) in layer_loader:
                batch_hidden = batch_hidden.to(device)
                
                # We need to compute position embeddings for each batch
                # as they are sequence length dependent and not constant
                cos, sin = position_embeddings
                batch_pos_emb = (cos[:, :batch_hidden.shape[1]], sin[:, :batch_hidden.shape[1]])
                batch_attn_mask = attention_mask[:, :, :batch_hidden.shape[1], :batch_hidden.shape[1]]


                loss, student_scores, teacher_scores = distill_wrapper(batch_hidden, batch_pos_emb, batch_attn_mask)
                loss.backward()
                loss_history[i].append(loss.item())
                
                torch.nn.utils.clip_grad_norm_(distill_wrapper.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                step += 1
                pbar.set_description(f"Loss: {loss.item():.4f}")
                pbar.update(1)
                
                if step == train_steps_per_layer - 1:
                    recall_history[i] = cal_recall(student_scores, teacher_scores, topk_ratio=model.config.topk_ratio)
                
                if step >= train_steps_per_layer: break
        pbar.close()
        
        # Replace the layer's attention module with the distilled one
        model.model.layers[i].self_attn = compressed_attn
        
        print(f"  Generating inputs for Layer {i+1}...")
        new_hidden_states_list = []
        gen_loader = DataLoader(layer_dataset, batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            for (batch_hidden,) in tqdm(gen_loader, desc="Forwarding"):
                batch_hidden = batch_hidden.to(device)
                cos, sin = position_embeddings
                batch_pos_emb = (cos[:, :batch_hidden.shape[1]], sin[:, :batch_hidden.shape[1]])
                batch_attn_mask = attention_mask[:, :, :batch_hidden.shape[1], :batch_hidden.shape[1]]
                
                # Must use the full LlamaDecoderLayer forward pass
                layer_output = model.model.layers[i]( 
                    batch_hidden, 
                    attention_mask=batch_attn_mask,
                    position_embeddings=batch_pos_emb
                )[0] # layer_output is a tuple
                
                new_hidden_states_list.append(layer_output.cpu())

        hidden_states = torch.cat(new_hidden_states_list, dim=0)
        
        del distill_wrapper
        torch.cuda.empty_cache()

    print("\nDistillation Complete!")
    return model, loss_history, recall_history

```

ppl测试关键代码
``` python

#scripts/run_pg16_ppl_test.py
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
    
    
"""
  
python scripts/run_pg16_ppl_test.py  \
--base-model /home/tgx/models/Llama-3.2-1B \
--eval-original

python scripts/run_pg16_ppl_test.py  \
  --base-model /home/tgx/models/Llama-3.2-1B \
  --eval-compressed \
  --quant-mode int --topk-ratio 0.1 --sink-size 4 --local-window 16 --escaped-layers 0 1 14 15\
  --num-eval-tokens 1024

python scripts/run_pg16_ppl_test.py \
  --base-model /home/tgx/models/Llama-3.2-1B \
  --eval-h2o \
  --topk-ratio 0.1 --sink-size 4 --local-window 16 \
  --num-eval-tokens 1024
"""
```

## 计划
你需要去参考上述的代码。目前计划去收集mse loss和rank loss下每层的训练loss以及每层训练完成后的recall分数，recall分数可以最后再计算一个平均数对于总层数。以及蒸馏前，蒸馏后（两种方法）的recall对比、ppl对比。

## 要求
收集到的数据存在json中，相关图表可等json收集完成后，代码运行成功后再去完成，图表优先级靠后，优先级最高的应该是收集到json数据
