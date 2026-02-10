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

from src.pruned_attention.calibration import get_c4_simple, get_compression_indices, apply_compression_to_model
from src.pruned_attention.attention import CompressedLlamaAttention, repeat_kv
from src.pruned_attention.distillation import CompressedLlamaAttentionTopKDistillWrapper, _get_layer_inputs

# Try to import ppl_eval_parallel from run_pg16_ppl_test.py in the same directory
try:
    from run_pg16_ppl_test import ppl_eval_parallel
except ImportError:
    try:
        from scripts.run_pg16_ppl_test import ppl_eval_parallel
    except ImportError:
        print("Warning: Could not import ppl_eval_parallel. PPL testing will be skipped.")
        def ppl_eval_parallel(model, tokenizer, args=None):
            return 0.0

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
             valid_mask = (attention_mask > -1000).expand_as(student_scores)
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
    model_name,
    loss_func:str,
    tokenizer,
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

    # Fix: Initialize as dict of lists, not set comprehension
    loss_history = {i: [] for i in range(model.config.num_hidden_layers)}
    recall_history = {}

    print("Preparing data...")
    c4_data = get_c4_simple(tokenizer, num_calibration_samples, seq_len)
    dataset = TensorDataset(c4_data)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    hidden_states, attention_mask, position_embeddings = _get_layer_inputs(model, dataloader, device, num_samples=num_calibration_samples)

    indices_path = os.path.join("data", model_name, "indices.json")
    print(f"Loading indices from {indices_path}...")

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
                #print(batch_hidden.shape)
                batch_hidden = batch_hidden.to(device)
                cos, sin = position_embeddings
                batch_pos_emb = (cos[:, :batch_hidden.shape[1]], sin[:, :batch_hidden.shape[1]])
                batch_attn_mask = attention_mask[:, :, :batch_hidden.shape[1], :batch_hidden.shape[1]]

                # Must use the full LlamaDecoderLayer forward pass
                layer_output = model.model.layers[i](
                    batch_hidden,
                    attention_mask=batch_attn_mask,
                    position_embeddings=batch_pos_emb
                ) # layer_output is a tuple
                #print(layer_output.shape)
                new_hidden_states_list.append(layer_output.cpu())

        hidden_states = torch.cat(new_hidden_states_list, dim=0)

        del distill_wrapper
        torch.cuda.empty_cache()

    print("\nDistillation Complete!")
    return model, loss_history, recall_history


def measure_ppl(model, tokenizer, device):
    print("Measuring PPL...")
    class Args:
        pass
    args = Args()
    args.device = str(device)
    args.num_eval_tokens = 1024
    try:
        ppl = ppl_eval_parallel(model, tokenizer, args=args)
        print(f"PPL: {ppl}")
        return ppl
    except Exception as e:
        print(f"PPL evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description="Experiment 5: Distillation Effect (MSE vs TopK)")
    parser.add_argument('--model_path', type=str, required=True, help="Path to Llama model")
    parser.add_argument('--output_dir', type=str, default='experiment_5_results', help="Directory to save results")
    parser.add_argument('--num_samples', type=int, default=512, help="Number of calibration samples")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length for calibration and distillation data.")
    parser.add_argument('--steps_per_layer', type=int, default=50, help="Training steps per layer")
    parser.add_argument('--compare', action='store_true', help="Run both MSE and TopK for comparison")
    parser.add_argument('--method', type=str, default='topk', choices=['mse', 'topk'], help="Distillation method if not comparing")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load Tokenizer
    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {}
    model_name = os.path.basename(args.model_path)

    # --- Phase 1: Baseline (Before Distillation) ---
    print(f"\n{'='*40}")
    print(f"Phase 1: Measuring Baseline (Before Distillation)")
    print(f"{'='*40}")

    # We load model just for baseline PPL check
    baseline_model = LlamaForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager"
    )

    # Create Compressed Model (Without Distillation - i.e. just pruned, random/initial small weights)
    # Note: apply_compression_to_model will initialize small projections.
    # If we want to check "before distillation", we just check this state.
    baseline_model = apply_compression_to_model(
        baseline_model,
        model_name,
        quant_mode="int",
        topk_ratio=0.2, # Must match distillation config
        sink_size=0,
        local_window=0,
        escaped_layers=[]
    )

    ppl_before = measure_ppl(baseline_model, tokenizer, baseline_model.device)
    results["before"] = {"ppl": ppl_before}

    del baseline_model
    torch.cuda.empty_cache()

    # Define runs
    if args.compare:
        methods = ["mse", "topk"]
    else:
        methods = [args.method]

    for method in methods:
        print(f"\n\n{'='*40}")
        print(f"Running Distillation with {method.upper()} Loss")
        print(f"{'='*40}")

        # Load Model (Reload for each method to ensure fresh start)
        print(f"Loading model from {args.model_path}...")
        model = LlamaForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager"
        )

        # Run Distillation
        # Capture the distilled model
        distilled_model, loss_hist, recall_hist = collect_layer_wise_distillation(
            model=model,
            model_name=os.path.basename(args.model_path),
            loss_func=method,
            tokenizer=tokenizer,
            train_steps_per_layer=args.steps_per_layer,
            num_calibration_samples=args.num_samples,
            seq_len=args.seq_len
        )

        # Measure PPL After Distillation
        ppl_after = measure_ppl(distilled_model, tokenizer, distilled_model.device)

        results[method] = {
            "loss": loss_hist,
            "recall": recall_hist,
            "ppl": ppl_after
        }

        # Cleanup
        del distilled_model
        torch.cuda.empty_cache()

    # Save Results
    output_file = os.path.join(args.output_dir, "distillation_comparison.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")

    # Plotting
    try:
        plt.figure(figsize=(18, 5))

        # Plot Recall Comparison
        plt.subplot(1, 3, 1)
        for method in methods:
            if method not in results: continue
            data = results[method]
            # Handle both string and int keys from potential json reload or in-memory dict
            keys = list(data["recall"].keys())
            layers = sorted([int(k) for k in keys])

            recalls = []
            for l in layers:
                # Try int key first, then string key
                if l in data["recall"]:
                    recalls.append(data["recall"][l])
                elif str(l) in data["recall"]:
                    recalls.append(data["recall"][str(l)])

            if len(recalls) == len(layers):
                plt.plot(layers, recalls, marker='o', label=f"{method.upper()} Recall")
        plt.xlabel("Layer Index")
        plt.ylabel("Top-K Recall")
        plt.title("Distillation Recall per Layer")
        plt.legend()
        plt.grid(True)

        # Plot Loss (Layer 0)
        plt.subplot(1, 3, 2)
        for method in methods:
            if method not in results: continue
            data = results[method]

            loss_data = None
            if 0 in data["loss"]:
                loss_data = data["loss"][0]
            elif "0" in data["loss"]:
                loss_data = data["loss"]["0"]

            if loss_data:
                plt.plot(loss_data, label=f"{method.upper()} Layer 0 Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Layer 0 Training Loss")
        plt.legend()
        plt.grid(True)

        # Plot PPL Comparison
        plt.subplot(1, 3, 3)
        labels = ["Before"]
        values = [results["before"]["ppl"]]

        for method in methods:
            if method in results:
                labels.append(f"{method.upper()}")
                values.append(results[method]["ppl"])

        # Filter None values
        clean_labels = []
        clean_values = []
        for l, v in zip(labels, values):
            if v is not None:
                clean_labels.append(l)
                clean_values.append(v)

        bars = plt.bar(clean_labels, clean_values)
        plt.ylabel("PPL")
        plt.title("Perplexity Comparison (Lower is Better)")
        plt.grid(axis='y')

        # Add labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "distillation_comparison.png"))
        print(f"Plot saved to {os.path.join(args.output_dir, 'distillation_comparison.png')}")
    except Exception as e:
        print(f"Plotting failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

"""
  python scripts/run_distillation_experiment.py \
      --model_path /home/tgx/models/Llama-3.2-1B \
      --compare \
      --steps_per_layer 50 \
      --output_dir experiment_5_results

"""
