import argparse
import torch
import torch.nn as nn
import json
import random
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from transformers import LlamaForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

# Import from package
from pruned_attention.calibration import get_compression_indices, get_c4_simple, repeat_kv

def calculate_recall(full_scores, compressed_scores, k):
    """
    Calculates the recall of the top-k indices.
    """
    # Get Top-K indices from full scores (Ground Truth)
    _, full_topk_indices = torch.topk(full_scores, k, dim=-1)

    # Get Top-K indices from compressed scores (Prediction)
    _, compressed_topk_indices = torch.topk(compressed_scores, k, dim=-1)

    # Calculate intersection
    # We expand dims to broadcast for comparison
    # full: [..., K, 1], compressed: [..., 1, K] -> match: [..., K, K]
    # This is memory intensive for large batches, so we do it iteratively or carefully

    bsz, num_heads, q_len, _ = full_scores.shape

    recall_sum = 0
    total_count = 0

    # Iterate to save memory
    for b in range(bsz):
        for h in range(num_heads):
            # sets of indices
            ft = full_topk_indices[b, h] # [q_len, k]
            ct = compressed_topk_indices[b, h] # [q_len, k]

            # Simple way: use sets for each query
            for q in range(q_len):
                gt_set = set(ft[q].tolist())
                pred_set = set(ct[q].tolist())
                intersection = len(gt_set.intersection(pred_set))
                recall_sum += intersection / k
                total_count += 1

    return recall_sum / total_count

@torch.no_grad()
def evaluate_strategies(model, dataloader, device, compression_ratio=0.125, topk_ratio=0.1):
    model.eval()
    layers = model.model.layers
    config = model.config
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // num_heads
    num_kv_groups = num_heads // num_kv_heads

    # Storage for results
    results = {
        "Sensitivity": [],
        "Magnitude": [],
        "Random": []
    }

    # Hook to catch inputs
    cache = {}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            cache['hidden_states'] = inp
            cache['attention_mask'] = kwargs.get('attention_mask')
            cache['position_ids'] = kwargs.get('position_ids')
            cache['position_embeddings'] = kwargs.get('position_embeddings')
            raise StopIteration

    # We only process a few layers to save time for this experiment script
    # Or we can process all. Let's process a subset (e.g., layers 0, 10, 20, 31) or middle layers
    # For a robust experiment, we should do all, but for demonstration:
    target_layers = list(range(0, len(layers), 4)) # Every 4th layer

    print(f"Evaluating on {len(target_layers)} layers...")

    for batch in tqdm(dataloader, desc="Batches"):
        batch = batch.to(device)
        # We need to get layer inputs.
        # Since we can't easily hook all at once without running full model multiple times,
        # we will run the model layer by layer.

        hidden_states = batch
        attention_mask = None # Simplified for C4 chunks
        position_ids = torch.arange(batch.shape[1], device=device).unsqueeze(0)

        # Get position embeddings (Rotary)
        # We need to manually generate them or get them from the model
        # Llama usually generates them inside the model.
        # Let's run the model up to the first layer to get common args

        # Helper to get layer inputs
        original_layer_0 = layers[0]
        layers[0] = Catcher(layers[0])
        try:
            model(batch)
        except StopIteration:
            pass
        layers[0] = original_layer_0

        hidden_states = cache['hidden_states']
        attention_mask = cache['attention_mask']
        position_ids = cache['position_ids']
        position_embeddings = cache['position_embeddings']

        # Loop through all layers
        for i, layer in enumerate(layers):
            if i in target_layers:
                # --- EVALUATE THIS LAYER ---

                # 1. Prepare Q, K, V
                bsz, q_len, _ = hidden_states.shape
                q_proj = layer.self_attn.q_proj(hidden_states).view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
                k_proj = layer.self_attn.k_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

                # Use the captured position embeddings
                if position_embeddings is None:
                    # Fallback if not captured (should not happen if Catcher works)
                     print("Warning: position_embeddings is None!")
                     cos, sin = None, None
                else:
                    cos, sin = position_embeddings

                # Apply RoPE for our metrics
                q_states, k_states = apply_rotary_pos_emb(q_proj, k_proj, cos, sin)

                # Full Attention Scores (Ground Truth)
                k_states_expanded = repeat_kv(k_states, num_kv_groups)
                full_scores = torch.matmul(q_states, k_states_expanded.transpose(2, 3)) / (head_dim**0.5)
                if attention_mask is not None:
                    full_scores = full_scores + attention_mask

                k_val = max(1, int(q_len * topk_ratio))

                # --- Strategy 1: Sensitivity (Ours) ---
                ours_indices = get_compression_indices(
                    layer.self_attn,
                    hidden_states,
                    position_embeddings,
                    attention_mask,
                    compression_ratio=compression_ratio,
                    topk_ratio=topk_ratio
                )

                # Calculate Recall for Ours
                recall_ours = evaluate_indices(q_states, k_states_expanded, full_scores, ours_indices, k_val, head_dim, num_kv_groups)
                results["Sensitivity"].append(recall_ours)

                # --- Strategy 2: Magnitude-based ---
                # Calculate magnitude of pairs in K (and Q).
                # Usually we care about Key importance for retrieval, or QK interaction.
                # "Magnitude-based pruning" usually retains weights/activations with largest L2 norm.
                # We calculate L2 norm of each feature pair across the batch and sequence.

                num_pairs = head_dim // 2
                pair_magnitudes = torch.zeros(num_kv_heads, num_pairs, device=device)

                # Use Key magnitude as a proxy for information content (common baseline)
                # Or sum of Q and K magnitudes. Let's use K magnitude.
                for p in range(num_pairs):
                    idx_start, idx_end = 2 * p, 2 * p + 2
                    k_slice = k_states[..., idx_start:idx_end] # [bsz, kv_heads, seq, 2]
                    pair_magnitudes[:, p] = k_slice.norm(dim=-1).mean(dim=(0, 2)) # Avg L2 norm over Batch and Seq

                num_keep = int(num_pairs * compression_ratio)
                _, mag_top_pairs = torch.topk(pair_magnitudes, num_keep, dim=1)

                mag_indices = torch.zeros(num_kv_heads, num_keep * 2, dtype=torch.long, device=device)
                for h in range(num_kv_heads):
                    pairs = mag_top_pairs[h]
                    cols = torch.cat([pairs * 2, pairs * 2 + 1], dim=-1)
                    mag_indices[h] = torch.sort(cols).values

                recall_mag = evaluate_indices(q_states, k_states_expanded, full_scores, mag_indices, k_val, head_dim, num_kv_groups)
                results["Magnitude"].append(recall_mag)

                # --- Strategy 3: Random ---
                rnd_indices = torch.zeros(num_kv_heads, num_keep * 2, dtype=torch.long, device=device)
                all_pairs = list(range(num_pairs))
                for h in range(num_kv_heads):
                    pairs = torch.tensor(random.sample(all_pairs, num_keep), device=device)
                    cols = torch.cat([pairs * 2, pairs * 2 + 1], dim=-1)
                    rnd_indices[h] = torch.sort(cols).values

                recall_rnd = evaluate_indices(q_states, k_states_expanded, full_scores, rnd_indices, k_val, head_dim, num_kv_groups)
                results["Random"].append(recall_rnd)

            # Forward pass for next layer
            try:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings
                )[0]
            except RuntimeError as e:
                print(f"Error in layer {i}: {e}")
                print(f"hidden_states: {hidden_states.shape}")
                if position_embeddings:
                    print(f"cos shape: {position_embeddings[0].shape}")
                raise e

    # Aggregate results
    avg_results = {k: sum(v)/len(v) for k, v in results.items()}
    return avg_results, results

def evaluate_indices(q_states, k_states_expanded, full_scores, indices, k_val, head_dim, num_kv_groups):
    """Helper to compute recall given indices."""
    bsz, num_heads, q_len, _ = q_states.shape

    # Select dimensions
    # Indices are per KV head. We need to expand for Query heads.
    # indices shape: [num_kv_heads, compressed_dim]

    # To do this efficiently without complex gathering:
    # We can mask the unselected dimensions with 0.

    mask_q = torch.zeros_like(q_states)
    mask_k = torch.zeros_like(k_states_expanded)

    num_kv_heads = indices.shape[0]

    for kv_h in range(num_kv_heads):
        idx = indices[kv_h]
        # Map KV head to corresponding Q heads
        q_start = kv_h * num_kv_groups
        q_end = (kv_h + 1) * num_kv_groups

        mask_q[:, q_start:q_end, :, idx] = 1
        mask_k[:, q_start:q_end, :, idx] = 1 # k_states_expanded has num_heads dim

    q_pruned = q_states * mask_q
    k_pruned = k_states_expanded * mask_k

    pruned_scores = torch.matmul(q_pruned, k_pruned.transpose(2, 3)) / (head_dim**0.5)

    # We don't add attention mask here for ranking comparison usually,
    # but consistency with full_scores matters.
    # If full_scores had mask, pruned should too.
    # But usually mask is -inf, which stays -inf.
    # The relative order of valid tokens matters.

    return calculate_recall(full_scores, pruned_scores, k_val)

def main():
    parser = argparse.ArgumentParser(description="Run pruning strategy comparison experiment")
    parser.add_argument("--model_path", type=str, required=True, help="Path to Llama model")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of calibration samples")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length")
    parser.add_argument("--compression_ratio", type=float, default=0.125, help="Compression ratio (e.g. 1/8)")
    parser.add_argument("--save_path", type=str, default="experiment_results.json", help="Path to save results")
    parser.add_argument("--plot_path", type=str, default="pruning_comparison.png", help="Path to save plot")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model from {args.model_path}...")
    model = LlamaForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    print("Loading calibration data...")
    calib_data = get_c4_simple(tokenizer, args.num_samples, args.seq_len)
    # Batch it
    dataloader = torch.utils.data.DataLoader(calib_data, batch_size=1, shuffle=False) # Batch size 1 to save memory

    print("Running evaluation...")
    avg_results, raw_results = evaluate_strategies(
        model,
        dataloader,
        device,
        compression_ratio=args.compression_ratio
    )

    print("\nResults (Top-K Recall):")
    for strategy, score in avg_results.items():
        print(f"{strategy}: {score:.4f}")

    # Save results
    with open(args.save_path, 'w') as f:
        json.dump({"average": avg_results, "raw": raw_results}, f, indent=4)

    # Plotting
    strategies = list(avg_results.keys())
    scores = list(avg_results.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(strategies, scores, color=['#2ca02c', '#1f77b4', '#ff7f0e'])
    plt.ylim(0, 1.1)
    plt.ylabel(f"Top-K Recall (Ratio={args.compression_ratio})")
    plt.title(f"Pruning Strategy Comparison\n(Seq Len: {args.seq_len}, Model: {os.path.basename(args.model_path)})")

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}',
                 ha='center', va='bottom')

    plt.savefig(args.plot_path)
    print(f"Plot saved to {args.plot_path}")

if __name__ == "__main__":
    main()
