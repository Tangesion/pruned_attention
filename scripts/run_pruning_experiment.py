import torch
import torch.nn as nn
import argparse
import os
import sys
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pruned_attention.calibration import get_c4_simple, get_compression_indices
from src.pruned_attention.attention import repeat_kv

def get_magnitude_indices(model_layer, compression_ratio=0.125):
    """
    Selects indices based on the magnitude (L1 norm) of the weights.
    Respects RoPE pairs.
    """
    config = model_layer.config
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    num_kv_groups = num_heads // num_kv_heads
    head_dim = model_layer.head_dim
    device = model_layer.q_proj.weight.device

    # Weights: [Out, In] -> [Hidden, Heads, HeadDim]
    # We want magnitude of columns (input features to the projection?)
    # Wait, the compression gathers columns from the projected output (HeadDim).
    # In attention.py:
    #   full_wk = k_proj.weight.T.view(Hidden, KV_Heads, Head_Dim)
    #   small_wk = gather(full_wk, 2, indices)
    # So we are selecting dimensions in the Head_Dim.
    # So we should look at the norm of the columns in the weight matrix corresponding to these output dimensions.
    # W_k shape in HF is [Out, In] = [Num_KV * HeadDim, Hidden]
    # Transpose to [Hidden, Num_KV * HeadDim]
    # View as [Hidden, Num_KV, HeadDim]
    # We want to select indices in dim 2.
    # Measure importance by L1 norm of the column vector (dim 0).

    with torch.no_grad():
        w_q = model_layer.q_proj.weight.T.view(config.hidden_size, num_heads, head_dim)
        w_k = model_layer.k_proj.weight.T.view(config.hidden_size, num_kv_heads, head_dim)

        # Calculate L1 norm along Hidden dimension
        q_norm = w_q.abs().sum(dim=0) # [Num_Q, HeadDim]
        k_norm = w_k.abs().sum(dim=0) # [Num_KV, HeadDim]

        # Aggregate Q norms to KV groups
        q_norm_grouped = q_norm.view(num_kv_heads, num_kv_groups, head_dim).sum(dim=1) # [Num_KV, HeadDim]

        # Total importance = Q_norm + K_norm
        total_importance = q_norm_grouped + k_norm # [Num_KV, HeadDim]

        # Handle RoPE pairs
        num_pairs = head_dim // 2
        pair_importance = torch.zeros(num_kv_heads, num_pairs, device=device)

        for i in range(num_pairs):
            pair_importance[:, i] = total_importance[:, 2*i] + total_importance[:, 2*i+1]

        # Select Top-K pairs
        num_keep_pairs = int(num_pairs * compression_ratio)
        _, top_pair_indices = torch.topk(pair_importance, num_keep_pairs, dim=1)
        top_pair_indices = torch.sort(top_pair_indices, dim=1).values

        compressed_dim = num_keep_pairs * 2
        keep_indices = torch.zeros(num_kv_heads, compressed_dim, dtype=torch.long, device=device)

        for i in range(num_kv_heads):
            pairs = top_pair_indices[i]
            cols = torch.cat([pairs * 2, pairs * 2 + 1], dim=-1)
            keep_indices[i] = torch.sort(cols).values

        return keep_indices

def get_random_indices(model_layer, compression_ratio=0.125):
    """
    Selects indices randomly. Respects RoPE pairs.
    """
    config = model_layer.config
    num_kv_heads = config.num_key_value_heads
    head_dim = model_layer.head_dim
    device = model_layer.q_proj.weight.device

    num_pairs = head_dim // 2
    num_keep_pairs = int(num_pairs * compression_ratio)
    compressed_dim = num_keep_pairs * 2

    keep_indices = torch.zeros(num_kv_heads, compressed_dim, dtype=torch.long, device=device)

    for i in range(num_kv_heads):
        # Random sample
        perm = torch.randperm(num_pairs, device=device)[:num_keep_pairs]
        pairs = torch.sort(perm).values
        cols = torch.cat([pairs * 2, pairs * 2 + 1], dim=-1)
        keep_indices[i] = torch.sort(cols).values

    return keep_indices

@torch.no_grad()
def calculate_recall(
    model_layer,
    hidden_states,
    position_embeddings,
    attention_mask,
    indices,
    topk_ratio=0.1
):
    """
    Simulates the compressed attention and calculates Top-K Recall against Full Attention.
    """
    bsz, q_len, _ = hidden_states.shape
    config = model_layer.config
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    num_kv_groups = num_heads // num_kv_heads
    head_dim = model_layer.head_dim

    # 1. Full Attention Scores (Ground Truth)
    query_states = model_layer.q_proj(hidden_states).view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    key_states = model_layer.k_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    key_states_expanded = repeat_kv(key_states, num_kv_groups)
    full_scores = torch.matmul(query_states, key_states_expanded.transpose(2, 3)) / (head_dim**0.5)

    if attention_mask is not None:
        full_scores = full_scores + attention_mask

    k_val = max(1, int(q_len * topk_ratio))
    _, full_topk_indices = torch.topk(full_scores, k_val, dim=-1) # [B, H, Q, K]

    # 2. Compressed Attention Scores (Prediction)
    # We need to gather the weights and compute proxy scores
    # Note: For efficiency in this test script, we can just gather the projected states
    # This is mathematically equivalent to gathering weights if we ignore quantization for now.

    # Indices: [Num_KV, Compressed_Dim]
    # We need to apply these to Q and K

    # K: [B, Num_KV, Seq, Head_Dim]
    # Q: [B, Num_Q, Seq, Head_Dim]

    # Expand indices for K
    # [1, Num_KV, 1, Comp_Dim]
    k_indices_expanded = indices.unsqueeze(0).unsqueeze(2).expand(bsz, -1, q_len, -1)
    k_small = torch.gather(key_states, 3, k_indices_expanded)

    # Expand indices for Q
    # indices: [Num_KV, Comp_Dim] -> [Num_Q, Comp_Dim]
    q_indices = indices.repeat_interleave(num_kv_groups, dim=0)
    q_indices_expanded = q_indices.unsqueeze(0).unsqueeze(2).expand(bsz, -1, q_len, -1)
    q_small = torch.gather(query_states, 3, q_indices_expanded)

    # Calculate Proxy Scores
    # Note: We are using the ROTATED states here.
    # In the actual implementation, we rotate AFTER gathering (Mixed RoPE).
    # However, for measuring "Recall" of the *importance*, doing it on rotated states
    # is a reasonable approximation for the "effectiveness of information retention".
    # BUT, to be strictly faithful to the method, we should gather from unrotated and then apply Mixed RoPE?
    # The paper says: "apply Mixed RoPE...".
    # For this experiment script, let's stick to the simpler approximation of gathering from already rotated states
    # if it simplifies things, BUT checking attention.py:
    # It gathers from weights, then applies Mixed RoPE.
    # Mixed RoPE gathers cos/sin.
    # Let's try to be accurate.

    # Re-project unrotated
    q_unrot = model_layer.q_proj(hidden_states).view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    k_unrot = model_layer.k_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

    # Gather
    q_small = torch.gather(q_unrot, 3, q_indices_expanded)
    k_small = torch.gather(k_unrot, 3, k_indices_expanded)

    # Mixed RoPE
    # We need to gather cos/sin
    # cos, sin: [B, Seq, HeadDim] -> [B, 1, Seq, HeadDim] -> [B, Heads, Seq, HeadDim]
    # Actually apply_rotary_pos_emb handles broadcasting.
    # We need to gather the relevant dimensions from cos/sin

    # cos: [B, Seq, HeadDim]
    # We need to select dimensions.
    # Note: RoPE is applied to pairs. Our indices are pair-aligned.
    # So we can just gather from cos/sin.

    # Indices are different for each head!
    # cos/sin are shared across heads usually (unless it's a specific variant).
    # In Llama, cos/sin is [B, Seq, HeadDim].
    # We have [B, Heads, Seq, CompDim].
    # This is tricky because cos/sin is broadcasted over heads.
    # But here each head keeps different dimensions.
    # So we must expand cos/sin to [B, Heads, Seq, HeadDim] and then gather.

    # Expand to batch size
    cos_exp = cos.unsqueeze(1).expand(bsz, num_heads, -1, -1) # [B, H, S, D]
    sin_exp = sin.unsqueeze(1).expand(bsz, num_heads, -1, -1)

    cos_small = torch.gather(cos_exp, 3, q_indices_expanded)
    sin_small = torch.gather(sin_exp, 3, q_indices_expanded)

    def rotate_half(x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    q_small_rot = (q_small * cos_small) + (rotate_half(q_small) * sin_small)

    # For K, we need to handle KV heads
    cos_exp_k = cos.unsqueeze(1).expand(bsz, num_kv_heads, -1, -1)
    sin_exp_k = sin.unsqueeze(1).expand(bsz, num_kv_heads, -1, -1)
    cos_small_k = torch.gather(cos_exp_k, 3, k_indices_expanded)
    sin_small_k = torch.gather(sin_exp_k, 3, k_indices_expanded)

    k_small_rot = (k_small * cos_small_k) + (rotate_half(k_small) * sin_small_k)

    # Expand K
    k_small_rot = repeat_kv(k_small_rot, num_kv_groups)

    # Score
    proxy_scores = torch.matmul(q_small_rot, k_small_rot.transpose(2, 3)) / (q_small_rot.shape[-1]**0.5)

    if attention_mask is not None:
        proxy_scores = proxy_scores + attention_mask

    _, pred_topk_indices = torch.topk(proxy_scores, k_val, dim=-1)

    # 3. Recall Calculation
    # Intersection of indices
    # We can use scatter to count matches

    # Convert indices to masks? No, K is small.
    # Just check overlap.
    # full_topk_indices: [B, H, Q, K]
    # pred_topk_indices: [B, H, Q, K]

    # We can gather values from a dummy tensor or use specialized recall func
    # A simple way for small tensors:
    # Expand and compare? [..., K, 1] == [..., 1, K] -> Any match

    # B, H, Q, K_val
    recall_sum = 0
    total_k = full_topk_indices.numel()

    # Iterate or vectorized? Vectorized is better.
    # [B, H, Q, K, 1] == [B, H, Q, 1, K] -> [B, H, Q, K, K] (bool)
    # Any match along last dim -> count
    matches = (full_topk_indices.unsqueeze(-1) == pred_topk_indices.unsqueeze(-2)).any(dim=-1).sum()

    recall = matches.item() / total_k
    return recall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to Llama model")
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--compression_ratio", type=float, default=0.125)
    parser.add_argument("--topk_ratio", type=float, default=0.1)
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    model = LlamaForCausalLM.from_pretrained(args.model_path, device_map="auto", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    print("Generating calibration data...")
    calibration_data = get_c4_simple(tokenizer, args.num_samples, args.seq_len)

    # Capture inputs
    print("Capturing layer inputs...")
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    device = model.device
    dtype = next(iter(model.parameters())).dtype

    # Storage for inputs
    # We only need to process one batch of samples really, or loop them.
    # Let's stick to the structure of calibration.py but simplify.

    inps = torch.zeros((calibration_data.shape[0], calibration_data.shape[1], model.config.hidden_size), dtype=dtype, device=device)
    cache = {'i': 0, 'attention_mask': None, 'position_ids': None, 'position_embeddings': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs.get('attention_mask')
            cache['position_ids'] = kwargs.get('position_ids')
            cache['position_embeddings'] = kwargs.get('position_embeddings')
            raise ValueError

    layers[0] = Catcher(layers[0])

    for i in range(calibration_data.shape[0]):
        try:
            model(calibration_data[i].unsqueeze(0).to(device))
        except ValueError:
            pass

    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    pos_emb = cache['position_embeddings']

    # Experiment
    results = {
        'ours': [],
        'magnitude': [],
        'random': []
    }

    print("Running Experiment 4: Pruning Strategy Effectiveness...")

    for i in tqdm(range(len(layers)), desc="Layers"):
        layer = layers[i]
        sample_input = inps.to(device)

        # 1. Ours (Sensitivity)
        idx_ours = get_compression_indices(
            layer.self_attn,
            sample_input,
            pos_emb,
            attention_mask=attention_mask,
            compression_ratio=args.compression_ratio,
            topk_ratio=args.topk_ratio
        )
        rec_ours = calculate_recall(layer.self_attn, sample_input, pos_emb, attention_mask, idx_ours, args.topk_ratio)
        results['ours'].append(rec_ours)

        # 2. Magnitude
        idx_mag = get_magnitude_indices(layer.self_attn, args.compression_ratio)
        rec_mag = calculate_recall(layer.self_attn, sample_input, pos_emb, attention_mask, idx_mag, args.topk_ratio)
        results['magnitude'].append(rec_mag)

        # 3. Random
        idx_rand = get_random_indices(layer.self_attn, args.compression_ratio)
        rec_rand = calculate_recall(layer.self_attn, sample_input, pos_emb, attention_mask, idx_rand, args.topk_ratio)
        results['random'].append(rec_rand)

        # Forward to next layer
        for j in range(calibration_data.shape[0]):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=pos_emb
                )[0]
        inps.copy_(outs)
        torch.cuda.empty_cache()

    print("\n=== Experiment 4 Results ===")
    print(f"Top-K Ratio: {args.topk_ratio}, Compression Ratio: {args.compression_ratio}")
    print(f"Ours (Sensitivity) Average Recall: {np.mean(results['ours']):.4f}")
    print(f"Magnitude-based Average Recall:    {np.mean(results['magnitude']):.4f}")
    print(f"Random Average Recall:             {np.mean(results['random']):.4f}")

    # Save results
    os.makedirs("experiment_pruning_results", exist_ok=True)
    save_path = os.path.join("experiment_pruning_results", "pruning_results.json")
    import json
    with open(save_path, 'w') as f:
        json.dump(results, f)
    print(f"Detailed results saved to {save_path}")

if __name__ == "__main__":
    main()
