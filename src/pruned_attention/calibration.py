import torch
import torch.nn as nn
from tqdm import tqdm
import json
import random
import os
from datasets import load_dataset
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from .attention import CompressedLlamaAttention, repeat_kv


@torch.no_grad()
def get_compression_indices(
    model_layer, 
    hidden_states, 
    position_embeddings, 
    attention_mask=None,
    compression_ratio=0.125,
    topk_ratio=0.1  
):
    """
    Calculates the most important dimension indices to keep for KV-cache compression.

    This function performs a sensitivity analysis on the head dimensions of a single
    attention layer to determine which dimensions have the most impact on the attention
    scores of the most important query-key pairs.

    Args:
        model_layer: The LlamaAttention layer to analyze.
        hidden_states (torch.Tensor): Input hidden states for the layer.
        position_embeddings (tuple): A tuple of (cos, sin) for rotary embeddings.
        attention_mask (torch.Tensor, optional): The attention mask.
        compression_ratio (float, optional): The ratio of dimensions to keep. Defaults to 0.125.
        topk_ratio (float, optional): The ratio of important QK pairs to focus on. Defaults to 0.1.

    Returns:
        torch.Tensor: A tensor of shape [Num_KV_Heads, Compressed_Dim] containing the
                      indices of the dimensions to keep.
    """
    bsz, q_len, _ = hidden_states.shape
    config = model_layer.config
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    num_kv_groups = num_heads // num_kv_heads
    head_dim = model_layer.head_dim
    device = hidden_states.device
    
    query_states = model_layer.q_proj(hidden_states).view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    key_states = model_layer.k_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    
    key_states_expanded = repeat_kv(key_states, num_kv_groups)
    full_attn_scores = torch.matmul(query_states, key_states_expanded.transpose(2, 3)) / (head_dim**0.5)
    
    if attention_mask is not None:
        full_attn_scores = full_attn_scores + attention_mask
    
    k_val = max(1, int(q_len * topk_ratio))
    topk_thresh = torch.topk(full_attn_scores, k_val, dim=-1).values[..., -1, None]
    topk_mask = (full_attn_scores >= topk_thresh).float()
    
    num_pairs = head_dim // 2
    q_head_sensitivity = torch.zeros(num_heads, num_pairs, device=device)
    
    for i in range(num_pairs):
        idx_start, idx_end = 2 * i, 2 * i + 2
        q_slice = query_states[..., idx_start:idx_end]
        k_slice = key_states_expanded[..., idx_start:idx_end]
        pair_score_contribution = torch.matmul(q_slice, k_slice.transpose(2, 3)) / (head_dim**0.5)
        impact = (pair_score_contribution * topk_mask).abs().sum(dim=(0, 2, 3)) 
        q_head_sensitivity[:, i] = impact

    kv_pair_importance = torch.zeros(num_kv_heads, num_pairs, device=device)
    for kv_i in range(num_kv_heads):
        q_start, q_end = kv_i * num_kv_groups, (kv_i + 1) * num_kv_groups
        kv_pair_importance[kv_i] = q_head_sensitivity[q_start:q_end].sum(dim=0)

    num_keep_pairs = int(num_pairs * compression_ratio)
    _, top_pair_indices = torch.topk(kv_pair_importance, num_keep_pairs, dim=1)
    top_pair_indices = torch.sort(top_pair_indices, dim=1).values

    compressed_dim = num_keep_pairs * 2
    keep_indices = torch.zeros(num_kv_heads, compressed_dim, dtype=torch.long, device=device)
    
    for i in range(num_kv_heads):
        pairs = top_pair_indices[i]
        cols = torch.cat([pairs * 2, pairs * 2 + 1], dim=-1)
        keep_indices[i] = torch.sort(cols).values

    return keep_indices


def apply_compression_to_model(model, model_name, quant_mode=None, topk_ratio=0.1, sink_size=4, local_window=64, escaped_layers=[]):
    """
    Replaces the standard LlamaAttention layers in a model with CompressedLlamaAttention
    layers, configured with pre-calculated indices.

    Args:
        model: The LlamaForCausalLM model to modify.
        indices_path (str, optional): Path to the JSON file with dimension indices.
        quant_mode (str, optional): Quantization mode for the compressed attention.
    
    Returns:
        The modified model.
    """
    
    indices_path = os.path.join("data", model_name, "indices.json")
    
    print(f"Loading indices from {indices_path}...")
    with open(indices_path, 'r') as f:
        indices_data = json.load(f)
        
    config = model.config
    config.topk_ratio = topk_ratio
    config.sink_size = sink_size
    config.local_window = local_window
    config.escaped_layers = escaped_layers
    
    for i, layer in enumerate(tqdm(model.model.layers, desc="Replacing Attention Layers")):
        layer_idx_str = str(i)
        if layer_idx_str not in indices_data:
            print(f"Warning: No indices found for layer {i}, skipping.")
            continue
            
        keep_indices = torch.tensor(indices_data[layer_idx_str], dtype=torch.long).to(model.device)
        
        compressed_attn = CompressedLlamaAttention(
            config, 
            layer.self_attn, 
            group_keep_indices=keep_indices,
            mode=quant_mode
        ).to(model.device)
        
        model.model.layers[i].self_attn = compressed_attn
        
    print("Model compression applied successfully!")
    return model


def run_calibration_and_save_indices(model: LlamaForCausalLM, calibration_data: torch.Tensor, device: torch.device, model_name="llama3"):
    """
    Runs the calibration process and saves the calculated dimension indices to a file.
    
    This function hooks into the model to capture intermediate activations, calculates
    the importance of each dimension in the attention layers, and saves the indices
of
    the most important dimensions to a JSON file.

    Args:
        model (LlamaForCausalLM): The model to calibrate.
        calibration_data (torch.Tensor): A batch of tokenized text for calibration.
        device (torch.device): The device to run calibration on.
        save_path (str, optional): Path to save the output indices JSON file.
    """
    print("Preparing calibration inputs...")
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    
    dtype = next(iter(model.parameters())).dtype
    # calibration_data: [Batch, Seq] (token ids)
    inps = torch.zeros((calibration_data.shape[0], calibration_data.shape[1], model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

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
    pos_emb_tuple = cache['position_embeddings']
    

    indices_dict = {} 

    print("Processing layers to calculate compression indices...")
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        
        sample_input = inps.to(device) # [bsz, Seq, Hidden]
        
        layer_indices = get_compression_indices(
            layer.self_attn, 
            sample_input, 
            pos_emb_tuple, 
            attention_mask=attention_mask,
            compression_ratio=0.125,
            topk_ratio=0.2
        )
        indices_dict[str(i)] = layer_indices.cpu().tolist()
        
        for j in range(calibration_data.shape[0]):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0), 
                    attention_mask=attention_mask, 
                    position_ids=position_ids,
                    position_embeddings=pos_emb_tuple
                )[0]
        
        # Swap buffers
        inps.copy_(outs) 
        
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache
    
    base_dirname = "data"
    os.makedirs(base_dirname, exist_ok=True)
    model_dirname = os.path.join(base_dirname, model_name)
    os.makedirs(model_dirname, exist_ok=True)
    save_path = os.path.join(model_dirname, "indices.json")
    with open(save_path, "w") as f:
        json.dump(indices_dict, f, indent=4)
    print(f"All indices saved to {save_path}")
    

def get_c4_simple(tokenizer, n_samples, seq_len):
    ds_dict = load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"}
    )
    ds = ds_dict["train"]
    n_items = len(ds)

    tokenized_samples, used_indices = [], set()

    for _ in range(n_samples):
        while True:
            sample_idx = random.randint(0, n_items - 1)
            if sample_idx in used_indices:
                continue
            text = ds[sample_idx]["text"]
            tokenized_sample = tokenizer(text, return_tensors="pt")
            if tokenized_sample.input_ids.shape[1] >= seq_len:
                used_indices.add(sample_idx)
                break

        max_start = tokenized_sample.input_ids.shape[1] - seq_len
        start = random.randint(0, max_start)
        tokenized_samples.append(
            tokenized_sample.input_ids[:, start:start + seq_len]
        )

    return torch.cat(tokenized_samples, dim=0)
