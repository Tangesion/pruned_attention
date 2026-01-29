import torch
import torch.nn as nn
from tqdm import tqdm
import json
import random
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


def apply_compression_to_model(model, indices_path="indices.json", quant_mode=None, topk_ratio=0.1, sink_size=4, local_window=64):
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
    print(f"Loading indices from {indices_path}...")
    with open(indices_path, 'r') as f:
        indices_data = json.load(f)
        
    config = model.config
    config.topk_ratio = topk_ratio
    config.sink_size = sink_size
    config.local_window = local_window
    
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


def run_calibration_and_save_indices(model: LlamaForCausalLM, calibration_data: torch.Tensor, device: torch.device, save_path="indices.json"):
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
    inps = torch.zeros((calibration_data.shape[0], calibration_data.shape[1], model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    
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

    original_layer_0 = layers[0]
    layers[0] = Catcher(layers[0])
    
    try:
        model(calibration_data.to(device))
    except StopIteration:
        pass # Expected exception
    finally:
        layers[0] = original_layer_0
    
    
    inps = cache['hidden_states']
    attention_mask = cache['attention_mask']
    pos_emb_tuple = cache['position_embeddings']
    position_ids = cache['position_ids']

    indices_dict = {} 

    print("Processing layers to calculate compression indices...")
    with torch.no_grad():
        for i in tqdm(range(len(layers))):
            layer = layers[i]
            
            layer_indices = get_compression_indices(
                layer.self_attn, 
                inps, 
                pos_emb_tuple, 
                attention_mask=attention_mask,
                compression_ratio=0.125,
                topk_ratio=0.2
            )
            indices_dict[str(i)] = layer_indices.cpu().tolist()
            
            # Pass the outputs of the current layer as inputs to the next
            inps = layer(
                inps, 
                attention_mask=attention_mask, 
                position_ids=position_ids,
                position_embeddings=pos_emb_tuple
            )[0]
            
            torch.cuda.empty_cache()

    model.config.use_cache = use_cache
    
    with open(save_path, "w") as f:
        json.dump(indices_dict, f, indent=4)
    print(f"All indices saved to {save_path}")
    

def get_c4_simple(tokenizer, n_samples: int, seq_len: int):
    """
    Downloads and prepares a small subset of the C4 dataset for calibration.
    """
    ds_dict = load_dataset("allenai/c4", data_files={"train": "en/c4-train.00000-of-01024.json.gz"}, split="train")
    
    tokenized_samples = []
    
    for example in ds_dict.shuffle(seed=42).select(range(n_samples * 2)): # Oversample to ensure we get enough long samples
        if len(tokenized_samples) == n_samples:
            break
        
        text = example["text"]
        tokens = tokenizer(text, return_tensors="pt", max_length=seq_len, truncation=True).input_ids
        if tokens.shape[1] == seq_len:
            tokenized_samples.append(tokens)

    if len(tokenized_samples) < n_samples:
        raise ValueError(f"Could not find {n_samples} samples with length {seq_len}. Found {len(tokenized_samples)}.")

    return torch.cat(tokenized_samples, dim=0)
