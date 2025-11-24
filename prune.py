import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, LlamaRotaryEmbedding
from Llama import CompressedLlamaAttention
import json
import random
from datasets import load_dataset



def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


@torch.no_grad()
def get_compression_indices(
    model_layer, 
    hidden_states, 
    position_embeddings, 
    attention_mask=None,
    compression_ratio=0.125,
    topk_ratio=0.1  
):

    bsz, q_len, _ = hidden_states.shape
    config = model_layer.config
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    num_kv_groups = num_heads // num_kv_heads
    head_dim = model_layer.head_dim
    device = hidden_states.device
    
    # [Batch, Seq, Hidden] -> [Batch, Heads, Seq, Dim]
    query_states = model_layer.q_proj(hidden_states).view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    key_states = model_layer.k_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(
        query_states,  # [B, num_heads, S, D]
        key_states,    # [B, num_kv_heads, S, D]
        cos,
        sin,
    )
    
    # [Batch, Num_Q_Heads, Seq, Dim]
    key_states_expanded = repeat_kv(key_states, num_kv_groups)

    # [Batch, Num_Q_Heads, Seq, Seq]
    full_attn_scores = torch.matmul(query_states, key_states_expanded.transpose(2, 3)) / (head_dim**0.5)
    
    if attention_mask is not None:
        full_attn_scores = full_attn_scores + attention_mask
    
    k_val = max(1, int(q_len * topk_ratio))
    topk_thresh = torch.topk(full_attn_scores, k_val, dim=-1).values[..., -1, None]
    topk_mask = (full_attn_scores >= topk_thresh).float()

    
    num_pairs = head_dim // 2
    q_head_sensitivity = torch.zeros(num_heads, num_pairs, device=device)
    
    for i in range(num_pairs):
        # [Batch, Heads, Seq, 2]
        idx_start = 2 * i
        idx_end = 2 * i + 2
        
        q_slice = query_states[..., idx_start:idx_end]
        k_slice = key_states_expanded[..., idx_start:idx_end]
        
        # [Batch, Heads, Seq, 2] @ [Batch, Heads, 2, Seq] -> [Batch, Heads, Seq, Seq]
        pair_score_contribution = torch.matmul(q_slice, k_slice.transpose(2, 3)) / (head_dim**0.5)
        
        impact = (pair_score_contribution * topk_mask).abs().sum(dim=(0, 2, 3)) 
        
        q_head_sensitivity[:, i] = impact

    kv_pair_importance = torch.zeros(num_kv_heads, num_pairs, device=device)  # (num_kv_heads, num_pairs)
    
    for kv_i in range(num_kv_heads):
        q_start = kv_i * num_kv_groups
        q_end = (kv_i + 1) * num_kv_groups
        # Sum across the group of Query heads
        kv_pair_importance[kv_i] = q_head_sensitivity[q_start:q_end].sum(dim=0)

    num_keep_pairs = int(num_pairs * compression_ratio)

    # top_pair_indices[Num_KV_Heads, Num_Keep_Pairs]
    _, top_pair_indices = torch.topk(kv_pair_importance, num_keep_pairs, dim=1)
    
    top_pair_indices = torch.sort(top_pair_indices, dim=1).values

    # [Num_KV_Heads, Compressed_Dim]
    #  Group 0  [0,1, 10,11], Group 1  [2,3, 10,11]
    compressed_dim = num_keep_pairs * 2
    keep_indices = torch.zeros(num_kv_heads, compressed_dim, dtype=torch.long, device=device)
    
    for i in range(num_kv_heads):
        pairs = top_pair_indices[i]
        #col: [p1, p2] -> [2*p1, 2*p1+1, 2*p2, 2*p2+1]
        cols = torch.cat([pairs * 2, pairs * 2 + 1], dim=-1)
        keep_indices[i] = torch.sort(cols).values

    return keep_indices # [Num_KV_Heads, Compressed_Dim]


def apply_compression_to_model(model, indices_path="indices.json"):
    print(f"Loading indices from {indices_path}...")
    with open(indices_path, 'r') as f:
        indices_data = json.load(f)
        
    config = model.config
    
    for i, layer in enumerate(tqdm(model.model.layers, desc="Replacing Attention Layers")):
        layer_idx = str(i)
        if layer_idx not in indices_data:
            continue
            
        keep_indices = torch.tensor(indices_data[layer_idx], dtype=torch.long).to(model.device)
        compressed_attn = CompressedLlamaAttention(
            config, 
            layer.self_attn, 
            group_keep_indices=keep_indices
        ).to(model.device)
        
        model.model.layers[i].self_attn = compressed_attn
        
    print("Model compression applied successfully!")
    return model

def prepare_calibration_input(model, data, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    dtype = next(iter(model.parameters())).dtype
    inps_cache = []
    inps = torch.zeros((data.shape[0], data.shape[1], model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}
    
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_embeddings'] = kwargs['position_embeddings']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(data.shape[0]):
        try:
            model(data[i].unsqueeze(0).to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache
    
    print("Getting activations for 32 layers...")
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        for j in range(data.shape[0]):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps_cache.append(inps)
        inps = outs
    #del inps, outs  # Free memory
    #torch.cuda.empty_cache()  # Clear cache to free memory
    return inps_cache, attention_mask, position_ids


def run_calibration_and_save_indices(model, calibration_data, device, save_path="indices.json"):
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
    
    with open(save_path, "w") as f:
        json.dump(indices_dict, f)
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
    
    
if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = '/home/tgx/data/models/Llama-3-8B-Instruct'
    model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="eager").to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    data = get_c4_simple(tokenizer, n_samples=16, seq_len=512)
    
    run_calibration_and_save_indices(model, data, device, save_path="indices.json")
    