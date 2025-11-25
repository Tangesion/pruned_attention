import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from .attention import CompressedLlamaAttention, repeat_kv
from .calibration import get_c4_simple


class CompressedLlamaAttentionTopKDistillWrapper(nn.Module):
    """
    A wrapper for distillation that trains the small projection layers of 
    CompressedLlamaAttention using a margin ranking loss.
    """
    def __init__(self, compressed_layer, original_layer_config):
        super().__init__()
        self.layer = compressed_layer 
        self.config = original_layer_config
        self.head_dim = self.layer.head_dim
        self.scaling = self.head_dim**-0.5
        self.num_key_value_groups = self.layer.num_key_value_groups
        
        # Freeze all parameters except the small projections
        for p in self.layer.parameters():
            p.requires_grad = False
        
        self.layer.q_proj_small.weight = nn.Parameter(self.layer.q_proj_small.weight.float())
        self.layer.k_proj_small.weight = nn.Parameter(self.layer.k_proj_small.weight.float())
        
        self.layer.q_proj_small.weight.requires_grad = True
        self.layer.k_proj_small.weight.requires_grad = True
        
    def forward(self, hidden_states, position_embeddings, attention_mask=None, temperature: float = 2.0):
        bsz, q_len, _ = hidden_states.shape
        
        # Teacher model (frozen) provides the ground truth top-k mask
        with torch.no_grad():
            q_full = self.layer.q_proj(hidden_states).view(bsz, q_len, self.layer.num_q_heads, self.head_dim).transpose(1, 2)
            k_full = self.layer.k_proj(hidden_states).view(bsz, q_len, self.layer.num_kv_heads, self.head_dim).transpose(1, 2)
            
            cos, sin = position_embeddings
            q_full, k_full = apply_rotary_pos_emb(q_full, k_full, cos, sin)
            
            k_full = repeat_kv(k_full, self.num_key_value_groups)
            
            teacher_scores = torch.matmul(q_full, k_full.transpose(2, 3)) * self.scaling
            if attention_mask is not None:
                teacher_scores += attention_mask
            
            k_val = max(1, int(q_len * self.layer.topk_ratio))
            threshold = torch.topk(teacher_scores, k_val, dim=-1).values[..., -1, None]
            is_important_mask = (teacher_scores >= threshold)

       
        # Student model (trainable) produces scores to be optimized
        q_small = self.layer.q_proj_small(hidden_states).view(bsz, q_len, self.layer.num_q_heads, self.layer.compressed_dim).transpose(1, 2)
        k_small = self.layer.k_proj_small(hidden_states).view(bsz, q_len, self.layer.num_kv_heads, self.layer.compressed_dim).transpose(1, 2)
        
        q_small, k_small = self.layer.apply_mixed_index_rope(q_small, k_small, position_embeddings)
        k_small = repeat_kv(k_small, self.num_key_value_groups)
        
        student_scores = torch.matmul(q_small, k_small.transpose(2, 3)) * (self.layer.compressed_dim**-0.5)
        if attention_mask is not None:
            student_scores += attention_mask
            
        # Calculate Margin Ranking Loss
        flat_student_scores = student_scores.view(-1)
        flat_is_topk = is_important_mask.view(-1)
        
        is_valid_token = torch.ones_like(flat_student_scores, dtype=torch.bool)
        if attention_mask is not None:
            flat_attn_mask = attention_mask.expand_as(student_scores).reshape(-1)
            is_valid_token = (flat_attn_mask > -10000)

        pos_mask = flat_is_topk & is_valid_token
        neg_mask = (~flat_is_topk) & is_valid_token
        
        pos_scores = flat_student_scores[pos_mask]
        neg_scores = flat_student_scores[neg_mask]
        
        # Balance positive and negative samples
        if len(neg_scores) > len(pos_scores):
            perm = torch.randperm(len(neg_scores), device=hidden_states.device)[:len(pos_scores)]
            neg_scores_sampled = neg_scores[perm]
        else:
            neg_scores_sampled = neg_scores
            pos_scores = pos_scores[:len(neg_scores)]
        
        loss_fct = nn.MarginRankingLoss(margin=1.0)
        target = torch.ones(len(pos_scores), device=hidden_states.device)
        
        return loss_fct(pos_scores, neg_scores_sampled, target)


class CompressedLlamaAttentionKLDistillWrapper(nn.Module):
    """
    A wrapper for distillation that trains the small projection layers of 
    CompressedLlamaAttention using KL-divergence loss.
    """
    def __init__(self, compressed_layer, original_layer_config):
        super().__init__()
        self.layer = compressed_layer 
        self.config = original_layer_config
        # ... (rest of the implementation is similar to TopKDistillWrapper)

    def forward(self, hidden_states, position_embeddings, attention_mask=None, temperature: float = 2.0):
        # ... (Implementation of KL-divergence loss)
        pass # Not used in the main script, so can be left as a stub for now


def _get_layer_inputs(model, dataloader, device, num_samples=128):
    """Helper to capture the input hidden states for the first layer."""
    print("Capturing inputs for Layer 0...")
    model.config.use_cache = False
    layers = model.model.layers
    
    inputs_list = []
    masks_list = []
    cache = {}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inputs_list.append(inp.detach().cpu()) 
            masks_list.append(kwargs.get('attention_mask'))
            cache['position_embeddings'] = kwargs.get('position_embeddings')
            raise StopIteration
            
    original_layer0 = layers[0]
    layers[0] = Catcher(original_layer0)

    for i, batch in enumerate(dataloader):
        if i * batch[0].shape[0] >= num_samples:
            break
        try:
            model(batch[0].to(device))
        except StopIteration:
            pass
            
    layers[0] = original_layer0
    all_inputs = torch.cat(inputs_list, dim=0)
    return all_inputs, masks_list[0], cache["position_embeddings"]


def train_layer_wise_distillation(
    model, 
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
    
    print("Preparing data...")
    c4_data = get_c4_simple(tokenizer, num_calibration_samples, seq_len)
    dataset = TensorDataset(c4_data)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    hidden_states, attention_mask, position_embeddings = _get_layer_inputs(model, dataloader, device, num_samples=num_calibration_samples)

    with open(indices_path, 'r') as f:
        indices_data = json.load(f)

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


                loss = distill_wrapper(batch_hidden, batch_pos_emb, batch_attn_mask)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(distill_wrapper.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                step += 1
                pbar.set_description(f"Loss: {loss.item():.4f}")
                pbar.update(1)
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
    return model
