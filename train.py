import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import copy
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from Llama import repeat_kv, CompressedLlamaAttention
from prune import get_c4_simple
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import random

class CompressedLlamaAttentionTopKDistillWrapper(nn.Module):
 
    def __init__(self, compressed_layer, original_layer_config):
        super().__init__()
        self.layer = compressed_layer 
        self.config = original_layer_config
        self.head_dim = self.layer.head_dim
        self.scaling = self.head_dim**-0.5
        self.num_key_value_groups = self.layer.num_key_value_groups
        
    
        for p in self.layer.parameters():
            p.requires_grad = False
        
        self.layer.q_proj_small.weight = nn.Parameter(self.layer.q_proj_small.weight.float())
        self.layer.k_proj_small.weight = nn.Parameter(self.layer.k_proj_small.weight.float())
        
        self.layer.q_proj_small.weight.requires_grad = True
        self.layer.k_proj_small.weight.requires_grad = True
        
    def forward(self, hidden_states, position_embeddings, attention_mask=None, temperature: float = 2.0):
      
        bsz, q_len, _ = hidden_states.shape
        
        # Teacher
        with torch.no_grad():
            q_full = self.layer.q_proj(hidden_states).view(bsz, q_len, self.layer.num_q_heads, self.head_dim).transpose(1, 2)
            k_full = self.layer.k_proj(hidden_states).view(bsz, q_len, self.layer.num_kv_heads, self.head_dim).transpose(1, 2)
            
            cos, sin = position_embeddings
            q_full, k_full = apply_rotary_pos_emb(q_full, k_full, cos, sin, position_ids=None)
            
            # GQA Expand
            k_full = repeat_kv(k_full, self.num_key_value_groups)
            
            # Score
            teacher_scores = torch.matmul(q_full, k_full.transpose(2, 3)) * self.scaling
            if attention_mask is not None:
                teacher_scores = teacher_scores + attention_mask
            
            k_val = max(1, int(q_len * self.layer.topk_ratio))
            threshold = torch.topk(teacher_scores, k_val, dim=-1).values[..., -1, None]
            
            is_important_mask = (teacher_scores >= threshold)

       
        # Student
        q_small = self.layer.q_proj_small(hidden_states).view(bsz, q_len, self.layer.num_q_heads, self.layer.compressed_dim).transpose(1, 2)
        k_small = self.layer.k_proj_small(hidden_states).view(bsz, q_len, self.layer.num_kv_heads, self.layer.compressed_dim).transpose(1, 2)
        
        q_small, k_small = self.layer.apply_mixed_index_rope(q_small, k_small, position_embeddings)
   
        k_small = repeat_kv(k_small, self.layer.num_key_value_groups)
        
        student_scores = torch.matmul(q_small, k_small.transpose(2, 3)) * (self.layer.compressed_dim**-0.5)
        if attention_mask is not None:
            student_scores = student_scores + attention_mask
            
        flat_student_scores = student_scores.reshape(-1)
        flat_is_topk = is_important_mask.reshape(-1)
        
        if attention_mask is not None:
            flat_attn_mask = attention_mask.expand_as(student_scores).reshape(-1)
            
            is_valid_token = (flat_attn_mask > -10000)
        else:
            is_valid_token = torch.ones_like(flat_student_scores, dtype=torch.bool)

        pos_mask = flat_is_topk & is_valid_token
        
        neg_mask = (~flat_is_topk) & is_valid_token
        
        pos_scores = flat_student_scores[pos_mask]
        neg_scores = flat_student_scores[neg_mask]
        
        if len(neg_scores) > len(pos_scores):
            perm = torch.randperm(len(neg_scores), device=hidden_states.device)[:len(pos_scores)]
            neg_scores_sampled = neg_scores[perm]
        else:
            neg_scores_sampled = neg_scores
            pos_scores = pos_scores[:len(neg_scores)]
        
        loss_fct = nn.MarginRankingLoss(margin=1.0)
        
        target = torch.ones(len(pos_scores), device=hidden_states.device)
        
        loss = loss_fct(pos_scores, neg_scores_sampled, target)
        
        return loss

class CompressedLlamaAttentionKLDistillWrapper(nn.Module):

    def __init__(self, compressed_layer, original_layer_config):
        super().__init__()
        self.layer = compressed_layer 
        self.config = original_layer_config
        self.head_dim = self.layer.head_dim
        self.scaling = self.head_dim**-0.5
        
        for p in self.layer.parameters():
            p.requires_grad = False
        
        self.layer.q_proj_small.weight = nn.Parameter(self.layer.q_proj_small.weight.float())
        self.layer.k_proj_small.weight = nn.Parameter(self.layer.k_proj_small.weight.float())
        
        self.layer.q_proj_small.weight.requires_grad = True
        self.layer.k_proj_small.weight.requires_grad = True
        

    def forward(self, hidden_states, position_embeddings, attention_mask=None, temperature: float = 2.0):
      
        bsz, q_len, _ = hidden_states.shape
        
        with torch.no_grad():
            q_full = self.layer.q_proj(hidden_states).view(bsz, q_len, self.layer.num_q_heads, self.head_dim).transpose(1, 2)
            k_full = self.layer.k_proj(hidden_states).view(bsz, q_len, self.layer.num_kv_heads, self.head_dim).transpose(1, 2)
            
            cos, sin = position_embeddings
            q_full, k_full = apply_rotary_pos_emb(q_full, k_full, cos, sin, position_ids=None)
            
            k_full = repeat_kv(k_full, self.layer.num_key_value_groups)
            
            teacher_scores = torch.matmul(q_full, k_full.transpose(2, 3)) * self.scaling
            if attention_mask is not None:
                teacher_scores = teacher_scores + attention_mask
            
            teacher_probs = F.softmax(teacher_scores / temperature, dim=-1) 

        q_small = self.layer.q_proj_small(hidden_states).view(bsz, q_len, self.layer.num_q_heads, self.layer.compressed_dim).transpose(1, 2)
        k_small = self.layer.k_proj_small(hidden_states).view(bsz, q_len, self.layer.num_kv_heads, self.layer.compressed_dim).transpose(1, 2)
        
        q_small, k_small = self.layer.apply_mixed_index_rope(q_small, k_small, position_embeddings)
        
        k_small = repeat_kv(k_small, self.layer.num_key_value_groups)
        
        student_scores = torch.matmul(q_small, k_small.transpose(2, 3)) * (self.layer.compressed_dim**-0.5)
        if attention_mask is not None:
            student_scores = student_scores + attention_mask
            
        student_log_probs = F.log_softmax(student_scores / temperature, dim=-1)
        loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
        
        return loss
    
    
def get_layer_inputs(model, dataloader, device, num_samples=128):
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
            cache['postion_embeddings'] = kwargs.get('position_embeddings')
            raise ValueError 
            
    original_layer0 = layers[0]
    layers[0] = Catcher(original_layer0)

    for i, batch in enumerate(dataloader):
        (input_ids,) = batch                     
        if i * input_ids.shape[0] >= num_samples:
            break
        try:
            model(input_ids.to(device))
        except ValueError:
            pass
            
    layers[0] = original_layer0
    all_inputs = torch.cat(inputs_list, dim=0) # [N, Seq, Hidden]
    return all_inputs, masks_list[0], cache["postion_embeddings"]


def train_layer_wise_distillation(
    model, 
    tokenizer, 
    indices_path="indices.json", 
    train_steps_per_layer=100, 
    batch_size=4,
    lr=1e-3,
    seq_len=128, #
    num_calibration_samples=128 
):
    device = model.device
    

    print("Preparing data...")
    c4_data = get_c4_simple(tokenizer, num_calibration_samples, seq_len)
    dataset = TensorDataset(c4_data)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    hidden_states, attention_mask, position_embeddings = get_layer_inputs(model, dataloader, device)

    import json
    with open(indices_path, 'r') as f:
        indices_data = json.load(f)

    num_layers = len(model.model.layers)
    
    for i in range(num_layers):
        print(f"\n=== Processing Layer {i}/{num_layers} ===")
        layer = model.model.layers[i]
        
        keep_indices = torch.tensor(indices_data[str(i)], dtype=torch.long).to(device)
        compressed_attn = CompressedLlamaAttention(
            model.config, 
            layer.self_attn, 
            group_keep_indices=keep_indices
        ).to(device)
        
        distill_wrapper = CompressedLlamaAttentionTopKDistillWrapper(compressed_attn, model.config).to(device)

        print(f"  Training Router (Small Projections)...")
        optimizer = torch.optim.AdamW(distill_wrapper.parameters(), lr=lr)
        
        layer_dataset = TensorDataset(hidden_states)
        layer_loader = DataLoader(layer_dataset, batch_size=batch_size, shuffle=True)
        
        pbar = tqdm(total=train_steps_per_layer)
        step = 0
        while step < train_steps_per_layer:
            for (batch_hidden,) in layer_loader:
                batch_hidden = batch_hidden.to(device)
                loss = distill_wrapper(batch_hidden, position_embeddings, attention_mask)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(distill_wrapper.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                step += 1
                pbar.set_description(f"Loss: {loss.item():.4f}")
                pbar.update(1)
                if step >= train_steps_per_layer: break
        pbar.close()
        
        model.model.layers[i].self_attn = compressed_attn
        
        print(f"  Forwarding to generate inputs for Layer {i+1}...")
        new_hidden_states_list = []
        gen_loader = DataLoader(layer_dataset, batch_size=batch_size, shuffle=False)
        
        for (batch_hidden,) in gen_loader:
            batch_hidden = batch_hidden.to(device)
            with torch.no_grad():
                position_ids = torch.arange(batch_hidden.shape[1], device=device).unsqueeze(0).expand(batch_hidden.shape[0], -1)
                
                layer_output = model.model.layers[i](
                    batch_hidden, 
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings
                ) 
                
                new_hidden_states_list.append(layer_output.cpu())

        hidden_states = torch.cat(new_hidden_states_list, dim=0)
        
        
        # 清理显存
        del distill_wrapper
        torch.cuda.empty_cache()

    print("\nDistillation Complete! Saving model...")
    return model


def main():
    # 配置参数
    MODEL_ID = "/home/tgx/data/models/Llama-3-8B-Instruct" 
    INDICES_PATH = "indices.json"
    SAVE_PATH = "./compressed_llama_distilled_topk_wo_sink"
    SEQ_LEN = 512
    NUM_SAMPLES = 512  
    TRAIN_STEPS = 300   
    BATCH_SIZE = 4
    
    print(f"Loading model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float32, 
        device_map="auto",
        attn_implementation="eager"
    )
    
    if not os.path.exists(INDICES_PATH):
        print(f"Error: {INDICES_PATH} not found. Please run calibration first.")
        return

    print("Starting Layer-wise Distillation...")
    distilled_model = train_layer_wise_distillation(
        model=model,
        tokenizer=tokenizer,
        indices_path=INDICES_PATH,
        train_steps_per_layer=TRAIN_STEPS,
        batch_size=BATCH_SIZE,
        lr=1e-5,
        seq_len=SEQ_LEN,
        num_calibration_samples=NUM_SAMPLES
    )
    
    print(f"Saving distilled model to {SAVE_PATH}...")
    os.makedirs(SAVE_PATH, exist_ok=True)
    small_state = {}
    for i, layer in enumerate(distilled_model.model.layers):
        attn = layer.self_attn
        if hasattr(attn, "q_proj_small") and hasattr(attn, "k_proj_small"):
            small_state[f"layers.{i}.self_attn.q_proj_small.weight"] = attn.q_proj_small.weight.detach().cpu()
            small_state[f"layers.{i}.self_attn.k_proj_small.weight"] = attn.k_proj_small.weight.detach().cpu()

    small_path = os.path.join(SAVE_PATH, "small_attn_weights.pt")
    torch.save(small_state, small_path)
    print(f"Saved small attention weights to {small_path}")
    print("Done!")

if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    main()