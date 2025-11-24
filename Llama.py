from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaConfig
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Any
from typing_extensions import Unpack
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils import logging, TransformersKwargs
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, LlamaRotaryEmbedding
from transformers.modeling_rope_utils import dynamic_rope_update
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from transformers.utils.deprecation import deprecate_kwarg
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP, LlamaRMSNorm
from transformers.models.llama.modeling_llama import create_causal_mask
from transformers.utils.generic import check_model_inputs
from transformers.models.llama.modeling_llama import eager_attention_forward


logger = logging.get_logger(__name__)


import torch
import torch.nn as nn
from transformers.models.llama.configuration_llama import LlamaConfig


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_partial_rotary_pos_emb(q, k, cos, sin, position_ids, keep_indices):
    """
    对裁剪后的 Q/K 应用 RoPE，严格保持原始频率语义。
    keep_indices 必须成对出现 (e.g., [0, 1, 10, 11])。
    """
    # 1. 选取当前位置的 cos/sin: [batch, seq_len, full_dim]
    cos = cos[position_ids].squeeze(1)
    sin = sin[position_ids].squeeze(1)
    
    # 2. 根据裁剪索引提取对应的频率: [batch, seq_len, compressed_dim]
    cos = cos[..., keep_indices]
    sin = sin[..., keep_indices]
    
    # 3. 扩展维度以匹配 head 维度: [batch, 1, seq_len, compressed_dim]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    
    # 4. 应用旋转
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed



def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    
    
class CompressedLlamaAttention(nn.Module):
    def __init__(self, config, original_layer, group_keep_indices, pretrained_small=None):
        super().__init__()
        self.config = config
        self.head_dim = original_layer.head_dim
        self.num_q_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_q_heads // self.num_kv_heads
        # group_keep_indices: [Num_KV_Heads, Compressed_Dim]
        self.register_buffer("keep_indices", group_keep_indices)
        self.compressed_dim = group_keep_indices.shape[1]
        self.scaling = self.head_dim**-0.5
        self.topk_ratio = 0.2
        self.layer_idx = original_layer.layer_idx
        self.escaped_layer = []
        self.attention_dropout = original_layer.attention_dropout
        
        if pretrained_small is not None:
            in_dim_q, out_dim_q = pretrained_small["q_proj_small.weight"].shape[1], pretrained_small["q_proj_small.weight"].shape[0]
            in_dim_k, out_dim_k = pretrained_small["k_proj_small.weight"].shape[1], pretrained_small["k_proj_small.weight"].shape[0]

            self.q_proj_small = nn.Linear(in_dim_q, out_dim_q, bias=False)
            self.k_proj_small = nn.Linear(in_dim_k, out_dim_k, bias=False)

            with torch.no_grad():
                self.q_proj_small.weight.copy_(pretrained_small["q_proj_small.weight"])
                self.k_proj_small.weight.copy_(pretrained_small["k_proj_small.weight"])

            q_keep_indices = self.keep_indices.unsqueeze(1).expand(-1, self.num_key_value_groups, -1).reshape(self.num_q_heads, -1)
            self.register_buffer("q_keep_indices", q_keep_indices)
        
        else:

            with torch.no_grad():
                
                full_wk = original_layer.k_proj.weight.T.view(config.hidden_size, self.num_kv_heads, self.head_dim)
                
                # indices [1, Num_KV_Heads, Compressed_Dim] 
                gather_idx_k = self.keep_indices.unsqueeze(0).expand(config.hidden_size, -1, -1)
                
                # [Hidden, Num_KV_Heads, Compressed_Dim]
                small_wk_view = torch.gather(full_wk, 2, gather_idx_k)
                
                # [Hidden, Num_KV_Heads * Compressed_Dim]
                small_wk_flat = small_wk_view.reshape(config.hidden_size, -1)
                self.k_proj_small = nn.Linear(config.hidden_size, small_wk_flat.shape[1], bias=False)
                self.k_proj_small.weight.data = small_wk_flat.T

                # keep_indices: [Num_KV_Heads, D] -> [Num_KV_Heads, Group_Size, D] -> [Num_Q_Heads, D]
                q_keep_indices = self.keep_indices.unsqueeze(1).expand(-1, self.num_key_value_groups, -1).reshape(self.num_q_heads, -1)
                
                #  [Hidden, Num_Q_Heads, Head_Dim]
                full_wq = original_layer.q_proj.weight.T.view(config.hidden_size, self.num_q_heads, self.head_dim)
                
                gather_idx_q = q_keep_indices.unsqueeze(0).expand(config.hidden_size, -1, -1)
                small_wq_view = torch.gather(full_wq, 2, gather_idx_q)
                
                # Flatten
                small_wq_flat = small_wq_view.reshape(config.hidden_size, -1)
                self.q_proj_small = nn.Linear(config.hidden_size, small_wq_flat.shape[1], bias=False)
                self.q_proj_small.weight.data = small_wq_flat.T
                
                self.register_buffer("q_keep_indices", q_keep_indices)
        
        self.q_proj = original_layer.q_proj
        self.k_proj = original_layer.k_proj
        self.v_proj = original_layer.v_proj
        self.o_proj = original_layer.o_proj
        
    def apply_mixed_index_rope(
        self, 
        q, 
        k, 
        position_embeddings, 
    ):
        """
        q: [Batch, Num_Q_Heads, Seq, Compressed_Dim]
        k: [Batch, Num_KV_Heads, Seq, Compressed_Dim]
        position_embeddings: (cos, sin), 形状均为 [Batch, Seq, Full_Head_Dim]
        """
        cos, sin = position_embeddings
        
        bsz, seq_len, _ = cos.shape
    
        #  [Batch, Seq, Full_Dim] -> [Batch, 1, Seq, Full_Dim] 
        cos_exp = cos.unsqueeze(1)
        sin_exp = sin.unsqueeze(1)

        # Q Indices: [Num_Q, Comp_D] 
        
        # expand: [1, Num_Q, 1, Comp_D] -> [B, Num_Q, S, Comp_D]
        idx_q = self.q_keep_indices.unsqueeze(0).unsqueeze(2).expand(bsz, -1, seq_len, -1)
        
        # [Batch, 1, Seq, Full_Dim] -> [Batch, Num_Q, Seq, Full_Dim]
        cos_source_q = cos_exp.expand(-1, self.num_q_heads, -1, -1)
        sin_source_q = sin_exp.expand(-1, self.num_q_heads, -1, -1)
        
        cos_q = torch.gather(cos_source_q, 3, idx_q)
        sin_q = torch.gather(sin_source_q, 3, idx_q)

        # K Indices: [Num_KV, Comp_D] -> [B, Num_KV, S, Comp_D]
        idx_k = self.keep_indices.unsqueeze(0).unsqueeze(2).expand(bsz, -1, seq_len, -1)
        
        # [Batch, 1, Seq, Full_Dim] -> [Batch, Num_KV, Seq, Full_Dim]
        cos_source_k = cos_exp.expand(-1, self.num_kv_heads, -1, -1)
        sin_source_k = sin_exp.expand(-1, self.num_kv_heads, -1, -1)
        
        cos_k = torch.gather(cos_source_k, 3, idx_k)
        sin_k = torch.gather(sin_source_k, 3, idx_k)

        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        q_embed = (q * cos_q) + (rotate_half(q) * sin_q)
        k_embed = (k * cos_k) + (rotate_half(k) * sin_k)
        
        return q_embed, k_embed
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        bsz, q_len = input_shape[0], input_shape[1]

        kv_seq_len = hidden_states.shape[1]
        
        q_full = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k_full = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        q_full, k_full = apply_rotary_pos_emb(q_full, k_full, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k_full, value_states = past_key_values.update(k_full, value_states, self.layer_idx, cache_kwargs)
        
        
        if self.layer_idx in self.escaped_layer:
            attn_output, attn_weights = eager_attention_forward(
                self,
                q_full,
                k_full,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs,
            )

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output, attn_weights
        
        else:
            k_full = repeat_kv(k_full, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
        
            q_small = self.q_proj_small(hidden_states).view(bsz, q_len, self.num_q_heads, self.compressed_dim).transpose(1, 2)
            k_small = self.k_proj_small(hidden_states).view(bsz, q_len, self.num_kv_heads, self.compressed_dim).transpose(1, 2)
            
            # Mixed RoPE
            q_small, k_small = self.apply_mixed_index_rope(q_small, k_small, position_embeddings)
            
            k_small = repeat_kv(k_small, self.num_key_value_groups)
            proxy_scores = torch.matmul(q_small, k_small.transpose(2, 3)) * (self.compressed_dim**-0.5)
            
            if attention_mask is not None:
                proxy_scores = proxy_scores + attention_mask
            
            min_k = min(kv_seq_len, 32) 
            k_val = max(min_k, int(kv_seq_len * self.topk_ratio))

            #sink_size = 4
            
            if q_len > 1:
                # [B, H, Q, KV]
                real_scores = torch.matmul(q_full, k_full.transpose(2, 3)) * self.scaling
                if attention_mask is not None:
                    real_scores = real_scores + attention_mask

                # topk_indices: [B, H, Q, k_val]
                _, topk_indices = torch.topk(proxy_scores, k_val, dim=-1)
                
                # mask shape: [B, H, Q, KV]
                mask = torch.zeros_like(real_scores, dtype=torch.bool)
                mask.scatter_(3, topk_indices, True)
                
                #if sink_size > 0:
                #    mask[..., :sink_size] = True
                
                min_dtype = torch.finfo(real_scores.dtype).min
                real_scores = torch.where(mask, real_scores, torch.tensor(min_dtype, dtype=real_scores.dtype))
                
                attn_weights = torch.softmax(real_scores, dim=-1)
                attn_output = torch.matmul(attn_weights, value_states)
                attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
                attn_output = self.o_proj(attn_output)
            
            else:
                _, topk_indices = torch.topk(proxy_scores, k_val, dim=-1)

                # Gather Logic: [B, Num_Q, Seq, Full_Dim]
                # Expand indices: [B, Num_Q, 1, TopK] -> [B, Num_Q, TopK, Full_Dim]
                idx_gather = topk_indices.view(bsz, self.num_q_heads, k_val, 1).expand(-1, -1, -1, self.head_dim)
                
                k_selected = torch.gather(k_full, 2, idx_gather)
                v_selected = torch.gather(value_states, 2, idx_gather)
                
                refined_scores = torch.matmul(q_full, k_selected.transpose(2, 3)) * self.scaling
                
                attn_weights = F.softmax(refined_scores, dim=-1)
                
                attn_output = torch.matmul(attn_weights, v_selected)
                attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
                attn_output = self.o_proj(attn_output)
                
            return attn_output, attn_weights