import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Any
from transformers.models.llama.modeling_llama import LlamaConfig
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer, CacheLayerMixin, DynamicSlidingWindowLayer
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils import logging, TransformersKwargs
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import eager_attention_forward

from .utils import PseudoQuantizer

logger = logging.get_logger(__name__)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Expands the key-value states for Grouped-Query Attention.
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class KVWithSmallKLayer(DynamicLayer):  
    def __init__(self):
        super().__init__()
        self.small_keys: Optional[torch.Tensor] = None
    
    def lazy_initialization(self, key_states: torch.Tensor):
        self.dtype, self.device = key_states.dtype, key_states.device
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.small_keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.is_initialized = True
        
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states)

        self.keys = torch.cat([self.keys, key_states], dim=-2)
        self.values = torch.cat([self.values, value_states], dim=-2)

        if cache_kwargs is not None and "small_key_states" in cache_kwargs:
            self.small_keys = torch.cat([self.small_keys, cache_kwargs["small_key_states"]], dim=-2)
        
        return self.keys, self.values, self.small_keys


class CustomCache(Cache):
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.layer_class_to_replicate is not None:
            while len(self.layers) <= layer_idx:
                self.layers.append(self.layer_class_to_replicate())

        if self.offloading:
            torch.cuda.default_stream(key_states.device).wait_stream(self.prefetch_stream)
            self.prefetch(layer_idx + 1, self.only_non_sliding)

        keys, values, small_keys = self.layers[layer_idx].update(key_states, value_states, cache_kwargs)

        if self.offloading:
            self.offload(layer_idx, self.only_non_sliding)

        return keys, values, small_keys


class KVWithSmallKCache(CustomCache):
    def __init__(
        self,
        config: LlamaConfig,
        offloading: bool = False,
    ):
        super().__init__()
        self.layers = [KVWithSmallKLayer() for _ in range(config.num_hidden_layers)]
        self.offloading = offloading
        self.only_non_sliding = False # Not implemented for this custom cache


class CompressedLlamaAttention(nn.Module):
    """
    A compressed attention mechanism that uses a smaller, proxy attention calculation
    to select important key-value pairs for the full attention calculation.
    """
    def __init__(self, config, original_layer, group_keep_indices, mode=None, pretrained_small=None):
        super().__init__()
        self.config = config
        self.head_dim = original_layer.head_dim
        self.num_q_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_q_heads // self.num_kv_heads
        
        self.register_buffer("keep_indices", group_keep_indices)
        self.compressed_dim = group_keep_indices.shape[1]
        self.scaling = self.head_dim**-0.5
        self.topk_ratio = 0.2
        self.layer_idx = original_layer.layer_idx
        self.attention_dropout = original_layer.attention_dropout
        
        self.quant = PseudoQuantizer(bits=4, mode=mode) if mode else None

        # Initialize small projection layers
        self._create_small_projections(config, original_layer, pretrained_small)
        
        # Keep original projection layers
        self.q_proj = original_layer.q_proj
        self.k_proj = original_layer.k_proj
        self.v_proj = original_layer.v_proj
        self.o_proj = original_layer.o_proj
        
    def _create_small_projections(self, config, original_layer, pretrained_small):
        """Initializes the smaller q and k projection layers."""
        if pretrained_small:
            self._load_pretrained_small(pretrained_small)
        else:
            self._slice_original_projections(config, original_layer)

    def _load_pretrained_small(self, pretrained_small):
        """Loads weights from a pretrained small model."""
        in_dim_q, out_dim_q = pretrained_small["q_proj_small.weight"].shape[1], pretrained_small["q_proj_small.weight"].shape[0]
        in_dim_k, out_dim_k = pretrained_small["k_proj_small.weight"].shape[1], pretrained_small["k_proj_small.weight"].shape[0]

        self.q_proj_small = nn.Linear(in_dim_q, out_dim_q, bias=False)
        self.k_proj_small = nn.Linear(in_dim_k, out_dim_k, bias=False)

        with torch.no_grad():
            self.q_proj_small.weight.copy_(pretrained_small["q_proj_small.weight"])
            self.k_proj_small.weight.copy_(pretrained_small["k_proj_small.weight"])

        q_keep_indices = self.keep_indices.unsqueeze(1).expand(-1, self.num_key_value_groups, -1).reshape(self.num_q_heads, -1)
        self.register_buffer("q_keep_indices", q_keep_indices)

    def _slice_original_projections(self, config, original_layer):
        """Creates small projection layers by slicing the original ones."""
        with torch.no_grad():
            # Slice K projection
            full_wk = original_layer.k_proj.weight.T.view(config.hidden_size, self.num_kv_heads, self.head_dim)
            gather_idx_k = self.keep_indices.unsqueeze(0).expand(config.hidden_size, -1, -1)
            small_wk_view = torch.gather(full_wk, 2, gather_idx_k)
            small_wk_flat = small_wk_view.reshape(config.hidden_size, -1)
            
            self.k_proj_small = nn.Linear(config.hidden_size, small_wk_flat.shape[1], bias=False, device=original_layer.k_proj.weight.device, dtype=original_layer.k_proj.weight.dtype)
            self.k_proj_small.weight.data = small_wk_flat.T

            # Slice Q projection
            q_keep_indices = self.keep_indices.unsqueeze(1).expand(-1, self.num_key_value_groups, -1).reshape(self.num_q_heads, -1)
            full_wq = original_layer.q_proj.weight.T.view(config.hidden_size, self.num_q_heads, self.head_dim)
            gather_idx_q = q_keep_indices.unsqueeze(0).expand(config.hidden_size, -1, -1)
            small_wq_view = torch.gather(full_wq, 2, gather_idx_q)
            small_wq_flat = small_wq_view.reshape(config.hidden_size, -1)

            self.q_proj_small = nn.Linear(config.hidden_size, small_wq_flat.shape[1], bias=False, device=original_layer.q_proj.weight.device, dtype=original_layer.q_proj.weight.dtype)
            self.q_proj_small.weight.data = small_wq_flat.T
            
            self.register_buffer("q_keep_indices", q_keep_indices)

    def apply_mixed_index_rope(self, q, k, position_embeddings):
        """
        Applies Rotary Position Embeddings to the compressed q and k tensors
        using the selected indices.
        """
        cos, sin = position_embeddings
        bsz, seq_len, _ = cos.shape
    
        cos_exp = cos.unsqueeze(1)
        sin_exp = sin.unsqueeze(1)

        idx_q = self.q_keep_indices.unsqueeze(0).unsqueeze(2).expand(bsz, -1, seq_len, -1)
        cos_source_q = cos_exp.expand(-1, self.num_q_heads, -1, -1)
        sin_source_q = sin_exp.expand(-1, self.num_q_heads, -1, -1)
        cos_q = torch.gather(cos_source_q, 3, idx_q)
        sin_q = torch.gather(sin_source_q, 3, idx_q)

        idx_k = self.keep_indices.unsqueeze(0).unsqueeze(2).expand(bsz, -1, seq_len, -1)
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
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, q_len, _ = hidden_states.shape

        # Full projections
        q_full = self.q_proj(hidden_states).view(bsz, q_len, self.num_q_heads, self.head_dim).transpose(1, 2)
        k_full = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_q_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        q_full, k_full = apply_rotary_pos_emb(q_full, k_full, cos, sin)
        
        # Small projections
        q_small = self.q_proj_small(hidden_states).view(bsz, q_len, self.num_q_heads, self.compressed_dim).transpose(1, 2)
        k_small = self.k_proj_small(hidden_states).view(bsz, q_len, self.num_kv_heads, self.compressed_dim).transpose(1, 2)
        
        q_small, k_small = self.apply_mixed_index_rope(q_small, k_small, position_embeddings)
        
        if self.quant:
            q_small, k_small = self.quant(q_small), self.quant(k_small)
        
        if past_key_values is not None:
            cache_kwargs = {"small_key_states": k_small, "sin": sin, "cos": cos, "cache_position": cache_position}
            k_full, value_states, k_small = past_key_values.update(k_full, value_states, self.layer_idx, cache_kwargs)
        
        k_full_grouped = repeat_kv(k_full, self.num_key_value_groups)
        value_states_grouped = repeat_kv(value_states, self.num_key_value_groups)
        k_small_grouped = repeat_kv(k_small, self.num_key_value_groups)

        # Use small projections to find top-k indices
        proxy_scores = torch.matmul(q_small, k_small_grouped.transpose(2, 3)) * (self.compressed_dim**-0.5)
        if attention_mask is not None:
            proxy_scores += attention_mask
        
        kv_seq_len = k_full_grouped.size(2)
        min_k = min(kv_seq_len, 32) 
        k_val = max(min_k, int(kv_seq_len * self.topk_ratio))

        if q_len > 1:
            # Use top-k indices to create a mask for full attention
            real_scores = torch.matmul(q_full, k_full_grouped.transpose(2, 3)) * self.scaling
            if attention_mask is not None:
                real_scores += attention_mask

            _, topk_indices = torch.topk(proxy_scores, k_val, dim=-1)
            
            mask = torch.zeros_like(real_scores, dtype=torch.bool)
            mask.scatter_(3, topk_indices, True)
            
            min_dtype = torch.finfo(real_scores.dtype).min
            real_scores.masked_fill_(~mask, min_dtype)
            
            attn_weights = torch.softmax(real_scores, dim=-1)
            attn_output = torch.matmul(attn_weights, value_states_grouped)
        else: # Special path for single-token generation to use gather instead of mask
            _, topk_indices = torch.topk(proxy_scores, k_val, dim=-1)

            idx_gather = topk_indices.view(bsz, self.num_q_heads, k_val, 1).expand(-1, -1, -1, self.head_dim)
            
            k_selected = torch.gather(k_full_grouped, 2, idx_gather)
            v_selected = torch.gather(value_states_grouped, 2, idx_gather)
            
            refined_scores = torch.matmul(q_full, k_selected.transpose(2, 3)) * self.scaling
            
            attn_weights = F.softmax(refined_scores, dim=-1)
            attn_output = torch.matmul(attn_weights, v_selected)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
            
        return attn_output, attn_weights
