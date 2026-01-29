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
from typing_extensions import Unpack

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
    def __init__(
        self,
        layers: Optional[list[CacheLayerMixin]] = None,
        layer_class_to_replicate: Optional[type[CacheLayerMixin]] = None,
        offloading: bool = False,
        offload_only_non_sliding: bool = True,
    ):
        super().__init__(layers, layer_class_to_replicate, offloading, offload_only_non_sliding)
        
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

class CustomDynamicCache(CustomCache):
    def __init__(
        self,
        ddp_cache_data = None,
        config = None,
        offloading: bool = False,
        offload_only_non_sliding: bool = False,
    ):
        layers = []
        # If a config is passed, use it to infer the layer types and initialize accordingly
        if config is not None:
            decoder_config = config.get_text_config(decoder=True)
            sliding_window = getattr(decoder_config, "sliding_window", None) or getattr(
                decoder_config, "attention_chunk_size", None
            )
            layer_types = getattr(decoder_config, "layer_types", None)
            if layer_types is None:
                layer_types = [
                    "sliding_attention" if sliding_window is not None else "full_attention"
                    for _ in range(decoder_config.num_hidden_layers)
                ]
            # Some models have shared layers thus no cache is needed for them (e.g. Gemma3n)
            if hasattr(decoder_config, "num_kv_shared_layers"):
                layer_types = layer_types[: -decoder_config.num_kv_shared_layers]

            for layer_type in layer_types:
                # From a cache point of view, both sliding and chunked are the same in how they should behave and how many
                # states they should return - only the mask changes to make them different at the end!
                if layer_type in ("sliding_attention", "chunked_attention"):
                    layers.append(DynamicSlidingWindowLayer(sliding_window=sliding_window))
                else:
                    layers.append(DynamicLayer())
            super().__init__(layers=layers, offloading=offloading, offload_only_non_sliding=offload_only_non_sliding)

class KVWithSmallKCache(CustomDynamicCache):
    def __init__(
        self,
        ddp_cache_data = None,
        config = None,
        offloading: bool = False,
        offload_only_non_sliding: bool = False,
    ):
        super().__init__(ddp_cache_data, config, offloading, offload_only_non_sliding)
        for i in range(len(self.layers)):
            self.layers[i] = KVWithSmallKLayer()


class KVWithSmallKCache(CustomDynamicCache):
    def __init__(
        self,
        ddp_cache_data = None,
        config = None,
        offloading: bool = False,
        offload_only_non_sliding: bool = False,
    ):
        super().__init__(ddp_cache_data, config, offloading, offload_only_non_sliding)
        for i in range(len(self.layers)):
            self.layers[i] = KVWithSmallKLayer()

    
class CompressedLlamaAttention(nn.Module):
    def __init__(self, config, original_layer, group_keep_indices, mode=None):
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
        #self.topk_ratio = 0.05
        self.topk_ratio = config.topk_ratio
        self.layer_idx = original_layer.layer_idx
        self.escaped_layer = [0, 1, 30, 31]
        self.attention_dropout = original_layer.attention_dropout
        self.sink_size = config.sink_size
        self.local_window = config.local_window
        #self.sink_size = 4
        #self.local_window = 64
        
        if mode is not None:
            self.quant = PseudoQuantizer(bits=4, mode=mode)
        else:
            self.quant = None
        

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

        
        q_full = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k_full = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        q_full, k_full = apply_rotary_pos_emb(q_full, k_full, cos, sin)
        
        
        if self.layer_idx in self.escaped_layer:
            
            if past_key_values is not None:
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                k_full, value_states, _ = past_key_values.update(k_full, value_states, self.layer_idx, cache_kwargs)
                
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
            
            q_small = self.q_proj_small(hidden_states).view(bsz, q_len, self.num_q_heads, self.compressed_dim).transpose(1, 2)
            k_small = self.k_proj_small(hidden_states).view(bsz, q_len, self.num_kv_heads, self.compressed_dim).transpose(1, 2)
            
            # Mixed RoPE
            q_small, k_small = self.apply_mixed_index_rope(q_small, k_small, position_embeddings)
            
            #quantization
            if self.quant is not None:
                q_small = self.quant(q_small)
                k_small = self.quant(k_small)
            
            #k_small = self.k_cache.update(k_small) 
            
            if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
                #cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                cache_kwargs = {"small_key_states": k_small, "sin": sin, "cos": cos, "cache_position": cache_position}
                k_full, value_states, k_small = past_key_values.update(k_full, value_states, self.layer_idx, cache_kwargs)
            
            k_full = repeat_kv(k_full, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
            
            k_small = repeat_kv(k_small, self.num_key_value_groups)
            proxy_scores = torch.matmul(q_small, k_small.transpose(2, 3)) * (self.compressed_dim**-0.5)
            
            if attention_mask is not None:
                proxy_scores = proxy_scores + attention_mask
            kv_seq_len = k_full.size(2)
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
                
                if self.sink_size > 0:
                    mask[..., :self.sink_size] = True
                
                # --- Local Window Logic ---
                #if self.local_window > 0:
                #    ones = torch.ones_like(mask, dtype=torch.bool)
                #    loc_start = torch.triu(ones, diagonal=-self.local_window)
                #    loc_end = torch.tril(ones, diagonal=0)
                #    
                #    local_mask = loc_start & loc_end
                #    mask = mask | local_mask
                if self.local_window > 0:
                    kv_seq_len = real_scores.size(-1)
                    q_len = real_scores.size(-2)

                    # For typical decoding/training with cache: KV = past + Q
                    past_len = kv_seq_len - q_len
                    if past_len < 0:
                        past_len = 0  # safety

                    # q_pos: [Q], absolute positions of the queries in the KV timeline
                    q_pos = torch.arange(q_len, device=real_scores.device) + past_len  # [Q]
                    # k_pos: [KV], absolute positions of keys
                    k_pos = torch.arange(kv_seq_len, device=real_scores.device)  # [KV]

                    # Broadcast to [Q, KV]
                    # causal: k_pos <= q_pos
                    # local:  k_pos >= q_pos - (local_window - 1)
                    q_pos_ = q_pos.view(q_len, 1)
                    k_pos_ = k_pos.view(1, kv_seq_len)

                    local_mask_2d = (k_pos_ <= q_pos_) & (k_pos_ >= (q_pos_ - (self.local_window - 1)))

                    # Expand to [B, H, Q, KV] and OR into existing mask
                    mask = mask | local_mask_2d.view(1, 1, q_len, kv_seq_len)
                
                min_dtype = torch.finfo(real_scores.dtype).min
                real_scores = torch.where(mask, real_scores, torch.tensor(min_dtype, dtype=real_scores.dtype))
                
                attn_weights = torch.softmax(real_scores, dim=-1)
                attn_output = torch.matmul(attn_weights, value_states)
                attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
                attn_output = self.o_proj(attn_output)
            
            else:
                sink_indices = torch.arange(self.sink_size, device=proxy_scores.device) if self.sink_size > 0 else torch.tensor([], device=proxy_scores.device, dtype=torch.long)
                
                local_start = max(0, kv_seq_len - self.local_window)
                local_indices = torch.arange(local_start, kv_seq_len, device=proxy_scores.device) if self.local_window > 0 else torch.tensor([], device=proxy_scores.device, dtype=torch.long)

                # 2. 为了避免 TopK 选出重复的 Sink/Local token，先将这些位置的分数设为负无穷
                # clone 一份 proxy_scores 以免影响后续（虽然这里是局部变量）
                temp_scores = proxy_scores.clone()
                if self.sink_size > 0:
                    temp_scores[..., :self.sink_size] = -float('inf')
                if self.local_window > 0:
                    temp_scores[..., local_start:] = -float('inf')

                # 3. 从剩余部分选出 TopK
                _, topk_indices = torch.topk(temp_scores, k_val, dim=-1) # [B, H, 1, k_val]
                """
                # =============== 新增 Block-Sparse 逻辑 ===============
                BLOCK_SIZE = 4  # 你可以定义成类的属性 self.block_size
                
                # 1. Padding: 如果序列长度不能被 BLOCK_SIZE 整除，需要补齐
                # temp_scores shape: [B, H, 1, Seq_Len]
                seq_len = temp_scores.shape[-1]
                pad_len = (BLOCK_SIZE - (seq_len % BLOCK_SIZE)) % BLOCK_SIZE
                
                if pad_len > 0:
                    # 补 -inf，保证补的这些不会被选中
                    padding = torch.full((*temp_scores.shape[:-1], pad_len), float('-inf'), device=temp_scores.device, dtype=temp_scores.dtype)
                    padded_scores = torch.cat([temp_scores, padding], dim=-1)
                else:
                    padded_scores = temp_scores

                # 2. Reshape & Max Pooling
                # [B, H, 1, Num_Blocks, Block_Size]
                reshaped_scores = padded_scores.view(*padded_scores.shape[:-1], -1, BLOCK_SIZE)
                
                # 取每个块的最大值作为该块的分数
                # block_scores: [B, H, 1, Num_Blocks]
                block_scores, _ = reshaped_scores.max(dim=-1)

                # 3. Block-level TopK
                # k_val 是你要选的 Token 总数，所以块的数量 = k_val // BLOCK_SIZE
                k_blocks = max(1, k_val // BLOCK_SIZE) 
                _, topk_block_indices = torch.topk(block_scores, k_blocks, dim=-1) # [B, H, 1, k_blocks]

                # 4. Expand: 将块索引还原回 Token 索引
                # 例如 block_idx=2, BLOCK_SIZE=4 -> indices 8, 9, 10, 11
                # [B, H, 1, k_blocks, 1] * BLOCK_SIZE -> [..., base_index]
                base_indices = topk_block_indices.unsqueeze(-1) * BLOCK_SIZE 
                
                # [1, 1, 1, 1, BLOCK_SIZE] -> [0, 1, 2, 3]
                offsets = torch.arange(BLOCK_SIZE, device=base_indices.device).view(1, 1, 1, 1, -1)
                
                # Broadcasting add: [B, H, 1, k_blocks, BLOCK_SIZE]
                full_indices = base_indices + offsets
                
                # Flatten back to [B, H, 1, Total_Tokens]
                topk_indices = full_indices.view(*topk_block_indices.shape[:-1], -1)

                # 5. 边界检查 (非常重要!)
                # 因为我们可能有 Padding，还原出来的某些索引可能会越界 (>= seq_len)
                # clamp 一下或者 mask 掉，但在 gather 之前最好确保索引有效。
                # 由于我们 padding 填的是 -inf，正常情况下选不到越界的块，除非 k_val 很大。
                # 为了安全，clamp 到最大有效索引
                topk_indices = topk_indices.clamp(max=seq_len - 1)
                """
                # 4. 拼接索引: Sink + Local + TopK
                # 需要将 sink/local 索引扩展维度以匹配 batch 和 head
                # topk_indices shape: [B, H, 1, k_val]
                bsz_rt, num_heads_rt, _, _ = topk_indices.shape
                
                indices_list = []
                if self.sink_size > 0:
                    indices_list.append(sink_indices.view(1, 1, 1, -1).expand(bsz_rt, num_heads_rt, 1, -1))
                if self.local_window > 0:
                    indices_list.append(local_indices.view(1, 1, 1, -1).expand(bsz_rt, num_heads_rt, 1, -1))
                indices_list.append(topk_indices)
                
                # [B, H, 1, Total_Selected]
                all_indices = torch.cat(indices_list, dim=-1)
                
                # 排序索引，有助于内存访问连续性 (RoPE 依赖位置)
                all_indices, _ = torch.sort(all_indices, dim=-1)

                # Gather Logic: [B, Num_Q, Seq, Full_Dim]
                # Expand indices: [B, Num_Q, 1, Total_Selected] -> [B, Num_Q, Total_Selected, Full_Dim]
                idx_gather = all_indices.view(bsz_rt, num_heads_rt, -1, 1).expand(-1, -1, -1, self.head_dim)
                
                k_selected = torch.gather(k_full, 2, idx_gather)
                v_selected = torch.gather(value_states, 2, idx_gather)
                
                refined_scores = torch.matmul(q_full, k_selected.transpose(2, 3)) * self.scaling
                
                attn_weights = F.softmax(refined_scores, dim=-1)
                
                attn_output = torch.matmul(attn_weights, v_selected)
                attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
                attn_output = self.o_proj(attn_output)
                
            return attn_output, attn_weights
