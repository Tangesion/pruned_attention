from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaConfig
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer, CacheLayerMixin, DynamicSlidingWindowLayer
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
from utils import PseudoQuantizer


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class KCache:
    def __init__(self):
        self.k_cache = None

    def update(self, k):
        if self.k_cache is None:
            self.k_cache = k
        else:
            self.k_cache = torch.cat([self.k_cache, k], dim=2)
        return self.k_cache
    
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
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`dict[str, Any]`, *optional*):
                Additional arguments for the cache subclass. These are specific to each subclass and allow new types of
                cache to be created.

        Return:
            A tuple containing the updated key and value states.
        """
        # In this case, the `layers` were not provided, and we must append as much as `layer_idx`
        if self.layer_class_to_replicate is not None:
            while len(self.layers) <= layer_idx:
                self.layers.append(self.layer_class_to_replicate())

        if self.offloading:
            # Wait for the stream to finish if needed, and start prefetching the next layer
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

    
class CompressedLlamaAttention(nn.Module):
    def __init__(self, config, original_layer, group_keep_indices, mode=None, pretrained_small=None):
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
        
        if mode is not None:
            self.quant = PseudoQuantizer(bits=4, mode=mode)
        else:
            self.quant = None
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
        
        
class LlamaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
    

@auto_docstring
class LlamaPreTrainedModel(PreTrainedModel):
    config: LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": LlamaDecoderLayer,
        "attentions": CompressedLlamaAttention,
    }
    


@auto_docstring
class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )




@auto_docstring
class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
