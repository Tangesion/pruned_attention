
import torch
from torch import nn
from typing import Optional
from tqdm import tqdm

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
__all__ = ['convert_kvcache_llama_sliding_window', 'LlamaAttention_sliding_window']

class LlamaAttention_sliding_window(nn.Module):
    """Sliding-window attention: attend only to the most recent local_window tokens."""

    def __init__(self, config: LlamaConfig, original_layer, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = original_layer.q_proj
        self.k_proj = original_layer.k_proj
        self.v_proj = original_layer.v_proj
        self.o_proj = original_layer.o_proj

        self.sink_size = getattr(config, "sink_size", 4)
        self.local_window = getattr(config, "local_window", 64)
        self.local_window_ratio = getattr(config, "local_window_ratio", 0.0)
        self.escaped_layers = getattr(config, "escaped_layers", [])

    def _get_local_window(self, k_len: int) -> int:
        if self.local_window_ratio and self.local_window_ratio > 0:
            return max(1, int(k_len * self.local_window_ratio))
        return self.local_window

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values=None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, q_len, dim = hidden_states.size()
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # Escape layers: use full attention
        if self.layer_idx in self.escaped_layers:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, dim)
            attn_output = self.o_proj(attn_output)
            return attn_output, attn_weights

        # Sliding-window + sink tokens
        k_len = key_states.shape[-2]
        local_window = self._get_local_window(k_len)
        if (self.sink_size or local_window) and k_len > (self.sink_size + local_window):
            mask = torch.full((1, 1, 1, k_len), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)

            # keep sink tokens
            if self.sink_size > 0:
                mask[..., :self.sink_size] = 0

            # keep recent local window
            if local_window > 0:
                mask[..., -local_window:] = 0

            attn_weights = attn_weights + mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


def convert_kvcache_llama_sliding_window(model, config):
    for i, layer in enumerate(tqdm(model.model.layers, desc="Replacing Attention Layers (Sliding Window)")):
        sw_attn = LlamaAttention_sliding_window(config, layer.self_attn, layer_idx=i).to(model.device)
        model.model.layers[i].self_attn = sw_attn

    print("sliding-window attention applied successfully!")
    return model
