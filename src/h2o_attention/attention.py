import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F
import math
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from typing import Optional, Tuple
from tqdm import tqdm

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaAttention, apply_rotary_pos_emb, repeat_kv


__all__ = ['convert_kvcache_llama_heavy_recent', 'LlamaAttention_heavy_hitter']


class LlamaAttention_heavy_hitter(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, original_layer, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_attention_heads = config.num_attention_heads  # Ensure this attribute exists
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = original_layer.q_proj
        self.k_proj = original_layer.k_proj
        self.v_proj = original_layer.v_proj
        self.o_proj = original_layer.o_proj

        # --- New Configurations ---
        self.escaped_layers = getattr(config, "escaped_layers", [])

        # heavy_ratio is still a ratio for Heavy Hitters
        self.heavy_budget_ratio = getattr(config, "heavy_ratio", 0.1)
        self.sink_size = getattr(config, "sink_size", 4)
        self.local_window = getattr(config, "local_window", 64)

        self.attention_masks_next = None
        self.heavy_budget = None
        self.previous_scores = None

    def _reset_masks(self):
        self.attention_masks_next = None
        self.heavy_budget = None
        self.previous_scores = None

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
        # [bsz, nh, t, hd]

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # --- Escape Layer Logic ---
        if self.layer_idx in self.escaped_layers:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, dim)
            attn_output = self.o_proj(attn_output)
            return attn_output, attn_weights

        # --- H2O Logic ---
        if self.attention_masks_next is not None:
            attn_weights = attn_weights * self.attention_masks_next + (1 - self.attention_masks_next) * torch.finfo(attn_weights.dtype).min

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # Accumulate attention scores for H2O selection
        # attn_weights shape: (bsz, heads, q-tokens, k-tokens)
        # Sum over q-tokens -> importance per (bsz, head, k-token)
        current_scores_sum = attn_weights.sum(dim=2)  # (bsz, heads, k-tokens)

        if self.previous_scores is not None:
            current_scores_sum[..., :-1] += self.previous_scores

        # Update budget every step based on current sequence length
        attn_tokens_all = current_scores_sum.shape[-1]
        self.heavy_budget = max(1, int(self.heavy_budget_ratio * attn_tokens_all)) if self.heavy_budget_ratio > 0 else 0

        dtype_attn_weights = attn_weights.dtype
        attn_weights_devices = attn_weights.device

        # Update previous scores
        self.previous_scores = current_scores_sum

        # --- Construct Next Step Mask ---
        # Mask shape: [bsz, heads, Total_KV_Len + 1] (Prepare for next token)
        attn_mask = torch.ones(
            current_scores_sum.shape[0],
            current_scores_sum.shape[1],
            current_scores_sum.shape[2] + 1,
            device=attn_weights_devices,
            dtype=dtype_attn_weights,
        )

        # If sequence is long enough to need pruning
        if attn_tokens_all > (self.sink_size + self.local_window + self.heavy_budget):
            # 1. Initialize mask to 0 (prune everything by default)
            attn_mask.zero_()

            # 2. Keep Sink Tokens
            if self.sink_size > 0:
                attn_mask[..., :self.sink_size] = 1

            # 3. Keep Local Window (Recent tokens)
            if self.local_window > 0:
                attn_mask[..., -self.local_window:] = 1
                candidate_start = self.sink_size
                candidate_end = max(candidate_start, attn_tokens_all - self.local_window)
                selected_set = self.previous_scores[..., candidate_start:candidate_end]
            else:
                candidate_start = self.sink_size
                selected_set = self.previous_scores[..., candidate_start:]

            # 4. Keep Heavy Hitters (Top-K from candidates)
            if self.heavy_budget > 0 and selected_set.shape[-1] > 0:
                k_val = min(self.heavy_budget, selected_set.shape[-1])
                _, keep_topk = selected_set.topk(k=k_val, dim=-1, largest=True)
                keep_topk = keep_topk + candidate_start
                attn_mask = attn_mask.scatter(-1, keep_topk, 1)

        # Save mask for next step: [bsz, heads, 1, Seq_Len]
        self.attention_masks_next = attn_mask.unsqueeze(2)

        # Apply mask to previous_scores to "forget" pruned tokens scores
        score_mask = attn_mask[..., :-1]
        if self.local_window > 0:
            score_mask[..., -self.local_window:] = 1
        self.previous_scores = self.previous_scores * score_mask

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


def convert_kvcache_llama_heavy_recent(model, config):
    for i, layer in enumerate(tqdm(model.model.layers, desc="Replacing Attention Layers")):
        heavy_recent_attn = LlamaAttention_heavy_hitter(config, layer.self_attn, layer_idx=i).to(model.device)
        model.model.layers[i].self_attn = heavy_recent_attn

    print("h2o applied successfully!")
    return model