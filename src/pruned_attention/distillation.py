import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from .attention import CompressedLlamaAttention, repeat_kv
from .calibration import get_c4_simple


def slice_position_embeddings(position_embeddings, seq_len):
    cos, sin = position_embeddings
    return cos[:, :seq_len], sin[:, :seq_len]


def slice_attention_mask(attention_mask, seq_len):
    if attention_mask is None:
        return None
    return attention_mask[:, :, :seq_len, :seq_len]


def build_valid_mask(attention_mask, scores):
    if attention_mask is None:
        return torch.ones_like(scores, dtype=torch.bool)
    return (attention_mask > -10000).expand_as(scores)


def build_topk_mask(scores, valid_mask, topk_ratio):
    masked_scores = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)

    valid_counts = valid_mask.sum(dim=-1)
    safe_valid_counts = valid_counts.clamp(min=1)
    k_per_query = torch.clamp((safe_valid_counts.float() * topk_ratio).long(), min=1)
    k_per_query = torch.minimum(k_per_query, safe_valid_counts)

    max_k = int(k_per_query.max().item())
    topk_indices = torch.topk(masked_scores, k=max_k, dim=-1).indices
    take_mask = torch.arange(max_k, device=scores.device).view(1, 1, 1, -1) < k_per_query.unsqueeze(-1)

    topk_mask = torch.zeros_like(valid_mask)
    topk_mask.scatter_(-1, topk_indices, take_mask)
    topk_mask &= valid_mask
    return topk_mask, k_per_query, valid_counts


def compute_topk_recall(student_scores, teacher_scores, topk_ratio=0.1, attention_mask=None, return_counts=False):
    with torch.no_grad():
        valid_mask = build_valid_mask(attention_mask, teacher_scores)
        teacher_topk, _, _ = build_topk_mask(teacher_scores.float(), valid_mask, topk_ratio)
        student_topk, _, _ = build_topk_mask(student_scores.float(), valid_mask, topk_ratio)

        matched = (teacher_topk & student_topk).sum(dtype=torch.float32)
        total = teacher_topk.sum(dtype=torch.float32).clamp(min=1.0)
        recall = matched / total

        if return_counts:
            return float(matched.item()), float(total.item())
        return float(recall.item())


def compute_query_mean_std(scores, valid_mask, eps=1e-6):
    valid_mask_f = valid_mask.to(scores.dtype)
    valid_counts = valid_mask_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
    valid_scores = scores.masked_fill(~valid_mask, 0.0)

    mean = valid_scores.sum(dim=-1, keepdim=True) / valid_counts
    centered = (valid_scores - mean) * valid_mask_f
    var = centered.square().sum(dim=-1, keepdim=True) / valid_counts
    std = torch.sqrt(var + eps)
    return mean, std


def select_masked_topk_values(scores, candidate_mask, take_counts, largest):
    max_take = int(take_counts.max().item())
    empty_shape = scores.shape[:-1] + (0,)
    if max_take == 0:
        return scores.new_empty(empty_shape), candidate_mask.new_zeros(empty_shape)

    fill_value = torch.finfo(scores.dtype).min if largest else torch.finfo(scores.dtype).max
    candidates = scores.masked_fill(~candidate_mask, fill_value)
    values = torch.topk(candidates, k=max_take, dim=-1, largest=largest).values
    take_mask = torch.arange(max_take, device=scores.device).view(1, 1, 1, -1) < take_counts.unsqueeze(-1)
    return values, take_mask


def select_masked_topk_indices(scores, candidate_mask, take_counts, largest):
    max_take = int(take_counts.max().item())
    empty_shape = scores.shape[:-1] + (0,)
    if max_take == 0:
        return torch.empty(empty_shape, device=scores.device, dtype=torch.long), candidate_mask.new_zeros(empty_shape)

    fill_value = torch.finfo(scores.dtype).min if largest else torch.finfo(scores.dtype).max
    candidates = scores.masked_fill(~candidate_mask, fill_value)
    indices = torch.topk(candidates, k=max_take, dim=-1, largest=largest).indices
    take_mask = torch.arange(max_take, device=scores.device).view(1, 1, 1, -1) < take_counts.unsqueeze(-1)
    return indices, take_mask


def gather_last_dim(values, indices):
    return torch.gather(values, dim=-1, index=indices)


def build_teacher_boundary_mask(teacher_scores, valid_mask, teacher_topk, k_per_query, boundary_ratio):
    neg_mask = valid_mask & ~teacher_topk
    neg_counts = neg_mask.sum(dim=-1)

    boundary_counts = torch.ceil(k_per_query.float() * boundary_ratio).long()
    boundary_counts = torch.clamp(boundary_counts, min=1)
    boundary_counts = torch.minimum(boundary_counts, neg_counts)

    max_boundary = int(boundary_counts.max().item())
    if max_boundary == 0:
        return torch.zeros_like(neg_mask), neg_mask, neg_counts, boundary_counts

    neg_candidates = teacher_scores.masked_fill(~neg_mask, torch.finfo(teacher_scores.dtype).min)
    boundary_indices = torch.topk(neg_candidates, k=max_boundary, dim=-1, largest=True).indices
    take_mask = torch.arange(max_boundary, device=teacher_scores.device).view(1, 1, 1, -1) < boundary_counts.unsqueeze(-1)

    boundary_mask = torch.zeros_like(neg_mask)
    boundary_mask.scatter_(-1, boundary_indices, take_mask)
    boundary_mask &= neg_mask
    return boundary_mask, neg_mask, neg_counts, boundary_counts


def build_teacher_boundary_positive_mask(teacher_scores, teacher_topk, k_per_query, boundary_ratio):
    pos_counts = teacher_topk.sum(dim=-1)

    boundary_counts = torch.ceil(k_per_query.float() * boundary_ratio).long()
    boundary_counts = torch.clamp(boundary_counts, min=1)
    boundary_counts = torch.minimum(boundary_counts, pos_counts)

    max_boundary = int(boundary_counts.max().item())
    if max_boundary == 0:
        return torch.zeros_like(teacher_topk), pos_counts, boundary_counts

    pos_candidates = teacher_scores.masked_fill(~teacher_topk, torch.finfo(teacher_scores.dtype).max)
    boundary_indices = torch.topk(pos_candidates, k=max_boundary, dim=-1, largest=False).indices
    take_mask = torch.arange(max_boundary, device=teacher_scores.device).view(1, 1, 1, -1) < boundary_counts.unsqueeze(-1)

    boundary_mask = torch.zeros_like(teacher_topk)
    boundary_mask.scatter_(-1, boundary_indices, take_mask)
    boundary_mask &= teacher_topk
    return boundary_mask, pos_counts, boundary_counts


class _CompressedLlamaAttentionDistillWrapper(nn.Module):
    def __init__(self, compressed_layer, original_layer_config):
        super().__init__()
        self.layer = compressed_layer
        self.config = original_layer_config
        self.head_dim = self.layer.head_dim
        self.scaling = self.head_dim ** -0.5
        self.num_key_value_groups = self.layer.num_key_value_groups

        for p in self.layer.parameters():
            p.requires_grad = False

        self.layer.q_proj_small.weight = nn.Parameter(self.layer.q_proj_small.weight)
        self.layer.k_proj_small.weight = nn.Parameter(self.layer.k_proj_small.weight)
        self.layer.q_proj_small.weight.requires_grad = True
        self.layer.k_proj_small.weight.requires_grad = True

    def _compute_teacher_student_scores(self, hidden_states, position_embeddings, attention_mask=None):
        bsz, q_len, _ = hidden_states.shape

        with torch.no_grad():
            q_full = self.layer.q_proj(hidden_states).view(
                bsz, q_len, self.layer.num_q_heads, self.head_dim
            ).transpose(1, 2)
            k_full = self.layer.k_proj(hidden_states).view(
                bsz, q_len, self.layer.num_kv_heads, self.head_dim
            ).transpose(1, 2)

            cos, sin = position_embeddings
            q_full, k_full = apply_rotary_pos_emb(q_full, k_full, cos, sin)
            k_full = repeat_kv(k_full, self.num_key_value_groups)

            teacher_scores = torch.matmul(q_full, k_full.transpose(2, 3)) * self.scaling
            if attention_mask is not None:
                teacher_scores = teacher_scores + attention_mask

        q_small = self.layer.q_proj_small(hidden_states).view(
            bsz, q_len, self.layer.num_q_heads, self.layer.compressed_dim
        ).transpose(1, 2)
        k_small = self.layer.k_proj_small(hidden_states).view(
            bsz, q_len, self.layer.num_kv_heads, self.layer.compressed_dim
        ).transpose(1, 2)

        q_small, k_small = self.layer.apply_mixed_index_rope(q_small, k_small, position_embeddings)
        k_small = repeat_kv(k_small, self.num_key_value_groups)

        student_scores = torch.matmul(q_small, k_small.transpose(2, 3)) * (self.layer.compressed_dim ** -0.5)
        if attention_mask is not None:
            student_scores = student_scores + attention_mask

        return student_scores.float(), teacher_scores.float()

    def _valid_mask(self, attention_mask, scores):
        return build_valid_mask(attention_mask, scores)


class CompressedLlamaAttentionMSEDistillWrapper(_CompressedLlamaAttentionDistillWrapper):
    def __init__(self, compressed_layer, original_layer_config):
        super().__init__(compressed_layer, original_layer_config)
        self.loss_fct = nn.MSELoss()

    def forward(self, hidden_states, position_embeddings, attention_mask=None):
        student_scores, teacher_scores = self._compute_teacher_student_scores(
            hidden_states, position_embeddings, attention_mask
        )
        valid_mask = self._valid_mask(attention_mask, teacher_scores)

        if valid_mask.any():
            loss = self.loss_fct(student_scores[valid_mask], teacher_scores[valid_mask])
        else:
            loss = student_scores.new_zeros(())
        return loss, student_scores, teacher_scores


class CompressedLlamaAttentionTopKDistillWrapper(_CompressedLlamaAttentionDistillWrapper):
    """
    Ranking distillation on a per-head, per-query basis.

    Student scores are scaled by the teacher query std to avoid layer/head scale
    mismatch without destroying the raw geometry used at inference time. Pairs
    are built around the teacher top-k boundary on both sides, with optional
    student-hard negatives as fallback.
    """

    def __init__(
        self,
        compressed_layer,
        original_layer_config,
        margin=1.0,
        max_pairs_per_query=8,
        ranking_temperature=0.5,
        teacher_boundary_ratio=1.0,
        student_hard_negative_ratio=0.0,
    ):
        super().__init__(compressed_layer, original_layer_config)
        self.margin = float(margin)
        self.max_pairs_per_query = int(max_pairs_per_query)
        self.ranking_temperature = max(float(ranking_temperature), 1e-6)
        self.teacher_boundary_ratio = max(float(teacher_boundary_ratio), 0.0)
        self.student_hard_negative_ratio = min(max(float(student_hard_negative_ratio), 0.0), 1.0)

    def _ranking_loss(self, student_scores, teacher_scores, valid_mask):
        _, teacher_std = compute_query_mean_std(teacher_scores, valid_mask)
        teacher_scale = teacher_std.clamp(min=1e-6)

        student_scores = (student_scores / teacher_scale).masked_fill(~valid_mask, 0.0)
        teacher_scores = (teacher_scores / teacher_scale).masked_fill(~valid_mask, 0.0)

        teacher_topk, k_per_query, _ = build_topk_mask(teacher_scores, valid_mask, self.layer.topk_ratio)
        positive_boundary_mask, _, pos_boundary_counts = build_teacher_boundary_positive_mask(
            teacher_scores,
            teacher_topk,
            k_per_query,
            boundary_ratio=self.teacher_boundary_ratio,
        )
        boundary_mask, _, _, neg_boundary_counts = build_teacher_boundary_mask(
            teacher_scores,
            valid_mask,
            teacher_topk,
            k_per_query,
            boundary_ratio=self.teacher_boundary_ratio,
        )

        pos_take_counts = torch.clamp(pos_boundary_counts, max=self.max_pairs_per_query)
        neg_take_counts = torch.clamp(neg_boundary_counts, max=self.max_pairs_per_query)
        max_pos = int(pos_take_counts.max().item())
        max_neg = int(neg_take_counts.max().item())
        if max_pos == 0 or max_neg == 0:
            return student_scores.new_zeros(())

        pos_indices, pos_take_mask = select_masked_topk_indices(
            teacher_scores,
            positive_boundary_mask,
            pos_take_counts,
            largest=False,
        )
        neg_indices, neg_take_mask = select_masked_topk_indices(
            teacher_scores,
            boundary_mask,
            neg_take_counts,
            largest=True,
        )

        student_pos = gather_last_dim(student_scores, pos_indices).masked_fill(~pos_take_mask, 0.0)
        teacher_pos = gather_last_dim(teacher_scores, pos_indices).masked_fill(~pos_take_mask, 0.0)
        student_neg = gather_last_dim(student_scores, neg_indices).masked_fill(~neg_take_mask, 0.0)
        teacher_neg = gather_last_dim(teacher_scores, neg_indices).masked_fill(~neg_take_mask, 0.0)

        pair_mask = pos_take_mask.unsqueeze(-1) & neg_take_mask.unsqueeze(-2)
        teacher_gap = (teacher_pos.unsqueeze(-1) - teacher_neg.unsqueeze(-2)).clamp(min=0.0)
        student_gap = student_pos.unsqueeze(-1) - student_neg.unsqueeze(-2)

        pair_mask_f = pair_mask.to(teacher_gap.dtype)
        mean_gap = (teacher_gap * pair_mask_f).sum(dim=(-1, -2), keepdim=True) / pair_mask_f.sum(dim=(-1, -2), keepdim=True).clamp(min=1.0)
        gap_weight = (teacher_gap / mean_gap.clamp(min=1e-6)).clamp(min=0.5, max=2.0)
        losses = F.softplus(-student_gap / self.ranking_temperature) * gap_weight
        if pair_mask.any():
            return losses[pair_mask].mean()
        return student_scores.new_zeros(())

    def forward(self, hidden_states, position_embeddings, attention_mask=None):
        student_scores, teacher_scores = self._compute_teacher_student_scores(
            hidden_states, position_embeddings, attention_mask
        )
        valid_mask = self._valid_mask(attention_mask, teacher_scores)
        loss = self._ranking_loss(student_scores, teacher_scores, valid_mask)
        return loss, student_scores, teacher_scores


def build_distill_wrapper(
    loss_func,
    compressed_attn,
    config,
    ranking_margin=1.0,
    max_pairs_per_query=8,
    ranking_temperature=0.5,
    teacher_boundary_ratio=1.0,
    student_hard_negative_ratio=0.0,
):
    if loss_func == "mse":
        return CompressedLlamaAttentionMSEDistillWrapper(compressed_attn, config)
    if loss_func == "topk":
        return CompressedLlamaAttentionTopKDistillWrapper(
            compressed_attn,
            config,
            margin=ranking_margin,
            max_pairs_per_query=max_pairs_per_query,
            ranking_temperature=ranking_temperature,
            teacher_boundary_ratio=teacher_boundary_ratio,
            student_hard_negative_ratio=student_hard_negative_ratio,
        )
    raise ValueError(f"Unsupported distillation loss: {loss_func}")


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
            masks_list.append(kwargs.get("attention_mask"))
            cache["position_embeddings"] = kwargs.get("position_embeddings")
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
    num_calibration_samples=128,
    loss_func="topk",
    topk_ratio=0.2,
    ranking_margin=1.0,
    max_pairs_per_query=8,
    ranking_temperature=0.5,
    teacher_boundary_ratio=1.0,
    student_hard_negative_ratio=0.0,
):
    """
    Performs layer-wise distillation to fine-tune the compressed attention projections.
    """
    device = model.device

    print("Preparing data...")
    c4_data = get_c4_simple(tokenizer, num_calibration_samples, seq_len)
    dataset = TensorDataset(c4_data)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    hidden_states, attention_mask, position_embeddings = _get_layer_inputs(
        model, dataloader, device, num_samples=num_calibration_samples
    )

    with open(indices_path, "r") as f:
        indices_data = json.load(f)

    model.config.topk_ratio = topk_ratio
    model.config.sink_size = 0
    model.config.local_window = 0
    model.config.escaped_layers = []

    for i in range(model.config.num_hidden_layers):
        print(f"\n=== Processing Layer {i}/{model.config.num_hidden_layers} ===")
        layer = model.model.layers[i]

        keep_indices = torch.tensor(indices_data[str(i)], dtype=torch.long).to(device)

        compressed_attn = CompressedLlamaAttention(
            model.config,
            layer.self_attn,
            group_keep_indices=keep_indices,
        ).to(device)

        distill_wrapper = build_distill_wrapper(
            loss_func,
            compressed_attn,
            model.config,
            ranking_margin=ranking_margin,
            max_pairs_per_query=max_pairs_per_query,
            ranking_temperature=ranking_temperature,
            teacher_boundary_ratio=teacher_boundary_ratio,
            student_hard_negative_ratio=student_hard_negative_ratio,
        ).to(device)

        print("  Training small projections...")
        optimizer = torch.optim.AdamW(distill_wrapper.parameters(), lr=lr)

        layer_dataset = TensorDataset(hidden_states)
        layer_loader = DataLoader(layer_dataset, batch_size=batch_size, shuffle=True)

        pbar = tqdm(total=train_steps_per_layer, desc=f"Layer {i} Distill")
        step = 0
        while step < train_steps_per_layer:
            for (batch_hidden,) in layer_loader:
                batch_hidden = batch_hidden.to(device)
                batch_pos_emb = slice_position_embeddings(position_embeddings, batch_hidden.shape[1])
                batch_attn_mask = slice_attention_mask(attention_mask, batch_hidden.shape[1])

                loss, _, _ = distill_wrapper(batch_hidden, batch_pos_emb, batch_attn_mask)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(distill_wrapper.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

                step += 1
                pbar.set_description(f"Loss: {loss.item():.4f}")
                pbar.update(1)
                if step >= train_steps_per_layer:
                    break
        pbar.close()

        model.model.layers[i].self_attn = compressed_attn

        print(f"  Generating inputs for Layer {i+1}...")
        new_hidden_states_list = []
        gen_loader = DataLoader(layer_dataset, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for (batch_hidden,) in tqdm(gen_loader, desc="Forwarding"):
                batch_hidden = batch_hidden.to(device)
                batch_pos_emb = slice_position_embeddings(position_embeddings, batch_hidden.shape[1])
                batch_attn_mask = slice_attention_mask(attention_mask, batch_hidden.shape[1])

                layer_output = model.model.layers[i](
                    batch_hidden,
                    attention_mask=batch_attn_mask,
                    position_embeddings=batch_pos_emb,
                )
                if isinstance(layer_output, tuple):
                    layer_output = layer_output[0]
                new_hidden_states_list.append(layer_output.detach().cpu())

        hidden_states = torch.cat(new_hidden_states_list, dim=0)

        del distill_wrapper
        torch.cuda.empty_cache()

    print("\nDistillation Complete!")
    return model
