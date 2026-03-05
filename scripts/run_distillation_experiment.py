import torch
import torch.nn as nn
import argparse
import os
import sys
import json
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pruned_attention.calibration import get_c4_simple, apply_compression_to_model
from src.pruned_attention.attention import CompressedLlamaAttention, repeat_kv
from src.pruned_attention.distillation import CompressedLlamaAttentionTopKDistillWrapper, _get_layer_inputs

# 优先使用并行PPL评估
try:
    from run_pg16_ppl_test import ppl_eval_parallel
except ImportError:
    try:
        from scripts.run_pg16_ppl_test import ppl_eval_parallel
    except ImportError:
        print("Warning: Could not import ppl_eval_parallel. PPL testing will be skipped.")
        def ppl_eval_parallel(model, tokenizer, args=None):
            return None


class CompressedLlamaAttentionMSEDistillWrapper(nn.Module):
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

        self.loss_fct = nn.MSELoss()

    def forward(self, hidden_states, position_embeddings, attention_mask=None):
        bsz, q_len, _ = hidden_states.shape

        with torch.no_grad():
            q_full = self.layer.q_proj(hidden_states).view(bsz, q_len, self.layer.num_q_heads, self.head_dim).transpose(1, 2)
            k_full = self.layer.k_proj(hidden_states).view(bsz, q_len, self.layer.num_kv_heads, self.head_dim).transpose(1, 2)
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

        if attention_mask is not None:
            valid_mask = (attention_mask > -1000).expand_as(student_scores)
            return self.loss_fct(student_scores[valid_mask], teacher_scores[valid_mask]), student_scores, teacher_scores
        return self.loss_fct(student_scores, teacher_scores), student_scores, teacher_scores


def cal_recall(student_scores, teacher_scores, topk_ratio=0.1):
    with torch.no_grad():
        _, _, _, kv_len = teacher_scores.shape
        k_val = max(1, int(kv_len * topk_ratio))
        _, indices_true = torch.topk(teacher_scores, k=k_val, dim=-1)
        _, indices_pred = torch.topk(student_scores, k=k_val, dim=-1)
        matches = (indices_true.unsqueeze(-1) == indices_pred.unsqueeze(-2)).any(dim=-1).sum()
        total_k = indices_true.numel()
        return (matches.float() / total_k).item()


def collect_layer_wise_distillation(
    model,
    model_name,
    loss_func: str,
    tokenizer,
    train_steps_per_layer=100,
    batch_size=4,
    lr=1e-3,
    seq_len=128,
    num_calibration_samples=128,
):
    device = model.device
    loss_history = {i: [] for i in range(model.config.num_hidden_layers)}
    recall_history = {}

    print("Preparing data...")
    c4_data = get_c4_simple(tokenizer, num_calibration_samples, seq_len)
    dataset = TensorDataset(c4_data)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    hidden_states, attention_mask, position_embeddings = _get_layer_inputs(
        model, dataloader, device, num_samples=num_calibration_samples
    )

    indices_path = os.path.join("data", model_name, "indices.json")
    print(f"Loading indices from {indices_path}...")
    with open(indices_path, "r") as f:
        indices_data = json.load(f)

    model.config.topk_ratio = 0.2
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

        if loss_func == "mse":
            distill_wrapper = CompressedLlamaAttentionMSEDistillWrapper(compressed_attn, model.config).to(device)
        else:
            distill_wrapper = CompressedLlamaAttentionTopKDistillWrapper(compressed_attn, model.config).to(device)

        optimizer = torch.optim.AdamW(distill_wrapper.parameters(), lr=lr)
        layer_dataset = TensorDataset(hidden_states)
        layer_loader = DataLoader(layer_dataset, batch_size=batch_size, shuffle=True)

        pbar = tqdm(total=train_steps_per_layer, desc=f"Layer {i} Distill")
        step = 0
        while step < train_steps_per_layer:
            for (batch_hidden,) in layer_loader:
                batch_hidden = batch_hidden.to(device)
                cos, sin = position_embeddings
                batch_pos_emb = (cos[:, :batch_hidden.shape[1]], sin[:, :batch_hidden.shape[1]])
                batch_attn_mask = attention_mask[:, :, :batch_hidden.shape[1], :batch_hidden.shape[1]]

                loss, student_scores, teacher_scores = distill_wrapper(batch_hidden, batch_pos_emb, batch_attn_mask)
                loss.backward()
                loss_history[i].append(loss.item())

                torch.nn.utils.clip_grad_norm_(distill_wrapper.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

                step += 1
                pbar.set_description(f"Loss: {loss.item():.4f}")
                pbar.update(1)

                if step == train_steps_per_layer - 1:
                    recall_history[i] = cal_recall(
                        student_scores, teacher_scores, topk_ratio=model.config.topk_ratio
                    )

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
                cos, sin = position_embeddings
                batch_pos_emb = (cos[:, :batch_hidden.shape[1]], sin[:, :batch_hidden.shape[1]])
                batch_attn_mask = attention_mask[:, :, :batch_hidden.shape[1], :batch_hidden.shape[1]]
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

    return model, loss_history, recall_history


def _safe_eval_name(name: str):
    return name.replace("/", "_").replace(" ", "_")


def _pick_main_ppl_from_multi(multi_ret, prefer_order=("wikitext", "pg19", "c4")):
    if not multi_ret:
        return None
    d = multi_ret.get("ppl_by_dataset", {})
    for k in prefer_order:
        if k in d and d[k] is not None:
            return d[k]
    for _, v in d.items():
        if v is not None:
            return v
    return None


def run_ppl_multidataset(
    model,
    tokenizer,
    device,
    output_dir,
    method_name,
    eval_specs,
    num_eval_tokens=1024,
    log_interval=64,
    seq_len=2048,
    batch_size=2,
):
    """
    使用并行PPL评估（ppl_eval_parallel），并支持 num_eval_tokens 截断。
    注意：要求 run_pg16_ppl_test.py 的 ppl_eval_parallel 支持 args.dataset / args.num_eval_tokens / args.output_file。
    """
    curves = {}
    scalars = {}

    os.makedirs(output_dir, exist_ok=True)

    for spec in eval_specs:
        name = spec["name"]
        dataset = spec.get("dataset", name)

        curve_path = os.path.join(
            output_dir, f"ppl_curve_{_safe_eval_name(method_name)}_{_safe_eval_name(name)}.json"
        )

        class EvalArgs:
            pass

        ea = EvalArgs()
        ea.device = str(device)
        ea.dataset = dataset
        ea.num_eval_tokens = int(num_eval_tokens) if num_eval_tokens is not None else None
        ea.output_file = curve_path
        ea.log_interval = int(log_interval)
        ea.model_seq_len = int(seq_len)
        ea.batch_size = int(batch_size)

        try:
            ppl = ppl_eval_parallel(model, tokenizer, args=ea)
            scalars[name] = ppl
            curves[name] = curve_path if os.path.exists(curve_path) else None
        except Exception as e:
            print(f"[WARN] PPL eval failed on {name}: {e}")
            scalars[name] = None
            curves[name] = None
    print(f"PPL Results for {method_name}: {scalars}")
    return {"ppl_by_dataset": scalars, "ppl_curve_files": curves}


def main():
    parser = argparse.ArgumentParser(description="Experiment 5: Distillation Effect (MSE vs TopK)")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="experiment_5_results")
    parser.add_argument("--num_samples", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--steps_per_layer", type=int, default=50)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--method", type=str, default="topk", choices=["mse", "topk"])

    parser.add_argument("--num_eval_tokens", type=int, default=1024)
    parser.add_argument("--log_interval", type=int, default=64)

    parser.add_argument("--eval_pg19", action="store_true")
    parser.add_argument("--eval_wikitext", action="store_true")
    parser.add_argument("--eval_c4", action="store_true")
    parser.add_argument("--eval_ptb", action="store_true")
    parser.add_argument("--eval_lambada", action="store_true")
    parser.add_argument("--eval_all", action="store_true")

    parser.add_argument("--pg19_path", type=str, default=None)
    parser.add_argument("--wikitext_path", type=str, default=None)
    parser.add_argument("--c4_path", type=str, default=None)
    parser.add_argument("--ptb_path", type=str, default=None)
    parser.add_argument("--lambada_path", type=str, default=None)

    parser.add_argument("--ppl_seq_len", type=int, default=2048)
    parser.add_argument("--ppl_batch_size", type=int, default=2)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {}
    model_name = os.path.basename(args.model_path)

    if args.eval_all:
        args.eval_pg19 = True
        args.eval_wikitext = True
        args.eval_c4 = True
        args.eval_ptb = True
        args.eval_lambada = True

    eval_specs = []
    if args.eval_pg19:
        eval_specs.append({"name": "pg19", "dataset": "pg19-test", "data_path": args.pg19_path})
    if args.eval_wikitext:
        eval_specs.append({"name": "wikitext", "dataset": "wikitext2", "data_path": args.wikitext_path})
    if args.eval_c4:
        eval_specs.append({"name": "c4", "dataset": "c4", "data_path": args.c4_path})
    if args.eval_ptb:
        eval_specs.append({"name": "ptb", "dataset": "ptb", "data_path": args.ptb_path})
    if args.eval_lambada:
        eval_specs.append({"name": "lambada", "dataset": "lambada", "data_path": args.lambada_path})

    if len(eval_specs) == 0:
        eval_specs.append({"name": "wikitext", "dataset": "wikitext2", "data_path": None})

    print("Phase 1: Baseline")
    baseline_model = LlamaForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map="auto", attn_implementation="eager"
    )
    baseline_model = apply_compression_to_model(
        baseline_model, model_name, quant_mode="int", topk_ratio=0.2, sink_size=0, local_window=0, escaped_layers=[]
    )

    before_multi = run_ppl_multidataset(
        baseline_model,
        tokenizer,
        baseline_model.device,
        output_dir=args.output_dir,
        method_name="before",
        eval_specs=eval_specs,
        num_eval_tokens=args.num_eval_tokens,
        log_interval=args.log_interval,
        seq_len=args.ppl_seq_len,
        batch_size=args.ppl_batch_size,
    )
    results["before"] = {
        "multi_dataset": before_multi,
        "ppl": _pick_main_ppl_from_multi(before_multi),
    }

    del baseline_model
    torch.cuda.empty_cache()

    methods = ["mse", "topk"] if args.compare else [args.method]
    for method in methods:
        print(f"Running {method.upper()} distillation...")
        model = LlamaForCausalLM.from_pretrained(
            args.model_path, dtype=torch.bfloat16, device_map="auto", attn_implementation="eager"
        )
        distilled_model, loss_hist, recall_hist = collect_layer_wise_distillation(
            model=model,
            model_name=model_name,
            loss_func=method,
            tokenizer=tokenizer,
            train_steps_per_layer=args.steps_per_layer,
            num_calibration_samples=args.num_samples,
            seq_len=args.seq_len,
        )

        method_multi = run_ppl_multidataset(
            distilled_model,
            tokenizer,
            distilled_model.device,
            output_dir=args.output_dir,
            method_name=method,
            eval_specs=eval_specs,
            num_eval_tokens=args.num_eval_tokens,
            log_interval=args.log_interval,
            seq_len=args.ppl_seq_len,
            batch_size=args.ppl_batch_size,
        )

        results[method] = {
            "loss": loss_hist,
            "recall": recall_hist,
            "multi_dataset": method_multi,
            "ppl": _pick_main_ppl_from_multi(method_multi),
        }

        del distilled_model
        torch.cuda.empty_cache()

    output_file = os.path.join(args.output_dir, "distillation_comparison.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()