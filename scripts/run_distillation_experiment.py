import torch
import argparse
import os
import sys
import json
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, LlamaForCausalLM

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pruned_attention.calibration import get_c4_simple, apply_compression_to_model
from src.pruned_attention.attention import CompressedLlamaAttention
from src.pruned_attention.distillation import (
    _get_layer_inputs,
    build_distill_wrapper,
    compute_topk_recall,
    slice_attention_mask,
    slice_position_embeddings,
)

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

@torch.no_grad()
def evaluate_layer_recall(distill_wrapper, hidden_states, position_embeddings, attention_mask, batch_size, topk_ratio, device):
    layer_dataset = TensorDataset(hidden_states)
    recall_loader = DataLoader(layer_dataset, batch_size=batch_size, shuffle=False)

    was_training = distill_wrapper.training
    distill_wrapper.eval()

    matched_sum = 0.0
    total_sum = 0.0
    for (batch_hidden,) in recall_loader:
        batch_hidden = batch_hidden.to(device)
        batch_pos_emb = slice_position_embeddings(position_embeddings, batch_hidden.shape[1])
        batch_attn_mask = slice_attention_mask(attention_mask, batch_hidden.shape[1])

        _, student_scores, teacher_scores = distill_wrapper(batch_hidden, batch_pos_emb, batch_attn_mask)
        matched, total = compute_topk_recall(
            student_scores,
            teacher_scores,
            topk_ratio=topk_ratio,
            attention_mask=batch_attn_mask,
            return_counts=True,
        )
        matched_sum += matched
        total_sum += total

    if was_training:
        distill_wrapper.train()

    return matched_sum / max(total_sum, 1.0)


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
    topk_ratio=0.2,
    ranking_margin=1.0,
    max_pairs_per_query=8,
    ranking_temperature=0.5,
    teacher_boundary_ratio=1.0,
    student_hard_negative_ratio=0.0,
    hybrid_alpha=0.25,
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
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite distillation loss at layer {i}, step {step}: {loss.item()}")
                loss.backward()
                loss_history[i].append(loss.item())

                torch.nn.utils.clip_grad_norm_(distill_wrapper.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

                step += 1
                pbar.set_description(f"Loss: {loss.item():.4f}")
                pbar.update(1)

                if step >= train_steps_per_layer:
                    break
        pbar.close()

        recall_history[i] = evaluate_layer_recall(
            distill_wrapper,
            hidden_states,
            position_embeddings,
            attention_mask,
            batch_size=batch_size,
            topk_ratio=model.config.topk_ratio,
            device=device,
        )

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
    parser.add_argument("--distill_batch_size", type=int, default=4)
    parser.add_argument("--distill_lr", type=float, default=1e-3)
    parser.add_argument("--distill_topk_ratio", type=float, default=0.2)
    parser.add_argument("--ranking_margin", type=float, default=1.0)
    parser.add_argument("--max_pairs_per_query", type=int, default=8)
    parser.add_argument("--ranking_temperature", type=float, default=0.5)
    parser.add_argument("--teacher_boundary_ratio", type=float, default=1.0)
    parser.add_argument("--student_hard_negative_ratio", type=float, default=0.0)

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
        baseline_model,
        model_name,
        quant_mode=None,
        topk_ratio=args.distill_topk_ratio,
        sink_size=0,
        local_window=0,
        escaped_layers=[],
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

    if args.compare:
        methods = ["mse", "topk"]
    else:
        methods = [args.method]
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
            batch_size=args.distill_batch_size,
            lr=args.distill_lr,
            num_calibration_samples=args.num_samples,
            seq_len=args.seq_len,
            topk_ratio=args.distill_topk_ratio,
            ranking_margin=args.ranking_margin,
            max_pairs_per_query=args.max_pairs_per_query,
            ranking_temperature=args.ranking_temperature,
            teacher_boundary_ratio=args.teacher_boundary_ratio,
            student_hard_negative_ratio=args.student_hard_negative_ratio,
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
