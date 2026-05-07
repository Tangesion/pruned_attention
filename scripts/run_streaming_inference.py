import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, LlamaForCausalLM, StoppingCriteria, StoppingCriteriaList


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.h2o_attention.attention import convert_kvcache_llama_heavy_recent
from src.pruned_attention.attention import KVWithSmallKCache
from src.pruned_attention.calibration import apply_compression_to_model
from src.slide_attention.attention import convert_kvcache_llama_sliding_window


METHOD_ALIASES = {
    "compress": "ours",
    "streaming": "slide",
    "base": "full",
    "original": "full",
}

MODEL_REGISTRY_ALIAS_MAP = {
    "llama3-8b-compress": "llama3-8b-instruct",
    "llama3-8b-h2o": "llama3-8b-instruct",
    "llama3-8b-slide": "llama3-8b-instruct",
    "llama-3.2-1b-compress": "llama-3.2-1B",
    "llama-3.2-h2o": "llama-3.2-1B",
    "llama-3.2-slide": "llama-3.2-1B",
}

ARTIFACT_NAME_MAP = {
    "llama-3.2-1b": "Llama-3.2-1B",
    "llama-3.2-1b-compress": "Llama-3.2-1B",
    "llama-3.2-h2o": "Llama-3.2-1B",
    "llama-3.2-slide": "Llama-3.2-1B",
    "llama3-8b-instruct": "Llama-3-8B-Instruct",
    "llama3-8b-compress": "Llama-3-8B-Instruct",
    "llama3-8b-h2o": "Llama-3-8B-Instruct",
    "llama3-8b-slide": "Llama-3-8B-Instruct",
}


class StreamTokens(StoppingCriteria):
    def __init__(self, tokenizer, input_len: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.last_printed = input_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        seq = input_ids[0]
        cur_len = seq.shape[0]

        if cur_len > self.last_printed:
            new_tokens = seq[self.last_printed:cur_len]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            print(text, end="", flush=True)
            self.last_printed = cur_len

        return False


def parse_args():
    parser = argparse.ArgumentParser(description="Unified long-context streaming inference demo.")
    parser.add_argument(
        "--method",
        type=str,
        default="ours",
        choices=["full", "ours", "h2o", "slide", "all", "compress", "streaming", "base", "original"],
        help="Method to run. 'all' runs full, ours, h2o, and slide sequentially.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama-3.2-1B-compress",
        help="Logical model name. If present in LongBench/config/model2path.json, its path will be used.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional explicit model path or HF repo id. Overrides --model path lookup.",
    )
    parser.add_argument(
        "--artifact-name",
        type=str,
        default=None,
        help="Optional artifact directory name under ./data used by our method.",
    )
    parser.add_argument("--prompt-file", type=str, required=True, help="Path to the prompt file.")
    parser.add_argument(
        "--min-new-tokens",
        type=int,
        default=0,
        help="Minimum number of new tokens to generate before EOS is allowed.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--topk-ratio", type=float, default=0.1)
    parser.add_argument("--sink-size", type=int, default=4)
    parser.add_argument("--local-window", type=int, default=64)
    parser.add_argument(
        "--slide-window-ratio",
        type=float,
        default=None,
        help="Window ratio for slide/streaming attention. Defaults to --topk-ratio if unset.",
    )
    parser.add_argument("--escaped-layers", type=int, nargs="*", default=[])
    parser.add_argument(
        "--quant-mode",
        type=str,
        default="int",
        help="Quant mode for our compressed attention. Use 'none' to disable quantization.",
    )
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling during generation.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Print the full prompt before generation. Disabled by default for long-context demos.",
    )
    parser.add_argument(
        "--attn-impl",
        type=str,
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="HF attention backend for the base model load.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional directory to save each method's generated text.",
    )
    parser.add_argument(
        "--score-profile",
        type=str,
        default="auto",
        choices=["auto", "none", "long_continuation_demo"],
        help="Optional heuristic scoring profile for generated outputs.",
    )
    parser.add_argument(
        "--score-only-dir",
        type=str,
        default=None,
        help="Skip generation and only score existing outputs in this directory.",
    )
    return parser.parse_args()


def normalize_method(method: str) -> str:
    return METHOD_ALIASES.get(method, method)


def load_model_registry():
    config_path = PROJECT_ROOT / "LongBench" / "config" / "model2path.json"
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_model_path(model_name: str, model_path: str | None) -> str:
    if model_path:
        return model_path

    registry = load_model_registry()
    registry_lower = {k.lower(): v for k, v in registry.items()}

    if model_name in registry:
        return registry[model_name]

    lowered = model_name.lower()
    if lowered in registry_lower:
        return registry_lower[lowered]

    alias_name = MODEL_REGISTRY_ALIAS_MAP.get(lowered)
    if alias_name and alias_name in registry:
        return registry[alias_name]

    return model_name


def infer_artifact_name(model_name: str | None, model_path: str, explicit_name: str | None) -> str:
    if explicit_name:
        return explicit_name

    if model_name:
        lowered = model_name.lower()
        if lowered in ARTIFACT_NAME_MAP:
            return ARTIFACT_NAME_MAP[lowered]

    return Path(model_path.rstrip("/")).name


def load_small_state(model, artifact_name: str):
    distilled_dir = PROJECT_ROOT / "data" / artifact_name
    small_state_path = distilled_dir / "small_attn_weights.pt"
    if not small_state_path.exists():
        raise FileNotFoundError(f"small_attn_weights.pt not found: {small_state_path}")

    try:
        small_state = torch.load(small_state_path, map_location="cpu", weights_only=True)
    except TypeError:
        small_state = torch.load(small_state_path, map_location="cpu")
    base_state = model.state_dict()
    matched = 0

    for name, param in small_state.items():
        if name in base_state and base_state[name].shape == param.shape:
            base_state[name] = param
            matched += 1

    model.load_state_dict(base_state)

    if matched == 0:
        raise RuntimeError(f"No tensors from {small_state_path} matched the current model state_dict.")

    print(f"Loaded small-state tensors: {matched}/{len(small_state)} matched from {small_state_path}")
    return model


def load_base_model_and_tokenizer(model_path: str, attn_impl: str):
    print(f"Loading model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = LlamaForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )
    model.eval()
    return model, tokenizer


def build_model_for_method(method: str, model_path: str, artifact_name: str, args):
    model, tokenizer = load_base_model_and_tokenizer(model_path, args.attn_impl)

    if method == "full":
        print("Using Full Attention (original model) ...")
        return model, tokenizer

    if method == "ours":
        quant_mode = None if str(args.quant_mode).lower() == "none" else args.quant_mode
        print(
            f"Applying Ours attention (topk={args.topk_ratio}, sink={args.sink_size}, "
            f"local_window={args.local_window}, quant={quant_mode}) ..."
        )
        old_cwd = os.getcwd()
        os.chdir(PROJECT_ROOT)
        try:
            model = apply_compression_to_model(
                model,
                artifact_name,
                quant_mode=quant_mode,
                topk_ratio=args.topk_ratio,
                sink_size=args.sink_size,
                local_window=args.local_window,
                escaped_layers=args.escaped_layers,
            )
        finally:
            os.chdir(old_cwd)
        model = load_small_state(model, artifact_name)
        return model, tokenizer

    if method == "h2o":
        print(
            f"Applying H2O attention (heavy_ratio={args.topk_ratio}, "
            f"sink={args.sink_size}, local_window={args.local_window}) ..."
        )
        model.config.heavy_ratio = args.topk_ratio
        model.config.sink_size = args.sink_size
        model.config.local_window = args.local_window
        model.config.escaped_layers = args.escaped_layers
        model = convert_kvcache_llama_heavy_recent(model, model.config)
        return model, tokenizer

    if method == "slide":
        slide_window_ratio = args.slide_window_ratio
        if slide_window_ratio is None:
            slide_window_ratio = args.topk_ratio

        print(
            f"Applying Slide/Streaming attention (sink={args.sink_size}, "
            f"local_window={args.local_window}, window_ratio={slide_window_ratio}) ..."
        )
        model.config.sink_size = args.sink_size
        model.config.local_window = args.local_window
        model.config.local_window_ratio = slide_window_ratio
        model.config.escaped_layers = args.escaped_layers
        model = convert_kvcache_llama_sliding_window(model, model.config)
        return model, tokenizer

    raise ValueError(f"Unsupported method: {method}")


def reset_runtime_state(model):
    for module in model.modules():
        if hasattr(module, "_reset_masks"):
            module._reset_masks()


def build_generate_kwargs(method: str, model, tokenizer, inputs, args):
    input_len = inputs["input_ids"].shape[1]
    streamer = StreamTokens(tokenizer, input_len)

    kwargs = {
        **inputs,
        "min_new_tokens": args.min_new_tokens,
        "max_new_tokens": args.max_new_tokens,
        "use_cache": True,
        "repetition_penalty": args.repetition_penalty,
        "stopping_criteria": StoppingCriteriaList([streamer]),
        "num_beams": 1,
        "pad_token_id": tokenizer.eos_token_id,
    }

    if args.do_sample:
        kwargs["do_sample"] = True
        kwargs["temperature"] = args.temperature
        kwargs["top_p"] = args.top_p
    else:
        kwargs["do_sample"] = False

    if method == "ours":
        kwargs["past_key_values"] = KVWithSmallKCache(config=model.config)

    return kwargs


def read_prompt(prompt_file: str) -> str:
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    with open(prompt_file, "r", encoding="utf-8") as f:
        return f.read()


def save_output(output_dir: str, method: str, prompt_file: str, generated_text: str):
    os.makedirs(output_dir, exist_ok=True)
    prompt_stem = Path(prompt_file).stem
    output_path = Path(output_dir) / f"{prompt_stem}.{method}.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(generated_text)
    print(f"Saved output to {output_path}")


def infer_score_profile(prompt_file: str, explicit_profile: str) -> str | None:
    if explicit_profile == "none":
        return None
    if explicit_profile != "auto":
        return explicit_profile

    stem = Path(prompt_file).stem
    if stem.startswith("long_continuation_demo"):
        return "long_continuation_demo"
    return None


def _collect_matches(text: str, terms: list[str]) -> list[str]:
    found = []
    for term in terms:
        if term in text and term not in found:
            found.append(term)
    return found


def score_long_continuation_demo(text: str) -> dict:
    anchors = [
        "旧剧院",
        "南塔",
        "归零室",
        "第二封信",
        "林伯文",
        "蓝色铁箱",
        "四点十七分",
        "钟塔",
    ]
    article_markers = [
        "作者:",
        "译者:",
        "出版商",
        "作者介绍",
        "出版社",
        "博士生导师",
        "教授",
        "## ",
        "——《",
    ]
    drift_terms = [
        "西安",
        "纽约",
        "洛杉矶",
        "华盛顿",
        "海边",
        "北港",
        "临江",
        "北京",
        "上海",
        "宁安镇",
    ]

    score = 0
    breakdown = []

    def add(points: int, label: str, detail: str):
        nonlocal score
        score += points
        breakdown.append({"label": label, "points": points, "detail": detail})

    has_protagonist = "沈越" in text
    add(20 if has_protagonist else 0, "主角保持", "包含“沈越”" if has_protagonist else "未检测到“沈越”")

    has_location = "宁州" in text
    add(20 if has_location else 0, "地点保持", "包含“宁州”" if has_location else "未检测到“宁州”")

    has_year = "2031年" in text or "2031" in text
    add(10 if has_year else 0, "年份保持", "包含“2031”" if has_year else "未检测到“2031”")

    anchor_hits = _collect_matches(text, anchors)
    anchor_points = min(len(anchor_hits), 3) * 8
    add(
        anchor_points,
        "关键线索延续",
        "命中: " + (", ".join(anchor_hits) if anchor_hits else "无"),
    )

    article_hits = _collect_matches(text, article_markers)
    if article_hits:
        penalty = min(len(article_hits), 3) * 8
        add(-penalty, "文体漂移惩罚", "出现元数据/说明文标记: " + ", ".join(article_hits))
    else:
        add(10, "文体连续性", "未出现明显的作者介绍/书目信息")

    drift_hits = _collect_matches(text, drift_terms)
    if drift_hits:
        penalty = min(len(drift_hits), 3) * 6
        add(-penalty, "设定漂移惩罚", "出现地点/场景漂移词: " + ", ".join(drift_hits))
    else:
        add(10, "场景稳定性", "未检测到明显的地点漂移词")

    normalized_score = max(0, min(100, score))
    return {
        "profile": "long_continuation_demo",
        "score": normalized_score,
        "breakdown": breakdown,
    }


def score_output_text(text: str, score_profile: str | None) -> dict | None:
    if score_profile is None:
        return None
    if score_profile == "long_continuation_demo":
        return score_long_continuation_demo(text)
    raise ValueError(f"Unsupported score profile: {score_profile}")


def load_saved_outputs(output_dir: str, prompt_file: str, methods: list[str]) -> dict[str, str]:
    stem = Path(prompt_file).stem
    outputs = {}
    for method in methods:
        output_path = Path(output_dir) / f"{stem}.{method}.txt"
        if not output_path.exists():
            print(f"Warning: saved output not found for {method}: {output_path}")
            continue
        with open(output_path, "r", encoding="utf-8") as f:
            outputs[method] = f.read()
    return outputs


def print_score_summary(scores: dict[str, dict]):
    if not scores:
        return

    print(f"\n{'=' * 24} SCORES {'=' * 24}")
    ranking = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)
    for method, result in ranking:
        print(f"[{method}] {result['score']}/100")
        for item in result["breakdown"]:
            sign = "+" if item["points"] >= 0 else ""
            print(f"  {sign}{item['points']:>2} {item['label']}: {item['detail']}")


def run_single_method(method: str, model_path: str, artifact_name: str, prompt: str, args):
    banner = f"{'=' * 24} {method.upper()} {'=' * 24}"
    print(f"\n{banner}")

    model, tokenizer = build_model_for_method(method, model_path, artifact_name, args)

    input_device = model.device
    inputs = tokenizer(prompt, return_tensors="pt").to(input_device)
    input_len = inputs["input_ids"].shape[1]

    print(f"Prompt tokens: {input_len}")
    if args.show_prompt:
        print("\n--- Prompt ---\n")
        print(prompt, end="" if prompt.endswith("\n") else "\n")

    print("\n--- Start Generation ---\n")
    model.generation_config.do_sample = args.do_sample
    if not args.do_sample:
        model.generation_config.temperature = None
        model.generation_config.top_p = None
    generate_kwargs = build_generate_kwargs(method, model, tokenizer, inputs, args)
    output = model.generate(**generate_kwargs)
    print("\n\n--- End Generation ---")

    generated_text = tokenizer.decode(output[0][input_len:], skip_special_tokens=True)
    reset_runtime_state(model)

    del output
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return generated_text


def main():
    args = parse_args()
    method = normalize_method(args.method)
    model_path = resolve_model_path(args.model, args.model_path)
    artifact_name = infer_artifact_name(args.model, model_path, args.artifact_name)
    prompt = read_prompt(args.prompt_file)
    score_profile = infer_score_profile(args.prompt_file, args.score_profile)

    if method == "all":
        methods = ["full", "ours", "h2o", "slide"]
    else:
        methods = [method]

    if args.score_only_dir is not None:
        outputs = load_saved_outputs(args.score_only_dir, args.prompt_file, methods)
        scores = {
            current_method: score_output_text(text, score_profile)
            for current_method, text in outputs.items()
            if score_output_text(text, score_profile) is not None
        }
        print_score_summary(scores)
        return

    outputs = {}
    for current_method in methods:
        outputs[current_method] = run_single_method(current_method, model_path, artifact_name, prompt, args)
        if args.output_dir is not None:
            save_output(args.output_dir, current_method, args.prompt_file, outputs[current_method])

    if len(outputs) > 1:
        print(f"\n{'=' * 24} SUMMARY {'=' * 24}")
        for current_method, text in outputs.items():
            preview = text.strip().replace("\n", " ")
            if len(preview) > 200:
                preview = preview[:200] + " ..."
            print(f"[{current_method}] {preview}")

    scores = {
        current_method: score_output_text(text, score_profile)
        for current_method, text in outputs.items()
        if score_output_text(text, score_profile) is not None
    }
    print_score_summary(scores)


if __name__ == "__main__":
    main()
