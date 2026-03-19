import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def get_key(d, i, default=None):
    if i in d:
        return d[i]
    if str(i) in d:
        return d[str(i)]
    return default


def setup_fonts():
    simsun_path = r"C:\Windows\Fonts\simsun.ttc"
    if os.path.exists(simsun_path):
        fm.fontManager.addfont(simsun_path)
        cn_font = fm.FontProperties(fname=simsun_path, weight="bold")
    else:
        cn_font = fm.FontProperties(family="SimHei", weight="bold")

    available = {f.name for f in fm.fontManager.ttflist}
    if "Times New Roman" in available:
        en_font = "Times New Roman"
    elif "Times" in available:
        en_font = "Times"
    else:
        en_font = "DejaVu Serif"

    return cn_font, en_font


def method_label(m):
    if m == "mse":
        return "MSE Distillation"
    if m == "topk":
        return "Ranking Loss (Ours)"
    return m.upper()


def load_optional_ppl_curve(path):
    if path is None or not os.path.exists(path):
        return None, None
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)

    if isinstance(d, list):
        steps, ppls = [], []
        for row in d:
            if "step" in row and "ppl" in row:
                steps.append(float(row["step"]))
                ppls.append(float(row["ppl"]))
        return steps, ppls

    if isinstance(d, dict) and "steps" in d and "ppl" in d:
        return [float(x) for x in d["steps"]], [float(x) for x in d["ppl"]]

    return None, None


def _collect_loss_series(loss_d):
    layer_ids = sorted([int(k) for k in loss_d.keys()])
    series = []
    for lid in layer_ids:
        x = get_key(loss_d, lid, default=[])
        if isinstance(x, list) and len(x) > 0:
            arr = np.array(x, dtype=float)
            if np.all(np.isfinite(arr)):
                series.append(arr)
    return series


def _mean_std_curve(series, normalize=False):
    if len(series) == 0:
        return None, None, None

    proc = []
    for s in series:
        if normalize:
            denom = s[0] if abs(s[0]) > 1e-12 else 1.0
            proc.append(s / denom)
        else:
            proc.append(s)

    min_len = min(len(s) for s in proc)
    mat = np.stack([s[:min_len] for s in proc], axis=0)
    mean_curve = mat.mean(axis=0)
    std_curve = mat.std(axis=0)
    steps = np.arange(1, min_len + 1)
    return steps, mean_curve, std_curve


def _set_style(en_font):
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": en_font,
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 1.0,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.8,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })


def save_fig(fig, output_dir, stem):
    png = os.path.join(output_dir, f"{stem}.png")
    pdf = os.path.join(output_dir, f"{stem}.pdf")
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"Saved:\n- {png}\n- {pdf}")


def plot_31_convergence(results, output_dir, en_font, normalize_loss=True):
    _set_style(en_font)
    method_colors = {"mse": "#D55E00", "topk": "#0072B2"}

    fig, ax = plt.subplots(figsize=(6.8, 4.2))

    debug_info = {}
    for m in ["mse", "topk"]:
        if m not in results:
            continue
        c = method_colors[m]
        loss_d = results[m].get("loss", {})
        series = _collect_loss_series(loss_d)
        if len(series) == 0:
            continue

        debug_info[m] = {
            "layer0_first": float(series[0][0]),
            "layer0_last": float(series[0][-1]),
            "num_layers": len(series),
            "len_min": min(len(s) for s in series),
        }

        steps, mean_curve, std_curve = _mean_std_curve(series, normalize=normalize_loss)
        if steps is None:
            continue

        ax.plot(steps, mean_curve, color=c, linewidth=2.2, label=method_label(m))
        ax.fill_between(steps, mean_curve - std_curve, mean_curve + std_curve, color=c, alpha=0.18)

    ylabel = "Normalized Loss" if normalize_loss else "Distillation Loss"
    ax.set_xlabel("Training Step")
    ax.set_ylabel(ylabel)
    ax.set_title("3.1 Distillation Convergence")
    ax.legend(loc="upper right", frameon=True, framealpha=0.95)
    ax.grid(True)
    for s in ax.spines.values():
        s.set_alpha(0.85)

    fig.tight_layout()
    save_fig(fig, output_dir, "fig_3_1_distillation_convergence")
    plt.close(fig)

    if "mse" in debug_info and "topk" in debug_info:
        print("[DEBUG] mse:", debug_info["mse"])
        print("[DEBUG] topk:", debug_info["topk"])


def plot_32_recall(results, output_dir, en_font):
    _set_style(en_font)
    c_before = "#7f7f7f"
    method_styles = {
        "mse": dict(color="#D55E00", marker="s", linewidth=2.0),
        "topk": dict(color="#0072B2", marker="o", linewidth=2.2),
    }

    fig, ax = plt.subplots(figsize=(6.8, 4.2))

    before_recall = results.get("before", {}).get("recall", None)
    ref = None
    for k in ["topk", "mse"]:
        if k in results and "recall" in results[k] and len(results[k]["recall"]) > 0:
            ref = results[k]["recall"]
            break

    if ref:
        layers = sorted([int(k) for k in ref.keys()])
        x = np.array(layers, dtype=int)

        if before_recall is not None:
            ax.plot(
                x,
                np.full_like(x, float(before_recall), dtype=float),
                color=c_before,
                linestyle="--",
                linewidth=1.8,
                label="Before Distillation",
            )

        for method in ["mse", "topk"]:
            if method not in results:
                continue
            d = results[method].get("recall", {})
            y = [get_key(d, i, np.nan) for i in layers]
            style = method_styles[method]
            ax.plot(
                x,
                y,
                color=style["color"],
                marker=style["marker"],
                linewidth=style["linewidth"],
                markersize=4.5,
                label=method_label(method),
            )

        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Top-k Recall")
        ax.set_title("3.2 Layer-wise Recall Comparison")
        ax.set_xticks(x)
        ax.legend(loc="best", frameon=True, framealpha=0.95)
        ax.grid(True)
    else:
        ax.text(0.5, 0.5, "No recall data found", ha="center", va="center")
        ax.set_title("3.2 Layer-wise Recall Comparison")

    for s in ax.spines.values():
        s.set_alpha(0.85)

    fig.tight_layout()
    save_fig(fig, output_dir, "fig_3_2_layerwise_recall")
    plt.close(fig)


def load_last_ppl(path):
    if path is None or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)

    # [{"step":..., "ppl":...}, ...]
    if isinstance(d, list) and len(d) > 0:
        vals = [row.get("ppl", None) for row in d if isinstance(row, dict)]
        vals = [v for v in vals if v is not None]
        return float(vals[-1]) if len(vals) > 0 else None

    # {"steps":[...], "ppl":[...]}
    if isinstance(d, dict) and "ppl" in d and isinstance(d["ppl"], list) and len(d["ppl"]) > 0:
        return float(d["ppl"][-1])

    return None


def plot_ppl_multidataset_bar_by_method(output_dir, en_font):
    """
    每个数据集一条线：
    x轴 = [Before, MSE, Ranking]
    y轴 = 各方法最后一步PPL
    """
    _set_style(en_font)

    datasets = ["pg19", "wikitext", "c4", "ptb", "lambada"]
    methods = [("before", "Before"), ("mse", "MSE"), ("topk", "Ranking")]
    x = np.arange(len(methods))

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]

    has_any = False
    for i, ds in enumerate(datasets):
        y = []
        for m, _ in methods:
            p = os.path.join(output_dir, f"ppl_curve_{m}_{ds}.json")
            v = load_last_ppl(p)
            y.append(np.nan if v is None else float(v))

        if np.all(np.isnan(np.array(y, dtype=float))):
            continue

        has_any = True
        ax.plot(
            x, y,
            marker="o",
            linewidth=2.0,
            markersize=5,
            color=palette[i % len(palette)],
            label=ds.upper(),
        )

    ax.set_xticks(x)
    ax.set_xticklabels([name for _, name in methods])
    ax.set_xlabel("Method")
    ax.set_ylabel("PPL (last step)")
    ax.set_title("PPL Comparison Across Methods")
    if has_any:
        ax.legend(loc="best", frameon=True, framealpha=0.95)
    ax.grid(True, axis="y")

    fig.tight_layout()
    save_fig(fig, output_dir, "fig_3_3_ppl_by_method_laststep")
    plt.close(fig)

def plot_33_ppl_bar(results, output_dir, en_font):
    _set_style(en_font)
    c_before = "#7f7f7f"
    c_mse = "#D55E00"
    c_rank = "#0072B2"

    fig, ax = plt.subplots(figsize=(6.0, 4.2))

    labels, vals, colors = [], [], []
    b = results.get("before", {}).get("ppl", None)
    if b is not None:
        labels.append("Before")
        vals.append(float(b))
        colors.append(c_before)

    if "mse" in results and results["mse"].get("ppl", None) is not None:
        labels.append("MSE")
        vals.append(float(results["mse"]["ppl"]))
        colors.append(c_mse)

    if "topk" in results and results["topk"].get("ppl", None) is not None:
        labels.append("Ranking")
        vals.append(float(results["topk"]["ppl"]))
        colors.append(c_rank)

    if len(vals) > 0:
        bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.8, alpha=0.92)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, h, f"{h:.2f}", ha="center", va="bottom")
        ax.set_ylabel("PPL")
        ax.set_title("3.3 PPL Comparison")
        ax.grid(axis="y")
    else:
        ax.text(0.5, 0.5, "No PPL data found", ha="center", va="center")
        ax.set_title("3.3 PPL Comparison")

    for s in ax.spines.values():
        s.set_alpha(0.85)

    fig.tight_layout()
    save_fig(fig, output_dir, "fig_3_3_ppl_comparison")
    plt.close(fig)


def plot_ppl_multidataset(optional_curves, output_dir, cn_font, en_font):
    valid = [(name, s, p) for name, (s, p) in optional_curves.items() if s is not None and p is not None and len(s) > 0]
    if len(valid) == 0:
        return

    _set_style(en_font)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]

    for i, (name, steps, ppls) in enumerate(valid):
        ax.plot(steps, ppls, linewidth=2.0, color=palette[i % len(palette)], label=name)

    ax.set_xlabel("Evaluated Tokens")
    ax.set_ylabel("Cumulative PPL")
    ax.set_title("多数据集 PPL 曲线对比", fontproperties=cn_font)
    ax.legend(loc="best", frameon=True, framealpha=0.95)
    ax.grid(True)
    for s in ax.spines.values():
        s.set_alpha(0.85)

    fig.tight_layout()
    save_fig(fig, output_dir, "fig_3_3_ppl_multidataset")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot distillation results as THREE separate figures")
    parser.add_argument("--input_json", type=str, required=True, help="Path to distillation_comparison.json")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--raw_loss", action="store_true", help="Use raw loss in Fig 3.1 (default: normalized)")

    parser.add_argument("--ppl_curve_pg19", type=str, default=None)
    parser.add_argument("--ppl_curve_wikitext", type=str, default=None)
    parser.add_argument("--ppl_curve_c4", type=str, default=None)
    parser.add_argument("--ppl_curve_ptb", type=str, default=None)
    parser.add_argument("--ppl_curve_lambada", type=str, default=None)

    # 新增：自动加载 all 数据集（before/mse/topk）
    parser.add_argument("--ppl_all", action="store_true", help="Auto load all ppl_curve_{method}_{dataset}.json from output_dir")
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        results = json.load(f)

    output_dir = args.output_dir or str(Path(args.input_json).parent)
    os.makedirs(output_dir, exist_ok=True)

    cn_font, en_font = setup_fonts()

    # 生成三幅独立图
    plot_31_convergence(results, output_dir, en_font, normalize_loss=(not args.raw_loss))
    plot_32_recall(results, output_dir, en_font)
    plot_33_ppl_bar(results, output_dir, en_font)

    #if args.ppl_all:
    #    optional_curves = {}
    #    methods = ["before", "mse", "topk"]
    #    datasets = ["pg19", "wikitext", "c4", "ptb", "lambada"]
    #    for ds in datasets:
    #        for m in methods:
    #            p = os.path.join(output_dir, f"ppl_curve_{m}_{ds}.json")
    #            key = f"{m.upper()}-{ds.upper()}"
    #            optional_curves[key] = load_optional_ppl_curve(p)
    #else:
    #    optional_curves = {
    #        "PG19": load_optional_ppl_curve(args.ppl_curve_pg19),
    #        "WikiText": load_optional_ppl_curve(args.ppl_curve_wikitext),
    #        "C4": load_optional_ppl_curve(args.ppl_curve_c4),
    #        "PTB": load_optional_ppl_curve(args.ppl_curve_ptb),
    #        "LAMBADA": load_optional_ppl_curve(args.ppl_curve_lambada),
    #    }
#
    #plot_ppl_multidataset(optional_curves, output_dir, cn_font, en_font)
    plot_ppl_multidataset_bar_by_method(output_dir, en_font)


if __name__ == "__main__":
    main()
