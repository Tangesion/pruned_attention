import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def setup_fonts():
    """
    CN: Songti (Windows simsun.ttc preferred)
    EN/Number: Times New Roman
    """
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


def main():
    # ====== path ======
    json_path = Path("./experiment_pruning_results/pruning_results.json")
    out_dir = json_path.parent

    # ====== font/style ======
    CN_FONT, EN_FONT = setup_fonts()

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": EN_FONT,  # 英文数字默认 Times
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 1.0,
        "grid.linewidth": 0.8,
        "grid.alpha": 0.25,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })

    # ====== load data ======
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ours = np.array(data["ours"], dtype=float)
    mag = np.array(data["magnitude"], dtype=float)
    rnd = np.array(data["random"], dtype=float)

    n_layers = len(ours)
    x = np.arange(n_layers)

    means = [ours.mean(), mag.mean(), rnd.mean()]
    stds = [ours.std(ddof=1), mag.std(ddof=1), rnd.std(ddof=1)]

    # ====== figure with 2 subplots ======
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.0, 4.2), gridspec_kw={"width_ratios": [2.2, 1.2]}
    )

    # -------- Subplot (a): Line chart (layer-wise recall) --------
    ax1.plot(x, ours, color="#0072B2", marker="o", linewidth=2.2, markersize=4.5, label="Ours (Sensitivity)")
    ax1.plot(x, mag,  color="#D55E00", marker="s", linewidth=2.0, markersize=4.5, linestyle="-.", label="Magnitude")
    ax1.plot(x, rnd,  color="#009E73", marker="^", linewidth=2.0, markersize=4.5, linestyle=":",  label="Random")

    ax1.axhline(ours.mean(), color="#0072B2", linestyle="--", linewidth=1.2, alpha=0.9)
    ax1.axhline(mag.mean(),  color="#D55E00", linestyle="--", linewidth=1.2, alpha=0.9)
    ax1.axhline(rnd.mean(),  color="#009E73", linestyle="--", linewidth=1.2, alpha=0.9)

    ax1.set_xlabel("层索引", fontproperties=CN_FONT)
    ax1.set_ylabel("Top-k 召回率", fontproperties=CN_FONT)
    ax1.set_title("(a) 分层召回率对比", fontproperties=CN_FONT)
    ax1.set_xticks(x)
    ax1.set_xlim(-0.3, n_layers - 0.7)
    y_max = max(ours.max(), mag.max(), rnd.max())
    y_min = min(ours.min(), mag.min(), rnd.min())
    pad = max(0.03, 0.08 * (y_max - y_min))
    ax1.set_ylim(max(0.0, y_min - pad), min(1.0, y_max + pad))

    leg1 = ax1.legend(loc="lower right", frameon=True, framealpha=0.95)
    leg1.get_frame().set_linewidth(0.8)

    # -------- Subplot (b): Bar chart (mean ± std) --------
    labels = ["Ours", "Magnitude", "Random"]
    colors = ["#0072B2", "#D55E00", "#009E73"]

    bars = ax2.bar(
        labels,
        means,
        yerr=stds,
        capsize=4,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.9
    )

    for b, m in zip(bars, means):
        ax2.text(b.get_x() + b.get_width()/2, m + 0.005, f"{m:.3f}", ha="center", va="bottom", fontsize=10)

    ax2.set_ylabel("平均 Top-k 召回率", fontproperties=CN_FONT)
    ax2.set_title("(b) 均值 ± 标准差", fontproperties=CN_FONT)
    ax2.set_ylim(0.0, min(1.0, max(means) + max(stds) + 0.08))

    # make spines subtle
    for ax in (ax1, ax2):
        for spine in ax.spines.values():
            spine.set_alpha(0.85)

    # overall title
    fig.suptitle("剪枝策略有效性实验结果", fontproperties=CN_FONT, y=1.02, fontsize=13)

    fig.tight_layout()

    # ====== save ======
    png_path = out_dir / "experiment_4_combined_plot.png"
    pdf_path = out_dir / "experiment_4_combined_plot.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    print(f"CN font: {CN_FONT.get_name()}")
    print(f"EN font: {EN_FONT}")
    print(f"Saved:\n- {png_path}\n- {pdf_path}")


if __name__ == "__main__":
    main()