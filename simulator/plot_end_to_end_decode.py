import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MPL_CONFIG_DIR = ROOT / ".mplconfig"

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties


DEFAULT_CSV_PATH = ROOT / "results" / "end_to_end_decode.csv"
DEFAULT_OUT_DIR = ROOT / "results"

METHOD_ORDER = [
    "Dense-FullKV",
    "Sparse-LinearNaive",
    "Sparse-HashNoMerge",
    "Sparse-HashMerge",
    "Sparse-HashMergePipeline",
]

PLOT_METHOD_ORDER = [
    "Dense-FullKV",
    "Sparse-LinearNaive",
    "Sparse-HashNoMerge",
    "Sparse-HashMerge",
    "Sparse-HashMergePipeline",
]

COLOR_MAP = {
    "Dense-FullKV": "#6E6E6E",
    "Sparse-LinearNaive": "#B22222",
    "Sparse-HashNoMerge": "#D98C10",
    "Sparse-HashMerge": "#2A6DBB",
    "Sparse-HashMergePipeline": "#1F7A4D",
}

MARKER_MAP = {
    "Dense-FullKV": "o",
    "Sparse-LinearNaive": "s",
    "Sparse-HashNoMerge": "^",
    "Sparse-HashMerge": "D",
    "Sparse-HashMergePipeline": "P",
}


def pick_font(candidates):
    for name in candidates:
        try:
            path = font_manager.findfont(name, fallback_to_default=False)
        except Exception:
            continue
        if path and Path(path).exists():
            return path
    return None


def build_fonts():
    chinese_candidates = [
        "SimSun",
        "Songti SC",
        "Noto Serif CJK SC",
        "Source Han Serif SC",
        "AR PL UMing CN",
        "WenQuanYi Zen Hei",
    ]
    english_candidates = [
        "Times New Roman",
        "Nimbus Roman",
        "Liberation Serif",
        "DejaVu Serif",
    ]

    cn_path = pick_font(chinese_candidates)
    en_path = pick_font(english_candidates)

    if cn_path is None:
        raise RuntimeError("未找到可用中文字体。")
    if en_path is None:
        raise RuntimeError("未找到可用英文字体。")

    cn_font = FontProperties(fname=cn_path)
    en_font = FontProperties(fname=en_path)
    return cn_font, en_font, cn_path, en_path


def configure_style(en_font):
    MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": en_font.get_name(),
            "font.size": 15,
            "axes.labelsize": 18,
            "axes.titlesize": 18,
            "legend.fontsize": 14,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "axes.linewidth": 1.1,
            "grid.linewidth": 0.9,
            "grid.alpha": 0.35,
            "axes.unicode_minus": False,
            "figure.dpi": 220,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def resolve_paths():
    csv_candidates = [
        DEFAULT_CSV_PATH,
        ROOT.parent / "experiment_data" / "fpga" / "end_to_end_decode.csv",
    ]
    for path in csv_candidates:
        if path.exists():
            return path, path.parent
    raise FileNotFoundError("未找到 end_to_end_decode.csv。")


def load_rows(csv_path):
    rows = []
    with csv_path.open("r", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(
                {
                    "context_length": int(row["context_length"]),
                    "sparsity": float(row["sparsity"]),
                    "method": row["method"],
                    "latency_ms": float(row["latency_us"]) / 1000.0,
                }
            )
    return rows


def format_context_length(value):
    if value >= 1024:
        return f"{int(value // 1024)}K"
    return str(int(value))


def set_tick_font(axis, font_prop, size):
    for label in axis.get_xticklabels():
        label.set_fontproperties(font_prop)
        label.set_fontsize(size)
    for label in axis.get_yticklabels():
        label.set_fontproperties(font_prop)
        label.set_fontsize(size)


def main():
    csv_path, out_dir = resolve_paths()
    out_dir.mkdir(parents=True, exist_ok=True)

    cn_font, en_font, cn_path, en_path = build_fonts()
    configure_style(en_font)
    rows = load_rows(csv_path)

    context_lengths = sorted({row["context_length"] for row in rows})
    sparsities = sorted({row["sparsity"] for row in rows})

    fig, axes = plt.subplots(1, len(sparsities), figsize=(18.0, 6.8), sharey=True)
    if len(sparsities) == 1:
        axes = [axes]

    legend_handles = []
    legend_labels = []

    for index, sparsity in enumerate(sparsities):
        ax = axes[index]
        subset = [row for row in rows if row["sparsity"] == sparsity]

        for method in PLOT_METHOD_ORDER:
            method_rows = sorted(
                [row for row in subset if row["method"] == method],
                key=lambda item: item["context_length"],
            )
            if not method_rows:
                continue

            x_values = [row["context_length"] for row in method_rows]
            y_values = [row["latency_ms"] for row in method_rows]

            handle = ax.plot(
                x_values,
                y_values,
                color=COLOR_MAP[method],
                marker=MARKER_MAP[method],
                linewidth=2.8 if method != "Dense-FullKV" else 3.2,
                markersize=8.8 if method != "Dense-FullKV" else 9.8,
                markerfacecolor="white" if method == "Dense-FullKV" else COLOR_MAP[method],
                markeredgecolor=COLOR_MAP[method],
                markeredgewidth=1.8,
                alpha=0.98,
                label=method,
            )[0]

            if index == 0:
                legend_handles.append(handle)
                legend_labels.append(method)

        ax.set_yscale("log")
        ax.grid(True, which="major", linestyle="--", linewidth=0.9, alpha=0.38)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.65, alpha=0.18)
        ax.set_xticks(context_lengths)
        ax.set_xticklabels([format_context_length(value) for value in context_lengths])
        ax.set_xlabel("上下文长度（Token）", fontproperties=cn_font, fontsize=17, labelpad=8)
        ax.set_title(
            f"稀疏率 {int(round(sparsity * 100))}%",
            fontproperties=cn_font,
            fontsize=18,
            pad=12,
        )
        set_tick_font(ax, en_font, 15)
        ax.tick_params(axis="both", width=1.0, length=5)
        for spine in ax.spines.values():
            spine.set_linewidth(1.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("单步解码时延（ms，对数坐标）", fontproperties=cn_font, fontsize=18, labelpad=10)

    legend = fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=5,
        frameon=False,
        prop=en_font,
        handlelength=2.2,
        columnspacing=1.3,
    )
    for text in legend.get_texts():
        text.set_fontproperties(en_font)
        text.set_fontsize(14)

    fig.tight_layout(rect=[0.02, 0.05, 0.995, 0.92])

    png_path = out_dir / "end_to_end_decode_latency_zh.png"
    pdf_path = out_dir / "end_to_end_decode_latency_zh.pdf"
    svg_path = out_dir / "end_to_end_decode_latency_zh.svg"

    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"输入 CSV: {csv_path}")
    print(f"中文字体: {cn_path}")
    print(f"英文字体: {en_path}")
    print(f"PNG 输出: {png_path}")
    print(f"PDF 输出: {pdf_path}")
    print(f"SVG 输出: {svg_path}")


if __name__ == "__main__":
    main()
