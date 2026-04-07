import csv
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
SUMMARY_CSV = RESULTS_DIR / "pipeline_overlap_workload_summary.csv"
TIMELINE_CSV = RESULTS_DIR / "pipeline_overlap_representative_timeline.csv"
MPL_CONFIG_DIR = ROOT / ".mplconfig"
FONT_CACHE_DIR = ROOT / ".cache"

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(FONT_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


STAGE_ORDER = ["P", "F", "C"]
STAGE_LABELS = {
    "P": "P-stage (HBM scan)",
    "F": "F-stage (DDR fetch)",
    "C": "C-stage (BF16 compute)",
}
STAGE_STYLES = {
    "P": {"facecolor": "#D55E00", "hatch": "////"},
    "F": {"facecolor": "#0072B2", "hatch": "...."},
    "C": {"facecolor": "#009E73", "hatch": "xx"},
}

SPARSITY_ORDER = [0.01, 0.05, 0.10, 0.20]
SPARSITY_LABELS = {
    0.01: "1%",
    0.05: "5%",
    0.10: "10%",
    0.20: "20%",
}
SPARSITY_COLORS = {
    0.01: "#F7C07A",
    0.05: "#82D18B",
    0.10: "#C8B4E3",
    0.20: "#ECE384",
}
SPARSITY_HATCHES = {
    0.01: "",
    0.05: "///",
    0.10: "\\\\",
    0.20: "xxx",
}


def configure_style():
    MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    FONT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 11.5,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "legend.fontsize": 10.5,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.linewidth": 1.0,
            "grid.linewidth": 0.8,
            "grid.alpha": 0.20,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_summary_rows(path):
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "context_length": int(row["context_length"]),
                    "sparsity_ratio": float(row["sparsity_ratio"]),
                    "num_channels": int(row["num_channels"]),
                    "selected_kv_per_group": int(row["selected_kv_per_group"]),
                    "total_cycles": int(row["total_cycles"]),
                    "serial_cycles": int(row["serial_cycles"]),
                    "speedup_vs_serial": float(row["speedup_vs_serial"]),
                }
            )
    return rows


def load_timeline_rows(path):
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "context_length": int(row["context_length"]),
                    "sparsity_ratio": float(row["sparsity_ratio"]),
                    "num_channels": int(row["num_channels"]),
                    "group_id": int(row["group_id"]),
                    "stage": row["stage"],
                    "start_cycle": int(row["start_cycle"]),
                    "end_cycle": int(row["end_cycle"]),
                    "duration_cycles": int(row["duration_cycles"]),
                }
            )
    return rows


def format_context_tick(context_length):
    return f"{context_length // 1024}"


def format_kcycles(cycles):
    return cycles / 1000.0


def plot_pipeline_schedule(ax, timeline_rows):
    rows_by_key = {(row["group_id"], row["stage"]): row for row in timeline_rows}
    makespan = max(row["end_cycle"] for row in timeline_rows)
    context_length = timeline_rows[0]["context_length"]
    sparsity_ratio = timeline_rows[0]["sparsity_ratio"]
    num_channels = timeline_rows[0]["num_channels"]
    groups = sorted({row["group_id"] for row in timeline_rows}, reverse=True)

    for idx, group_id in enumerate(groups):
        y = idx
        if idx % 2 == 0:
            ax.axhspan(y - 0.46, y + 0.46, color="#F5F5F5", zorder=0)

        for stage in STAGE_ORDER:
            row = rows_by_key[(group_id, stage)]
            start_pct = 100.0 * row["start_cycle"] / makespan
            width_pct = 100.0 * row["duration_cycles"] / makespan
            style = STAGE_STYLES[stage]
            ax.barh(
                y,
                width_pct,
                left=start_pct,
                height=0.70,
                color=style["facecolor"],
                hatch=style["hatch"],
                edgecolor="#333333",
                linewidth=0.9,
                zorder=3,
            )

    ax.set_title(
        "(a) Representative pipeline Gantt chart\n"
        f"({context_length // 1024}K tokens, {int(round(sparsity_ratio * 100))}% sparsity, fixed {num_channels}-channel DDR)",
    )
    ax.set_xlabel("Normalized timeline within makespan (%)")
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([f"G{group_id}" for group_id in groups])
    ax.invert_yaxis()
    ax.set_xlim(0.0, 100.0)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(True, axis="x", linestyle="--", linewidth=0.8, alpha=0.25)
    ax.grid(False, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_grouped_bars(ax, summary_rows):
    context_lengths = sorted({row["context_length"] for row in summary_rows})
    x = np.arange(len(context_lengths), dtype=float)
    width = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width

    max_speedup = max(row["speedup_vs_serial"] for row in summary_rows)
    upper = max(1.9, math.ceil((max_speedup + 0.06) * 10.0) / 10.0)

    for index, sparsity in enumerate(SPARSITY_ORDER):
        series = []
        for context_length in context_lengths:
            row = next(
                item
                for item in summary_rows
                if item["context_length"] == context_length and abs(item["sparsity_ratio"] - sparsity) < 1e-9
            )
            series.append(row["speedup_vs_serial"])

        ax.bar(
            x + offsets[index],
            np.array(series) - 1.0,
            width=width,
            bottom=1.0,
            color=SPARSITY_COLORS[sparsity],
            edgecolor="#7A7A7A",
            linewidth=0.9,
            hatch=SPARSITY_HATCHES[sparsity],
            label=SPARSITY_LABELS[sparsity],
            zorder=3,
        )

    ax.axhline(1.0, color="#7A7A7A", linewidth=1.1, linestyle=":", alpha=0.95)
    ax.text(
        0.02,
        0.96,
        "Higher is better",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.4,
        color="#4C566A",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([format_context_tick(item) for item in context_lengths])
    ax.set_xlabel("Context length (K tokens)")
    ax.set_ylabel("Speedup over serial (x)")
    ax.set_title("(b) Speedup trend across long-context workloads")
    ax.set_ylim(1.0, upper)
    ax.set_yticks(np.arange(1.0, upper + 1e-9, 0.2))
    ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.25, zorder=0)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend = ax.legend(loc="upper right", frameon=False, ncol=2, title="Sparsity")
    legend._legend_box.align = "left"


def main():
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Missing summary CSV: {SUMMARY_CSV}")
    if not TIMELINE_CSV.exists():
        raise FileNotFoundError(f"Missing timeline CSV: {TIMELINE_CSV}")

    configure_style()
    summary_rows = load_summary_rows(SUMMARY_CSV)
    timeline_rows = load_timeline_rows(TIMELINE_CSV)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15.0, 6.2),
        gridspec_kw={"width_ratios": [1.10, 1.0]},
    )

    plot_pipeline_schedule(axes[0], timeline_rows)
    plot_grouped_bars(axes[1], summary_rows)

    stage_handles = [
        Patch(
            facecolor=STAGE_STYLES[stage]["facecolor"],
            hatch=STAGE_STYLES[stage]["hatch"],
            edgecolor="#333333",
            label=STAGE_LABELS[stage],
        )
        for stage in STAGE_ORDER
    ]
    fig.legend(
        handles=stage_handles,
        labels=[STAGE_LABELS[stage] for stage in STAGE_ORDER],
        loc="upper center",
        bbox_to_anchor=(0.28, 1.01),
        ncol=3,
        frameon=False,
    )

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94], pad=1.0, w_pad=2.2)

    out_png = RESULTS_DIR / "pipeline_overlap_combined.png"
    out_pdf = RESULTS_DIR / "pipeline_overlap_combined.pdf"
    out_svg = RESULTS_DIR / "pipeline_overlap_combined.svg"
    fig.savefig(out_png, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(out_svg, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
