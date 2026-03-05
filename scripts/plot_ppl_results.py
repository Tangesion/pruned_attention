import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---- Load data ----
json_path = Path("/home/tgx/projects/pruned_attention/experiment_ppl_results/ppl_results.json")
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

topk = [float(x) for x in data["config"]["topk_ratios"]]
results = data["results"]

# Method display order
methods = ["full_attention", "ours_bf16", "ours_int4", "h2o", "slide"]

# Conference-paper-like style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
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
    "pdf.fonttype": 42,   # editable text in PDF
    "ps.fonttype": 42,
})

# Colorblind-friendly palette + line styles
style_map = {
    "full_attention": dict(color="#000000", marker="o", linestyle="--", linewidth=2.0, markersize=5, label="Full Attention"),
    "ours_bf16":      dict(color="#0072B2", marker="s", linestyle="-",  linewidth=2.2, markersize=5, label="Ours (BF16)"),
    "ours_int4":      dict(color="#009E73", marker="D", linestyle="-",  linewidth=2.2, markersize=5, label="Ours (INT4)"),
    "h2o":            dict(color="#D55E00", marker="^", linestyle="-.", linewidth=2.0, markersize=5, label="H2O"),
    "slide":          dict(color="#CC79A7", marker="v", linestyle=":",  linewidth=2.0, markersize=5, label="Slide"),
}

fig, ax = plt.subplots(figsize=(6.6, 4.2))

for m in methods:
    y = [results[m][str(x)] for x in topk]
    ax.plot(topk, y, **style_map[m])

# Axes formatting
ax.set_xlabel("Top-k Ratio")
ax.set_ylabel("Perplexity (↓)")
ax.set_title("Perplexity vs. Top-k Ratio")
ax.set_xticks(topk)
ax.set_xlim(min(topk) - 0.02, max(topk) + 0.02)

# Tight y-range for better readability
all_vals = []
for m in methods:
    all_vals.extend([results[m][str(x)] for x in topk])
ymin, ymax = min(all_vals), max(all_vals)
pad = max(0.08 * (ymax - ymin), 0.15)
ax.set_ylim(ymin - pad, ymax + pad)

# Legend (inside, top-right)
leg = ax.legend(loc="upper right", frameon=True, framealpha=0.95)
leg.get_frame().set_linewidth(0.8)

# Light spines for paper style
for spine in ax.spines.values():
    spine.set_alpha(0.8)

fig.tight_layout()

out_dir = json_path.parent
png_path = out_dir / "ppl_results_lineplot.png"
pdf_path = out_dir / "ppl_results_lineplot.pdf"

fig.savefig(png_path, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
print(f"Saved:\n- {png_path}\n- {pdf_path}")