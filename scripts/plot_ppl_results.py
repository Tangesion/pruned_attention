import json
import matplotlib.pyplot as plt
import os
import glob
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def plot_ppl_results(results_dir="experiment_ppl_results", output_file="experiment_ppl_results/ppl_comparison.png"):
    # Map filenames to legend labels
    label_map = {
        "original.json": "Full Attention",
        "compressed.json": "Ours (Compressed)",
        "h2o.json": "H2O",
        "sliding_window.json": "Sliding Window"
    }

    # Colors for each method
    colors = {
        "Full Attention": "black",
        "Ours (Compressed)": "red",
        "H2O": "blue",
        "Sliding Window": "green"
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    # Find all json files
    json_files = glob.glob(os.path.join(results_dir, "*.json"))

    if not json_files:
        print(f"No JSON files found in {results_dir}")
        return

    for json_file in json_files:
        basename = os.path.basename(json_file)
        if basename not in label_map:
            continue

        label = label_map[basename]

        with open(json_file, 'r') as f:
            data = json.load(f)

        steps = [d["step"] for d in data if "step" in d and "ppl" in d]
        ppl = [d["ppl"] for d in data if "step" in d and "ppl" in d]

        if steps and ppl:
            ax.plot(steps, ppl, label=label, color=colors.get(label, "gray"), linewidth=2)

    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Perplexity (PPL)")
    ax.set_title("PPL vs. Sequence Length on PG19")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_yscale("linear")  # PPL often spans orders of magnitude, but linear is fine if close

    # Zoomed inset
    zoom_x_min, zoom_x_max = 3072, 4096
    zoom_y_min, zoom_y_max = 9.0, 13.0

    axins = inset_axes(ax, width="38%", height="38%", loc="lower right", borderpad=1)
    for json_file in json_files:
        basename = os.path.basename(json_file)
        if basename not in label_map:
            continue
        label = label_map[basename]
        with open(json_file, 'r') as f:
            data = json.load(f)
        steps = [d["step"] for d in data if "step" in d and "ppl" in d]
        ppl = [d["ppl"] for d in data if "step" in d and "ppl" in d]
        if steps and ppl:
            axins.plot(steps, ppl, color=colors.get(label, "gray"), linewidth=2)

    axins.set_xlim(zoom_x_min, zoom_x_max)
    axins.set_ylim(zoom_y_min, zoom_y_max)
    axins.grid(True, linestyle='--', alpha=0.5)
    axins.set_xticks([])
    axins.set_yticks([])

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    plot_ppl_results()