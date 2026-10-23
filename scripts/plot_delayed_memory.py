"""Plot the delayed-goal benchmark from published aggregate metrics."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=Path("reports/delayed-memory-v1/final.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/delayed-memory-v1/comparison.png"))
    args = parser.parse_args()
    cache = Path("runs/plot-cache").resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads(args.report.read_text(encoding="utf-8"))
    seeds = report["config"]["optimizer_seeds"]
    policies = ["untrained", "random", "fifo", "graph_scheduled", "imitation_only", "learned"]
    labels = ["Untrained", "Random", "FIFO", "Scheduled\ngraph", "Imitation\nonly", "Learned"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.1), layout="constrained")
    for axis, metric, scale, title in zip(axes, ["accuracy", "utility"], [100, 1],
                                          ["Delayed-question accuracy (%)", "Accuracy minus write cost"]):
        values = [[report["results_by_seed"][str(seed)][policy][metric] * scale for seed in seeds]
                  for policy in policies]
        means = [sum(row) / len(row) for row in values]
        axis.bar(range(len(policies)), means, color=["#b5bdc7"] * 5 + ["#267dba"], width=0.65)
        for i, row in enumerate(values):
            offsets = [(j - (len(row) - 1) / 2) * 0.10 for j in range(len(row))]
            axis.scatter([i + offset for offset in offsets], row, color="black", s=17, zorder=3)
        axis.set_xticks(range(len(policies)), labels, fontsize=9)
        axis.set_title(title, fontsize=11)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    bound = report["references"]["offline_best_subset_mean_accuracy"] * 100
    axes[0].axhline(bound, color="#a33b26", linestyle="--", label=f"Offline storage-only bound: {bound:.1f}%")
    axes[0].set_ylim(0, bound + 5)
    axes[0].legend(fontsize=8, loc="upper left")
    axes[1].axhline(0, color="gray", linewidth=1)
    figure.suptitle("Preliminary experiment: 200 shared worlds; black points are 3 optimizer seeds", fontsize=11)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
