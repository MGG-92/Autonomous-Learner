"""Recreate static learning curves from the published synthetic experiment data."""
import argparse
import csv
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--output", type=Path, default=Path("reports/learning-curves.png"))
    args = parser.parse_args()
    cache = Path("runs/plot-cache").resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    paths = sorted(args.reports.glob("seed-*-curve.csv"))
    if not paths:
        parser.error("No seed learning curves found in --reports")
    for path in paths:
        with path.open(encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        seed = path.stem.split("-")[1]
        x = list(range(len(rows)))
        axes[0].plot(x, [float(r["accuracy"]) for r in rows], marker=".", label=f"Seed {seed}")
        axes[1].plot(x, [float(r["mean_questions"]) for r in rows], marker=".", label=f"Seed {seed}")
        boundary = max(i for i, r in enumerate(rows) if r["stage"] == "imitation") + 0.5
    for axis in axes:
        axis.axvline(boundary, color="gray", linestyle="--", linewidth=1)
        axis.set_xlabel("Validation checkpoint index")
        axis.grid(alpha=0.25)
    axes[0].set(title="Validation goal accuracy", ylabel="Fraction correct", ylim=(-0.03, 1.05))
    axes[1].set(title="Validation teacher questions", ylabel="Mean calls per task", ylim=(-0.2, 8.4))
    axes[0].legend(loc="lower right")
    figure.suptitle("Preliminary synthetic experiment — dashed line starts RL fine-tuning", fontsize=11)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
