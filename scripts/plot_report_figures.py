"""Create publication figures from verified saved results, without test inference."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from har_project.data import file_hash, write_json
from har_project.finalization import verify_results


def main():
    verify_results(ROOT / "results")
    output = ROOT / "report/figures"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    names = ["Walking", "Upstairs", "Downstairs", "Sitting", "Standing", "Laying"]
    activities = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]
    metrics = json.loads((ROOT / "results/runs/cnn_b_seed42/test_metrics.json").read_text())
    confusion = np.array(metrics["confusion_matrix"])
    normalized = confusion / confusion.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), layout="constrained")
    for ax, values, title, is_count in zip(axes, (confusion, normalized), ("Number of windows", "Recall by actual class"), (True, False)):
        ax.imshow(values, cmap="Blues", vmin=0, vmax=confusion.max() if is_count else 1)
        ax.set(xticks=range(6), yticks=range(6), xticklabels=names, yticklabels=names, xlabel="Predicted activity", ylabel="Actual activity", title=title)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
        for i in range(6):
            for j in range(6):
                label = str(values[i, j]) if is_count else f"{values[i,j]:.1%}"
                ax.text(j, i, label, ha="center", va="center", color="white" if values[i,j] > 0.55 * values.max() else "black", fontsize=9)
    fig.suptitle("CNN-B, seed 42 — official test set, 2,947 windows")
    fig.savefig(output / "confusion_cnn_b_seed42.png", dpi=200)
    plt.close(fig)

    test = pd.read_csv(ROOT / "results/metrics_summary.csv").set_index("model")
    validation = pd.read_csv(ROOT / "results/validation_metrics_summary.csv").set_index("model")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), layout="constrained")
    for ax, metric, title in zip(axes, ("accuracy", "macro_f1"), ("Accuracy", "Macro-F1")):
        x = np.arange(3)
        for offset, (label, table) in enumerate((("Validation", validation), ("Test", test))):
            table = table.loc[["cnn_a", "cnn_b", "cnn_c"]]
            ax.bar(x + (offset - 0.5) * 0.35, table[f"{metric}_mean"], width=0.35, yerr=table[f"{metric}_std"], capsize=4, label=label)
        ax.set(xticks=x, xticklabels=["CNN-A", "CNN-B", "CNN-C"], ylim=(0, 1.04), ylabel=title)
        ax.legend(loc="lower right")
    fig.suptitle("Mean and sample SD over 3 seeds — selected by validation: CNN-B")
    fig.savefig(output / "model_comparison.png", dpi=200)
    plt.close(fig)

    classes = pd.read_csv(ROOT / "results/test_class_metrics.csv")
    table = classes[classes.model == "cnn_b"].groupby("activity").recall.agg(["mean", "std"]).loc[activities]
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.bar(names, table["mean"], yerr=table["std"], capsize=4, color="#2166ac")
    ax.set(ylabel="Recall", ylim=(0, 1.07), title="CNN-B test recall — mean and sample SD over 3 seeds")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.savefig(output / "cnn_b_recall_by_activity.png", dpi=200)
    plt.close(fig)

    subjects = pd.read_csv(ROOT / "results/test_subject_metrics.csv")
    table = subjects[subjects.model == "cnn_b"].groupby("subject_id").accuracy.agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    positions = np.arange(len(table))
    ax.bar(positions, table["mean"], yerr=table["std"], capsize=4, color="#2166ac")
    ax.set(xticks=positions, xticklabels=[str(subject) for subject in table.index], xlabel="Held-out subject ID", ylabel="Accuracy", ylim=(0, 1.07), title="CNN-B test accuracy by subject — mean and sample SD over 3 seeds")
    fig.savefig(output / "cnn_b_accuracy_by_subject.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(10, 9), layout="constrained")
    for row, seed in enumerate((42, 123, 2026)):
        history = pd.read_csv(ROOT / f"results/runs/cnn_b_seed{seed}/history.csv")
        best = json.loads((ROOT / f"results/runs/cnn_b_seed{seed}/validation_metrics.json").read_text())["best_epoch"]
        for col, metric in enumerate(("loss", "accuracy")):
            ax = axes[row, col]
            ax.plot(history.epoch + 1, history[metric], label="Fit-train")
            ax.plot(history.epoch + 1, history[f"val_{metric}"], label="Validation")
            ax.axvline(best, linestyle=":", color="gray", label="Best val-loss epoch")
            ax.set(xlabel="Epoch", ylabel=metric.capitalize(), title=f"Seed {seed}")
            ax.legend(fontsize=8)
    fig.suptitle("CNN-B training history; checkpoints selected by validation loss")
    fig.savefig(output / "cnn_b_learning_curves.png", dpi=180)
    plt.close(fig)
    write_json(output / "provenance.json", {
        "protocol_lock_sha256": file_hash(ROOT / "results/protocol_lock.json"),
        "script_sha256": file_hash(__file__),
        "source_csv_sha256": {p.name: file_hash(p) for p in (ROOT / "results").glob("*.csv")},
        "figure_sha256": {p.name: file_hash(p) for p in output.glob("*.png")},
        "test_inference_run": False,
    })
    print("Created five report figures from verified existing results.")


if __name__ == "__main__":
    main()
