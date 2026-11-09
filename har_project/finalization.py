"""Freeze validation-based decisions, then evaluate immutable checkpoints on test."""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ACTIVITIES, CHANNELS, file_hash, write_json
from .experiment import classification_metrics

MODELS = ("cnn_a", "cnn_b", "cnn_c")
SEEDS = (42, 123, 2026)
RUN_FILES = ("config.json", "environment.json", "architecture.json", "history.csv", "validation_metrics.json", "validation_predictions.csv", "model.keras")


def select_model(records):
    """Rank complete groups by mean validation Macro-F1 rounded to four decimals."""
    expected = {(model, seed) for model in MODELS for seed in SEEDS}
    actual = [(row["model"], row["seed"]) for row in records]
    if len(actual) != 9 or set(actual) != expected:
        raise ValueError("Expected nine unique model/seed combinations with the same three seeds")
    if any(row.get("split") != "validation" or row.get("kind") != "official" for row in records):
        raise ValueError("Selection requires official validation results only")
    ranked = []
    for model in MODELS:
        rows = [row for row in records if row["model"] == model]
        if len({row["parameters"] for row in rows}) != 1:
            raise ValueError(f"Inconsistent parameter counts: {model}")
        scores = np.array([row["macro_f1"] for row in rows], dtype=float)
        if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            raise ValueError("Invalid validation score")
        ranked.append((-round(float(scores.mean()), 4), rows[0]["parameters"], model))
    return min(ranked)[2]


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _exclusive_json(path, payload):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Lock path escapes outside its root: {relative}")
    return path


def verify_file_hashes(root, hashes):
    root = Path(root).resolve()
    for relative, expected in hashes.items():
        path = _inside(root, relative)
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"Locked file changed or SHA-256 hash mismatch: {relative}")


def verify_checkpoint_predictions(saved, current):
    if saved.shape != current.shape or not np.allclose(saved, current, rtol=1e-5, atol=1e-6):
        raise ValueError("Current checkpoint differs from the saved validation predictions")


def verify_lock(root="results"):
    root = Path(root).resolve()
    lock_path = root / "protocol_lock.json"
    if not lock_path.exists():
        raise FileNotFoundError(f"Test evaluation requires a protocol lock: {lock_path}")
    lock = _read_json(lock_path)
    if lock.get("schema_version") != 1:
        raise ValueError("Unsupported protocol lock schema")
    runs = [f"runs/{model}_seed{seed}" for model in MODELS for seed in SEEDS]
    expected_files = {"prepared/dataset.npz", "data_summary.json", "split.json", "normalization.npz"}
    expected_files.update(f"{run}/{name}" for run in runs for name in RUN_FILES)
    expected_project = {"docs/research/protocol.md"} | {f"har_project/{name}" for name in ("data.py", "experiment.py", "finalization.py", "__main__.py")}
    for split in ("train", "test"):
        expected_project.update(f"Dataset/UCI HAR Dataset/{split}/Inertial Signals/{channel}_{split}.txt" for channel in CHANNELS)
        expected_project.update(f"Dataset/UCI HAR Dataset/{split}/{prefix}_{split}.txt" for prefix in ("y", "subject"))
    if set(lock.get("files", {})) != expected_files or set(lock.get("project_files", {})) != expected_project:
        raise ValueError("Incomplete protocol lock inventory/schema")
    if lock.get("runs") != runs or lock.get("models") != list(MODELS) or lock.get("seeds") != list(SEEDS):
        raise ValueError("Incomplete nine-run protocol lock schema")
    if lock.get("selection_split") != "validation" or lock.get("selection_metric") != "mean_macro_f1" or lock.get("test_evaluated_at_lock") is not False or lock.get("checkpoint_validation_rechecked") is not True:
        raise ValueError("Invalid pre-test selection metadata in protocol lock")
    verify_file_hashes(root, lock["files"])
    verify_file_hashes(root.parent, lock["project_files"])
    records = [_read_json(root / run / "validation_metrics.json") for run in runs]
    selected = select_model(records)
    if lock.get("selected_model") != selected or lock.get("primary_figure") != "cnn_b_seed42" or lock.get("selected_figure") != f"{selected}_seed42":
        raise ValueError("Locked model/figure decision differs from validation-only selection")
    with np.load(root / "prepared/dataset.npz") as arrays:
        baseline_class = int(np.bincount(arrays["y_train"], minlength=6).argmax())
    if lock.get("baseline_class") != baseline_class:
        raise ValueError("Locked baseline was not chosen from training labels")
    for model in MODELS:
        mean = float(np.mean([r["macro_f1"] for r in records if r["model"] == model]))
        if not np.isclose(lock.get("validation_means", {}).get(model, -1), mean, rtol=0, atol=1e-12):
            raise ValueError("Locked validation mean differs from recorded runs")
    return lock


def freeze_experiment(root="results"):
    root = Path(root).resolve()
    if (root / "protocol_lock.json").exists():
        raise FileExistsError("Protocol lock already exists; never overwrite a pre-test decision")
    if (root / "test_evaluation_status.json").exists() or list((root / "runs").glob("*/test_metrics.json")):
        raise ValueError("Cannot create a pre-test lock after test evaluation has started")
    protocol = root.parent / "docs/research/protocol.md"
    summary = _read_json(root / "data_summary.json")
    prepared = root / "prepared/dataset.npz"
    if file_hash(prepared) != summary["prepared_sha256"]:
        raise ValueError("Prepared data SHA-256 mismatch")
    files = {relative: file_hash(root / relative) for relative in ("prepared/dataset.npz", "data_summary.json", "split.json", "normalization.npz")}
    project_files = {"docs/research/protocol.md": file_hash(protocol)}
    for relative, expected in summary["input_sha256"].items():
        source = root.parent / "Dataset/UCI HAR Dataset" / relative
        if file_hash(source) != expected:
            raise ValueError(f"Source dataset changed: {relative}")
        project_files[source.relative_to(root.parent).as_posix()] = expected
    for name in ("data.py", "experiment.py", "finalization.py", "__main__.py"):
        project_files[f"har_project/{name}"] = file_hash(Path(__file__).with_name(name))
    with np.load(prepared) as arrays:
        y_train, y_validation, x_validation = arrays["y_train"], arrays["y_validation"], arrays["x_validation"]
    records, runs = [], []
    common = {
        "kind": "official", "epochs_max": 50, "batch_size": 64, "optimizer": "Adam", "learning_rate": 0.001,
        "early_stopping_monitor": "val_loss", "early_stopping_patience": 8, "restore_best_weights": True,
        "dropout_rate": 0.3, "padding": "same", "channels": list(CHANNELS),
        "train_samples": len(y_train), "validation_samples": len(y_validation), "test_evaluated": False,
        "validation_subjects": [5, 11, 19, 26], "prepared_sha256": summary["prepared_sha256"],
        "protocol_sha256": file_hash(protocol),
    }
    for model in MODELS:
        for seed in SEEDS:
            relative = f"runs/{model}_seed{seed}"
            folder = root / relative
            config = _read_json(folder / "config.json")
            if any(config.get(key) != value for key, value in {**common, "model": model, "seed": seed}.items()):
                raise ValueError(f"Run does not follow the common locked protocol: {relative}")
            for name in ("data.py", "experiment.py"):
                if config["source_sha256"][name] != project_files[f"har_project/{name}"]:
                    raise ValueError(f"Training source differs between runs: {relative}/{name}")
            metrics = _read_json(folder / "validation_metrics.json")
            if not metrics.get("checkpoint_reload_verified") or metrics.get("test_evaluated") or metrics.get("model") != model or metrics.get("seed") != seed or metrics.get("parameters") != config["parameters"]:
                raise ValueError(f"Run is not a verified pre-test checkpoint: {relative}")
            predictions = pd.read_csv(folder / "validation_predictions.csv")
            np.testing.assert_array_equal(predictions["y_true"], y_validation)
            probabilities = predictions[[f"p_{name}" for name in ACTIVITIES]].to_numpy()
            if not np.isfinite(probabilities).all() or (probabilities < 0).any() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
                raise ValueError(f"Invalid saved validation probabilities: {relative}")
            np.testing.assert_array_equal(predictions["y_pred"], probabilities.argmax(axis=1))
            recalculated = classification_metrics(y_validation, predictions["y_pred"].to_numpy())
            if any(not np.isclose(metrics[key], recalculated[key], rtol=0, atol=1e-12) for key in ("accuracy", "macro_f1")):
                raise ValueError(f"Validation metrics differ from saved predictions: {relative}")
            records.append({**metrics, "kind": config["kind"]})
            runs.append(relative)
            for name in RUN_FILES:
                files[f"{relative}/{name}"] = file_hash(folder / name)
    selected = select_model(records)
    # Bind each checkpoint to the validation evidence used to select it, before test access.
    import tensorflow as tf
    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()
    for relative in runs:
        folder = root / relative
        tf.keras.backend.clear_session()
        model = tf.keras.models.load_model(folder / "model.keras")
        config = _read_json(folder / "config.json")
        expected_architecture = _read_json(folder / "architecture.json")["config"]
        if model.name != config["model"] or model.count_params() != config["parameters"] or json.loads(json.dumps(model.get_config())) != expected_architecture:
            raise ValueError(f"Checkpoint architecture differs from training evidence: {relative}")
        saved = pd.read_csv(folder / "validation_predictions.csv")[[f"p_{name}" for name in ACTIVITIES]].to_numpy()
        verify_checkpoint_predictions(saved, model.predict(x_validation, batch_size=64, verbose=0))
        print(f"Validated current checkpoint against saved validation predictions: {folder.name}", flush=True)
    verify_file_hashes(root, files)
    verify_file_hashes(root.parent, project_files)
    lock = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_split": "validation", "selection_metric": "mean_macro_f1",
        "tie_rule": "round mean Macro-F1 to 4 decimals, then fewer parameters, then model name",
        "selected_model": selected, "models": list(MODELS), "seeds": list(SEEDS), "runs": runs,
        "primary_figure": "cnn_b_seed42", "selected_figure": f"{selected}_seed42",
        "baseline_class": int(np.bincount(y_train, minlength=6).argmax()),
        "validation_means": {model: float(np.mean([r["macro_f1"] for r in records if r["model"] == model])) for model in MODELS},
        "test_evaluated_at_lock": False, "checkpoint_validation_rechecked": True, "files": files, "project_files": project_files,
    }
    _exclusive_json(root / "protocol_lock.json", lock)
    verify_lock(root)
    return lock


def evaluate_frozen(root="results"):
    root = Path(root).resolve()
    lock = verify_lock(root)
    status_path = root / "test_evaluation_status.json"
    if status_path.exists() or (root / "baseline_test_metrics.json").exists():
        raise FileExistsError("Test evaluation already started; inspect saved evidence instead of scoring again")
    for relative in lock["runs"]:
        if any((root / relative / name).exists() for name in ("test_metrics.json", "test_predictions.csv")):
            raise FileExistsError(f"Existing test evidence in {relative}")
    # Only now, after checking the complete pre-test lock, import TF and read test arrays.
    import tensorflow as tf
    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()
    with np.load(root / "prepared/dataset.npz") as arrays:
        x_test, y_test, subjects = arrays["x_test"], arrays["y_test"], arrays["subject_test"]
    split = _read_json(root / "split.json")
    indices = np.array(split["test_indices_in_official_test"])
    if not (len(indices) == len(x_test) == len(y_test) == len(subjects)):
        raise ValueError("Test indices, samples, labels and subjects are misaligned")
    lock_hash = file_hash(root / "protocol_lock.json")
    status = {"state": "started", "lock_sha256": lock_hash, "started_utc": datetime.now(timezone.utc).isoformat()}
    _exclusive_json(status_path, status)
    baseline_predictions = np.full_like(y_test, lock["baseline_class"])
    baseline = {"split": "test", "predicted_class": lock["baseline_class"], "activity": ACTIVITIES[lock["baseline_class"]], "lock_sha256": lock_hash, **classification_metrics(y_test, baseline_predictions)}
    write_json(root / "baseline_test_metrics.json", baseline)
    pd.DataFrame({"sample_index": indices, "subject_id": subjects, "y_true": y_test, "y_pred": baseline_predictions}).to_csv(root / "baseline_test_predictions.csv", index=False)
    for relative in lock["runs"]:
        folder = root / relative
        config = _read_json(folder / "config.json")
        training = _read_json(folder / "validation_metrics.json")
        tf.keras.backend.clear_session()
        model = tf.keras.models.load_model(folder / "model.keras")
        probabilities = model.predict(x_test, batch_size=64, verbose=0)
        if probabilities.shape != (len(y_test), 6) or not np.isfinite(probabilities).all() or (probabilities < 0).any() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
            raise ValueError(f"Invalid test probabilities: {relative}")
        predictions = probabilities.argmax(axis=1)
        metrics = {"split": "test", "kind": "official", "model": config["model"], "seed": config["seed"],
                   "parameters": config["parameters"], "training_seconds": training["training_seconds"],
                   "epochs_run": training["epochs_run"], "best_epoch": training["best_epoch"],
                   "lock_sha256": lock_hash, "checkpoint_sha256": lock["files"][f"{relative}/model.keras"],
                   **classification_metrics(y_test, predictions)}
        frame = pd.DataFrame({"sample_index": indices, "subject_id": subjects, "y_true": y_test, "y_pred": predictions})
        for index, name in enumerate(ACTIVITIES):
            frame[f"p_{name}"] = probabilities[:, index]
        frame.to_csv(folder / "test_predictions.csv", index=False)
        write_json(folder / "test_metrics.json", metrics)
        print(f"{folder.name}: test accuracy={metrics['accuracy']:.6f}, macro_f1={metrics['macro_f1']:.6f}", flush=True)
    rebuild_tables(root)
    plot_results(root)
    verify_lock(root)
    result_paths = list((root / "runs").glob("*/test_*.json")) + list((root / "runs").glob("*/test_predictions.csv"))
    result_paths += [root / name for name in ("baseline_test_metrics.json", "baseline_test_predictions.csv", "metrics_per_run.csv", "metrics_summary.csv", "validation_metrics_per_run.csv", "validation_metrics_summary.csv", "test_class_metrics.csv", "test_subject_metrics.csv", "test_error_pairs.csv")]
    result_paths += list((root / "figures").glob("test_*.png"))
    status.update(state="completed", completed_utc=datetime.now(timezone.utc).isoformat(), result_sha256={p.relative_to(root).as_posix(): file_hash(p) for p in result_paths})
    write_json(status_path, status)
    return pd.read_csv(root / "metrics_summary.csv")


def rebuild_tables(root="results"):
    """Recalculate tables from saved predictions without loading any model."""
    root = Path(root).resolve()
    lock = verify_lock(root)
    with np.load(root / "prepared/dataset.npz") as arrays:
        truth, subjects = arrays["y_test"], arrays["subject_test"]
    test_records, validation_records, class_records, subject_records, pairs = [], [], [], [], []
    for relative in lock["runs"]:
        folder = root / relative
        metrics, validation = _read_json(folder / "test_metrics.json"), _read_json(folder / "validation_metrics.json")
        frame = pd.read_csv(folder / "test_predictions.csv")
        np.testing.assert_array_equal(frame["sample_index"], np.arange(len(truth)))
        np.testing.assert_array_equal(frame["y_true"], truth)
        np.testing.assert_array_equal(frame["subject_id"], subjects)
        probabilities = frame[[f"p_{name}" for name in ACTIVITIES]].to_numpy()
        np.testing.assert_array_equal(frame["y_pred"], probabilities.argmax(axis=1))
        derived = classification_metrics(truth, frame["y_pred"].to_numpy())
        if any(not np.isclose(metrics[k], derived[k], rtol=0, atol=1e-12) for k in ("accuracy", "macro_f1")) or metrics["confusion_matrix"] != derived["confusion_matrix"]:
            raise ValueError(f"Test metrics do not match predictions: {relative}")
        fields = ("model", "seed", "split", "accuracy", "macro_f1", "parameters", "training_seconds", "epochs_run", "best_epoch")
        test_records.append({key: metrics[key] for key in fields})
        validation_records.append({key: validation[key] for key in fields})
        identity = {"model": metrics["model"], "seed": metrics["seed"]}
        for activity in ACTIVITIES:
            class_records.append({**identity, "activity": activity, **derived["report"][activity]})
        for subject, group in frame.groupby("subject_id"):
            result = classification_metrics(group.y_true.to_numpy(), group.y_pred.to_numpy())
            subject_records.append({**identity, "subject_id": int(subject), "samples": len(group), "accuracy": result["accuracy"], "macro_f1": result["macro_f1"]})
        confusion = np.array(derived["confusion_matrix"])
        for first in range(6):
            for second in range(first + 1, 6):
                pairs.append({**identity, "class_a": ACTIVITIES[first], "class_b": ACTIVITIES[second], "a_to_b": int(confusion[first, second]), "b_to_a": int(confusion[second, first]), "total_errors": int(confusion[first, second] + confusion[second, first])})
    for records, prefix in ((test_records, ""), (validation_records, "validation_")):
        frame = pd.DataFrame(records).sort_values(["model", "seed"])
        frame.to_csv(root / f"{prefix}metrics_per_run.csv", index=False)
        summary = frame.groupby("model").agg(n_runs=("seed", "count"), accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"), macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"), parameters=("parameters", "first"), training_seconds_mean=("training_seconds", "mean"))
        summary.to_csv(root / f"{prefix}metrics_summary.csv")
    pd.DataFrame(class_records).to_csv(root / "test_class_metrics.csv", index=False)
    pd.DataFrame(subject_records).to_csv(root / "test_subject_metrics.csv", index=False)
    pd.DataFrame(pairs).sort_values(["model", "seed", "total_errors"], ascending=[True, True, False]).to_csv(root / "test_error_pairs.csv", index=False)


def verify_results(root="results"):
    """Verify immutable evidence and independently rescore CSV predictions, without inference."""
    root = Path(root).resolve()
    status_path = root / "test_evaluation_status.json"
    if not status_path.exists() or _read_json(status_path).get("state") != "completed":
        raise ValueError("Test evaluation must be completed before verifying final results")
    status = _read_json(status_path)
    lock = verify_lock(root)
    if status.get("lock_sha256") != file_hash(root / "protocol_lock.json"):
        raise ValueError("Test evaluation references a different protocol lock hash")
    if not status.get("result_sha256"):
        raise ValueError("Missing result SHA-256 inventory")
    verify_file_hashes(root, status["result_sha256"])
    with np.load(root / "prepared/dataset.npz") as arrays:
        truth, subjects = arrays["y_test"], arrays["subject_test"]
    for relative in lock["runs"]:
        metrics = _read_json(root / relative / "test_metrics.json")
        frame = pd.read_csv(root / relative / "test_predictions.csv")
        np.testing.assert_array_equal(frame["sample_index"], np.arange(len(truth)))
        np.testing.assert_array_equal(frame["y_true"], truth)
        np.testing.assert_array_equal(frame["subject_id"], subjects)
        probabilities = frame[[f"p_{name}" for name in ACTIVITIES]].to_numpy()
        if not np.isfinite(probabilities).all() or (probabilities < 0).any() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
            raise ValueError(f"Invalid saved test probabilities: {relative}")
        np.testing.assert_array_equal(frame["y_pred"], probabilities.argmax(axis=1))
        derived = classification_metrics(truth, frame["y_pred"].to_numpy())
        if any(not np.isclose(metrics[key], derived[key], rtol=0, atol=1e-12) for key in ("accuracy", "macro_f1")) or metrics["confusion_matrix"] != derived["confusion_matrix"]:
            raise ValueError(f"Saved prediction metrics differ: {relative}")
        if metrics.get("lock_sha256") != status["lock_sha256"] or metrics.get("checkpoint_sha256") != lock["files"][f"{relative}/model.keras"]:
            raise ValueError(f"Test metrics reference different locked evidence: {relative}")
    baseline = _read_json(root / "baseline_test_metrics.json")
    derived = classification_metrics(truth, np.full_like(truth, lock["baseline_class"]))
    if any(not np.isclose(baseline[key], derived[key], rtol=0, atol=1e-12) for key in ("accuracy", "macro_f1")):
        raise ValueError("Baseline test metrics differ from the train-selected baseline")
    return {"state": "verified", "runs": len(lock["runs"]), "test_windows_per_run": len(truth), "selected_model": lock["selected_model"]}


def plot_results(root="results"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(root).resolve()
    lock = verify_lock(root)
    labels = ["Walk", "Upstairs", "Downstairs", "Sit", "Stand", "Lay"]
    for name in sorted({lock["primary_figure"], lock["selected_figure"]}):
        metrics = _read_json(root / "runs" / name / "test_metrics.json")
        confusion = np.array(metrics["confusion_matrix"])
        normalized = confusion / np.maximum(confusion.sum(axis=1, keepdims=True), 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
        for ax, values, title in zip(axes, (confusion, normalized), ("Window counts", "Recall by actual class")):
            ax.imshow(values, cmap="Blues", vmin=0, vmax=int(confusion.max()) if title == "Window counts" else 1)
            ax.set(xticks=range(6), yticks=range(6), xticklabels=labels, yticklabels=labels, xlabel="Predicted class", ylabel="Actual class", title=title)
            for i in range(6):
                for j in range(6):
                    text = str(values[i, j]) if title == "Window counts" else f"{values[i,j]:.1%}"
                    ax.text(j, i, text, ha="center", va="center", fontsize=8, color="white" if values[i,j] > values.max() * 0.55 else "black")
        fig.suptitle(f"Test confusion matrix — {name}, N={confusion.sum()}")
        fig.savefig(root / "figures" / f"test_confusion_{name}.png", dpi=180)
        plt.close(fig)
    test = pd.read_csv(root / "metrics_summary.csv").set_index("model").loc[list(MODELS)]
    validation = pd.read_csv(root / "validation_metrics_summary.csv").set_index("model").loc[list(MODELS)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for ax, metric, title in zip(axes, ("accuracy", "macro_f1"), ("Accuracy", "Macro-F1")):
        positions = np.arange(3)
        for offset, (label, table) in enumerate((("Validation", validation), ("Test", test))):
            ax.bar(positions + (offset - 0.5) * 0.35, table[f"{metric}_mean"], width=0.35, yerr=table[f"{metric}_std"], capsize=4, label=label)
        ax.set(xticks=positions, xticklabels=["CNN-A", "CNN-B", "CNN-C"], ylabel=title, ylim=(0, 1.04), title=title)
        ax.legend(loc="lower right")
    fig.suptitle("Mean ± sample SD over 3 seeds; model selection uses validation only")
    fig.savefig(root / "figures/test_model_comparison.png", dpi=180)
    plt.close(fig)
