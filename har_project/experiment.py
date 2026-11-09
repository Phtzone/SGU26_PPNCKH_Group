"""CNN training and validation only; no test-set scoring in this module."""
import csv
import importlib.metadata
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .data import ACTIVITIES, CHANNELS, file_hash, write_json


def classification_metrics(y_true, y_pred):
    labels = list(range(6))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "report": classification_report(y_true, y_pred, labels=labels, target_names=ACTIVITIES, zero_division=0, output_dict=True),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def baseline_validation(y_train, y_validation):
    predicted_class = int(np.bincount(y_train, minlength=6).argmax())
    result = classification_metrics(y_validation, np.full_like(y_validation, predicted_class))
    return {"split": "validation", "predicted_class": predicted_class, "activity": ACTIVITIES[predicted_class], **result}


def build_model(name):
    if name not in ("cnn_a", "cnn_b", "cnn_c"):
        raise ValueError("Model must be cnn_a, cnn_b or cnn_c")
    import tensorflow as tf

    layers = tf.keras.layers
    network = [layers.Input((128, 9)), layers.Conv1D(64, 5, padding="same", activation="relu"), layers.MaxPooling1D(2)]
    if name in ("cnn_b", "cnn_c"):
        network.append(layers.Conv1D(128, 3, padding="same", activation="relu"))
    if name == "cnn_c":
        network.append(layers.Conv1D(128, 3, padding="same", activation="relu"))
    network += [layers.GlobalAveragePooling1D(), layers.Dense(64, activation="relu"), layers.Dropout(0.3), layers.Dense(6, activation="softmax")]
    model = tf.keras.Sequential(network, name=name)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def _dataset(tf, x, y, seed, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        dataset = dataset.shuffle(len(y), seed=seed, reshuffle_each_iteration=True)
    options = tf.data.Options()
    options.deterministic = True
    options.threading.private_threadpool_size = 2
    options.threading.max_intra_op_parallelism = 2
    return dataset.batch(64).with_options(options).prefetch(1)


def train_run(prepared_path, output, model_name="cnn_b", seed=42, smoke=False):
    output, prepared_path = Path(output), Path(prepared_path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing evidence: {output}")
    if not smoke and seed not in (42, 123, 2026):
        raise ValueError("Protocol 1.0 permits official seeds 42, 123, 2026")
    import tensorflow as tf

    try:
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError:
        # A notebook may already have initialized TensorFlow; record actual values.
        pass
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    with np.load(prepared_path) as arrays:
        # Test arrays are deliberately not accessed by training or validation.
        x_train, y_train = arrays["x_train"], arrays["y_train"]
        x_validation, y_validation = arrays["x_validation"], arrays["y_validation"]
    summary_path = prepared_path.parent.parent / "data_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if file_hash(prepared_path) != summary["prepared_sha256"]:
        raise ValueError("Prepared arrays differ from their recorded SHA-256")
    model = build_model(model_name)
    output.mkdir(parents=True)
    config = {
        "model": model_name, "seed": seed, "kind": "smoke" if smoke else "official",
        "epochs_max": 1 if smoke else 50, "batch_size": 64, "optimizer": "Adam", "learning_rate": 0.001,
        "early_stopping_monitor": "val_loss", "early_stopping_patience": 8, "restore_best_weights": True,
        "dropout_rate": 0.3, "padding": "same", "channels": list(CHANNELS),
        "train_samples": len(y_train), "validation_samples": len(y_validation), "parameters": model.count_params(),
        "test_evaluated": False, "prepared_sha256": summary["prepared_sha256"],
        "validation_subjects": [5, 11, 19, 26],
        "source_sha256": {p.name: file_hash(p) for p in (Path(__file__), Path(__file__).with_name("data.py"))},
    }
    protocol = Path(__file__).resolve().parents[1] / "docs/research/protocol.md"
    config["protocol_sha256"] = file_hash(protocol)
    write_json(output / "config.json", config)
    packages = ("tensorflow", "keras", "numpy", "pandas", "scikit-learn", "matplotlib", "nbformat", "nbclient", "ipykernel")
    write_json(output / "environment.json", {
        "python": sys.version, "platform": platform.platform(), "processor": platform.processor(),
        "cpu_count": os.cpu_count(), "devices": [str(device) for device in tf.config.list_physical_devices()],
        "versions": {package: importlib.metadata.version(package) for package in packages},
        "deterministic_ops": True, "oneDNN": os.environ.get("TF_ENABLE_ONEDNN_OPTS"),
        "intra_op_threads": tf.config.threading.get_intra_op_parallelism_threads(),
        "inter_op_threads": tf.config.threading.get_inter_op_parallelism_threads(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    })
    write_json(output / "baseline_validation.json", baseline_validation(y_train, y_validation))
    (output / "architecture.json").write_text(model.to_json(), encoding="utf-8")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(output / "model.keras"), monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.CSVLogger(str(output / "history.csv")),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    start = time.perf_counter()
    history = model.fit(
        _dataset(tf, x_train, y_train, seed, shuffle=True),
        validation_data=_dataset(tf, x_validation, y_validation, seed),
        epochs=config["epochs_max"], callbacks=callbacks, shuffle=False, verbose=2,
    )
    duration = time.perf_counter() - start
    if not all(np.isfinite(value) for values in history.history.values() for value in values):
        raise ValueError("Nonfinite training history; run must not be reported as valid")
    best_model = tf.keras.models.load_model(output / "model.keras")
    probabilities = best_model.predict(x_validation, batch_size=64, verbose=0)
    in_memory = model.predict(x_validation[:64], batch_size=64, verbose=0)
    np.testing.assert_allclose(probabilities[:64], in_memory, rtol=1e-5, atol=1e-6)
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Model probabilities are invalid")
    metrics = {
        "split": "validation", "kind": config["kind"], "model": model_name, "seed": seed,
        "epochs_run": len(history.history["loss"]), "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
        "training_seconds": duration, "parameters": model.count_params(), "checkpoint_reload_verified": True,
        "test_evaluated": False, **classification_metrics(y_validation, probabilities.argmax(axis=1)),
    }
    write_json(output / "validation_metrics.json", metrics)
    with (output / "validation_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["validation_index", "y_true", "y_pred"] + [f"p_{name}" for name in ACTIVITIES])
        for index, (truth, row) in enumerate(zip(y_validation, probabilities)):
            writer.writerow([index, int(truth), int(row.argmax())] + row.tolist())
    _plot_history(history.history, output / "learning_curves.png", f"{model_name}, seed {seed}, {config['kind']}")
    return metrics


def _plot_history(history, output, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    epochs = np.arange(1, len(history["loss"]) + 1)
    for ax, metric in zip(axes, ("loss", "accuracy")):
        ax.plot(epochs, history[metric], label="Fit-train")
        ax.plot(epochs, history[f"val_{metric}"], label="Validation")
        ax.set(xlabel="Epoch", ylabel=metric.capitalize())
        ax.legend()
    fig.suptitle(title)
    fig.savefig(output, dpi=180)
    plt.close(fig)

