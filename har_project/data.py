"""Load UCI HAR windows and prepare subject-disjoint, train-scaled arrays."""
import hashlib
import json
from pathlib import Path

import numpy as np

CHANNELS = (
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
)
ACTIVITIES = ("WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING")
VALIDATION_SUBJECTS = (5, 11, 19, 26)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_split(root, split):
    if split not in ("train", "test"):
        raise ValueError("split must be train or test")
    folder = Path(root) / split
    # Load as floats first so malformed fractional labels/IDs cannot be truncated.
    raw_y = np.loadtxt(folder / f"y_{split}.txt", ndmin=1)
    raw_subjects = np.loadtxt(folder / f"subject_{split}.txt", ndmin=1)
    if raw_y.ndim != 1 or raw_subjects.shape != raw_y.shape or raw_y.size == 0:
        raise ValueError("Label and subject rows must be aligned and nonempty")
    if not np.isfinite(raw_y).all() or not np.isin(raw_y, np.arange(1, 7)).all():
        raise ValueError("Labels must be integers from 1 through 6")
    if not np.isfinite(raw_subjects).all() or not np.isin(raw_subjects, np.arange(1, 31)).all():
        raise ValueError("Subject IDs must be integers from 1 through 30")
    signals = []
    for channel in CHANNELS:
        path = folder / "Inertial Signals" / f"{channel}_{split}.txt"
        values = np.loadtxt(path, dtype=np.float32, ndmin=2)
        if values.shape != (len(raw_y), 128):
            raise ValueError(f"{path.name}: shape {values.shape}; expected aligned rows {(len(raw_y), 128)}")
        if not np.isfinite(values).all():
            raise ValueError(f"{path.name}: nonfinite signal value")
        signals.append(values)
    return np.stack(signals, axis=-1), raw_y.astype(np.int64) - 1, raw_subjects.astype(np.int64)


def subject_indices(subjects, validation_subjects=VALIDATION_SUBJECTS):
    requested = set(validation_subjects)
    missing = requested - set(np.unique(subjects).tolist())
    if missing:
        raise ValueError(f"Validation subjects absent/missing from training data: {sorted(missing)}")
    mask = np.isin(subjects, validation_subjects)
    if mask.all() or not mask.any():
        raise ValueError("Both fit-train and validation must contain samples")
    return np.flatnonzero(~mask), np.flatnonzero(mask)


def fit_normalizer(x_train):
    if x_train.ndim != 3 or x_train.shape[0] == 0 or not np.isfinite(x_train).all():
        raise ValueError("Expected finite, nonempty (samples, time, channels) training data")
    mean = x_train.mean(axis=(0, 1), keepdims=True, dtype=np.float64)
    std = x_train.std(axis=(0, 1), keepdims=True, dtype=np.float64)
    return mean, np.maximum(std, 1e-8)


def normalize(x, mean, std):
    return ((x - mean) / std).astype(np.float32)


def describe_split(x, y, subjects):
    return {
        "shape": list(x.shape), "dtype": str(x.dtype), "finite": bool(np.isfinite(x).all()),
        "subjects": np.unique(subjects).tolist(),
        "class_counts": {name: int(np.sum(y == i)) for i, name in enumerate(ACTIVITIES)},
    }


def prepare(root="Dataset/UCI HAR Dataset", output="results"):
    root, output = Path(root), Path(output)
    x, y, subjects = load_split(root, "train")
    x_test, y_test, test_subjects = load_split(root, "test")
    overlap = set(subjects.tolist()) & set(test_subjects.tolist())
    if overlap:
        raise ValueError(f"Train/test subject overlap: {sorted(overlap)}")
    fit, validation = subject_indices(subjects)
    for selected in (y[fit], y[validation], y_test):
        if set(selected.tolist()) != set(range(6)):
            raise ValueError("All six classes must be present in each split")
    mean, std = fit_normalizer(x[fit])
    arrays = {
        "x_train": normalize(x[fit], mean, std), "y_train": y[fit], "subject_train": subjects[fit],
        "x_validation": normalize(x[validation], mean, std), "y_validation": y[validation],
        "subject_validation": subjects[validation], "x_test": normalize(x_test, mean, std),
        "y_test": y_test, "subject_test": test_subjects,
    }
    prepared = output / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(prepared / "dataset.npz", **arrays)
    np.savez(output / "normalization.npz", mean=mean, std=std, channels=np.array(CHANNELS))
    summary = {
        "channels": list(CHANNELS), "sampling_hz": 50, "window_samples": 128,
        "label_encoding": {str(i): name for i, name in enumerate(ACTIVITIES)},
        "train_original": describe_split(x, y, subjects),
        "fit_train": describe_split(x[fit], y[fit], subjects[fit]),
        "validation": describe_split(x[validation], y[validation], subjects[validation]),
        "test": describe_split(x_test, y_test, test_subjects), "subject_overlap": [],
        "normalization": {"fit_on": "fit_train", "axes": [0, 1], "mean": mean.ravel().tolist(), "std": std.ravel().tolist()},
        "input_sha256": {}, "prepared_sha256": file_hash(prepared / "dataset.npz"),
    }
    for split in ("train", "test"):
        paths = list((root / split / "Inertial Signals").glob(f"*_{split}.txt"))
        paths += [root / split / f"y_{split}.txt", root / split / f"subject_{split}.txt"]
        for path in sorted(paths):
            summary["input_sha256"][path.relative_to(root).as_posix()] = file_hash(path)
    write_json(output / "split.json", {
        "validation_subjects": list(VALIDATION_SUBJECTS), "fit_train_indices_in_official_train": fit.tolist(),
        "validation_indices_in_official_train": validation.tolist(),
        "test_indices_in_official_test": list(range(len(y_test))),
        "fit_train_subjects": np.unique(subjects[fit]).tolist(), "test_subjects": np.unique(test_subjects).tolist(),
    })
    write_json(output / "data_summary.json", summary)
    _plot_data(x, y, fit, validation, y_test, output / "figures")
    return summary


def _plot_data(x, y, fit, validation, y_test, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    positions = np.arange(6)
    for offset, (name, labels) in enumerate((("Fit-train", y[fit]), ("Validation", y[validation]), ("Test", y_test))):
        ax.bar(positions + (offset - 1) * 0.25, np.bincount(labels, minlength=6), width=0.25, label=name)
    ax.set(xticks=positions, xticklabels=["Walk", "Upstairs", "Downstairs", "Sit", "Stand", "Lay"], ylabel="Windows", title="UCI HAR: class distribution by subject-disjoint split")
    ax.legend()
    fig.savefig(output / "class_distribution.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, layout="constrained")
    index = int(fit[np.flatnonzero(y[fit] == 0)[0]])
    time = np.arange(128) / 50
    for group, (title, unit) in enumerate((("Body acceleration", "g"), ("Angular velocity", "rad/s"), ("Total acceleration", "g"))):
        for axis, axis_name in enumerate("XYZ"):
            axes[group].plot(time, x[index, :, 3 * group + axis], label=axis_name, linewidth=1.1)
        axes[group].set(ylabel=unit, title=title)
        axes[group].legend(loc="upper right", ncols=3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"WALKING — official training window {index}, before project standardization")
    fig.savefig(output / "sample_signal_walking.png", dpi=180)
    plt.close(fig)
