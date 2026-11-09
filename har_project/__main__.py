"""Run preparation or protocol-fixed training from the repository root."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare", help="Validate signals, split subjects and fit train-only scaling")
    preparation.add_argument("--data", default="Dataset/UCI HAR Dataset")
    preparation.add_argument("--output", default="results")
    training = commands.add_parser("train", help="Train and score validation only; never score test")
    training.add_argument("--prepared", default="results/prepared/dataset.npz")
    training.add_argument("--output", default="results")
    training.add_argument("--models", nargs="+", choices=("cnn_a", "cnn_b", "cnn_c"), default=["cnn_b"])
    training.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    training.add_argument("--smoke", action="store_true", help="One epoch, written under results/smoke")
    for name, help_text in (("freeze", "Lock nine runs and select from validation before test"), ("evaluate", "Score test once using the frozen checkpoints"), ("verify", "Verify saved results without running inference")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--output", default="results")
    args = parser.parse_args()
    if args.command == "prepare":
        if (Path(args.output) / "protocol_lock.json").exists():
            raise ValueError("Dataset is locked for final evaluation; use a new output for exploratory work")
        from .data import prepare
        summary = prepare(args.data, args.output)
        print(json.dumps({key: summary[key]["shape"] for key in ("fit_train", "validation", "test")}, indent=2))
    elif args.command == "train":
        if (Path(args.output) / "protocol_lock.json").exists():
            raise ValueError("Experiment is locked; use a new output for exploratory training")
        from .experiment import train_run
        seeds = args.seeds
        if args.smoke and seeds == [42, 123, 2026]:
            seeds = [42]
        # Check the entire batch before any work, rather than discovering a collision mid-batch.
        runs = [(model, seed, Path(args.output) / ("smoke" if args.smoke else "runs") / f"{model}_seed{seed}") for model in args.models for seed in seeds]
        if len({path for _, _, path in runs}) != len(runs):
            raise ValueError("Duplicate model/seed requested")
        for _, _, path in runs:
            if path.exists():
                raise FileExistsError(f"Existing run: {path}; choose a separate output to retain evidence")
        for model, seed, path in runs:
            result = train_run(args.prepared, path, model, seed, smoke=args.smoke)
            print(json.dumps({key: result[key] for key in ("model", "seed", "kind", "accuracy", "macro_f1", "training_seconds")}, indent=2))
    else:
        from .finalization import freeze_experiment, evaluate_frozen, verify_results
        if args.command == "freeze":
            lock = freeze_experiment(args.output)
            print(json.dumps({key: lock[key] for key in ("selected_model", "validation_means", "test_evaluated_at_lock")}, indent=2))
        elif args.command == "evaluate":
            print(evaluate_frozen(args.output).to_string(index=False))
        else:
            print(json.dumps(verify_results(args.output), indent=2))


if __name__ == "__main__":
    main()
