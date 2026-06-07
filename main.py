from __future__ import annotations

import argparse
import sys


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Pipeline orchestrator for PT↔Old Tupi translation experiments.",
    )

    stages = parser.add_argument_group("Stages")
    stages.add_argument("--prepare-data", action="store_true", help="Create clean train/validation/test splits.")
    stages.add_argument("--zero-shot", action="store_true", help="Run zero-shot evaluation.")
    stages.add_argument("--train", action="store_true", help="Run fine-tuning.")
    stages.add_argument("--predict", action="store_true", help="Run evaluation for fine-tuned models.")
    stages.add_argument("--corpus-stats", action="store_true", help="Create corpus alignment and adaptation summary tables.")

    exp = parser.add_argument_group("Experiment parameters")
    exp.add_argument("--train-dataset", choices=["adapted", "historical"], default="adapted")
    exp.add_argument("--eval-dataset", choices=["all", "adapted", "historical"], default="all")
    exp.add_argument("--direction", choices=["pt_to_tupi", "tupi_to_pt"], default="pt_to_tupi")
    exp.add_argument("--model-id", default="facebook/nllb-200-distilled-600M")
    exp.add_argument("--epochs", type=float, default=5)
    exp.add_argument("--learning-rate", type=float, default=5e-5)
    exp.add_argument("--batch-size", type=int, default=4)
    exp.add_argument("--inference-batch-size", type=int, default=16)
    exp.add_argument("--gradient-accumulation-steps", type=int, default=2)
    exp.add_argument("--max-length", type=int, default=128)
    exp.add_argument("--seed", type=int, default=42)
    exp.add_argument("--force", action="store_true", help="Recreate processed splits when preparing data.")

    parser.add_argument(
        "--allow-downloads",
        action="store_true",
        help="Allow stages that may download models/data (zero-shot/train/predict).",
    )

    return parser.parse_args(argv)


def _require_allow_downloads(args: argparse.Namespace, stage: str) -> bool:
    if args.allow_downloads:
        return True
    print(
        f"Stage '{stage}' skipped because it may download models/data. "
        "Re-run with --allow-downloads to execute it.",
        file=sys.stderr,
    )
    return False


def _run_prepare_data(args: argparse.Namespace) -> int:
    from src.utils import ProjectPaths, prepare_parallel_splits

    paths = ProjectPaths.from_file(__file__)
    summary = prepare_parallel_splits(paths, seed=args.seed, force=args.force)
    print(f"Prepared data splits (seed={args.seed}):")
    for corpus, counts in summary.items():
        print(
            f"- {corpus}: "
            f"train={counts['train']}, val={counts['val']}, test={counts['test']}"
        )
    _run_corpus_stats(args, required=False)
    return 0


def _run_corpus_stats(args: argparse.Namespace, *, required: bool = True) -> int:
    from src.corpus_stats import existing_splits_dir, write_reports
    from src.utils import ProjectPaths

    paths = ProjectPaths.from_file(__file__)
    audit_csv = paths.data_raw / "oldtupi_portuguese_adapted_audit.csv"
    if not audit_csv.exists():
        if required:
            raise FileNotFoundError(f"Adaptation audit not found: {audit_csv}")
        print("Corpus stats skipped because the adaptation audit file is missing.")
        return 0

    splits_dir = existing_splits_dir(paths.project_root)
    split_report_path, adaptation_report_path = write_reports(
        paths.project_root,
        splits_dir,
        audit_csv,
    )
    print(f"Saved split alignment table: {split_report_path}")
    print(f"Saved adaptation summary table: {adaptation_report_path}")
    return 0


def _run_zero_shot(args: argparse.Namespace) -> int:
    if not _require_allow_downloads(args, "zero-shot"):
        return 2

    from src import zero_shot

    zero_shot.main(
        direction=args.direction,
        eval_dataset=args.eval_dataset,
        seed=args.seed,
        batch_size=args.inference_batch_size,
        max_length=args.max_length,
        force_prepare=args.force,
    )
    return 0


def _run_train(args: argparse.Namespace) -> int:
    if not _require_allow_downloads(args, "train"):
        return 2

    from src import train

    train.main(
        train_dataset=args.train_dataset,
        direction=args.direction,
        model_id=args.model_id,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        seed=args.seed,
    )
    return 0


def _run_predict(args: argparse.Namespace) -> int:
    if not _require_allow_downloads(args, "predict"):
        return 2

    from src import predict

    predict.main(
        train_dataset=args.train_dataset,
        direction=args.direction,
        eval_dataset=args.eval_dataset,
        seed=args.seed,
        batch_size=args.inference_batch_size,
        max_length=args.max_length,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    selected = [args.prepare_data, args.zero_shot, args.train, args.predict, args.corpus_stats]
    if not any(selected):
        print("No stage selected. Use --help to see available options.")
        return 0

    exit_code = 0
    if args.prepare_data:
        exit_code = max(exit_code, _run_prepare_data(args))
    if args.zero_shot:
        exit_code = max(exit_code, _run_zero_shot(args))
    if args.train:
        exit_code = max(exit_code, _run_train(args))
    if args.predict:
        exit_code = max(exit_code, _run_predict(args))
    if args.corpus_stats:
        exit_code = max(exit_code, _run_corpus_stats(args))

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
