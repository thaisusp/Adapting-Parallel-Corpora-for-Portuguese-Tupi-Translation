from __future__ import annotations

import argparse
from pathlib import Path

import evaluate
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .eval_stats import aggregate_run_metrics
from .experiment import (
    DIRECTION_CHOICES,
    EVAL_DATASET_CHOICES,
    MODEL_ID,
    dataset_paths,
    direction_columns,
    direction_languages,
    normalize_direction,
    translate_list,
)
from .utils import (
    ProjectPaths,
    clear_memory,
    compute_mt_metrics,
    get_device,
    prepare_parallel_splits,
    read_tabular,
)

DEVICE = get_device()


def _prediction_file(
    predictions_dir: Path,
    *,
    direction: str,
    dataset: str,
    seed: int,
) -> Path:
    return predictions_dir / f"zeroshot_{direction}_{dataset}_seed{seed}.csv"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run zero-shot NLLB evaluation for one translation direction.")
    parser.add_argument("--direction", choices=DIRECTION_CHOICES, default="pt_to_tupi")
    parser.add_argument("--eval-dataset", choices=EVAL_DATASET_CHOICES, default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--force-prepare", action="store_true")
    return parser.parse_args(argv)


def main(
    *,
    direction: str = "pt_to_tupi",
    eval_dataset: str = "all",
    seed: int = 42,
    batch_size: int = 16,
    max_length: int = 128,
    force_prepare: bool = False,
):
    direction = normalize_direction(direction)

    paths = ProjectPaths.from_file(__file__)
    predictions_dir = paths.results_dir / "predictions" / "zeroshot"
    metrics_dir = paths.results_dir / "metrics" / "zeroshot"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    clear_memory()
    print(f"Loading model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    ).to(DEVICE)

    src_lang, tgt_lang = direction_languages(direction)

    bleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")
    ter = evaluate.load("ter")
    run_rows: list[dict[str, object]] = []

    prepare_parallel_splits(paths, seed=seed, force=force_prepare)
    selected = dataset_paths(paths, eval_dataset)

    for dataset_name, dataset_path in selected.items():
        if not dataset_path.exists():
            print(f"Skipping missing dataset: {dataset_path}")
            continue

        df = read_tabular(dataset_path)
        source_col, target_col = direction_columns(direction, df)

        predictions = translate_list(
            df[source_col].tolist(),
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            device=DEVICE,
            desc_prefix="Zero-shot",
        )

        metrics = compute_mt_metrics(
            predictions,
            df[target_col].tolist(),
            bleu_metric=bleu,
            chrf_metric=chrf,
            ter_metric=ter,
        )

        output_path = _prediction_file(
            predictions_dir,
            direction=direction,
            dataset=dataset_name,
            seed=seed,
        )
        pd.DataFrame(
            {
                "source": df[source_col],
                "target": df[target_col],
                "prediction": predictions,
            }
        ).to_csv(output_path, index=False)

        run_rows.append(
            {
                "ModelType": "zeroshot",
                "Direction": direction,
                "Dataset": dataset_name,
                "Seed": seed,
                "PredictionFile": str(output_path),
                **metrics,
            }
        )
        print(f"Saved predictions: {output_path}")

    if not run_rows:
        print("No zero-shot results generated.")
        return

    runs_df = pd.DataFrame(run_rows).sort_values(["Direction", "Dataset", "Seed"])
    runs_file = metrics_dir / f"zeroshot_{direction}_runs.csv"
    runs_df.to_csv(runs_file, index=False)

    aggregate_df = aggregate_run_metrics(
        runs_df,
        group_cols=["ModelType", "Direction", "Dataset"],
        metric_cols=["BLEU", "TER", "chrF1", "chrF3"],
    )
    aggregate_file = metrics_dir / f"zeroshot_{direction}_aggregate.csv"
    aggregate_df.to_csv(aggregate_file, index=False)

    print(f"Saved run metrics: {runs_file}")
    print(f"Saved aggregate metrics: {aggregate_file}")

    del model, tokenizer
    clear_memory()


if __name__ == "__main__":
    cli_args = _parse_args()
    main(
        direction=cli_args.direction,
        eval_dataset=cli_args.eval_dataset,
        seed=cli_args.seed,
        batch_size=cli_args.batch_size,
        max_length=cli_args.max_length,
        force_prepare=cli_args.force_prepare,
    )
