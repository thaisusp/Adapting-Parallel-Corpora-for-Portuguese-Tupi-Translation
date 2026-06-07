from __future__ import annotations

import argparse
from pathlib import Path

import evaluate
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .eval_stats import aggregate_run_metrics
from .experiment import (
    DIRECTION_CHOICES,
    EVAL_DATASET_CHOICES,
    TRAIN_DATASET_CHOICES,
    dataset_paths,
    direction_columns,
    direction_languages,
    model_run_dir,
    normalize_direction,
    translate_list,
)
from .utils import (
    ProjectPaths,
    clear_memory,
    compute_mt_metrics,
    get_device,
    has_model_weights,
    read_tabular,
)

DEVICE = get_device()


def _prediction_file(
    out_dir: Path,
    *,
    direction: str,
    dataset: str,
    train_dataset: str,
    seed: int,
) -> Path:
    return out_dir / f"finetuned_{direction}_{dataset}_seed{seed}_train_{train_dataset}.csv"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned NLLB models by direction.")
    parser.add_argument("--train-dataset", choices=TRAIN_DATASET_CHOICES, default="adapted")
    parser.add_argument("--direction", choices=DIRECTION_CHOICES, default="pt_to_tupi")
    parser.add_argument("--eval-dataset", choices=EVAL_DATASET_CHOICES, default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    return parser.parse_args(argv)


def main(
    train_dataset: str = "adapted",
    direction: str = "pt_to_tupi",
    eval_dataset: str = "all",
    seed: int = 42,
    batch_size: int = 16,
    max_length: int = 128,
):
    direction = normalize_direction(direction)

    paths = ProjectPaths.from_file(__file__)
    out_predictions = paths.results_dir / "predictions" / "finetuned"
    out_metrics = paths.results_dir / "metrics" / "finetuned"
    out_predictions.mkdir(parents=True, exist_ok=True)
    out_metrics.mkdir(parents=True, exist_ok=True)

    src_lang, tgt_lang = direction_languages(direction)

    bleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")
    ter = evaluate.load("ter")
    selected_dataset_paths = dataset_paths(paths, eval_dataset)
    run_rows: list[dict[str, object]] = []

    model_final_dir = model_run_dir(
        paths,
        train_dataset=train_dataset,
        direction=direction,
        seed=seed,
    ) / "final"

    if not has_model_weights(model_final_dir):
        raise FileNotFoundError(
            "Missing fine-tuned model weights for "
            f"direction={direction}, train_dataset={train_dataset}, seed={seed}: {model_final_dir}"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_final_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_final_dir).to(DEVICE)

    for dataset_name, dataset_path in selected_dataset_paths.items():
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
            desc_prefix="Fine-tuned",
        )

        metrics = compute_mt_metrics(
            predictions,
            df[target_col].tolist(),
            bleu_metric=bleu,
            chrf_metric=chrf,
            ter_metric=ter,
        )

        pred_file = _prediction_file(
            out_predictions,
            direction=direction,
            dataset=dataset_name,
            train_dataset=train_dataset,
            seed=seed,
        )
        pd.DataFrame(
            {
                "source": df[source_col],
                "target": df[target_col],
                "prediction": predictions,
            }
        ).to_csv(pred_file, index=False)

        run_rows.append(
            {
                "ModelType": "finetuned",
                "TrainDataset": train_dataset,
                "Direction": direction,
                "Dataset": dataset_name,
                "Seed": seed,
                "PredictionFile": str(pred_file),
                **metrics,
            }
        )
        print(f"Saved predictions: {pred_file}")

    del model, tokenizer
    clear_memory(verbose=False)

    if not run_rows:
        print("No fine-tuned predictions generated.")
        return

    runs_df = pd.DataFrame(run_rows).sort_values(
        ["TrainDataset", "Direction", "Dataset", "Seed"]
    )
    run_file = out_metrics / f"finetuned_{direction}_train_{train_dataset}_runs.csv"
    runs_df.to_csv(run_file, index=False)

    aggregate_df = aggregate_run_metrics(
        runs_df,
        group_cols=["ModelType", "TrainDataset", "Direction", "Dataset"],
        metric_cols=["BLEU", "TER", "chrF1", "chrF3"],
    )
    aggregate_file = out_metrics / f"finetuned_{direction}_train_{train_dataset}_aggregate.csv"
    aggregate_df.to_csv(aggregate_file, index=False)

    print(f"Saved run metrics: {run_file}")
    print(f"Saved aggregate metrics: {aggregate_file}")
    clear_memory()


if __name__ == "__main__":
    cli_args = _parse_args()
    main(
        train_dataset=cli_args.train_dataset,
        direction=cli_args.direction,
        eval_dataset=cli_args.eval_dataset,
        seed=cli_args.seed,
        batch_size=cli_args.batch_size,
        max_length=cli_args.max_length,
    )
