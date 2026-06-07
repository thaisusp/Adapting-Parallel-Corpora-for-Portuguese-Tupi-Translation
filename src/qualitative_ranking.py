from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sacrebleu.metrics import BLEU, CHRF, TER

DIRECTIONS = ["pt_to_tupi", "tupi_to_pt"]
EVAL_DATASETS = ["historical", "adapted"]
METRICS = ["BLEU", "TER", "chrF1", "chrF3"]


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _prediction_file(
    root: Path,
    *,
    direction: str,
    eval_dataset: str,
    train_dataset: str,
    seed: int,
) -> Path:
    return (
        root
        / "results"
        / "predictions"
        / "finetuned"
        / f"finetuned_{direction}_{eval_dataset}_seed{seed}_train_{train_dataset}.csv"
    )


def _read_prediction_pair(
    root: Path,
    *,
    direction: str,
    eval_dataset: str,
    seed: int,
) -> pd.DataFrame:
    adapted_file = _prediction_file(
        root,
        direction=direction,
        eval_dataset=eval_dataset,
        train_dataset="adapted",
        seed=seed,
    )
    historical_file = _prediction_file(
        root,
        direction=direction,
        eval_dataset=eval_dataset,
        train_dataset="historical",
        seed=seed,
    )

    missing = [path for path in [adapted_file, historical_file] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing prediction file(s):\n- " + "\n- ".join(str(path) for path in missing))

    adapted = pd.read_csv(adapted_file).fillna("")
    historical = pd.read_csv(historical_file).fillna("")

    required_cols = {"source", "target", "prediction"}
    for path, df in [(adapted_file, adapted), (historical_file, historical)]:
        missing_cols = required_cols.difference(df.columns)
        if missing_cols:
            raise ValueError(f"{path} is missing columns: {sorted(missing_cols)}")

    if len(adapted) != len(historical):
        raise ValueError(
            f"Prediction files have different row counts for direction={direction}, "
            f"eval_dataset={eval_dataset}: adapted={len(adapted)}, historical={len(historical)}"
        )

    for col in ["source", "target"]:
        mismatch = adapted[col].astype(str).ne(historical[col].astype(str))
        if bool(mismatch.any()):
            first = int(mismatch[mismatch].index[0])
            raise ValueError(
                f"Prediction files are not aligned for direction={direction}, "
                f"eval_dataset={eval_dataset}, column={col}, first_mismatch_row={first}"
            )

    return pd.DataFrame(
        {
            "Direction": direction,
            "EvalDataset": eval_dataset,
            "SentenceIndex": range(len(adapted)),
            "Source": adapted["source"].astype(str),
            "Reference": adapted["target"].astype(str),
            "Prediction_Finetuned_Adapted": adapted["prediction"].astype(str),
            "Prediction_Finetuned_Historical": historical["prediction"].astype(str),
        }
    )


def _score_sentence(metric: object, prediction: str, reference: str) -> float:
    return float(metric.sentence_score(str(prediction), [str(reference)]).score)


def _score_pairs(df: pd.DataFrame) -> pd.DataFrame:
    scorers = {
        "BLEU": BLEU(effective_order=True),
        "TER": TER(),
        "chrF1": CHRF(beta=1),
        "chrF3": CHRF(beta=3),
    }

    rows: list[dict[str, object]] = []
    for row in df.itertuples(index=False):
        for metric_name, scorer in scorers.items():
            historical_score = _score_sentence(
                scorer,
                row.Prediction_Finetuned_Historical,
                row.Reference,
            )
            adapted_score = _score_sentence(
                scorer,
                row.Prediction_Finetuned_Adapted,
                row.Reference,
            )
            raw_delta = adapted_score - historical_score
            quality_delta = -raw_delta if metric_name == "TER" else raw_delta
            rows.append(
                {
                    "Direction": row.Direction,
                    "EvalDataset": row.EvalDataset,
                    "SentenceIndex": row.SentenceIndex,
                    "Metric": metric_name,
                    "Score_Finetuned_Historical": round(historical_score, 4),
                    "Score_Finetuned_Adapted": round(adapted_score, 4),
                    "Delta_AdaptedMinusHistorical": round(raw_delta, 4),
                    "QualityDelta": round(quality_delta, 4),
                    "Source": row.Source,
                    "Reference": row.Reference,
                    "Prediction_Finetuned_Historical": row.Prediction_Finetuned_Historical,
                    "Prediction_Finetuned_Adapted": row.Prediction_Finetuned_Adapted,
                }
            )

    return pd.DataFrame(rows)


def _rank_top_bottom(scores: pd.DataFrame, *, top_k: int) -> pd.DataFrame:
    ranked_parts: list[pd.DataFrame] = []
    sentence_key = [
        "Direction",
        "Source",
        "Reference",
        "Prediction_Finetuned_Historical",
        "Prediction_Finetuned_Adapted",
    ]

    for metric in METRICS:
        metric_df = scores[scores["Metric"].eq(metric)].copy()
        if metric_df.empty:
            continue

        helps = (
            metric_df.sort_values(
                ["QualityDelta", "Direction", "EvalDataset", "SentenceIndex"],
                ascending=[False, True, True, True],
            )
            .drop_duplicates(sentence_key, keep="first")
            .head(top_k)
            .copy()
        )
        helps.insert(4, "RankGroup", "adaptation_helps")
        helps.insert(5, "Rank", range(1, len(helps) + 1))

        hurts = (
            metric_df.sort_values(
                ["QualityDelta", "Direction", "EvalDataset", "SentenceIndex"],
                ascending=[True, True, True, True],
            )
            .drop_duplicates(sentence_key, keep="first")
            .head(top_k)
            .copy()
        )
        hurts.insert(4, "RankGroup", "adaptation_hurts")
        hurts.insert(5, "Rank", range(1, len(hurts) + 1))

        ranked_parts.extend([helps, hurts])

    if not ranked_parts:
        return pd.DataFrame()

    out = pd.concat(ranked_parts, ignore_index=True)
    return out.sort_values(["Metric", "RankGroup", "Rank"], ignore_index=True)


def _write_outputs(ranked: pd.DataFrame, output_dir: Path, *, top_k: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    for pattern in [
        "adaptation_delta_top_bottom_*.csv",
        "best_adaptation_top_*.csv",
        "worst_adaptation_top_*.csv",
    ]:
        for path in output_dir.glob(pattern):
            if path.is_file():
                path.unlink()

    column_order = [
        "Metric",
        "Rank",
        "Direction",
        "EvalDataset",
        "SentenceIndex",
        "Source",
        "Expected_Reference",
        "Prediction_Finetuned_Adapted",
        "Prediction_Finetuned_Historical",
        "Score_Finetuned_Adapted",
        "Score_Finetuned_Historical",
        "Delta_AdaptedMinusHistorical",
        "QualityDelta",
    ]
    rename_cols = {
        "Reference": "Expected_Reference",
    }

    output_files: list[Path] = []
    outputs = [
        ("adaptation_helps", output_dir / f"best_adaptation_top_{top_k}.csv"),
        ("adaptation_hurts", output_dir / f"worst_adaptation_top_{top_k}.csv"),
    ]

    for rank_group, path in outputs:
        group_df = ranked[ranked["RankGroup"].eq(rank_group)].copy()
        group_df = group_df.rename(columns=rename_cols)
        group_df = group_df[column_order].sort_values(["Metric", "Rank"], ignore_index=True)
        group_df.to_csv(path, index=False)
        output_files.append(path)

    return output_files


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rank sentence-level examples where the adapted fine-tuned model "
            "helps or hurts relative to the historical fine-tuned model."
        )
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--direction", choices=["all", *DIRECTIONS], default="all")
    parser.add_argument("--eval-dataset", choices=["all", *EVAL_DATASETS], default="all")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(
    *,
    project_root: Path | None = None,
    seed: int = 42,
    top_k: int = 20,
    direction: str = "all",
    eval_dataset: str = "all",
    output_dir: Path | None = None,
) -> None:
    if top_k < 1:
        raise ValueError("--top-k must be at least 1")

    root = project_root.resolve() if project_root else _project_root()
    selected_directions = DIRECTIONS if direction == "all" else [direction]
    selected_eval_datasets = EVAL_DATASETS if eval_dataset == "all" else [eval_dataset]
    out_dir = output_dir or root / "results" / "qualitative"

    paired_frames: list[pd.DataFrame] = []
    for selected_direction in selected_directions:
        for selected_eval_dataset in selected_eval_datasets:
            paired_frames.append(
                _read_prediction_pair(
                    root,
                    direction=selected_direction,
                    eval_dataset=selected_eval_dataset,
                    seed=seed,
                )
            )

    pairs = pd.concat(paired_frames, ignore_index=True)
    scores = _score_pairs(pairs)
    ranked = _rank_top_bottom(scores, top_k=top_k)
    output_files = _write_outputs(ranked, out_dir, top_k=top_k)

    for path in output_files:
        print(f"Saved qualitative ranking: {path}")


if __name__ == "__main__":
    args = _parse_args()
    main(
        project_root=args.project_root,
        seed=args.seed,
        top_k=args.top_k,
        direction=args.direction,
        eval_dataset=args.eval_dataset,
        output_dir=args.output_dir,
    )
