from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .adaptation_taxonomy import CHANGE_TYPE_ORDER, parse_change_types
    from .utils import split_alignment_table
except ImportError:  # pragma: no cover
    from adaptation_taxonomy import CHANGE_TYPE_ORDER, parse_change_types
    from utils import split_alignment_table


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def existing_splits_dir(root: Path) -> Path:
    processed = root / "data" / "processed"
    if any(processed.glob("*_historical.csv")) and any(processed.glob("*_adapted.csv")):
        return processed
    return root / "data"


def parse_bool_series(series: pd.Series) -> pd.Series:
    return series.map(lambda value: str(value).strip().lower() in {"true", "1", "yes"})


def adaptation_summary_table(audit_csv: Path) -> pd.DataFrame:
    audit_df = pd.read_csv(audit_csv)
    changed = (
        parse_bool_series(audit_df["changed"])
        if "changed" in audit_df.columns
        else pd.Series(dtype=bool)
    )

    rows: list[dict[str, Any]] = [
        {
            "item": "adapted_corpus_sentences",
            "count": len(audit_df),
            "rate_over_corpus": 1.0 if len(audit_df) else 0.0,
        },
        {
            "item": "sentences_with_changes",
            "count": int(changed.sum()) if len(changed) else 0,
            "rate_over_corpus": round(float(changed.mean()), 4) if len(changed) else 0.0,
        },
        {
            "item": "sentences_without_changes",
            "count": int((~changed).sum()) if len(changed) else 0,
            "rate_over_corpus": round(float((~changed).mean()), 4) if len(changed) else 0.0,
        },
    ]

    type_counts: dict[str, int] = {}
    for value in audit_df.get("change_types", pd.Series(dtype=str)).fillna(""):
        for change_type in parse_change_types(value):
            if change_type == "none":
                continue
            type_counts[change_type] = type_counts.get(change_type, 0) + 1

    ordered_types = CHANGE_TYPE_ORDER + sorted(
        change_type for change_type in type_counts if change_type not in CHANGE_TYPE_ORDER
    )
    for change_type in ordered_types:
        count = type_counts.get(change_type, 0)
        rows.append(
            {
                "item": f"change_type_{change_type}",
                "count": count,
                "rate_over_corpus": round(count / len(audit_df), 4) if len(audit_df) else 0.0,
            }
        )

    return pd.DataFrame(rows)


def write_reports(root: Path, splits_dir: Path, audit_csv: Path) -> tuple[Path, Path]:
    output_dir = root / "results" / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    split_report = split_alignment_table(splits_dir)
    split_report_path = output_dir / "corpus_split_alignment.csv"
    split_report.to_csv(split_report_path, index=False)

    adaptation_report = adaptation_summary_table(audit_csv)
    adaptation_report_path = output_dir / "corpus_adaptation_summary.csv"
    adaptation_report.to_csv(adaptation_report_path, index=False)

    return split_report_path, adaptation_report_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description="Create corpus alignment and adaptation-type summary tables."
    )
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--splits-dir", type=Path, default=None)
    parser.add_argument(
        "--audit-csv",
        type=Path,
        default=root / "data" / "raw" / "oldtupi_portuguese_adapted_audit.csv",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.project_root.resolve()
    splits_dir = args.splits_dir.resolve() if args.splits_dir else existing_splits_dir(root)
    audit_csv = args.audit_csv.resolve()

    if not audit_csv.exists():
        raise FileNotFoundError(f"Adaptation audit not found: {audit_csv}")

    split_report_path, adaptation_report_path = write_reports(root, splits_dir, audit_csv)
    print(f"Saved split alignment table: {split_report_path}")
    print(f"Saved adaptation summary table: {adaptation_report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
