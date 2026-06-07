from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Any

import gc
import re
import unicodedata

import pandas as pd

# Torch is optional so that I/O utilities can run without GPU/torch installed.
try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


_RX_SPACES = re.compile(r"\s+")
_RX_BRACKETS_OR_TAIL = re.compile(r"\[[^\]]*(?:\]|$)")
_RX_PARENTHESES_OR_TAIL = re.compile(r"\([^)]*(?:\)|$)")
SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class ProjectPaths:
    """Basic project paths derived from a script file."""
    script_dir: Path
    project_root: Path
    data_raw: Path
    data_processed: Path
    results_dir: Path

    @staticmethod
    def from_file(file: str | Path) -> "ProjectPaths":
        script_dir = Path(file).resolve().parent
        if (script_dir / "data").exists() and (script_dir / "src").exists():
            project_root = script_dir
        else:
            project_root = script_dir.parent
        return ProjectPaths(
            script_dir=script_dir,
            project_root=project_root,
            data_raw=project_root / "data" / "raw",
            data_processed=project_root / "data" / "processed",
            results_dir=project_root / "results",
        )


def ensure_dir(*paths: str | Path) -> None:
    """Create directories if they do not exist."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def get_device() -> str:
    """Return 'cuda', 'mps', or 'cpu' depending on available hardware."""
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def clean_text(
    text: Any,
    *,
    normalize_form: str = "NFKC",
    remove_brackets: bool = False,
    remove_parenthetical_content: bool = False,
    remove_quotes_symbols: bool = False,
    to_lower: bool = False,
) -> str:
    """
    Normalize and clean text strings.

    Parameters
    ----------
    normalize_form : Unicode normalization form (NFKC is a good default)
    remove_brackets : remove text inside brackets like [comments], including trailing unclosed spans
    remove_parenthetical_content : remove text inside parentheses like (comments)
    remove_quotes_symbols : remove decorative quote symbols often found in corpora
    to_lower : convert text to lowercase
    """
    if not isinstance(text, str):
        return ""

    s = unicodedata.normalize(normalize_form, text)

    if remove_brackets:
        s = _RX_BRACKETS_OR_TAIL.sub("", s)

    if remove_parenthetical_content:
        s = _RX_PARENTHESES_OR_TAIL.sub("", s)

    if remove_quotes_symbols:
        s = re.sub(r'[«»""“”…*_]', "", s)

    if to_lower:
        s = s.lower()

    s = _RX_SPACES.sub(" ", s).strip()

    return s


def normalize_portuguese_text_with_notes(text: Any) -> tuple[str, list[str]]:
    """Normalize Portuguese source text and report deterministic edit notes."""
    raw = "" if pd.isna(text) else str(text)
    nfkc = unicodedata.normalize("NFKC", raw)
    without_brackets = _RX_BRACKETS_OR_TAIL.sub("", nfkc)
    without_parentheses = _RX_PARENTHESES_OR_TAIL.sub("", without_brackets)
    lowered = without_parentheses.lower()
    collapsed = _RX_SPACES.sub(" ", lowered)
    normalized = collapsed.strip()

    notes: list[str] = []
    if raw != nfkc:
        notes.append("unicode_nfkc")
    if nfkc != without_brackets:
        notes.append("brackets_removed")
    if without_brackets != without_parentheses:
        notes.append("parentheses_removed")
    if without_parentheses != lowered:
        notes.append("lowercased")
    if lowered != collapsed:
        notes.append("whitespace_collapsed")
    if collapsed != normalized:
        notes.append("trimmed")
    return normalized, notes


def normalize_portuguese_text(text: Any) -> str:
    return normalize_portuguese_text_with_notes(text)[0]


def clear_memory(verbose: bool = True) -> None:
    """Free Python memory and torch caches when available."""
    gc.collect()

    if torch is not None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    if verbose:
        print("Memory cleared.")


def compute_mt_metrics(
    preds: Sequence[str],
    refs: Sequence[str],
    *,
    bleu_metric: Any,
    chrf_metric: Any,
    ter_metric: Any,
    beta1: int = 1,
    beta3: int = 3,
) -> dict[str, float]:

    preds_clean = [str(x) if pd.notna(x) else "" for x in preds]
    refs_clean = [[str(x) if pd.notna(x) else ""] for x in refs]

    return {
        "BLEU": round(
            bleu_metric.compute(
                predictions=preds_clean,
                references=refs_clean
            )["score"],
            2,
        ),
        "TER": round(
            ter_metric.compute(
                predictions=preds_clean,
                references=refs_clean,
            )["score"],
            2,
        ),
        "chrF1": round(
            chrf_metric.compute(
                predictions=preds_clean,
                references=refs_clean,
                beta=beta1,
            )["score"],
            2,
        ),
        "chrF3": round(
            chrf_metric.compute(
                predictions=preds_clean,
                references=refs_clean,
                beta=beta3,
            )["score"],
            2,
        ),
    }


def detect_parallel_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Return the Portuguese and Tupi columns used by the project datasets."""
    col_pt = (
        "pt_adaptado_clean" if "pt_adaptado_clean" in df.columns
        else "adapted_source_text" if "adapted_source_text" in df.columns
        else "source_text" if "source_text" in df.columns
        else df.columns[0]
    )

    col_tupi = (
        "tupi_clean" if "tupi_clean" in df.columns
        else "target_text" if "target_text" in df.columns
        else df.columns[1]
    )

    return col_pt, col_tupi


def has_model_weights(model_dir: str | Path) -> bool:
    """Check whether a Hugging Face seq2seq model directory has weight files."""
    p = Path(model_dir)
    return any(
        (p / filename).exists()
        for filename in (
            "model.safetensors",
            "pytorch_model.bin",
            "tf_model.h5",
            "flax_model.msgpack",
        )
    )


def read_tabular(path: str | Path, *, encoding: str = "utf-8") -> pd.DataFrame:
    """
    Read CSV or Excel files with simple fallback.

    - CSV: tries ',' first and then ';'
    - Excel: supports .xls / .xlsx
    """

    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    ext = p.suffix.lower()

    if ext in {".xls", ".xlsx"}:
        return pd.read_excel(p)

    if ext == ".csv":
        try:
            return pd.read_csv(p, sep=None, engine="python", encoding=encoding)
        except Exception:
            try:
                return pd.read_csv(p, sep=";", encoding=encoding)
            except Exception:
                return pd.read_csv(p, encoding=encoding)

    raise ValueError(f"Unsupported file format: {ext} ({p})")


def parallel_text_keys(series: pd.Series) -> pd.Series:
    """Normalize a text series for corpus alignment comparisons."""
    return (
        series.astype(str)
        .str.normalize("NFKC")
        .str.replace(_RX_SPACES, " ", regex=True)
        .str.strip()
        .str.casefold()
    )


def _normalize_parallel_dataframe(
    df: pd.DataFrame,
    *,
    remove_brackets: bool,
    remove_quotes_symbols: bool,
    drop_duplicate_pairs: bool,
) -> pd.DataFrame:
    if df.shape[1] < 2:
        raise ValueError("Parallel corpus must have at least two columns.")

    out = df.copy()
    out = out.rename(columns={out.columns[0]: "source_text", out.columns[1]: "target_text"})
    out = out[["source_text", "target_text"]].dropna(subset=["source_text", "target_text"])

    out["source_text"] = out["source_text"].apply(
        lambda x: clean_text(
            normalize_portuguese_text(x),
            remove_brackets=remove_brackets,
        )
    )
    out["target_text"] = out["target_text"].apply(
        lambda x: clean_text(
            x,
            normalize_form="NFKC",
            remove_quotes_symbols=remove_quotes_symbols,
        )
    )
    out = out[(out["source_text"].str.len() > 0) & (out["target_text"].str.len() > 0)]
    if drop_duplicate_pairs:
        out["_source_key"] = parallel_text_keys(out["source_text"])
        out["_target_key"] = parallel_text_keys(out["target_text"])
        out = out.drop_duplicates(subset=["_source_key", "_target_key"], keep="first")
        out = out.drop(columns=["_source_key", "_target_key"])
    return out.reset_index(drop=True)


def normalize_parallel_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Keep a clean two-column parallel corpus with duplicate pairs removed."""
    return _normalize_parallel_dataframe(
        df,
        remove_brackets=True,
        remove_quotes_symbols=True,
        drop_duplicate_pairs=True,
    )


def normalize_technical_parallel_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply only technical text normalization to the Portuguese side."""
    return _normalize_parallel_dataframe(
        df,
        remove_brackets=False,
        remove_quotes_symbols=False,
        drop_duplicate_pairs=False,
    )


def split_alignment_table(splits_dir: Path) -> pd.DataFrame:
    """Report whether historical/adapted splits share the same target order."""
    rows: list[dict[str, Any]] = []
    for split in SPLIT_NAMES:
        historical_path = splits_dir / f"{split}_historical.csv"
        adapted_path = splits_dir / f"{split}_adapted.csv"
        row: dict[str, Any] = {
            "split": split,
            "historical_rows": 0,
            "adapted_rows": 0,
            "same_row_count": False,
            "same_target_order": False,
            "same_target_set": False,
            "target_intersection": 0,
            "historical_only_targets": 0,
            "adapted_only_targets": 0,
            "first_mismatch_index": "",
        }

        if not historical_path.exists() or not adapted_path.exists():
            row["status"] = "missing_split_file"
            rows.append(row)
            continue

        historical_df = read_tabular(historical_path)
        adapted_df = read_tabular(adapted_path)
        historical_targets = parallel_text_keys(historical_df["target_text"]).tolist()
        adapted_targets = parallel_text_keys(adapted_df["target_text"]).tolist()
        historical_set = set(historical_targets)
        adapted_set = set(adapted_targets)

        same_order = historical_targets == adapted_targets
        row.update(
            {
                "historical_rows": len(historical_df),
                "adapted_rows": len(adapted_df),
                "same_row_count": len(historical_df) == len(adapted_df),
                "same_target_order": same_order,
                "same_target_set": historical_set == adapted_set,
                "target_intersection": len(historical_set & adapted_set),
                "historical_only_targets": len(historical_set - adapted_set),
                "adapted_only_targets": len(adapted_set - historical_set),
                "status": "ok" if same_order else "not_aligned",
            }
        )

        if not same_order:
            for idx, (historical_target, adapted_target) in enumerate(
                zip(historical_targets, adapted_targets)
            ):
                if historical_target != adapted_target:
                    row["first_mismatch_index"] = idx
                    break
            if len(historical_targets) != len(adapted_targets) and row["first_mismatch_index"] == "":
                row["first_mismatch_index"] = min(len(historical_targets), len(adapted_targets))

        rows.append(row)

    return pd.DataFrame(rows)


def validate_split_alignment(splits_dir: Path) -> pd.DataFrame:
    """Raise when historical/adapted splits are missing or not sentence-paired."""
    report = split_alignment_table(splits_dir)
    bad = report[report["status"] != "ok"]
    if not bad.empty:
        details = "; ".join(
            f"{row.split}: {row.status}"
            for row in bad.itertuples(index=False)
        )
        raise ValueError(f"Historical/adapted split alignment failed: {details}")
    return report


def split_parallel_dataframe(
    df: pd.DataFrame,
    *,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> dict[str, pd.DataFrame]:
    """Create deterministic train/validation/test splits from one dataframe."""
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1:
        raise ValueError("train_ratio and val_ratio must be between 0 and 1.")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must leave room for test data.")

    shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return {
        "train": shuffled.iloc[:train_end].reset_index(drop=True),
        "val": shuffled.iloc[train_end:val_end].reset_index(drop=True),
        "test": shuffled.iloc[val_end:].reset_index(drop=True),
    }


def prepare_normalized_historical_corpus(paths: ProjectPaths, *, force: bool = False) -> Path:
    """Create the normalized historical corpus used by train/validation/test splits."""
    output_path = paths.data_raw / "oldtupi_portuguese_normalized.csv"
    if output_path.exists() and not force:
        return output_path

    candidates = [
        paths.data_raw / "oldtupi_portuguese_original.csv",
    ]
    input_path = next((p for p in candidates if p.exists()), candidates[0])
    raw_df = read_tabular(input_path)
    normalized_df = normalize_technical_parallel_dataframe(raw_df)
    ensure_dir(output_path.parent)
    normalized_df.to_csv(output_path, index=False, encoding="utf-8")

    return output_path


def _select_largest_existing_tabular(candidates: Sequence[Path]) -> Path:
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return candidates[0]

    best_path = existing[0]
    best_rows = -1
    for path in existing:
        try:
            rows = len(read_tabular(path))
        except Exception:
            rows = -1
        if rows > best_rows:
            best_rows = rows
            best_path = path
    return best_path


def _clean_parallel_with_row_id(df: pd.DataFrame) -> pd.DataFrame:
    if df.shape[1] < 2:
        raise ValueError("Parallel corpus must have at least two columns.")

    out = df.rename(columns={df.columns[0]: "source_text", df.columns[1]: "target_text"})
    out = out[["source_text", "target_text"]].copy()
    out["source_text"] = out["source_text"].apply(normalize_portuguese_text)
    out["target_text"] = out["target_text"].apply(lambda x: clean_text(x, normalize_form="NFKC"))
    out = out[(out["source_text"].str.len() > 0) & (out["target_text"].str.len() > 0)]
    return out.reset_index().rename(columns={"index": "_row_id"})


def _load_paired_corpora(
    historical_file: Path,
    adapted_file: Path,
    audit_file: Path,
) -> dict[str, pd.DataFrame] | None:
    if not audit_file.exists():
        return None

    audit_df = read_tabular(audit_file)
    if "row_id" not in audit_df.columns:
        return None

    historical_df = _clean_parallel_with_row_id(read_tabular(historical_file))
    adapted_df = _clean_parallel_with_row_id(read_tabular(adapted_file))

    row_ids = audit_df["row_id"].dropna().astype(int).tolist()
    if len(row_ids) != len(adapted_df):
        raise ValueError(
            "Adaptation audit and adapted corpus have different row counts: "
            f"audit={len(row_ids)}, adapted={len(adapted_df)}."
        )

    historical_by_id = historical_df.set_index("_row_id", drop=False)
    missing = [row_id for row_id in row_ids if row_id not in historical_by_id.index]
    if missing:
        preview = ", ".join(str(row_id) for row_id in missing[:5])
        raise ValueError(f"Historical corpus is missing adapted row_id values: {preview}")

    historical_paired = historical_by_id.loc[row_ids].reset_index(drop=True)
    adapted_paired = adapted_df.copy()
    adapted_paired["_row_id"] = row_ids

    historical_targets = parallel_text_keys(historical_paired["target_text"])
    adapted_targets = parallel_text_keys(adapted_paired["target_text"])
    mismatches = historical_targets.ne(adapted_targets)
    if bool(mismatches.any()):
        first_idx = int(mismatches[mismatches].index[0])
        raise ValueError(
            "Historical/adapted paired corpora do not share the same Tupi target "
            f"at paired position {first_idx}."
        )

    return {
        "historical": historical_paired,
        "adapted": adapted_paired,
    }


def prepare_parallel_splits(
    paths: ProjectPaths,
    *,
    seed: int = 42,
    force: bool = False,
) -> dict[str, dict[str, int]]:
    """
    Build processed adapted/historical splits from canonical raw files.

    When the adaptation audit is available, both corpora are split from the same
    row_id list so historical/adapted evaluations are sentence-paired. If the
    adaptation is incomplete, the historical corpus is restricted to the adapted
    subset.
    """
    adapted_file = _select_largest_existing_tabular(
        [
            paths.data_raw / "oldtupi_portuguese_adapted.csv",
        ]
    )
    historical_file = prepare_normalized_historical_corpus(paths, force=force)

    corpus_files = {
        "adapted": adapted_file,
        "historical": historical_file,
    }

    required_outputs = [
        paths.data_processed / f"{split}_{corpus}.csv"
        for corpus in corpus_files
        for split in SPLIT_NAMES
    ]

    if not force and all(path.exists() for path in required_outputs):
        validate_split_alignment(paths.data_processed)
        return {
            corpus: {
                split: int(len(read_tabular(paths.data_processed / f"{split}_{corpus}.csv")))
                for split in SPLIT_NAMES
            }
            for corpus in corpus_files
        }

    ensure_dir(paths.data_processed)

    paired = _load_paired_corpora(
        historical_file=historical_file,
        adapted_file=adapted_file,
        audit_file=paths.data_raw / "oldtupi_portuguese_adapted_audit.csv",
    )

    summary: dict[str, dict[str, int]] = {}
    if paired is not None:
        split_ids = split_parallel_dataframe(paired["historical"][["_row_id"]], seed=seed)
        for corpus, clean_df in paired.items():
            by_id = clean_df.set_index("_row_id", drop=False)
            summary[corpus] = {}
            for split, split_df in split_ids.items():
                out_path = paths.data_processed / f"{split}_{corpus}.csv"
                out_df = by_id.loc[split_df["_row_id"].tolist(), ["source_text", "target_text"]]
                out_df.reset_index(drop=True).to_csv(out_path, index=False, encoding="utf-8")
                summary[corpus][split] = int(len(out_df))
        validate_split_alignment(paths.data_processed)
        return summary

    for corpus, input_path in corpus_files.items():
        raw_df = read_tabular(input_path)
        clean_df = normalize_parallel_dataframe(raw_df)
        splits = split_parallel_dataframe(clean_df, seed=seed)
        summary[corpus] = {}
        for split, split_df in splits.items():
            out_path = paths.data_processed / f"{split}_{corpus}.csv"
            split_df.to_csv(out_path, index=False, encoding="utf-8")
            summary[corpus][split] = int(len(split_df))

    validate_split_alignment(paths.data_processed)
    return summary
