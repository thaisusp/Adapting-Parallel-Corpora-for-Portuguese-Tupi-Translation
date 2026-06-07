from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .utils import ProjectPaths, detect_parallel_columns

MODEL_ID = "facebook/nllb-200-distilled-600M"
LANG_PT = "por_Latn"
LANG_TUPI_PROXY = "grn_Latn"
DIRECTION_CHOICES = ["pt_to_tupi", "tupi_to_pt"]
TRAIN_DATASET_CHOICES = ["adapted", "historical"]
EVAL_DATASET_CHOICES = ["all", "adapted", "historical"]
EVAL_DATASETS = ["historical", "adapted"]


def normalize_direction(direction: str) -> str:
    value = direction.strip().lower().replace("-", "_")
    if value not in DIRECTION_CHOICES:
        raise ValueError(f"Unsupported direction: {direction}")
    return value


def direction_languages(direction: str) -> tuple[str, str]:
    direction = normalize_direction(direction)
    if direction == "pt_to_tupi":
        return LANG_PT, LANG_TUPI_PROXY
    return LANG_TUPI_PROXY, LANG_PT


def direction_columns(direction: str, df: pd.DataFrame) -> tuple[str, str]:
    direction = normalize_direction(direction)
    col_pt, col_tupi = detect_parallel_columns(df)
    if direction == "pt_to_tupi":
        return col_pt, col_tupi
    return col_tupi, col_pt


def model_run_dir(
    paths: ProjectPaths,
    *,
    train_dataset: str,
    direction: str,
    seed: int,
) -> Path:
    return (
        paths.results_dir
        / "models"
        / "finetuned"
        / normalize_direction(direction)
        / train_dataset
        / f"seed{seed}"
    )


def dataset_paths(paths: ProjectPaths, eval_dataset: str) -> dict[str, Path]:
    all_paths = {
        "historical": paths.data_processed / "test_historical.csv",
        "adapted": paths.data_processed / "test_adapted.csv",
    }
    if eval_dataset == "all":
        return all_paths
    return {eval_dataset: all_paths[eval_dataset]}


def translate_list(
    texts: list[str],
    *,
    src_lang: str,
    tgt_lang: str,
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    batch_size: int,
    max_length: int,
    device: str,
    desc_prefix: str,
) -> list[str]:
    model.eval()
    tokenizer.src_lang = src_lang
    predictions: list[str] = []

    for i in tqdm(range(0, len(texts), batch_size), desc=f"{desc_prefix} {src_lang}->{tgt_lang}"):
        batch = [str(x) if pd.notna(x) else "" for x in texts[i : i + batch_size]]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        bos_id = tokenizer.convert_tokens_to_ids(tgt_lang)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                forced_bos_token_id=bos_id,
                max_length=max_length,
            )

        predictions.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))

    return predictions
