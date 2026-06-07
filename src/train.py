from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

from .experiment import (
    DIRECTION_CHOICES,
    LANG_PT,
    LANG_TUPI_PROXY,
    MODEL_ID,
    TRAIN_DATASET_CHOICES,
    direction_languages,
    model_run_dir,
    normalize_direction,
)
from .utils import (
    ProjectPaths,
    clear_memory,
    clean_text,
    get_device,
    read_tabular,
)


def load_parallel_split(paths: ProjectPaths, filename: str) -> pd.DataFrame:
    path = paths.data_processed / filename
    df = read_tabular(path)

    cols = df.columns
    if len(cols) < 2:
        raise ValueError(f"Parallel split requires at least 2 columns: {path}")

    df = df.rename(columns={cols[0]: "pt", cols[1]: "tupi"})
    df["pt"] = df["pt"].apply(clean_text).astype(str)
    df["tupi"] = df["tupi"].apply(clean_text).astype(str)
    df = df[(df["pt"].str.len() > 0) & (df["tupi"].str.len() > 0)].reset_index(drop=True)
    return df


def preprocess_function(
    examples,
    tokenizer,
    *,
    source_col: str,
    target_col: str,
    src_lang: str,
    tgt_lang: str,
    max_length: int,
):
    inputs = examples[source_col]
    targets = examples[target_col]

    tokenizer.src_lang = src_lang
    model_inputs = tokenizer(inputs, max_length=max_length, truncation=True)

    tokenizer.src_lang = tgt_lang
    labels_batch = tokenizer(targets, max_length=max_length, truncation=True)
    model_inputs["labels"] = labels_batch["input_ids"]

    return model_inputs


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune NLLB for Portuguese/Tupi translation.")
    parser.add_argument("--train-dataset", choices=TRAIN_DATASET_CHOICES, default="adapted")
    parser.add_argument("--direction", choices=DIRECTION_CHOICES, default="pt_to_tupi")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--epochs", type=float, default=5)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(
    train_dataset: str = "adapted",
    direction: str = "pt_to_tupi",
    model_id: str = MODEL_ID,
    epochs: float = 5,
    learning_rate: float = 5e-5,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 2,
    max_length: int = 128,
    seed: int = 42,
):
    direction = normalize_direction(direction)
    clear_memory()
    set_seed(seed)

    paths = ProjectPaths.from_file(__file__)
    output_dir = model_run_dir(paths, train_dataset=train_dataset, direction=direction, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Running on: {device.upper()}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        src_lang=LANG_PT if direction == "pt_to_tupi" else LANG_TUPI_PROXY,
        use_fast=False,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)

    df_train = load_parallel_split(paths, f"train_{train_dataset}.csv")
    df_val = load_parallel_split(paths, f"val_{train_dataset}.csv")

    if direction == "pt_to_tupi":
        source_col, target_col = "pt", "tupi"
    else:
        source_col, target_col = "tupi", "pt"
    src_lang, tgt_lang = direction_languages(direction)

    raw_datasets = DatasetDict(
        {
            "train": Dataset.from_pandas(df_train),
            "validation": Dataset.from_pandas(df_val),
        }
    )

    preprocess = partial(
        preprocess_function,
        tokenizer=tokenizer,
        source_col=source_col,
        target_col=target_col,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        max_length=max_length,
    )

    tokenized_datasets = raw_datasets.map(
        preprocess,
        batched=True,
        load_from_cache_file=False,
        remove_columns=raw_datasets["train"].column_names,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )

    args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=epochs,
        fp16=(device == "cuda"),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=50,
        dataloader_num_workers=0,
        report_to="none",
        seed=seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print(
        f"Training fine-tuned model | train_dataset={train_dataset} | direction={direction} | seed={seed}"
    )
    trainer.train()

    final_path = Path(output_dir) / "final"
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)

    print(f"Model saved at: {final_path}")


if __name__ == "__main__":
    cli_args = _parse_args()
    main(
        train_dataset=cli_args.train_dataset,
        direction=cli_args.direction,
        model_id=cli_args.model_id,
        epochs=cli_args.epochs,
        learning_rate=cli_args.learning_rate,
        batch_size=cli_args.batch_size,
        gradient_accumulation_steps=cli_args.gradient_accumulation_steps,
        max_length=cli_args.max_length,
        seed=cli_args.seed,
    )
