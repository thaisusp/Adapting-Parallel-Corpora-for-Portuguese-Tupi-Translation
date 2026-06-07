# Low-Resource Machine Translation: Portuguese and Old Tupi

This repository contains experiments for Portuguese/Old Tupi machine translation using a methodologically symmetric evaluation protocol. The main goal is to test whether adapting the Portuguese side of a historical parallel corpus into contemporary Portuguese improves translation quality.

## Scope

- Base model: `facebook/nllb-200-distilled-600M`
- Old Tupi proxy language code: `grn_Latn`
- Translation directions:
  - `pt_to_tupi`
  - `tupi_to_pt`
- Training corpora:
  - `historical`: technically normalized historical Portuguese source text
  - `adapted`: contemporary Portuguese adaptation of the same aligned examples
- Evaluation corpora:
  - `historical`
  - `adapted`

Zero-shot and fine-tuned systems are always compared within the same translation direction and evaluation corpus.

## Project Structure

- `main.py`: pipeline orchestrator.
- `src/adapt_pt_corpus.py`: Portuguese-side normalization and auditable adaptation.
- `src/adaptation_taxonomy.py`: adaptation change-type taxonomy.
- `src/corpus_stats.py`: corpus alignment and adaptation summary reports.
- `src/eval_stats.py`: aggregation of run-level metrics.
- `src/experiment.py`: shared experiment constants and translation helpers.
- `src/predict.py`: fine-tuned inference and evaluation.
- `src/plots.py`: final comparison plots.
- `src/qualitative_ranking.py`: sentence-level ranking for qualitative analysis.
- `src/train.py`: fine-tuning.
- `src/utils.py`: shared I/O, normalization, split, and metric utilities.
- `src/zero_shot.py`: zero-shot evaluation.
- `data/raw/`: source, normalized, adapted, and audit corpora.
- `data/processed/`: train/validation/test splits.
- `results/metrics/`: run-level and aggregate metric CSVs.
- `results/predictions/`: generated predictions.
- `results/plots/`: generated figures.
- `results/qualitative/`: qualitative best/worst adaptation rankings.

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Corpus Adaptation

The Portuguese side of the corpus can be adapted with:

- deterministic technical normalization;
- LLM-based adaptation into contemporary Brazilian Portuguese;
- sentence-level change taxonomy;
- audit tables with deterministic change metrics;
- raw prompt/response batch logs for reproducibility;
- fallback to the normalized original only for technical failures, such as missing or empty model output.

Input:

- `data/raw/oldtupi_portuguese_original.csv`

Main outputs:

- `data/raw/oldtupi_portuguese_normalized.csv`
- `data/raw/oldtupi_portuguese_normalized_audit.csv`
- `data/raw/oldtupi_portuguese_adapted.csv`
- `data/raw/oldtupi_portuguese_adapted_audit.csv`
- `data/raw/oldtupi_portuguese_adapted_raw_batches.jsonl`

The normalized corpus is used for the historical splits. The adapted corpus is used for the adapted splits. After changing either corpus, regenerate the processed splits.

Normalize only, without API calls:

```bash
./venv/bin/python -m src.adapt_pt_corpus --mode normalize
```

Sample review without writing final adaptation files:

```bash
export OPENAI_API_KEY="your_api_key_here"
./venv/bin/python -m src.adapt_pt_corpus --mode sample --sample-size 20
```

Full adaptation:

```bash
export OPENAI_API_KEY="your_api_key_here"
./venv/bin/python -m src.adapt_pt_corpus --mode full
```

Useful optional parameters:

```bash
./venv/bin/python -m src.adapt_pt_corpus \
  --mode full \
  --batch-size 30 \
  --model gpt-5-mini
```

Adaptation CLI arguments:

- `--mode {normalize,sample,full}`: execution mode.
- `--sample-size <int>`: number of rows used by `sample` mode.
- `--input-csv <path>`: source parallel corpus. Defaults to `data/raw/oldtupi_portuguese_original.csv`.
- `--normalized-csv <path>`: normalized corpus output. Defaults to `data/raw/oldtupi_portuguese_normalized.csv`.
- `--normalization-audit-csv <path>`: technical normalization audit output. Defaults to `data/raw/oldtupi_portuguese_normalized_audit.csv`.
- `--output-csv <path>`: final adapted corpus output. Defaults to `data/raw/oldtupi_portuguese_adapted.csv`.
- `--audit-csv <path>`: adaptation audit output. Defaults to `data/raw/oldtupi_portuguese_adapted_audit.csv`.
- `--raw-batches-jsonl <path>`: raw prompt/response batch log. Defaults to `data/raw/oldtupi_portuguese_adapted_raw_batches.jsonl`.
- `--model <model>`: OpenAI model used for adaptation. Defaults to `gpt-5-mini`.
- `--batch-size <int>`: number of corpus rows sent per model call. Defaults to `30`.
- `--temperature <float>`: model sampling temperature. Defaults to `1.0`.
- `--max-retries <int>`: retry limit for model calls and JSON parsing. Defaults to `6`.
- `--sleep-between-batches <float>`: pause between batches in seconds. Defaults to `0.2`.
- `--max-length-ratio <float>`: audit flag threshold for overly long outputs. Defaults to `1.8`.
- `--min-length-ratio <float>`: audit flag threshold for overly short outputs. Defaults to `0.55`.
- `--max-novel-token-ratio <float>`: audit flag threshold for high lexical novelty. Defaults to `0.65`.

## Main Pipeline Arguments

`main.py` supports:

- `--prepare-data`
- `--zero-shot`
- `--train`
- `--predict`
- `--corpus-stats`
- `--allow-downloads`
- `--train-dataset {adapted,historical}`
- `--eval-dataset {all,adapted,historical}`
- `--direction {pt_to_tupi,tupi_to_pt}`
- `--model-id <model>`
- `--epochs <float>`
- `--learning-rate <float>`
- `--batch-size <int>`
- `--inference-batch-size <int>`
- `--gradient-accumulation-steps <int>`
- `--max-length <int>`
- `--seed <int>`
- `--force`

## Runbook

### 1. Prepare Data

```bash
./venv/bin/python main.py --prepare-data --seed 42 --force
```

This creates deterministic train/validation/test splits and validates that the historical and adapted splits use the same Old Tupi target order.

Corpus reports can also be regenerated directly:

```bash
./venv/bin/python main.py --corpus-stats
```

### 2. Fine-Tune Portuguese to Old Tupi

```bash
./venv/bin/python main.py --train --allow-downloads --train-dataset historical --direction pt_to_tupi --seed 42
./venv/bin/python main.py --train --allow-downloads --train-dataset adapted --direction pt_to_tupi --seed 42
```

### 3. Fine-Tune Old Tupi to Portuguese

```bash
./venv/bin/python main.py --train --allow-downloads --train-dataset historical --direction tupi_to_pt --seed 42
./venv/bin/python main.py --train --allow-downloads --train-dataset adapted --direction tupi_to_pt --seed 42
```

### 4. Evaluate Zero-Shot Systems

```bash
./venv/bin/python main.py --zero-shot --allow-downloads --direction pt_to_tupi --eval-dataset all --seed 42
./venv/bin/python main.py --zero-shot --allow-downloads --direction tupi_to_pt --eval-dataset all --seed 42
```

### 5. Evaluate Fine-Tuned Systems

```bash
./venv/bin/python main.py --predict --allow-downloads --train-dataset historical --direction pt_to_tupi --eval-dataset all --seed 42
./venv/bin/python main.py --predict --allow-downloads --train-dataset adapted --direction pt_to_tupi --eval-dataset all --seed 42
./venv/bin/python main.py --predict --allow-downloads --train-dataset historical --direction tupi_to_pt --eval-dataset all --seed 42
./venv/bin/python main.py --predict --allow-downloads --train-dataset adapted --direction tupi_to_pt --eval-dataset all --seed 42
```

### 6. Generate Plots

```bash
./venv/bin/python -m src.plots
```

Plot generation validates that the complete metric matrix exists before writing figures: zero-shot results for both directions and evaluation corpora, plus fine-tuned results for both directions, both training corpora, both evaluation corpora, and all reported metrics.

### 7. Generate Qualitative Rankings

```bash
./venv/bin/python -m src.qualitative_ranking
```

This creates:

- `results/qualitative/best_adaptation_top_20.csv`
- `results/qualitative/worst_adaptation_top_20.csv`

Each file contains 20 sentence-level examples per metric, with the expected reference and both fine-tuned predictions side by side.

## Output Convention

Predictions use explicit names:

- `zeroshot_{direction}_{dataset}_seed{seed}.csv`
- `finetuned_{direction}_{dataset}_seed{seed}_train_{train_dataset}.csv`

Metric outputs are split into:

1. run-level files (`*_runs.csv`) with one row per evaluation dataset;
2. aggregate files (`*_aggregate.csv`) with one metric row per model, direction, and evaluation dataset.

Reported MT metrics:

- `BLEU`
- `TER`
- `chrF1`
- `chrF3`

Higher is better for BLEU and chrF. Lower is better for TER.

## Methodological Notes

- Zero-shot evaluation uses NLLB with `grn_Latn` as the proxy source/target code for Old Tupi.
- Fine-tuned models are stored per `(direction, train_dataset, seed)`.
- The current experiment uses a fixed split and a single training seed (`42`).
- For valid comparisons, compare systems only within the same translation direction and evaluation corpus.
- Historical splits use technically normalized Portuguese, not the raw spreadsheet text.
- Adapted splits use the adapted Portuguese corpus, normalized again during split preparation with the same deterministic normalization.
- The raw adaptation JSONL includes full prompts and model responses; review it before publishing if prompt content or model outputs should remain private.

## GitHub Hygiene

The repository is configured to ignore virtual environments, model checkpoints, model weights, local system files, legacy scratch data, and redundant plot archives. Before publishing, confirm that no large artifacts are staged:

```bash
git status --short
git check-ignore -v venv results/models old
```
