from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .adaptation_taxonomy import CHANGE_TYPES
    from .utils import normalize_portuguese_text_with_notes, read_tabular
except ImportError:  # pragma: no cover
    from adaptation_taxonomy import CHANGE_TYPES
    from utils import normalize_portuguese_text_with_notes, read_tabular


PROMPT_VERSION = "adapter-change-taxonomy-v2"
_JSON_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
AUDIT_OUTPUT_COLUMNS = [
    "row_id",
    "source_text_raw",
    "source_text_normalized",
    "normalization_changed",
    "adapted_candidate",
    "adapted_final",
    "changed",
    "change_types",
    "change_summary",
    "similarity_ratio",
    "novel_token_ratio",
    "audit_flags",
    "used_fallback_original",
    "fallback_reason",
    "parsed_response_valid",
    "parse_error",
    "attempts",
]


@dataclass(frozen=True)
class AdaptationConfig:
    input_csv: Path
    normalized_csv: Path
    normalization_audit_csv: Path
    output_csv: Path
    audit_csv: Path
    raw_batches_jsonl: Path
    model: str = "gpt-5-mini"
    batch_size: int = 30
    temperature: float = 1.0
    max_retries: int = 6
    sleep_between_batches: float = 0.2
    max_length_ratio: float = 1.8
    min_length_ratio: float = 0.55
    max_novel_token_ratio: float = 0.65


@dataclass(frozen=True)
class JsonCallResult:
    parsed: list[dict[str, Any]]
    raw_content: str
    requested_model: str
    response_model: str
    attempts: int
    request_timestamp_utc: str
    response_timestamp_utc: str
    system_prompt_hash: str
    user_prompt_hash: str
    parse_error: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def detect_project_root() -> Path:
    cwd = Path.cwd().resolve()
    if (cwd / "data").exists() and (cwd / "src").exists():
        return cwd
    if (cwd.parent / "data").exists() and (cwd.parent / "src").exists():
        return cwd.parent
    return cwd


def default_config(project_root: Path | None = None) -> AdaptationConfig:
    root = project_root or detect_project_root()
    return AdaptationConfig(
        input_csv=root / "data" / "raw" / "oldtupi_portuguese_original.csv",
        normalized_csv=root / "data" / "raw" / "oldtupi_portuguese_normalized.csv",
        normalization_audit_csv=root / "data" / "raw" / "oldtupi_portuguese_normalized_audit.csv",
        output_csv=root / "data" / "raw" / "oldtupi_portuguese_adapted.csv",
        audit_csv=root / "data" / "raw" / "oldtupi_portuguese_adapted_audit.csv",
        raw_batches_jsonl=root / "data" / "raw" / "oldtupi_portuguese_adapted_raw_batches.jsonl",
    )


def create_client() -> Any:
    try:
        from openai import OpenAI
    except Exception as exc:  
        raise RuntimeError(
            "The 'openai' library is not available. Install the dependencies before running the script."
        ) from exc

    import os

    token = os.environ.get("OPENAI_API_KEY")
    if not token:
        raise RuntimeError("Set OPENAI_API_KEY before running adaptation.")
    return OpenAI(api_key=token)


def load_parallel_corpus(path: str | Path) -> pd.DataFrame:
    df = read_tabular(path)
    if df.shape[1] < 2:
        raise ValueError("The corpus must have at least two parallel columns.")
    if list(df.columns[:2]) != ["source_text", "target_text"]:
        df = df.rename(columns={df.columns[0]: "source_text", df.columns[1]: "target_text"})

    rows: list[dict[str, Any]] = []
    for row_id, row in df[["source_text", "target_text"]].iterrows():
        raw_source = "" if pd.isna(row["source_text"]) else str(row["source_text"])
        normalized_source, notes = normalize_portuguese_text_with_notes(raw_source)
        target_text = "" if pd.isna(row["target_text"]) else str(row["target_text"]).strip()
        if not normalized_source or not target_text:
            continue
        rows.append(
            {
                "row_id": int(row_id),
                "source_text_raw": raw_source,
                "source_text_normalized": normalized_source,
                "source_text": normalized_source,
                "target_text": target_text,
                "normalization_changed": raw_source != normalized_source,
                "normalization_notes": " | ".join(notes),
            }
        )
    return pd.DataFrame(rows).reset_index(drop=True)


def save_normalized_outputs(df: pd.DataFrame, config: AdaptationConfig) -> None:
    config.normalized_csv.parent.mkdir(parents=True, exist_ok=True)
    config.normalization_audit_csv.parent.mkdir(parents=True, exist_ok=True)

    normalized_df = df[["source_text_normalized", "target_text"]].rename(
        columns={"source_text_normalized": "source_text"}
    )
    normalized_df.to_csv(config.normalized_csv, index=False)

    audit_df = df[
        [
            "row_id",
            "source_text_raw",
            "source_text_normalized",
            "target_text",
            "normalization_changed",
            "normalization_notes",
        ]
    ]
    audit_df.to_csv(config.normalization_audit_csv, index=False)


def _strip_json_fence(content: str) -> str:
    stripped = (content or "").strip()
    stripped = _JSON_FENCE_RE.sub("", stripped).strip()
    start = stripped.find("[")
    end = stripped.rfind("]")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1].strip()
    return stripped


def _chat_json(
    client: Any,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_retries: int,
) -> JsonCallResult:
    raw_content = ""
    response_model = model
    parse_error = ""
    request_timestamp = utc_now_iso()
    system_hash = sha256_text(system_prompt)
    user_hash = sha256_text(user_prompt)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            response_model = str(getattr(response, "model", model))
            raw_content = response.choices[0].message.content or ""
            parsed = json.loads(_strip_json_fence(raw_content))
            if not isinstance(parsed, list):
                raise ValueError("The response is not a JSON list.")
            return JsonCallResult(
                parsed=parsed,
                raw_content=raw_content,
                requested_model=model,
                response_model=response_model,
                attempts=attempt + 1,
                request_timestamp_utc=request_timestamp,
                response_timestamp_utc=utc_now_iso(),
                system_prompt_hash=system_hash,
                user_prompt_hash=user_hash,
                parse_error="",
            )
        except Exception as exc:  # pragma: no cover
            parse_error = str(exc)
            delay = min(2**attempt, 20) + (0.05 * attempt)
            print(f"[Warning] Model call failed ({attempt + 1}/{max_retries}): {exc}")
            print(f"[Warning] Retrying in {delay:.2f}s.")
            time.sleep(delay)

    return JsonCallResult(
        parsed=[],
        raw_content=raw_content,
        requested_model=model,
        response_model=response_model,
        attempts=max_retries,
        request_timestamp_utc=request_timestamp,
        response_timestamp_utc=utc_now_iso(),
        system_prompt_hash=system_hash,
        user_prompt_hash=user_hash,
        parse_error=parse_error or "unknown_parse_error",
    )


ADAPTATION_SYSTEM_PROMPT = f"""
You are an assistant specialized in adapting historical Portuguese into contemporary Brazilian Portuguese.

Goal:

- Rewrite each text while fully preserving the original meaning.
- Adapt the language into natural contemporary standard Brazilian Portuguese.
- Report, in a structured way, which types of changes were made.

Target register:

- Adapt texts into natural contemporary written Brazilian Portuguese.
- Archaic syntax, obsolete word order, archaic second-person forms, mesoclisis, and obsolete vocabulary should normally be modernized.
- Semantic preservation is more important than literal wording.

Mandatory instructions:

- Do not summarize, interpret, or invent new information.
- Do not omit informative content from the original.
- Do not alter proper names, dates, historical references, or facts.
- Do not change the capitalization of the original text without a clear linguistic reason.
- Do not introduce sentences, examples, or explanations.
- If you are unsure about a historical expression, keep a conservative version.
- Large syntactic restructuring is allowed when necessary to produce natural contemporary Portuguese, provided the original meaning is preserved.
- Write change_summary in English.
- Do not modify the "id" field.
- Return only a valid JSON array, with no additional text, comments, or code blocks.

Representative adaptation examples:

Example:
Original: "Como quando olhais para o céu, vedes sua beleza"
Adapted: "Como quando vocês olham para o céu, veem sua beleza"

Example:
Original: "A terra, todas as nuvens que caem do céu cobri-la-ão"
Adapted: "Todas as nuvens que caem do céu cobrirão a terra"

Example:
Original: "E por outra parte, semelhantemente, o próprio mar será mais terrível do que é seu costume"
Adapted: "Por outro lado, o próprio mar será mais terrível do que costuma ser"

Example:
Original: "Caem às vezes os frutos de suas árvores"
Adapted: "Às vezes os frutos de suas árvores caem"

Example:
Original: "Derrama tuas lágrimas, pranteando a vergonha de tua alma"
Adapted: "Derrame suas lágrimas, lamentando a vergonha de sua alma"

Allowed categories for change_types:
{", ".join(sorted(CHANGE_TYPES))}

Use the categories as follows:

- none: no relevant linguistic adjustment.
- orthography: spelling, accentuation, or historical orthography.
- punctuation: punctuation changes.
- direct_order: shift to direct order or local syntactic reordering.
- second_person_pronoun: changes involving tu/vós/a ti/te/você/vocês.
- other_pronoun: other pronominal changes.
- verb_form: verbal form, mood, placement, or inflection.
- vocabulary: replacement of historical vocabulary with a contemporary equivalent.
- connector: connectors, conjunctions, or discourse particles.
- syntax_simplification: simplification of historical syntax.
- addition: something was made explicit in the adapted text.
- omission: something from the original does not appear literally in the adapted version.
- uncertain: there is meaningful uncertainty in the adaptation.
- other: a relevant change that does not fit the categories above.

Required format:
[
{{
"id": "...",
"adapted_text": "...",
"change_types": ["vocabulary", "direct_order"],
"change_summary": "Short English summary of what was changed.",
"confidence": 0.0
}}
]

Rules:

- If the text does not change, use "change_types": ["none"].
- If there is a possible addition, omission, or uncertainty, include addition, omission, or uncertain.
- confidence must be a number between 0 and 1.
- The output must be JSON ONLY.
  """.strip()


def _build_adaptation_payload(df: pd.DataFrame, start: int, end: int) -> list[dict[str, str]]:
    payload: list[dict[str, str]] = []
    for pos in range(start, end):
        row = df.iloc[pos]
        payload.append({"id": str(row["row_id"]), "text": row["source_text_normalized"]})
    return payload


def request_adaptation_batch(
    client: Any,
    config: AdaptationConfig,
    batch: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], JsonCallResult, str]:
    user_prompt = (
        "Adapt each item in the batch below and classify the changes.\n"
        "Return JSON only, with the fields id, adapted_text, change_types, "
        "change_summary, and confidence.\n"
        "Write change_summary in English.\n\n"
        f"Batch:\n{json.dumps(batch, ensure_ascii=False)}"
    )
    call_result = _chat_json(
        client,
        model=config.model,
        system_prompt=ADAPTATION_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=config.temperature,
        max_retries=config.max_retries,
    )
    return call_result.parsed, call_result, user_prompt


def similarity_ratio(a: str, b: str) -> float:
    import difflib

    return difflib.SequenceMatcher(None, a, b).ratio()


def _strip_accents(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn"
    )


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(_strip_accents(text.lower()))


def novel_token_ratio(original: str, candidate: str) -> float:
    orig_tokens = set(_tokens(original))
    cand_tokens = _tokens(candidate)
    if not cand_tokens:
        return 0.0
    novel = [tok for tok in cand_tokens if tok not in orig_tokens]
    return len(novel) / len(cand_tokens)


def audit_flags(original: str, final_text: str, config: AdaptationConfig) -> list[str]:
    flags: list[str] = []
    ratio = (len(final_text) / len(original)) if original else 1.0
    novelty = novel_token_ratio(original, final_text)

    if not final_text.strip():
        flags.append("empty_final_text")
    if ratio < config.min_length_ratio:
        flags.append("too_short")
    if ratio > config.max_length_ratio:
        flags.append("too_long")
    if novelty > config.max_novel_token_ratio:
        flags.append("high_novel_token_ratio")
    return flags


def normalize_change_types(value: Any, *, changed: bool) -> list[str]:
    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, str):
        raw_items = re.split(r"[,;|]", value)
    else:
        raw_items = []

    normalized: list[str] = []
    synonyms = {
        "direct order": "direct_order",
        "word_order": "direct_order",
        "word order": "direct_order",
        "pronoun change of 2nd person": "second_person_pronoun",
        "2nd_person_pronoun": "second_person_pronoun",
        "second person pronoun": "second_person_pronoun",
        "syntax": "syntax_simplification",
    }
    for item in raw_items:
        key = str(item).strip().lower().replace("-", "_").replace(" ", "_")
        key = synonyms.get(str(item).strip().lower(), key)
        if key in CHANGE_TYPES and key not in normalized:
            normalized.append(key)

    if not normalized:
        normalized = ["other"] if changed else ["none"]
    if not changed and normalized == ["other"]:
        normalized = ["none"]
    if "none" in normalized and len(normalized) > 1:
        normalized = [item for item in normalized if item != "none"]
    return normalized


def build_audit_rows(
    df: pd.DataFrame,
    batch: list[dict[str, str]],
    adapted: list[dict[str, Any]],
    call_result: JsonCallResult,
    config: AdaptationConfig,
) -> list[dict[str, Any]]:
    adaptation_map = {str(item.get("id", "")): item for item in adapted if isinstance(item, dict)}
    df_by_row_id = {str(row["row_id"]): row for _, row in df.iterrows()}

    rows: list[dict[str, Any]] = []
    for item in batch:
        row_id = str(item["id"])
        source_row = df_by_row_id[row_id]
        original = str(source_row["source_text_normalized"])
        record = adaptation_map.get(row_id)

        fallback_reason = ""
        if record is None:
            candidate = ""
            fallback_reason = "missing_item"
        else:
            candidate = normalize_portuguese_text_with_notes(record.get("adapted_text", ""))[0]
            if not candidate:
                fallback_reason = "empty_adapted_text"
        if call_result.parse_error and not adapted:
            fallback_reason = "invalid_batch_response"

        used_fallback = bool(fallback_reason)
        final_text = original if used_fallback else candidate
        changed = final_text != original
        change_types = normalize_change_types(
            record.get("change_types") if record else [],
            changed=changed,
        )
        flags = audit_flags(original, final_text, config)

        rows.append(
            {
                "row_id": int(row_id),
                "source_text_raw": source_row["source_text_raw"],
                "source_text_normalized": original,
                "normalization_changed": bool(source_row["normalization_changed"]),
                "target_text": source_row["target_text"],
                "adapted_candidate": candidate,
                "adapted_final": final_text,
                "changed": changed,
                "change_types": " | ".join(change_types),
                "change_summary": "" if record is None else str(record.get("change_summary", "")),
                "similarity_ratio": round(similarity_ratio(original, final_text), 4),
                "novel_token_ratio": round(novel_token_ratio(original, final_text), 4),
                "audit_flags": " | ".join(dict.fromkeys(flags)),
                "used_fallback_original": used_fallback,
                "fallback_reason": fallback_reason,
                "parsed_response_valid": not bool(call_result.parse_error),
                "parse_error": call_result.parse_error,
                "attempts": call_result.attempts,
            }
        )
    return rows


def append_raw_batch_log(
    config: AdaptationConfig,
    *,
    run_id: str,
    raw_batch_id: str,
    batch_idx: int,
    batch: list[dict[str, str]],
    user_prompt: str,
    call_result: JsonCallResult,
) -> None:
    config.raw_batches_jsonl.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "run_id": run_id,
        "raw_batch_id": raw_batch_id,
        "batch_idx": batch_idx,
        "prompt_version": PROMPT_VERSION,
        "requested_model": call_result.requested_model,
        "response_model": call_result.response_model,
        "temperature": config.temperature,
        "batch_size": config.batch_size,
        "attempts": call_result.attempts,
        "request_timestamp_utc": call_result.request_timestamp_utc,
        "response_timestamp_utc": call_result.response_timestamp_utc,
        "system_prompt_hash": call_result.system_prompt_hash,
        "user_prompt_hash": call_result.user_prompt_hash,
        "system_prompt": ADAPTATION_SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "batch": batch,
        "raw_response": call_result.raw_content,
        "parsed_response": call_result.parsed,
        "parsed_response_valid": not bool(call_result.parse_error),
        "parse_error": call_result.parse_error,
    }
    with config.raw_batches_jsonl.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_adapted_outputs(audit_df: pd.DataFrame, config: AdaptationConfig) -> None:
    config.audit_csv.parent.mkdir(parents=True, exist_ok=True)
    config.output_csv.parent.mkdir(parents=True, exist_ok=True)

    audit_output_df = audit_df.reindex(columns=AUDIT_OUTPUT_COLUMNS)
    audit_output_df.to_csv(config.audit_csv, index=False)
    final_df = audit_df[["adapted_final", "target_text"]].rename(
        columns={"adapted_final": "source_text"}
    )
    final_df.to_csv(config.output_csv, index=False)


def summarize_audit(audit_df: pd.DataFrame) -> pd.DataFrame:
    if audit_df.empty:
        return pd.DataFrame(columns=["metric", "value"])

    rows: list[dict[str, Any]] = [
        {"metric": "rows", "value": len(audit_df)},
        {"metric": "changed_rows", "value": int(audit_df["changed"].sum())},
        {"metric": "changed_rate", "value": round(float(audit_df["changed"].mean()), 4)},
        {
            "metric": "normalization_changed_rows",
            "value": int(audit_df["normalization_changed"].sum()),
        },
        {
            "metric": "normalization_changed_rate",
            "value": round(float(audit_df["normalization_changed"].mean()), 4),
        },
        {
            "metric": "fallback_rate",
            "value": round(float(audit_df["used_fallback_original"].mean()), 4),
        },
        {
            "metric": "invalid_response_rate",
            "value": round(float((~audit_df["parsed_response_valid"]).mean()), 4),
        },
        {"metric": "mean_similarity_ratio", "value": round(float(audit_df["similarity_ratio"].mean()), 4)},
        {"metric": "mean_novel_token_ratio", "value": round(float(audit_df["novel_token_ratio"].mean()), 4)},
    ]

    type_counts: dict[str, int] = {}
    for value in audit_df["change_types"].fillna(""):
        for change_type in [item.strip() for item in str(value).split("|") if item.strip()]:
            type_counts[change_type] = type_counts.get(change_type, 0) + 1
    for change_type, count in sorted(type_counts.items()):
        rows.append({"metric": f"change_type_{change_type}", "value": count})

    return pd.DataFrame(rows)


def normalize_corpus_only(config: AdaptationConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_parallel_corpus(config.input_csv)
    save_normalized_outputs(df, config)
    summary = pd.DataFrame(
        [
            {"metric": "rows", "value": len(df)},
            {"metric": "normalization_changed_rows", "value": int(df["normalization_changed"].sum())},
            {
                "metric": "normalization_changed_rate",
                "value": round(float(df["normalization_changed"].mean()) if len(df) else 0.0, 4),
            },
        ]
    )
    return df, summary


def adapt_corpus(
    config: AdaptationConfig,
    *,
    client: Any | None = None,
    limit_rows: int | None = None,
    persist_outputs: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_parallel_corpus(config.input_csv)
    if limit_rows is not None:
        df = df.head(limit_rows).copy()

    if persist_outputs:
        save_normalized_outputs(df, config)

    active_client = client or create_client()
    audit_rows: list[dict[str, Any]] = []
    run_id = str(uuid.uuid4())

    total = len(df)
    total_batches = (total + config.batch_size - 1) // config.batch_size
    for batch_idx in range(total_batches):
        start = batch_idx * config.batch_size
        end = min(start + config.batch_size, total)
        batch = _build_adaptation_payload(df, start, end)
        raw_batch_id = f"{run_id}:{batch_idx + 1}"

        print(f"[Batch {batch_idx + 1}/{total_batches}] Adapting {len(batch)} rows...")
        adapted, call_result, user_prompt = request_adaptation_batch(active_client, config, batch)

        if persist_outputs:
            append_raw_batch_log(
                config,
                run_id=run_id,
                raw_batch_id=raw_batch_id,
                batch_idx=batch_idx + 1,
                batch=batch,
                user_prompt=user_prompt,
                call_result=call_result,
            )

        audit_rows.extend(
            build_audit_rows(
                df,
                batch,
                adapted,
                call_result,
                config,
            )
        )

        if persist_outputs:
            partial_audit_df = pd.DataFrame(audit_rows).sort_values("row_id").reset_index(drop=True)
            save_adapted_outputs(partial_audit_df, config)
        time.sleep(config.sleep_between_batches)

    audit_df = pd.DataFrame(audit_rows).sort_values("row_id").reset_index(drop=True)
    if persist_outputs:
        save_adapted_outputs(audit_df, config)
    summary_df = summarize_audit(audit_df)
    return audit_df, summary_df


def sample_review(
    config: AdaptationConfig,
    *,
    sample_size: int = 20,
    client: Any | None = None,
) -> pd.DataFrame:
    audit_df, _ = adapt_corpus(
        config,
        client=client,
        limit_rows=sample_size,
        persist_outputs=False,
    )
    columns = [
        "row_id",
        "source_text_raw",
        "source_text_normalized",
        "normalization_changed",
        "adapted_candidate",
        "adapted_final",
        "changed",
        "change_types",
        "change_summary",
        "novel_token_ratio",
        "audit_flags",
        "used_fallback_original",
    ]
    return audit_df[columns].copy()


def print_summary(summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        print("No data to summarize.")
        return
    print("\nSummary:")
    for _, row in summary_df.iterrows():
        print(f"- {row['metric']}: {row['value']}")


def build_config_from_args(args: argparse.Namespace) -> AdaptationConfig:
    base = default_config()
    return AdaptationConfig(
        input_csv=Path(args.input_csv).resolve() if args.input_csv else base.input_csv,
        normalized_csv=Path(args.normalized_csv).resolve() if args.normalized_csv else base.normalized_csv,
        normalization_audit_csv=(
            Path(args.normalization_audit_csv).resolve()
            if args.normalization_audit_csv
            else base.normalization_audit_csv
        ),
        output_csv=Path(args.output_csv).resolve() if args.output_csv else base.output_csv,
        audit_csv=Path(args.audit_csv).resolve() if args.audit_csv else base.audit_csv,
        raw_batches_jsonl=(
            Path(args.raw_batches_jsonl).resolve()
            if args.raw_batches_jsonl
            else base.raw_batches_jsonl
        ),
        model=args.model,
        batch_size=args.batch_size,
        temperature=args.temperature,
        max_retries=args.max_retries,
        sleep_between_batches=args.sleep_between_batches,
        max_length_ratio=args.max_length_ratio,
        min_length_ratio=args.min_length_ratio,
        max_novel_token_ratio=args.max_novel_token_ratio,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize and adapt the Portuguese side of the corpus with descriptive auditing."
    )
    parser.add_argument("--mode", choices=["normalize", "sample", "full"], default="sample")
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--input-csv")
    parser.add_argument("--normalized-csv")
    parser.add_argument("--normalization-audit-csv")
    parser.add_argument("--output-csv")
    parser.add_argument("--audit-csv")
    parser.add_argument("--raw-batches-jsonl")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--sleep-between-batches", type=float, default=0.2)
    parser.add_argument("--max-length-ratio", type=float, default=1.8)
    parser.add_argument("--min-length-ratio", type=float, default=0.55)
    parser.add_argument("--max-novel-token-ratio", type=float, default=0.65)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = build_config_from_args(args)

    if args.mode == "normalize":
        normalized_df, summary_df = normalize_corpus_only(config)
        print_summary(summary_df)
        print(f"\nNormalized CSV saved to: {config.normalized_csv}")
        print(f"Normalization audit saved to: {config.normalization_audit_csv}")
        print(f"Processed rows: {len(normalized_df)}")
        return 0

    if args.mode == "sample":
        review_df = sample_review(config, sample_size=args.sample_size)
        print(review_df.to_string(index=False))
        return 0

    audit_df, summary_df = adapt_corpus(config)
    print_summary(summary_df)
    print(f"\nNormalized CSV saved to: {config.normalized_csv}")
    print(f"Normalization audit saved to: {config.normalization_audit_csv}")
    print(f"Adapted CSV saved to: {config.output_csv}")
    print(f"Adaptation audit saved to: {config.audit_csv}")
    print(f"Raw batch log saved to: {config.raw_batches_jsonl}")
    print(f"Processed rows: {len(audit_df)}")
    print("Rerun data preparation to rebuild the splits after changing the corpus.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
