from __future__ import annotations

from typing import Any

import pandas as pd


CHANGE_TYPE_ORDER = [
    "orthography",
    "punctuation",
    "direct_order",
    "second_person_pronoun",
    "other_pronoun",
    "verb_form",
    "vocabulary",
    "connector",
    "syntax_simplification",
    "addition",
    "omission",
    "uncertain",
    "other",
]
CHANGE_TYPES = {"none", *CHANGE_TYPE_ORDER}


def parse_change_types(value: Any) -> list[str]:
    if pd.isna(value):
        return []
    return [item.strip() for item in str(value).split("|") if item.strip()]
