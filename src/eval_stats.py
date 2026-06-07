from __future__ import annotations

import pandas as pd


def aggregate_run_metrics(
    runs_df: pd.DataFrame,
    *,
    group_cols: list[str],
    metric_cols: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    grouped = runs_df.groupby(group_cols, dropna=False, sort=True)
    for group_values, group_df in grouped:
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        if len(group_df) != 1:
            group_text = ", ".join(f"{col}={value}" for col, value in zip(group_cols, group_values))
            raise ValueError(
                "Expected exactly one run per aggregate group. "
                f"Found {len(group_df)} rows for {group_text}."
            )

        group_dict = dict(zip(group_cols, group_values))
        run = group_df.iloc[0]

        for metric in metric_cols:
            rows.append(
                {
                    **group_dict,
                    "Metric": metric,
                    "Mean": round(float(run[metric]), 4),
                }
            )

    return pd.DataFrame(rows)
