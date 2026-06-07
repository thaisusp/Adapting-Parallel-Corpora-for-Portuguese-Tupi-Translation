from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

METRICS = ["BLEU", "TER", "chrF1", "chrF3"]
DIRECTION_ORDER = ["pt_to_tupi", "tupi_to_pt"]
DATASET_ORDER = ["historical", "adapted"]
TRAIN_DATASET_ORDER = ["historical", "adapted"]
MODEL_LABEL_ORDER = ["Zero-shot", "Fine-tuned (historical)", "Fine-tuned (adapted)"]
DATASET_LABELS = {
    "historical": "Historical",
    "adapted": "Adapted",
}
DIRECTION_LABELS = {
    "pt_to_tupi": "Portuguese -> Tupi",
    "tupi_to_pt": "Tupi -> Portuguese",
}
MODEL_COLORS = {
    "Zero-shot": "#4c78a8",
    "Fine-tuned (historical)": "#59a14f",
    "Fine-tuned (adapted)": "#f28e2b",
}
MARKERS = {
    "historical": "o",
    "adapted": "s",
}
TRAIN_DATASET_COLORS = {
    "adapted": ["#8B0000", "#DC143C", "#CD5C5C", "#F08080"],
    "historical": ["#00008B", "#0000CD", "#4169E1", "#6495ED"],
    "zeroshot": ["#4F4F4F", "#808080", "#A9A9A9", "#D3D3D3"],
}


def _project_root() -> Path:
    here = Path(__file__).resolve()
    return here.parent.parent


def load_aggregate_metrics(root: Path) -> pd.DataFrame:
    metrics_root = root / "results" / "metrics"
    files = sorted(metrics_root.glob("**/*_aggregate.csv"))

    rows: list[pd.DataFrame] = []
    for path in files:
        if path.name == "comparison_aggregate.csv":
            continue

        df = pd.read_csv(path)
        if df.empty:
            continue

        if "ModelType" not in df.columns:
            continue
        if "Direction" not in df.columns or "Dataset" not in df.columns:
            continue

        if "TrainDataset" not in df.columns:
            df["TrainDataset"] = "-"

        df["SourceFile"] = str(path.relative_to(root))
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)

    out["Direction"] = out["Direction"].astype(str).str.lower().str.replace("-", "_", regex=False)
    out["Dataset"] = out["Dataset"].astype(str).str.lower()
    out["ModelType"] = out["ModelType"].astype(str).str.lower()
    out["TrainDataset"] = out["TrainDataset"].astype(str).str.lower()
    out["Metric"] = out["Metric"].astype(str)
    out["Mean"] = pd.to_numeric(out["Mean"], errors="coerce")

    out = out[out["Metric"].isin(METRICS)].copy()

    def label_model(row: pd.Series) -> str:
        if row["ModelType"] == "zeroshot":
            return "Zero-shot"
        return f"Fine-tuned ({row['TrainDataset']})"

    out["ModelLabel"] = out.apply(label_model, axis=1)
    return out


def validate_expected_metric_matrix(df: pd.DataFrame) -> None:
    missing: list[str] = []
    duplicate_keys: list[str] = []

    key_cols = ["ModelType", "TrainDataset", "Direction", "Dataset", "Metric"]
    duplicates = df[df.duplicated(key_cols, keep=False)]
    if not duplicates.empty:
        for row in duplicates[key_cols].drop_duplicates().itertuples(index=False):
            duplicate_keys.append(
                f"model={row.ModelType}, train_dataset={row.TrainDataset}, "
                f"direction={row.Direction}, eval_dataset={row.Dataset}, metric={row.Metric}"
            )

    expected: list[dict[str, str]] = []
    for direction in DIRECTION_ORDER:
        for dataset in DATASET_ORDER:
            for metric in METRICS:
                expected.append(
                    {
                        "ModelType": "zeroshot",
                        "TrainDataset": "-",
                        "Direction": direction,
                        "Dataset": dataset,
                        "Metric": metric,
                    }
                )
        for train_dataset in TRAIN_DATASET_ORDER:
            for dataset in DATASET_ORDER:
                for metric in METRICS:
                    expected.append(
                        {
                            "ModelType": "finetuned",
                            "TrainDataset": train_dataset,
                            "Direction": direction,
                            "Dataset": dataset,
                            "Metric": metric,
                        }
                    )

    for item in expected:
        mask = pd.Series(True, index=df.index)
        for col, value in item.items():
            mask &= df[col].eq(value)
        if not bool(mask.any()):
            missing.append(
                f"model={item['ModelType']}, train_dataset={item['TrainDataset']}, "
                f"direction={item['Direction']}, eval_dataset={item['Dataset']}, metric={item['Metric']}"
            )

    problems = []
    if missing:
        problems.append("Missing metric rows:\n- " + "\n- ".join(missing))
    if duplicate_keys:
        problems.append("Duplicate metric rows:\n- " + "\n- ".join(duplicate_keys))

    if problems:
        raise ValueError(
            "Cannot generate plots because the metric matrix is incomplete.\n\n"
            + "\n\n".join(problems)
        )


def _ordered_model_labels(df: pd.DataFrame) -> list[str]:
    labels = df["ModelLabel"].dropna().unique().tolist()
    ordered = [label for label in MODEL_LABEL_ORDER if label in labels]
    ordered.extend(sorted(label for label in labels if label not in ordered))
    return ordered


def _ordered_directions(df: pd.DataFrame) -> list[str]:
    directions = df["Direction"].dropna().unique().tolist()
    ordered = [direction for direction in DIRECTION_ORDER if direction in directions]
    ordered.extend(sorted(direction for direction in directions if direction not in ordered))
    return ordered


def _ordered_datasets(df: pd.DataFrame) -> list[str]:
    datasets = df["Dataset"].dropna().unique().tolist()
    ordered = [dataset for dataset in DATASET_ORDER if dataset in datasets]
    ordered.extend(sorted(dataset for dataset in datasets if dataset not in ordered))
    return ordered


def _display_direction(direction: str) -> str:
    return DIRECTION_LABELS.get(direction, direction.replace("_", " "))



def _format_metric_value(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.2f}"


def _safe_metric_name(metric: str) -> str:
    return metric.lower().replace(" ", "_")


def _heatmap_row_label(row: pd.Series) -> str:
    if row["ModelType"] == "zeroshot":
        return "Zero-shot"
    return f"Train: {DATASET_LABELS.get(row['TrainDataset'], row['TrainDataset'])}"


def _heatmap_row_order(df: pd.DataFrame) -> list[str]:
    labels = df.apply(_heatmap_row_label, axis=1).dropna().unique().tolist()
    preferred = ["Zero-shot"] + [f"Train: {DATASET_LABELS[item]}" for item in TRAIN_DATASET_ORDER]
    ordered = [label for label in preferred if label in labels]
    ordered.extend(sorted(label for label in labels if label not in ordered))
    return ordered


def _clear_generated_plot_files(root: Path) -> None:
    plots_dir = root / "results" / "plots"
    if not plots_dir.exists():
        return

    patterns = [
        "paired_*.png",
        "delta_*.png",
        "comparison_*.png",
        "heatmap_train_eval_*.png",
        "heatmap_combined_*.png",
        "summary_table*.png",
        "scatter_bleu_*.png",
        "delta_finetuned_*.png",
        "comparative_performance_analysis.png",
    ]
    for pattern in patterns:
        for path in plots_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def _metric_plot_order(df: pd.DataFrame) -> list[str]:
    metrics = [metric for metric in METRICS if metric in df["Metric"].unique()]
    if "TER" in metrics:
        metrics.remove("TER")
        metrics.append("TER")
    return metrics


def _configuration_palette(configurations: list[str]) -> dict[str, str]:
    palette: dict[str, str] = {}
    color_indexes = {"adapted": 0, "historical": 0, "zeroshot": 0}

    for configuration in sorted(configurations):
        name = configuration.lower()
        if "adapted" in name:
            color_group = "adapted"
        elif "historical" in name:
            color_group = "historical"
        else:
            color_group = "zeroshot"

        colors = TRAIN_DATASET_COLORS[color_group]
        palette[configuration] = colors[color_indexes[color_group] % len(colors)]
        color_indexes[color_group] += 1

    return palette


def save_comparative_performance_analysis(df: pd.DataFrame, root: Path) -> list[Path]:
    plots_dir = root / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df_plot = df.copy()
    df_plot["Configuration"] = df_plot["ModelLabel"] + " (" + df_plot["Direction"] + ")"
    metrics = _metric_plot_order(df_plot)
    if not metrics:
        return []

    palette = _configuration_palette(df_plot["Configuration"].dropna().unique().tolist())
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.8 * len(metrics), 6.5), sharey=False)
    if len(metrics) == 1:
        axes = [axes]

    fig.suptitle("Comparative Performance Analysis (Aggregated Data)", fontsize=16, y=1.02, fontweight="bold")

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        df_metric = df_plot[df_plot["Metric"].eq(metric)]
        df_mean = df_metric.pivot_table(
            index="Dataset",
            columns="Configuration",
            values="Mean",
            aggfunc="first",
        ).reindex(index=_ordered_datasets(df_metric))

        df_mean.index = [f"Evaluated on:\n{DATASET_LABELS.get(str(item), str(item).capitalize())}" for item in df_mean.index]
        colors = [palette.get(column, "#808080") for column in df_mean.columns]

        df_mean.plot(
            kind="bar",
            ax=ax,
            color=colors,
            width=0.8,
            alpha=0.9,
            edgecolor="black",
            linewidth=0.6,
        )

        ax.set_title(f"Metric: {metric}", fontsize=13, fontweight="bold")
        ax.set_xlabel("")
        ylabel = "Mean score - lower is better" if metric == "TER" else "Mean score"
        ax.set_ylabel(ylabel, fontsize=11)
        ax.tick_params(axis="x", rotation=0, labelsize=11)
        ax.grid(axis="y", alpha=0.25)
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Models and directions",
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.05),
    )

    fig.subplots_adjust(bottom=0.35)
    fig.tight_layout()
    out_file = plots_dir / "comparative_performance_analysis.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [out_file]





def _normalized_metric_values(values: np.ndarray, metric: str) -> np.ndarray:
    scores = -values if metric == "TER" else values
    valid = scores[np.isfinite(scores)]
    if valid.size == 0:
        return np.full_like(scores, np.nan, dtype=float)
    low = float(valid.min())
    high = float(valid.max())
    if np.isclose(low, high):
        return np.where(np.isfinite(scores), 0.5, np.nan)
    return (scores - low) / (high - low)


def save_combined_heatmap_plots(df: pd.DataFrame, root: Path) -> list[Path]:
    plots_dir = root / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []

    for direction in _ordered_directions(df):
        df_dir = df[df["Direction"] == direction].copy()
        datasets = _ordered_datasets(df_dir)

        df_dir = df_dir.copy()
        df_dir["HeatmapRow"] = df_dir.apply(_heatmap_row_label, axis=1)
        row_order = _heatmap_row_order(df_dir)
        columns: list[tuple[str, str]] = [(metric, dataset) for metric in METRICS for dataset in datasets]
        values = np.full((len(row_order), len(columns)), np.nan)
        normalized = np.full_like(values, np.nan, dtype=float)

        for metric_idx, metric in enumerate(METRICS):
            pivot = df_dir[df_dir["Metric"] == metric].pivot_table(
                index="HeatmapRow",
                columns="Dataset",
                values="Mean",
                aggfunc="first",
            ).reindex(index=row_order, columns=datasets)
            block = pivot.to_numpy(dtype=float)
            start_col = metric_idx * len(datasets)
            end_col = start_col + len(datasets)
            values[:, start_col:end_col] = block
            normalized[:, start_col:end_col] = _normalized_metric_values(block, metric)

        masked = np.ma.masked_invalid(normalized)
        cmap = plt.colormaps["YlGnBu"].copy()
        cmap.set_bad(color="#f2f2f2")

        fig, ax = plt.subplots(figsize=(12, max(4.8, 0.75 * len(row_order) + 2.3)))
        image = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=0, vmax=1)

        ax.set_title(
            f"Combined heatmap by metric and evaluation corpus | {_display_direction(direction)}\n"
            "Color is normalized within each metric; darker means better",
            fontsize=13,
            weight="bold",
            pad=14,
        )
        ax.set_xticks(np.arange(len(columns)))
        ax.set_xticklabels(
            [f"{metric}\n{DATASET_LABELS.get(dataset, dataset)}" for metric, dataset in columns],
            rotation=0,
            ha="center",
        )
        ax.set_yticks(np.arange(len(row_order)))
        ax.set_yticklabels(row_order)
        ax.set_xlabel("Metric / evaluation corpus")
        ax.set_ylabel("Training setup")

        for metric_idx in range(1, len(METRICS)):
            ax.axvline(metric_idx * len(datasets) - 0.5, color="white", linewidth=3)

        for row_idx in range(values.shape[0]):
            for col_idx in range(values.shape[1]):
                value = values[row_idx, col_idx]
                if np.isfinite(value):
                    ax.text(
                        col_idx,
                        row_idx,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="black",
                    )

        fig.colorbar(image, ax=ax, shrink=0.78, label="Normalized quality within metric")
        fig.tight_layout()
        out_file = plots_dir / f"heatmap_combined_{direction}.png"
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_file)

    return saved


def _summary_body_rows(df: pd.DataFrame) -> list[list[object]]:
    rows: list[list[object]] = []
    model_labels = [label for label in MODEL_LABEL_ORDER if label in df["ModelLabel"].unique()]

    for direction in _ordered_directions(df):
        for dataset in _ordered_datasets(df[df["Direction"] == direction]):
            for metric in METRICS:
                subset = df[
                    df["Direction"].eq(direction)
                    & df["Dataset"].eq(dataset)
                    & df["Metric"].eq(metric)
                ]
                pivot = subset.pivot_table(
                    index="Metric",
                    columns="ModelLabel",
                    values="Mean",
                    aggfunc="first",
                )
                values = []
                for model_label in model_labels:
                    if metric in pivot.index and model_label in pivot.columns:
                        values.append(pivot.at[metric, model_label])
                    else:
                        values.append(np.nan)
                rows.append([_display_direction(direction), DATASET_LABELS.get(dataset, dataset), metric, *values])
    return rows


def _summary_cell_colors(rows: list[list[object]], metric_col: int, first_value_col: int) -> list[list[str]]:
    body_colors: list[list[str]] = []
    for row in rows:
        colors = ["#f7f7f7"] * first_value_col
        values = [float(value) if pd.notna(value) else np.nan for value in row[first_value_col:]]
        valid_positions = [idx for idx, value in enumerate(values) if np.isfinite(value)]
        value_colors = ["#ffffff"] * len(values)

        if valid_positions:
            reverse = row[metric_col] != "TER"
            ranked_positions = sorted(valid_positions, key=lambda idx: values[idx], reverse=reverse)
            rank_colors = ["#d9ead3", "#fff2cc", "#f4cccc"]
            for rank, value_idx in enumerate(ranked_positions):
                value_colors[value_idx] = rank_colors[min(rank, len(rank_colors) - 1)]

        colors.extend(value_colors)
        body_colors.append(colors)
    return body_colors


def save_summary_table_plot(df: pd.DataFrame, root: Path) -> list[Path]:
    plots_dir = root / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    model_labels = [label for label in MODEL_LABEL_ORDER if label in df["ModelLabel"].unique()]
    columns = ["Direction", "Eval corpus", "Metric", *model_labels]
    raw_rows = _summary_body_rows(df)
    display_rows = [
        [*row[:3], *[_format_metric_value(value) for value in row[3:]]]
        for row in raw_rows
    ]
    cell_colors = _summary_cell_colors(raw_rows, metric_col=2, first_value_col=3)

    fig_height = max(1, 0.42 * len(display_rows))
    fig, ax = plt.subplots(figsize=(11, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=display_rows,
        colLabels=columns,
        cellColours=cell_colors,
        cellLoc="center",
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.35)

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d0d0")
        cell.set_linewidth(0.5)
        if row_idx == 0:
            cell.set_facecolor("#BBBBBB")
            cell.set_text_props(color="black", weight="bold")
        elif col_idx < 3:
            cell.set_text_props(weight="bold" if col_idx == 2 else "normal")


    fig.tight_layout()
    out_file = plots_dir / "summary_table_conditional.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [out_file]


def save_finetuned_delta_heatmap(df: pd.DataFrame, root: Path) -> list[Path]:
    plots_dir = root / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fine_tuned = df[df["ModelType"].eq("finetuned")].copy()
    rows: list[str] = []
    values = np.full((len(_ordered_directions(df)) * len(DATASET_ORDER), len(METRICS)), np.nan)
    quality_delta = np.full_like(values, np.nan, dtype=float)

    row_idx = 0
    for direction in _ordered_directions(df):
        for dataset in _ordered_datasets(df[df["Direction"] == direction]):
            rows.append(f"{_display_direction(direction)}\n{DATASET_LABELS.get(dataset, dataset)}")
            subset = fine_tuned[
                fine_tuned["Direction"].eq(direction)
                & fine_tuned["Dataset"].eq(dataset)
                & fine_tuned["TrainDataset"].isin(TRAIN_DATASET_ORDER)
            ]
            pivot = subset.pivot_table(
                index="TrainDataset",
                columns="Metric",
                values="Mean",
                aggfunc="first",
            )
            for col_idx, metric in enumerate(METRICS):
                if all(item in pivot.index for item in TRAIN_DATASET_ORDER) and metric in pivot.columns:
                    delta = float(pivot.at["adapted", metric]) - float(pivot.at["historical", metric])
                    values[row_idx, col_idx] = delta
                    quality_delta[row_idx, col_idx] = -delta if metric == "TER" else delta
            row_idx += 1

    values = values[: len(rows), :]
    quality_delta = quality_delta[: len(rows), :]
    max_abs = np.nanmax(np.abs(quality_delta))
    if not np.isfinite(max_abs) or np.isclose(max_abs, 0):
        max_abs = 1.0

    cmap = plt.colormaps["RdYlGn"].copy()
    cmap.set_bad(color="#f2f2f2")
    norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    fig, ax = plt.subplots(figsize=(8.6, max(4.8, 0.72 * len(rows) + 2.2)))
    image = ax.imshow(np.ma.masked_invalid(quality_delta), cmap=cmap, norm=norm, aspect="auto")

    ax.set_title(
        "Delta heatmap: Fine-tuned (adapted) - Fine-tuned (historical)\n"
        "Green means the adapted-trained model is better; numbers show raw deltas",
        fontsize=13,
        weight="bold",
        pad=14,
    )
    ax.set_xticks(np.arange(len(METRICS)))
    ax.set_xticklabels(METRICS)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Direction / evaluation corpus")

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if np.isfinite(value):
                ax.text(
                    col,
                    row,
                    f"{value:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="black",
                )

    fig.colorbar(image, ax=ax, shrink=0.78, label="Quality delta after TER inversion")
    fig.tight_layout()
    out_file = plots_dir / "delta_finetuned_adapted_minus_historical.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [out_file]


def save_bleu_chrf_scatter_plots(df: pd.DataFrame, root: Path) -> list[Path]:
    plots_dir = root / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    chr_metrics = ["chrF1", "chrF3"]
    index_cols = ["ModelType", "TrainDataset", "Direction", "Dataset", "ModelLabel"]

    wide = (
        df[df["Metric"].isin(["BLEU", *chr_metrics])]
        .pivot_table(index=index_cols, columns="Metric", values="Mean", aggfunc="first")
        .reset_index()
    )

    for direction in _ordered_directions(df):
        df_dir = wide[wide["Direction"] == direction].copy()
        if df_dir.empty or "BLEU" not in df_dir.columns:
            continue

        for chr_metric in chr_metrics:
            if chr_metric not in df_dir.columns:
                continue

            plot_df = df_dir[["Dataset", "ModelLabel", "BLEU", chr_metric]].dropna()
            if plot_df.empty:
                continue

            fig, ax = plt.subplots(figsize=(8, 6.5))
            for dataset in _ordered_datasets(plot_df):
                df_dataset = plot_df[plot_df["Dataset"] == dataset]
                for model_label in _ordered_model_labels(df_dataset):
                    row = df_dataset[df_dataset["ModelLabel"] == model_label]
                    if row.empty:
                        continue
                    ax.scatter(
                        row["BLEU"],
                        row[chr_metric],
                        s=95,
                        marker=MARKERS.get(dataset, "o"),
                        color=MODEL_COLORS.get(model_label, "#8a8f98"),
                        edgecolor="white",
                        linewidth=0.8,
                        label=f"{model_label} | {DATASET_LABELS.get(dataset, dataset)}",
                    )
                    for _, point in row.iterrows():
                        ax.annotate(
                            DATASET_LABELS.get(dataset, dataset),
                            xy=(point["BLEU"], point[chr_metric]),
                            xytext=(6, 5),
                            textcoords="offset points",
                            fontsize=8,
                        )

            ax.set_title(f"BLEU vs {chr_metric} | {_display_direction(direction)}")
            ax.set_xlabel("BLEU (higher is better)")
            ax.set_ylabel(f"{chr_metric} (higher is better)")
            ax.grid(alpha=0.22)
            ax.set_axisbelow(True)

            handles, labels = ax.get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            ax.legend(
                unique.values(),
                unique.keys(),
                title="Model | evaluation corpus",
                fontsize=8,
                loc="best",
            )

            fig.tight_layout()
            out_file = plots_dir / f"scatter_bleu_{_safe_metric_name(chr_metric)}_{direction}.png"
            fig.savefig(out_file, dpi=300, bbox_inches="tight")
            plt.close(fig)
            saved.append(out_file)

    return saved


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate comparison plots from aggregate metrics.")
    parser.add_argument("--project-root", type=Path, default=None)
    return parser.parse_args(argv)


def main(project_root: Path | None = None) -> None:
    root = project_root.resolve() if project_root else _project_root()

    df = load_aggregate_metrics(root)
    if df.empty:
        raise FileNotFoundError(
            "No aggregate metrics found. Run zero-shot/predict first to generate *_aggregate.csv files."
        )
    validate_expected_metric_matrix(df)

    metrics_out = root / "results" / "metrics" / "comparison_aggregate.csv"
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(metrics_out, index=False)

    _clear_generated_plot_files(root)
    saved = save_comparative_performance_analysis(df, root)
    saved.extend(save_combined_heatmap_plots(df, root))
    saved.extend(save_summary_table_plot(df, root))
    saved.extend(save_finetuned_delta_heatmap(df, root))
    saved.extend(save_bleu_chrf_scatter_plots(df, root))

    print(f"Saved combined aggregate table: {metrics_out}")
    for path in saved:
        print(f"Saved plot: {path}")


if __name__ == "__main__":
    args = _parse_args()
    main(project_root=args.project_root)
