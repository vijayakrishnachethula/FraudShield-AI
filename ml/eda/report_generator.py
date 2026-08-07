"""EDA report generation pipeline for the PaySim dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.settings import settings
from ml.eda.analyzer import EDAAnalyzer
from ml.eda.insights import BusinessInsightsGenerator
from ml.eda.loader import load_paysim_dataset
from ml.eda.visualizer import EDAVisualizer


def _to_markdown_table(dataframe: pd.DataFrame) -> str:
    """Convert a dataframe to a markdown table."""
    headers = [str(column) for column in dataframe.columns]
    separator = ["---"] * len(headers)
    rows = [
        [str(value) for value in row]
        for row in dataframe.fillna("").itertuples(index=False, name=None)
    ]
    table_rows = [headers, separator, *rows]
    return "\n".join("| " + " | ".join(row) + " |" for row in table_rows)


def _build_validation_markdown(validation_report: object) -> str:
    """Build markdown for the validation report."""
    missing_table = pd.DataFrame(
        {
            "column": list(validation_report.missing_values_by_column.keys()),
            "missing_count": list(validation_report.missing_values_by_column.values()),
            "missing_percentage": list(
                validation_report.missing_percentage_by_column.values()
            ),
        }
    )

    invalid_dtype_text = (
        ", ".join(
            f"{column}={dtype}"
            for column, dtype in validation_report.invalid_dtypes.items()
        )
        if validation_report.invalid_dtypes
        else "None"
    )

    return f"""# Validation Report

## Summary

- Dataset path: `{validation_report.dataset_path}`
- Rows: {validation_report.row_count}
- Columns: {validation_report.column_count}
- Duplicate rows: {validation_report.duplicate_rows}
- Required columns present: {validation_report.required_columns_present}
- Invalid dtypes: {invalid_dtype_text}

## Missing Values

{_to_markdown_table(missing_table)}
"""


def _build_eda_summary_markdown(
    overview: dict[str, object],
    target_summary: dict[str, float | int],
    transaction_summary: pd.DataFrame,
    feature_statistics: pd.DataFrame,
    missing_values: pd.DataFrame,
    target_correlations: pd.DataFrame,
) -> str:
    """Build markdown for the EDA summary report."""
    top_feature_stats = feature_statistics[
        ["feature", "mean", "std", "min", "25%", "50%", "75%", "max", "outlier_count"]
    ]

    return f"""# EDA Summary

## Dataset Overview

- Shape: {overview['shape']}
- Memory usage: {overview['memory_usage_mb']} MB
- Duplicate rows: {overview['duplicate_rows']}

## Data Types

{_to_markdown_table(pd.DataFrame(list(overview['dtypes'].items()), columns=['column', 'dtype']))}

## Target Analysis

- Fraud count: {target_summary['fraud_count']}
- Legitimate count: {target_summary['legitimate_count']}
- Fraud percentage: {target_summary['fraud_percentage']:.6f}%
- Class imbalance ratio: {target_summary['class_imbalance_ratio']:.6f}

## Transaction Type Analysis

{_to_markdown_table(transaction_summary)}

## Numerical Feature Statistics

{_to_markdown_table(top_feature_stats)}

## Missing Values

{_to_markdown_table(missing_values)}

## Correlation With Target

{_to_markdown_table(target_correlations)}
"""


def run_eda_pipeline(
    dataset_path: str | Path | None = None,
    reports_dir: str | Path = "reports/eda",
) -> None:
    """Run validation, analysis, visualization, and report generation.

    Args:
        dataset_path: Optional override for the dataset path.
        reports_dir: Target directory for generated reports.
    """
    resolved_dataset_path = dataset_path or settings.dataset_path
    output_dir = Path(reports_dir)
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    dataframe, validation_report = load_paysim_dataset(resolved_dataset_path)
    analyzer = EDAAnalyzer(dataframe=dataframe)

    overview = analyzer.dataset_overview()
    target_summary = analyzer.target_analysis()
    transaction_summary = analyzer.transaction_type_analysis()
    feature_statistics = analyzer.numerical_feature_statistics()
    correlation_matrix = analyzer.correlation_analysis()
    target_correlations = analyzer.correlation_with_target()
    missing_values = analyzer.missing_values_analysis()
    leakage_summary = analyzer.leakage_analysis()

    visualizer = EDAVisualizer(plots_dir)
    visualizer.plot_target_count(dataframe)
    visualizer.plot_target_pie(dataframe)
    visualizer.plot_transaction_type_frequency(transaction_summary)
    visualizer.plot_transaction_type_fraud_rate(transaction_summary)
    visualizer.plot_numerical_distributions(dataframe)
    visualizer.plot_correlation_heatmap(correlation_matrix)

    insights_markdown = BusinessInsightsGenerator(
        target_summary=target_summary,
        transaction_summary=transaction_summary,
        feature_statistics=feature_statistics,
        target_correlations=target_correlations,
        leakage_summary=leakage_summary,
    ).build_markdown()

    (output_dir / "feature_statistics.csv").write_text(
        feature_statistics.to_csv(index=False),
        encoding="utf-8",
    )
    (output_dir / "missing_values.csv").write_text(
        missing_values.to_csv(index=False),
        encoding="utf-8",
    )
    (output_dir / "correlation.csv").write_text(
        correlation_matrix.to_csv(),
        encoding="utf-8",
    )
    (output_dir / "validation_report.md").write_text(
        _build_validation_markdown(validation_report),
        encoding="utf-8",
    )
    (output_dir / "eda_summary.md").write_text(
        _build_eda_summary_markdown(
            overview=overview,
            target_summary=target_summary,
            transaction_summary=transaction_summary,
            feature_statistics=feature_statistics,
            missing_values=missing_values,
            target_correlations=target_correlations,
        ),
        encoding="utf-8",
    )
    (output_dir / "business_insights.md").write_text(
        insights_markdown,
        encoding="utf-8",
    )


if __name__ == "__main__":
    run_eda_pipeline()
