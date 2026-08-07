"""Business insight generation for the PaySim EDA phase."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class BusinessInsightsGenerator:
    """Generate narrative insights from EDA outputs."""

    target_summary: dict[str, float | int]
    transaction_summary: pd.DataFrame
    feature_statistics: pd.DataFrame
    target_correlations: pd.DataFrame
    leakage_summary: dict[str, object]

    def build_markdown(self) -> str:
        """Return a markdown-formatted business insights report."""
        top_volume_type = self.transaction_summary.iloc[0]
        top_fraud_rate_type = self.transaction_summary.sort_values(
            "fraud_rate", ascending=False
        ).iloc[0]
        highest_outlier_feature = self.feature_statistics.sort_values(
            "outlier_count", ascending=False
        ).iloc[0]
        strongest_positive_corr = (
            self.target_correlations[self.target_correlations["feature"] != "isFraud"]
            .sort_values("correlation_with_isFraud", ascending=False)
            .iloc[0]
        )

        isflaggedfraud_summary = self.leakage_summary["isflaggedfraud_summary"]
        flagged_fraud_rate = isflaggedfraud_summary.loc[
            isflaggedfraud_summary["isFlaggedFraud"] == 1, "fraud_rate"
        ]
        flagged_rate_text = (
            f"{float(flagged_fraud_rate.iloc[0]):.4f}%"
            if not flagged_fraud_rate.empty
            else "0.0000%"
        )

        return f"""# Business Insights

## Executive Summary

- Fraud is extremely rare in the dataset at {self.target_summary['fraud_percentage']:.6f}%, which confirms a highly imbalanced fraud detection problem.
- The class imbalance ratio is approximately {self.target_summary['class_imbalance_ratio']:.2f}:1 legitimate-to-fraud transactions, so later modeling phases will need imbalance-aware evaluation.
- `{top_volume_type['type']}` is the highest-volume transaction type with {int(top_volume_type['transaction_count'])} transactions.
- `{top_fraud_rate_type['type']}` has the highest observed fraud rate at {float(top_fraud_rate_type['fraud_rate']):.4f}% among transaction types.

## Numerical Risk Signals

- `{highest_outlier_feature['feature']}` shows the largest outlier count ({int(highest_outlier_feature['outlier_count'])}), suggesting heavy-tailed behavior that may matter during later preprocessing and thresholding.
- The strongest positive linear relationship with `isFraud` among numeric features is `{strongest_positive_corr['feature']}` with correlation {float(strongest_positive_corr['correlation_with_isFraud']):.6f}.

## Leakage Review

- `isFlaggedFraud` may be a leakage-prone feature because transactions where it equals 1 show a fraud rate of {flagged_rate_text}, which is far from the baseline rate.
- `nameOrig` has a unique-value ratio of {self.leakage_summary['nameorig_unique_ratio']:.4f}% and `nameDest` has {self.leakage_summary['namedest_unique_ratio']:.4f}%, indicating they are high-cardinality identifiers.
- High-cardinality identity fields may behave as memorization shortcuts rather than stable business predictors, so they should be treated cautiously in future phases.

## Analyst Takeaways

- Fraud concentration by transaction type should guide later feature review and threshold analysis.
- Extremely low fraud prevalence means accuracy alone will be misleading in future modeling work.
- Outliers and account-balance behavior appear important enough to inspect carefully during future preprocessing, but no transformations are applied in this phase.
"""
