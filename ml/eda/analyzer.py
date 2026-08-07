"""Core exploratory analysis routines for the PaySim dataset."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


NUMERICAL_FEATURES: tuple[str, ...] = (
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
)


@dataclass(slots=True)
class EDAAnalyzer:
    """Perform structured EDA over the PaySim dataset."""

    dataframe: pd.DataFrame

    def dataset_overview(self) -> dict[str, object]:
        """Return high-level dataset profile details."""
        memory_usage_mb = self.dataframe.memory_usage(deep=True).sum() / (1024**2)
        return {
            "row_count": int(self.dataframe.shape[0]),
            "column_count": int(self.dataframe.shape[1]),
            "shape": self.dataframe.shape,
            "dtypes": {
                column: str(dtype) for column, dtype in self.dataframe.dtypes.items()
            },
            "memory_usage_mb": round(memory_usage_mb, 2),
            "duplicate_rows": int(self.dataframe.duplicated().sum()),
        }

    def target_analysis(self) -> dict[str, float | int]:
        """Return target distribution metrics for `isFraud`."""
        counts = self.dataframe["isFraud"].value_counts().sort_index()
        fraud_count = int(counts.get(1, 0))
        legitimate_count = int(counts.get(0, 0))
        total_count = fraud_count + legitimate_count
        fraud_percentage = (fraud_count / total_count * 100) if total_count else 0.0
        imbalance_ratio = (
            legitimate_count / fraud_count if fraud_count else float("inf")
        )
        return {
            "fraud_count": fraud_count,
            "legitimate_count": legitimate_count,
            "fraud_percentage": round(fraud_percentage, 6),
            "class_imbalance_ratio": round(float(imbalance_ratio), 6),
        }

    def transaction_type_analysis(self) -> pd.DataFrame:
        """Return transaction frequencies and fraud rates by transaction type."""
        grouped = self.dataframe.groupby("type", dropna=False)["isFraud"]
        summary = grouped.agg(["count", "sum", "mean"]).reset_index()
        summary.columns = ["type", "transaction_count", "fraud_count", "fraud_rate"]
        summary["fraud_rate"] = summary["fraud_rate"] * 100
        return summary.sort_values("transaction_count", ascending=False)

    def numerical_feature_statistics(self) -> pd.DataFrame:
        """Return summary statistics and outlier counts for core numeric features."""
        statistics = self.dataframe.loc[:, NUMERICAL_FEATURES].describe().transpose()
        quartiles = self.dataframe.loc[:, NUMERICAL_FEATURES].quantile([0.25, 0.75])
        statistics["iqr"] = quartiles.loc[0.75] - quartiles.loc[0.25]
        statistics["lower_bound"] = quartiles.loc[0.25] - 1.5 * statistics["iqr"]
        statistics["upper_bound"] = quartiles.loc[0.75] + 1.5 * statistics["iqr"]

        outlier_counts: dict[str, int] = {}
        for feature in NUMERICAL_FEATURES:
            series = self.dataframe[feature]
            lower_bound = statistics.loc[feature, "lower_bound"]
            upper_bound = statistics.loc[feature, "upper_bound"]
            outlier_counts[feature] = int(
                ((series < lower_bound) | (series > upper_bound)).sum()
            )

        statistics["outlier_count"] = pd.Series(outlier_counts)
        return statistics.reset_index(names="feature")

    def correlation_analysis(self) -> pd.DataFrame:
        """Return the correlation matrix for numeric columns."""
        numeric_columns = self.dataframe.select_dtypes(include=["number"]).columns
        return self.dataframe.loc[:, numeric_columns].corr(numeric_only=True)

    def correlation_with_target(self) -> pd.DataFrame:
        """Return feature correlations with the target column."""
        correlation_matrix = self.correlation_analysis()
        target_correlations = correlation_matrix["isFraud"].sort_values(ascending=False)
        return target_correlations.reset_index().rename(
            columns={"index": "feature", "isFraud": "correlation_with_isFraud"}
        )

    def missing_values_analysis(self) -> pd.DataFrame:
        """Return missing value counts and percentages per column."""
        missing_counts = self.dataframe.isna().sum()
        missing_percentages = (missing_counts / len(self.dataframe.index)) * 100
        return (
            pd.DataFrame(
                {
                    "column": missing_counts.index,
                    "missing_count": missing_counts.values.astype(int),
                    "missing_percentage": missing_percentages.values,
                }
            )
            .sort_values(["missing_count", "column"], ascending=[False, True])
            .reset_index(drop=True)
        )

    def leakage_analysis(self) -> dict[str, object]:
        """Assess columns that may create target leakage or identity leakage risks."""
        flagged_fraud_rates = (
            self.dataframe.groupby("isFlaggedFraud")["isFraud"]
            .agg(["count", "sum", "mean"])
            .reset_index()
        )
        flagged_fraud_rates["mean"] = flagged_fraud_rates["mean"] * 100

        return {
            "isflaggedfraud_summary": flagged_fraud_rates.rename(
                columns={
                    "count": "transaction_count",
                    "sum": "fraud_count",
                    "mean": "fraud_rate",
                }
            ),
            "nameorig_unique_ratio": round(
                self.dataframe["nameOrig"].nunique(dropna=True)
                / len(self.dataframe.index)
                * 100,
                6,
            ),
            "namedest_unique_ratio": round(
                self.dataframe["nameDest"].nunique(dropna=True)
                / len(self.dataframe.index)
                * 100,
                6,
            ),
        }
