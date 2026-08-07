"""Visualization utilities for PaySim exploratory analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from ml.eda.analyzer import NUMERICAL_FEATURES


class EDAVisualizer:
    """Create and persist EDA figures."""

    def __init__(self, output_dir: str | Path) -> None:
        """Initialize the visualizer with a target output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save_figure(self, file_name: str) -> None:
        """Save and close the active Matplotlib figure."""
        plt.tight_layout()
        plt.savefig(self.output_dir / file_name, dpi=200, bbox_inches="tight")
        plt.close()

    def plot_target_count(self, dataframe: pd.DataFrame) -> None:
        """Save a count plot for the target distribution."""
        counts = dataframe["isFraud"].value_counts().sort_index()
        labels = ["Legitimate", "Fraud"]
        values = [int(counts.get(0, 0)), int(counts.get(1, 0))]

        plt.figure(figsize=(8, 5))
        plt.bar(labels, values, color=["#2c7fb8", "#d95f0e"])
        plt.title("Target Distribution: isFraud")
        plt.ylabel("Transaction Count")
        self._save_figure("target_count.png")

    def plot_target_pie(self, dataframe: pd.DataFrame) -> None:
        """Save a pie chart for target class share."""
        counts = dataframe["isFraud"].value_counts().sort_index()
        values = [int(counts.get(0, 0)), int(counts.get(1, 0))]

        plt.figure(figsize=(7, 7))
        plt.pie(
            values,
            labels=["Legitimate", "Fraud"],
            autopct="%1.4f%%",
            startangle=90,
            colors=["#2c7fb8", "#d95f0e"],
        )
        plt.title("Fraud vs Legitimate Transactions")
        self._save_figure("target_pie.png")

    def plot_transaction_type_frequency(self, transaction_summary: pd.DataFrame) -> None:
        """Save a bar chart of transaction frequency by type."""
        plt.figure(figsize=(9, 5))
        plt.bar(
            transaction_summary["type"],
            transaction_summary["transaction_count"],
            color="#1b9e77",
        )
        plt.title("Transaction Frequency by Type")
        plt.xlabel("Transaction Type")
        plt.ylabel("Transaction Count")
        plt.xticks(rotation=25)
        self._save_figure("transaction_type_frequency.png")

    def plot_transaction_type_fraud_rate(self, transaction_summary: pd.DataFrame) -> None:
        """Save a bar chart of fraud rate by transaction type."""
        plt.figure(figsize=(9, 5))
        plt.bar(
            transaction_summary["type"],
            transaction_summary["fraud_rate"],
            color="#e7298a",
        )
        plt.title("Fraud Rate by Transaction Type")
        plt.xlabel("Transaction Type")
        plt.ylabel("Fraud Rate (%)")
        plt.xticks(rotation=25)
        self._save_figure("transaction_type_fraud_rate.png")

    def plot_numerical_distributions(self, dataframe: pd.DataFrame) -> None:
        """Save histogram, KDE, and boxplot for each core numeric feature."""
        sample_size = min(len(dataframe.index), 50_000)
        sample = dataframe.loc[:, NUMERICAL_FEATURES].sample(
            n=sample_size, random_state=42
        )

        for feature in NUMERICAL_FEATURES:
            series = dataframe[feature].dropna()
            sample_series = sample[feature].dropna()

            plt.figure(figsize=(9, 5))
            plt.hist(series, bins=60, color="#4c78a8", alpha=0.75)
            plt.title(f"Histogram: {feature}")
            plt.xlabel(feature)
            plt.ylabel("Frequency")
            self._save_figure(f"{feature}_histogram.png")

            plt.figure(figsize=(9, 5))
            if sample_series.nunique() > 1:
                kde = gaussian_kde(sample_series.astype(float))
                x_grid = pd.Series(sample_series).quantile([0.01, 0.99]).tolist()
                grid = np.linspace(x_grid[0], x_grid[1], 500)
                plt.plot(grid, kde(grid), color="#d95f02")
            plt.title(f"KDE Plot: {feature}")
            plt.xlabel(feature)
            plt.ylabel("Density")
            self._save_figure(f"{feature}_kde.png")

            plt.figure(figsize=(9, 3))
            plt.boxplot(sample_series, vert=False)
            plt.title(f"Boxplot: {feature}")
            plt.xlabel(feature)
            self._save_figure(f"{feature}_boxplot.png")

    def plot_correlation_heatmap(self, correlation_matrix: pd.DataFrame) -> None:
        """Save a correlation heatmap for numeric columns."""
        plt.figure(figsize=(8, 6))
        plt.imshow(correlation_matrix, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
        plt.colorbar(label="Correlation")
        plt.xticks(
            ticks=range(len(correlation_matrix.columns)),
            labels=correlation_matrix.columns,
            rotation=45,
            ha="right",
        )
        plt.yticks(
            ticks=range(len(correlation_matrix.index)),
            labels=correlation_matrix.index,
        )
        plt.title("Correlation Heatmap")
        self._save_figure("correlation_heatmap.png")
