import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from gold_analysis import (
    load_data,
    preprocess_data,
    filter_above_average,
    yearly_summary,
    train_and_evaluate,
)


def plot_price_trend(data, output_path):
    """Save a GLD price trend chart without modifying the input data."""
    sorted_data = data.sort_values("Date")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        sorted_data["Date"],
        sorted_data["GLD"],
        color="goldenrod",
        linewidth=1.5,
    )
    ax.set_title("GLD Price Trend from 2015 to 2025")
    ax.set_xlabel("Date")
    ax.set_ylabel("GLD Price")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_model_results(results, output_path):
    """Save actual and estimated GLD prices for the test period."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        results["test_dates"],
        results["actual"],
        label="Actual GLD",
        color="goldenrod",
    )
    ax.plot(
        results["test_dates"],
        results["predictions"],
        label="Estimated GLD",
        color="navy",
        linestyle="--",
    )
    ax.set_title("Actual vs. Estimated GLD Prices")
    ax.set_xlabel("Date")
    ax.set_ylabel("GLD Price")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    """Run the analysis using configurable input and output paths."""
    parser = argparse.ArgumentParser(description="Analyze GLD market data.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("gold_data_2015_25.csv"),
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for generated charts.",
    )
    args = parser.parse_args()

    gold = load_data(args.data)

    print("First five rows:")
    print(gold.head())
    print("\nData info:")
    gold.info()
    print("\nSummary statistics:")
    print(gold.describe())
    print("\nMissing values:")
    print(gold.isnull().sum())
    print("\nNumber of duplicate rows:")
    print(gold.duplicated().sum())
    print("\nData types:")
    print(gold.dtypes)

    gold = preprocess_data(gold)
    above_average = filter_above_average(gold)
    annual_summary = yearly_summary(gold)

    print("\nOverall average GLD price:")
    print(round(gold["GLD"].mean(), 2))
    print("\nFirst five rows where GLD is above average:")
    print(above_average.head())
    print("\nNumber of rows where GLD is above average:")
    print(len(above_average))
    print("\nYearly GLD summary:")
    print(annual_summary.round(2))

    results = train_and_evaluate(gold)

    print("\nTraining observations:")
    print(results["train_count"])
    print("\nTesting observations:")
    print(results["test_count"])
    print("\nLinear Regression Results:")
    print("Mean Absolute Error:", round(results["mae"], 2))
    print("R-squared:", round(results["r_squared"], 3))
    print("\nModel coefficients:")
    print(results["coefficients"].round(3))
    print("\nModel intercept:")
    print(round(results["intercept"], 3))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_price_trend(gold, args.output_dir / "gld_price_trend.png")
    plot_model_results(results, args.output_dir / "actual_vs_estimated_gld.png")

    print("\nCharts saved to:")
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
