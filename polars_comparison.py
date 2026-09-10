import timeit
import pandas as pd
import polars as pl

FILE_PATH = "gold_data_2015_25.csv"


def run_pandas():
    """Read, filter, and group the data using Pandas."""

    df = pd.read_csv(FILE_PATH)

    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year

    average_gld = df["GLD"].mean()
    above_average = df[df["GLD"] > average_gld]

    yearly_summary = (
        df.groupby("Year")["GLD"]
        .agg(["mean", "min", "max", "count"])
    )

    return above_average, yearly_summary


def run_polars():
    """Read, filter, and group the data using Polars."""

    df = pl.read_csv(FILE_PATH)

    df = df.with_columns(
        pl.col("Date")
        .str.to_date("%Y-%m-%d")
        .alias("Date")
    )

    df = df.with_columns(
        pl.col("Date")
        .dt.year()
        .alias("Year")
    )

    average_gld = df["GLD"].mean()

    above_average = df.filter(
        pl.col("GLD") > average_gld
    )

    yearly_summary = (
        df.group_by("Year")
        .agg(
            pl.col("GLD").mean().alias("average_price"),
            pl.col("GLD").min().alias("minimum_price"),
            pl.col("GLD").max().alias("maximum_price"),
            pl.col("GLD").count().alias("observation_count"),
        )
        .sort("Year")
    )

    return above_average, yearly_summary


# Run each version once and display comparable results
pandas_filtered, pandas_summary = run_pandas()
polars_filtered, polars_summary = run_polars()

print("Pandas rows above average:")
print(len(pandas_filtered))

print("\nPolars rows above average:")
print(polars_filtered.height)

print("\nPolars yearly summary:")
print(polars_summary)

# Repeat the same work to compare approximate performance
runs = 100

pandas_seconds = timeit.timeit(run_pandas, number=runs)
polars_seconds = timeit.timeit(run_polars, number=runs)

print(f"\nPerformance comparison over {runs} runs:")
print(f"Pandas total time: {pandas_seconds:.4f} seconds")
print(f"Polars total time: {polars_seconds:.4f} seconds")

print(f"Pandas average per run: {pandas_seconds / runs:.6f} seconds")
print(f"Polars average per run: {polars_seconds / runs:.6f} seconds")

if polars_seconds < pandas_seconds:
    speedup = pandas_seconds / polars_seconds
    print(f"Polars was approximately {speedup:.2f} times faster.")
else:
    speedup = polars_seconds / pandas_seconds
    print(f"Pandas was approximately {speedup:.2f} times faster.")