import pandas as pd


REQUIRED_COLUMNS = ["Date", "SPX", "GLD", "USO", "SLV", "EUR/USD"]


def load_data(file_path):
    """Read the CSV and check required columns."""
    data = pd.read_csv(file_path)

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return data


def preprocess_data(data):
    """Clean values, sort by date, and add the Year column."""
    cleaned = data.copy()

    cleaned["Date"] = pd.to_datetime(
        cleaned["Date"],
        format="%Y-%m-%d",
        errors="coerce",
    )

    for column in REQUIRED_COLUMNS[1:]:
        cleaned[column] = pd.to_numeric(
            cleaned[column],
            errors="coerce",
        )

    cleaned = cleaned.dropna(subset=REQUIRED_COLUMNS)
    cleaned = cleaned.drop_duplicates()
    cleaned = cleaned.sort_values("Date").reset_index(drop=True)

    if cleaned.empty:
        raise ValueError("No valid rows remain after preprocessing.")

    cleaned["Year"] = cleaned["Date"].dt.year

    return cleaned


def filter_above_average(data):
    """Return rows where GLD is strictly above its overall mean."""
    average = data["GLD"].mean()
    return data.loc[data["GLD"] > average].copy()


def yearly_summary(data):
    """Calculate GLD summary statistics for each year."""
    return data.groupby("Year")["GLD"].agg(
        average_price="mean",
        minimum_price="min",
        maximum_price="max",
        observation_count="count",
    )