import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

REQUIRED_COLUMNS = ["Date", "SPX", "GLD", "USO", "SLV", "EUR/USD"]


def load_data(file_path):
    """Read the CSV and check required columns."""
    data = pd.read_csv(file_path)

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in data.columns
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


def train_and_evaluate(data, train_fraction=0.8):
    """Fit and evaluate a linear model using a chronological split."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")

    ordered = data.sort_values("Date").reset_index(drop=True)
    split_index = int(len(ordered) * train_fraction)

    if split_index < 2 or len(ordered) - split_index < 2:
        raise ValueError("Training and test sets must each have at least two rows.")

    features = ["SPX", "USO", "SLV", "EUR/USD"]
    train = ordered.iloc[:split_index]
    test = ordered.iloc[split_index:]

    model = LinearRegression()
    model.fit(train[features], train["GLD"])
    predictions = model.predict(test[features])

    return {
        "train_count": len(train),
        "test_count": len(test),
        "test_dates": test["Date"],
        "actual": test["GLD"],
        "predictions": predictions,
        "mae": mean_absolute_error(test["GLD"], predictions),
        "r_squared": r2_score(test["GLD"], predictions),
        "coefficients": pd.DataFrame(
            {
                "Variable": features,
                "Coefficient": model.coef_,
            }
        ),
        "intercept": model.intercept_,
    }


def calculate_daily_returns(data):
    """Calculate price returns between consecutive trading observations."""
    columns = ["GLD", "SPX", "USO", "SLV"]
    ordered = data.sort_values("Date").set_index("Date")
    prices = ordered[columns]

    if len(prices) < 2:
        raise ValueError("At least two observations are required.")

    if (
        prices.isna().any().any()
        or prices.isin([float("inf"), float("-inf")]).any().any()
        or (prices <= 0).any().any()
    ):
        raise ValueError("Prices must be finite, positive, and non-missing.")

    return prices.pct_change(fill_method=None).iloc[1:]
