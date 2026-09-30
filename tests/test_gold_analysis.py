import pandas as pd
import pytest

from gold_analysis import (
    load_data,
    preprocess_data,
    filter_above_average,
    yearly_summary,
    train_and_evaluate,
    calculate_daily_returns,
)


def make_data(dates, prices):
    """Create small, predictable data for testing."""
    size = len(dates)

    return pd.DataFrame(
        {
            "Date": dates,
            "SPX": [2000.0] * size,
            "GLD": prices,
            "USO": [50.0] * size,
            "SLV": [15.0] * size,
            "EUR/USD": [1.1] * size,
        }
    )


def test_load_data_reads_csv(tmp_path):
    expected = make_data(
        ["2020-01-01", "2020-01-02"],
        [100.0, 200.0],
    )
    file_path = tmp_path / "gold.csv"
    expected.to_csv(file_path, index=False)

    result = load_data(file_path)

    pd.testing.assert_frame_equal(result, expected)


def test_preprocess_sorts_dates_and_adds_year():
    data = make_data(
        ["2021-01-01", "2020-01-01"],
        ["200", "100"],
    )

    result = preprocess_data(data)

    assert result["Date"].tolist() == [
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2021-01-01"),
    ]
    assert result["Year"].tolist() == [2020, 2021]
    assert result["GLD"].tolist() == [100, 200]

    # Preprocessing should not change the original data.
    assert data["GLD"].tolist() == ["200", "100"]
    assert "Year" not in data.columns


def test_filter_above_average_excludes_equal_values():
    data = make_data(
        ["2020-01-01", "2020-01-02", "2020-01-03"],
        [100.0, 200.0, 300.0],
    )

    result = filter_above_average(data)

    # The average is 200, so only 300 should remain.
    assert result["GLD"].tolist() == [300.0]


def test_yearly_summary_calculates_each_year():
    data = pd.DataFrame(
        {
            "Year": [2020, 2020, 2021],
            "GLD": [100.0, 200.0, 300.0],
        }
    )

    result = yearly_summary(data)

    expected = pd.DataFrame(
        {
            "average_price": [150.0, 300.0],
            "minimum_price": [100.0, 300.0],
            "maximum_price": [200.0, 300.0],
            "observation_count": [2, 1],
        },
        index=pd.Index([2020, 2021], name="Year"),
    )

    pd.testing.assert_frame_equal(result, expected)


def test_load_data_rejects_missing_columns(tmp_path):
    file_path = tmp_path / "missing_column.csv"
    data = make_data(["2020-01-01"], [100.0])
    data.drop(columns=["GLD"]).to_csv(file_path, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        load_data(file_path)


def test_preprocess_removes_invalid_and_duplicate_rows():
    data = make_data(
        [
            "2020-01-01",
            "not-a-date",
            "2020-01-03",
            "2020-01-01",
        ],
        [100.0, 200.0, "bad", 100.0],
    )

    result = preprocess_data(data)

    # Keep one valid row; remove the invalid date,
    # invalid price, and duplicate.
    assert len(result) == 1
    assert result.iloc[0]["GLD"] == 100.0
    assert result.iloc[0]["Date"] == pd.Timestamp("2020-01-01")


def test_preprocess_rejects_all_invalid_data():
    data = make_data(["not-a-date"], [100.0])

    with pytest.raises(ValueError, match="No valid rows"):
        preprocess_data(data)


def test_model_predicts_linear_data_in_chronological_order():
    """Verify known predictions and keep the latest dates for testing."""
    dates = pd.date_range("2020-01-01", periods=10)
    data = make_data(dates, [2 * value + 10 for value in range(10)])
    data["SPX"] = list(range(10))

    # Shuffle input to verify that the function sorts before splitting.
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    original = data.copy(deep=True)

    result = train_and_evaluate(data)

    assert result["train_count"] == 8
    assert result["test_count"] == 2
    assert result["test_dates"].tolist() == dates[-2:].tolist()
    assert result["actual"].tolist() == [26, 28]
    assert result["predictions"] == pytest.approx([26, 28])
    assert result["mae"] == pytest.approx(0, abs=1e-8)
    assert result["r_squared"] == pytest.approx(1)

    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize("train_fraction", [-0.1, 0, 1, 1.1])
def test_model_rejects_invalid_train_fraction(train_fraction):
    data = make_data(
        pd.date_range("2020-01-01", periods=10),
        list(range(100, 110)),
    )

    with pytest.raises(ValueError, match="train_fraction"):
        train_and_evaluate(data, train_fraction=train_fraction)


@pytest.mark.parametrize("train_fraction", [0.1, 0.9])
def test_model_rejects_too_small_training_or_test_set(train_fraction):
    data = make_data(
        pd.date_range("2020-01-01", periods=10),
        list(range(100, 110)),
    )

    with pytest.raises(ValueError, match="at least two rows"):
        train_and_evaluate(data, train_fraction=train_fraction)


def test_daily_returns_calculates_correctly_and_sorts_dates():
    data = make_data(
        ["2020-01-03", "2020-01-01", "2020-01-02"],
        [99.0, 100.0, 110.0],
    )
    data["Date"] = pd.to_datetime(data["Date"])
    original = data.copy(deep=True)

    result = calculate_daily_returns(data)

    assert result.index.tolist() == [
        pd.Timestamp("2020-01-02"),
        pd.Timestamp("2020-01-03"),
    ]
    assert result.columns.tolist() == ["GLD", "SPX", "USO", "SLV"]

    # GLD moves from 100 to 110 (+10%), then to 99 (-10%).
    assert result["GLD"].tolist() == pytest.approx([0.1, -0.1])

    # The other prices are constant in this test dataset.
    for column in ["SPX", "USO", "SLV"]:
        assert result[column].tolist() == pytest.approx([0.0, 0.0])

    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize("row_count", [0, 1])
def test_daily_returns_rejects_insufficient_data(row_count):
    data = make_data(
        pd.date_range("2020-01-01", periods=row_count),
        [100.0] * row_count,
    )

    with pytest.raises(ValueError, match="At least two observations"):
        calculate_daily_returns(data)


@pytest.mark.parametrize(
    "invalid_price",
    [0.0, -1.0, float("nan"), float("inf"), float("-inf")],
)
def test_daily_returns_rejects_invalid_prices(invalid_price):
    data = make_data(
        pd.date_range("2020-01-01", periods=3),
        [100.0, invalid_price, 110.0],
    )

    with pytest.raises(ValueError, match="finite, positive, and non-missing"):
        calculate_daily_returns(data)
