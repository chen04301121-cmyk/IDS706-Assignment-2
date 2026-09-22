import pandas as pd
import pytest

from gold_analysis import (
    load_data,
    preprocess_data,
    filter_above_average,
    yearly_summary,
)


def make_data(dates, prices):
    """Create small, predictable data for testing."""
    size = len(dates)

    return pd.DataFrame({
        "Date": dates,
        "SPX": [2000.0] * size,
        "GLD": prices,
        "USO": [50.0] * size,
        "SLV": [15.0] * size,
        "EUR/USD": [1.1] * size,
    })


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
    data = pd.DataFrame({
        "Year": [2020, 2020, 2021],
        "GLD": [100.0, 200.0, 300.0],
    })

    result = yearly_summary(data)

    expected = pd.DataFrame({
        "average_price": [150.0, 300.0],
        "minimum_price": [100.0, 300.0],
        "maximum_price": [200.0, 300.0],
        "observation_count": [2, 1],
    }, index=pd.Index([2020, 2021], name="Year"))

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