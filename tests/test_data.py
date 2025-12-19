from pathlib import Path

import numpy as np
import pandas as pd

import hydrostats.data as hd


def test_julian_to_gregorian() -> None:
    julian_dates = np.array(
        [
            2444239.5,
            2444239.5416666665,
            2444239.5833333335,
            2444239.625,
            2444239.6666666665,
            2444239.7083333335,
            2444239.75,
            2444239.7916666665,
            2444239.8333333335,
            2444239.875,
        ]
    )
    expected_dates = pd.date_range("1980-01-01", periods=10, freq="H")

    rng = np.random.default_rng()
    data = rng.random((10, 2))
    test_df = pd.DataFrame(
        data=data,
        columns=("Simulated Data", "Observed Data"),
        index=julian_dates,
    )

    test_df_gregorian = hd.julian_to_gregorian(test_df, frequency="H")
    assert np.all(test_df_gregorian.index == expected_dates)

    hd.julian_to_gregorian(test_df, inplace=True, frequency="H")
    assert np.all(test_df.index == expected_dates)


def test_daily_average(merged_df: pd.DataFrame, comparison_files: Path) -> None:
    original_df = pd.read_csv(comparison_files / "daily_average.csv", index_col=0)
    original_df.index = original_df.index.astype(object)
    test_df = hd.daily_average(merged_df)
    pd.testing.assert_frame_equal(original_df, test_df)


def test_daily_std_dev(merged_df: pd.DataFrame, comparison_files: Path) -> None:
    original_df = pd.read_csv(comparison_files / "daily_std_dev.csv", index_col=0)
    original_df.index = original_df.index.astype(object)
    test_df = hd.daily_std_dev(merged_df)
    pd.testing.assert_frame_equal(original_df, test_df)


def test_daily_std_error(merged_df: pd.DataFrame, comparison_files: Path) -> None:
    original_df = pd.read_csv(comparison_files / "daily_std_error.csv", index_col=0)
    original_df.index = original_df.index.astype(object)
    test_df = hd.daily_std_error(merged_df)
    pd.testing.assert_frame_equal(original_df, test_df)


def test_monthly_average(merged_df: pd.DataFrame, comparison_files: Path) -> None:
    original_df = pd.read_csv(comparison_files / "monthly_average.csv", index_col=0)
    original_df.index = np.array(
        ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"],
        dtype=object,
    )
    test_df = hd.monthly_average(merged_df)
    pd.testing.assert_frame_equal(original_df, test_df)


def test_monthly_std_dev(merged_df: pd.DataFrame, comparison_files: Path) -> None:
    original_df = pd.read_csv(comparison_files / "monthly_std_dev.csv", index_col=0)
    original_df.index = np.array(
        ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"],
        dtype=object,
    )
    test_df = hd.monthly_std_dev(merged_df)
    pd.testing.assert_frame_equal(original_df, test_df)


def test_monthly_std_error(merged_df: pd.DataFrame, comparison_files: Path) -> None:
    original_df = pd.read_csv(comparison_files / "monthly_std_error.csv", index_col=0)
    original_df.index = np.array(
        ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"],
        dtype=object,
    )
    test_df = hd.monthly_std_error(merged_df)
    pd.testing.assert_frame_equal(original_df, test_df)


def test_remove_nan_df() -> None:
    rng = np.random.default_rng()
    data = rng.random((15, 2))
    data[0, 0] = data[1, 1] = np.nan
    data[2, 0] = data[3, 1] = np.inf
    data[4, 0] = data[5, 1] = 0
    data[6, 0] = data[7, 1] = -0.1

    test_df = hd.remove_nan_df(
        pd.DataFrame(data=data, index=pd.date_range("1980-01-01", periods=15))
    )
    original_df = pd.DataFrame(data=data[8:, :], index=pd.date_range("1980-01-09", periods=7))
    # The frequency gets dropped when slicing/filtering, so don't compare it
    pd.testing.assert_frame_equal(original_df, test_df, check_freq=False)
