from pathlib import Path

import pandas as pd

import hydrostats.analyze as ha
import hydrostats.data as hd
import hydrostats.metrics as he


def test_make_table(merged_df: pd.DataFrame) -> None:
    my_metrics = ["MAE", "r2", "NSE", "KGE (2012)"]
    seasonal = [["01-01", "03-31"], ["04-01", "06-30"], ["07-01", "09-30"], ["10-01", "12-31"]]

    # Using the Function
    table = ha.make_table(merged_df, my_metrics, seasonal, remove_neg=True, remove_zero=True)

    # Calculating manually to test
    metric_functions = [he.mae, he.r_squared, he.nse, he.kge_2012]

    season0 = merged_df
    season1 = hd.seasonal_period(merged_df, daily_period=("01-01", "03-31"))
    season2 = hd.seasonal_period(merged_df, daily_period=("04-01", "06-30"))
    season3 = hd.seasonal_period(merged_df, daily_period=("07-01", "09-30"))
    season4 = hd.seasonal_period(merged_df, daily_period=("10-01", "12-31"))

    all_seasons = [season0, season1, season2, season3, season4]

    test_list = []
    for season in all_seasons:
        temp_list = []
        for metric in metric_functions:
            sim = season.iloc[:, 0].to_numpy()
            obs = season.iloc[:, 1].to_numpy()
            temp_list.append(metric(sim, obs, remove_neg=True, remove_zero=True))
        test_list.append(temp_list)

    test_table = pd.DataFrame(
        test_list,
        index=[
            "Full Time Series",
            "January-01:March-31",
            "April-01:June-30",
            "July-01:September-30",
            "October-01:December-31",
        ],
        columns=["MAE", "r2", "NSE", "KGE (2012)"],
    )

    pd.testing.assert_frame_equal(test_table, table)


def test_lag_analysis(merged_df: pd.DataFrame, comparison_files: Path) -> None:
    time_lag_df, summary_df = ha.time_lag(
        merged_df, metrics=["r2", "RMSE", "KGE (2012)", "NSE"]
    )

    time_lag_df_original = pd.read_csv(comparison_files / "time_lag_df.csv", index_col=0)
    summary_df_original = pd.read_csv(comparison_files / "summary_df.csv", index_col=0)

    pd.testing.assert_frame_equal(time_lag_df, time_lag_df_original, rtol=1e-5, atol=1e-8)
    pd.testing.assert_frame_equal(summary_df, summary_df_original, rtol=1e-5, atol=1e-8)
