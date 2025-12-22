from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pytest

import hydrostats.data as hd
import hydrostats.visual as hv

if TYPE_CHECKING:
    import pandas as pd


@pytest.mark.mpl_image_compare
def test_plot_full1(merged_df: "pd.DataFrame") -> plt.Figure:
    return hv.plot(
        merged_data_df=merged_df,
        title="Hydrograph of Entire Time Series",
        linestyles=("r-", "k-"),
        legend=("SFPT", "GLOFAS"),
        labels=["Datetime", "Streamflow (cfs)"],
        metrics=["ME", "NSE", "SA"],
        grid=True,
    )


@pytest.mark.mpl_image_compare
def test_plot_seasonal(merged_df: "pd.DataFrame") -> plt.Figure:
    daily_avg_df = hd.daily_average(df=merged_df)
    daily_std_error = hd.daily_std_error(merged_data=merged_df)

    return hv.plot(
        merged_data_df=daily_avg_df,
        title="Daily Average Streamflow (Standard Error)",
        legend=("SFPT", "GLOFAS"),
        x_season=True,
        labels=["Datetime", "Streamflow (csm)"],
        linestyles=("r-", "k-"),
        fig_size=(14, 8),
        ebars=daily_std_error,
        ecolor=("r", "k"),
        tight_xlim=True,
    )


@pytest.mark.mpl_image_compare
def test_hist_df(merged_df: "pd.DataFrame") -> plt.Figure:
    return hv.hist(
        merged_data_df=merged_df,
        num_bins=100,
        title="Histogram of Streamflows",
        legend=("SFPT", "GLOFAS"),
        labels=("Bins", "Frequency"),
        grid=True,
    )


@pytest.mark.mpl_image_compare
def test_hist_arrays(merged_df: "pd.DataFrame") -> plt.Figure:
    sim_array = merged_df.iloc[:, 0].to_numpy()
    obs_array = merged_df.iloc[:, 1].to_numpy()

    hv.hist(
        sim_array=sim_array,
        obs_array=obs_array,
        num_bins=100,
        title="Histogram of Streamflows",
        legend=("SFPT", "GLOFAS"),
        labels=("Bins", "Frequency"),
        grid=True,
    )
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_hist_znorm(merged_df: "pd.DataFrame") -> plt.Figure:
    return hv.hist(
        merged_data_df=merged_df,
        num_bins=100,
        title="Histogram of Streamflows",
        labels=("Bins", "Frequency"),
        grid=True,
        z_norm=True,
        legend=None,
        prob_dens=True,
    )


def test_hist_error(merged_df: "pd.DataFrame") -> None:
    sim_array = merged_df.iloc[:, 0].to_numpy()
    with pytest.raises(ValueError, match=r"You must either pass in a dataframe or two arrays."):
        hv.hist(merged_data_df=merged_df, sim_array=sim_array)


@pytest.mark.mpl_image_compare
def test_scatter(merged_df: "pd.DataFrame") -> plt.Figure:
    return hv.scatter(
        merged_data_df=merged_df,
        grid=True,
        title="Scatter Plot (Normal Scale)",
        labels=("SFPT", "GLOFAS"),
        best_fit=True,
    )


@pytest.mark.mpl_image_compare
def test_scatterlog(merged_df: "pd.DataFrame") -> plt.Figure:
    sim_array = merged_df.iloc[:, 0].to_numpy()
    obs_array = merged_df.iloc[:, 1].to_numpy()
    return hv.scatter(
        sim_array=sim_array,
        obs_array=obs_array,
        grid=True,
        title="Scatter Plot (Log-Log Scale)",
        labels=("SFPT", "GLOFAS"),
        line45=True,
        metrics=["ME", "KGE (2012)"],
        log_scale=True,
    )


@pytest.mark.mpl_image_compare
def test_qq_plot(merged_df: "pd.DataFrame") -> plt.Figure:
    return hv.qqplot(
        merged_data_df=merged_df,
        title="Quantile-Quantile Plot of Data",
        xlabel="SFPT Data Quantiles",
        ylabel="GLOFAS Data Quantiles",
        legend=True,
        figsize=(8, 6),
    )


@pytest.mark.mpl_image_compare
def test_qq_plot2(merged_df: "pd.DataFrame") -> plt.Figure:
    sim_array = merged_df.iloc[:, 0].to_numpy()
    obs_array = merged_df.iloc[:, 1].to_numpy()
    return hv.qqplot(
        sim_array=sim_array,
        obs_array=obs_array,
        title="Quantile-Quantile Plot of Data",
        xlabel="SFPT Data Quantiles",
        ylabel="GLOFAS Data Quantiles",
        figsize=(8, 6),
    )
