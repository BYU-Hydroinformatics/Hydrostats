from __future__ import annotations

from io import BytesIO
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import hydrostats.data as hd
import hydrostats.visual as hv


def _png_from_current_figure() -> np.ndarray:
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img = mpimg.imread(buf)
    buf.close()
    return img


def test_plot_full1(merged_df: pd.DataFrame, baseline_plots: Path):
    hv.plot(
        merged_data_df=merged_df,
        title="Hydrograph of Entire Time Series",
        linestyles=["r-", "k-"],
        legend=("SFPT", "GLOFAS"),
        labels=["Datetime", "Streamflow (cfs)"],
        metrics=["ME", "NSE", "SA"],
        grid=True,
    )
    img_test = _png_from_current_figure()
    img_original = mpimg.imread(baseline_plots / "plot_full1.png")
    assert np.allclose(img_test, img_original)


def test_plot_seasonal(merged_df: pd.DataFrame, baseline_plots: Path):
    daily_avg_df = hd.daily_average(df=merged_df)
    daily_std_error = hd.daily_std_error(merged_data=merged_df)

    hv.plot(
        merged_data_df=daily_avg_df,
        title="Daily Average Streamflow (Standard Error)",
        legend=("SFPT", "GLOFAS"),
        x_season=True,
        labels=["Datetime", "Streamflow (csm)"],
        linestyles=["r-", "k-"],
        fig_size=(14, 8),
        ebars=daily_std_error,
        ecolor=("r", "k"),
        tight_xlim=True,
    )
    img_test = _png_from_current_figure()
    img_original = mpimg.imread(baseline_plots / "plot_seasonal.png")
    assert np.allclose(img_test, img_original)


def test_hist_df(merged_df: pd.DataFrame, baseline_plots: Path):
    hv.hist(
        merged_data_df=merged_df,
        num_bins=100,
        title="Histogram of Streamflows",
        legend=("SFPT", "GLOFAS"),
        labels=("Bins", "Frequency"),
        grid=True,
    )
    img_test = _png_from_current_figure()
    img_original = mpimg.imread(baseline_plots / "hist1.png")
    assert np.allclose(img_test, img_original)


def test_hist_arrays(merged_df: pd.DataFrame, baseline_plots: Path):
    sim_array = merged_df.iloc[:, 0].values
    obs_array = merged_df.iloc[:, 1].values

    hv.hist(
        sim_array=sim_array,
        obs_array=obs_array,
        num_bins=100,
        title="Histogram of Streamflows",
        legend=("SFPT", "GLOFAS"),
        labels=("Bins", "Frequency"),
        grid=True,
    )
    img_test = _png_from_current_figure()
    img_original = mpimg.imread(baseline_plots / "hist1.png")
    assert np.allclose(img_test, img_original)


def test_hist_znorm(merged_df: pd.DataFrame, baseline_plots: Path):
    hv.hist(
        merged_data_df=merged_df,
        num_bins=100,
        title="Histogram of Streamflows",
        labels=("Bins", "Frequency"),
        grid=True,
        z_norm=True,
        legend=None,
        prob_dens=True,
    )
    img_test = _png_from_current_figure()
    img_original = mpimg.imread(baseline_plots / "hist_znorm.png")
    assert np.allclose(img_test, img_original)


def test_hist_error(merged_df: pd.DataFrame):
    sim_array = merged_df.iloc[:, 0].values
    with pytest.raises(RuntimeError):
        hv.hist(merged_data_df=merged_df, sim_array=sim_array)


def test_scatter(merged_df: pd.DataFrame, baseline_plots: Path):
    hv.scatter(
        merged_data_df=merged_df,
        grid=True,
        title="Scatter Plot (Normal Scale)",
        labels=("SFPT", "GLOFAS"),
        best_fit=True,
    )
    img_test = _png_from_current_figure()
    img_original = mpimg.imread(baseline_plots / "scatter.png")
    assert np.allclose(img_test, img_original)


def test_scatterlog(merged_df: pd.DataFrame, baseline_plots: Path):
    sim_array = merged_df.iloc[:, 0].values
    obs_array = merged_df.iloc[:, 1].values
    hv.scatter(
        sim_array=sim_array,
        obs_array=obs_array,
        grid=True,
        title="Scatter Plot (Log-Log Scale)",
        labels=("SFPT", "GLOFAS"),
        line45=True,
        metrics=["ME", "KGE (2012)"],
        log_scale=True,
    )
    img_test = _png_from_current_figure()
    img_original = mpimg.imread(baseline_plots / "scatterlog.png")
    assert np.allclose(img_test, img_original)


def test_qq_plot(merged_df: pd.DataFrame, baseline_plots: Path):
    hv.qqplot(
        merged_data_df=merged_df,
        title="Quantile-Quantile Plot of Data",
        xlabel="SFPT Data Quantiles",
        ylabel="GLOFAS Data Quantiles",
        legend=True,
        figsize=(8, 6),
    )
    img_test = _png_from_current_figure()
    img_original = mpimg.imread(baseline_plots / "qqplot.png")
    assert np.allclose(img_test, img_original)


def test_qq_plot2(merged_df: pd.DataFrame, baseline_plots: Path):
    sim_array = merged_df.iloc[:, 0].values
    obs_array = merged_df.iloc[:, 1].values
    hv.qqplot(
        sim_array=sim_array,
        obs_array=obs_array,
        title="Quantile-Quantile Plot of Data",
        xlabel="SFPT Data Quantiles",
        ylabel="GLOFAS Data Quantiles",
        figsize=(8, 6),
    )
    img_test = _png_from_current_figure()
    img_original = mpimg.imread(baseline_plots / "qqplot2.png")
    assert np.allclose(img_test, img_original)
