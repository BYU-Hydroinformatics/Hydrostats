from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.patches import Rectangle

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


@pytest.mark.parametrize("z_norm", [False, True])
@pytest.mark.parametrize("legend", [None, ("Simulated", "Observed")])
@pytest.mark.parametrize("prob_dens", [False, True])
def test_hist_shared_bins(z_norm: bool, legend: tuple[str, str] | None, prob_dens: bool) -> None:
    sim = np.array([0, 1, 4])
    obs = np.array([1, 3, 5])
    fig = hv.hist(
        sim_array=sim,
        obs_array=obs,
        num_bins=4,
        z_norm=z_norm,
        legend=legend,
        prob_dens=prob_dens,
    )

    if z_norm:
        expected_edges = np.linspace(-np.sqrt(1.5) - 0.01, 7 / np.sqrt(26) + 0.01, 4)
    else:
        expected_edges = np.linspace(-0.01, 5.01, 4)
    expected_counts = np.array([2, 0, 1, 1, 1, 1])
    if prob_dens:
        expected_counts = expected_counts / (3 * np.diff(expected_edges)[0])

    ax = fig.axes[0]
    for patch, left, width, height in zip(
        ax.patches,
        np.tile(expected_edges[:-1], 2),
        np.tile(np.diff(expected_edges), 2),
        expected_counts,
        strict=True,
    ):
        assert isinstance(patch, Rectangle)
        assert patch.get_x() == pytest.approx(left)
        assert patch.get_width() == pytest.approx(width)
        assert patch.get_height() == pytest.approx(height)
    assert (ax.get_legend() is not None) == (legend is not None)
    plt.close(fig)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("z_norm", [False, True])
def test_hist_bin_precision(
    dtype: type[np.float16] | type[np.float32] | type[np.float64],
    z_norm: bool,
) -> None:
    sim = np.array([0.1, 0.3, 0.7, 1.3], dtype=dtype)
    obs = np.array([0.2, 0.4, 0.8, 1.2], dtype=dtype)
    fig = hv.hist(sim_array=sim, obs_array=obs, num_bins=4, z_norm=z_norm, prob_dens=True)

    if z_norm:
        sim = (sim - np.mean(sim)) / np.std(sim)
        obs = (obs - np.mean(obs)) / np.std(obs)
    expected_edges = np.linspace(
        np.min([np.min(sim), np.min(obs)]) - 0.01,
        np.max([np.max(sim), np.max(obs)]) + 0.01,
        4,
    )
    expected_heights = np.concatenate(
        [np.histogram(data, bins=expected_edges, density=True)[0] for data in (sim, obs)]
    )
    heights = []
    for patch in fig.axes[0].patches:
        assert isinstance(patch, Rectangle)
        heights.append(patch.get_height())
    np.testing.assert_array_equal(heights, expected_heights)
    plt.close(fig)


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


@pytest.mark.parametrize("dtype", [np.int64, np.float64])
def test_scatter_best_fit_arrays(dtype: type[np.int64] | type[np.float64]) -> None:
    sim = np.array([1, 2, 3], dtype=dtype)
    obs = np.array([3, 5, 7], dtype=dtype)
    fig = hv.scatter(sim_array=sim, obs_array=obs, best_fit=True)

    fit_line = fig.axes[0].get_lines()[1]
    np.testing.assert_allclose(np.asarray(fit_line.get_xdata(), dtype=float), [0, 1.5, 3])
    np.testing.assert_allclose(np.asarray(fit_line.get_ydata(), dtype=float), [1, 4, 7])
    plt.close(fig)


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
