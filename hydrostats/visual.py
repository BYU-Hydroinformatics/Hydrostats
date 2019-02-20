# python 3.6
# -*- coding: utf-8 -*-
"""
The visual module contains different plotting functions for time series visualization. It allows
users to plot hydrographs, scatter plots, histograms, and quantile-quantile (qq) plots to visualize
time series data. In some of the visualization functions, metrics can be added to the plots for a
more complete summary of the data.
"""
from __future__ import division
from hydrostats.metrics import function_list, metric_abbr
from HydroErr.HydroErr import treat_values
import numpy as np
import matplotlib.pyplot as plt
import calendar
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

__all__ = ['plot', 'hist', 'scatter', 'qqplot']


def plot(merged_data_df, legend=('Simulated Data', 'Observed Data'), metrics=None, grid=False, title=None,
         x_season=False, labels=None, linestyles=('ro', 'b^'), tight_xlim=False, fig_size=(10, 6),
         text_adjust=(-0.35, 0.75), plot_adjust=0.27, transparency=0.5, ebars=None, ecolor=None,
         markersize=2, errorevery=1, markevery=1):
    """
    Create a comparison time series line plot of simulated and observed time series data.

    The time series plot is a function that is available for viewing two times series plotted side
    by side vs. time. Goodness of fit metrics can also be viewed on the plot to compare the two
    time series.

    Parameters
    ----------
    merged_data_df: DataFrame
        DataFrame must contain datetime index and floating point type numbers in the two columns.
        The left columns must be simulated data and the right column observed data.

    legend: tuple of str
        Adds a Legend in the 'best' location determined by matplotlib. The entries in the tuple describe the left and
        right columns of the merged_data_df data.

    metrics: list of str
        Adds Metrics to the left side of the plot. Any metric from the Hydrostats library can
        be added to the plot as the abbreviation of the function. The entries must be in a list.
        (e.g. ['ME', 'r2', 'KGE (2012)']).

    grid: bool
        If True, adds a grid to the plot.

    title: str
        If given, adds a title to the plot.

    x_season: bool
        If True, the x-axis ticks will be monthly. This is a useful feature when plotting seasonal
        time series comparisons (e.g. daily averages).

    labels: list of str
        List of two str type inputs specifying x-axis labels and y-axis labels, respectively.

    linestyles: list of str
        List of two string type inputs thet will change the linestyle of the predicted and
        recorded data, respectively. Linestyle references can be found in
        :ref:`matplotlib_linestyles`.

    tight_xlim: bool
        If true, will set the padding to zero for the lines in the line plot.

    fig_size: tuple of floats
        Tuple of length two that specifies the horizontal and vertical lengths of the plot in
        inches, respectively.

    text_adjust: tuple
        Tuple of length two with float type inputs indicating the relative position of the text
        (x-coordinate, y-coordinate) when adding metrics to the plot.

    plot_adjust: float
        Specifies the relative position to shift the plot the the right when adding metrics to the
        plot.

    transparency: float
        Value between 0 to 1 indicating the transparency of the two lines that are plotted and
        error bars if they are plotted, lower means more transparent.

    ebars: DataFrame
        DataFrame must contain datetime index and two columns of data that specify the error
        of the plots, with the simulated error on the left and the observed error on the right.
        These dataframes can be created with the
        :doc:`daily_std_error <hydrostats.data.daily_std_error>`,
        :doc:`daily_std_dev <hydrostats.data.daily_std_dev>`,
        :doc:`monthly_std_error <hydrostats.data.monthly_std_error>`, and
        :doc:`monthly_std_dev <hydrostats.data.monthly_std_dev>` functions.

    ecolor: tuple of str
        Tuple of two sting type inputs specifying the colors of the errorbars (e.g. ['r', 'k'] would
        make red and black errorbars on the simulated and observed data, respectively).

    markersize: float
        Indicates the size of the markers on the plot, if markers are used.

    errorevery: int
        Specifies how often to put error bars on the plot.

    markevery: int
        Specifies how often to put markers on the plot if markers are used.

    Returns
    -------
    fig : Matplotlib figure instance
        A matplotlib figure handle is returned, which can be viewed with the matplotlib.pyplot.show() command.

    Examples
    --------

    In this example two models are compared.

    >>> import hydrostats.data as hd
    >>> import hydrostats.visual as hv
    >>> import matplotlib.pyplot as plt

    >>> sfpt_url = r'https://github.com/waderoberts123/Hydrostats/raw/master/Sample_data/sfpt_data/magdalena-calamar_interim_data.csv'
    >>> glofas_url = r'https://github.com/waderoberts123/Hydrostats/raw/master/Sample_data/GLOFAS_Data/magdalena-calamar_ECMWF_data.csv'

    >>> merged_df = hd.merge_data(sfpt_url, glofas_url, column_names=('SFPT', 'GLOFAS'))
    >>> seasonal_df = hd.seasonal_period(merged_df, ('04-01', '07-31'), time_range=('1986-01-01', '1992-12-31'))
    >>> daily_avg_df = hd.daily_average(merged_data=merged_df)  # Seasonal Daily Averages
    >>> daily_std_error = hd.daily_std_error(merged_data=merged_df)  # Seasonal Daily Standard Deviation

    The entire timeseries is plotted below

    >>> plot(merged_data_df=merged_df,
    >>>      title='Hydrograph of Entire Time Series',
    >>>      linestyles=['r-', 'k-'],
    >>>      legend=('SFPT', 'GLOFAS'),
    >>>      labels=['Datetime', 'Streamflow (cfs)'],
    >>>      metrics=['ME', 'NSE', 'SA'],
    >>>      grid=True)
    >>> plt.show()

    .. image:: /Figures/plot_full1.png

    The seasonal averages with standard error bars is plotted below

    >>> plot(merged_data_df=daily_avg_df,
    >>>      title='Daily Average Streamflow (Standard Error)',
    >>>      legend=('SFPT', 'GLOFAS'),
    >>>      x_season=True,
    >>>      labels=['Datetime', 'Streamflow (csm)'],
    >>>      linestyles=['r-', 'k-'],
    >>>      fig_size=(14, 8),
    >>>      ebars=daily_std_error,
    >>>      ecolor=('r', 'k'),
    >>>      tight_xlim=True
    >>>      )
    >>> plt.show()

    .. image:: /Figures/plot_seasonal.png

    """
    fig = plt.figure(figsize=fig_size, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)

    # Setting Variable for the simulated data, observed data, and time stamps
    sim = merged_data_df.iloc[:, 0].values
    obs = merged_data_df.iloc[:, 1].values
    time = merged_data_df.index.values

    # Plotting the Data
    if ebars is None:
        plt.plot(time, sim, linestyles[0], markersize=markersize,
                 label=legend[0], alpha=transparency, markevery=markevery)
        plt.plot(time, obs, linestyles[1], markersize=markersize,
                 label=legend[1], alpha=transparency, markevery=markevery)
        plt.legend(fontsize=14)
    elif ebars is not None:
        plt.errorbar(x=time, y=sim, yerr=ebars.iloc[:, 0].values,
                     fmt=linestyles[0], markersize=markersize, label=legend[0], alpha=transparency, ecolor=ecolor[0],
                     markevery=markevery, errorevery=errorevery)
        plt.errorbar(x=time, y=obs, yerr=ebars.iloc[:, 1].values,
                     fmt=linestyles[1], markersize=markersize, label=legend[1], alpha=transparency, ecolor=ecolor[1],
                     markevery=markevery, errorevery=errorevery)
        plt.legend(fontsize=14)

    # Adjusting the plot if user wants tight x axis limits
    if tight_xlim:
        plt.xlim(time[0], time[-1])

    # Changing the X axis for a better seasonal plot if seasonal and adjusting tick sizes
    if x_season:
        seasons = calendar.month_abbr[1:13]
        day_month = np.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        day_month_cum = np.cumsum(day_month)
        fractions = day_month_cum[:11] / 366
        fractions = np.insert(fractions, 0, 0)
        index = np.rint(fractions * len(time)).astype(np.integer)
        plt.xticks(time[index], seasons, fontsize=14, rotation=45)
    else:
        plt.xticks(fontsize=14, rotation=45)

    plt.yticks(fontsize=14)

    # Placing Labels if requested
    if labels:
        # Plotting Labels
        plt.xlabel(labels[0], fontsize=18)
        plt.ylabel(labels[1], fontsize=18)
    if title:
        title_dict = {'family': 'sans-serif',
                      'color': 'black',
                      'weight': 'normal',
                      'size': 20,
                      }
        ax.set_title(label=title, fontdict=title_dict, pad=25)

    # Placing a grid if requested
    if grid:
        plt.grid(True)

    # Fixes issues with parts of plot being cut off
    plt.tight_layout()

    # Placing Metrics on the Plot if requested
    if metrics:
        function_list_str = metric_abbr

        assert isinstance(metrics, list)

        for metric in metrics:
            assert metric in function_list_str

        index = []
        for metric in metrics:
            index.append(function_list_str.index(metric))

        selected_metrics = []
        for i in index:
            selected_metrics.append(
                function_list_str[i] + '=' + str(round(function_list[i](sim, obs), 3)))

        formatted_selected_metrics = ''
        for i in selected_metrics:
            formatted_selected_metrics += i + '\n'

        font = {'family': 'sans-serif',
                'weight': 'normal',
                'size': 14}
        plt.text(text_adjust[0], text_adjust[1], formatted_selected_metrics, ha='left', va='center',
                 transform=ax.transAxes, fontdict=font)

        plt.subplots_adjust(left=plot_adjust)

    return fig


def hist(merged_data_df=None, sim_array=None, obs_array=None, num_bins=100, z_norm=False,
         legend=('Simulated', 'Observed'), grid=False, title=None, labels=None, prob_dens=False,
         figsize=(12, 6)):
    """Plots a histogram comparing simulated and observed data.

    The histogram plot is a function that is available for comparing the histograms of two time
    series. Data can be Z-score normalized as well as fit in a probability density function.

    Parameters
    ----------
    merged_data_df: DataFrame
        Dataframe must contain a datetime type index and floating point type numbers in two
        columns. The left column must be simulated data and the right column must be observed data.
        If given, sim_array and obs_array must be None.

    sim_array: 1D ndarray
        Array of simulated data. If given, merged_data_df parameter must be None and obs_array must
        be given.

    obs_array: 1D ndarray
        Array of observed data. If given, merged_data_df parameter must be None and sim_array must
        be given.

    num_bins: int
        Specifies the number of bins in the histogram.

    z_norm: bool
        If True, the data will be Z-score normalized.

    legend: tuple of str
        Tuple of length two with str inputs. Adds a Legend in the 'best' location determined by
        matplotlib. The entries in the tuple label the simulated and observed data
        (e.g. ['Simulated Data', 'Predicted Data']).

    grid: bool
        If True, adds a grid to the plot.

    title: str
        If given, sets the title of the plot.

    labels: tuple of str
        Tuple of two string type objects to set the x-axis labels and y-axis labels, respectively.

    prob_dens: bool
        If True, normalizes both histograms to form a probability density, i.e., the area
        (or integral) under each histogram will sum to 1.

    figsize: tuple of float
        Tuple of length two that specifies the horizontal and vertical lengths of the plot in
        inches, respectively.

    Returns
    -------
    fig : Matplotlib figure instance
        A matplotlib figure handle is returned, which can be viewed with the matplotlib.pyplot.show() command.

    Examples
    --------

    In this example the histograms of two models are compared to check their distributions

    >>> import hydrostats.data as hd
    >>> import hydrostats.visual as hv
    >>> import matplotlib.pyplot as plt

    >>> sfpt_url = r'https://github.com/waderoberts123/Hydrostats/raw/master/Sample_data/sfpt_data/magdalena-calamar_interim_data.csv'
    >>> glofas_url = r'https://github.com/waderoberts123/Hydrostats/raw/master/Sample_data/GLOFAS_Data/magdalena-calamar_ECMWF_data.csv'
    >>> merged_df = hd.merge_data(sfpt_url, glofas_url, column_names=('SFPT', 'GLOFAS'))

    The histogram with 100 bins is plotted below

    >>> hist(merged_data_df=merged_df,
    >>>      num_bins=100,
    >>>      title='Histogram of Streamflows',
    >>>      legend=('SFPT', 'GLOFAS'),
    >>>      labels=('Bins', 'Frequency'),
    >>>      grid=True)
    >>> plt.show()

    .. image:: /Figures/hist1.png

    """
    # Getting the fig and axis handles
    fig, ax1 = plt.subplots(figsize=figsize)

    if merged_data_df is not None and sim_array is None and obs_array is None:
        # Creating a simulated and observed data array
        sim = merged_data_df.iloc[:, 0].values
        obs = merged_data_df.iloc[:, 1].values
    elif sim_array is not None and obs_array is not None and merged_data_df is None:
        sim = sim_array
        obs = obs_array
    else:
        raise RuntimeError("You must either pass in a dataframe or two arrays.")

    if z_norm:
        # Calculating the Z Scores for the simulated data
        sim_mean = np.mean(sim)
        sim_std_dev = np.std(sim)
        # The z scores override sim from before because we are plotting the Z scores
        sim = ((sim - sim_mean) / sim_std_dev)

        # Calculating the Z Scores for the observed data
        obs_mean = np.mean(obs)
        obs_std_dev = np.std(obs)
        # The z scores override obs from before because we are plotting the Z scores
        obs = ((obs - obs_mean) / obs_std_dev)

        # Finding the maximum and minimum Z scores
        sim_max = np.max(sim)
        sim_min = np.min(sim)
        obs_max = np.max(obs)
        obs_min = np.min(obs)

        total_max = np.max([sim_max, obs_max])
        total_min = np.min([sim_min, obs_min])

        # Creating the bins based on the max and min
        bins = np.linspace(total_min - 0.01, total_max + 0.01, num_bins)
    else:
        # Calculating the max and min of both data sets
        sim_max = np.max(sim)
        sim_min = np.min(sim)
        obs_max = np.max(obs)
        obs_min = np.min(obs)

        total_max = np.max([sim_max, obs_max])
        total_min = np.min([sim_min, obs_min])

        # Creating the bins based on the max and min
        bins = np.linspace(total_min - 0.01, total_max + 0.01, num_bins)

    if legend is None:
        # Plotting the data without the legend
        ax1.hist(sim, bins, alpha=0.5, edgecolor='black', linewidth=0.5, density=prob_dens)
        ax1.hist(obs, bins, alpha=0.5, edgecolor='black', linewidth=0.5, density=prob_dens)
    else:
        # Plotting the data with the legend
        ax1.hist(sim, bins, alpha=0.5, label=legend[0], edgecolor='black', linewidth=0.5, density=prob_dens)
        ax1.hist(obs, bins, alpha=0.5, label=legend[1], edgecolor='black', linewidth=0.5, density=prob_dens)
        ax1.legend(framealpha=1)

    # Setting the x and y tick size
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Creating a grid
    if grid:
        plt.grid(True)

    # Creating a title
    if title:
        title_dict = {'family': 'sans-serif',
                      'color': 'black',
                      'weight': 'normal',
                      'size': 20,
                      }
        ax1.set_title(label=title, fontdict=title_dict, pad=15)

    # Creating x and y axis labels
    if labels:
        plt.xlabel(labels[0], fontsize=18)
        plt.ylabel(labels[1], fontsize=18)

    # Assuring a tight layout
    plt.tight_layout()

    return fig


def scatter(merged_data_df=None, sim_array=None, obs_array=None, grid=False, title=None, labels=None, best_fit=False,
            marker_style='ko', metrics=None, log_scale=False, line45=False, figsize=(12, 8)):
    """Creates a scatter plot of the observed and simulated data.

    Parameters
    ----------
    merged_data_df: DataFrame
        Dataframe must contain a datetime type index and floating point type numbers in two
        columns. The left column must be simulated data and the right column must be observed data.
        If given, sim_array and obs_array must be None.

    sim_array: 1D ndarray
        Array of simulated data. If given, merged_data_df parameter must be None and obs_array must
        be given.

    obs_array: 1D ndarray
        Array of observed data. If given, merged_data_df parameter must be None and sim_array must
        be given.

    grid: bool
        If True, adds a grid to the plot.

    title: str
        If given, sets the title of the plot.

    labels: tuple of str
        Tuple of two string type objects to set the x-axis labels and y-axis labels, respectively.

    best_fit: bool
        If True, adds a best linear regression line on the graph
        with the equation for the line in the legend.

    marker_style: str
        If give, changes the markerstyle of the points on the scatter plot. Matplotlib styling
        guides are found in :ref:`matplotlib_linestyles`.

    metrics: list of str
        Adds Metrics to the left side of the plot. Any metric from the Hydrostats library can
        be added to the plot as the abbreviation of the function. The entries must be in a list.
        (e.g. ['ME', 'r2', 'KGE (2012)']).

    log_scale: bool
        If True, log-log scale will be used on the scatter plot.

    line45: bool
        IF Trre, adds a 45 degree line to the plot and the legend.

    figsize: tuple of float
        Tuple of length two that specifies the horizontal and vertical lengths of the plot in
        inches, respectively.

    Returns
    -------
    fig : Matplotlib figure instance
        A matplotlib figure handle is returned, which can be viewed with the matplotlib.pyplot.show() command.


    Examples
    --------

    A scatter plot is created in this example comparing two models.

    >>> import hydrostats.data as hd
    >>> import hydrostats.visual as hv
    >>> import matplotlib.pyplot as plt

    >>> sfpt_url = r'https://github.com/waderoberts123/Hydrostats/raw/master/Sample_data/sfpt_data/magdalena-calamar_interim_data.csv'
    >>> glofas_url = r'https://github.com/waderoberts123/Hydrostats/raw/master/Sample_data/GLOFAS_Data/magdalena-calamar_ECMWF_data.csv'
    >>> merged_df = hd.merge_data(sfpt_url, glofas_url, column_names=('SFPT', 'GLOFAS'))

    >>> sim_array = merged_df.iloc[:, 0].values
    >>> obs_array = merged_df.iloc[:, 1].values

    >>> scatter(merged_data_df=merged_df, grid=True, title='Scatter Plot (Normal Scale)',
    >>>         labels=('SFPT', 'GLOFAS'), best_fit=True)
    >>> plt.show()

    .. image:: /Figures/scatter.png

    Arrays can be used as well in the parameters, as demonstrated below.

    >>> scatter(sim_array=sim_array, obs_array=obs_array, grid=True, title='Scatter Plot (Log-Log Scale)',
    >>>         labels=('SFPT', 'GLOFAS'), line45=True, metrics=['ME', 'KGE (2012)'])
    >>> plt.show()

    .. image:: /Figures/scatterlog.png

    """
    fig = plt.figure(figsize=figsize, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)

    if merged_data_df is not None and sim_array is None and obs_array is None:
        # Creating a simulated and observed data array
        sim = merged_data_df.iloc[:, 0].values
        obs = merged_data_df.iloc[:, 1].values
    elif sim_array is not None and obs_array is not None and merged_data_df is None:
        sim = sim_array
        obs = obs_array
    else:
        raise RuntimeError("You must either pass in a dataframe or two arrays.")

    max_both = max([np.max(sim), np.max(obs)])

    if not log_scale:
        plt.plot(sim, obs, marker_style)
    else:
        plt.loglog(sim, obs, marker_style)

    if line45:
        plt.plot(np.arange(0, int(max_both) + 1), np.arange(0, int(max_both) + 1), 'r--', label='45$^\circ$ Line')

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    if grid:
        plt.grid(True)

    if title:
        plt.title(title, fontsize=20)

    if labels:
        plt.xlabel(labels[0], fontsize=18)
        plt.ylabel(labels[1], fontsize=18)

    if best_fit:
        # Getting a polynomial fit and defining a function with it
        p = np.polyfit(sim, obs, 1)
        f = np.poly1d(p)

        # Calculating new x's and y's
        x_new = np.linspace(0, sim.max(), sim.size)
        y_new = f(x_new)

        # Formatting the best fit equation to be able to display in latex
        equation = "{} x + {}".format(np.round(p[0], 4), np.round(p[1], 4))

        # Plotting the best fit line with the equation as a legend in latex
        plt.plot(x_new, y_new, 'k', label="${}$".format(equation))

    if line45 or best_fit:
        plt.legend(fontsize=12)

    if metrics is not None:
        function_list_str = metric_abbr

        assert isinstance(metrics, list)
        for metric in metrics:
            assert metric in function_list_str
        index = []
        for metric in metrics:
            index.append(function_list_str.index(metric))
        selected_metrics = []
        for i in index:
            selected_metrics.append(function_list_str[i] + '=' +
                                    str(round(function_list[i](sim, obs), 3)))
        formatted_selected_metrics = ''
        for i in selected_metrics:
            formatted_selected_metrics += i + '\n'
        font = {'family': 'sans-serif',
                'weight': 'normal',
                'size': 14}
        plt.text(-0.35, 0.75, formatted_selected_metrics, ha='left', va='center', transform=ax.transAxes, fontdict=font)
        plt.subplots_adjust(left=0.25)

    return fig


def qqplot(merged_data_df=None, sim_array=None, obs_array=None, interpolate='linear', title=None,
           xlabel='Simulated Data Quantiles', ylabel='Observed Data Quantiles', legend=False, replace_nan=None,
           replace_inf=None, remove_neg=False, remove_zero=False, figsize=(12, 8)):
    """Plots a Quantile-Quantile plot of the simulated and observed data.

    Useful for comparing to see whether the two datasets come from the same distribution.

    Parameters
    ----------
    merged_data_df: DataFrame
        Dataframe must contain a datetime type index and floating point type numbers in two
        columns. The left column must be simulated data and the right column must be observed data.
        If given, sim_array and obs_array must be None.

    sim_array: 1D ndarray
        Array of simulated data. If given, merged_data_df parameter must be None and obs_array must
        be given.

    obs_array: 1D ndarray
        Array of observed data. If given, merged_data_df parameter must be None and sim_array must
        be given.

    interpolate: str
        Specifies the interpolation type when computing quantiles.

    title: str
        If given, sets the title of the plot.

    xlabel: str
        The label for the x axis that holds the simulated data quantiles.

    ylabel: str
        The label for the y axis that holds the observed data quantiles.

    legend: bool
        If True, a legend to explain the elements on the plot will be added.

    replace_nan: float, optional
        If given, indicates which value to replace NaN values with in the two arrays. If None, when
        a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
        of the observed and simulated array are removed before the computation.

    replace_inf: float, optional
        If given, indicates which value to replace Inf values with in the two arrays. If None, when
        an inf value is found at the i-th position in the observed OR simulated array, the i-th
        value of the observed and simulated array are removed before the computation.

    remove_neg: boolean, optional
        If True, when a negative value is found at the i-th position in the observed OR simulated
        array, the i-th value of the observed AND simulated array are removed before the
        computation.

    remove_zero: boolean, optional
        If true, when a zero value is found at the i-th position in the observed OR simulated
        array, the i-th value of the observed AND simulated array are removed before the
        computation.

    figsize: tuple of float
        Tuple of length two that specifies the horizontal and vertical lengths of the plot in
        inches, respectively.

    Returns
    -------
    fig : Matplotlib figure instance
        A matplotlib figure handle is returned, which can be viewed with the matplotlib.pyplot.show() command.

    Examples
    --------

    >>> import hydrostats.data as hd
    >>> import hydrostats.visual as hv
    >>> import matplotlib.pyplot as plt

    >>> sfpt_url = r'https://github.com/waderoberts123/Hydrostats/raw/master/Sample_data/sfpt_data/magdalena-calamar_interim_data.csv'
    >>> glofas_url = r'https://github.com/waderoberts123/Hydrostats/raw/master/Sample_data/GLOFAS_Data/magdalena-calamar_ECMWF_data.csv'
    >>> merged_df = hd.merge_data(sfpt_url, glofas_url, column_names=('SFPT', 'GLOFAS'))

    >>> qqplot(merged_data_df=merged_df, title='Quantile-Quantile Plot of Data',
    >>>        xlabel='SFPT Data Quantiles', ylabel='GLOFAS Data Quantiles', legend=True,
    >>>        figsize=(8, 6))
    >>> plt.show()

    .. image:: /Figures/qqplot.png

    """

    fig = plt.figure(figsize=figsize, facecolor='w', edgecolor='k')

    if merged_data_df is not None and sim_array is None and obs_array is None:
        # Creating a simulated and observed data array
        sim = merged_data_df.iloc[:, 0].values
        obs = merged_data_df.iloc[:, 1].values
    elif sim_array is not None and obs_array is not None and merged_data_df is None:
        sim = sim_array
        obs = obs_array
    else:
        raise RuntimeError("You must either pass in a dataframe or two arrays.")

    sim, obs = treat_values(sim, obs, replace_nan=replace_nan, replace_inf=replace_inf, remove_neg=remove_neg,
                            remove_zero=remove_zero)

    # Finding the size of n and creating a percentile vector:
    n = sim.size

    pvec = 100 * ((np.arange(1, n + 1) - 0.5) / n)

    sim_perc = np.percentile(sim, pvec, interpolation=interpolate)
    obs_perc = np.percentile(obs, pvec, interpolation=interpolate)

    # Finding the interquartile range to plot the best fit line
    quant_1_sim = np.percentile(sim, 25, interpolation=interpolate)
    quant_3_sim = np.percentile(sim, 75, interpolation=interpolate)
    quant_1_obs = np.percentile(obs, 25, interpolation=interpolate)
    quant_3_obs = np.percentile(obs, 75, interpolation=interpolate)
    quant_sim = np.array([quant_1_sim, quant_3_sim])
    quant_obs = np.array([quant_1_obs, quant_3_obs])

    dsim = quant_3_sim - quant_1_sim
    dobs = quant_3_obs - quant_1_obs
    slope = dobs / dsim
    centersim = (quant_1_sim + quant_3_sim) / 2
    centerobs = (quant_1_obs + quant_3_obs) / 2
    maxsim = np.max(sim)
    minsim = np.min(sim)
    maxobs = centerobs + slope * (maxsim - centersim)
    minobs = centerobs - slope * (centersim - minsim)

    msim = np.array([minsim, maxsim])
    mobs = np.array([minobs, maxobs])

    if not legend:
        plt.plot(sim_perc, obs_perc, 'b^', markersize=2)
        plt.plot(msim, mobs, 'r-.', lw=1)
        plt.plot(quant_sim, quant_obs, 'r-', marker='o', mfc='k', lw=2)
    else:
        plt.plot(sim_perc, obs_perc, 'b^', markersize=2, label='Quantiles')
        plt.plot(msim, mobs, 'r-.', lw=1, label='Entire Range of Quantiles')
        plt.plot(quant_sim, quant_obs, 'r-', marker='o', mfc='w', lw=2, label='Inter-Quartile Range')
        plt.legend(fontsize=14)

    if title is not None:
        plt.title(title, fontsize=18)

    # Formatting things
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()

    return fig


if __name__ == "__main__":
    pass

    # import pandas as pd
    #
    # merged_df = pd.read_pickle("tests/Files_for_tests/merged_df.pkl")
    #
    # plot(merged_data_df=merged_df,
    #      title='Hydrograph of Entire Time Series',
    #      linestyles=['r-', 'k-'],
    #      legend=('SFPT', 'GLOFAS'),
    #      labels=['Datetime', 'Streamflow (cfs)'],
    #      metrics=['ME', 'NSE', 'SA'],
    #      grid=True)
    #
    # plt.savefig(r"tests/baseline_images/plot_tests/plot_full1_test.png")
    #
    # plot(merged_data_df=daily_avg_df,
    #         title='Daily Average Streamflow (Standard Error)',
    #         legend=('SFPT', 'GLOFAS'),
    #         x_season=True,
    #         labels=['Datetime', 'Streamflow (csm)'],
    #         linestyles=['r-', 'k-'],
    #         fig_size=(14, 8),
    #         ebars=daily_std_error,
    #         ecolor=('r', 'k'),
    #         tight_xlim=True
    #         )
