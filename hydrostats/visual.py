# python 3.6
# -*- coding: utf-8 -*-
"""
Created on Jan 5 3:25:56 2018
@author: Wade Roberts
"""
from __future__ import division
from hydrostats import HydrostatsVariables, remove_values
from hydrostats.data import HydrostatsError
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import calendar


def plot(merged_data_df, legend=None, metrics=None, grid=False, title=None, x_season=False, labels=None,
         savefigure=None, linestyles=['ro', 'b^'], tight_xlim=False, fig_size=(10, 6), text_adjust=[-0.35, 0.75],
         plot_adjust=0.27, transparency=0.5, ebars=None, ecolor=None, markersize=2, errorevery=1, markevery=1):
    """ A function that will plot a simulated and observed data vs time and show metrics to the left of the plot if
    specified. See the documentation at:
    https://github.com/waderoberts123/Hydrostats/blob/master/docs/README.md#hydrostatsvisualplot. """
    fig = plt.figure(num=1, figsize=fig_size, dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)

    # Setting Variable for the simulated data, observed data, and time stamps
    sim = merged_data_df.iloc[:, 0].values
    obs = merged_data_df.iloc[:, 1].values
    time = merged_data_df.index

    if legend is not None and ebars is None:
        plt.plot(time, sim, linestyles[0], markersize=markersize,
                 label=legend[0], alpha=transparency, markevery=markevery)
        plt.plot(time, obs, linestyles[1], markersize=markersize,
                 label=legend[1], alpha=transparency, markevery=markevery)
        plt.legend(fontsize=14)
    elif legend is not None and ebars is not None:
        plt.errorbar(x=time, y=sim, yerr=ebars.iloc[:, 0].values,
                     fmt=linestyles[0], markersize=markersize, label=legend[0], alpha=transparency, ecolor=ecolor[0],
                     markevery=markevery, errorevery=errorevery)
        plt.errorbar(x=time, y=obs, yerr=ebars.iloc[:, 1].values,
                     fmt=linestyles[1], markersize=markersize, label=legend[1], alpha=transparency, ecolor=ecolor[1],
                     markevery=markevery, errorevery=errorevery)
        plt.legend(fontsize=14)
    elif legend is None and ebars is not None:
        plt.errorbar(time, sim, fmt=linestyles[0], yerr=ebars.iloc[:, 0].values,
                     markersize=markersize, alpha=transparency, ecolor=ecolor[0], markevery=markevery,
                     errorevery=errorevery)
        plt.errorbar(time, obs, fmt=linestyles[1], yerr=ebars.iloc[:, 0].values,
                     markersize=markersize, alpha=transparency, ecolor=ecolor[1], markevery=markevery,
                     errorevery=errorevery)
    else:
        plt.plot(time, sim, linestyles[0], markersize=markersize,
                 alpha=transparency, markevery=markevery)
        plt.plot(time, obs, linestyles[1], markersize=markersize,
                 alpha=transparency, markevery=markevery)
    if tight_xlim:
        plt.xlim(time[0], time[-1])
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

    if grid:
        plt.grid(True)

    plt.tight_layout()

    if metrics:
        function_list = HydrostatsVariables.function_list
        function_list_str = HydrostatsVariables.metric_abbr
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
    if savefigure is not None:
        plt.savefig(savefigure)
        plt.close()
    else:
        plt.show()


def hist(merged_data_df=None, sim_array=None, obs_array=None, num_bins=100, z_norm=False, legend=None, grid=False,
         title=None, labels=None, savefigure=None, prob_dens=False, figsize=(12, 6)):
    # Getting the fig and axis handles
    fig, ax1 = plt.subplots(figsize=figsize)

    if merged_data_df is not None:
        # Creating a simulated and observed data array
        sim = merged_data_df.iloc[:, 0].values
        obs = merged_data_df.iloc[:, 1].values
    elif sim_array is not None and obs_array is not None:
        sim = sim_array
        obs = obs_array
    else:
        raise HydrostatsError("You must either pass in a dataframe or two arrays.")

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

    if savefigure is not None:
        # Saving the figure
        plt.savefig(savefigure)
        plt.close()
    else:
        # Showing the figure
        plt.show()


def scatter(merged_data_df=None, sim_array=None, obs_array=None, grid=False, title=None, labels=None, best_fit=False,
            savefigure=None, marker_style='ko', metrics=None, log_scale=False, line45=False, figsize=(12, 8)):

    fig = plt.figure(num=1, figsize=figsize, dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)

    if merged_data_df is not None:
        # Creating a simulated and observed data array
        sim = merged_data_df.iloc[:, 0].values
        obs = merged_data_df.iloc[:, 1].values
    elif sim_array is not None and obs_array is not None:
        sim = sim_array
        obs = obs_array
    else:
        raise HydrostatsError("You must either pass in a dataframe or two arrays.")

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
        x = sp.symbols("x")
        poly = sum(sp.S("{:6.4f}".format(v)) * x ** i for i, v in enumerate(p[::-1]))
        eq_latex = sp.printing.latex(poly)

        # Plotting the best fit line with the equation as a legend in latex
        plt.plot(x_new, y_new, 'k', label="${}$".format(eq_latex))

    if line45 or best_fit:
        plt.legend(fontsize=12)

    if metrics is not None:

        function_list = [me, mae, mse, ed, ned, rmse, rmsle, mase, r_squared, acc, mape, mapd, smap1, smap2, d, d1, dr,
                         drel, dmod, watt_m, mb_r, nse, nse_mod, nse_rel, lm_index, sa, sc, sid, sga]

        function_list_str = ['ME', 'MAE', 'MSE', 'ED', 'NED', 'RMSE', 'RMSLE', 'MASE', 'R^2', 'ACC', 'MAPE',
                             'MAPD', 'SMAP1', 'SMAP2', 'D', 'D1', 'DR', 'D-Rel', 'D-Mod', 'M', 'R', 'NSE', 'NSE-Mod',
                             'NSE-Rel', 'E_1', 'SA', 'SC', 'SID', 'SGA']

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
    if savefigure is not None:
        plt.savefig(savefigure)
        plt.close()
    else:
        plt.show()


def qqplot(merged_data_df=None, sim_array=None, obs_array=None, interpolate='linear', title=None, xlabel='Simulated Data Quantiles',
           ylabel='Observed Data Quantiles', legend=False, replace_nan=None, replace_inf=None, remove_neg=False,
           remove_zero=False, figsize=(12, 8), savefigure=None):

    plt.figure(num=1, figsize=figsize, dpi=80, facecolor='w', edgecolor='k')

    if merged_data_df is not None:
        # Creating a simulated and observed data array
        sim = merged_data_df.iloc[:, 0].values
        obs = merged_data_df.iloc[:, 1].values
    elif sim_array is not None and obs_array is not None:
        sim = sim_array
        obs = obs_array
    else:
        raise HydrostatsError("You must either pass in a dataframe or two arrays.")

    sim, obs = remove_values(sim, obs, replace_nan=replace_nan, replace_inf=replace_inf, remove_neg=remove_neg,
                             remove_zero=remove_zero)

    # Finding the size of n and creating a percentile vector:
    n = sim.size

    pvec = 100 * ((np.arange(1, n + 1) - 0.5) / n)

    sim_perc = np.percentile(sim, pvec, interpolation=interpolate)
    obs_perc = np.percentile(obs, pvec, interpolation=interpolate)

    # Finding the interquartile range to plot the best fit line
    quant_1_sim = np.percentile(sim, 25, interpolation=interpolate)
    quant_2_sim = np.percentile(sim, 50, interpolation=interpolate)
    quant_3_sim = np.percentile(sim, 75, interpolation=interpolate)
    quant_1_obs = np.percentile(obs, 25, interpolation=interpolate)
    quant_2_obs = np.percentile(sim, 50, interpolation=interpolate)
    quant_3_obs = np.percentile(obs, 75, interpolation=interpolate)
    quant_sim = np.array([quant_1_sim, quant_2_sim, quant_3_sim])
    quant_obs = np.array([quant_1_obs, quant_2_obs, quant_3_obs])

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
        plt.title(title, fontsize=20)

    # Formatting things
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    if savefigure is not None:
        plt.savefig(savefigure)
        plt.close()
    else:
        plt.show()
