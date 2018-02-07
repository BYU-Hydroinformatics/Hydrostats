from hydrostats import *
import matplotlib.patches as mpatches
import numpy as np
import matplotlib.pyplot as plt
import hydrostats.data as hd
import sympy as sp
import scipy.stats as stats


def plot(merged_data_df, legend=None, metrics=None, grid=False, title=None, force_x=None, labels=None):
    """Daily means you want to plot daily values, in which we need to space the x ticks
    legend - put a list of the two legend values [sim, obs]
    metrics - include the metrics you want placed on the graph
    grid - boolean
    title - string indicating the title
    force_x - number that indicates if you want to force x values every nth time series, only use when x axis is cramped
    labels - x and y axis labels, have a string list ["x", 'y']"""
    fig = plt.figure(num=1, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)
    plt.plot(merged_data_df.index, merged_data_df.iloc[:, 0], 'ro', markersize=2)
    plt.plot(merged_data_df.index, merged_data_df.iloc[:, 1], 'b^', markersize=2)
    if force_x:
        # Get the current axis
        ax = plt.gca()
        # Only label every 20th value
        ticks_to_use = merged_data_df.index[::force_x]
        # Now set the ticks and labels
        ax.set_xticks(ticks_to_use)
        ax.set_xticklabels(ticks_to_use)
        plt.xticks(fontsize=14, rotation=45)
    else:
        plt.xticks(fontsize=14, rotation=45)
    plt.yticks(fontsize=14)
    if labels:
        #Plotting Labels
        plt.xlabel(labels[0], fontsize=18)
        plt.ylabel(labels[1], fontsize=18)
    if title:
        plt.title(title, fontsize=20)
    if legend:
        sim = mpatches.Patch(color='red', label=legend[0])
        obs = mpatches.Patch(color='blue', label=legend[1])
        plt.legend(handles=[sim, obs])
    if grid:
        plt.grid(True)
    if metrics:
        forecasted_array = merged_data_df.iloc[:, 0].as_matrix()
        observed_array = merged_data_df.iloc[:, 1].as_matrix()
        function_list = [me, mae, mse, ed, ned, rmse, rmsle, mase, r_squared, acc, mape, mapd, smap1, smap2, d, d1, dr,
                         drel, dmod, M, R, E, Emod, Erel, E_1, sa, sc, sid, sga
                         ]
        function_list_str = ['ME', 'MAE', 'MSE', 'ED', 'NED', 'RMSE', 'RMSLE', 'MASE', 'R^2', 'ACC', 'MAPE',
                             'MAPD', 'SMAP1', 'SMAP2', 'D', 'D1', 'DR', 'D-Rel', 'D-Mod', 'M', 'R', 'E', 'E-Mod',
                             'E-Rel', 'E_1', 'SA', 'SC', 'SID', 'SGA'
                             ]
        assert isinstance(metrics, list)
        for metric in metrics:
            assert metric in function_list_str
        index = []
        for metric in metrics:
            index.append(function_list_str.index(metric))
        selected_metrics = []
        for i in index:
            selected_metrics.append(function_list_str[i] + '=' + str(round(function_list[i](forecasted_array, observed_array), 3)))
        formatted_selected_metrics = ''
        for i in selected_metrics:
            formatted_selected_metrics += i + '\n'
        font = {'family': 'sans-serif',
                'weight': 'normal',
                'size': 14}
        plt.text(-0.35, 0.75, formatted_selected_metrics, ha='left', va='center', transform=ax.transAxes, fontdict=font)
        plt.subplots_adjust(left=0.25)
    plt.show()

    
def hist(merged_data_df, legend, bins, grid=False, title=None, labels=None, savefigure=None):
    """Bins needs to be a numpy array"""
    fig, ax1 = plt.subplots(figsize=(12, 7))
    sim = merged_data_df.iloc[:, 0].as_matrix()
    obs = merged_data_df.iloc[:, 1].as_matrix()
    ax1.hist(sim, bins, alpha=0.5, label=legend[0], edgecolor='black', linewidth=0.5)
    ax1.hist(obs, bins, alpha=0.5, label=legend[1], edgecolor='black', linewidth=0.5)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if grid:
        plt.grid(True)
    if title:
        plt.title(title, fontsize=20)
    if labels:
        plt.xlabel(labels[0], fontsize=18)
        plt.ylabel(labels[1], fontsize=18)
    ax1.legend(loc='upper right', framealpha=1)
    plt.tight_layout()
    if savefigure:
        plt.savefig(savefigure)
    else:
        plt.show()

      
def scatter(merged_data_df, grid=False, title=None, labels=None, best_fit=None, savefigure=None):
    plt.figure(figsize=(12, 7))
    sim = merged_data_df.iloc[:, 0].as_matrix()
    obs = merged_data_df.iloc[:, 1].as_matrix()
    plt.plot(sim, obs, 'ko')
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

        # calculate new x's and y's
        x_new = np.linspace(0, sim.max(), sim.size)
        y_new = f(x_new)

        # Formatting the best fit equation to be able to display in latex
        x = sp.symbols("x")
        poly = sum(sp.S("{:6.4f}".format(v)) * x ** i for i, v in enumerate(p[::-1]))
        eq_latex = sp.printing.latex(poly)

        # Plotting the best fit line with the equation as a legend in latex
        plt.plot(x_new, y_new, 'k', label="${}$".format(eq_latex))
        plt.legend(fontsize=12)
    if savefigure:
        plt.savefig(savefigure)
    else:
        plt.show()
        
        
def qq_plot(merged_data_df, grid=False, title=None, labels=None, savefigure=None, rvalue=None):
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111)
    measurements = merged_data_df.iloc[:, 0] - merged_data_df.iloc[:, 1]
    (osm, osr), (slope, intercept, r) = stats.probplot(measurements, dist="norm")
    print(osm, osr, slope, intercept, r)
    plt.plot(osm, osr, 'bo', osm, slope * osm + intercept, 'r-')
    if rvalue:
        xmin = np.min(osm)
        xmax = np.max(osm)
        ymin = np.min(measurements)
        ymax = np.max(measurements)
        posx = xmin + 0.70 * (xmax - xmin)
        posy = ymin + 0.01 * (ymax - ymin)
        plt.text(posx, posy, "$R^2=%1.4f$" % r**2)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if grid:
        ax.grid(True)
    if title:
        ax.set_title(title, fontsize=20)
    if labels:
        ax.set_xlabel(labels[0], fontsize=18)
        ax.set_ylabel(labels[1], fontsize=18)
    if savefigure:
        plt.savefig(savefigure)
    else:
        plt.show()
