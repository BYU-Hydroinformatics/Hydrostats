from hydrostats import me, mae, mse, ed, ned, rmse, rmsle, mase, r_squared, acc, mape, mapd, smap1, smap2, d, d1, dr, \
    drel, dmod, M, R, NSE, NSEmod, NSErel, E_1, sa, sc, sid, sga
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import scipy.stats as stats
import calendar


def plot(merged_data_df, legend=None, metrics=None, grid=False, title=None, x_season=False, labels=None,
         savefigure=None, linestyles=['ro', 'b^'], tight_xlim=False):
    """A function that will plot a simulated and observed data vs time and show metrics to the left of the plot if specified.
    Arguments:
    merged_data_df - A pandas dataframe with a datetime type index, and two columns of simulated and observed data of
    float type, respectively.
    legend - A list of two string type inputs specifying the two legend text values (e.g. ['sim', 'obs']).
    metrics - A list of string type input indicating the metrics you want placed to the left of the graph
    (See documentation for a full list of the available metrics).
    grid - Boolean type indicating whether a graph is desired
    title - String type input indicating the title of the plot.
    x_season - For use when a dataframe has been converted to monthly or daily averages or standard error with a
    hydrostats.data function. A boolean type argument that will change the string values to monthly values.
    labels - A list of string type inputs specifying the x and y axis labels, respectively.
    savefigure - A string type input requiring the path and the filename where the figure will be save instead of
    displayed (e.g. r'path/to/file/image.png)
    linestyles - A list of two string type inputs specifying the linestyles (see documentation for the available
    options.
    tight_xlim - A boolean type input indicating if the user would like the gap in the x limits of the graph to be
    removed."""
    fig = plt.figure(num=1, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)
    if legend:
        plt.legend()
        plt.plot(merged_data_df.index, merged_data_df.iloc[:, 0], linestyles[0], markersize=2,
                 label=legend[0])
        plt.plot(merged_data_df.index, merged_data_df.iloc[:, 1], linestyles[1], markersize=2,
                 label=legend[1])
    else:
        plt.plot(merged_data_df.index, merged_data_df.iloc[:, 0], linestyles[0], markersize=2)
        plt.plot(merged_data_df.index, merged_data_df.iloc[:, 1], linestyles[1], markersize=2)
    if tight_xlim:
        plt.xlim(merged_data_df.index[0], merged_data_df.index[-1])
    if x_season:
        seasons = calendar.month_abbr[1:13]
        day_month = np.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        day_month_cum = np.cumsum(day_month)
        fractions = day_month_cum[:11] / 366
        fractions = np.insert(fractions, 0, 0)
        index = np.rint(fractions * len(merged_data_df.index)).astype(np.integer)
        plt.xticks(merged_data_df.index[index], seasons, fontsize=14, rotation=45)
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
    if metrics:
        forecasted_array = merged_data_df.iloc[:, 0].as_matrix()
        observed_array = merged_data_df.iloc[:, 1].as_matrix()
        function_list = [me, mae, mse, ed, ned, rmse, rmsle, mase, r_squared, acc, mape, mapd, smap1, smap2, d, d1, dr,
                         drel, dmod, watt_m, mb_r, nse, nse_mod, nse_rel, lm_index, sa, sc, sid, sga
                         ]
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
            selected_metrics.append(function_list_str[i] + '=' + str(round(function_list[i](forecasted_array, observed_array), 3)))
        formatted_selected_metrics = ''
        for i in selected_metrics:
            formatted_selected_metrics += i + '\n'
        font = {'family': 'sans-serif',
                'weight': 'normal',
                'size': 14}
        plt.text(-0.35, 0.75, formatted_selected_metrics, ha='left', va='center', transform=ax.transAxes, fontdict=font)
        plt.subplots_adjust(left=0.25)
    if savefigure:
        plt.savefig(savefigure)
        plt.close()
    else:
        plt.show()

    
def hist(merged_data_df, num_bins, z_norm=False, legend=None, grid=False, title=None, labels=None,
         savefigure=None, prob_dens=False):
    # Getting the fig and axis handles
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Creating a simulated and observed data array
    sim = merged_data_df.iloc[:, 0].as_matrix()
    obs = merged_data_df.iloc[:, 1].as_matrix()

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

    if savefigure:
        # Saving the figure
        plt.savefig(savefigure)
        plt.close()
    else:
        # Showing the figure
        plt.show()

      
def scatter(merged_data_df, grid=False, title=None, labels=None, best_fit=False, savefigure=None, marker_style='ko',
            metrics=None):
    fig = plt.figure(num=1, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)
    sim = merged_data_df.iloc[:, 0].as_matrix()
    obs = merged_data_df.iloc[:, 1].as_matrix()
    plt.plot(sim, obs, marker_style)
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
        plt.legend(fontsize=12)
    if metrics is not None:
        forecasted_array = merged_data_df.iloc[:, 0].as_matrix()
        observed_array = merged_data_df.iloc[:, 1].as_matrix()
        function_list = [me, mae, mse, ed, ned, rmse, rmsle, mase, r_squared, acc, mape, mapd, smap1, smap2, d, d1, dr,
                         drel, dmod, watt_m, mb_r, nse, nse_mod, nse_rel, lm_index, sa, sc, sid, sga
                         ]
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
    if savefigure:
        plt.savefig(savefigure)
        plt.close()
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
