from hydrostats import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


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
        plt.title(title)
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
