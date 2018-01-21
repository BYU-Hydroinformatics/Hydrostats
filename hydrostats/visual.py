from hydrostats import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot(merged_data_df, legend=False, metrics=[], grid=False):
    fig = plt.figure(num=1, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)
    plt.plot(merged_data_df.index, merged_data_df['predicted streamflow'], 'ro', markersize=2)
    plt.plot(merged_data_df.index, merged_data_df['recorded streamflow'], 'b^', markersize=2)
    plt.ylabel('Streamflow', fontsize=18)
    plt.yticks(fontsize=14)
    plt.xlabel('Date', fontsize=18)
    plt.xticks(fontsize=14, rotation=45)
    if legend:
        sim = mpatches.Patch(color='red', label='Forecaster Streamflow')
        obs = mpatches.Patch(color='blue', label='Observed Streamflow')
        plt.legend(handles=[sim, obs])
    if grid:
        plt.grid(True)
    if metrics:
        forecasted_array = merged_data_df['predicted streamflow'].as_matrix()
        observed_array = merged_data_df['recorded streamflow'].as_matrix()
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
        plt.text(-0.35, 1.0, formatted_selected_metrics, ha='left', va='top', transform=ax.transAxes, fontdict=font)
        plt.subplots_adjust(left=0.25)
    plt.show()

    
def plot_daily(merged_data_df, legend=False, metrics=[], grid=False):
    """ Plotting function used to plot daily values from the daily average time series dataframe created with the 
    hydrostats.data.daily_average function."""
    fig = plt.figure(num=1, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)
    plt.plot(merged_data_df.index, merged_data_df['predicted streamflow'], 'ro', markersize=2)
    plt.plot(merged_data_df.index, merged_data_df['recorded streamflow'], 'b^', markersize=2)
    plt.ylabel('Streamflow', fontsize=18)
    plt.yticks(fontsize=14)
    plt.xlabel('Date', fontsize=18)
    # Get the current axis
    ax = plt.gca()
    # Only label every 20th value
    ticks_to_use = merged_data_df.index[::30]
    # Now set the ticks and labels
    ax.set_xticks(ticks_to_use)
    ax.set_xticklabels(ticks_to_use)
    plt.xticks(fontsize=14, rotation=45)
    if legend:
        sim = mpatches.Patch(color='red', label='Forecaster Streamflow')
        obs = mpatches.Patch(color='blue', label='Observed Streamflow')
        plt.legend(handles=[sim, obs])
    if grid:
        plt.grid(True)
    if metrics:
        forecasted_array = merged_data_df['predicted streamflow'].as_matrix()
        observed_array = merged_data_df['recorded streamflow'].as_matrix()
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
        plt.text(-0.35, 1.0, formatted_selected_metrics, ha='left', va='top', transform=ax.transAxes, fontdict=font)
        plt.subplots_adjust(left=0.25)
    plt.show()
