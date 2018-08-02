# python 3.6
# -*- coding: utf-8 -*-
"""

The analyze module contains functions that perform a more complex analysis of simulated and observed
time series data. It allows users to make tables with metrics that they choose as well as different
date ranges. It also allows users to run a time lag analysis of two time series.

"""
from __future__ import division
from hydrostats.metrics import list_of_metrics, metric_names, metric_abbr, remove_values, \
    HydrostatsError
from hydrostats.data import seasonal_period
import calendar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

__all__ = ['make_table', 'time_lag']


def make_table(merged_dataframe, metrics, seasonal_periods=None, mase_m=1, dmod_j=1,
               nse_mod_j=1, h6_mhe_k=1, h6_ahe_k=1, h6_rmshe_k=1, d1_p_obs_bar_p=None,
               lm_x_obs_bar_p=None, replace_nan=None, replace_inf=None, remove_neg=False,
               remove_zero=False, to_csv=None, to_excel=None, location=None):
    """Create a table of user selected metrics with optional seasonal analysis.

    Creates a table with metrics as specified by the user. Seasonal periods can also be
    specified in order to compare different seasons and how well the simulated data matches the
    observed data. Has options to save the table to either a csv or an excel workbook. Also has
    an option to add a column for the location of the data.

    Parameters
    ----------
    merged_dataframe: DataFrame, optional
        A pandas dataframe that has two columns of predicted data (Col 0) and observed data (Col 1)
        with a datetime index.

    metrics: list of str
        A list of all the metrics that the user wants to calculate. The metrics abbreviations must
        be used and can be found in the table of all the metrics in the documentation.

    seasonal_periods: 2D list of str, optional
        If given, specifies the seasonal periods that the user wants to analyze (e.g. [['06-01',
        '06-30'], ['08-12', '11-23']] would analyze the dates from June 1st to June 30th and also
        August 8th to November 23). Note that the entire time series is analyzed with the selected
        metrics by default.

    mase_m: int, Optional
        Parameter for the mean absolute scaled error (MASE) metric.

    dmod_j: int or float, optional
        Parameter for the modified index of agreement (dmod) metric.

    nse_mod_j: int or float, optional
        Parameter for the modified Nash-Sutcliffe (nse_mod) metric.

    h6_mhe_k: int or float, optional
        Parameter for the H6 (MHE) metric.

    h6_ahe_k: int or float, optional
        Parameter for the H6 (AHE) metric

    h6_rmshe_k: int or float, optional
        Parameter for the H6 (RMSHE) metric

    d1_p_obs_bar_p: float, optional
        Parameter fot the Legate McCabe Index of Agreement (d1_p).

    lm_x_obs_bar_p: float, optional
        Parameter for the Lagate McCabe Efficiency Index (lm_index).

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

    to_csv: str
        Filepath and file name of the csv that is written (e.g. r'/path/to/output_dir/file.csv').

    to_excel: str
        Filepath and file name of the excel workbook that is written
        (e.g. r'/path/to/output_dir/file.xlsx').

    location: str
        The name of the location that will be created as a column in the table that is created.
        Useful for creating a large table with different datasets.

    Returns
    -------
    DataFrame
        Dataframe with rows containing the metric values at the different time ranges, and columns
        containing the metrics specified.

    Examples
    --------
    First we need to get some data. The data here is pulled from the Streamflow Predication Tool
    model and the ECMWF forecasting model. We are comparing the two models in this example.

    >>> import hydrostats as hs
    >>> import hydrostats.data as hd
    >>>
    >>> # Defining the URLs of the datasets
    >>> sfpt_url = r'https://github.com/waderoberts123/Hydrostats/raw/master/Sample_data/sfpt_data/magdalena-calamar_interim_data.csv'
    >>> glofas_url = r'https://github.com/waderoberts123/Hydrostats/raw/master/Sample_data/GLOFAS_Data/magdalena-calamar_ECMWF_data.csv'
    >>> # Merging the data
    >>> merged_df = hd.merge_data(sfpt_url, glofas_url, column_names=['SFPT', 'GLOFAS'])

    Here we make a table and print the results

    >>> table = hs.make_table(merged_dataframe=merged_df, metrics=['MAE', 'r2', 'NSE', 'KGE (2012)'],
    >>>                       seasonal_periods=[['01-01', '03-31'], ['04-01', '06-30'],
    >>>                                         ['07-01', '09-30'], ['10-01', '12-31']],
    >>>                       remove_neg=True, remove_zero=True, location='Magdalena')
    >>> table
                             Location          MAE        r2       NSE  KGE (2012)
    Full Time Series        Magdalena  1157.669988  0.907503  0.873684    0.872871
    January-01:March-31     Magdalena   631.984177  0.887249  0.861163    0.858187
    April-01:June-30        Magdalena  1394.640050  0.882599  0.813737    0.876890
    July-01:September-30    Magdalena  1188.542871  0.884249  0.829492    0.831188
    October-01:December-31  Magdalena  1410.852917  0.863800  0.793927    0.791257

    We can also write the table to a CSV or Excel worksheet.

    >>> hs.make_table(merged_dataframe=merged_df, metrics=['MAE', 'r2', 'NSE', 'KGE (2012)'],
    >>>               seasonal_periods=[['01-01', '03-31'], ['04-01', '06-30'], ['07-01', '09-30'],
    >>>                                 ['10-01', '12-31']],
    >>>               remove_neg=True, remove_zero=True, location='Magdalena',
    >>>               to_csv='magdalena_table.csv')
    """

    # Creating a list for all of the metrics for all of the seasons
    complete_metric_list = []

    # Creating an index list
    index_array = ['Full Time Series']
    if seasonal_periods is not None:
        seasonal_periods_names = []
        for i in seasonal_periods:
            month_1 = calendar.month_name[int(i[0][:2])]
            month_2 = calendar.month_name[int(i[1][:2])]
            name = month_1 + i[0][2:] + ':' + month_2 + i[1][2:]
            seasonal_periods_names.append(name)
        index_array.extend(seasonal_periods_names)

    # Creating arrays for sim and obs with all the values if a merged dataframe is given
    sim_array = merged_dataframe.iloc[:, 0].values
    obs_array = merged_dataframe.iloc[:, 1].values

    # Getting a list of the full time series
    full_time_series_list = list_of_metrics(metrics=metrics, abbr=True, sim_array=sim_array, obs_array=obs_array,
                                            mase_m=mase_m, dmod_j=dmod_j, nse_mod_j=nse_mod_j, h6_mhe_k=h6_mhe_k,
                                            h6_ahe_k=h6_ahe_k, h6_rmshe_k=h6_rmshe_k, d1_p_obs_bar_p=d1_p_obs_bar_p,
                                            lm_x_obs_bar_p=lm_x_obs_bar_p, replace_nan=replace_nan,
                                            replace_inf=replace_inf, remove_neg=remove_neg, remove_zero=remove_zero)

    # Appending the full time series list to the entire list:
    complete_metric_list.append(full_time_series_list)

    # Calculating metrics for the seasonal periods
    if seasonal_periods is not None:
        for time in seasonal_periods:
            temp_df = seasonal_period(merged_dataframe, time)
            sim_array = temp_df.iloc[:, 0].values
            obs_array = temp_df.iloc[:, 1].values

            seasonal_metric_list = list_of_metrics(metrics=metrics, sim_array=sim_array, abbr=True, obs_array=obs_array,
                                                   mase_m=mase_m, dmod_j=dmod_j, nse_mod_j=nse_mod_j,
                                                   h6_mhe_k=h6_mhe_k, h6_ahe_k=h6_ahe_k, h6_rmshe_k=h6_rmshe_k,
                                                   d1_p_obs_bar_p=d1_p_obs_bar_p, lm_x_obs_bar_p=lm_x_obs_bar_p,
                                                   replace_nan=replace_nan, replace_inf=replace_inf,
                                                   remove_neg=remove_neg, remove_zero=remove_zero)

            complete_metric_list.append(seasonal_metric_list)

    table_df_final = pd.DataFrame(complete_metric_list, index=index_array, columns=metrics)

    if location is not None:
        col_values = [location for i in range(table_df_final.shape[0])]
        table_df_final.insert(loc=0, column='Location', value=np.array(col_values))

    if to_csv is None and to_excel is None:
        return table_df_final

    elif to_csv is None and to_excel is not None:
        table_df_final.to_excel(to_excel, index_label='Datetime')

    elif to_csv is not None and to_excel is None:
        table_df_final.to_csv(to_csv, index_label='Datetime')

    else:
        table_df_final.to_excel(to_excel, index_label='Datetime')
        table_df_final.to_csv(to_csv, index_label='Datetime')


def time_lag(merged_dataframe, metrics, interp_freq='6H', interp_type='pchip',
             shift_range=[-30, 30], mase_m=1, dmod_j=1, nse_mod_j=1, h6_mhe_k=1,
             h6_ahe_k=1, h6_rmshe_k=1, d1_p_obs_bar_p=None, lm_x_obs_bar_p=None, replace_nan=None,
             replace_inf=None, remove_neg=False, remove_zero=False,
             plot_title='Metric Values as Different Lags', ylabel='Metric Value',
             xlabel='Number of Lags', save_fig=None, figsize=(10, 6), station=None, to_csv=None,
             to_excel=None):
    """Check metric values between simulated and observed data at different time lags.

    Runs a time lag analysis to check for potential timing errors in datasets. Returns a dataframe
    with all of the metric values at different time lags, the maximum and minimum metric value
    throughout the time lag, and the index of the maximum and minimum time lag values.

    """

    metrics_list = metric_names
    abbreviations = metric_abbr

    abbr_indices = []
    for i in metrics:
        abbr_indices.append(metrics_list.index(i))

    abbr_list = []
    for i in abbr_indices:
        abbr_list.append(abbreviations[i])

    # Making a new time index to be able to interpolate the time series to the required input
    new_index = pd.date_range(merged_dataframe.index[0], merged_dataframe.index[-1], freq=interp_freq)

    # Reindexing the dataframe and interpolating it
    try:
        merged_dataframe = merged_dataframe.reindex(new_index)
        merged_dataframe = merged_dataframe.interpolate(interp_type)
    except ValueError:
        raise HydrostatsError('ValueError Raised while interpolating, you may want to check for duplicates in'
                              ' your dates.')

    # Making arrays to compare the metric value at different time steps
    sim_array = merged_dataframe.iloc[:, 0].values
    obs_array = merged_dataframe.iloc[:, 1].values

    sim_array, obs_array = remove_values(sim_array, obs_array, replace_nan=replace_nan, replace_inf=replace_inf,
                                         remove_zero=remove_zero, remove_neg=remove_neg)

    # Creating a list to append the values of shift to
    shift_list = []

    # Creating a list of all the time shifts specified by the user
    lag_array = np.arange(shift_range[0], shift_range[1] + 1)

    # Looping through the list of lags and appending the metric value to the shift list
    for i in lag_array:
        sim_array_temp = np.roll(sim_array, i)

        lag_metrics = list_of_metrics(metrics=metrics, sim_array=sim_array_temp, obs_array=obs_array, mase_m=mase_m,
                                      dmod_j=dmod_j, nse_mod_j=nse_mod_j, h6_mhe_k=h6_mhe_k,
                                      h6_ahe_k=h6_ahe_k, h6_rmshe_k=h6_rmshe_k, d1_p_obs_bar_p=d1_p_obs_bar_p,
                                      lm_x_obs_bar_p=lm_x_obs_bar_p, replace_nan=replace_nan, replace_inf=replace_inf,
                                      remove_neg=remove_neg, remove_zero=remove_zero)
        shift_list.append(lag_metrics)

    final_array = np.array(shift_list)

    plt.figure(figsize=figsize)

    for i, abbr in enumerate(abbr_list):
        shift_list_temp = final_array[:, i]
        plt.plot(lag_array, shift_list_temp, label=abbr, alpha=0.7)

    plt.title(plot_title, fontsize=18)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    if save_fig is None:
        plt.show()
    else:
        plt.savefig(save_fig)
        plt.close()

    max_lag_array = np.max(final_array, 0)
    max_lag_indices = np.argmax(final_array, 0)
    max_lag_locations = lag_array[max_lag_indices]
    min_lag_array = np.min(final_array, 0)
    min_lag_indices = np.argmin(final_array, 0)
    min_lag_locations = lag_array[min_lag_indices]

    data = np.column_stack((max_lag_array, max_lag_locations, min_lag_array, min_lag_locations))

    final_df = pd.DataFrame(data=data, index=metrics, columns=["Max", "Max Lag Number", "Min", "Min Lag Number"])

    if station is not None:
        col_values = [station for i in range(final_df.shape[0])]
        final_df.insert(loc=0, column='Station', value=np.array(col_values))

    if to_csv is None and to_excel is None:
        return final_df

    elif to_csv is None and to_excel is not None:
        final_df.to_excel(to_excel, index_label='Metric')

    elif to_csv is not None and to_excel is None:
        final_df.to_csv(to_csv, index_label='Metric')

    else:
        final_df.to_excel(to_excel, index_label='Metric')
        final_df.to_csv(to_csv, index_label='Metric')


if __name__ == "__main__":
    import hydrostats.data as hd

    # Defining the URLs of the datasets
    sfpt_url = r'https://raw.githubusercontent.com/waderoberts123/Hydrostats/master/Sample_data/' \
               r'sfpt_data/magdalena-calamar_interim_data.csv'
    glofas_url = r'https://raw.githubusercontent.com/waderoberts123/Hydrostats/master/Sample_data' \
                 r'/GLOFAS_Data/magdalena-calamar_ECMWF_data.csv'

    # Merging the data
    merged_df = hd.merge_data(sim_fpath=sfpt_url, obs_fpath=glofas_url, column_names=['SFPT', 'GLOFAS'])

    # print(pd.read_csv(sfpt_url, delimiter=','))

    pd.options.display.max_columns = 10

    table = make_table(merged_dataframe=merged_df, metrics=['MAE', 'r2', 'NSE', 'KGE (2012)'],
                       seasonal_periods=[['01-01', '03-31'], ['04-01', '06-30'],
                                         ['07-01', '09-30'], ['10-01', '12-31']],
                       remove_neg=True, remove_zero=True, location='Magdalena')
    print(table)