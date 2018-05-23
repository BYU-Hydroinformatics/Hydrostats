# python 3.6
# -*- coding: utf-8 -*-
"""
Created on Jan 7 1:56:32 2018
@author: Wade Roberts
"""
from __future__ import division
import pandas as pd
from numpy import inf, nan
import math


class HydrostatsError(Exception):
    pass


def merge_data(sim_fpath=None, obs_fpath=None, sim_df=None, obs_df=None, interpolate=None,
               column_names=['Simulated', 'Observed'], simulated_tz=None, observed_tz=None, interp_type='pchip'):
    """Takes two csv files or two pandas dataframes that have been formatted with 1 row as a header with date in the
    first column and streamflow values in the second column and combines them into a pandas dataframe with datetime type
     for the dates and float type for the streamflow value. Please note that the only acceptable time deltas are 15min,
    30min, 45min, and any number of hours in between.

        There are three scenarios to consider when merging your data.

        The first scenario is that the timezones and the spacing of the time series matches (eg. 1 Day). In this case,
        you will want to leave the simulated_tz, observed_tz, and interpolate arguments empty, and the function will
        simply join the two csv's into a dataframe.

        The second scenario is that you have two time series with matching time zones but not matching spacing. In this
        case you will want to leave the simulated_tz and observed_tz empty, and use the interpolate argument to tell the
        function which time series you would like to interpolate to match the other time series.

        The third scenario is that you have two time series with different time zones and possibly different spacings.
        In this case you will want to fill in the simulated_tz, observed_tz, and interpolate arguments. This will then
        take timezones into account when interpolating the selected time series.
        """
    # Reading the data into dataframes if from file
    if sim_fpath and obs_fpath is not None:
        # Importing data into a data-frame
        sim_df = pd.read_csv(sim_fpath, delimiter=",", header=None, names=[column_names[0]],
                             index_col=0, infer_datetime_format=True, skiprows=1)
        obs_df = pd.read_csv(obs_fpath, delimiter=",", header=None, names=[column_names[1]],
                             index_col=0, infer_datetime_format=True, skiprows=1)
        # Converting the index to datetime type
        obs_df.index = pd.to_datetime(obs_df.index, infer_datetime_format=True, errors='coerce')
        sim_df.index = pd.to_datetime(sim_df.index, infer_datetime_format=True, errors='coerce')

    elif sim_df is not None and obs_df is not None:
        # Overriding the column names to match the column name input
        sim_df.columns = sim_df.columns.astype(str)
        sim_df.columns.values[0] = column_names[0]
        obs_df.columns = obs_df.columns.astype(str)
        obs_df.columns.values[0] = column_names[1]

        # Checking to make sure that both dataframes have datetime indices if they are not read from file.
        if sim_df.index.dtype == "datetime64[ns]" and obs_df.index.dtype == "datetime64[ns]":
            pass
        else:
            obs_df.index = pd.to_datetime(obs_df.index, infer_datetime_format=True, errors='coerce')
            sim_df.index = pd.to_datetime(sim_df.index, infer_datetime_format=True, errors='coerce')

    else:
        raise HydrostatsError('either sim_fpath and obs_fpath or sim_df and obs_df are required inputs.')

    # Checking to see if the necessary arguments in the function are fulfilled
    if simulated_tz is None and observed_tz is not None:

        raise HydrostatsError('Either Both Timezones are required or neither')

    elif simulated_tz is not None and observed_tz is None:

        raise HydrostatsError('Either Both Timezones are required or neither')

    elif simulated_tz is not None and observed_tz is not None and interpolate is None:

        raise HydrostatsError("You must specify whether to interpolate the 'simulated' or 'observed' data.")

    elif simulated_tz is None and observed_tz is None and interpolate is None:
        """Scenario 1"""
        # Merging and joining the two dataframes
        return pd.DataFrame.join(sim_df, obs_df).dropna()

    elif simulated_tz is None and observed_tz is None and interpolate is not None:
        """Scenario 2"""

        if interpolate == 'simulated':
            # Condition for a half hour time delta
            if (obs_df.index[1] - obs_df.index[0]).seconds / 3600 == 0.5:
                # Making a new index of half hour time spacing for interpolation
                simulated_index_interpolate = pd.date_range(sim_df.index[0], sim_df.index[-1],
                                                            freq='30min', tz=simulated_tz)
                # Reindexing and interpolating the dataframe to match the observed data
                sim_df = sim_df.reindex(simulated_index_interpolate).interpolate(interp_type)
            elif (obs_df.index[1] - obs_df.index[0]).seconds / 3600 == 0.25 or \
                    (obs_df.index[1] - obs_df.index[0]).seconds / 3600 == 0.75:
                # Making a new index of quarter hour time spacing for interpolation
                simulated_index_interpolate = pd.date_range(sim_df.index[0], sim_df.index[-1],
                                                            freq='15min', tz=simulated_tz)
                # Reindexing and interpolating the dataframe to match the observed data
                sim_df = sim_df.reindex(simulated_index_interpolate).interpolate(interp_type)
            else:
                # Making a new index of one hour time spacing for interpolation
                simulated_index_interpolate = pd.date_range(sim_df.index[0], sim_df.index[-1],
                                                            freq='1H', tz=simulated_tz)
                # Reindexing and interpolating the dataframe to match the observed data
                sim_df = sim_df.reindex(simulated_index_interpolate).interpolate(interp_type)

        if interpolate == 'observed':

            # Condition for a half hour time delta
            if (sim_df.index[1] - sim_df.index[0]).seconds / 3600 == 0.5:

                # Making a new index of half hour time spacing for interpolation
                observed_index_interpolate = pd.date_range(obs_df.index[0], obs_df.index[-1],
                                                           freq='30min', tz=observed_tz)
                # Reindexing and interpolating the dataframe to match the observed data
                obs_df = obs_df.reindex(observed_index_interpolate).interpolate(interp_type)

            elif (sim_df.index[1] - sim_df.index[0]).seconds / 3600 == 0.25 or \
                    (sim_df.index[1] - sim_df.index[0]).seconds / 3600 == 0.75:

                # Making a new index of quarter hour time spacing for interpolation
                observed_index_interpolate = pd.date_range(obs_df.index[0], obs_df.index[-1],
                                                           freq='15min', tz=observed_tz)
                # Reindexing and interpolating the dataframe to match the observed data
                obs_df = obs_df.reindex(observed_index_interpolate).interpolate(interp_type)

            else:
                # Making a new index of one hour time spacing for interpolation
                observed_index_interpolate = pd.date_range(obs_df.index[0], obs_df.index[-1],
                                                           freq='1H', tz=observed_tz)
                # Reindexing and interpolating the dataframe to match the observed data
                obs_df = obs_df.reindex(observed_index_interpolate).interpolate(interp_type)

        return pd.DataFrame.join(sim_df, obs_df).dropna()

    elif simulated_tz is not None and observed_tz is not None and interpolate is not None:
        """Scenario 3"""

        # Finding the frequency of the timeseries for observed and simulated
        td_simulated = (sim_df.index[1] - sim_df.index[0]).days + \
                       ((sim_df.index[1] - sim_df.index[0]).seconds / 3600) / 24
        td_observed = (obs_df.index[1] - obs_df.index[0]).days + \
                      ((obs_df.index[1] - obs_df.index[0]).seconds / 3600) / 24

        # converting the time delta to a tuple with days and hours
        td_tuple_simulated = math.modf(td_simulated)
        td_tuple_observed = math.modf(td_observed)

        # Converting the time delta to a frequency
        freq_simulated = str(td_tuple_simulated[1]) + 'D' + str(td_tuple_simulated[0] * 24) + 'H'
        freq_observed = str(td_tuple_observed[1]) + 'D' + str(td_tuple_observed[0] * 24) + 'H'

        # Making a new index for reindexing the time series
        simulated_df_new_index = pd.date_range(sim_df.index[0], sim_df.index[-1],
                                               freq=freq_simulated, tz=simulated_tz)
        observed_df_new_index = pd.date_range(obs_df.index[0], obs_df.index[-1],
                                              freq=freq_observed, tz=observed_tz)

        # Changing the time series index to reflect the changes in the timezones
        sim_df.index = simulated_df_new_index
        obs_df.index = observed_df_new_index

        if interpolate == 'simulated':
            # Checking if the time zone is a half hour off of UTC
            if int(obs_df.index[0].strftime('%z')[-2:]) == 30:
                # Checking if the time delta is either 15 minutes or 45 minutes
                if td_tuple_observed[0] * 24 == 0.25 or td_tuple_simulated[0] * 24 == 0.75:
                    # Making a new index of quarter hour time spacing for interpolation
                    simulated_index_interpolate = pd.date_range(sim_df.index[0], sim_df.index[-1],
                                                                freq='15min', tz=simulated_tz)
                    # Reindexing and interpolating the dataframe to match the observed data
                    sim_df = sim_df.reindex(simulated_index_interpolate).interpolate(interp_type)

                else:
                    # Making a new index of half hour time spacing for interpolation
                    simulated_index_interpolate = pd.date_range(sim_df.index[0], sim_df.index[-1],
                                                                freq='30min', tz=simulated_tz)
                    # Reindexing and interpolating the dataframe to match the observed data
                    sim_df = sim_df.reindex(simulated_index_interpolate).interpolate(interp_type)

            else:
                # Checking if the time delta is either 15 minutes or 45 minutes
                if td_tuple_observed[0] * 24 == 0.25 or td_tuple_observed[0] * 24 == 0.75 or \
                        int(obs_df.index[0].strftime('%z')[-2:]) == 45 or \
                        int(sim_df.index[0].strftime('%z')[-2:]) == 45:
                    # Making a new index of quarter hour time spacing for interpolation
                    simulated_index_interpolate = pd.date_range(sim_df.index[0], sim_df.index[-1],
                                                                freq='15min', tz=simulated_tz)
                    # Reindexing and interpolating the dataframe to match the observed data
                    sim_df = sim_df.reindex(simulated_index_interpolate).interpolate(interp_type)

                elif td_tuple_observed[0] * 24 == 0.5:
                    # Making a new index of half hour time spacing for interpolation
                    simulated_index_interpolate = pd.date_range(sim_df.index[0], sim_df.index[-1],
                                                                freq='30min', tz=simulated_tz)
                    # Reindexing and interpolating the dataframe to match the observed data
                    sim_df = sim_df.reindex(simulated_index_interpolate).interpolate(interp_type)

                else:
                    # Making a new index of half hour time spacing for interpolation
                    simulated_index_interpolate = pd.date_range(sim_df.index[0], sim_df.index[-1],
                                                                freq='1H', tz=simulated_tz)
                    # Reindexing and interpolating the dataframe to match the observed data
                    sim_df = sim_df.reindex(simulated_index_interpolate).interpolate(interp_type)

        elif interpolate == 'observed':
            # Checking if the time zone is a half hour off of UTC
            if int(sim_df.index[0].strftime('%z')[-2:]) == 30 or \
                    int(obs_df.index[0].strftime('%z')[-2:]) == 30:

                # Checking if the time delta is either 15 minutes or 45 minutes
                if td_tuple_simulated[0] * 24 == 0.25 or td_tuple_simulated[0] * 24 == 0.75:
                    # Making a new index of quarter hour time spacing for interpolation
                    observed_index_interpolate = pd.date_range(obs_df.index[0], obs_df.index[-1],
                                                               freq='15min', tz=observed_tz)
                    # Reindexing and interpolating the dataframe to match the observed data
                    obs_df = obs_df.reindex(observed_index_interpolate).interpolate(interp_type)

                else:
                    # Making a new index of half hour time spacing for interpolation
                    observed_index_interpolate = pd.date_range(obs_df.index[0], obs_df.index[-1],
                                                               freq='30min', tz=observed_tz)
                    # Reindexing and interpolating the dataframe to match the observed data
                    obs_df = obs_df.reindex(observed_index_interpolate).interpolate(interp_type)

            else:
                # Checking if the time delta is either 15 minutes or 45 minutes
                if td_tuple_simulated[0] * 24 == 0.25 or td_tuple_simulated[0] * 24 == 0.75 or \
                        int(sim_df.index[0].strftime('%z')[-2:]) == 45 or \
                        int(obs_df.index[0].strftime('%z')[-2:]) == 45:
                    # Making a new index of quarter hour time spacing for interpolation
                    observed_index_interpolate = pd.date_range(obs_df.index[0], obs_df.index[-1],
                                                               freq='15min', tz=observed_tz)
                    # Reindexing and interpolating the dataframe to match the observed data
                    obs_df = obs_df.reindex(observed_index_interpolate).interpolate(interp_type)

                elif td_tuple_simulated[0] * 24 == 0.5:
                    # Making a new index of half hour time spacing for interpolation
                    observed_index_interpolate = pd.date_range(obs_df.index[0], obs_df.index[-1],
                                                               freq='30min', tz=observed_tz)
                    # Reindexing and interpolating the dataframe to match the observed data
                    obs_df = obs_df.reindex(observed_index_interpolate).interpolate(interp_type)

                else:
                    # Making a new index of half hour time spacing for interpolation
                    observed_index_interpolate = pd.date_range(obs_df.index[0], obs_df.index[-1],
                                                               freq='1H', tz=observed_tz)
                    # Reindexing and interpolating the dataframe to match the observed data
                    obs_df = obs_df.reindex(observed_index_interpolate).interpolate(interp_type)
        else:
            raise HydrostatsError("You must specify the interpolation argument to be either 'simulated' or 'observed'.")

        return pd.DataFrame.join(sim_df, obs_df).dropna()


def daily_average(merged_data):
    """Takes a dataframe with an index of datetime type and both recorded and predicted streamflow values and
        calculates the daily average streamflow over the course of the data. Returns a dataframe with the date
        format as MM/DD in the index and the two columns of the data averaged"""
    # Calculating the daily average from the database
    a = merged_data.groupby(merged_data.index.strftime("%m/%d"))
    return a.mean()


def daily_std_error(merged_data):
    """Takes a dataframe with an index of datetime type and both recorded and predicted streamflow values and
        calculates the daily standard error of the streamflow over the course of the data. Returns a dataframe with
        the date format as MM/DD in the index and the two columns of the data's standard error"""
    # Calculating the daily average from the database
    a = merged_data.groupby(merged_data.index.strftime("%m/%d"))
    return a.sem()


def daily_std_dev(merged_data):
    """Takes a dataframe with an index of datetime type and both recorded and predicted streamflow values and
        calculates the daily standard deviation of the streamflow over the course of the data. Returns a dataframe with
        the date format as MM/DD in the index and the two columns of the data's standard deviation"""
    # Calculating the daily average from the database
    a = merged_data.groupby(merged_data.index.strftime("%m/%d"))
    return a.std()


def monthly_average(merged_data):
    """Takes a dataframe with an index of datetime type and both recorded and predicted streamflow values and
        calculates the monthly average streamflow over the course of the data. Returns a dataframe with the date
        format as MM in the index and the two columns of the data averaged"""
    # Calculating the daily average from the database
    a = merged_data.groupby(merged_data.index.strftime("%m"))
    return a.mean()


def monthly_std_error(merged_data):
    """Takes a dataframe with an index of datetime type and both recorded and predicted streamflow values and
        calculates the monthly standard error of the streamflow over the course of the data. Returns a dataframe
        with the date format as MM in the index and the two columns of the data's standard error"""
    # Calculating the daily average from the database
    a = merged_data.groupby(merged_data.index.strftime("%m"))
    return a.sem()


def monthly_std_dev(merged_data):
    """Takes a dataframe with an index of datetime type and both recorded and predicted streamflow values and
        calculates the monthly standard deviation of the streamflow over the course of the data.
        Returns a dataframe with the date format as MM in the index and the two columns of the data's standard
        deviation"""
    # Calculating the daily average from the database
    a = merged_data.groupby(merged_data.index.strftime("%m"))
    return a.std()


def remove_nan_df(merged_dataframe):
    """Drops rows with Nan, zero, negative, and inf values from a pandas dataframe"""
    # Drops Zeros and negatives
    merged_dataframe = merged_dataframe.loc[~(merged_dataframe <= 0).any(axis=1)]
    # Replaces infinites with nans
    merged_dataframe = merged_dataframe.replace([inf, -inf], nan)
    # Drops Nan
    merged_dataframe = merged_dataframe.dropna()
    return merged_dataframe


def seasonal_period(merged_dataframe, daily_period, time_range=None, numpy=False):
    """Returns the seasonal period specified for the time series, between the specified time range. Can return either a
    pandas DataFrame or two numpy arrays"""
    # Making a copy to avoid changing the original df
    merged_df_copy = merged_dataframe.copy()

    if time_range:
        # Setting the time range
        merged_df_copy = merged_df_copy.loc[time_range[0]: time_range[1]]

    # Setting a placeholder for the datetime string values
    merged_df_copy.insert(loc=0, column='placeholder', value=merged_df_copy.index.strftime('%m-%d'))

    # getting the start and end of the seasonal period
    start = daily_period[0]
    end = daily_period[1]

    # Getting the seasonal period
    if start < end:
        merged_df_copy = merged_df_copy.loc[(merged_df_copy['placeholder'] >= start) &
                                                (merged_df_copy['placeholder'] <= end)]
    else:
        merged_df_copy = merged_df_copy.loc[(merged_df_copy['placeholder'] >= start) |
                                                (merged_df_copy['placeholder'] <= end)]
    # Dropping the placeholder
    merged_df_copy = merged_df_copy.drop(columns=['placeholder'])

    if numpy:
        return merged_df_copy.iloc[:, 0].values, merged_df_copy.iloc[:, 1].values
    else:
        return merged_df_copy
