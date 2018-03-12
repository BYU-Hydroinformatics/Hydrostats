import pandas as pd
from numpy import inf, nan
import math


def merge_data(predicted_file_path, recorded_file_path, column_names=['Simulated', 'Observed']):
    def merge_data(predicted_file_path, recorded_file_path, interpolate=None, column_names=['Simulated', 'Observed'],
               predicted_tz=None, recorded_tz=None, interp_type='pchip'):
    """Takes two csv files that have been formatted with 1 row as a header with date in the first column and
        streamflow values in the second column and combines them into a pandas dataframe with datetime type for the
        dates and float type for the streamflow value. Please note that the only acceptable time deltas are 15min,
        30min, 45min, and any number of hours in between.

        There are three scenarios to consider when merging your data.

        The first scenario is that the timezones and the spacing of the time series matches (eg. 1 Day). In this case,
        you will want to leave the predicted_tz, recorded_tz, and interpolate arguments empty, and the function will
        simply join the two csv's into a dataframe.

        The second scenario is that you have two time series with matching time zones but not matching spacing. In this
        case you will want to leave the predicted_tz and recorded_tz empty, and use the interpolate argument to tell the
        function which time series you would like to interpolate to match the other time series.

        The third scenario is that you have two time series with different time zones and possibly different spacings.
        In this case you will want to fill in the predicted_tz, recorded_tz, and interpolate arguments. This will then
        take timezones into account when interpolating the selected time series.
        """
    if predicted_tz is None and recorded_tz:

        print('Either Both Timezones are required or neither')

    elif predicted_tz and recorded_tz is None:

        print('Either Both Timezones are required or neither')

    elif predicted_tz is None and recorded_tz is None and interpolate is None:
        """Scenario 1"""

        # Importing data into a data-frame
        df_predicted = pd.read_csv(predicted_file_path, delimiter=",", header=None, names=[column_names[0]],
                                   index_col=0, infer_datetime_format=True, skiprows=1)
        df_recorded = pd.read_csv(recorded_file_path, delimiter=",", header=None, names=[column_names[1]],
                                  index_col=0, infer_datetime_format=True, skiprows=1)
        # Converting the index to datetime type
        df_recorded.index = pd.to_datetime(df_recorded.index, infer_datetime_format=True)
        df_predicted.index = pd.to_datetime(df_predicted.index, infer_datetime_format=True)

        return pd.DataFrame.join(df_predicted, df_recorded).dropna()

    elif predicted_tz is None and recorded_tz is None and interpolate:
        """Scenario 2"""

        # Importing data into a data-frame
        df_predicted = pd.read_csv(predicted_file_path, delimiter=",", header=None, names=[column_names[0]],
                                   index_col=0, infer_datetime_format=True, skiprows=1)
        df_recorded = pd.read_csv(recorded_file_path, delimiter=",", header=None, names=[column_names[1]],
                                  index_col=0, infer_datetime_format=True, skiprows=1)
        # Converting the index to datetime type
        df_recorded.index = pd.to_datetime(df_recorded.index, infer_datetime_format=True)
        df_predicted.index = pd.to_datetime(df_predicted.index, infer_datetime_format=True)

        if interpolate == 'predicted':
            # Condition for a half hour time delta
            if (df_recorded.index[1] - df_recorded.index[0]).seconds / 3600 == 0.5:
                # Making a new index of half hour time spacing for interpolation
                predicted_index_interpolate = pd.date_range(df_predicted.index[0], df_predicted.index[-1],
                                                            freq='30min', tz=predicted_tz)
                # Reindexing and interpolating the dataframe to match the observed data
                df_predicted = df_predicted.reindex(predicted_index_interpolate).interpolate(interp_type)
            elif (df_recorded.index[1] - df_recorded.index[0]).seconds / 3600 == 0.25 or \
                    (df_recorded.index[1] - df_recorded.index[0]).seconds / 3600 == 0.75:
                # Making a new index of quarter hour time spacing for interpolation
                predicted_index_interpolate = pd.date_range(df_predicted.index[0], df_predicted.index[-1],
                                                            freq='15min', tz=predicted_tz)
                # Reindexing and interpolating the dataframe to match the observed data
                df_predicted = df_predicted.reindex(predicted_index_interpolate).interpolate(interp_type)
            else:
                # Making a new index of one hour time spacing for interpolation
                predicted_index_interpolate = pd.date_range(df_predicted.index[0], df_predicted.index[-1],
                                                            freq='1H', tz=predicted_tz)
                # Reindexing and interpolating the dataframe to match the observed data
                df_predicted = df_predicted.reindex(predicted_index_interpolate).interpolate(interp_type)

        if interpolate == 'recorded':

            # Condition for a half hour time delta
            if (df_predicted.index[1] - df_predicted.index[0]).seconds / 3600 == 0.5:

                # Making a new index of half hour time spacing for interpolation
                recorded_index_interpolate = pd.date_range(df_recorded.index[0], df_recorded.index[-1],
                                                           freq='30min', tz=recorded_tz)
                # Reindexing and interpolating the dataframe to match the observed data
                df_recorded = df_recorded.reindex(recorded_index_interpolate).interpolate(interp_type)

            elif (df_predicted.index[1] - df_predicted.index[0]).seconds / 3600 == 0.25 or \
                    (df_predicted.index[1] - df_predicted.index[0]).seconds / 3600 == 0.75:

                # Making a new index of quarter hour time spacing for interpolation
                recorded_index_interpolate = pd.date_range(df_recorded.index[0], df_recorded.index[-1],
                                                           freq='15min', tz=recorded_tz)
                # Reindexing and interpolating the dataframe to match the observed data
                df_recorded = df_recorded.reindex(recorded_index_interpolate).interpolate(interp_type)

            else:
                # Making a new index of one hour time spacing for interpolation
                recorded_index_interpolate = pd.date_range(df_recorded.index[0], df_recorded.index[-1],
                                                           freq='1H', tz=recorded_tz)
                # Reindexing and interpolating the dataframe to match the observed data
                df_recorded = df_recorded.reindex(recorded_index_interpolate).interpolate(interp_type)

        return pd.DataFrame.join(df_predicted, df_recorded).dropna()

    elif predicted_tz and recorded_tz and interpolate:
        """Scenario 3"""

        # Importing data into a data-frame
        df_predicted = pd.read_csv(predicted_file_path, delimiter=",", header=None, names=[column_names[0]],
                                   index_col=0, infer_datetime_format=True, skiprows=1)
        df_recorded = pd.read_csv(recorded_file_path, delimiter=",", header=None, names=[column_names[1]],
                                  index_col=0, infer_datetime_format=True, skiprows=1)
        # Converting the index to datetime type
        df_predicted.index = pd.to_datetime(df_predicted.index, infer_datetime_format=True)
        df_recorded.index = pd.to_datetime(df_recorded.index, infer_datetime_format=True)

        # finding the frequency of the timeseries for observed and predicted
        td_predicted = (df_predicted.index[1] - df_predicted.index[0]).days + \
                       ((df_predicted.index[1] - df_predicted.index[0]).seconds / 3600) / 24
        td_recorded = (df_recorded.index[1] - df_recorded.index[0]).days + \
                      ((df_recorded.index[1] - df_recorded.index[0]).seconds / 3600) / 24

        # converting the time delta to a tuple with days and hours
        td_tuple_predicted = math.modf(td_predicted)
        td_tuple_recorded = math.modf(td_recorded)

        # Converting the time delta to a frequency
        freq_predicted = str(td_tuple_predicted[1]) + 'D' + str(td_tuple_predicted[0] * 24) + 'H'
        freq_recorded = str(td_tuple_recorded[1]) + 'D' + str(td_tuple_recorded[0] * 24) + 'H'

        # Making a new index for reindexing the time series
        predicted_df_new_index = pd.date_range(df_predicted.index[0], df_predicted.index[-1],
                                               freq=freq_predicted, tz=predicted_tz)
        recorded_df_new_index = pd.date_range(df_recorded.index[0], df_recorded.index[-1],
                                              freq=freq_recorded, tz=recorded_tz)

        # Changing the time series index to reflect the changes in the timezones
        df_predicted.index = predicted_df_new_index
        df_recorded.index = recorded_df_new_index

        if interpolate == 'predicted':
            # Checking if the time zone is a half hour off of UTC
            if int(df_recorded.index[0].strftime('%z')[-2:]) == 30:
                # Checking if the time delta is either 15 minutes or 45 minutes
                if td_tuple_recorded[0] * 24 == 0.25 or td_tuple_predicted[0] * 24 == 0.75:
                    # Making a new index of quarter hour time spacing for interpolation
                    predicted_index_interpolate = pd.date_range(df_predicted.index[0], df_predicted.index[-1],
                                                                freq='15min', tz=predicted_tz)
                    # Reindexing and interpolating the dataframe to match the observed data
                    df_predicted = df_predicted.reindex(predicted_index_interpolate).interpolate(interp_type)
                else:
                    # Making a new index of half hour time spacing for interpolation
                    predicted_index_interpolate = pd.date_range(df_predicted.index[0], df_predicted.index[-1],
                                                                freq='30min', tz=predicted_tz)
                    # Reindexing and interpolating the dataframe to match the observed data
                    df_predicted = df_predicted.reindex(predicted_index_interpolate).interpolate(interp_type)
            else:
                # Checking if the time delta is either 15 minutes or 45 minutes
                if td_tuple_recorded[0] * 24 == 0.25 or td_tuple_recorded[0] * 24 == 0.75 or \
                        int(df_recorded.index[0].strftime('%z')[-2:]) == 45 or \
                        int(df_predicted.index[0].strftime('%z')[-2:]) == 45:
                    # Making a new index of quarter hour time spacing for interpolation
                    predicted_index_interpolate = pd.date_range(df_predicted.index[0], df_predicted.index[-1],
                                                                freq='15min', tz=predicted_tz)
                    # Reindexing and interpolating the dataframe to match the observed data
                    df_predicted = df_predicted.reindex(predicted_index_interpolate).interpolate(interp_type)
                elif td_tuple_recorded[0] * 24 == 0.5:
                    # Making a new index of half hour time spacing for interpolation
                    predicted_index_interpolate = pd.date_range(df_predicted.index[0], df_predicted.index[-1],
                                                                freq='30min', tz=predicted_tz)
                    # Reindexing and interpolating the dataframe to match the observed data
                    df_predicted = df_predicted.reindex(predicted_index_interpolate).interpolate(interp_type)
                else:
                    # Making a new index of half hour time spacing for interpolation
                    predicted_index_interpolate = pd.date_range(df_predicted.index[0], df_predicted.index[-1],
                                                                freq='1H', tz=predicted_tz)
                    # Reindexing and interpolating the dataframe to match the observed data
                    df_predicted = df_predicted.reindex(predicted_index_interpolate).interpolate(interp_type)

        if interpolate == 'recorded':
            # Checking if the time zone is a half hour off of UTC
            if int(df_predicted.index[0].strftime('%z')[-2:]) == 30 or\
                    int(df_recorded.index[0].strftime('%z')[-2:]) == 30:
                # Checking if the time delta is either 15 minutes or 45 minutes
                if td_tuple_predicted[0] * 24 == 0.25 or td_tuple_predicted[0] * 24 == 0.75:
                    # Making a new index of quarter hour time spacing for interpolation
                    recorded_index_interpolate = pd.date_range(df_recorded.index[0], df_recorded.index[-1],
                                                                freq='15min', tz=recorded_tz)
                    # Reindexing and interpolating the dataframe to match the observed data
                    df_recorded = df_recorded.reindex(recorded_index_interpolate).interpolate(interp_type)
                else:
                    # Making a new index of half hour time spacing for interpolation
                    recorded_index_interpolate = pd.date_range(df_recorded.index[0], df_recorded.index[-1],
                                                                freq='30min', tz=recorded_tz)
                    # Reindexing and interpolating the dataframe to match the observed data
                    df_recorded = df_recorded.reindex(recorded_index_interpolate).interpolate(interp_type)
            else:
                # Checking if the time delta is either 15 minutes or 45 minutes
                if td_tuple_predicted[0] * 24 == 0.25 or td_tuple_predicted[0] * 24 == 0.75 or \
                        int(df_predicted.index[0].strftime('%z')[-2:]) == 45 or \
                        int(df_recorded.index[0].strftime('%z')[-2:]) == 45:
                    # Making a new index of quarter hour time spacing for interpolation
                    recorded_index_interpolate = pd.date_range(df_recorded.index[0], df_recorded.index[-1],
                                                                freq='15min', tz=recorded_tz)
                    # Reindexing and interpolating the dataframe to match the observed data
                    df_recorded = df_recorded.reindex(recorded_index_interpolate).interpolate(interp_type)
                elif td_tuple_predicted[0] * 24 == 0.5:
                    # Making a new index of half hour time spacing for interpolation
                    recorded_index_interpolate = pd.date_range(df_recorded.index[0], df_recorded.index[-1],
                                                                freq='30min', tz=recorded_tz)
                    # Reindexing and interpolating the dataframe to match the observed data
                    df_recorded = df_recorded.reindex(recorded_index_interpolate).interpolate(interp_type)
                else:
                    # Making a new index of half hour time spacing for interpolation
                    recorded_index_interpolate = pd.date_range(df_recorded.index[0], df_recorded.index[-1],
                                                                freq='1H', tz=recorded_tz)
                    # Reindexing and interpolating the dataframe to match the observed data
                    df_recorded = df_recorded.reindex(recorded_index_interpolate).interpolate(interp_type)

        return pd.DataFrame.join(df_predicted, df_recorded).dropna()


def daily_average(merged_data):
    """Takes a dataframe with an index of datetime type and both recorded and predicted streamflow values and
        calculates the daily average streamflow over the course of the data. Please note that this function assumes
        that the column for predicted streamflow is labeled 'predicted streamflow' and the column for recorded
        streamflow is labeled 'recorded streamflow.' Returns a dataframe with the date format as MM/DD in the index
        and the two columns of the data averaged"""
    # Calculating the daily average from the database
    return merged_data.groupby(merged_data.index.strftime("%m/%d")).mean()


def daily_std_error(merged_data):
    """Takes a dataframe with an index of datetime type and both recorded and predicted streamflow values and
        calculates the daily standard error of the streamflow over the course of the data. Please note that this
        function assumes that the column for predicted streamflow is labeled 'predicted streamflow' and the column for
        recorded streamflow is labeled 'recorded streamflow.' Returns a dataframe with the date format as MM/DD in the
        index and the two columns of the data averaged"""
    # Calculating the daily average from the database
    return merged_data.groupby(merged_data.index.strftime("%m/%d")).sem()


def monthly_average(merged_data):
    """Takes a dataframe with an index of datetime type and both recorded and predicted streamflow values and
        calculates the monthly average streamflow over the course of the data. Please note that this function assumes
        that the column for predicted streamflow is labeled 'predicted streamflow' and the column for recorded
        streamflow is labeled 'recorded streamflow.' Returns a dataframe with the date format as MM in the index
        and the two columns of the data averaged"""
    # Calculating the daily average from the database
    return merged_data.groupby(merged_data.index.strftime("%m")).mean()


def monthly_std_error(merged_data):
    """Takes a dataframe with an index of datetime type and both recorded and predicted streamflow values and
        calculates the monthly standard error of the streamflow over the course of the data. Please note that this
        function assumes that the column for predicted streamflow is labeled 'predicted streamflow' and the column for
        recorded streamflow is labeled 'recorded streamflow.' Returns a dataframe with the date format as MM in the
        index and the two columns of the data averaged"""
    # Calculating the daily average from the database
    return merged_data.groupby(merged_data.index.strftime("%m")).sem()


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
    pandas dataframe or two numpy arrays"""
    if time_range:
        merged_dataframe = merged_dataframe.loc[time_range[0] : time_range[1]]
    merged_dataframe.index = merged_dataframe.index.strftime('%m-%d')
    start = daily_period[0]
    end = daily_period[1]
    merged_dataframe = merged_dataframe.loc[(merged_dataframe.index >= start) &
                                            (merged_dataframe.index <= end)]
    if numpy:
        return merged_dataframe.iloc[:, 0].as_matrix(), merged_dataframe.iloc[:, 1].as_matrix()
    else:
        return merged_dataframe
