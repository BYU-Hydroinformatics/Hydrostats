import pandas as pd


def merge_data(predicted_file_path, recorded_file_path, column_names=['Simulated', 'Observed']):
    """Takes two csv files that have been formatted with 1 row as a header with date in the first column and
        streamflow values in the second column and combines them into a pandas dataframe with datetime type for the
        dates and float type for the streamflow value"""

    # Importing data into a data-frame
    df_predicted = pd.read_csv(predicted_file_path, delimiter=",", header=None, names=[column_names[0]],
                               index_col=0, infer_datetime_format=True, skiprows=1)
    df_recorded = pd.read_csv(recorded_file_path , delimiter=",", header=None, names=[column_names[1]],
                              index_col=0, infer_datetime_format=True, skiprows=1)
    # Converting the index to datetime type
    df_recorded.index = pd.to_datetime(df_recorded.index, infer_datetime_format=True)
    df_predicted.index = pd.to_datetime(df_predicted.index, infer_datetime_format=True)
    # Joining the two dataframes
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
    merged_dataframe = merged_dataframe.replace([np.inf, -np.inf], np.nan)
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
