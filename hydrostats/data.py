import pandas as pd


def merge_data(predicted_file_path, recorded_file_path):
    """Takes two csv files that have been formatted with 1 row as a header with date in the first column and
        streamflow values in the second column and combines them into a pandas dataframe with datetime type for the
        dates and float type for the streamflow value"""

    # Importing data into a data-frame
    df_recorded = pd.read_csv('asaraghat_karnali_recorded_data.txt', delimiter=",", header=None, names=['recorded '
                                                                                                        'streamflow'],
                              index_col=0, infer_datetime_format=True, skiprows=1)
    df_predicted = pd.read_csv('asaraghat_karnali_interim_data.csv', delimiter=",", header=None, names=['predicted '
                                                                                                        'streamflow'],
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
