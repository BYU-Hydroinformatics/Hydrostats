# -*- coding: utf-8 -*-
"""

The data module contains tools for preprocessing data. It allows users to merge timeseries, compute
daily and monthly summary statistics, and get seasonal periods of a time series.

"""
from __future__ import division
import pandas as pd
from numpy import inf, nan

__all__ = ['julian_to_gregorian', 'merge_data', 'daily_average', 'daily_std_error', 'daily_std_dev', 'monthly_average',
           'monthly_std_error', 'monthly_std_dev', 'remove_nan_df', 'seasonal_period']


def julian_to_gregorian(dataframe, frequency=None, inplace=False):
    """
    Converts the index of the merged dataframe from julian float values to gregorian datetime
    values.

    Parameters
    ----------
    dataframe: Pandas DataFrame
        A DataFrame with an index of type float

    frequency: string
        Optional. Sometimes when converting from julian to gregorian there will be rounding errors
        due to the inability of computers to store floats as perfect decimals. Providing the
        frequency will automatically attempt to round the dates. A list of all the frequencies pandas provides is found
        `here <http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases/>`_. Common frequencies
        include daily ("D") and hourly ("H").

    inplace: bool
        Default False. If True, will modify the index of the dataframe in place rather than
        creating a copy and returning the copy. Use when the time series are very long and making
        a copy would take a large amount of memory

    Returns
    -------
    Pandas DataFrame
        A pandas DataFrame with gregorian index.

    Examples
    --------

    >>> import pandas as pd
    >>> import hydrostats.data as hd
    >>> import numpy as np

    >>> # The julian dates in an array
    >>> julian_dates = np.array([2444239.5, 2444239.5416666665, 2444239.5833333335, 2444239.625,
    >>>                          2444239.6666666665, 2444239.7083333335, 2444239.75,
    >>>                          2444239.7916666665, 2444239.8333333335, 2444239.875])
    >>> # Creating a test dataframe
    >>> test_df = pd.DataFrame(data=np.random.rand(10, 2),  # Random data in the columns
    >>>                        columns=("Simulated Data", "Observed Data"),
    >>>                        index=julian_dates)
    >>> test_df
                  Simulated Data  Observed Data
    2.444240e+06        0.764719       0.126610
    2.444240e+06        0.372736       0.141392
    2.444240e+06        0.008645       0.686477
    2.444240e+06        0.656825       0.480444
    2.444240e+06        0.555247       0.869409
    2.444240e+06        0.643896       0.549590
    2.444240e+06        0.242720       0.799617
    2.444240e+06        0.432421       0.185760
    2.444240e+06        0.694631       0.136986
    2.444240e+06        0.700422       0.390415

    >>> # Making a new df with gregorian index
    >>> test_df_gregorian = hd.julian_to_gregorian(test_df)
    >>> test_df_gregorian
                              Simulated Data   Observed Data
    1980-01-01 00:00:00.000000	    0.585454	    0.457238
    1980-01-01 01:00:00.028800	    0.524764	    0.083464
    1980-01-01 01:59:59.971200	    0.516821	    0.416683
    1980-01-01 03:00:00.000000	    0.948483	    0.553874
    1980-01-01 04:00:00.028800	    0.492280	    0.232901
    1980-01-01 04:59:59.971200	    0.527967	    0.296395
    1980-01-01 06:00:00.000000	    0.650018	    0.212802
    1980-01-01 07:00:00.028800	    0.585592	    0.802971
    1980-01-01 07:59:59.971200	    0.448243	    0.665814
    1980-01-01 09:00:00.000000	    0.137395	    0.201721

    >>> # Rounding can be applied due to floating point inaccuracy
    >>> test_df_gregorian_rounded = julian_to_gregorian(test_df, frequency="H")  # Hourly Rounding Frequency
    >>> test_df_gregorian_rounded
                         Simulated Data  Observed Data
    1980-01-01 00:00:00        0.309527       0.938991
    1980-01-01 01:00:00        0.872284       0.497708
    1980-01-01 02:00:00        0.168046       0.225845
    1980-01-01 03:00:00        0.954494       0.275607
    1980-01-01 04:00:00        0.875885       0.194380
    1980-01-01 05:00:00        0.236849       0.992770
    1980-01-01 06:00:00        0.639346       0.029808
    1980-01-01 07:00:00        0.855828       0.903927
    1980-01-01 08:00:00        0.638805       0.916124
    1980-01-01 09:00:00        0.273430       0.443980

    >>> # The DataFrame can also be modified in place, increasing efficiency with large time series
    >>> julian_to_gregorian(test_df, inplace=True, frequency="H")
    >>> test_df
                         Simulated Data  Observed Data
    1980-01-01 00:00:00        0.309527       0.938991
    1980-01-01 01:00:00        0.872284       0.497708
    1980-01-01 02:00:00        0.168046       0.225845
    1980-01-01 03:00:00        0.954494       0.275607
    1980-01-01 04:00:00        0.875885       0.194380
    1980-01-01 05:00:00        0.236849       0.992770
    1980-01-01 06:00:00        0.639346       0.029808
    1980-01-01 07:00:00        0.855828       0.903927
    1980-01-01 08:00:00        0.638805       0.916124
    1980-01-01 09:00:00        0.273430       0.443980

    """

    if inplace:
        dataframe.index = pd.to_datetime(dataframe.index, origin="julian", unit="D")

        if frequency is not None:
            dataframe.index = dataframe.index.round(frequency)

    else:
        # Copying to avoid modifying the original dataframe
        return_df = dataframe.copy()

        # Converting the dataframe index from julian to gregorian
        return_df.index = pd.to_datetime(return_df.index, origin="julian", unit="D")

        if frequency is not None:
            return_df.index = return_df.index.round(frequency)

        return return_df


def merge_data(sim_fpath=None, obs_fpath=None, sim_df=None, obs_df=None, interpolate=None,
               column_names=('Simulated', 'Observed'), simulated_tz=None, observed_tz=None, interp_type='pchip',
               return_tz="Etc/UTC", julian=False, julian_freq=None):
    """Merges two dataframes or csv files, depending on the input.

    Parameters
    ----------
    sim_fpath: str
        The filepath to the simulated csv of data. Can be a url if the page is formatted correctly.
        The csv must be formatted with the dates in the left column and the data in the right
        column.

    obs_fpath: str
        The filepath to the observed csv. Can be a url if the page is formatted correctly.
        The csv must be formatted with the dates in the left column and the data in the right
        column.

    sim_df: DataFrame
        A pandas DataFrame containing the simulated data. Must be formatted with a datetime index
        and the simulated data values in column 0.

    obs_df: DataFrame
        A pandas DataFrame containing the simulated data. Must be formatted with a datetime index
        and the simulated data values in column 0.

    interpolate: str
        Must be either 'observed' or 'simulated'. Specifies which data set you would like to
        interpolate if interpolation is needed to properly merge the data.

    column_names: tuple of str
        Tuple of length two containing the column names that the user would like to set for the
        DataFrame that is returned. Note that the simulated data will be in the left column and the
        observed data will be in the right column

    simulated_tz: str
        The timezone of the simulated data. A full list of timezones can be found in the
        :ref:`timezones`.

    observed_tz: str
        The timezone of the simulated data. A full list of timezones can be found in the
        :ref:`timezones`.

    interp_type: str
        Which interpolation method to use. Uses the default pandas interpolater.
        Available types are found at
        http://pandas.pydata.org/pandas-docs/version/0.16.2/generated/pandas.DataFrame.interpolate.html

    return_tz: str
        What timezone the merged dataframe's index should be returned as. Default is 'Etc/UTC', which is recommended
        for simplicity.

    julian: bool
        If True, will parse the first column of the file to a datetime index from julian floating point time
        representation, this is only valid when supplying the sim_fpath and obs_fpath parameters. Users supplying two
        DataFrame objects must convert the index from Julian to Gregorian using the julian_to_gregorian function in this
        module

    julian_freq: str
        A string representing the frequency of the julian dates so that they can be rounded. See examples for usage.

    Notes
    -----
    The only acceptable time frequencies in the data are 15min, 30min, 45min, and any number of hours or
    days in between.

    There are three scenarios to consider when merging your data:

    1. The first scenario is that the timezones and the spacing of the time series matches
       (eg. 1 Day). In this case, you will want to leave the simulated_tz, observed_tz, and
       interpolate arguments empty, and the function will simply join the two csv's into a dataframe.
    2. The second scenario is that you have two time series with matching time zones but not
       matching spacing. In this case you will want to leave the simulated_tz and observed_tz empty,
       and use the interpolate argument to tell the function which time series you would like to
       interpolate to match the other time series.
    3. The third scenario is that you have two time series with different time zones and possibly
       different spacings. In this case you will want to fill in the simulated_tz, observed_tz, and
       interpolate arguments. This will then take timezones into account when interpolating
       the selected time series.

    Examples
    --------

    >>> import hydrostats.data as hd
    >>> import pandas as pd
    >>> pd.options.display.max_rows = 15

    The data URLs contain streamflow data from two different models, and are provided from the Hydrostats Github page

    >>> sfpt_url = r'https://github.com/waderoberts123/Hydrostats/raw/master/Sample_data/sfpt_data/magdalena-calamar_interim_data.csv'
    >>> glofas_url = r'https://github.com/waderoberts123/Hydrostats/raw/master/Sample_data/GLOFAS_Data/magdalena-calamar_ECMWF_data.csv'
    >>> merged_df = hd.merge_data(sfpt_url, glofas_url, column_names=('Streamflow Prediction Tool', 'GLOFAS'))

    """
    # Reading the data into dataframes if from file
    if sim_fpath is not None and obs_fpath is not None:
        # Importing data into a data-frame
        sim_df_copy = pd.read_csv(sim_fpath, delimiter=",", header=None, names=[column_names[0]],
                                  index_col=0, infer_datetime_format=True, skiprows=1)
        obs_df_copy = pd.read_csv(obs_fpath, delimiter=",", header=None, names=[column_names[1]],
                                  index_col=0, infer_datetime_format=True, skiprows=1)

        # Converting the index to datetime type
        if julian:
            julian_to_gregorian(sim_df_copy, frequency=julian_freq, inplace=True)
            julian_to_gregorian(obs_df_copy, frequency=julian_freq, inplace=True)
        else:
            sim_df_copy.index = pd.to_datetime(sim_df_copy.index, infer_datetime_format=True, errors='coerce')
            obs_df_copy.index = pd.to_datetime(obs_df_copy.index, infer_datetime_format=True, errors='coerce')

    elif sim_df is not None and obs_df is not None:
        # Checking to make sure that both dataframes have datetime indices if they are not read from file.
        if not isinstance(sim_df.index, pd.DatetimeIndex) and not isinstance(obs_df.index, pd.DatetimeIndex):
            raise RuntimeError("Both the obs_df and the sim_df need to have a datetime index.")

        # Copying the user supplied DataFrame objects
        sim_df_copy = sim_df.copy()
        obs_df_copy = obs_df.copy()

    else:
        raise RuntimeError('either sim_fpath and obs_fpath or sim_df and obs_df are required inputs.')

    # Checking to see if the necessary arguments in the function are fulfilled
    if simulated_tz is None and observed_tz is not None:
        raise RuntimeError('Either Both Timezones are required or neither')

    elif simulated_tz is not None and observed_tz is None:
        raise RuntimeError('Either Both Timezones are required or neither')

    elif simulated_tz is not None and observed_tz is not None and interpolate is None:
        raise RuntimeError("You must specify with the interpolate parameter whether to interpolate the 'simulated' "
                           "or 'observed' data.")

    elif simulated_tz is None and observed_tz is None and interpolate is None:
        # Scenario 1

        # Merging and joining the two DataFrames
        merged_df = pd.DataFrame.join(sim_df_copy, obs_df_copy).dropna()
        merged_df.columns = column_names

        return merged_df

    elif simulated_tz is None and observed_tz is None and interpolate is not None:
        # Scenario 2

        if interpolate == 'simulated':
            # Resampling and interpolating the observed data to match
            sim_df_copy = sim_df_copy.resample("15min").interpolate(interp_type)

        elif interpolate == 'observed':
            # Resampling and interpolating the observed data to match
            obs_df_copy = obs_df_copy.resample("15min").interpolate(interp_type)

        else:
            raise RuntimeError("The interpolate argument must be either 'simulated' or 'observed'.")

        # Merging and joining the two DataFrames
        merged_df = pd.DataFrame.join(sim_df_copy, obs_df_copy).dropna()
        merged_df.columns = column_names

        return merged_df

    elif simulated_tz is not None and observed_tz is not None and interpolate is not None:
        # Scenario 3

        # Convert the DateTime Index of both DataFrames to User Specified Timezones
        sim_df_copy.index = sim_df_copy.index.tz_localize(simulated_tz).tz_convert(return_tz)
        obs_df_copy.index = obs_df_copy.index.tz_localize(observed_tz).tz_convert(return_tz)

        if interpolate == 'simulated':
            # Resampling the simulated DataFrame to 15 minute time increments, then interpolating
            sim_df_copy = sim_df_copy.resample("15min").interpolate(interp_type)

        elif interpolate == 'observed':
            # Resampling the observed DataFrame to 15 minute time increments, then interpolating
            obs_df_copy = obs_df_copy.resample("15min").interpolate(interp_type)

        else:
            raise RuntimeError("You must specify the interpolation argument to be either 'simulated' or "
                               "'observed'.")

        # Merging and joining the two DataFrames
        merged_df = pd.DataFrame.join(sim_df_copy, obs_df_copy).dropna()
        merged_df.columns = column_names

        return merged_df


def daily_average(df, rolling=False, **kwargs):
    """Calculates daily seasonal averages of the timeseries data in a DataFrame

    Parameters
    ----------

    df: DataFrame
        A pandas DataFrame with a datetime index and columns containing float type values.

    rolling: bool
        If True, will calculate the rolling seasonal average.

    **kwargs: pandas.DataFrame.rolling() properties, optional
        Options for how to compute the rolling averages. If not provided, the default is to use the following
        parameters: {window=6, min_periods=1, center=True, closed="right"}. Specifying **kwargs will clear the
        defaults, however.

    Returns
    -------
    DataFrame
        A pandas dataframe with a string type index of date representations and the daily seasonal averages as
        float values in the columns.

    Examples
    --------

    >>> import hydrostats.data as hd
    >>> import pandas as pd
    >>> pd.options.display.max_rows = 15

    The data URLs contain streamflow data from two different models, and are provided from the Hydrostats Github page

    >>> sfpt_url = r'https://github.com/waderoberts123/Hydrostats/raw/master/Sample_data/sfpt_data/magdalena-calamar_interim_data.csv'
    >>> glofas_url = r'https://github.com/waderoberts123/Hydrostats/raw/master/Sample_data/GLOFAS_Data/magdalena-calamar_ECMWF_data.csv'
    >>> merged_df = hd.merge_data(sfpt_url, glofas_url, column_names=('Streamflow Prediction Tool', 'GLOFAS'))

    >>> hd.daily_average(merged_df)
           Streamflow Prediction Tool       GLOFAS
    01/01                 7331.305278  7676.792460
    01/02                 7108.640746  7753.671798
    01/03                 6927.147740  7631.900453
    01/04                 6738.162886  7483.029897
    01/05                 6552.914171  7316.004227
    01/06                 6388.213829  7154.650963
    01/07                 6258.418600  7012.279722
                               ...          ...
    12/25                 8321.367143  8948.101821
    12/26                 8149.313143  8903.544978
    12/27                 7994.357429  8807.690639
    12/28                 7872.819143  8642.877365
    12/29                 7791.741143  8435.175677
    12/30                 7729.451143  8225.315074
    12/31                 7656.042286  8041.918136
    [366 rows x 2 columns]
    """
    # Calculating the daily average from the database
    if not rolling:
        daily_averages = df.groupby(df.index.strftime("%m/%d")).mean()
    else:
        if kwargs:
            rolling_averages = df.rolling(**kwargs).mean()
            daily_averages = daily_average(rolling_averages, rolling=False)
        else:
            rolling_averages = df.rolling(window=6, min_periods=1, center=True, closed="right").mean()
            daily_averages = daily_average(rolling_averages, rolling=False)

    return daily_averages


def daily_std_error(merged_data):
    """Calculates daily seasonal standard error of the timeseries data in a DataFrame

    Parameters
    ----------

    merged_data: DataFrame
        A pandas DataFrame with a datetime index and columns containing float type values.

    Returns
    -------
    DataFrame
        A pandas dataframe with a string type index of date representations and the daily seasonal standard error as
        float values in the columns.

    Examples
    --------

    >>> import hydrostats.data as hd
    >>> import pandas as pd
    >>> pd.options.display.max_rows = 15

    The data URLs contain streamflow data from two different models, and are provided from the Hydrostats Github page

    >>> sfpt_url = r'https://github.com/waderoberts123/Hydrostats/raw/master/Sample_data/sfpt_data/magdalena-calamar_interim_data.csv'
    >>> glofas_url = r'https://github.com/waderoberts123/Hydrostats/raw/master/Sample_data/GLOFAS_Data/magdalena-calamar_ECMWF_data.csv'
    >>> merged_df = hd.merge_data(sfpt_url, glofas_url, column_names=('Streamflow Prediction Tool', 'GLOFAS'))

    >>> hd.daily_std_error(merged_df)
           Streamflow Prediction Tool      GLOFAS
    01/01                  558.189895  494.958042
    01/02                  553.290181  442.497656
    01/03                  535.002487  432.096928
    01/04                  514.511095  422.915060
    01/05                  489.287216  411.861086
    01/06                  463.321927  401.023620
    01/07                  441.666108  395.703128
                               ...         ...
    12/25                  613.876851  566.669886
    12/26                  589.424434  567.179646
    12/27                  582.957832  557.932109
    12/28                  581.465297  540.021918
    12/29                  573.949000  517.494155
    12/30                  560.993945  495.040565
    12/31                  546.904139  474.742075
    [366 rows x 2 columns]

    """
    # Calculating the daily average from the database
    a = merged_data.groupby(merged_data.index.strftime("%m/%d"))
    return a.sem()


def daily_std_dev(merged_data):
    """Calculates daily seasonal standard deviation of the timeseries data in a DataFrame

    Parameters
    ----------

    merged_data: DataFrame
        A pandas DataFrame with a datetime index and columns containing float type values.

    Returns
    -------
    DataFrame
        A pandas dataframe with a string type index of date representations and the daily seasonal standard deviation as
        float values in the columns.

    Examples
    --------

    >>> import hydrostats.data as hd
    >>> import pandas as pd
    >>> pd.options.display.max_rows = 15

    The data URLs contain streamflow data from two different models, and are provided from the Hydrostats Github page

    >>> sfpt_url = r'https://github.com/waderoberts123/Hydrostats/raw/master/Sample_data/sfpt_data/magdalena-calamar_interim_data.csv'
    >>> glofas_url = r'https://github.com/waderoberts123/Hydrostats/raw/master/Sample_data/GLOFAS_Data/magdalena-calamar_ECMWF_data.csv'
    >>> merged_df = hd.merge_data(sfpt_url, glofas_url, column_names=('Streamflow Prediction Tool', 'GLOFAS'))

    >>> hd.daily_std_dev(merged_df)
           Streamflow Prediction Tool       GLOFAS
    01/01                 3349.139373  2969.748253
    01/02                 3273.308852  2617.851437
    01/03                 3165.117397  2556.319898
    01/04                 3043.888685  2501.999235
    01/05                 2894.662206  2436.603046
    01/06                 2741.049485  2372.487729
    01/07                 2612.931931  2341.011275
                               ...          ...
    12/25                 3631.744428  3352.464257
    12/26                 3487.081980  3355.480036
    12/27                 3448.825041  3300.770870
    12/28                 3439.995086  3194.812751
    12/29                 3395.528078  3061.536706
    12/30                 3318.884936  2928.699478
    12/31                 3235.528520  2808.611992
    [366 rows x 2 columns]

    """
    # Calculating the daily average from the database
    a = merged_data.groupby(merged_data.index.strftime("%m/%d"))
    return a.std()


def monthly_average(merged_data):
    """Calculates monthly seasonal averages of the timeseries data in a DataFrame

    Parameters
    ----------

    merged_data: DataFrame
        A pandas DataFrame with a datetime index and columns containing float type values.

    Returns
    -------
    DataFrame
        A pandas dataframe with a string type index of date representations and the monthly seasonal averages as
        float values in the columns.

    Examples
    --------

    >>> import hydrostats.data as hd
    >>> import pandas as pd
    >>> pd.options.display.max_rows = 15

    The data URLs contain streamflow data from two different models, and are provided from the Hydrostats Github page

    >>> sfpt_url = r'https://github.com/waderoberts123/Hydrostats/raw/master/Sample_data/sfpt_data/magdalena-calamar_interim_data.csv'
    >>> glofas_url = r'https://github.com/waderoberts123/Hydrostats/raw/master/Sample_data/GLOFAS_Data/magdalena-calamar_ECMWF_data.csv'
    >>> merged_df = hd.merge_data(sfpt_url, glofas_url, column_names=('Streamflow Prediction Tool', 'GLOFAS'))

    >>> hd.monthly_average(merged_df)
            Streamflow Prediction Tool        GLOFAS
    01                 5450.558878   6085.033102
    02                 4178.249788   4354.072332
    03                 4874.788452   4716.785701
    04                 7682.920219   7254.073875
    05                13062.175899  11748.583189
    06                12114.431105  11397.032335
    07                 9461.472599   9598.017209
    08                 8802.643954   8708.876388
    09                10358.254219   9944.071882
    10                12968.474415  12671.180449
    11                13398.111010  13355.019167
    12                 9853.288608  10275.652887

    """
    # Calculating the daily average from the database
    a = merged_data.groupby(merged_data.index.strftime("%m"))
    return a.mean()


def monthly_std_error(merged_data):
    """Calculates monthly seasonal standard error of the timeseries data in a DataFrame

    Parameters
    ----------

    merged_data: DataFrame
        A pandas DataFrame with a datetime index and columns containing float type values.

    Returns
    -------
    DataFrame
        A pandas dataframe with a string type index of date representations and the monthly seasonal standard error as
        float values in the columns.

    Examples
    --------

    >>> import hydrostats.data as hd
    >>> import pandas as pd
    >>> pd.options.display.max_rows = 15

    The data URLs contain streamflow data from two different models, and are provided from the Hydrostats Github page

    >>> sfpt_url = r'https://github.com/waderoberts123/Hydrostats/raw/master/Sample_data/sfpt_data/magdalena-calamar_interim_data.csv'
    >>> glofas_url = r'https://github.com/waderoberts123/Hydrostats/raw/master/Sample_data/GLOFAS_Data/magdalena-calamar_ECMWF_data.csv'
    >>> merged_df = hd.merge_data(sfpt_url, glofas_url, column_names=('Streamflow Prediction Tool', 'GLOFAS'))

    >>> hd.monthly_std_error(merged_df)
        Streamflow Prediction Tool      GLOFAS
    01                   75.348943   71.206858
    02                   65.182159   58.347438
    03                   83.865980   72.919782
    04                  131.199766  123.486202
    05                  165.707528  139.586564
    06                  149.938998  136.119774
    07                  136.449337  115.436215
    08                  129.371343  110.101251
    09                  143.367894  123.798265
    10                  133.911782  114.008446
    11                  146.896826  116.443188
    12                  135.092750  117.958715

    """
    # Calculating the daily average from the database
    a = merged_data.groupby(merged_data.index.strftime("%m"))
    return a.sem()


def monthly_std_dev(merged_data):
    """Calculates monthly seasonal standard deviation of the timeseries data in a DataFrame

    Parameters
    ----------

    merged_data: DataFrame
        A pandas DataFrame with a datetime index and columns containing float type values.

    Returns
    -------
    DataFrame
        A pandas dataframe with a string type index of date representations and the monthly seasonal standard deviation
        as float values in the columns.

    Examples
    --------

    >>> import hydrostats.data as hd
    >>> import pandas as pd
    >>> pd.options.display.max_rows = 15

    The data URLs contain streamflow data from two different models, and are provided from the Hydrostats Github page

    >>> sfpt_url = r'https://github.com/waderoberts123/Hydrostats/raw/master/Sample_data/sfpt_data/magdalena-calamar_interim_data.csv'
    >>> glofas_url = r'https://github.com/waderoberts123/Hydrostats/raw/master/Sample_data/GLOFAS_Data/magdalena-calamar_ECMWF_data.csv'
    >>> merged_df = hd.merge_data(sfpt_url, glofas_url, column_names=('Streamflow Prediction Tool', 'GLOFAS'))

    >>> hd.monthly_std_dev(merged_df)
        Streamflow Prediction Tool       GLOFAS
    01                 2483.087791  2346.587412
    02                 2049.872689  1834.931830
    03                 2762.489873  2401.929369
    04                 4251.358318  4001.410267
    05                 5458.296296  4597.889041
    06                 4858.578846  4410.784791
    07                 4494.550854  3802.392524
    08                 4261.406429  3626.662349
    09                 4645.650733  4011.522281
    10                 4410.965472  3755.362766
    11                 4760.001179  3773.190543
    12                 4449.865762  3885.481991

    """
    # Calculating the daily average from the database
    a = merged_data.groupby(merged_data.index.strftime("%m"))
    return a.std()


def remove_nan_df(merged_dataframe):
    """Drops rows with NaN, zero, negative, and inf values from a pandas dataframe

    Parameters
    ----------

    merged_dataframe: DataFrame
        A pandas DataFrame with a datetime index and columns containing float type values.

    Returns
    -------
    DataFrame
        Pandas dataframe with rows containing NaN, zero, negative, and inf values removed.

    Examples
    --------

    >>> import pandas as pd
    >>> import numpy as np
    >>> import hydrostats.data as hd

    An example dataframe is created with invalid values inserted into it.

    >>> data = np.random.rand(15, 2)
    >>> data[0, 0] = data[1, 1] = np.nan
    >>> data[2, 0] = data[3, 1] = np.inf
    >>> data[4, 0] = data[5, 1] = 0
    >>> data[6, 0] = data[7, 1] = -0.1
    >>> example_df = pd.DataFrame(data=data, index=pd.date_range('1980-01-01', periods=15))
    >>> example_df
                       0         1
    1980-01-01       NaN  0.358903
    1980-01-02  0.074718       NaN
    1980-01-03       inf  0.515931
    1980-01-04  0.498002       inf
    1980-01-05  0.000000  0.617974
    1980-01-06  0.522522  0.000000
    1980-01-07 -0.100000  0.116625
    1980-01-08  0.588054 -0.100000
    1980-01-09  0.691136  0.561178
    1980-01-10  0.998170  0.432411
    1980-01-11  0.424473  0.599110
    1980-01-12  0.330988  0.469158
    1980-01-13  0.633894  0.191701
    1980-01-14  0.183241  0.784494
    1980-01-15  0.681419  0.303280

    Now the NaN, inf, negative, and zero values can be remove with this function.

    >>> hd.remove_nan_df(example_df)
                       0         1
    1980-01-09  0.691136  0.561178
    1980-01-10  0.998170  0.432411
    1980-01-11  0.424473  0.599110
    1980-01-12  0.330988  0.469158
    1980-01-13  0.633894  0.191701
    1980-01-14  0.183241  0.784494
    1980-01-15  0.681419  0.303280

    """
    # Drops Zeros and negatives
    merged_dataframe = merged_dataframe.loc[~(merged_dataframe <= 0).any(axis=1)]
    # Replaces infinites with nans
    merged_dataframe = merged_dataframe.replace([inf, -inf], nan)
    # Drops Nan
    merged_dataframe = merged_dataframe.dropna()

    return merged_dataframe


def seasonal_period(merged_dataframe, daily_period, time_range=None, numpy=False):
    """Creates a dataframe with a specified seasonal period

    Parameters
    ----------

    merged_dataframe: DataFrame
        A pandas DataFrame with a datetime index and columns containing float type values.

    daily_period: tuple of str
        A list of length two with strings representing the start and end dates of the seasonal period (e.g.
        (01-01, 01-31) for Jan 1 to Jan 31.

    time_range: tuple of str
        A tuple of string values representing the start and end dates of the time range. Format is YYYY-MM-DD.

    numpy: bool
        If True, two numpy arrays will be returned instead of a pandas dataframe

    Returns
    -------
    DataFrame
        Pandas dataframe that has been truncated to fit the parameters specified for the seasonal period.

    Examples
    --------
    >>> import pandas
    >>> pd.options.display.max_rows = 15
    >>> import numpy as np
    >>> import hydrostats.data as hd

    Here an example DataFrame is made with appx three years of data.

    >>> example_df = pd.DataFrame(data=np.random.rand(1000, 2), index=pd.date_range('2000-01-01', periods=1000), columns=['Simulated', 'Observed'])
                Simulated  Observed
    2000-01-01   0.862726  0.056597
    2000-01-02   0.979643  0.915072
    2000-01-03   0.857667  0.965057
    2000-01-04   0.011238  0.033678
    2000-01-05   0.011390  0.401728
    2000-01-06   0.056505  0.047417
    2000-01-07   0.615151  0.134103
                   ...       ...
    2002-09-20   0.883156  0.272355
    2002-09-21   0.595319  0.406609
    2002-09-22   0.415106  0.826873
    2002-09-23   0.399449  0.656040
    2002-09-24   0.243404  0.561899
    2002-09-25   0.879932  0.551347
    2002-09-26   0.787526  0.887288
    [1000 rows x 2 columns]

    Using this function, a new dataframe containing only the data values in january is returned.

    >>> seasonal_df_jan = hd.seasonal_period(example_df, ('01-01', '01-31'))
                Simulated  Observed
    2000-01-01   0.862726  0.056597
    2000-01-02   0.979643  0.915072
    2000-01-03   0.857667  0.965057
    2000-01-04   0.011238  0.033678
    2000-01-05   0.011390  0.401728
    2000-01-06   0.056505  0.047417
    2000-01-07   0.615151  0.134103
                   ...       ...
    2002-01-25   0.230580  0.363213
    2002-01-26   0.579899  0.370847
    2002-01-27   0.317925  0.120410
    2002-01-28   0.196034  0.035715
    2002-01-29   0.245429  0.974162
    2002-01-30   0.156166  0.544797
    2002-01-31   0.158595  0.311630
    [93 rows x 2 columns]

    We can also specify a time range if we only want the months of January in the year 2000 and 2001

    >>> seasonal_df_jan = hd.seasonal_period(example_df, ('01-01', '01-31'), time_range=('2000-01-01', '2001-12-31'))
                Simulated  Observed
    2000-01-01   0.862726  0.056597
    2000-01-02   0.979643  0.915072
    2000-01-03   0.857667  0.965057
    2000-01-04   0.011238  0.033678
    2000-01-05   0.011390  0.401728
    2000-01-06   0.056505  0.047417
    2000-01-07   0.615151  0.134103
                   ...       ...
    2001-01-25   0.119188  0.043076
    2001-01-26   0.896280  0.282883
    2001-01-27   0.659078  0.230265
    2001-01-28   0.667826  0.383687
    2001-01-29   0.298459  0.738100
    2001-01-30   0.336499  0.189036
    2001-01-31   0.571562  0.783718
    [62 rows x 2 columns]

    """
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


if __name__ == "__main__":
    pass
