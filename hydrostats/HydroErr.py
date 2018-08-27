# python 3.6
# -*- coding: utf-8 -*-
"""

The metrics module contains all of the metrics included in hydrostats that measure hydrologic skill.
Each metric is contained in function, and every metric has the ability to treat missing values as
well as remove zero and negative values from the timeseries data.

"""
from __future__ import division
import numpy as np
from numba import njit, prange
from scipy.stats import gmean, rankdata
import warnings

__all__ = ['me', 'mae', 'mse', 'mle', 'male', 'msle', 'mde', 'mdae', 'mdse', 'ed', 'ned', 'rmse',
           'rmsle', 'nrmse_range', 'nrmse_mean', 'nrmse_iqr', 'irmse', 'mase', 'r_squared',
           'pearson_r', 'spearman_r', 'acc', 'mape', 'mapd', 'maape', 'smape1', 'smape2', 'd', 'd1',
           'dmod', 'drel', 'dr', 'watt_m', 'mb_r', 'nse', 'nse_mod', 'nse_rel', 'kge_2009',
           'kge_2012', 'lm_index', 'd1_p', 've', 'sa', 'sc', 'sid', 'sga', 'h1_mhe', 'h1_mahe',
           'h1_rmshe', 'h2_mhe', 'h2_mahe', 'h2_rmshe', 'h3_mhe', 'h3_mahe', 'h3_rmshe', 'h4_mhe',
           'h4_mahe', 'h4_rmshe', 'h5_mhe', 'h5_mahe', 'h5_rmshe', 'h6_mhe', 'h6_mahe', 'h6_rmshe',
           'h7_mhe', 'h7_mahe', 'h7_rmshe', 'h8_mhe', 'h8_mahe', 'h8_rmshe', 'h10_mhe', 'h10_mahe',
           'h10_rmshe', 'g_mean_diff', 'mean_var']


####################################################################################################
#                            General and Hydrological Error Metrics                                #
####################################################################################################


def me(simulated_array, observed_array, replace_nan=None, replace_inf=None,
       remove_neg=False, remove_zero=False):
    """Compute the mean error of the simulated and observed data.

    .. image:: /pictures/ME.png

    **Range:** -inf < MAE < inf, data units, closer to zero is better, indicates bias.

    **Notes:** The mean error (ME) measures the difference between the simulated data and the
    observed data. For the mean error, a smaller number indicates a better fit to the original
    data. Note that if the error is in the form of random noise, the mean error will be very small,
    which can skew the accuracy of this metric. ME is cumulative and will be small even if there
    are large positive and negative errors that balance.

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The mean error value.

    Examples
    --------
    Note that in this example the random noise cancels, leaving a very small ME.

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> # Seed for reproducibility
    >>> np.random.seed(54839)

    >>> x = np.arange(100) / 20
    >>> sim = np.sin(x) + 2
    >>> obs = sim * (((np.random.rand(100) - 0.5) / 10) + 1)
    >>> he.me(sim, obs)
    -0.006832220968967168

    References
    ----------
    - Fisher, R.A., 1920. A Mathematical Examination of the Methods of Determining the Accuracy of
      an Observation by the Mean Error, and by the Mean Square Error. Monthly Notices of the Royal
      Astronomical Society 80 758 - 770.
    """

    # Checking data to make sure it will work and the arrays are correct
    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")

    # Treating missing values
    simulated_array, observed_array = treat_values(simulated_array, observed_array,
                                                   replace_nan=replace_nan,
                                                   replace_inf=replace_inf,
                                                   remove_neg=remove_neg,
                                                   remove_zero=remove_zero)
    return np.mean(simulated_array - observed_array)


def mae(simulated_array, observed_array, replace_nan=None, replace_inf=None,
        remove_neg=False, remove_zero=False):
    """Compute the mean absolute error of the simulated and observed data.

    .. image:: /pictures/MAE.png

    **Range:** 0 ≤ MAE < inf, data units, smaller is better, does not indicate bias.

    **Notes:** The ME measures the absolute difference between the simulated data and the observed
    data. For the mean abolute error, a smaller number indicates a better fit to the original data.
    Also note that random errors do not cancel. Also referred to as an L1-norm.

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The mean absolute error value.

    References
    ----------
    - Willmott, Cort J., and Kenji Matsuura. “Advantages of the Mean Absolute Error (MAE) over the
      Root Mean Square Error (RMSE) in Assessing Average Model Performance.” Climate Research 30,
      no. 1 (2005): 79–82.
    - Willmott, Cort J., and Kenji Matsuura. “On the Use of Dimensioned Measures of Error to
      Evaluate the Performance of Spatial Interpolators.” International Journal of Geographical
      Information Science 20, no. 1 (2006): 89–102.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 6.8])
    >>> he.mae(sim, obs)
    0.5666666666666665
    """

    # Checking and cleaning the data
    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    return np.mean(np.absolute(simulated_array - observed_array))


def mse(simulated_array, observed_array, replace_nan=None, replace_inf=None,
        remove_neg=False, remove_zero=False):
    """
    Compute the mean squared error of the simulated and observed data.

    .. image:: /pictures/MSE.png

    **Range:** 0 ≤ MSE < inf, data units squared, smaller is better, does not indicate bias.

    **Notes:** Random errors do not cancel, highlights larger errors, also referred to as a
    squared L2-norm.

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The mean squared error value.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 6.8])
    >>> he.mse(sim, obs)
    0.4333333333333333

    References
    ----------
    - Wang, Zhou, and Alan C. Bovik. “Mean Squared Error: Love It or Leave It? A New Look at Signal
      Fidelity Measures.” IEEE Signal Processing Magazine 26, no. 1 (2009): 98–117.
    """

    # Checking and cleaning the data
    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    return np.mean((simulated_array - observed_array) ** 2)


def mle(simulated_array, observed_array, replace_nan=None, replace_inf=None,
        remove_neg=False, remove_zero=False):
    """
    Compute the mean log error of the simulated and observed data.

    .. image:: /pictures/MLE.png

    **Range:** -inf < MLE < inf, data units, closer to zero is better, indicates bias.

    **Notes** Same as ME only use log ratios as the error term. Limits the impact of outliers, more
    evenly weights high and low data values.

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The mean log error value.

    Examples
    --------

    Note that the value is very small because it is in log space.

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 6.8])
    >>> he.mle(sim, obs)
    0.002961767058151136

    References
    ----------
    - Törnqvist, Leo, Pentti Vartia, and Yrjö O. Vartia. “How Should Relative Changes Be Measured?”
      The American Statistician 39, no. 1 (1985): 43–46.
    """

    # Checking and cleaning the data
    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    sim_log = np.log1p(simulated_array)
    obs_log = np.log1p(observed_array)
    return np.mean(sim_log - obs_log)


def male(simulated_array, observed_array, replace_nan=None, replace_inf=None,
         remove_neg=False, remove_zero=False):
    """
    Compute the mean absolute log error of the simulated and observed data.

    .. image:: /pictures/MALE.png

    **Range:** 0 ≤ MALE < inf, data units squared, smaller is better, does not indicate bias.

    **Notes** Same as MAE only use log ratios as the error term. Limits the impact of outliers,
    more evenly weights high and low flows.

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The mean absolute log error value.

    Examples
    --------

    Note that the value is very small because it is in log space.

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 6.8])
    >>> np.round(he.male(sim, obs), 6)
    0.090417

    References
    ----------
    - Törnqvist, Leo, Pentti Vartia, and Yrjö O. Vartia. “How Should Relative Changes Be Measured?”
      The American Statistician 39, no. 1 (1985): 43–46.
    """

    # Checking and cleaning the data
    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    sim_log = np.log1p(simulated_array)
    obs_log = np.log1p(observed_array)
    return np.mean(np.abs(sim_log - obs_log))


def msle(simulated_array, observed_array, replace_nan=None, replace_inf=None,
         remove_neg=False, remove_zero=False):
    """
    Compute the mean squared log error of the simulated and observed data.

    .. image:: /pictures/MSLE.png

    **Range:** 0 ≤ MSLE < inf, data units squared, smaller is better, does not indicate bias.

    **Notes** Same as the mean squared error (MSE) only use log ratios as the error term. Limits
    the impact of outliers, more evenly weights high and low values.

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The mean squared log error value.

    Examples
    --------

    Note that the value is very small because it is in log space.

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 6.8])
    >>> np.round(he.msle(sim, obs), 6)
    0.010426

    References
    ----------
    - Törnqvist, Leo, Pentti Vartia, and Yrjö O. Vartia. “How Should Relative Changes Be Measured?”
      The American Statistician 39, no. 1 (1985): 43–46.

    """
    # Checking and cleaning the data
    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    sim_log = np.log1p(simulated_array)
    obs_log = np.log1p(observed_array)
    return np.mean((sim_log - obs_log) ** 2)


def mde(simulated_array, observed_array, replace_nan=None, replace_inf=None,
        remove_neg=False, remove_zero=False):
    """
    Compute the median error (MdE) between the simulated and observed data.

    .. image:: /pictures/MdE.png

    **Range** -inf < MdE < inf, closer to zero is better.

    **Notes** This metric indicates bias. It is similar to the mean error (ME), only it takes the
    median rather than the mean. Median measures reduces the impact of outliers.

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Examples
    --------

    Note that the last outlier residual in the time series is negated using the median.

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 100])
    >>> he.mde(sim, obs)
    -0.10000000000000009

    Returns
    -------
    float
        The median error value.

    """

    # Checking and cleaning the data
    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    return np.median(simulated_array - observed_array)


def mdae(simulated_array, observed_array, replace_nan=None, replace_inf=None,
         remove_neg=False, remove_zero=False):
    """
    Compute the median absolute error (MdAE) between the simulated and observed data.

    .. image:: /pictures/MdAE.png

    **Range** 0 ≤ MdAE < inf, closer to zero is better.

    **Notes** This metric does not indicates bias. Random errors (noise) do not cancel.
    It is similar to the mean absolute error (MAE), only it takes the median rather than
    the mean. Median measures reduces the impact of outliers.

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Examples
    --------

    Note that the last outlier residual in the time series is negated using the median.

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 100])
    >>> he.mdae(sim, obs)
    0.75

    Returns
    -------
    float
        The median absolute error value.

    """

    # Checking and cleaning the data
    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    return np.median(np.abs(simulated_array - observed_array))


def mdse(simulated_array, observed_array, replace_nan=None, replace_inf=None,
         remove_neg=False, remove_zero=False):
    """
    Compute the median squared error (MdSE) between the simulated and observed data.

    .. image:: /pictures/MdSE.png

    **Range** 0 ≤ MdSE < inf, closer to zero is better.

    **Notes** This metric does not indicates bias. Random errors (noise) do not cancel.
    It is similar to the mean squared error (MSE), only it takes the median rather than
    the mean. Median measures reduces the impact of outliers.

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Examples
    --------

    Note that the last outlier residual in the time series is negated using the median.

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 100])
    >>> he.mdse(sim, obs)
    0.625

    Returns
    -------
    float
        The median squared error value.

    """

    # Checking and cleaning the data
    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    return np.median((simulated_array - observed_array) ** 2)


def ed(simulated_array, observed_array, replace_nan=None, replace_inf=None,
       remove_neg=False, remove_zero=False):
    """
    Compute the Euclidean distance between predicted and observed values in vector space.

    .. image:: /pictures/ED.png

    **Range** 0 ≤ ED < inf, smaller is better.
    **Notes** This metric does not indicate bias. It is also sometimes referred to as the L2-norm.

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.ed(sim, obs)
    1.63707055437449

    Returns
    -------
    float
        The euclidean distance error value.

    References
    ----------
    - Kennard, M. J., Mackay, S. J., Pusey, B. J., Olden, J. D., & Marsh, N. (2010). Quantifying
      uncertainty in estimation of hydrologic metrics for ecohydrological studies. River Research
      and Applications, 26(2), 137-156.
    """

    # Checking and cleaning the data
    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    return np.linalg.norm(observed_array - simulated_array)


def ned(simulated_array, observed_array, replace_nan=None, replace_inf=None,
        remove_neg=False, remove_zero=False):
    """
    Compute the normalized Euclidian distance between the simulated and observed data in vector
    space.

    .. image:: /pictures/NED.png

    **Range** 0 ≤ NED < inf, smaller is better.

    **Notes** This metric does not indicate bias. It is also sometimes referred to as the squared
    L2-norm.

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The normalized euclidean distance value.

    Examples
    --------
    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.ned(sim, obs)
    0.2872053604165771

    References
    ----------
    - Kennard, M. J., Mackay, S. J., Pusey, B. J., Olden, J. D., & Marsh, N. (2010). Quantifying
      uncertainty in estimation of hydrologic metrics for ecohydrological studies. River Research
      and Applications, 26(2), 137-156.
    """

    # Checking and cleaning the data
    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    a = observed_array / np.mean(observed_array)
    b = simulated_array / np.mean(simulated_array)
    return np.linalg.norm(a - b)


def rmse(simulated_array, observed_array, replace_nan=None, replace_inf=None,
         remove_neg=False, remove_zero=False):
    """

    Compute the root mean square error between the simulated and observed data.

    .. image:: /pictures/RMSE.png

    **Range** 0 ≤ RMSE < inf, smaller is better.

    **Notes:** The standard deviation of the residuals. A lower spread indicates that the points
    are better concentrated around the line of best fit (linear). Random errors do not cancel.
    This metric will highlights larger errors.

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The root mean square error value.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.rmse(sim, obs)
    0.668331255192114

    References
    ----------
    - Willmott, C.J., Matsuura, K., 2005. Advantages of the mean absolute error (MAE) over the
      root mean square error (RMSE) in assessing average model performance.
      Climate Research 30(1) 79-82.
    - Hyndman, R.J., Koehler, A.B., 2006. Another look at measures of forecast accuracy.
      International Journal of Forecasting 22(4) 679-688.
    """

    # Checking and cleaning the data
    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    return np.sqrt(np.mean((simulated_array - observed_array) ** 2))


def rmsle(simulated_array, observed_array, replace_nan=None, replace_inf=None,
          remove_neg=False, remove_zero=False):
    """

    Compute the root mean square log error between the simulated and observed data.

    .. image:: /pictures/RMSLE.png

    **Range:** 0 ≤ RMSLE < inf. Smaller is better, and it does not indicate bias.

    **Notes:** Random errors do not cancel while using this metric. This metric also limits the
    impact of outliers by more evenly weighting high and low values. To calculate the log values,
    each value in the observed and simulated array is increased by one unit in order to avoid
    run-time errors and nan values (function np.log1p).

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The root mean square log error value.

    Examples
    --------

    Notice that the value is very small because it is in log space.

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> np.round(he.rmsle(sim, obs), 6)
    0.103161

    References
    ----------
    - Tornquist, L., Vartia, P., Vartia, Y.O., 1985. How Should Relative Changes be Measured?
      The American Statistician 43-46.
    - Willmott, C.J., Matsuura, K., 2005. Advantages of the mean absolute error (MAE) over the
      root mean square error (RMSE) in assessing average model performance.
      Climate Research 30(1) 79-82.
    """
    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(simulated_array, observed_array,
                                                   replace_nan=replace_nan,
                                                   replace_inf=replace_inf, remove_neg=remove_neg,
                                                   remove_zero=remove_zero)
    return np.sqrt(np.mean(np.power(np.log1p(simulated_array) - np.log1p(observed_array), 2)))


def nrmse_range(simulated_array, observed_array, replace_nan=None, replace_inf=None,
                remove_neg=False, remove_zero=False):
    """Compute the range normalized root mean square error between the simulated and observed data.

    .. image:: /pictures/NRMSE_Range.png

    **Range:** 0 ≤ NRMSE < inf.

    **Notes:** This metric is the RMSE normalized by the range of the observed time series (x).
    Normalizing allows comparison between data sets with different scales. The NRMSErange is the
    most sensitive to outliers of the three normalized rmse metrics.

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The range normalized root mean square error value.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.nrmse_range(sim, obs)
    0.0891108340256152

    References
    ----------
    - Pontius, R.G., Thontteh, O., Chen, H., 2008. Components of information for multiple
      resolution comparison between maps that share a real variable. Environmental and Ecological
      Statistics 15(2) 111-142.
    """

    # Checking and cleaning the data
    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    rmse_value = np.sqrt(np.mean((simulated_array - observed_array) ** 2))
    obs_max = np.max(observed_array)
    obs_min = np.min(observed_array)
    return rmse_value / (obs_max - obs_min)


def nrmse_mean(simulated_array, observed_array, replace_nan=None, replace_inf=None,
               remove_neg=False, remove_zero=False):
    """Compute the mean normalized root mean square error between the simulated and observed data.

    .. image:: /pictures/NRMSE_Mean.png

    **Range:** 0 ≤ NRMSE < inf.

    **Notes:** This metric is the RMSE normalized by the mean of the observed time series (x).
    Normalizing allows comparison between data sets with different scales.

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The mean normalized root mean square error.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.nrmse_mean(sim, obs)
    0.11725109740212526

    References
    ----------
    - Pontius, R.G., Thontteh, O., Chen, H., 2008. Components of information for multiple
      resolution comparison between maps that share a real variable. Environmental and Ecological
      Statistics 15(2) 111-142.

    """

    # Checking and cleaning the data
    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    rmse_value = np.sqrt(np.mean((simulated_array - observed_array) ** 2))
    obs_mean = np.mean(observed_array)
    return rmse_value / obs_mean


def nrmse_iqr(simulated_array, observed_array, replace_nan=None, replace_inf=None,
              remove_neg=False, remove_zero=False):
    """Compute the IQR normalized root mean square error between the simulated and observed data.

    .. image:: /pictures/NRMSE_IQR.png

    **Range:** 0 ≤ NRMSE < inf.

    **Notes:** This metric is the RMSE normalized by the interquartile range of the observed time
    series (x). Normalizing allows comparison between data sets with different scales.
    The NRMSEquartile is the least sensitive to outliers of the three normalized rmse metrics.

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The IQR normalized root mean square error.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.nrmse_iqr(sim, obs)
    0.2595461185212093

    References
    ----------
    - Pontius, R.G., Thontteh, O., Chen, H., 2008. Components of information for multiple
      resolution comparison between maps that share a real variable. Environmental and Ecological
      Statistics 15(2) 111-142.

    """

    # Checking and cleaning the data
    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    rmse_value = np.sqrt(np.mean((simulated_array - observed_array) ** 2))
    q1 = np.percentile(observed_array, 25)
    q3 = np.percentile(observed_array, 75)
    iqr = q3 - q1
    return rmse_value / iqr


def irmse(simulated_array, observed_array, replace_nan=None, replace_inf=None,
          remove_neg=False, remove_zero=False):
    """

    Compute the inertial root mean square error (IRMSE) between the simulated and observed data.

    .. image:: /pictures/IRMSE.png

    **Range:** 0 ≤ IRMSE < inf, lower is better.

    **Notes:** This metric is the RMSE devided by by the standard deviation of the gradient of the
    observed timeseries data. This metric is meant to be help understand the ability of the model
    to predict changes in observation.

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The inertial root mean square error.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.irmse(sim, obs)
    0.14572738134831856

    References
    ----------
    - Daga, M., Deo, M.C., 2009. Alternative data-driven methods to estimate wind from waves by
      inverse modeling. Natural Hazards 49(2) 293-310.
    """

    # Checking and cleaning the data
    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    # Getting the gradient of the observed data
    obs_len = observed_array.size
    obs_grad = observed_array[1:obs_len] - observed_array[0:obs_len - 1]

    # Standard deviation of the gradient
    obs_grad_std = np.std(obs_grad, ddof=1)

    # Divide RMSE by the standard deviation of the gradient of the observed data
    rmse_value = np.sqrt(np.mean((simulated_array - observed_array) ** 2))
    return rmse_value / obs_grad_std


def mase(simulated_array, observed_array, m=1, replace_nan=None, replace_inf=None,
         remove_neg=False, remove_zero=False):
    """Compute the mean absolute scaled error between the simulated and observed data.

    .. image:: /pictures/MASE.png

    **Range:**

    **Notes:**

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

    m: int
        If given, indicates the seasonal period m. If not given, the default is 1.

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

    Returns
    -------
    float
        The mean absolute scaled error.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.mase(sim, obs)
    0.17341040462427745

    References
    ----------
    - Hyndman, R.J., Koehler, A.B., 2006. Another look at measures of forecast accuracy.
      International Journal of Forecasting 22(4) 679-688.
    """

    # Checking and cleaning the data
    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")

    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    start = m
    end = simulated_array.size - m
    a = np.mean(np.abs(simulated_array - observed_array))
    b = np.abs(observed_array[start:observed_array.size] - observed_array[:end])
    return a / (np.sum(b) / end)


def pearson_r(simulated_array, observed_array, replace_nan=None, replace_inf=None,
              remove_neg=False, remove_zero=False):
    """

    Compute the pearson correlation coefficient.

    .. image:: /pictures/R_pearson.png

    **Range:** -1 ≤ R (Pearson) ≤ 1. 1 indicates perfect postive correlation, 0 indicates
    complete randomness, -1 indicate perfect negative correlation.

    **Notes:** The pearson r coefficient measures linear correlation. It is sensitive to outliers.

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The Pearson correlation coefficient.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.pearson_r(sim, obs)
    0.9610793632835262

    References
    ----------
    - Pearson, K. (1895). Note on regression and inheritance in the case of two parents.
      Proceedings of the Royal Society of London, 58, 240-242.

    """
    # Checking and cleaning the data
    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    sim_mean = np.mean(simulated_array)
    obs_mean = np.mean(observed_array)

    top = np.sum((observed_array - obs_mean) * (simulated_array - sim_mean))
    bot1 = np.sqrt(np.sum((observed_array - obs_mean) ** 2))
    bot2 = np.sqrt(np.sum((simulated_array - sim_mean) ** 2))

    return top / (bot1 * bot2)


def spearman_r(simulated_array, observed_array, replace_nan=None, replace_inf=None,
               remove_neg=False, remove_zero=False):
    """

    Compute the spearman rank correlation coefficient.

    .. image:: /pictures/R_spearman.png

    **Range:** -1 ≤ R (Pearson) ≤ 1. 1 indicates perfect postive correlation, 0 indicates
    complete randomness, -1 indicate perfect negative correlation.

    **Notes:** The spearman r coefficient measures the monotonic relation between simulated and
    observed data. Because it uses a nonparametric measure of rank correlation, it is less sensitive
    to outliers compared to the Pearson correlation coefficient.

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The Spearman rank correlation coefficient.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.spearman_r(sim, obs)
    0.942857142857143

    References
    ----------
    - Spearman C (1904). "The proof and measurement of association between two things". American
      Journal of Psychology. 15: 72–101. doi:10.2307/1412159
    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    rank_sim = rankdata(simulated_array)
    rank_obs = rankdata(observed_array)

    mean_rank_sim = np.mean(rank_sim)
    mean_rank_obs = np.mean(rank_obs)

    top = np.mean((rank_obs - mean_rank_obs) * (rank_sim - mean_rank_sim))
    bot = np.sqrt(
        np.mean((rank_obs - mean_rank_obs) ** 2) * np.mean((rank_sim - mean_rank_sim) ** 2))

    return top / bot


def r_squared(simulated_array, observed_array, replace_nan=None, replace_inf=None,
              remove_neg=False, remove_zero=False):
    """

    Compute the the Coefficient of Determination (r2).

    .. image:: /pictures/r2.png

    **Range:** 0 ≤ r2 ≤ 1. 1 indicates perfect correlation, 0 indicates complete randomness.

    **Notes:** The Coefficient of Determination measures the linear relation between simulated and
    observed data. Because it is the pearson correlation coefficient squared, it is more heavily
    affected by outliers than the pearson correlation coefficient.

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The coefficient of determination (R^2).

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.r_squared(sim, obs)
    0.9236735425294681

    References
    ----------

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    a = observed_array - np.mean(observed_array)
    b = simulated_array - np.mean(simulated_array)
    return (np.sum(a * b)) ** 2 / (np.sum(a ** 2) * np.sum(b ** 2))


def acc(simulated_array, observed_array, replace_nan=None, replace_inf=None,
        remove_neg=False, remove_zero=False):
    """

    Compute the the anomaly correlation coefficient (ACC).

    .. image:: /pictures/ACC.png

    **Range:** -1 ≤ ACC ≤ 1. -1 indicates perfect negative correlation of the variation
    pattern of the anomalies, 0 indicates complete randomness of the variation patterns of the
    anomalies, 1 indicates perfect correlation of the variation pattern of the anomalies.

    **Notes:** Common measure in the verification of spatial fields. Measures the correlation
    between the variation pattern of the simulated data compared to the observed data.

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The anomaly correlation coefficient.

    Examples
    --------
    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.acc(sim, obs)
    0.8008994694029383

    References
    ----------
    - Langland, Rolf H., and Ryan N. Maue. “Recent Northern Hemisphere Mid-Latitude Medium-Range
      Deterministic Forecast Skill.” Tellus A: Dynamic Meteorology and Oceanography 64,
      no. 1 (2012): 17531.
    - Miyakoda, K., G. D. Hembree, R. F. Strickler, and I. Shulman. “Cumulative Results of Extended
      Forecast Experiments I. Model Performance for Winter Cases.” Monthly Weather Review 100, no.
      12(1972): 836–55.
    - Murphy, Allan H., and Edward S. Epstein. “Skill Scores and Correlation Coefficients in Model
      Verification.” Monthly Weather Review 117, no. 3 (1989): 572–82.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    a = simulated_array - np.mean(simulated_array)
    b = observed_array - np.mean(observed_array)
    c = np.std(observed_array, ddof=1) * np.std(simulated_array, ddof=1) * simulated_array.size
    return np.dot(a, b / c)


def mape(simulated_array, observed_array, replace_nan=None, replace_inf=None,
         remove_neg=False, remove_zero=False):
    """

    Compute the the mean absolute percentage error (MAPE).

    .. image:: /pictures/MAPE.png

    **Range:** 0% ≤ MAPE ≤ inf. 0% indicates perfect correlation, a larger error indicates a
    larger percent error in the data.

    **Notes:**

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The mean absolute percentage error.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.mape(sim, obs)
    11.639226612630866

    References
    ----------
    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    a = simulated_array - observed_array
    b = np.abs(a / observed_array)
    c = 100 / simulated_array.size
    return c * np.sum(b)


def mapd(simulated_array, observed_array, replace_nan=None, replace_inf=None,
         remove_neg=False, remove_zero=False):
    """Compute the the mean absolute percentage deviation (MAPD).

    .. image:: /pictures/MAPD.png

    **Range:**

    **Notes:**

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The mean absolute percentage deviation.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.mapd(sim, obs)
    0.10526315789473682

    References
    ----------
    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    a = np.sum(np.abs(simulated_array - observed_array))
    b = np.sum(np.abs(observed_array))
    return a / b


def maape(simulated_array, observed_array, replace_nan=None, replace_inf=None,
          remove_neg=False, remove_zero=False):
    """Compute the the Mean Arctangent Absolute Percentage Error (MAAPE).

    .. image:: /pictures/MAAPE.png

    **Range:** 0 ≤ MAAPE < π/2, does not indicate bias, smaller is better.

    **Notes:** Represents the mean absolute error as a percentage of the observed values. Handles
    0s in the observed data. This metric is not as biased as MAPE by under-over predictions.

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The mean arctangent absolute percentage error.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.mape(sim, obs)
    11.639226612630866

    References
    ----------
    - Kim, S., Kim, H., 2016. A new metric of absolute percentage error for intermittent demand
      forecasts. International Journal of Forecasting 32(3) 669-679.
    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    a = simulated_array - observed_array
    b = np.abs(a / observed_array)
    return np.mean(np.arctan(b))


def smape1(simulated_array, observed_array, replace_nan=None, replace_inf=None,
           remove_neg=False, remove_zero=False):
    """

    Compute the the Symmetric Mean Absolute Percentage Error (1) (SMAPE1).

    .. image:: /pictures/SMAPE1.png

    **Range:** 0 ≤ SMAPE1 < 200%, does not indicate bias, smaller is better, symmetrical.

    **Notes:** This metric is an adjusted version of the MAPE.

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The symmetric mean absolute percentage error (1).

    Examples
    --------

    Note that if we switch the simulated and observed arrays the result is the same

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.smape1(sim, obs)
    5.871915694397428
    >>> he.smape1(obs, sim)
    5.871915694397428

    References
    ----------
    - Flores, B.E., 1986. A pragmatic view of accuracy measurement in forecasting. Omega 14(2)
      93-98.
    - Goodwin, P., Lawton, R., 1999. On the asymmetry of the symmetric MAPE. International Journal
      of Forecasting 15(4) 405-408.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    a = 100 / simulated_array.size
    b = np.abs(simulated_array - observed_array)
    c = np.abs(simulated_array) + np.abs(observed_array)
    return a * np.sum(b / c)


def smape2(simulated_array, observed_array, replace_nan=None, replace_inf=None,
           remove_neg=False, remove_zero=False):
    """

    Compute the the Symmetric Mean Absolute Percentage Error (2) (SMAPE2).

    .. image:: /pictures/SMAPE2.png

    **Range:** 0 ≤ SMAPE1 < 200%, does not indicate bias, smaller is better, symmetrical.

    **Notes:** This metric is an adjusted version of the MAPE with only positive metric values.

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The symmetric mean absolute percentage error (2).

    Examples
    --------

    Note that switching the simulated and observed arrays yields the same results

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.smape2(sim, obs)
    11.743831388794856
    >>> he.smape2(obs, sim)
    11.743831388794856


    References
    ----------
    - Flores, B.E., 1986. A pragmatic view of accuracy measurement in forecasting. Omega 14(2)
      93-98.
    - Goodwin, P., Lawton, R., 1999. On the asymmetry of the symmetric MAPE. International Journal
      of Forecasting 15(4) 405-408.
    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    a = simulated_array - observed_array
    b = (simulated_array + observed_array) / 2
    c = 100 / simulated_array.size
    return c * np.sum(np.abs(a / b))


def d(simulated_array, observed_array, replace_nan=None, replace_inf=None,
      remove_neg=False, remove_zero=False):
    """

    Compute the the index of agreement (d).

    .. image:: /pictures/d.png

    **Range:** 0 ≤ d < 1, does not indicate bias, larger is better.

    **Notes:** This metric is a modified approach to the Nash-Sutcliffe Efficiency metric.

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The index of agreement (1).

    Examples
    --------
    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.d(sim, obs)
    0.978477353035657

    References
    ----------
    - Legates, D.R., McCabe Jr, G.J., 1999. Evaluating the use of “goodness‐of‐fit” Measures in
      hydrologic and hydroclimatic model validation. Water Resources Research 35(1) 233-241.
    - Willmott, C.J., Robeson, S.M., Matsuura, K., 2012. A refined index of model performance.
      International Journal of Climatology 32(13) 2088-2094.
    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    a = (observed_array - simulated_array) ** 2
    b = np.abs(simulated_array - np.mean(observed_array))
    c = np.abs(observed_array - np.mean(observed_array))
    return 1 - (np.sum(a) / np.sum((b + c) ** 2))


def d1(simulated_array, observed_array, replace_nan=None, replace_inf=None,
       remove_neg=False, remove_zero=False):
    """

    Compute the the index of agreement (d1).

    .. image:: /pictures/d1.png

    **Range:** 0 ≤ d < 1, does not indicate bias, larger is better.

    **Notes:** This metric is a modified approach to the Nash-Sutcliffe Efficiency metric. Compared
    to the other index of agreement (d) it has a reduced impact of outliers.

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The index of agreement (d1).

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.d1(sim, obs)
    0.8434782608695652

    References
    ----------
    - Willmott, C.J., Robeson, S.M., Matsuura, K., 2012. A refined index of model performance.
      International Journal of Climatology 32(13) 2088-2094.
    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    obs_mean = np.mean(observed_array)

    a = np.sum(np.abs(simulated_array - observed_array))
    b = np.abs(simulated_array - obs_mean)
    c = np.abs(observed_array - obs_mean)
    return 1 - np.sum(a) / np.sum(b + c)


def dr(simulated_array, observed_array, replace_nan=None, replace_inf=None,
       remove_neg=False, remove_zero=False):
    """

    Compute the the refined index of agreement (dr).

    .. image:: /pictures/dr.png

    **Range:** -1 ≤ dr < 1, does not indicate bias, larger is better.

    **Notes:** Reformulation of Willmott’s index of agreement. This metric was created to address
    issues in the index of agreement and the Nash-Sutcliffe efficiency metric. Meant to be a
    flexible metric for use in climatology.

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The refined index of agreement.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.dr(sim, obs)
    0.847457627118644

    References
    ----------
    - Willmott, C.J., Robeson, S.M., Matsuura, K., 2012. A refined index of model performance.
      International Journal of Climatology 32(13) 2088-2094.
    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    a = np.sum(np.abs(simulated_array - observed_array))
    b = 2 * np.sum(np.abs(observed_array - observed_array.mean()))
    if a <= b:
        return 1 - (a / b)
    else:
        return (b / a) - 1


def drel(simulated_array, observed_array, replace_nan=None, replace_inf=None,
         remove_neg=False, remove_zero=False):
    """Compute the the relative index of agreement (drel).

    .. image:: /pictures/drel.png

    **Range:** 0 ≤ drel < 1, does not indicate bias, larger is better.

    **Notes:** Instead of absolute differences, this metric uses relative differences.

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The relative index of agreement.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.drel(sim, obs)
    0.9740868625579597

    References
    ----------
    - Krause, P., Boyle, D., Bäse, F., 2005. Comparison of different efficiency criteria for
      hydrological model assessment. Advances in geosciences 5 89-97.
    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    a = ((simulated_array - observed_array) / observed_array) ** 2
    b = np.abs(simulated_array - np.mean(observed_array))
    c = np.abs(observed_array - np.mean(observed_array))
    e = ((b + c) / np.mean(observed_array)) ** 2
    return 1 - (np.sum(a) / np.sum(e))


def dmod(simulated_array, observed_array, j=1, replace_nan=None, replace_inf=None,
         remove_neg=False, remove_zero=False):
    """

    Compute the the modified index of agreement (dmod).

    .. image:: /pictures/dmod.png

    **Range:** 0 ≤ dmod < 1, does not indicate bias, larger is better.

    **Notes:** When j=1, this metric is the same as d1. As j becomes larger, outliers have a larger
    impact on the value.

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

    j: int or float
        Optional input indicating the j values desired. A higher j places more emphasis on
        outliers. j is 1 by default.

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

    Returns
    -------
    float
        The modified index of agreement.

    Examples
    --------

    Note that using the default is the same as calculating the d1 metric. Changing the value of j
    modification of the metric.

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.dmod(sim, obs)  # Same as d1
    0.8434782608695652
    >>> he.dmod(sim, obs, j=1.5)
    0.9413310986805733

    References
    ----------

    - Krause, P., Boyle, D., Bäse, F., 2005. Comparison of different efficiency criteria for
      hydrological model assessment. Advances in geosciences 5 89-97.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    a = (np.abs(simulated_array - observed_array)) ** j
    b = np.abs(simulated_array - np.mean(observed_array))
    c = np.abs(observed_array - np.mean(observed_array))
    e = (b + c) ** j
    return 1 - (np.sum(a) / np.sum(e))


def watt_m(simulated_array, observed_array, replace_nan=None, replace_inf=None,
           remove_neg=False, remove_zero=False):
    """Compute Watterson's M (M).

    .. image:: /pictures/M.png

    **Range:** -1 ≤ M < 1, does not indicate bias, larger is better.

    **Notes:**

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        Watterson's M value.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.watt_m(sim, obs)
    0.8307913876595929

    References
    ----------
    - Watterson, I.G., 1996. Non‐dimensional measures of climate model performance. International
      Journal of Climatology 16(4) 379-391.
    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    a = 2 / np.pi
    b = np.mean((simulated_array - observed_array) ** 2)  # MSE
    c = np.std(observed_array, ddof=1) ** 2 + np.std(simulated_array, ddof=1) ** 2
    e = (np.mean(simulated_array) - np.mean(observed_array)) ** 2
    f = c + e
    return a * np.arcsin(1 - (b / f))


def mb_r(simulated_array, observed_array, replace_nan=None, replace_inf=None,
         remove_neg=False, remove_zero=False):
    """

    Compute Mielke-Berry R value (MB R).

    .. image:: /pictures/MB_R.png

    **Range:** 0 ≤ MB R < 1, does not indicate bias, larger is better.

    **Notes:** Compares prediction to probability it arose by chance.

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The Mielke-Berry R value.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.mb_r(sim, obs)
    0.7726315789473684

    References
    ----------
    - Berry, K.J., Mielke, P.W., 1988. A Generalization of Cohen's Kappa Agreement Measure to
      Interval Measurement and Multiple Raters. Educational and Psychological Measurement 48(4)
      921-933.
    - Mielke, P.W., Berry, K.J., 2007. Permutation methods: a distance function approach.
      Springer Science & Business Media.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    @njit(parallel=True)
    def numba_loop(simulated_array_numba, observed_array_numba):
        """Using numba for the double for loop"""
        assert len(observed_array_numba) == len(simulated_array_numba)
        size_numba = len(simulated_array_numba)
        total_numba = 0.
        for ii in prange(size_numba):
            observed = observed_array_numba[ii]
            for jj in prange(size_numba):
                total_numba += abs(simulated_array_numba[jj] - observed)
        return total_numba, size_numba

    # Using NumPy for the vectorized calculations
    total, size = numba_loop(simulated_array, observed_array)
    mae_value = np.mean(np.abs(simulated_array - observed_array))
    return 1 - (mae_value * size ** 2 / total)


def nse(simulated_array, observed_array, replace_nan=None, replace_inf=None,
        remove_neg=False, remove_zero=False):
    """Compute the Nash-Sutcliffe Efficiency.

    .. image:: /pictures/NSE.png

    **Range:** -inf < NSE < 1, does not indicate bias, larger is better.

    **Notes:** The Nash-Sutcliffe efficiency metric compares prediction values to naive predictions
    (i.e. average value). One major flaw of this metric is that it punishes a higher variance in
    the observed values (denominator). This metric is analogous to the mean absolute error skill
    score (MAESS) using the mean flow as a benchmark.

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The Nash-Sutcliffe Efficiency value.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.nse(sim, obs)
    0.922093023255814

    References
    ----------
    - Krause, P., Boyle, D., Bäse, F., 2005. Comparison of different efficiency criteria for
      hydrological model assessment. Advances in geosciences 5 89-97.
    - McCuen, R.H., Knight, Z., Cutter, A.G., 2006. Evaluation of the Nash-Sutcliffe Efficiency
      Index. Journal of Hydraulic Engineering.
    - Nash, J.E., Sutcliffe, J.V., 1970. River flow forecasting through conceptual models part
      I — A discussion of principles. Journal of Hydrology 282-290.
    - Willmott, C.J., Robeson, S.M., Matsuura, K., 2012. A refined index of model performance.
      International Journal of Climatology 32(13) 2088-2094.
    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    a = (np.abs(simulated_array - observed_array)) ** 2
    b = (np.abs(observed_array - np.mean(observed_array))) ** 2
    return 1 - (np.sum(a) / np.sum(b))


def nse_mod(simulated_array, observed_array, j=1, replace_nan=None, replace_inf=None,
            remove_neg=False, remove_zero=False):
    """Compute the modified Nash-Sutcliffe efficiency (NSE mod).

    .. image:: /pictures/NSEmod.png

    **Range:** -inf < NSE (mod) < 1, does not indicate bias, larger is better.

    **Notes:** The modified Nash-Sutcliffe efficiency metric gives less weight to outliers if j=1,
    or more weight to outliers if j is higher. Generally, j=1.

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

    j: int or float
        If given, sets the value of j to the input. j is 1 by default. A higher j gives more
        emphasis to outliers

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

    Returns
    -------
    float
        The modified Nash-Sutcliffe efficiency value.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.nse_mod(sim, obs)
    0.6949152542372882

    References
    ----------
    - Krause, P., Boyle, D., Bäse, F., 2005. Comparison of different efficiency criteria for
      hydrological model assessment. Advances in geosciences 5 89-97.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    a = (np.abs(simulated_array - observed_array)) ** j
    b = (np.abs(observed_array - np.mean(observed_array))) ** j
    return 1 - (np.sum(a) / np.sum(b))


def nse_rel(simulated_array, observed_array, replace_nan=None, replace_inf=None,
            remove_neg=False, remove_zero=False):
    """

    Compute the relative Nash-Sutcliffe efficiency (NSE rel).

    .. image:: /pictures/NSErel.png

    **Range:** -inf < NSE (rel) < 1, does not indicate bias, larger is better.

    **Notes:** The modified Nash-Sutcliffe efficiency metric gives less weight to outliers if j=1,
    or more weight to outliers if j is higher. Generally, j=1.

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The relative Nash-Sutcliffe efficiency value.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.nse_rel(sim, obs)
    0.9062004687708474

    References
    ----------
    - Krause, P., Boyle, D., Bäse, F., 2005. Comparison of different efficiency criteria for
      hydrological model assessment. Advances in geosciences 5 89-97.
    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    a = (np.abs((simulated_array - observed_array) / observed_array)) ** 2
    b = (np.abs((observed_array - np.mean(observed_array)) / np.mean(observed_array))) ** 2
    return 1 - (np.sum(a) / np.sum(b))


def kge_2009(simulated_array, observed_array, s=(1, 1, 1), replace_nan=None,
             replace_inf=None, remove_neg=False, remove_zero=False):
    """Compute the Kling-Gupta efficiency (2009).

    .. image:: /pictures/KGE_2009.png

    **Range:** -inf < KGE (2009) < 1, does not indicate bias, larger is better.

    **Notes:** Gupta et al. (2009) created this metric to demonstrate the relative importance of
    the three components of the NSE, which are correlation, bias and variability. This was done
    with hydrologic modeling as the context. This metric is meant to address issues with the NSE.

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

    s: tuple of length three
        Represents the scaling factors to be used for re-scaling the Pearson product-moment
        correlation coefficient (r), Alpha, and Beta, respectively.

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

    Returns
    -------
    float
        The Kling-Gupta (2009) efficiency value.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.kge_2009(sim, obs)
    0.912223072345668

    References
    ----------
    - Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009). Decomposition of the mean
      squared error and NSE performance criteria: Implications for improving hydrological modelling.
      Journal of Hydrology, 377(1-2), 80-91.
    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    # Means
    sim_mean = np.mean(simulated_array)
    obs_mean = np.mean(observed_array)

    # Standard Deviations
    sim_sigma = np.std(simulated_array, ddof=1)
    obs_sigma = np.std(observed_array, ddof=1)

    # Pearson R
    top_pr = np.sum((observed_array - obs_mean) * (simulated_array - sim_mean))
    bot1_pr = np.sqrt(np.sum((observed_array - obs_mean) ** 2))
    bot2_pr = np.sqrt(np.sum((simulated_array - sim_mean) ** 2))
    pr = top_pr / (bot1_pr * bot2_pr)

    # Ratio between mean of simulated and observed data
    beta = sim_mean / obs_mean

    # Relative variability between simulated and observed values
    alpha = sim_sigma / obs_sigma

    if obs_mean != 0 and obs_sigma != 0:
        kge = 1 - np.sqrt(
            (s[0] * (pr - 1)) ** 2 + (s[1] * (alpha - 1)) ** 2 + (s[2] * (beta - 1)) ** 2)
    else:
        if obs_mean == 0:
            warnings.warn(
                'Warning: The observed data mean is 0. Therefore, Beta is infinite and the KGE '
                'value cannot be computed.')
        if obs_sigma == 0:
            warnings.warn(
                'Warning: The observed data standard deviation is 0. Therefore, Alpha is infinite '
                'and the KGE value cannot be computed.')
        kge = np.nan

    return kge


def kge_2012(simulated_array, observed_array, s=(1, 1, 1), replace_nan=None,
             replace_inf=None, remove_neg=False, remove_zero=False):
    """

    Compute the Kling-Gupta efficiency (2012).

    .. image:: /pictures/KGE_2012.png

    **Range:** -inf < KGE (2012) < 1, does not indicate bias, larger is better.

    **Notes:** The modified version of the KGE (2009). Kling proposed this version to avoid
    cross-correlation between bias and variability ratios.

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

    s: tuple of length three
        Represents the scaling factors to be used for re-scaling the Pearson product-moment
        correlation coefficient (r), gamma, and Beta, respectively.

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

    Returns
    -------
    float
        The Kling-Gupta (2012) efficiency value.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.kge_2012(sim, obs)
    0.9122230723456678

    References
    ----------
    - Kling, H., Fuchs, M., & Paulin, M. (2012). Runoff conditions in the upper Danube basin under
      an ensemble of climate change scenarios. Journal of Hydrology, 424, 264-277.

    """
    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    # Means
    sim_mean = np.mean(simulated_array)
    obs_mean = np.mean(observed_array)

    # Standard Deviations
    sim_sigma = np.std(simulated_array)
    obs_sigma = np.std(observed_array)

    # Pearson R
    top_pr = np.sum((observed_array - obs_mean) * (simulated_array - sim_mean))
    bot1_pr = np.sqrt(np.sum((observed_array - obs_mean) ** 2))
    bot2_pr = np.sqrt(np.sum((simulated_array - sim_mean) ** 2))
    pr = top_pr / (bot1_pr * bot2_pr)

    # Ratio between mean of simulated and observed data
    beta = sim_mean / obs_mean

    # CV is the coefficient of variation (standard deviation / mean)
    sim_cv = sim_sigma / sim_mean
    obs_cv = obs_sigma / obs_mean

    # Variability Ratio, or the ratio of simulated CV to observed CV
    gam = sim_cv / obs_cv

    if obs_mean != 0 and obs_sigma != 0:
        kge = 1 - np.sqrt(
            (s[0] * (pr - 1)) ** 2 + (s[1] * (gam - 1)) ** 2 + (s[2] * (beta - 1)) ** 2)
    else:
        if obs_mean == 0:
            warnings.warn(
                'Warning: The observed data mean is 0. Therefore, Beta is infinite and the KGE '
                'value cannot be computed.')
        if obs_sigma == 0:
            warnings.warn(
                'Warning: The observed data standard deviation is 0. Therefore, Gamma is infinite '
                'and the KGE value cannot be computed.')
        kge = np.nan

    return kge


def lm_index(simulated_array, observed_array, obs_bar_p=None, replace_nan=None,
             replace_inf=None, remove_neg=False, remove_zero=False):
    """

    Compute the Legate-McCabe Efficiency Index.

    .. image:: /pictures/E1p.png

    **Range:** 0 ≤ E1' < 1, does not indicate bias, larger is better.

    **Notes:** The obs_bar_p argument represents a seasonal or other selected average.

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

    obs_bar_p: float
        Seasonal or other selected average. If None, the mean of the observed array will be used.

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

    Returns
    -------
    float
        The Legate-McCabe Efficiency index value.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.lm_index(sim, obs)
    0.6949152542372882

    References
    ----------
    - Legates, D.R., McCabe Jr, G.J., 1999. Evaluating the use of “goodness‐of‐fit” Measures in
      hydrologic and hydroclimatic model validation. Water Resources Research 35(1) 233-241.
      Lehmann, E.L., Casella, G., 1998. Springer Texts in Statistics. Springer-Verlag, New York.
    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    mean_obs = np.mean(observed_array)

    if obs_bar_p is not None:
        a = np.abs(simulated_array - observed_array)
        b = np.abs(observed_array - obs_bar_p)
        return 1 - (np.sum(a) / np.sum(b))
    else:
        a = np.abs(simulated_array - observed_array)
        b = np.abs(observed_array - mean_obs)
        return 1 - (np.sum(a) / np.sum(b))


def d1_p(simulated_array, observed_array, obs_bar_p=None, replace_nan=None,
         replace_inf=None, remove_neg=False, remove_zero=False):
    """Compute the Legate-McCabe Index of Agreement.

    .. image:: /pictures/D1p.png

    **Range:** 0 ≤ d1' < 1, does not indicate bias, larger is better.

    **Notes:** The obs_bar_p argument represents a seasonal or other selected average.

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

    obs_bar_p: float
        Seasonal or other selected average. If None, the mean of the observed array will be used.

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

    Returns
    -------
    float
        The Legate-McCabe Efficiency index of agreement.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.d1_p(sim, obs)
    0.8434782608695652

    References
    ----------
    - Legates, D.R., McCabe Jr, G.J., 1999. Evaluating the use of “goodness‐of‐fit” Measures in
      hydrologic and hydroclimatic model validation. Water Resources Research 35(1) 233-241.
      Lehmann, E.L., Casella, G., 1998. Springer Texts in Statistics. Springer-Verlag, New York.
    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    if obs_bar_p is not None:
        a = np.abs(observed_array - simulated_array)
        b = np.abs(simulated_array - obs_bar_p) + np.abs(observed_array - obs_bar_p)
        return 1 - (np.sum(a) / np.sum(b))
    else:
        mean_obs = np.mean(observed_array)
        a = np.abs(observed_array - simulated_array)
        b = np.abs(simulated_array - mean_obs) + np.abs(observed_array - mean_obs)
        return 1 - (np.sum(a) / np.sum(b))


def ve(simulated_array, observed_array, replace_nan=None, replace_inf=None,
       remove_neg=False, remove_zero=False):
    """

    Compute the Volumetric Efficiency (VE).

    .. image:: /pictures/VE.png

    **Range:** 0 ≤ VE < 1 smaller is better, does not indicate bias.

    **Notes:** Represents the error as a percentage of flow.

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The Volumetric Efficiency value.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.ve(sim, obs)
    0.8947368421052632

    References
    ----------
    - Criss, R.E., Winston, W.E., 2008. Do Nash values have value? Discussion and alternate
      proposals. Hydrological Processes 22(14) 2723.
    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    a = np.sum(np.abs(simulated_array - observed_array))
    b = np.sum(observed_array)
    return 1 - (a / b)


def sa(simulated_array, observed_array, replace_nan=None, replace_inf=None,
       remove_neg=False, remove_zero=False):
    """Compute the Spectral Angle (SA).

    .. image:: /pictures/SA.png

    **Range:** -π/2 ≤ SA < π/2, closer to 0 is better.

    **Notes:** The spectral angle metric measures the angle between the two vectors in hyperspace.
    It indicates how well the shape of the two series match – not magnitude.

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The Spectral Angle value.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.sa(sim, obs)
    0.10816831366492945

    References
    ----------
    - Robila, S.A., Gershman, A., 2005. Spectral matching accuracy in processing hyperspectral
      data, Signals, Circuits and Systems, 2005. ISSCS 2005. International Symposium on. IEEE,
      pp. 163-166.
    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    a = np.dot(simulated_array, observed_array)
    b = np.linalg.norm(simulated_array) * np.linalg.norm(observed_array)
    return np.arccos(a / b)


def sc(simulated_array, observed_array, replace_nan=None, replace_inf=None,
       remove_neg=False, remove_zero=False):
    """Compute the Spectral Correlation (SC).

    .. image:: /pictures/SC.png

    **Range:** -π/2 ≤ SA < π/2, closer to 0 is better.

    **Notes:** The spectral correlation metric measures the angle between the two vectors in
    hyperspace. It indicates how well the shape of the two series match – not magnitude.

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The Spectral Correlation value.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.sc(sim, obs)
    0.27991341383646606

    References
    ----------
    - Robila, S.A., Gershman, A., 2005. Spectral matching accuracy in processing hyperspectral
      data, Signals, Circuits and Systems, 2005. ISSCS 2005. International Symposium on. IEEE,
      pp. 163-166.
    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    a = np.dot(observed_array - np.mean(observed_array), simulated_array - np.mean(simulated_array))
    b = np.linalg.norm(observed_array - np.mean(observed_array))
    c = np.linalg.norm(simulated_array - np.mean(simulated_array))
    e = b * c
    return np.arccos(a / e)


def sid(simulated_array, observed_array, replace_nan=None, replace_inf=None,
        remove_neg=False, remove_zero=False):
    """Compute the Spectral Information Divergence (SID).

    .. image:: /pictures/SID.png

    **Range:** -π/2 ≤ SID < π/2, closer to 0 is better.

    **Notes:** The spectral information divergence measures the angle between the two vectors in
    hyperspace. It indicates how well the shape of the two series match – not magnitude.

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The Spectral information divergence value.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.sid(sim, obs)
    0.03517616895318012

    References
    ----------
    - Robila, S.A., Gershman, A., 2005. Spectral matching accuracy in processing hyperspectral
      data, Signals, Circuits and Systems, 2005. ISSCS 2005. International Symposium on. IEEE,
      pp. 163-166.
    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    first = (observed_array / np.mean(observed_array)) - (
            simulated_array / np.mean(simulated_array))
    second1 = np.log10(observed_array) - np.log10(np.mean(observed_array))
    second2 = np.log10(simulated_array) - np.log10(np.mean(simulated_array))
    return np.dot(first, second1 - second2)


def sga(simulated_array, observed_array, replace_nan=None, replace_inf=None,
        remove_neg=False, remove_zero=False):
    """Compute the Spectral Gradient Angle (SGA).

    .. image:: /pictures/SGA.png

    **Range:** -π/2 ≤ SID < π/2, closer to 0 is better.

    **Notes:** The spectral gradient angle measures the angle between the two vectors in
    hyperspace. It indicates how well the shape of the two series match – not magnitude.
    SG is the gradient of the simulated or observed time series.

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The Spectral Gradient Angle.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.sga(sim, obs)
    0.26764286472739834

    References
    ----------
    - Robila, S.A., Gershman, A., 2005. Spectral matching accuracy in processing hyperspectral
      data, Signals, Circuits and Systems, 2005. ISSCS 2005. International Symposium on. IEEE,
      pp. 163-166.
    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    sgx = observed_array[1:] - observed_array[:observed_array.size - 1]
    sgy = simulated_array[1:] - simulated_array[:simulated_array.size - 1]
    a = np.dot(sgx, sgy)
    b = np.linalg.norm(sgx) * np.linalg.norm(sgy)
    return np.arccos(a / b)


####################################################################################################
#               H Metrics: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985)               #
####################################################################################################


def h1_mhe(simulated_array, observed_array, replace_nan=None, replace_inf=None,
           remove_neg=False, remove_zero=False):
    """Compute the H1 mean error.

    .. image:: /pictures/H1.png
    .. image:: /pictures/MHE.png

    **Range:**

    **Notes:**

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The mean H1 error.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.h1_mhe(sim, obs)
    0.002106551840594386

    References
    ----------
    - Tornquist, L., Vartia, P., Vartia, Y.O., 1985. How Should Relative Changes be Measured?
      The American Statistician 43-46.
    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    h = (simulated_array - observed_array) / observed_array
    return np.mean(h)


def h1_mahe(simulated_array, observed_array, replace_nan=None, replace_inf=None,
            remove_neg=False, remove_zero=False):
    """

    Compute the H1 absolute error.

    .. image:: /pictures/H1.png
    .. image:: /pictures/AHE.png

    **Range:**

    **Notes:**

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The H1 absolute error.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.h1_mahe(sim, obs)
    0.11639226612630865

    References
    ----------
    - Tornquist, L., Vartia, P., Vartia, Y.O., 1985. How Should Relative Changes be Measured?
      The American Statistician 43-46.
    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    h = (simulated_array - observed_array) / observed_array
    return np.mean(np.abs(h))


def h1_rmshe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
             remove_zero=False):
    """Compute the H1 root mean square error.

    .. image:: /pictures/H1.png
    .. image:: /pictures/RMSHE.png

    **Range:**

    **Notes:**

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The root mean squared H1 error.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.h1_rmshe(sim, obs)
    0.12865571253672756

    References
    ----------
    - Tornquist, L., Vartia, P., Vartia, Y.O., 1985. How Should Relative Changes be Measured?
      The American Statistician 43-46.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    h = (simulated_array - observed_array) / observed_array
    return np.sqrt(np.mean(h ** 2))


def h2_mhe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
           remove_zero=False):
    """

    Compute the H2 mean error.

    .. image:: /pictures/H2.png
    .. image:: /pictures/MHE.png

    **Range:**

    **Notes:**

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The mean H2 error.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.h2_mhe(sim, obs)
    -0.015319829424307036

    References
    ----------
    - Tornquist, L., Vartia, P., Vartia, Y.O., 1985. How Should Relative Changes be Measured?
      The American Statistician 43-46.
    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    h = (simulated_array - observed_array) / simulated_array
    return np.mean(h)


def h2_mahe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
            remove_zero=False):
    """

    Compute the H2 mean absolute error.

    .. image:: /pictures/H2.png
    .. image:: /pictures/AHE.png

    **Range:**

    **Notes:**

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The mean absolute H2 error.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.h2_mahe(sim, obs)
    0.11997591408039167

    References
    ----------
    - Tornquist, L., Vartia, P., Vartia, Y.O., 1985. How Should Relative Changes be Measured?
      The American Statistician 43-46.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    h = (simulated_array - observed_array) / simulated_array
    return np.mean(np.abs(h))


def h2_rmshe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
             remove_zero=False):
    """

    Compute the H2 root mean square error.

    .. image:: /pictures/H1.png
    .. image:: /pictures/MHE.png

    **Range:**

    **Notes:**

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The root mean square H2 error.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.h2_rmshe(sim, obs)
    0.1373586680669673

    References
    ----------
    - Tornquist, L., Vartia, P., Vartia, Y.O., 1985. How Should Relative Changes be Measured?
      The American Statistician 43-46.

    """
    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    h = (simulated_array - observed_array) / simulated_array
    return np.sqrt(np.mean(h ** 2))


def h3_mhe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
           remove_zero=False):
    """

    Compute the H3 mean error.

    .. image:: /pictures/H3.png
    .. image:: /pictures/MHE.png

    **Range:**

    **Notes:**

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The mean H3 error.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.h3_mhe(sim, obs)
    -0.006322019630356533

    References
    ----------
    - Tornquist, L., Vartia, P., Vartia, Y.O., 1985. How Should Relative Changes be Measured?
      The American Statistician 43-46.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    h = (simulated_array - observed_array) / (0.5 * (simulated_array + observed_array))
    return np.mean(h)


def h3_mahe(simulated_array, observed_array, replace_nan=None, replace_inf=None,
            remove_neg=False, remove_zero=False):
    """

    Compute the H3 mean absolute error.

    .. image:: /pictures/H3.png
    .. image:: /pictures/AHE.png

    **Range:**

    **Notes:**

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The mean absolute H3 error.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.h3_mahe(sim, obs)
    0.11743831388794855

    References
    ----------
    - Tornquist, L., Vartia, P., Vartia, Y.O., 1985. How Should Relative Changes be Measured?
      The American Statistician 43-46.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    h = (simulated_array - observed_array) / (0.5 * (simulated_array + observed_array))
    return np.mean(np.abs(h))


def h3_rmshe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
             remove_zero=False):
    """

    Compute the H3 root mean square error.

    .. image:: /pictures/H3.png
    .. image:: /pictures/RMSHE.png

    **Range:**

    **Notes:**

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The root mean square H3 error.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.h3_rmshe(sim, obs)
    0.13147667616722278

    References
    ----------
    - Tornquist, L., Vartia, P., Vartia, Y.O., 1985. How Should Relative Changes be Measured?
      The American Statistician 43-46.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    h = (simulated_array - observed_array) / (0.5 * (simulated_array + observed_array))
    return np.sqrt(np.mean(h ** 2))


def h4_mhe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
           remove_zero=False):
    """

    Compute the H4 mean error.

    .. image:: /pictures/H4.png
    .. image:: /pictures/MHE.png

    **Range:**

    **Notes:**

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The mean H4 error.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.h4_mhe(sim, obs)
    -0.0064637371129817

    References
    ----------
    - Tornquist, L., Vartia, P., Vartia, Y.O., 1985. How Should Relative Changes be Measured?
      The American Statistician 43-46.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    h = (simulated_array - observed_array) / np.sqrt(simulated_array * observed_array)
    return np.mean(h)


def h4_mahe(simulated_array, observed_array, replace_nan=None, replace_inf=None,
            remove_neg=False, remove_zero=False):
    """

    Compute the H4 mean absolute error.

    .. image:: /pictures/H4.png
    .. image:: /pictures/AHE.png

    **Range:**

    **Notes:**

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The mean absolute H4 error.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.h4_mahe(sim, obs)
    0.11781032209144082

    References
    ----------
    - Tornquist, L., Vartia, P., Vartia, Y.O., 1985. How Should Relative Changes be Measured?
      The American Statistician 43-46.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    h = (simulated_array - observed_array) / np.sqrt(simulated_array * observed_array)
    return np.mean(np.abs(h))


def h4_rmshe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
             remove_zero=False):
    """

    Compute the H4 mean error.

    .. image:: /pictures/H4.png
    .. image:: /pictures/RMSHE.png

    **Range:**

    **Notes:**

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The root mean square H4 error.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.h4_rmshe(sim, obs)
    0.13200901963465006

    References
    ----------
    - Tornquist, L., Vartia, P., Vartia, Y.O., 1985. How Should Relative Changes be Measured?
      The American Statistician 43-46.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    h = (simulated_array - observed_array) / np.sqrt(simulated_array * observed_array)
    return np.sqrt(np.mean(h ** 2))


def h5_mhe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
           remove_zero=False):
    """

    Compute the H5 mean error.

    .. image:: /pictures/H5.png
    .. image:: /pictures/MHE.png

    **Range:**

    **Notes:**

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The mean H5 error.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.h5_mhe(sim, obs)
    -0.006606638791856322

    References
    ----------
    - Tornquist, L., Vartia, P., Vartia, Y.O., 1985. How Should Relative Changes be Measured?
      The American Statistician 43-46.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    top = (simulated_array - observed_array)
    bot = np.reciprocal(0.5 * (np.reciprocal(observed_array) + np.reciprocal(simulated_array)))
    h = top / bot
    return np.mean(h)


def h5_mahe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
            remove_zero=False):
    """

    Compute the H5 mean absolute error.

    .. image:: /pictures/H5.png
    .. image:: /pictures/AHE.png

    **Range:**

    **Notes:**

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The mean absolute H5 error.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.h5_mahe(sim, obs)
    0.11818409010335018

    References
    ----------
    - Tornquist, L., Vartia, P., Vartia, Y.O., 1985. How Should Relative Changes be Measured?
      The American Statistician 43-46.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    top = (simulated_array - observed_array)
    bot = np.reciprocal(0.5 * (np.reciprocal(observed_array) + np.reciprocal(simulated_array)))
    h = top / bot
    return np.mean(np.abs(h))


def h5_rmshe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
             remove_zero=False):
    """

    Compute the H5 root mean square error.

    .. image:: /pictures/H5.png
    .. image:: /pictures/RMSHE.png

    **Range:**

    **Notes:**

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The root mean square H5 error.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.h5_rmshe(sim, obs)
    0.13254476469410933

    References
    ----------
    - Tornquist, L., Vartia, P., Vartia, Y.O., 1985. How Should Relative Changes be Measured?
      The American Statistician 43-46.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    top = (simulated_array - observed_array)
    bot = np.reciprocal(0.5 * (np.reciprocal(observed_array) + np.reciprocal(simulated_array)))
    h = top / bot
    return np.sqrt(np.mean(h ** 2))


def h6_mhe(simulated_array, observed_array, k=1, replace_nan=None, replace_inf=None,
           remove_neg=False, remove_zero=False):
    """

    Compute the H6 mean error.

    .. image:: /pictures/H6.png
    .. image:: /pictures/MHE.png

    **Range:**

    **Notes:**

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

    k: int or float
        If given, sets the value of k. If None, k=1.

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

    Returns
    -------
    float
        The mean H6 error.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.h6_mhe(sim, obs)
    -0.006322019630356514

    References
    ----------
    - Tornquist, L., Vartia, P., Vartia, Y.O., 1985. How Should Relative Changes be Measured?
      The American Statistician 43-46.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    top = (simulated_array / observed_array - 1)
    bot = np.power(0.5 * (1 + np.power(simulated_array / observed_array, k)), 1 / k)
    h = top / bot
    return np.mean(h)


def h6_mahe(simulated_array, observed_array, k=1, replace_nan=None, replace_inf=None,
            remove_neg=False,
            remove_zero=False):
    """Compute the H6 mean absolute error.

    .. image:: /pictures/H6.png
    .. image:: /pictures/AHE.png

    **Range:**

    **Notes:**

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

    k: int or float
        If given, sets the value of k. If None, k=1.

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

    Returns
    -------
    float
        The mean absolute H6 error.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.h6_mahe(sim, obs)
    0.11743831388794852

    References
    ----------
    - Tornquist, L., Vartia, P., Vartia, Y.O., 1985. How Should Relative Changes be Measured?
      The American Statistician 43-46.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    top = (simulated_array / observed_array - 1)
    bot = np.power(0.5 * (1 + np.power(simulated_array / observed_array, k)), 1 / k)
    h = top / bot
    return np.mean(np.abs(h))


def h6_rmshe(simulated_array, observed_array, k=1, replace_nan=None, replace_inf=None,
             remove_neg=False,
             remove_zero=False):
    """

    Compute the H6 root mean square error.

    .. image:: /pictures/H6.png
    .. image:: /pictures/RMSHE.png

    **Range:**

    **Notes:**

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

    k: int or float
        If given, sets the value of k. If None, k=1.

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

    Returns
    -------
    float
        The root mean square H6 error.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.h6_rmshe(sim, obs)
    0.13147667616722278

    References
    ----------
    - Tornquist, L., Vartia, P., Vartia, Y.O., 1985. How Should Relative Changes be Measured?
      The American Statistician 43-46.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    top = (simulated_array / observed_array - 1)
    bot = np.power(0.5 * (1 + np.power(simulated_array / observed_array, k)), 1 / k)
    h = top / bot
    return np.sqrt(np.mean(h ** 2))


def h7_mhe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
           remove_zero=False):
    """

    Compute the H7 mean error.

    .. image:: /pictures/H7.png
    .. image:: /pictures/MHE.png

    **Range:**

    **Notes:**

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The mean H7 error.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.h7_mhe(sim, obs)
    0.0026331898007430263

    References
    ----------
    - Tornquist, L., Vartia, P., Vartia, Y.O., 1985. How Should Relative Changes be Measured?
      The American Statistician 43-46.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    h = (simulated_array / observed_array - 1) / np.min(simulated_array / observed_array)
    return np.mean(h)


def h7_mahe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
            remove_zero=False):
    """

    Compute the H7 mean absolute error.

    .. image:: /pictures/H7.png
    .. image:: /pictures/AHE.png

    **Range:**

    **Notes:**

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The mean absolute H7 error.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.h7_mahe(sim, obs)
    0.14549033265788583

    References
    ----------
    - Tornquist, L., Vartia, P., Vartia, Y.O., 1985. How Should Relative Changes be Measured?
      The American Statistician 43-46.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    h = (simulated_array / observed_array - 1) / np.min(simulated_array / observed_array)
    return np.mean(np.abs(h))


def h7_rmshe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
             remove_zero=False):
    """

    Compute the H7 root mean square error.

    .. image:: /pictures/H7.png
    .. image:: /pictures/RMSHE.png

    **Range:**

    **Notes:**

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The root mean square H7 error.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.h7_rmshe(sim, obs)
    0.16081964067090945

    References
    ----------
    - Tornquist, L., Vartia, P., Vartia, Y.O., 1985. How Should Relative Changes be Measured?
      The American Statistician 43-46.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    h = (simulated_array / observed_array - 1) / np.min(simulated_array / observed_array)
    return np.sqrt(np.mean(h ** 2))


def h8_mhe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
           remove_zero=False):
    """Compute the H8 mean error.

    .. image:: /pictures/H8.png
    .. image:: /pictures/MHE.png

    **Range:**

    **Notes:**

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The mean H8 error.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.h8_mhe(sim, obs)
    0.0018056158633666466

    References
    ----------
    - Tornquist, L., Vartia, P., Vartia, Y.O., 1985. How Should Relative Changes be Measured?
      The American Statistician 43-46.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    h = (simulated_array / observed_array - 1) / np.max(simulated_array / observed_array)
    return np.mean(h)


def h8_mahe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
            remove_zero=False):
    """

    Compute the H8 mean absolute error.

    .. image:: /pictures/H8.png
    .. image:: /pictures/AHE.png

    **Range:**

    **Notes:**

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The mean absolute H8 error.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.h8_mahe(sim, obs)
    0.099764799536836

    References
    ----------
    - Tornquist, L., Vartia, P., Vartia, Y.O., 1985. How Should Relative Changes be Measured?
      The American Statistician 43-46.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    h = (simulated_array / observed_array - 1) / np.max(simulated_array / observed_array)
    return np.mean(np.abs(h))


def h8_rmshe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
             remove_zero=False):
    """

    Compute the H8 root mean square error.

    .. image:: /pictures/H8.png
    .. image:: /pictures/RMSHE.png

    **Range:**

    **Notes:**

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The root mean square H8 error.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.h8_rmshe(sim, obs)
    0.11027632503148076

    References
    ----------
    - Tornquist, L., Vartia, P., Vartia, Y.O., 1985. How Should Relative Changes be Measured?
      The American Statistician 43-46.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    h = (simulated_array / observed_array - 1) / np.max(simulated_array / observed_array)
    return np.sqrt(np.mean(h ** 2))


# def h9(simulated_array, observed_array, h_type='mhe', k=1):
#     h = (simulated_array / observed_array - 1) / \
#         np.power(0.5 * (1 + np.power(simulated_array / observed_array, k)), 1 / k)
#     if h_type == 'mhe':
#         return h.mean()
#     elif h_type == 'ahe':
#         return np.abs(h).mean()
#     elif h_type == 'rmshe':
#         return np.sqrt((h**2).mean())
#     else:
#         raise HydrostatsError("The three types available are 'mhe', 'ahe', and 'rmshe'.")


def h10_mhe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
            remove_zero=False):
    """

    Compute the H10 mean error.

    .. image:: /pictures/H10.png
    .. image:: /pictures/MHE.png

    **Range:**

    **Notes:**

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The mean H10 error.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.h10_mhe(sim, obs)
    -0.0012578676058971154

    References
    ----------
    - Tornquist, L., Vartia, P., Vartia, Y.O., 1985. How Should Relative Changes be Measured?
      The American Statistician 43-46.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    h = np.log1p(simulated_array) - np.log1p(observed_array)
    return np.mean(h)


def h10_mahe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
             remove_zero=False):
    """

    Compute the H10 mean absolute error.

    .. image:: /pictures/H10.png
    .. image:: /pictures/AHE.png

    **Range:**

    **Notes:**

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The mean absolute H10 error.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> np.round(he.h10_mahe(sim, obs), 6)
    0.094636

    References
    ----------
    - Tornquist, L., Vartia, P., Vartia, Y.O., 1985. How Should Relative Changes be Measured?
      The American Statistician 43-46.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    h = np.log1p(simulated_array) - np.log1p(observed_array)
    return np.mean(np.abs(h))


def h10_rmshe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
              remove_zero=False):
    """

    Compute the H10 root mean square error.

    .. image:: /pictures/H10.png
    .. image:: /pictures/RMSHE.png

    **Range:**

    **Notes:**

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The root mean square H10 error.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> np.round(he.h10_rmshe(sim, obs), 6)
    0.103161

    References
    ----------
    - Tornquist, L., Vartia, P., Vartia, Y.O., 1985. How Should Relative Changes be Measured?
      The American Statistician 43-46.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    h = np.log1p(simulated_array) - np.log1p(observed_array)
    return np.sqrt(np.mean(h ** 2))


###################################################################################################
#                         Statistical Error Metrics for Distribution Testing                      #
###################################################################################################


def g_mean_diff(simulated_array, observed_array, replace_nan=None, replace_inf=None,
                remove_neg=False,
                remove_zero=False):
    """

    Compute the geometric mean difference.

    .. image:: /pictures/GMD.png

    **Range:**

    **Notes:** For the difference of geometric means, the geometric mean is computed for each of
    two samples then their difference is taken.

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The geometric mean difference value.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> he.g_mean_diff(sim, obs)
    0.988855412098022

    References
    ----------

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    sim_log = np.log1p(simulated_array)
    obs_log = np.log1p(observed_array)
    return np.exp(gmean(sim_log) - gmean(obs_log))


def mean_var(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
             remove_zero=False):
    """

    Compute the mean variance.

    .. image:: /pictures/MV.png

    **Range:**

    **Notes:**

    Parameters
    ----------
    simulated_array: one dimensional ndarray
        An array of simulated data from the time series.

    observed_array: one dimensional ndarray
        An array of observed data from the time series.

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

    Returns
    -------
    float
        The mean variance.

    Examples
    --------

    >>> import hydrostats.HydroErr as he
    >>> import numpy as np

    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 7])
    >>> np.round(he.mean_var(sim, obs), 6)
    0.010641

    References
    ----------

    """
    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = treat_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    return np.var(np.log1p(observed_array) - np.log1p(simulated_array))


####################################################################################################
#                                      HELPER FUNCTIONS                                            #
####################################################################################################


class HydrostatsError(Exception):
    pass


metric_names = [
    'Mean Error', 'Mean Absolute Error', 'Mean Squared Error', 'Mean Log Error',
    'Mean Absolute Log Error', 'Mean Squared Log Error', 'Median Error', 'Median Absolute Error',
    'Median Squared Error', 'Eclidean Distance', 'Normalized Eclidean Distance',
    'Root Mean Square Error', 'Root Mean Squared Log Error',
    'Normalized Root Mean Square Error - Range', 'Normalized Root Mean Square Error - Mean',
    'Normalized Root Mean Square Error - IQR', 'Inertial Root Mean Square Error',
    'Mean Absolute Scaled Error', 'Coefficient of Determination', 'Pearson Correlation Coefficient',
    'Spearman Rank Correlation Coefficient', 'Anomaly Correlation Coefficient',
    'Mean Absolute Percentage Error', 'Mean Absolute Percentage Deviation',
    'Mean Arctangent Absolute Percentage Error', 'Symmetric Mean Absolute Percentage Error (1)',
    'Symmetric Mean Absolute Percentage Error (2)', 'Index of Agreement (d)',
    'Index of Agreement (d1)', 'Modified Index of Agreement', 'Relative Index of Agreement',
    'Index of Agreement Refined (dr)', "Watterson's M", 'Mielke-Berry R',
    'Nash-Sutcliffe Efficiency', 'Modified Nash-Sutcliffe Efficiency',
    'Relative Nash-Sutcliffe Efficiency', 'Kling-Gupta Efficiency (2009)',
    'Kling-Gupta Efficiency (2012)', 'Legate-McCabe Efficiency Index',
    'Legate-McCabe Index of Agreement', 'Volumetric Efficiency', 'Spectral Angle',
    'Spectral Correlation', 'Spectral Information Divergence', 'Spectral Gradient Angle',
    'Mean H1 Error', 'Mean Absolute H1 Error', 'Root Mean Square H1 Error', 'Mean H2 Error',
    'Mean Absolute H2 Error', 'Root Mean Square H2 Error', 'Mean H3 Error',
    'Mean Absolute H3 Error', 'Root Mean Square H3 Error', 'Mean H4 Error',
    'Mean Absolute H4 Error', 'Root Mean Square H4 Error', 'Mean H5 Error',
    'Mean Absolute H5 Error', 'Root Mean Square H5 Error', 'Mean H6 Error',
    'Mean Absolute H6 Error', 'Root Mean Square H6 Error', 'Mean H7 Error',
    'Mean Absolute H7 Error', 'Root Mean Square H7 Error', 'Mean H8 Error',
    'Mean Absolute H8 Error', 'Root Mean Square H8 Error', 'Mean H10 Error',
    'Mean Absolute H10 Error', 'Root Mean Square H10 Error', 'Geometric Mean Difference',
    'Mean Variance'
]

metric_abbr = [
    'ME', 'MAE', 'MSE', 'MLE', 'MALE', 'MSLE', 'MdE', 'MdAE', 'MdSE', 'ED', 'NED', 'RMSE', 'RMSLE',
    'NRMSE (Range)', 'NRMSE (Mean)', 'NRMSE (IQR)', 'IRMSE', 'MASE', 'r2', 'R (Pearson)',
    'R (Spearman)', 'ACC', 'MAPE', 'MAPD', 'MAAPE', 'SMAPE1', 'SMAPE2', 'd', 'd1', 'd (Mod.)',
    'd (Rel.)', 'dr', 'M', '(MB) R', 'NSE', 'NSE (Mod.)', 'NSE (Rel.)', 'KGE (2009)', 'KGE (2012)',
    "E1'", "D1'", 'VE', 'SA', 'SC', 'SID', 'SGA', 'H1 (MHE)', 'H1 (MAHE)', 'H1 (RMSHE)', 'H2 (MHE)',
    'H2 (MAHE)', 'H2 (RMSHE)', 'H3 (MHE)', 'H3 (MAHE)', 'H3 (RMSHE)', 'H4 (MHE)', 'H4 (MAHE)',
    'H4 (RMSHE)', 'H5 (MHE)', 'H5 (MAHE)', 'H5 (RMSHE)', 'H6 (MHE)', 'H6 (MAHE)', 'H6 (RMSHE)',
    'H7 (MHE)', 'H7 (MAHE)', 'H7 (RMSHE)', 'H8 (MHE)', 'H8 (MAHE)', 'H8 (RMSHE)', 'H10 (MHE)',
    'H10 (MAHE)', 'H10 (RMSHE)', 'GMD', 'MV'
]

function_list = [
    me, mae, mse, mle, male, msle, mde, mdae, mdse, ed, ned, rmse, rmsle, nrmse_range, nrmse_mean,
    nrmse_iqr, irmse, mase, r_squared, pearson_r, spearman_r, acc, mape, mapd, maape, smape1,
    smape2, d, d1, dmod, drel, dr, watt_m, mb_r, nse, nse_mod, nse_rel, kge_2009, kge_2012,
    lm_index, d1_p, ve, sa, sc, sid, sga, h1_mhe, h1_mahe, h1_rmshe, h2_mhe, h2_mahe, h2_rmshe,
    h3_mhe, h3_mahe, h3_rmshe, h4_mhe, h4_mahe, h4_rmshe, h5_mhe, h5_mahe, h5_rmshe, h6_mhe,
    h6_mahe, h6_rmshe, h7_mhe, h7_mahe, h7_rmshe, h8_mhe, h8_mahe, h8_rmshe, h10_mhe, h10_mahe,
    h10_rmshe, g_mean_diff, mean_var,
]


def treat_values(simulated_array, observed_array, replace_nan=None, replace_inf=None,
                 remove_neg=False, remove_zero=False):
    """Removes the nan, negative, and inf values in two numpy arrays"""
    sim_copy = np.copy(simulated_array)
    obs_copy = np.copy(observed_array)

    # Checking to see if the vectors are the same length
    assert sim_copy.ndim == 1, "The simulated array is not one dimensional."
    assert obs_copy.ndim == 1, "The observed array is not one dimensional."

    if sim_copy.size != obs_copy.size:
        raise HydrostatsError("The two ndarrays are not the same size.")

    # Treat missing data in observed_array and simulated_array, rows in simulated_array or
    # observed_array that contain nan values
    all_treatment_array = np.ones(obs_copy.size, dtype=bool)

    if np.any(np.isnan(obs_copy)) or np.any(np.isnan(sim_copy)):
        if replace_nan is not None:
            # Finding the NaNs
            sim_nan = np.isnan(sim_copy)
            obs_nan = np.isnan(obs_copy)
            # Replacing the NaNs with the input
            sim_copy[sim_nan] = replace_nan
            obs_copy[obs_nan] = replace_nan

            warnings.warn("Elements(s) {} contained NaN values in the simulated array and "
                          "elements(s) {} contained NaN values in the observed array and have been "
                          "replaced (Elements are zero indexed).".format(np.where(sim_nan)[0],
                                                                         np.where(obs_nan)[0]),
                          UserWarning)
        else:
            # Getting the indices of the nan values, combining them, and informing user.
            nan_indices_fcst = ~np.isnan(sim_copy)
            nan_indices_obs = ~np.isnan(obs_copy)
            all_nan_indices = np.logical_and(nan_indices_fcst, nan_indices_obs)
            all_treatment_array = np.logical_and(all_treatment_array, all_nan_indices)

            warnings.warn("Row(s) {} contained NaN values and the row(s) have been "
                          "removed (Rows are zero indexed).".format(np.where(~all_nan_indices)[0]),
                          UserWarning)

    if np.any(np.isinf(obs_copy)) or np.any(np.isinf(sim_copy)):
        if replace_nan is not None:
            # Finding the NaNs
            sim_inf = np.isinf(sim_copy)
            obs_inf = np.isinf(obs_copy)
            # Replacing the NaNs with the input
            sim_copy[sim_inf] = replace_inf
            obs_copy[obs_inf] = replace_inf

            warnings.warn("Elements(s) {} contained NaN values in the simulated array and "
                          "elements(s) {} contained NaN values in the observed array and have been "
                          "replaced (Elements are zero indexed).".format(np.where(sim_inf)[0],
                                                                         np.where(obs_inf)[0]),
                          UserWarning)
        else:
            inf_indices_fcst = ~(np.isinf(sim_copy))
            inf_indices_obs = ~np.isinf(obs_copy)
            all_inf_indices = np.logical_and(inf_indices_fcst, inf_indices_obs)
            all_treatment_array = np.logical_and(all_treatment_array, all_inf_indices)

            warnings.warn(
                "Row(s) {} contained Inf or -Inf values and the row(s) have been removed (Rows "
                "are zero indexed).".format(np.where(~all_inf_indices)[0]),
                UserWarning
            )

    # Treat zero data in observed_array and simulated_array, rows in simulated_array or
    # observed_array that contain zero values
    if remove_zero:
        if (obs_copy == 0).any() or (sim_copy == 0).any():
            zero_indices_fcst = ~(sim_copy == 0)
            zero_indices_obs = ~(obs_copy == 0)
            all_zero_indices = np.logical_and(zero_indices_fcst, zero_indices_obs)
            all_treatment_array = np.logical_and(all_treatment_array, all_zero_indices)

            warnings.warn(
                "Row(s) {} contained zero values and the row(s) have been removed (Rows are "
                "zero indexed).".format(np.where(~all_zero_indices)[0]),
                UserWarning
            )

    # Treat negative data in observed_array and simulated_array, rows in simulated_array or
    # observed_array that contain negative values
    warnings.filterwarnings("ignore")  # Ignore runtime warnings from comparing
    if remove_neg:
        if (obs_copy < 0).any() or (sim_copy < 0).any():
            neg_indices_fcst = ~(sim_copy < 0)
            neg_indices_obs = ~(obs_copy < 0)
            all_neg_indices = np.logical_and(neg_indices_fcst, neg_indices_obs)
            all_treatment_array = np.logical_and(all_treatment_array, all_neg_indices)

            warnings.filterwarnings("always")

            warnings.warn("Row(s) {} contained negative values and the row(s) have been "
                          "removed (Rows are zero indexed).".format(np.where(~all_neg_indices)[0]),
                          UserWarning)
        else:
            warnings.filterwarnings("always")
    else:
        warnings.filterwarnings("always")

    obs_copy = obs_copy[all_treatment_array]
    sim_copy = sim_copy[all_treatment_array]

    return sim_copy, obs_copy


def list_of_metrics(metrics, sim_array, obs_array, abbr=False, mase_m=1, dmod_j=1,
                    nse_mod_j=1, h6_mhe_k=1, h6_ahe_k=1, h6_rmshe_k=1, d1_p_obs_bar_p=None,
                    lm_x_obs_bar_p=None, replace_nan=None, replace_inf=None, remove_neg=False,
                    remove_zero=False):
    if len(sim_array.shape) != 1 or len(obs_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if sim_array.size != obs_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")

    metrics_list = []

    if not abbr:
        for metric in metrics:
            if metric == 'Mean Absolute Scaled Error':
                metrics_list.append(mase(sim_array, obs_array, m=mase_m,
                                         replace_nan=replace_nan, replace_inf=replace_inf,
                                         remove_neg=remove_neg, remove_zero=remove_zero))

            elif metric == 'Modified Index of Agreement':
                metrics_list.append(dmod(sim_array, obs_array, j=dmod_j,
                                         replace_nan=replace_nan, replace_inf=replace_inf,
                                         remove_neg=remove_neg, remove_zero=remove_zero))

            elif metric == 'Modified Nash-Sutcliffe Efficiency':
                metrics_list.append(nse_mod(sim_array, obs_array, j=nse_mod_j,
                                            replace_nan=replace_nan, replace_inf=replace_inf,
                                            remove_neg=remove_neg, remove_zero=remove_zero))

            elif metric == 'Legate-McCabe Efficiency Index':
                metrics_list.append(lm_index(sim_array, obs_array, obs_bar_p=lm_x_obs_bar_p,
                                             replace_nan=replace_nan, replace_inf=replace_inf,
                                             remove_neg=remove_neg, remove_zero=remove_zero))

            elif metric == 'Mean H6 Error':
                metrics_list.append(h6_mhe(sim_array, obs_array, k=h6_mhe_k,
                                           replace_nan=replace_nan, replace_inf=replace_inf,
                                           remove_neg=remove_neg, remove_zero=remove_zero
                                           ))

            elif metric == 'Mean Absolute H6 Error':
                metrics_list.append(h6_mahe(sim_array, obs_array, k=h6_ahe_k,
                                            replace_nan=replace_nan, replace_inf=replace_inf,
                                            remove_neg=remove_neg, remove_zero=remove_zero
                                            ))

            elif metric == 'Root Mean Square H6 Error':
                metrics_list.append(h6_rmshe(sim_array, obs_array, k=h6_rmshe_k,
                                             replace_nan=replace_nan, replace_inf=replace_inf,
                                             remove_neg=remove_neg, remove_zero=remove_zero
                                             ))

            elif metric == 'Legate-McCabe Index of Agreement':
                metrics_list.append(d1_p(sim_array, obs_array, obs_bar_p=d1_p_obs_bar_p,
                                         replace_nan=replace_nan, replace_inf=replace_inf,
                                         remove_neg=remove_neg, remove_zero=remove_zero
                                         ))
            else:
                index = metric_names.index(metric)
                metric_func = function_list[index]
                metrics_list.append(metric_func(sim_array, obs_array, replace_nan=replace_nan,
                                                replace_inf=replace_inf, remove_neg=remove_neg,
                                                remove_zero=remove_zero))

    else:
        for metric in metrics:
            if metric == 'MASE':
                metrics_list.append(mase(sim_array, obs_array, m=mase_m,
                                         replace_nan=replace_nan, replace_inf=replace_inf,
                                         remove_neg=remove_neg, remove_zero=remove_zero))

            elif metric == 'd (Mod.)':
                metrics_list.append(dmod(sim_array, obs_array, j=dmod_j,
                                         replace_nan=replace_nan, replace_inf=replace_inf,
                                         remove_neg=remove_neg, remove_zero=remove_zero))

            elif metric == 'NSE (Mod.)':
                metrics_list.append(nse_mod(sim_array, obs_array, j=nse_mod_j,
                                            replace_nan=replace_nan,
                                            replace_inf=replace_inf,
                                            remove_neg=remove_neg, remove_zero=remove_zero))

            elif metric == "E1'":
                metrics_list.append(lm_index(sim_array, obs_array, obs_bar_p=lm_x_obs_bar_p,
                                             replace_nan=replace_nan,
                                             replace_inf=replace_inf,
                                             remove_neg=remove_neg,
                                             remove_zero=remove_zero))

            elif metric == 'H6 (MHE)':
                metrics_list.append(h6_mhe(sim_array, obs_array, k=h6_mhe_k,
                                           replace_nan=replace_nan, replace_inf=replace_inf,
                                           remove_neg=remove_neg, remove_zero=remove_zero
                                           ))

            elif metric == 'H6 (AHE)':
                metrics_list.append(h6_mahe(sim_array, obs_array, k=h6_ahe_k,
                                            replace_nan=replace_nan, replace_inf=replace_inf,
                                            remove_neg=remove_neg, remove_zero=remove_zero
                                            ))

            elif metric == 'H6 (RMSHE)':
                metrics_list.append(h6_rmshe(sim_array, obs_array, k=h6_rmshe_k,
                                             replace_nan=replace_nan,
                                             replace_inf=replace_inf,
                                             remove_neg=remove_neg, remove_zero=remove_zero
                                             ))

            elif metric == "D1'":
                metrics_list.append(d1_p(sim_array, obs_array, obs_bar_p=d1_p_obs_bar_p,
                                         replace_nan=replace_nan, replace_inf=replace_inf,
                                         remove_neg=remove_neg, remove_zero=remove_zero
                                         ))
            else:
                index = metric_abbr.index(metric)
                metric_func = function_list[index]
                metrics_list.append(
                    metric_func(sim_array, obs_array, replace_nan=replace_nan,
                                replace_inf=replace_inf, remove_neg=remove_neg,
                                remove_zero=remove_zero))
    return metrics_list


if __name__ == "__main__":
    pass
