# python 3.6
# -*- coding: utf-8 -*-
"""
Created on Dec 28 12:16:32 2017
@author: Wade Roberts
"""
from __future__ import division
import numpy as np
from numba import njit, prange
from scipy.stats import gmean, rankdata
import warnings

__all__ = ['me', 'mae', 'mse', 'ed', 'ned', 'rmse', 'rmsle', 'mase', 'r_squared', 'acc', 'mape', 'mapd', 'smape1',
           'smape2', 'd', 'd1', 'dr', 'drel', 'dmod', 'watt_m', 'mb_r', 'nse', 'nse_mod', 'nse_rel', 'lm_index', 'sa',
           'sc', 'sid', 'sga', 'h1_mhe', 'h1_ahe', 'h1_rmshe', 'h2_mhe', 'h2_ahe', 'h2_rmshe', 'h3_mhe', 'h3_ahe',
           'h3_rmshe', 'h4_mhe', 'h4_ahe', 'h4_rmshe', 'h5_mhe', 'h5_ahe', 'h5_rmshe', 'h6_mhe', 'h6_ahe', 'h6_rmshe',
           'h7_mhe', 'h7_ahe', 'h7_rmshe', 'h8_mhe', 'h8_ahe', 'h8_rmshe', 'h10_mhe', 'h10_ahe', 'h10_rmshe',
           'g_mean_diff', 'mean_var', 'mle', 'male', 'msle', 'nrmse_range', 'nrmse_mean', 'nrmse_iqr', 'maape', 'd1_p',
           've', 'pearson_r', 'spearman_r', 'kge_2009', 'kge_2012']

#######################################################################################################################
#                                         General and Hydrological Error Metrics                                      #
#######################################################################################################################


def me(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
       remove_zero=False):
    """

    :param simulated_array: A 1D Numpy array of forecasted data from the time series.
    :param observed_array: A 1D Numpy array of observed data from the time series.
    :param replace_nan: Float input indicating what value to replace NaN values with.
    :param replace_inf: Float input indicating what value to replace Inf values with.
    :param remove_neg: Boolean input indicating whether user wants to remove negative numbers.
    :param remove_zero: Boolean input indicating whether user wants to remove zero values.
    :return: the mean error (bias) of the two arrays

    Range: -∞ < ME < ∞, data units, closer to 0 is  better, indicates bias.
    Notes: random errors can cancel with ME than indicating a better fit than actual

    - Fisher, Ronald Aylmer. “012: A Mathematical Examination of the Methods of Determining the Accuracy of an
    Observation by the Mean Error, and by the Mean Square Error.,” 1920.
    """

    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    return np.mean(simulated_array - observed_array)


def mae(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
        remove_zero=False):
    """

    :param simulated_array: A 1D Numpy array of forecasted data from the time series.
    :param observed_array: A 1D Numpy array of observed data from the time series.
    :param replace_nan: Float input indicating what value to replace NaN values with.
    :param replace_inf: Float input indicating what value to replace Inf values with.
    :param remove_neg: Boolean input indicating whether user wants to remove negative numbers.
    :param remove_zero: Boolean input indicating whether user wants to remove zero values.
    :return: Mean Absolute Error metric of the two arrays

    Range: 0 ≤ MAE < ∞, data units, smaller is better, does not indicate bias.
    Notes: random errors do not cancel, an L1-norm.

    - Willmott, Cort J., and Kenji Matsuura. “Advantages of the Mean Absolute Error (MAE) over the Root Mean Square
    Error (RMSE) in Assessing Average Model Performance.” Climate Research 30, no. 1 (2005): 79–82.
    - Willmott, Cort J., and Kenji Matsuura. “On the Use of Dimensioned Measures of Error to Evaluate the Performance
    of Spatial Interpolators.” International Journal of Geographical Information Science 20, no. 1 (2006): 89–102.
    """

    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    return np.mean(np.absolute(simulated_array - observed_array))


def mse(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
        remove_zero=False):
    """

    :param simulated_array: A 1D Numpy array of forecasted data from the time series.
    :param observed_array: A 1D Numpy array of observed data from the time series.
    :param replace_nan: Float input indicating what value to replace NaN values with.
    :param replace_inf: Float input indicating what value to replace Inf values with.
    :param remove_neg: Boolean input indicating whether user wants to remove negative numbers.
    :param remove_zero: Boolean input indicating whether user wants to remove zero values.
    :return: Mean Squared Error of the two arrays

    Range: 0 ≤ MSE < ∞, data units squared, smaller is better, does not indicate bias.
    Notes: random errors do not cancel, highlights larger errors, a squared L2-norm.

    - Wang, Zhou, and Alan C. Bovik. “Mean Squared Error: Love It or Leave It? A New Look at Signal Fidelity Measures.”
    IEEE Signal Processing Magazine 26, no. 1 (2009): 98–117.
    """

    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    return np.mean((simulated_array - observed_array) ** 2)


def mle(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
        remove_zero=False):
    """

    :param simulated_array: A 1D Numpy array of forecasted data from the time series.
    :param observed_array: A 1D Numpy array of observed data from the time series.
    :param replace_nan: Float input indicating what value to replace NaN values with.
    :param replace_inf: Float input indicating what value to replace Inf values with.
    :param remove_neg: Boolean input indicating whether user wants to remove negative numbers.
    :param remove_zero: Boolean input indicating whether user wants to remove zero values.
    :return: The mean log error of the two arrays.

    Same as ME only use log ratios as the error term.
    Notes: Limits the impact of outliers, more evenly weights high and low flows.

    - Törnqvist, Leo, Pentti Vartia, and Yrjö O. Vartia. “How Should Relative Changes Be Measured?”
    The American Statistician 39, no. 1 (1985): 43–46.
    """

    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    sim_log = np.log(simulated_array)
    obs_log = np.log(observed_array)
    return np.mean(sim_log - obs_log)


def male(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
         remove_zero=False):
    """

    :param simulated_array: A 1D Numpy array of forecasted data from the time series.
    :param observed_array: A 1D Numpy array of observed data from the time series.
    :param replace_nan: Float input indicating what value to replace NaN values with.
    :param replace_inf: Float input indicating what value to replace Inf values with.
    :param remove_neg: Boolean input indicating whether user wants to remove negative numbers.
    :param remove_zero: Boolean input indicating whether user wants to remove zero values.
    :return: The mean absolute log error of the two arrays.

    Same as MAE only use log ratios as the error term.
    Notes: Limits the impact of outliers, more evenly weights high and low flows.

    - Törnqvist, Leo, Pentti Vartia, and Yrjö O. Vartia. “How Should Relative Changes Be Measured?”
    The American Statistician 39, no. 1 (1985): 43–46.
    """

    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    sim_log = np.log(simulated_array)
    obs_log = np.log(observed_array)
    return np.mean(np.abs(sim_log - obs_log))


def msle(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
         remove_zero=False):
    """Returns the Mean Squared Log Error."""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    sim_log = np.log(simulated_array)
    obs_log = np.log(observed_array)
    return np.mean((sim_log - obs_log) ** 2)


def ed(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
       remove_zero=False):
    """Returns the Euclidean Distance"""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    return np.linalg.norm(observed_array - simulated_array)


def ned(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
        remove_zero=False):
    """Returns the Normalized Euclidean Distance"""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    a = observed_array / np.mean(observed_array)
    b = simulated_array / np.mean(simulated_array)
    return np.linalg.norm(a - b)


def rmse(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
         remove_zero=False):
    """Returns the Root mean squared error
    arguments: simulated array, observed array"""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    return np.sqrt(np.mean((simulated_array - observed_array) ** 2))


def rmsle(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
          remove_zero=False):
    """"Return the Root Mean Square Log Error. Note that to calculate the log values, each value in the observed and
    simulated array is increased by one unit in order to avoid run-time errors and nan values.
    arguments: simulated array, observed array"""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    return np.sqrt(np.mean(np.power(np.log1p(simulated_array) - np.log1p(observed_array), 2)))


def nrmse_range(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
                remove_zero=False):
    """"Return the Normalized Root Mean Square Error. RMSE normalized by the range of the observed time series (x).
    This allows comparison between data sets with different scales. The NRMSErange and NRMSEquartile  are the most and
    least sensitive to outliers, respectively. (Pontius et al., 2008)"""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)

    rmse_value = rmse(simulated_array=simulated_array, observed_array=observed_array)
    obs_max = np.max(observed_array)
    obs_min = np.min(observed_array)
    return rmse_value / (obs_max - obs_min)


def nrmse_mean(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
               remove_zero=False):
    """"Return the mean Normalized Root Mean Square Error. RMSE normalized by the mean of the observed time series (x).
    This allows comparison between data sets with different scales. The NRMSErange and NRMSEquartile  are the most and
    least sensitive to outliers, respectively.
    (Pontius et al., 2008)"""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)

    rmse_value = rmse(simulated_array=simulated_array, observed_array=observed_array)
    obs_mean = np.mean(observed_array)
    return rmse_value / obs_mean


def nrmse_iqr(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False, remove_zero=False):
    """"Return the interquartile range Normalized Root Mean Square Error. RMSE normalized by the interquartile
    range of the observed time series (x). The nRMSE allows comparison between data sets with different scales.
    The NRMSErange and NRMSEquartile  are the most and least sensitive to outliers, respectively.
    (Pontius et al., 2008)"""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)

    rmse_value = rmse(simulated_array=simulated_array, observed_array=observed_array)
    q1 = np.percentile(observed_array, 25)
    q3 = np.percentile(observed_array, 75)
    iqr = q3 - q1
    return rmse_value / iqr


def mase(simulated_array, observed_array, m=1, replace_nan=None, replace_inf=None, remove_neg=False,
         remove_zero=False):
    """Returns the Mean Absolute Scaled Error, the default period for m (seasonal period) is 1.
    Using the default assumes that the data is non-seasonal
    arguments: simulated array, observed array, m where m is the seasonal period"""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    start = m
    end = simulated_array.size - m
    a = np.mean(np.abs(simulated_array - observed_array))
    b = np.abs(observed_array[start:observed_array.size] - observed_array[:end])
    return a / (np.sum(b) / end)


def pearson_r(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False, remove_zero=False):
    simulated_array, observed_array = remove_values(simulated_array, observed_array,
                                                    replace_nan=replace_nan, replace_inf=replace_inf,
                                                    remove_neg=remove_neg, remove_zero=remove_zero)
    sim_mean = np.mean(simulated_array)
    obs_mean = np.mean(observed_array)

    top = np.sum((observed_array - obs_mean)*(simulated_array - sim_mean))
    bot1 = np.sqrt(np.sum((observed_array - obs_mean)**2))
    bot2 = np.sqrt(np.sum((simulated_array - sim_mean)**2))

    return top / (bot1 * bot2)


def spearman_r(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
               remove_zero=False):
    """Returns the Spearman rank-order correlation coefficient. The rank method is the scipy default, or 'average'.

    Spearman C (1904). "The proof and measurement of association between two things". American Journal of Psychology.
    15: 72–101. doi:10.2307/1412159"""
    simulated_array, observed_array = remove_values(simulated_array, observed_array,
                                                    replace_nan=replace_nan, replace_inf=replace_inf,
                                                    remove_neg=remove_neg, remove_zero=remove_zero)
    rank_sim = rankdata(simulated_array)
    rank_obs = rankdata(observed_array)

    mean_rank_sim = np.mean(rank_sim)
    mean_rank_obs = np.mean(rank_obs)

    top = np.mean((rank_obs - mean_rank_obs) * (rank_sim - mean_rank_sim))
    bot = np.sqrt(np.mean((rank_obs - mean_rank_obs) ** 2) * np.mean((rank_sim - mean_rank_sim) ** 2))

    return top / bot


def r_squared(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
              remove_zero=False):
    """Returns the Coefficient of Determination
    arguments: simulated array, observed array"""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    a = observed_array - np.mean(observed_array)
    b = simulated_array - np.mean(simulated_array)
    return (np.sum(a * b)) ** 2 / (np.sum(a ** 2) * np.sum(b ** 2))


def acc(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
        remove_zero=False):
    """
    :param simulated_array: A 1D Numpy array of forecasted data from the time series.
    :param observed_array: A 1D Numpy array of observed data from the time series.
    :param replace_nan: Float input indicating what value to replace NaN values with.
    :param replace_inf: Float input indicating what value to replace Inf values with.
    :param remove_neg: Boolean input indicating whether user wants to remove negative numbers.
    :param remove_zero: Boolean input indicating whether user wants to remove zero values.
    :return: Anomaly Correlation Coefficient.

    - Langland, Rolf H., and Ryan N. Maue. “Recent Northern Hemisphere Mid-Latitude Medium-Range Deterministic Forecast
    Skill.” Tellus A: Dynamic Meteorology and Oceanography 64, no. 1 (2012): 17531.
    - Miyakoda, K., G. D. Hembree, R. F. Strickler, and I. Shulman. “Cumulative Results of Extended Forecast Experiments
     I. Model Performance for Winter Cases.” Monthly Weather Review 100, no. 12 (1972): 836–55.
    - Murphy, Allan H., and Edward S. Epstein. “Skill Scores and Correlation Coefficients in Model Verification.”
    Monthly Weather Review 117, no. 3 (1989): 572–82.

    """
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    a = simulated_array - np.mean(simulated_array)
    b = observed_array - np.mean(observed_array)
    c = np.std(observed_array) * np.std(simulated_array) * simulated_array.size
    return np.dot(a, b / c)


def mape(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
         remove_zero=False):
    """Returns the Mean Absolute Percentage Error. The answer is a percentage
    arguments: simulated array, observed array"""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    a = simulated_array - observed_array
    b = np.abs(a / observed_array)
    c = 100 / simulated_array.size
    return c * np.sum(b)


def mapd(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
         remove_zero=False):
    """Returns the Mean Absolute Percentage Deviation.
    arguments: simulated array, observed array"""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    a = np.sum(np.abs(simulated_array - observed_array))
    b = np.sum(np.abs(observed_array))
    return a / b


def maape(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
          remove_zero=False):
    """Returns the Mean Arctangent Absolute Percentage Error. Range: 0 ≤ MAAPE < π/2, unit less, does not indicate bias,
    smaller is better. Represents the mean absolute error as a percentage of the observed values. Handles 0s in the
    observed data. not as bias as MAPE by under-over predictions
    (Kim and Kim, 2016)
    """
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    a = simulated_array - observed_array
    b = np.abs(a / observed_array)
    return np.mean(np.arctan(b))


def smape1(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
           remove_zero=False):
    """Returns the Symmetric Mean Absolute Percentage Error (1).
    arguments: simulated array, observed array"""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    a = 100 / simulated_array.size
    b = np.abs(simulated_array - observed_array)
    c = np.abs(simulated_array) + np.abs(observed_array)
    return a * np.sum(b / c)


def smape2(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
           remove_zero=False):
    """Returns the Symmetric Mean Absolute Percentage Error (2).
    arguments: simulated array, observed array"""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    a = simulated_array - observed_array
    b = (simulated_array + observed_array) / 2
    c = 100 / simulated_array.size
    return c * np.sum(np.abs(a / b))


def d(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
      remove_zero=False):
    """Returns the Index of Agreement (d).
    arguments: simulated array, observed array"""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    a = (observed_array - simulated_array) ** 2
    b = np.abs(simulated_array - np.mean(observed_array))
    c = np.abs(observed_array - np.mean(observed_array))
    return 1 - (np.sum(a) / np.sum((b + c) ** 2))


def d1(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
       remove_zero=False):
    """Returns the Index of Agreement (d1).
    arguments: simulated array, observed array"""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)

    obs_mean = np.mean(observed_array)

    a = np.sum(np.abs(simulated_array - observed_array))
    b = np.abs(simulated_array - obs_mean)
    c = np.abs(observed_array - obs_mean)
    return 1 - np.sum(a) / np.sum(b + c)


def dr(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
       remove_zero=False):
    """Returns the Refined Index of Agreement.
    arguments: simulated array, observed array"""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    a = np.sum(np.abs(simulated_array - observed_array))
    b = 2 * np.sum(np.abs(observed_array - observed_array.mean()))
    if a <= b:
        return 1 - (a / b)
    else:
        return (b / a) - 1


def drel(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
         remove_zero=False):
    """Returns the Relative Index of Agreement.
    arguments: simulated array, observed array"""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    a = ((simulated_array - observed_array) / observed_array) ** 2
    b = np.abs(simulated_array - np.mean(observed_array))
    c = np.abs(observed_array - np.mean(observed_array))
    e = ((b + c) / np.mean(observed_array)) ** 2
    return 1 - (np.sum(a) / np.sum(e))


def dmod(simulated_array, observed_array, j=1, replace_nan=None, replace_inf=None, remove_neg=False,
         remove_zero=False):
    """Returns the modified index of agreement, with j=1 as the default.
    arguments: simulated array, observed array, j"""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    a = (np.abs(simulated_array - observed_array)) ** j
    b = np.abs(simulated_array - np.mean(observed_array))
    c = np.abs(observed_array - np.mean(observed_array))
    e = (b + c) ** j
    return 1 - (np.sum(a) / np.sum(e))


def watt_m(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
           remove_zero=False):
    """Returns Watterson's M value. Watterson IG. 1996. Non-dimensional measures of climate model performance.
    International Journal of Climatology 16: 379–391.
    arguments: simulated array, observed array"""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    a = 2 / np.pi
    b = mse(simulated_array, observed_array)
    c = np.std(observed_array) ** 2 + np.std(simulated_array) ** 2
    e = (np.mean(simulated_array) - np.mean(observed_array)) ** 2
    f = c + e
    return a * np.arcsin(1 - (b / f))


def mb_r(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
         remove_zero=False):
    """Returns the Mielke-Berry R value. Mielke PW Jr, Berry KJ. 2001. Permutation Methods: A Distance Function Approach.
    Springer-Verlag: New York; 352.
    arguments: simulated array, observed array"""
    # Removing Nan Values
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)

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
    return 1 - (mae(simulated_array, observed_array) * size ** 2 / total)


def nse(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
        remove_zero=False):
    """Returns the Nash-Sutcliffe Efficiency value (Nash JE, Sutcliffe JV. 1970. River flow forecasting through
    conceptual models part I—A discussion of principles. Journal of Hydrology 10(3): 282–290.)
    arguments: simulated array, observed array"""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    a = (np.abs(simulated_array - observed_array)) ** 2
    b = (np.abs(observed_array - np.mean(observed_array))) ** 2
    return 1 - (np.sum(a) / np.sum(b))


def nse_mod(simulated_array, observed_array, j=1, replace_nan=None, replace_inf=None, remove_neg=False,
            remove_zero=False):
    """Returns the modified Nash-Sutcliffe Efficiency value
    (Krause, P., Boyle, D. P., and Base, F.: Comparison of different efficiency criteria for hydrological model
    assessment, Adv. Geosci., 5, 89-97, 2005
    Legates, D. R., and G. J. McCabe Jr. (1999), Evaluating the Use of "Goodness-of-Fit"
    Measures in Hydrologic and Hydroclimatic Model Validation, Water Resour. Res., 35(1), 233-241)
    arguments: simulated array, observed array, j"""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)

    a = (np.abs(simulated_array - observed_array)) ** j
    b = (np.abs(observed_array - np.mean(observed_array))) ** j
    return 1 - (np.sum(a) / np.sum(b))


def nse_rel(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
            remove_zero=False):
    """Returns the relative Nash-Sutcliffe Efficiency value
    (Krause, P., Boyle, D. P., and Base, F.: Comparison of different efficiency criteria for hydrological model
    assessment, Adv. Geosci., 5, 89-97, 2005
    Legates, D. R., and G. J. McCabe Jr. (1999), Evaluating the Use of "Goodness-of-Fit"
    Measures in Hydrologic and Hydroclimatic Model Validation, Water Resour. Res., 35(1), 233-241)
    arguments: simulated array, observed array"""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)

    a = (np.abs((simulated_array - observed_array) / observed_array)) ** 2
    b = (np.abs((observed_array - np.mean(observed_array)) / np.mean(observed_array))) ** 2
    return 1 - (np.sum(a) / np.sum(b))


def kge_2009(simulated_array, observed_array, s=(1, 1, 1), replace_nan=None, replace_inf=None, remove_neg=False,
             remove_zero=False):
    simulated_array, observed_array = remove_values(simulated_array, observed_array,
                                                    replace_nan=replace_nan, replace_inf=replace_inf,
                                                    remove_neg=remove_neg, remove_zero=remove_zero)
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

    # Relative variability between simulated and observed values
    alpha = sim_sigma / obs_sigma

    if obs_mean != 0 and obs_sigma != 0:
        kge = 1 - np.sqrt((s[0] * (pr - 1))**2 + (s[1] * (alpha - 1))**2 + (s[2] * (beta - 1))**2)
    else:
        if obs_mean == 0:
            warnings.warn('Warning: The observed data mean is 0. Therefore, Beta is infinite and the KGE value cannot '
                          'be computed.')
        if obs_sigma == 0:
            warnings.warn('Warning: The observed data standard deviation is 0. Therefore, Alpha is infinite and the KGE'
                          ' value cannot be computed.')
        kge = np.nan

    return kge


def kge_2012(simulated_array, observed_array, s=(1, 1, 1), replace_nan=None, replace_inf=None, remove_neg=False,
             remove_zero=False):
    simulated_array, observed_array = remove_values(simulated_array, observed_array,
                                                    replace_nan=replace_nan, replace_inf=replace_inf,
                                                    remove_neg=remove_neg, remove_zero=remove_zero)
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
        kge = 1 - np.sqrt((s[0] * (pr - 1))**2 + (s[1] * (gam - 1))**2 + (s[2] * (beta-1))**2)
    else:
        if obs_mean == 0:
            warnings.warn('Warning: The observed data mean is 0. Therefore, Beta is infinite and the KGE value cannot '
                          'be computed.')
        if obs_sigma == 0:
            warnings.warn('Warning: The observed data standard deviation is 0. Therefore, Gamma is infinite and the KGE'
                          ' value cannot be computed.')
        kge = np.nan

    return kge


def lm_index(simulated_array, observed_array, obs_bar_p=None, replace_nan=None, replace_inf=None, remove_neg=False,
             remove_zero=False):
    """Returns the Legate-McCabe Efficiency Index. Range: 0 ≤ lm_index < 1, unit less, larger is better,
    does not indicate bias, less weight to outliers. The term obs_bar_p is a seasonal or other selected average. If no
    obs_bar_p is given, the function will use the average of the observed data instead.
    (Legates and McCabe Jr, 1999)"""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    mean_obs = np.mean(observed_array)

    if obs_bar_p is not None:
        a = np.abs(simulated_array - observed_array)
        b = np.abs(observed_array - obs_bar_p)
        return 1 - (np.sum(a) / np.sum(b))
    else:
        a = np.abs(simulated_array - observed_array)
        b = np.abs(observed_array - mean_obs)
        return 1 - (np.sum(a) / np.sum(b))


def d1_p(simulated_array, observed_array, obs_bar_p=None, replace_nan=None, replace_inf=None, remove_neg=False,
         remove_zero=False):
    """
    :param simulated_array: array of simulated data.
    :param observed_array: array of observed data
    :param obs_bar_p: The term (obs_bar_p) is a seasonal or other selected average. If not obs_bar_p is given,
     the mean of the observed data will be used instead.
    :param replace_nan: Float input indicating what value to replace NaN values with.
    :param replace_inf: Float input indicating what value to replace Inf values with.
    :param remove_neg: Boolean input indicating whether user wants to remove negative numbers.
    :param remove_zero: Boolean input indicating whether user wants to remove zero values.
    :return: the Legate-McCabe Index of Agreement. Range: 0 ≤ d1_p < 1, unit less, larger is better, does not indicate
     bias, less weight to outliers.

    (Legates and McCabe Jr, 1999)
    """
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)

    if obs_bar_p is not None:
        a = np.abs(observed_array - simulated_array)
        b = np.abs(simulated_array - obs_bar_p) + np.abs(observed_array - obs_bar_p)
        return 1 - (np.sum(a) / np.sum(b))
    else:
        mean_obs = np.mean(observed_array)
        a = np.abs(observed_array - simulated_array)
        b = np.abs(simulated_array - mean_obs) + np.abs(observed_array - mean_obs)
        return 1 - (np.sum(a) / np.sum(b))


def ve(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
       remove_zero=False):
    """Range: 0≤VE<1 (Unitless) smaller is better, does not indicate bias, error as a percentage of flow.
    (Criss and Winston, 2008)"""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    a = np.sum(np.abs(simulated_array - observed_array))
    b = np.sum(observed_array)
    return 1 - (a / b)


def sa(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
       remove_zero=False):
    """Returns the spectral angle. Robila, S.A.; Gershman, A. In Spectral matching accuracy in processing hyperspectral
    data, Signals, Circuits and Systems, 2005. ISSCS 2005. International Symposium on, 2005; IEEE: pp 163-166.
    arguments: simulated array, observed array"""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    a = np.dot(simulated_array, observed_array)
    b = np.linalg.norm(simulated_array) * np.linalg.norm(observed_array)
    return np.arccos(a / b)


def sc(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
       remove_zero=False):
    """Returns the spectral Correlation. Robila, S.A.; Gershman, A. In Spectral matching accuracy in processing
    hyperspectral data, Signals, Circuits and Systems, 2005. ISSCS 2005. International Symposium on,
    2005; IEEE: pp 163-166.
    arguments: simulated array, observed array"""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    a = np.dot(observed_array - np.mean(observed_array), simulated_array - np.mean(simulated_array))
    b = np.linalg.norm(observed_array - np.mean(observed_array))
    c = np.linalg.norm(simulated_array - np.mean(simulated_array))
    e = b * c
    return np.arccos(a / e)


def sid(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
        remove_zero=False):
    """Returns the ___
    arguments: simulated array, observed array"""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    first = (observed_array / np.mean(observed_array)) - (simulated_array / np.mean(simulated_array))
    second1 = np.log10(observed_array) - np.log10(np.mean(observed_array))
    second2 = np.log10(simulated_array) - np.log10(np.mean(simulated_array))
    return np.dot(first, second1 - second2)


def sga(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
        remove_zero=False):
    """Returns the spectral gradient angle
    arguments: simulated array, observed array"""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    sgx = observed_array[1:] - observed_array[:observed_array.size - 1]
    sgy = simulated_array[1:] - simulated_array[:simulated_array.size - 1]
    a = np.dot(sgx, sgy)
    b = np.linalg.norm(sgx) * np.linalg.norm(sgy)
    return np.arccos(a / b)


""" ###################################################################################################################
                        H Metrics: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985)                                       
    ################################################################################################################"""


def h1_mhe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False, remove_zero=False):
    """H1 Metric: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985)."""

    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    h = (simulated_array - observed_array) / observed_array
    return np.mean(h)


def h1_ahe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
           remove_zero=False):
    """H1 Metric: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985)."""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    h = (simulated_array - observed_array) / observed_array
    return np.mean(np.abs(h))


def h1_rmshe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False, remove_zero=False):
    """H1 Metric: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985)."""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    h = (simulated_array - observed_array) / observed_array
    return np.mean(np.sqrt((h ** 2)))


def h2_mhe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False, remove_zero=False):
    """H2 Metric: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985)."""

    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    h = (simulated_array - observed_array) / simulated_array
    return np.mean(h)


def h2_ahe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False, remove_zero=False):
    """H2 Metric: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985)."""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    h = (simulated_array - observed_array) / simulated_array
    return np.mean(np.abs(h))


def h2_rmshe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False, remove_zero=False):
    """H2 Metric: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985)."""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    h = (simulated_array - observed_array) / simulated_array
    return np.mean(np.sqrt((h ** 2)))


def h3_mhe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False, remove_zero=False):
    """H3 Metric: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985)."""

    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    h = (simulated_array - observed_array) / (0.5 * (simulated_array + observed_array))
    return np.mean(h)


def h3_ahe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False, remove_zero=False):
    """H3 Metric: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985)."""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    h = (simulated_array - observed_array) / (0.5 * (simulated_array + observed_array))
    return np.mean(np.abs(h))


def h3_rmshe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False, remove_zero=False):
    """H3 Metric: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985)."""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    h = (simulated_array - observed_array) / (0.5 * (simulated_array + observed_array))
    return np.mean(np.sqrt((h ** 2)))


def h4_mhe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False, remove_zero=False):
    """H4 Metric: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985)."""

    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    h = (simulated_array - observed_array) / np.sqrt(simulated_array * observed_array)
    return np.mean(h)


def h4_ahe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False, remove_zero=False):
    """H4 Metric: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985)."""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    h = (simulated_array - observed_array) / np.sqrt(simulated_array * observed_array)
    return np.mean(np.abs(h))


def h4_rmshe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False, remove_zero=False):
    """H4 Metric: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985)."""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    h = (simulated_array - observed_array) / np.sqrt(simulated_array * observed_array)
    return np.mean(np.sqrt((h ** 2)))


def h5_mhe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False, remove_zero=False):
    """H5 Metric: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985)."""

    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    top = (simulated_array - observed_array)
    bot = np.reciprocal(0.5 * (np.reciprocal(observed_array) + np.reciprocal(simulated_array)))
    h = top / bot
    return np.mean(h)


def h5_ahe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False, remove_zero=False):
    """H5 Metric: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985)."""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    top = (simulated_array - observed_array)
    bot = np.reciprocal(0.5 * (np.reciprocal(observed_array) + np.reciprocal(simulated_array)))
    h = top / bot
    return np.mean(np.abs(h))


def h5_rmshe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False, remove_zero=False):
    """H5 Metric: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985)."""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    top = (simulated_array - observed_array)
    bot = np.reciprocal(0.5 * (np.reciprocal(observed_array) + np.reciprocal(simulated_array)))
    h = top / bot
    return np.mean(np.sqrt((h ** 2)))


def h6_mhe(simulated_array, observed_array, k=1, replace_nan=None, replace_inf=None, remove_neg=False,
           remove_zero=False):
    """H6 Metric: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985)."""

    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    top = (simulated_array / observed_array - 1)
    bot = np.power(0.5 * (1 + np.power(simulated_array / observed_array, k)), 1 / k)
    h = top / bot
    return np.mean(h)


def h6_ahe(simulated_array, observed_array, k=1, replace_nan=None, replace_inf=None, remove_neg=False,
           remove_zero=False):
    """H6 Metric: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985)."""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    top = (simulated_array / observed_array - 1)
    bot = np.power(0.5 * (1 + np.power(simulated_array / observed_array, k)), 1 / k)
    h = top / bot
    return np.mean(np.abs(h))


def h6_rmshe(simulated_array, observed_array, k=1, replace_nan=None, replace_inf=None, remove_neg=False,
             remove_zero=False):
    """H6 Metric: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985)."""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    top = (simulated_array / observed_array - 1)
    bot = np.power(0.5 * (1 + np.power(simulated_array / observed_array, k)), 1 / k)
    h = top / bot
    return np.mean(np.sqrt((h ** 2)))


def h7_mhe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False, remove_zero=False):
    """H7 Metric: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985)."""

    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    h = (simulated_array / observed_array - 1) / np.min(simulated_array / observed_array)
    return np.mean(h)


def h7_ahe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False, remove_zero=False):
    """H7 Metric: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985)."""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    h = (simulated_array / observed_array - 1) / np.min(simulated_array / observed_array)
    return np.mean(np.abs(h))


def h7_rmshe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False, remove_zero=False):
    """H7 Metric: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985)."""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    h = (simulated_array / observed_array - 1) / np.min(simulated_array / observed_array)
    return np.mean(np.sqrt((h ** 2)))


def h8_mhe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False, remove_zero=False):
    """H8 Metric: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985)."""

    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    h = (simulated_array / observed_array - 1) / np.max(simulated_array / observed_array)
    return np.mean(h)


def h8_ahe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False, remove_zero=False):
    """H8 Metric: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985)."""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    h = (simulated_array / observed_array - 1) / np.max(simulated_array / observed_array)
    return np.mean(np.abs(h))


def h8_rmshe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False, remove_zero=False):
    """H8 Metric: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985)."""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    h = (simulated_array / observed_array - 1) / np.max(simulated_array / observed_array)
    return np.mean(np.sqrt((h ** 2)))


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


def h10_mhe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False, remove_zero=False):
    """H10 Metric: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985)."""

    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    h = np.log1p(simulated_array) - np.log1p(observed_array)
    return np.mean(h)


def h10_ahe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False, remove_zero=False):
    """H10 Metric: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985)."""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    h = np.log1p(simulated_array) - np.log1p(observed_array)
    return np.mean(np.abs(h))


def h10_rmshe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False, remove_zero=False):
    """H10 Metric: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985)."""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    h = np.log1p(simulated_array) - np.log1p(observed_array)
    return np.mean(np.sqrt((h ** 2)))


""" ###################################################################################################################
                                Statistical Error Metrics for Distribution Testing                                       
    ################################################################################################################"""


def g_mean_diff(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
                remove_zero=False):
    """Returns the geometric mean difference."""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    sim_log = np.log1p(simulated_array)
    obs_log = np.log1p(observed_array)
    return np.exp(gmean(sim_log) - gmean(obs_log))


def mean_var(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
             remove_zero=False):
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    return np.var(np.log1p(observed_array) - np.log1p(simulated_array))


#######################################################################################################################
#                                             HELPER FUNCTIONS                                                        #
#######################################################################################################################


class HydrostatsError(Exception):
    pass


metric_names = ['Mean Error', 'Mean Absolute Error', 'Mean Squared Error', 'Eclidean Distance',
                'Normalized Eclidean Distance', 'Root Mean Square Error',
                'Root Mean Squared Log Error', 'Mean Absolute Scaled Error', 'Coefficient of Determination',
                'Anomaly Correlation Coefficient', 'Mean Absolute Percentage Error',
                'Mean Absolute Percentage Deviation', 'Symmetric Mean Absolute Percentage Error (1)',
                'Symmetric Mean Absolute Percentage Error (2)', 'Index of Agreement (d)',
                'Index of Agreement (d1)', 'Index of Agreement Refined (dr)',
                'Relative Index of Agreement', 'Modified Index of Agreement', "Watterson's M",
                'Mielke-Berry R', 'Nash-Sutcliffe Efficiency', 'Modified Nash-Sutcliffe Efficiency',
                'Relative Nash-Sutcliffe Efficiency', 'Legate-McCabe Efficiency Index',
                'Spectral Angle', 'Spectral Correlation', 'Spectral Information Divergence',
                'Spectral Gradient Angle', 'H1 - MHE', 'H1 - AHE', 'H1 - RMSHE', 'H2 - MHE',
                'H2 - AHE', 'H2 - RMSHE', 'H3 - MHE', 'H3 - AHE', 'H3 - RMSHE', 'H4 - MHE',
                'H4 - AHE', 'H4 - RMSHE', 'H5 - MHE', 'H5 - AHE', 'H5 - RMSHE', 'H6 - MHE',
                'H6 - AHE', 'H6 - RMSHE', 'H7 - MHE', 'H7 - AHE', 'H7 - RMSHE', 'H8 - MHE',
                'H8 - AHE', 'H8 - RMSHE', 'H10 - MHE', 'H10 - AHE', 'H10 - RMSHE',
                'Geometric Mean Difference', 'Mean Variance', 'Mean Log Error',
                'Mean Absolute Log Error', 'Mean Squared Log Error',
                'Normalized Root Mean Square Error - Range',
                'Normalized Root Mean Square Error - Mean',
                'Normalized Root Mean Square Error - IQR',
                'Mean Arctangent Absolute Percentage Error',
                'Legate-McCabe Index of Agreement', 'Volumetric Efficiency', 'Pearson R', 'Spearman R',
                'Kling-Gupta Efficiency (2009)', 'Kling-Gupta Efficiency (2012)']

metric_abbr = ['ME', 'MAE', 'MSE', 'ED', 'NED', 'RMSE', 'RMSLE', 'MASE', 'r2', 'ACC', 'MAPE', 'MAPD', 'SMAPE1',
               'SMAPE2', 'd', 'd1', 'dr', 'd (Rel.)', 'd (Mod.)', 'M', '(MB) R', 'NSE', 'NSE (Mod.)', 'NSE (Rel.)',
               "E1'", 'SA', 'SC', 'SID', 'SGA', 'H1 (MHE)', 'H1 (AHE)', 'H1 (RMSHE)', 'H2 (MHE)', 'H2 (AHE)',
               'H2 (RMSHE)', 'H3 (MHE)', 'H3 (AHE)', 'H3 (RMSHE)', 'H4 (MHE)', 'H4 (AHE)', 'H4 (RMSHE)',
               'H5 (MHE)', 'H5 (AHE)', 'H5 (RMSHE)', 'H6 (MHE)', 'H6 (AHE)', 'H6 (RMSHE)', 'H7 (MHE)',
               'H7 (AHE)', 'H7 (RMSHE)', 'H8 (MHE)', 'H8 (AHE)', 'H8 (RMSHE)', 'H10 (MHE)', 'H10 (AHE)',
               'H10 (RMSHE)', 'GMD', 'MV', 'MLE', 'MALE', 'MSLE', 'NRMSE (Range)', 'NRMSE (Mean)', 'NRMSE (IQR)',
               'MAAPE', "D1'", 'VE', 'R (Pearson)', 'R (Spearman)', 'KGE (2009)', 'KGE (2012)']


function_list = [me, mae, mse, ed, ned, rmse, rmsle, mase, r_squared, acc, mape, mapd, smape1, smape2, d, d1, dr, drel,
                 dmod, watt_m, mb_r, nse, nse_mod, nse_rel, lm_index, sa, sc, sid, sga, h1_mhe, h1_ahe, h1_rmshe,
                 h2_mhe, h2_ahe, h2_rmshe, h3_mhe, h3_ahe, h3_rmshe, h4_mhe, h4_ahe, h4_rmshe, h5_mhe, h5_ahe, h5_rmshe,
                 h6_mhe, h6_ahe, h6_rmshe, h7_mhe, h7_ahe, h7_rmshe, h8_mhe, h8_ahe, h8_rmshe, h10_mhe, h10_ahe,
                 h10_rmshe, g_mean_diff, mean_var, mle, male, msle, nrmse_range, nrmse_mean, nrmse_iqr, maape, d1_p, ve,
                 pearson_r, spearman_r, kge_2009, kge_2012]


def remove_values(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
                  remove_zero=False):
    """Removes the nan, negative, and inf values in two numpy arrays"""
    # Filtering warnings so that user doesn't see them while we remove the nans
    warnings.filterwarnings("ignore")
    # Checking to see if the vectors are the same length
    assert len(observed_array) == len(simulated_array)
    # Finding the original length of the two arrays
    original_length = simulated_array.size
    if replace_nan is not None:
        # Finding the NaNs
        sim_nan = np.isnan(simulated_array)
        obs_nan = np.isnan(observed_array)
        # Replacing the NaNs with the input
        simulated_array[sim_nan] = replace_nan
        observed_array[obs_nan] = replace_nan
    else:
        # Finding the nan values and combining them
        sim_nan = ~np.isnan(simulated_array)
        obs_nan = ~np.isnan(observed_array)
        nan_indices = np.logical_and(sim_nan, obs_nan)
        simulated_array = simulated_array[nan_indices]
        observed_array = observed_array[nan_indices]

    if replace_inf is not None:
        # Finding the NaNs
        sim_inf = np.isinf(simulated_array)
        obs_inf = np.isinf(observed_array)
        # Replacing the NaNs with the input
        simulated_array[sim_inf] = replace_inf
        observed_array[obs_inf] = replace_inf
    else:
        # Getting the indices of the nan values, combining them, and removing them from both arrays
        sim_inf = ~np.isinf(simulated_array)
        obs_inf = ~np.isinf(observed_array)
        inf_indices = np.logical_and(sim_inf, obs_inf)
        simulated_array = simulated_array[inf_indices]
        observed_array = observed_array[inf_indices]

    if remove_neg:
        # Finding the negative indices and combining them
        sim_neg = simulated_array >= 0
        obs_neg = observed_array >= 0
        neg_indices = np.logical_and(sim_neg, obs_neg)
        # Removing the negative indices
        simulated_array = simulated_array[neg_indices]
        observed_array = observed_array[neg_indices]

    if remove_zero:
        # Finding the zero indices and combining them
        sim_zero = simulated_array != 0
        obs_zero = observed_array != 0
        zero_indices = np.logical_and(sim_zero, obs_zero)
        # Removing the zero indices
        simulated_array = simulated_array[zero_indices]
        observed_array = observed_array[zero_indices]

    # Finding the final length of the arrays
    final_length = simulated_array.size

    warnings.filterwarnings("always")
    # Checking to see if any of the values were removed
    if final_length != original_length:
        warnings.warn("One of the arrays contained negative, nan, or inf values and they have been removed.",
                      Warning)
    return simulated_array, observed_array


def list_of_metrics(metrics, sim_array, obs_array, abbr=False, mase_m=1, dmod_j=1, nse_mod_j=1, h6_mhe_k=1, h6_ahe_k=1,
                    h6_rmshe_k=1, d1_p_obs_bar_p=None, lm_x_obs_bar_p=None, replace_nan=None, replace_inf=None,
                    remove_neg=False, remove_zero=False):
    # Removing values or replacing them based on user input
    sim_array, obs_array = remove_values(sim_array, obs_array, replace_nan=replace_nan, replace_inf=replace_inf,
                                         remove_neg=remove_neg, remove_zero=remove_zero)

    # Empty list for the metrics that are returned
    metrics_list = []

    # creating a list of indices for the selected metrics
    metrics_indices = []
    if not abbr:
        for i in metrics:
            metrics_indices.append(metric_names.index(i))
    else:
        for i in metrics:
            metrics_indices.append(metric_abbr.index(i))

    # Creating a list of selected metric functions
    selected_metrics = []
    for i in metrics_indices:
        selected_metrics.append(function_list[i])

    for index, func in zip(metrics_indices, selected_metrics):
        if index == 7:
            metrics_list.append(func(sim_array, obs_array, m=mase_m))

        elif index == 18:
            metrics_list.append(func(sim_array, obs_array, j=dmod_j))

        elif index == 22:
            metrics_list.append(func(sim_array, obs_array, j=nse_mod_j))

        elif index == 24:
            metrics_list.append(func(sim_array, obs_array, obs_bar_p=lm_x_obs_bar_p))

        elif index == 44:
            metrics_list.append(func(sim_array, obs_array, k=h6_mhe_k))

        elif index == 45:
            metrics_list.append(func(sim_array, obs_array, k=h6_ahe_k))

        elif index == 46:
            metrics_list.append(func(sim_array, obs_array, k=h6_rmshe_k))

        elif index == 65:
            metrics_list.append(func(sim_array, obs_array, obs_bar_p=d1_p_obs_bar_p))

        else:
            metrics_list.append(func(sim_array, obs_array))

    return metrics_list


if __name__ == "__main__":
    import pandas as pd
    import scipy.stats as stat

    # long_str = ''
    # for i in __all__:
    #     long_str += i + ', '
    # print(long_str)

    sim = np.random.rand(1000) * 20
    obs = np.random.rand(1000) * 20

    print(mase(sim, obs, m=3))
