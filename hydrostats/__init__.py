# python 3.6
# -*- coding: utf-8 -*-
"""
Created on Dec 28 12:16:32 2017
@author: Wade Roberts
"""
from __future__ import division
import numpy as np
import pandas as pd
from numba import njit, prange
import warnings
import calendar
import hydrostats.data as hd
import scipy.stats
import matplotlib.pyplot as plt
from hydrostats.data import HydrostatsError

""" ###################################################################################################################
                                         General and Hydrological Error Metrics                                       
    ################################################################################################################"""


def me(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
       remove_zero=False):
    """Returns the mean error of two 1 dimensional arrays
    arguments: simulated array, observed array"""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    return np.mean(simulated_array - observed_array)


def mae(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
        remove_zero=False):
    """Returns the Mean Absolute Error
    arguments: simulated array, observed array"""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    return np.mean(np.absolute(simulated_array - observed_array))


def mse(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
        remove_zero=False):
    """Returns the Mean Squared Error
    arguments: simulated array, observed array"""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    return np.mean((simulated_array - observed_array) ** 2)


def mle(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
        remove_zero=False):
    """Returns the Mean Log Error."""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    sim_log = np.log(simulated_array)
    obs_log = np.log(observed_array)
    return np.mean(sim_log - obs_log)


def male(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
         remove_zero=False):
    """Returns the Mean Absolute Log Error."""
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
    """Returns the Euclidean Distance
    arguments: simulated array, observed array"""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    return np.linalg.norm(observed_array - simulated_array)


def ned(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
        remove_zero=False):
    """Returns the Normalized Euclidean Distance
    arguments: simulated array, observed array"""
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


def nrmse(simulated_array, observed_array, nrmse_type, replace_nan=None, replace_inf=None, remove_neg=False,
          remove_zero=False):
    """"Return the Normalized Root Mean Square Error. Different types are 'range', 'mean', and 'iqr'.
    RMSE normalized by the range, the mean, or interquartile range of the observed time series (x), respectively.
    This allows comparison between data sets with different scales. The NRMSErange and NRMSEquartile  are the most and
    least sensitive to outliers, respectively. (Pontius et al., 2008)"""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    rmse_value = rmse(simulated_array=simulated_array, observed_array=observed_array)
    if nrmse_type == 'range':
        obs_max = np.max(observed_array)
        obs_min = np.min(observed_array)
        return rmse_value / (obs_max - obs_min)
    elif nrmse_type == 'mean':
        obs_mean = np.mean(observed_array)
        return rmse_value / obs_mean
    elif nrmse_type == 'iqr':
        q1 = np.percentile(observed_array, 25)
        q3 = np.percentile(observed_array, 75)
        iqr = q3 - q1
        return rmse_value / iqr
    else:
        raise HydrostatsError("Available types are 'range', 'mean', and 'iqr'. Please Specify one of these types.")


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
    """Returns the Anomaly Correlation Coefficient.
    arguments: simulated array, observed array"""
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
    a = np.sum((np.abs(simulated_array - observed_array)) ** 2)
    b = np.abs(simulated_array - np.mean(observed_array))
    c = np.abs(observed_array - np.mean(observed_array))
    return np.sum(a) / np.sum(b + c)


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
        for i in prange(size_numba):
            observed = observed_array_numba[i]
            for j in prange(size_numba):
                total_numba += abs(simulated_array_numba[j] - observed)
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


def lm_index(simulated_array, observed_array, x_bar_p=None, replace_nan=None, replace_inf=None, remove_neg=False,
             remove_zero=False):
    """Returns the Legate-McCabe Efficiency Index. Range: 0 ≤ lm_index < 1, unit less, larger is better,
    does not indicate bias, less weight to outliers. The term x_bar_p is a seasonal or other selected average. If no
    x_bar_p is given, the function will use the average of the observed data instead.
    (Legates and McCabe Jr, 1999)"""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    if x_bar_p is not None:
        a = np.abs(observed_array - simulated_array)
        b = np.abs(observed_array - np.mean(x_bar_p))
        return 1 - (np.sum(a) / np.sum(b))
    else:
        a = np.abs(observed_array - simulated_array)
        b = np.abs(observed_array - np.mean(observed_array))
        return 1 - (np.sum(a) / np.sum(b))


def d1_p(simulated_array, observed_array, x_bar_p=None, replace_nan=None, replace_inf=None, remove_neg=False,
         remove_zero=False):
    """Returns the Legate-McCabe Index of Agreement. Range: 0 ≤ d1_p < 1, unit less, larger is better, does not indicate
     bias, less weight to outliers. The term (x_bar_p) is a seasonal or other selected average. If not x_bar_p is given,
     the mean of the observed data will be used instead.
     (Legates and McCabe Jr, 1999)"""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    if x_bar_p is not None:
        a = (np.abs(simulated_array - observed_array)) ** 2
        b = np.abs(simulated_array - x_bar_p) + np.abs(observed_array - x_bar_p)
        return 1 - (np.sum(a) / np.sum(b))
    else:
        a = (np.abs(simulated_array - observed_array)) ** 2
        b = np.abs(simulated_array - np.mean(observed_array)) + np.abs(observed_array - np.mean(observed_array))
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


def h1(simulated_array, observed_array, h_type='mhe', replace_nan=None, replace_inf=None, remove_neg=False,
       remove_zero=False):
    """H1 Metric: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985).
    arguments: simulated array, observed array, h_type where the three h_types are mean, AHE, and rmhe."""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    h = (simulated_array - observed_array) / observed_array
    if h_type == 'mhe':
        return h.mean()
    elif h_type == 'ahe':
        return np.abs(h).mean()
    elif h_type == 'rmshe':
        return np.sqrt((h ** 2).mean())
    else:
        raise HydrostatsError("The three types available are 'mhe', 'ahe', and 'rmshe'.")


def h2(simulated_array, observed_array, h_type='mhe', replace_nan=None, replace_inf=None, remove_neg=False,
       remove_zero=False):
    """H2 Metric: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985).
    arguments: simulated array, observed array, h_type where the three h_types are mean, AHE, and rmhe."""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    h = (simulated_array - observed_array) / simulated_array
    if h_type == 'mhe':
        return h.mean()
    elif h_type == 'ahe':
        return np.abs(h).mean()
    elif h_type == 'rmshe':
        return np.sqrt((h ** 2).mean())
    else:
        raise HydrostatsError("The three types available are 'mhe', 'ahe', and 'rmshe'.")


def h3(simulated_array, observed_array, h_type='mhe', replace_nan=None, replace_inf=None, remove_neg=False,
       remove_zero=False):
    """H3 Metric: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985).
    arguments: simulated array, observed array, h_type where the three h_types are mean, AHE, and rmhe."""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    h = (simulated_array - observed_array) / (0.5 * (simulated_array + observed_array))
    if h_type == 'mhe':
        return h.mean()
    elif h_type == 'ahe':
        return np.abs(h).mean()
    elif h_type == 'rmshe':
        return np.sqrt((h ** 2).mean())
    else:
        raise HydrostatsError("The three types available are 'mhe', 'ahe', and 'rmshe'.")


def h4(simulated_array, observed_array, h_type='mhe', replace_nan=None, replace_inf=None, remove_neg=False,
       remove_zero=False):
    """H4 Metric: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985).
    arguments: simulated array, observed array, h_type where the three h_types are mean, AHE, and rmhe."""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    h = (simulated_array - observed_array) / np.sqrt(simulated_array * observed_array)
    if h_type == 'mhe':
        return h.mean()
    elif h_type == 'ahe':
        return np.abs(h).mean()
    elif h_type == 'rmshe':
        return np.sqrt((h ** 2).mean())
    else:
        raise HydrostatsError("The three types available are 'mhe', 'ahe', and 'rmshe'.")


def h5(simulated_array, observed_array, h_type='mhe', replace_nan=None, replace_inf=None, remove_neg=False,
       remove_zero=False):
    """H5 Metric: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985).
    arguments: simulated array, observed array, h_type where the three h_types are mean, AHE, and rmhe."""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    h = (simulated_array - observed_array) / \
        np.reciprocal(0.5 * (np.reciprocal(observed_array) + np.reciprocal(simulated_array)))
    if h_type == 'mhe':
        return h.mean()
    elif h_type == 'ahe':
        return np.abs(h).mean()
    elif h_type == 'rmshe':
        return np.sqrt((h ** 2).mean())
    else:
        raise HydrostatsError("The three types available are 'mhe', 'ahe', and 'rmshe'.")


def h6(simulated_array, observed_array, h_type='mhe', k=1, replace_nan=None, replace_inf=None, remove_neg=False,
       remove_zero=False):
    """H6 Metric: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985).
    arguments: simulated array, observed array, h_type where the three h_types are mean, AHE, and rmhe."""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    h = (simulated_array / observed_array - 1) / \
        np.power(0.5 * (1 + np.power(simulated_array / observed_array, k)), 1 / k)
    if h_type == 'mhe':
        return h.mean()
    elif h_type == 'ahe':
        return np.abs(h).mean()
    elif h_type == 'rmshe':
        return np.sqrt((h ** 2).mean())
    else:
        raise HydrostatsError("The three types available are 'mhe', 'ahe', and 'rmshe'.")


def h7(simulated_array, observed_array, h_type='mhe', replace_nan=None, replace_inf=None, remove_neg=False,
       remove_zero=False):
    """H7 Metric: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985).
    arguments: simulated array, observed array, h_type where the three h_types are mean, AHE, and rmhe."""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    h = (simulated_array / observed_array - 1) / np.min(simulated_array / observed_array)
    if h_type == 'mhe':
        return h.mean()
    elif h_type == 'ahe':
        return np.abs(h).mean()
    elif h_type == 'rmshe':
        return np.sqrt((h ** 2).mean())
    else:
        raise HydrostatsError("The three types available are 'mhe', 'ahe', and 'rmshe'.")


def h8(simulated_array, observed_array, h_type='mhe', replace_nan=None, replace_inf=None, remove_neg=False,
       remove_zero=False):
    """H8 Metric: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985).
    arguments: simulated array, observed array, h_type where the three h_types are mean, AHE, and rmhe."""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    h = (simulated_array / observed_array - 1) / (simulated_array / observed_array).max()
    if h_type == 'mhe':
        return h.mean()
    elif h_type == 'ahe':
        return np.abs(h).mean()
    elif h_type == 'rmshe':
        return np.sqrt((h ** 2).mean())
    else:
        raise HydrostatsError("The three types available are 'mhe', 'ahe', and 'rmshe'.")


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


def h10(simulated_array, observed_array, h_type='mhe', replace_nan=None, replace_inf=None, remove_neg=False,
        remove_zero=False):
    """H10 Metric: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985).
    arguments: simulated array, observed array, h_type where the three h_types are mean, AHE, and rmhe."""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    h = np.log1p(simulated_array) - np.log1p(observed_array)
    if h_type == 'mhe':
        return h.mean()
    elif h_type == 'ahe':
        return np.abs(h).mean()
    elif h_type == 'rmshe':
        return np.sqrt((h ** 2).mean())
    else:
        raise HydrostatsError("The three types available are 'mhe', 'ahe', and 'rmshe'.")


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
    return np.exp(scipy.stats.gmean(sim_log) - scipy.stats.gmean(obs_log))


def mean_var(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
             remove_zero=False):
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    return np.var(np.log1p(observed_array) - np.log1p(simulated_array))


""" ###################################################################################################################
                                                Helper Functions and Classes                                       
    ################################################################################################################"""


class HydrostatsVariables:
    metric_names = ['Mean Error', 'Mean Absolute Error', 'Mean Squared Error', 'Eclidean Distance',
                    'Normalized Eclidean Distance', 'Root Mean Square Error',
                    'Root Mean Squared Log Error', 'Mean Absolute Scaled Error', 'R^2',
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
                    'Legate-McCabe Index of Agreement', 'Volumetric Efficiency']

    metric_abbr = ['ME', 'MAE', 'MSE', 'ED', 'NED', 'RMSE', 'RMSLE', 'MASE', 'R^2', 'ACC', 'MAPE', 'MAPD', 'SMAPE1',
                   'SMAPE2', 'd', 'd1', 'dr', 'd (Rel.)', 'd (Mod.)', 'M', '(MB) R', 'NSE', 'NSE (Mod.)', 'NSE (Rel.)',
                   "E1'", 'SA', 'SC', 'SID', 'SGA', 'H1 (MHE)', 'H1 (AHE)', 'H1 (RMSHE)', 'H2 (MHE)', 'H2 (AHE)',
                   'H2 (RMSHE)', 'H3 (MHE)', 'H3 (AHE)', 'H3 (RMSHE)', 'H4 (MHE)', 'H4 (AHE)', 'H4 (RMSHE)',
                   'H5 (MHE)', 'H5 (AHE)', 'H5 (RMSHE)', 'H6 (MHE)', 'H6 (AHE)', 'H6 (RMSHE)', 'H7 (MHE)',
                   'H7 (AHE)', 'H7 (RMSHE)', 'H8 (MHE)', 'H8 (AHE)', 'H8 (RMSHE)', 'H10 (MHE)', 'H10 (AHE)',
                   'H10 (RMSHE)', 'GMD', 'MV', 'MLE', 'MALE', 'MSLE', 'NRMSE (Range)', 'NRMSE (Mean)', 'NRMSE (IQR)',
                   'MAAPE', "D1'", 'VE']

    function_list = [me, mae, mse, ed, ned, rmse, rmsle, mase, r_squared, acc, mape, mapd, smape1, smape2, d, d1, dr,
                     drel, dmod, watt_m, mb_r, nse, nse_mod, nse_rel, lm_index, sa, sc, sid, sga, h1, h1, h1, h2, h2,
                     h2, h3, h3, h3, h4, h4, h4, h5, h5, h5, h6, h6, h6, h7, h7, h7, h8, h8, h8, h10, h10, h10,
                     g_mean_diff, mean_var, mle, male, msle, nrmse, nrmse, nrmse, maape, d1_p, ve]


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
        sim_neg = simulated_array > 0
        obs_neg = observed_array > 0
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
        pass
        warnings.warn("One of the arrays contained negative, nan, or inf values and they have been removed.",
                      Warning)
    return simulated_array, observed_array


def list_of_metrics(metrics, sim_array, obs_array, mase_m=1, dmod_j=1, nse_mod_j=1, h6_mhe_k=1, h6_ahe_k=1,
                    h6_rmshe_k=1, d1_p_x_bar=None, lm_x_bar=None, replace_nan=None, replace_inf=None, remove_neg=False,
                    remove_zero=False):
    # Empty list for the metrics that are returned
    metrics_list = []

    metric_names = HydrostatsVariables.metric_names
    function_list = HydrostatsVariables.function_list

    # creating a list of indices for the selected metrics
    metrics_indices = []
    for i in metrics:
        metrics_indices.append(metric_names.index(i))

    # Creating a list of selected metric functions
    selected_metrics = []
    for i in metrics_indices:
        selected_metrics.append(function_list[i])

    for index, func in zip(metrics_indices, selected_metrics):
        if index == 7:
            metrics_list.append(func(sim_array, obs_array, m=mase_m, replace_nan=replace_nan,
                                     replace_inf=replace_inf, remove_neg=remove_neg, remove_zero=remove_zero))
        elif index == 18:
            metrics_list.append(func(sim_array, obs_array, j=dmod_j, replace_nan=replace_nan,
                                     replace_inf=replace_inf, remove_neg=remove_neg, remove_zero=remove_zero))
        elif index == 22:
            metrics_list.append(func(sim_array, obs_array, j=nse_mod_j, replace_nan=replace_nan,
                                     replace_inf=replace_inf, remove_neg=remove_neg, remove_zero=remove_zero))
        elif index == 24:
            metrics_list.append(func(sim_array, obs_array, x_bar_p=lm_x_bar, replace_nan=replace_nan,
                                     replace_inf=replace_inf, remove_neg=remove_neg, remove_zero=remove_zero))
        elif index == 29 or index == 32 or index == 35 or index == 38 or index == 41 or index == 47 or index == 50 \
                or index == 53:
            metrics_list.append(func(sim_array, obs_array, h_type='mhe', replace_nan=replace_nan,
                                     replace_inf=replace_inf, remove_neg=remove_neg, remove_zero=remove_zero))
        elif index == 30 or index == 33 or index == 36 or index == 39 or index == 42 or index == 48 or index == 51 \
                or index == 54:
            metrics_list.append(func(sim_array, obs_array, h_type='ahe', replace_nan=replace_nan,
                                     replace_inf=replace_inf, remove_neg=remove_neg, remove_zero=remove_zero))
        elif index == 31 or index == 34 or index == 37 or index == 40 or index == 43 or index == 49 or index == 52 \
                or index == 55:
            metrics_list.append(func(sim_array, obs_array, h_type='rmshe', replace_nan=replace_nan,
                                     replace_inf=replace_inf, remove_neg=remove_neg, remove_zero=remove_zero))
        elif index == 44:
            metrics_list.append(func(sim_array, obs_array, k=h6_mhe_k, h_type='mhe', replace_nan=replace_nan,
                                     replace_inf=replace_inf, remove_neg=remove_neg, remove_zero=remove_zero))
        elif index == 45:
            metrics_list.append(func(sim_array, obs_array, k=h6_ahe_k, h_type='ahe', replace_nan=replace_nan,
                                     replace_inf=replace_inf, remove_neg=remove_neg, remove_zero=remove_zero))
        elif index == 46:
            metrics_list.append(func(sim_array, obs_array, k=h6_rmshe_k, h_type='rmshe', replace_nan=replace_nan,
                                     replace_inf=replace_inf, remove_neg=remove_neg, remove_zero=remove_zero))
        elif index == 61:
            metrics_list.append(func(sim_array, obs_array, nrmse_type='range', replace_nan=replace_nan,
                                     replace_inf=replace_inf, remove_neg=remove_neg, remove_zero=remove_zero))
        elif index == 62:
            metrics_list.append(func(sim_array, obs_array, nrmse_type='mean', replace_nan=replace_nan,
                                     replace_inf=replace_inf, remove_neg=remove_neg, remove_zero=remove_zero))
        elif index == 63:
            metrics_list.append(func(sim_array, obs_array, nrmse_type='iqr', replace_nan=replace_nan,
                                     replace_inf=replace_inf, remove_neg=remove_neg, remove_zero=remove_zero))
        elif index == 65:
            metrics_list.append(func(sim_array, obs_array, x_bar_p=d1_p_x_bar, replace_nan=replace_nan,
                                     replace_inf=replace_inf, remove_neg=remove_neg, remove_zero=remove_zero))
        else:
            metrics_list.append(
                func(sim_array, obs_array, replace_nan=replace_nan, replace_inf=replace_inf,
                     remove_neg=remove_neg, remove_zero=remove_zero))

    return metrics_list


""" ###################################################################################################################
                                          Tools for Tables, Lag Analysis, Etc.                                       
    ################################################################################################################"""


def all_metrics(simulated_array, observed_array, mase_m=1, dmod_j=1, nse_mod_j=1, h6_k=1, d1_p_x_bar=None,
                lm_x_bar=None, replace_nan=None, replace_inf=None, remove_neg=False, remove_zero=False):
    """Takes two numpy arrays and returns a pandas dataframe with all of the metrics included."""
    metrics_list = HydrostatsVariables.metric_names

    # Creating the Metrics Matrix
    metrics_array = np.zeros(len(metrics_list), dtype=np.float64)

    # Removing Values based on User Input
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)

    if d1_p_x_bar is None:
        d1_p_x_bar = np.mean(observed_array)
    if lm_x_bar is None:
        lm_x_bar = np.mean(observed_array)

    metrics_array[0] = me(simulated_array, observed_array)
    warnings.filterwarnings("ignore")
    metrics_array[1] = mae(simulated_array, observed_array)
    metrics_array[2] = mse(simulated_array, observed_array)
    metrics_array[3] = ed(simulated_array, observed_array)
    metrics_array[4] = ned(simulated_array, observed_array)
    metrics_array[5] = rmse(simulated_array, observed_array)
    metrics_array[6] = rmsle(simulated_array, observed_array)
    metrics_array[7] = mase(simulated_array, observed_array, m=mase_m)
    metrics_array[8] = r_squared(simulated_array, observed_array)
    metrics_array[9] = acc(simulated_array, observed_array)
    metrics_array[10] = mape(simulated_array, observed_array)
    metrics_array[11] = mapd(simulated_array, observed_array)
    metrics_array[12] = smape1(simulated_array, observed_array)
    metrics_array[13] = smape2(simulated_array, observed_array)
    metrics_array[14] = d(simulated_array, observed_array)
    metrics_array[15] = d1(simulated_array, observed_array)
    metrics_array[16] = dr(simulated_array, observed_array)
    metrics_array[17] = drel(simulated_array, observed_array)
    metrics_array[18] = dmod(simulated_array, observed_array, j=dmod_j)
    metrics_array[19] = watt_m(simulated_array, observed_array)
    metrics_array[20] = mb_r(simulated_array, observed_array)
    metrics_array[21] = nse(simulated_array, observed_array)
    metrics_array[22] = nse_mod(simulated_array, observed_array, j=nse_mod_j)
    metrics_array[23] = nse_rel(simulated_array, observed_array)
    metrics_array[24] = lm_index(simulated_array, observed_array, lm_x_bar)
    metrics_array[25] = sa(simulated_array, observed_array)
    metrics_array[26] = sc(simulated_array, observed_array)
    metrics_array[27] = sid(simulated_array, observed_array)
    metrics_array[28] = sga(simulated_array, observed_array)
    metrics_array[29] = h1(simulated_array, observed_array, 'mhe')
    metrics_array[30] = h1(simulated_array, observed_array, 'ahe')
    metrics_array[31] = h1(simulated_array, observed_array, 'rmshe')
    metrics_array[32] = h2(simulated_array, observed_array, 'mhe')
    metrics_array[33] = h2(simulated_array, observed_array, 'ahe')
    metrics_array[34] = h2(simulated_array, observed_array, 'rmshe')
    metrics_array[35] = h3(simulated_array, observed_array, 'mhe')
    metrics_array[36] = h3(simulated_array, observed_array, 'ahe')
    metrics_array[37] = h3(simulated_array, observed_array, 'rmshe')
    metrics_array[38] = h4(simulated_array, observed_array, 'mhe')
    metrics_array[39] = h4(simulated_array, observed_array, 'ahe')
    metrics_array[40] = h4(simulated_array, observed_array, 'rmshe')
    metrics_array[41] = h5(simulated_array, observed_array, 'mhe')
    metrics_array[42] = h5(simulated_array, observed_array, 'ahe')
    metrics_array[43] = h5(simulated_array, observed_array, 'rmshe')
    metrics_array[44] = h6(simulated_array, observed_array, 'mhe', k=h6_k)
    metrics_array[45] = h6(simulated_array, observed_array, 'ahe', k=h6_k)
    metrics_array[46] = h6(simulated_array, observed_array, 'rmshe', k=h6_k)
    metrics_array[47] = h7(simulated_array, observed_array, 'mhe')
    metrics_array[48] = h7(simulated_array, observed_array, 'ahe')
    metrics_array[49] = h7(simulated_array, observed_array, 'rmshe')
    metrics_array[50] = h8(simulated_array, observed_array, 'mhe')
    metrics_array[51] = h8(simulated_array, observed_array, 'ahe')
    metrics_array[52] = h8(simulated_array, observed_array, 'rmshe')
    metrics_array[53] = h10(simulated_array, observed_array, 'mhe')
    metrics_array[54] = h10(simulated_array, observed_array, 'ahe')
    metrics_array[55] = h10(simulated_array, observed_array, 'rmshe')
    metrics_array[56] = g_mean_diff(simulated_array, observed_array)
    metrics_array[57] = mean_var(simulated_array, observed_array)
    metrics_array[58] = mle(simulated_array, observed_array)
    metrics_array[59] = male(simulated_array, observed_array)
    metrics_array[60] = msle(simulated_array, observed_array)
    metrics_array[61] = nrmse(simulated_array, observed_array, 'range')
    metrics_array[62] = nrmse(simulated_array, observed_array, 'mean')
    metrics_array[63] = nrmse(simulated_array, observed_array, 'iqr')
    metrics_array[64] = maape(simulated_array, observed_array)
    metrics_array[65] = d1_p(simulated_array, observed_array, d1_p_x_bar)
    metrics_array[66] = ve(simulated_array, observed_array)
    warnings.filterwarnings("always")

    return pd.DataFrame(data=metrics_array, columns=['Metric Values'], index=metrics_list)


def make_table(merged_dataframe, metrics, seasonal_periods=None, mase_m=1, dmod_j=1, nse_mod_j=1, h6_mhe_k=1,
               h6_ahe_k=1, h6_rmshe_k=1, d1_p_x_bar=None, lm_x_bar=None, replace_nan=None, replace_inf=None,
               remove_neg=False, remove_zero=False, to_csv=None, to_excel=None, location=None):
    """Creates a table with metrics as specified by the user. Seasonal periods can also be specified in order to compare
    different seasons and how well the simulated data matches the observed data. Has options to save the table to either
    a csv or an excel workbook. Also has an option to add a column for the location of the data. See the official
    documentation at https://waderoberts123.github.io/Hydrostats/ for a full explanation of all of the function
    arguments as well as examples."""

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
    full_time_series_list = list_of_metrics(metrics=metrics, sim_array=sim_array, obs_array=obs_array,
                                            mase_m=mase_m, dmod_j=dmod_j, nse_mod_j=nse_mod_j, h6_mhe_k=h6_mhe_k,
                                            h6_ahe_k=h6_ahe_k, h6_rmshe_k=h6_rmshe_k,
                                            d1_p_x_bar=d1_p_x_bar, lm_x_bar=lm_x_bar, replace_nan=replace_nan,
                                            replace_inf=replace_inf, remove_neg=remove_neg, remove_zero=remove_zero)

    # Appending the full time series list to the entire list:
    complete_metric_list.append(full_time_series_list)

    if seasonal_periods is not None:
        for time in seasonal_periods:
            temp_df = hd.seasonal_period(merged_dataframe, time)
            sim_array = temp_df.iloc[:, 0].values
            obs_array = temp_df.iloc[:, 1].values

            seasonal_metric_list = list_of_metrics(metrics=metrics, sim_array=sim_array, obs_array=obs_array,
                                                   mase_m=mase_m, dmod_j=dmod_j, nse_mod_j=nse_mod_j, h6_mhe_k=h6_mhe_k,
                                                   h6_ahe_k=h6_ahe_k, h6_rmshe_k=h6_rmshe_k, d1_p_x_bar=d1_p_x_bar,
                                                   lm_x_bar=lm_x_bar, replace_nan=replace_nan, replace_inf=replace_inf,
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


def time_lag(merged_dataframe, metrics, interp_freq='6H', interp_type='pchip', shift_range=[-30, 30], mase_m=1,
             dmod_j=1, nse_mod_j=1, h6_mhe_k=1, h6_ahe_k=1, h6_rmshe_k=1, d1_p_x_bar=None, lm_x_bar=None,
             replace_nan=None, replace_inf=None, remove_neg=False, remove_zero=False,
             plot_title='Metric Values as Different Lags', ylabel='Metric Value', xlabel='Number of Lags',
             save_fig=None, figsize=(10, 6), station=None, to_csv=None, to_excel=None):
    """Runs a time lag analysis to check for potential timing errors in datasets. Returns a dataframe with all of the
    metric values at different time lag, as well as the max and min metric value throughout the time lag as well as the
    position of the max and min time lag values. See the official documentation at
    https://waderoberts123.github.io/Hydrostats/ for a full explanation of all of the function arguments as well
    as examples."""
    metrics_list = HydrostatsVariables.metric_names
    abbreviations = HydrostatsVariables.metric_abbr

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
        raise HydrostatsError('ValueError Raised while interpolating, you may want to check for duplicates in your '
                              'dates.')

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
                                      h6_ahe_k=h6_ahe_k, h6_rmshe_k=h6_rmshe_k, d1_p_x_bar=d1_p_x_bar,
                                      lm_x_bar=lm_x_bar, replace_nan=replace_nan, replace_inf=replace_inf,
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
