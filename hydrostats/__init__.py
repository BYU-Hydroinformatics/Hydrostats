# python 3.6
# -*- coding: utf-8 -*-
"""
Created on Dec 28 12:16:32 2017
@author: Wade Roberts
"""
import numpy as np
import pandas as pd
from numba import njit, prange
import warnings


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
    return np.mean(np.abs(simulated_array - observed_array) / np.abs(observed_array)) * 100


def mapd(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
         remove_zero=False):
    """Returns the Mean Absolute Percentage Deviation.
    arguments: simulated array, observed array"""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan, 
                                                    replace_inf=replace_inf, remove_neg=remove_neg, 
                                                    remove_zero=remove_zero)
    return (np.sum(np.abs(simulated_array - observed_array))) / np.abs(observed_array.sum())


def smap1(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
          remove_zero=False):
    """Returns the Symmetric Mean Absolute Percentage Error (1).
    arguments: simulated array, observed array"""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan, 
                                                    replace_inf=replace_inf, remove_neg=remove_neg, 
                                                    remove_zero=remove_zero)
    a = 100 / simulated_array.size
    b = np.abs(simulated_array - observed_array)
    c = np.abs(simulated_array) - np.abs(observed_array)
    return a * np.sum(b / c)


def smap2(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
          remove_zero=False):
    """Returns the Symmetric Mean Absolute Percentage Error (2).
    arguments: simulated array, observed array"""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan, 
                                                    replace_inf=replace_inf, remove_neg=remove_neg, 
                                                    remove_zero=remove_zero)
    a = np.sum(np.abs(simulated_array - observed_array))
    b = np.sum(simulated_array + observed_array)
    return (100 / simulated_array.size) * (a / b)


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
    num = np.sum(np.abs(simulated_array - observed_array))
    den = np.sum((np.abs(simulated_array - observed_array.mean()) + np.abs(observed_array - observed_array.mean())))
    return 1 - (num / den)


def dr(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
       remove_zero=False):
    """Returns the Refined Index of Agreement.
    arguments: simulated array, observed array"""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan, 
                                                    replace_inf=replace_inf, remove_neg=remove_neg, 
                                                    remove_zero=remove_zero)
    if np.abs(simulated_array - observed_array).sum() <= 2 * np.abs(simulated_array - simulated_array.mean()).sum():
        return 1 - (np.abs(simulated_array - observed_array).sum() /
                    np.abs(simulated_array - simulated_array.mean()).sum())
    else:
        return (np.sum(np.abs(observed_array - observed_array.mean())) /
                np.sum(np.abs(simulated_array - observed_array))) - 1


def drel(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
         remove_zero=False):
    """Returns the Relative Index of Agreement.
    arguments: simulated array, observed array"""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan, 
                                                    replace_inf=replace_inf, remove_neg=remove_neg, 
                                                    remove_zero=remove_zero)
    return 1 - (np.sum(((observed_array - simulated_array) / observed_array) ** 2) /
                np.sum(((np.abs(simulated_array - np.mean(observed_array)) +
                         np.abs(observed_array - np.mean(observed_array))) / np.mean(observed_array)) ** 2))


def dmod(simulated_array, observed_array, j=1, replace_nan=None, replace_inf=None, remove_neg=False,
         remove_zero=False):
    """Returns the modified index of agreement, with j=1 as the default.
    arguments: simulated array, observed array, j"""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan, 
                                                    replace_inf=replace_inf, remove_neg=remove_neg, 
                                                    remove_zero=remove_zero)
    return (1 - (np.sum((np.abs(observed_array - simulated_array)) ** j)) /
            np.sum((np.abs(simulated_array - np.mean(observed_array)) +
                    np.abs(observed_array - np.mean(observed_array))) ** j))


def watt_m(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
           remove_zero=False):
    """Returns Watterson's M value. Watterson IG. 1996. Non-dimensional measures of climate model performance.
    International Journal of Climatology 16: 379–391.
    arguments: simulated array, observed array"""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan, 
                                                    replace_inf=replace_inf, remove_neg=remove_neg, 
                                                    remove_zero=remove_zero)
    return (2 / np.pi) * np.arcsin(1 - mse(simulated_array, observed_array) /
                                   (np.std(observed_array) ** 2 + np.std(simulated_array) ** 2 +
                                    (np.mean(simulated_array) - np.mean(observed_array)) ** 2))


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
    def numba_loop(simulated_array, observed_array):
        """Using numba for the double for loop"""
        assert len(observed_array) == len(simulated_array)
        size = len(simulated_array)
        total = 0.
        for i in prange(size):
            observed = observed_array[i]
            for j in prange(size):
                total += abs(simulated_array[j] - observed)
        return total, size

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
    return 1 - (
            np.sum((simulated_array - observed_array) ** 2) / np.sum((observed_array - observed_array.mean()) ** 2))


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
    return 1 - (np.sum(np.abs(observed_array - simulated_array) ** j) / np.sum(
        np.abs(observed_array - np.mean(observed_array)) ** j))


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
    return 1 - (np.sum(((observed_array - simulated_array) / observed_array) ** 2) /
                np.sum(np.abs((observed_array - np.mean(observed_array)) / np.mean(observed_array)) ** 2))


def lm_index(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
             remove_zero=False):
    """Returns the Legate-McCabe index. Legates DR, McCabe GJ Jr. 1999. Evaluating the use of “goodness-of-fit” measures
    in hydrologic and hydroclimatic model validation. Water Resources Research 35(1): 233–241.
    arguments: simulated array, observed array"""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan, 
                                                    replace_inf=replace_inf, remove_neg=remove_neg, 
                                                    remove_zero=remove_zero)
    return 1 - (np.sum(np.abs(observed_array - simulated_array)) / np.sum(
        np.abs(observed_array - np.mean(observed_array))))


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
    return np.arccos(np.dot((observed_array - observed_array.mean()), (simulated_array - simulated_array.mean())) /
                     (np.linalg.norm(observed_array - observed_array.mean()) *
                      np.linalg.norm(simulated_array - simulated_array.mean())))


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


# def sga(forecasted_array, observed_array):
#     """Returns the spectral gradient angle
#     arguments: forecasted array, observed array"""
#     assert len(observed_array) == len(forecasted_array)
#     forecasted_array, observed_array = remove_values(forecasted_array, observed_array)
#     SGx = observed_array[1:] - observed_array[:observed_array.size - 1]
#     SGy = forecasted_array[1:] - forecasted_array[:forecasted_array.size - 1]
#     return sa(SGx, SGy)


""" ###################################################################################################################
                        H Metrics: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985)                                       
    ################################################################################################################"""


def h1(simulated_array, observed_array, h_type='mean', replace_nan=None, replace_inf=None, remove_neg=False,
       remove_zero=False):
    """H1 Metric: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985).
    arguments: simulated array, observed array, h_type where the three h_types are mean, absolute, and rmhe."""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan, 
                                                    replace_inf=replace_inf, remove_neg=remove_neg, 
                                                    remove_zero=remove_zero)
    h = (simulated_array - observed_array) / observed_array
    if h_type == 'mean':
        return h.mean()
    elif h_type == 'absolute':
        return np.abs(h).mean()
    elif h_type == 'rmhe':
        return np.sqrt((h ** 2).mean())
    else:
        print('Please make a valid h_type selection')


def h2(simulated_array, observed_array, h_type='mean', replace_nan=None, replace_inf=None, remove_neg=False,
       remove_zero=False):
    """H2 Metric: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985).
    arguments: simulated array, observed array, h_type where the three h_types are mean, absolute, and rmhe."""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan, 
                                                    replace_inf=replace_inf, remove_neg=remove_neg, 
                                                    remove_zero=remove_zero)
    h = (simulated_array - observed_array) / simulated_array
    if h_type == 'mean':
        return h.mean()
    elif h_type == 'absolute':
        return np.abs(h).mean()
    elif h_type == 'rmhe':
        return np.sqrt((h ** 2).mean())
    else:
        return 'Please make a valid h_type selection'


def h3(simulated_array, observed_array, h_type='mean', replace_nan=None, replace_inf=None, remove_neg=False,
       remove_zero=False):
    """H3 Metric: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985).
    arguments: simulated array, observed array, h_type where the three h_types are mean, absolute, and rmhe."""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan, 
                                                    replace_inf=replace_inf, remove_neg=remove_neg, 
                                                    remove_zero=remove_zero)
    h = (simulated_array - observed_array) / (0.5 * (simulated_array + observed_array))
    if h_type == 'mean':
        return h.mean()
    elif h_type == 'absolute':
        return np.abs(h).mean()
    elif h_type == 'rmhe':
        return np.sqrt((h ** 2).mean())
    else:
        return 'Please make a valid h_type selection'


def h4(simulated_array, observed_array, h_type='mean', replace_nan=None, replace_inf=None, remove_neg=False,
       remove_zero=False):
    """H4 Metric: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985).
    arguments: simulated array, observed array, h_type where the three h_types are mean, absolute, and rmhe."""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan, 
                                                    replace_inf=replace_inf, remove_neg=remove_neg, 
                                                    remove_zero=remove_zero)
    h = (simulated_array - observed_array) / np.sqrt(simulated_array * observed_array)
    if h_type == 'mean':
        return h.mean()
    elif h_type == 'absolute':
        return np.abs(h).mean()
    elif h_type == 'rmhe':
        return np.sqrt((h ** 2).mean())
    else:
        return 'Please make a valid h_type selection'


def h5(simulated_array, observed_array, h_type='mean', replace_nan=None, replace_inf=None, remove_neg=False,
       remove_zero=False):
    """H5 Metric: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985).
    arguments: simulated array, observed array, h_type where the three h_types are mean, absolute, and rmhe."""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan, 
                                                    replace_inf=replace_inf, remove_neg=remove_neg, 
                                                    remove_zero=remove_zero)
    h = (simulated_array - observed_array) / \
        np.reciprocal(0.5 * (np.reciprocal(observed_array) + np.reciprocal(simulated_array)))
    if h_type == 'mean':
        return h.mean()
    elif h_type == 'absolute':
        return np.abs(h).mean()
    elif h_type == 'rmhe':
        return np.sqrt((h ** 2).mean())
    else:
        return 'Please make a valid h_type selection'


def h6(simulated_array, observed_array, h_type='mean', k=1, replace_nan=None, replace_inf=None, remove_neg=False,
       remove_zero=False):
    """H6 Metric: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985).
    arguments: simulated array, observed array, h_type where the three h_types are mean, absolute, and rmhe."""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan, 
                                                    replace_inf=replace_inf, remove_neg=remove_neg, 
                                                    remove_zero=remove_zero)
    h = (simulated_array / observed_array - 1) / \
        np.power(0.5 * (1 + np.power(simulated_array / observed_array, k)), 1 / k)
    if h_type == 'mean':
        return h.mean()
    elif h_type == 'absolute':
        return np.abs(h).mean()
    elif h_type == 'rmhe':
        return np.sqrt((h ** 2).mean())
    else:
        return 'Please make a valid h_type selection'


def h7(simulated_array, observed_array, h_type='mean', replace_nan=None, replace_inf=None, remove_neg=False,
       remove_zero=False):
    """H7 Metric: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985).
    arguments: simulated array, observed array, h_type where the three h_types are mean, absolute, and rmhe."""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan, 
                                                    replace_inf=replace_inf, remove_neg=remove_neg, 
                                                    remove_zero=remove_zero)
    h = (simulated_array / observed_array - 1) / np.min(simulated_array / observed_array)
    if h_type == 'mean':
        return h.mean()
    elif h_type == 'absolute':
        return np.abs(h).mean()
    elif h_type == 'rmhe':
        return np.sqrt((h ** 2).mean())
    else:
        return 'Please make a valid h_type selection'


def h8(simulated_array, observed_array, h_type='mean', replace_nan=None, replace_inf=None, remove_neg=False,
       remove_zero=False):
    """H8 Metric: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985).
    arguments: simulated array, observed array, h_type where the three h_types are mean, absolute, and rmhe."""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan, 
                                                    replace_inf=replace_inf, remove_neg=remove_neg, 
                                                    remove_zero=remove_zero)
    h = (simulated_array / observed_array - 1) / (simulated_array / observed_array).max()
    if h_type == 'mean':
        return h.mean()
    elif h_type == 'absolute':
        return np.abs(h).mean()
    elif h_type == 'rmhe':
        return np.sqrt((h ** 2).mean())
    else:
        return 'Please make a valid h_type selection'


# def h9(simulated_array, observed_array, h_type='mean', k=1):
#     h = (simulated_array / observed_array - 1) / \
#         np.power(0.5 * (1 + np.power(simulated_array / observed_array, k)), 1 / k)
#     if h_type == 'mean':
#         return h.mean()
#     elif h_type == 'absolute':
#         return np.abs(h).mean()
#     elif h_type == 'rmhe':
#         return np.sqrt((h**2).mean())
#     else:
#         return 'Please make a valid h_type selection'


def h10(simulated_array, observed_array, h_type='mean', replace_nan=None, replace_inf=None, remove_neg=False,
        remove_zero=False):
    """H10 Metric: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985).
    arguments: simulated array, observed array, h_type where the three h_types are mean, absolute, and rmhe."""
    assert len(observed_array) == len(simulated_array)
    simulated_array, observed_array = remove_values(simulated_array, observed_array, replace_nan=replace_nan, 
                                                    replace_inf=replace_inf, remove_neg=remove_neg, 
                                                    remove_zero=remove_zero)
    h = np.log(simulated_array) - np.log(observed_array)
    if h_type == 'mean':
        return h.mean()
    elif h_type == 'absolute':
        return np.abs(h).mean()
    elif h_type == 'rmhe':
        return np.sqrt((h ** 2).mean())
    else:
        return 'Please make a valid h_type selection'


def all_metrics(simulated_array, observed_array, mase_m=1, dmod_j = 1, nse_mod_j = 1, h6_k = 1):
    """Takes two numpy arrays and returns a pandas dataframe with all of the metrics included."""
    metrics_list = ['Mean Error', 'Mean Absolute Error', 'Mean Squared Error', 'Eclidean Distance',
                    'Normalized Eclidean Distance', 'Root Mean Square Error', 'Root Mean Squared Log Error',
                    'Mean Absolute Scaled Error', 'R^2', 'Anomoly Correlation Coefficient',
                    'Mean Absolute Percentage Error', 'Mean Absolute Percentage Deviation',
                    'Symmetric Mean Absolute Percentage Error (1)', 'Symmetric Mean Absolute Percentage Error (2)',
                    'Index of Agreement (d)', 'Index of Agreement (d1)', 'Index of Agreement Refined (dr)',
                    'Relative Index of Agreement', 'Modified Index of Agreement', "Watterson's M", 'Mielke-Berry R',
                    'Nash-Sutcliffe Efficiency', 'Modified Nash-Sutcliffe Efficiency',
                    'Relative Nash-Sutcliffe Efficiency',
                    'Legate-McCabe Index', 'Spectral Angle', 'Spectral Correlation',
                    'Spectral Information Divergence', 'Spectral Gradient Angle', 'H1 - Mean', 'H1 - Absolute',
                    'H1 - Root', 'H2 - Mean', 'H2 - Absolute', 'H2 - Root', 'H3 - Mean', 'H3 - Absolute', 'H3 - Root',
                    'H4 - Mean', 'H4 - Absolute', 'H4 - Root', 'H5 - Mean', 'H5 - Absolute', 'H5 - Root', 'H6 - Mean',
                    'H6 - Absolute', 'H6 - Root', 'H7 - Mean', 'H7 - Absolute', 'H7 - Root', 'H8 - Mean',
                    'H8 - Absolute', 'H8 - Root', 'H10 - Mean', 'H10 - Absolute', 'H10 - Root']

    # Creating the Metrics Matrix
    metrics_array = np.zeros(len(metrics_list), dtype=float)

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
    metrics_array[12] = smap1(simulated_array, observed_array)
    metrics_array[13] = smap2(simulated_array, observed_array)
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
    metrics_array[24] = lm_index(simulated_array, observed_array)
    metrics_array[25] = sa(simulated_array, observed_array)
    metrics_array[26] = sc(simulated_array, observed_array)
    metrics_array[27] = sid(simulated_array, observed_array)
    metrics_array[28] = sga(simulated_array, observed_array)
    metrics_array[29] = h1(simulated_array, observed_array, 'mean')
    metrics_array[30] = h1(simulated_array, observed_array, 'absolute')
    metrics_array[31] = h1(simulated_array, observed_array, 'rmhe')
    metrics_array[32] = h2(simulated_array, observed_array, 'mean')
    metrics_array[33] = h2(simulated_array, observed_array, 'absolute')
    metrics_array[34] = h2(simulated_array, observed_array, 'rmhe')
    metrics_array[35] = h3(simulated_array, observed_array, 'mean')
    metrics_array[36] = h3(simulated_array, observed_array, 'absolute')
    metrics_array[37] = h3(simulated_array, observed_array, 'rmhe')
    metrics_array[38] = h4(simulated_array, observed_array, 'mean')
    metrics_array[39] = h4(simulated_array, observed_array, 'absolute')
    metrics_array[40] = h4(simulated_array, observed_array, 'rmhe')
    metrics_array[41] = h5(simulated_array, observed_array, 'mean')
    metrics_array[42] = h5(simulated_array, observed_array, 'absolute')
    metrics_array[43] = h5(simulated_array, observed_array, 'rmhe')
    metrics_array[44] = h6(simulated_array, observed_array, 'mean', k=h6_k)
    metrics_array[45] = h6(simulated_array, observed_array, 'absolute', k=h6_k)
    metrics_array[46] = h6(simulated_array, observed_array, 'rmhe', k=h6_k)
    metrics_array[47] = h7(simulated_array, observed_array, 'mean')
    metrics_array[48] = h7(simulated_array, observed_array, 'absolute')
    metrics_array[49] = h7(simulated_array, observed_array, 'rmhe')
    metrics_array[50] = h8(simulated_array, observed_array, 'mean')
    metrics_array[51] = h8(simulated_array, observed_array, 'absolute')
    metrics_array[52] = h8(simulated_array, observed_array, 'rmhe')
    metrics_array[53] = h10(simulated_array, observed_array, 'mean')
    metrics_array[54] = h10(simulated_array, observed_array, 'absolute')
    metrics_array[55] = h10(simulated_array, observed_array, 'rmhe')
    warnings.filterwarnings("always")

    return pd.DataFrame(np.column_stack([metrics_list, metrics_array]), columns=['Metrics', 'Values'])


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
