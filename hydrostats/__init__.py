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

"""####################################################################################################################
            Please note that all of these metrics require inputs of numpy arrays. This is because numpy
            is considerably faster than using only python, and its arrays are very useful for these metrics. 
                        For help, see the numpy documentation at http://www.numpy.org/.
####################################################################################################################"""


def me(forecasted_array, observed_array):
    """Returns the mean error of two 1 dimensional arrays
    arguments: forecasted array, observed array"""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    return (forecasted_array - observed_array).mean()


def mae(forecasted_array, observed_array):
    """Returns the Mean Absolute Error
    arguments: forecasted array, observed array"""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    return (np.absolute(forecasted_array - observed_array)).mean()


def mse(forecasted_array, observed_array):
    """Returns the Mean Squared Error
    arguments: forecasted array, observed array"""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    return ((forecasted_array - observed_array) ** 2).mean()


def ed(forecasted_array, observed_array):
    """Returns the Euclidean Distance
    arguments: forecasted array, observed array"""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    return np.sqrt(((observed_array - forecasted_array) ** 2).sum())


def ned(forecasted_array, observed_array):
    """Returns the Normalized Euclidean Distance
    arguments: forecasted array, observed array"""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    return np.sqrt(
        (((observed_array / observed_array.mean()) - (forecasted_array / forecasted_array.mean())) ** 2).sum())


def rmse(forecasted_array, observed_array):
    """Returns the Root mean squared error
    arguments: forecasted array, observed array"""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    return np.sqrt(((forecasted_array - observed_array) ** 2).mean())


def rmsle(forecasted_array, observed_array):
    """"Return the Root Mean Square Log Error. Note that to calculate the log values, each value in the observed and
    forecasted array is increased by one unit in order to avoid run-time errors and nan values.
    arguments: forecasted array, observed array"""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    return np.sqrt(np.mean(np.power(np.log1p(forecasted_array) - np.log1p(observed_array), 2)))


def mase(forecasted_array, observed_array, m=1):
    """Returns the Mean Absolute Scaled Error, the default period for m (seasonal period) is 1.
    Using the default assumes that the data is non-seasonal
    arguments: forecasted array, observed array, m where m is the seasonal period"""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    start = m
    end = forecasted_array.size - m
    return mae(forecasted_array, observed_array) / \
           (np.sum(np.abs(observed_array[start:observed_array.size] - observed_array[:end])) / end)


def r_squared(forecasted_array, observed_array):
    """Returns the Coefficient of Determination
    arguments: forecasted array, observed array"""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    return (((observed_array - observed_array.mean()) * (forecasted_array - forecasted_array.mean())).sum()) ** 2 / \
           (((observed_array - observed_array.mean()) ** 2).sum() * (
                   (forecasted_array - forecasted_array.mean()) ** 2).sum())


def acc(forecasted_array, observed_array):
    """Returns the Anomaly Correlation Coefficient.
    arguments: forecasted array, observed array"""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    return np.dot((forecasted_array - forecasted_array.mean()), (observed_array - observed_array.mean())) / \
           (np.std(observed_array) * np.std(forecasted_array) * forecasted_array.size)


def mape(forecasted_array, observed_array):
    """Returns the Mean Absolute Percentage Error. The answer is a percentage
    arguments: forecasted array, observed array"""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    return np.mean(np.abs(forecasted_array - observed_array) / np.abs(observed_array)) * 100


def mapd(forecasted_array, observed_array):
    """Returns the Mean Absolute Percentage Deviation.
    arguments: forecasted array, observed array"""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    return (np.sum(np.abs(forecasted_array - observed_array))) / np.abs(observed_array.sum())


def smap1(forecasted_array, observed_array):
    """Returns the Symmetric Mean Absolute Percentage Error (1).
    arguments: forecasted array, observed array"""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    return (100 / forecasted_array.size) * np.sum(np.abs(forecasted_array - observed_array) /
                                                  (np.abs(forecasted_array) - np.abs(observed_array)))


def smap2(forecasted_array, observed_array):
    """Returns the Symmetric Mean Absolute Percentage Error (2).
    arguments: forecasted array, observed array"""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    num = np.sum(np.abs(forecasted_array - observed_array))
    den = np.sum(forecasted_array + observed_array)
    return (100 / forecasted_array.size) * (num / den)


def d(forecasted_array, observed_array):
    """Returns the Index of Agreement (d).
    arguments: forecasted array, observed array"""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    return 1 - (np.sum((observed_array - forecasted_array) ** 2) /
                np.sum((np.abs(forecasted_array - np.mean(observed_array)) +
                        np.abs(observed_array - np.mean(observed_array))) ** 2))


def d1(forecasted_array, observed_array):
    """Returns the Index of Agreement (d1).
    arguments: forecasted array, observed array"""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    num = np.sum(np.abs(forecasted_array - observed_array))
    den = np.sum((np.abs(forecasted_array - observed_array.mean()) + np.abs(observed_array - observed_array.mean())))
    return 1 - (num / den)


def dr(forecasted_array, observed_array):
    """Returns the Refined Index of Agreement.
    arguments: forecasted array, observed array"""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    if np.abs(forecasted_array - observed_array).sum() <= 2 * np.abs(forecasted_array - forecasted_array.mean()).sum():
        return 1 - (np.abs(forecasted_array - observed_array).sum() /
                    np.abs(forecasted_array - forecasted_array.mean()).sum())
    else:
        return (np.sum(np.abs(observed_array - observed_array.mean())) /
                np.sum(np.abs(forecasted_array - observed_array))) - 1


def drel(forecasted_array, observed_array):
    """Returns the Relative Index of Agreement.
    arguments: forecasted array, observed array"""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    return 1 - (np.sum(((observed_array - forecasted_array) / observed_array) ** 2) /
                np.sum(((np.abs(forecasted_array - np.mean(observed_array)) +
                         np.abs(observed_array - np.mean(observed_array))) / np.mean(observed_array)) ** 2))


def dmod(forecasted_array, observed_array, j=1):
    """Returns the modified index of agreement, with j=1 as the default.
    arguments: forecasted array, observed array, j"""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    return (1 - (np.sum((np.abs(observed_array - forecasted_array)) ** j)) /
            np.sum((np.abs(forecasted_array - np.mean(observed_array)) +
                    np.abs(observed_array - np.mean(observed_array))) ** j))


def M(forecasted_array, observed_array):
    """Returns Watterson's M value. Watterson IG. 1996. Non-dimensional measures of climate model performance.
    International Journal of Climatology 16: 379–391.
    arguments: forecasted array, observed array"""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    return (2 / np.pi) * np.arcsin(1 - mse(forecasted_array, observed_array) /
                                   (np.std(observed_array) ** 2 + np.std(forecasted_array) ** 2 +
                                    (np.mean(forecasted_array) - np.mean(observed_array)) ** 2))


def R(forecasted_array, observed_array):
    """Returns the Mielke-Berry R value. Mielke PW Jr, Berry KJ. 2001. Permutation Methods: A Distance Function Approach.
    Springer-Verlag: New York; 352.
    arguments: forecasted array, observed array"""
    # Removing Nan Values
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)

    @njit(parallel=True)
    def numba_loop(forecasted_array, observed_array):
        """Using numba for the double for loop"""
        assert len(observed_array) == len(forecasted_array)
        size = len(forecasted_array)
        total = 0.
        for i in prange(size):
            observed = observed_array[i]
            for j in prange(size):
                total += abs(forecasted_array[j] - observed)
        return total, size

    # Using NumPy for the vectorized calculations
    total, size = numba_loop(forecasted_array, observed_array)
    return 1 - (mae(forecasted_array, observed_array) * size ** 2 / total)


def NSE(forecasted_array, observed_array):
    """Returns the Nash-Sutcliffe Efficiency value (Nash JE, Sutcliffe JV. 1970. River flow forecasting through
    conceptual models part I—A discussion of principles. Journal of Hydrology 10(3): 282–290.)
    arguments: forecasted array, observed array"""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    return 1 - (
            np.sum((forecasted_array - observed_array) ** 2) / np.sum((observed_array - observed_array.mean()) ** 2))


def NSEmod(forecasted_array, observed_array, j=1):
    """Returns the modified Nash-Sutcliffe Efficiency value
    (Krause, P., Boyle, D. P., and Base, F.: Comparison of different efficiency criteria for hydrological model
    assessment, Adv. Geosci., 5, 89-97, 2005
    Legates, D. R., and G. J. McCabe Jr. (1999), Evaluating the Use of "Goodness-of-Fit"
    Measures in Hydrologic and Hydroclimatic Model Validation, Water Resour. Res., 35(1), 233-241)
    arguments: forecasted array, observed array, j"""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    return 1 - (np.sum(np.abs(observed_array - forecasted_array) ** j) / np.sum(
        np.abs(observed_array - np.mean(observed_array)) ** j))


def NSErel(forecasted_array, observed_array):
    """Returns the relative Nash-Sutcliffe Efficiency value
    (Krause, P., Boyle, D. P., and Base, F.: Comparison of different efficiency criteria for hydrological model
    assessment, Adv. Geosci., 5, 89-97, 2005
    Legates, D. R., and G. J. McCabe Jr. (1999), Evaluating the Use of "Goodness-of-Fit"
    Measures in Hydrologic and Hydroclimatic Model Validation, Water Resour. Res., 35(1), 233-241)
    arguments: forecasted array, observed array"""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    return 1 - (np.sum(((observed_array - forecasted_array) / observed_array) ** 2) /
                np.sum(np.abs((observed_array - np.mean(observed_array)) / np.mean(observed_array)) ** 2))


def E_1(forecasted_array, observed_array):
    """Returns the Legate-McCabe index. Legates DR, McCabe GJ Jr. 1999. Evaluating the use of “goodness-of-fit” measures
    in hydrologic and hydroclimatic model validation. Water Resources Research 35(1): 233–241.
    arguments: forecasted array, observed array"""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    return 1 - (np.sum(np.abs(observed_array - forecasted_array)) / np.sum(
        np.abs(observed_array - np.mean(observed_array))))


def sa(forecasted_array, observed_array):
    """Returns the spectral angle. Robila, S.A.; Gershman, A. In Spectral matching accuracy in processing hyperspectral
    data, Signals, Circuits and Systems, 2005. ISSCS 2005. International Symposium on, 2005; IEEE: pp 163-166.
    arguments: forecasted array, observed array"""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    return np.arccos(np.dot(forecasted_array, observed_array) /
                     (np.linalg.norm(forecasted_array) * np.linalg.norm(observed_array)))


def sc(forecasted_array, observed_array):
    """Returns the spectral Correlation. Robila, S.A.; Gershman, A. In Spectral matching accuracy in processing
    hyperspectral data, Signals, Circuits and Systems, 2005. ISSCS 2005. International Symposium on,
    2005; IEEE: pp 163-166.
    arguments: forecasted array, observed array"""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    return np.arccos(np.dot((observed_array - observed_array.mean()), (forecasted_array - forecasted_array.mean())) /
                     (np.linalg.norm(observed_array - observed_array.mean()) *
                      np.linalg.norm(forecasted_array - forecasted_array.mean())))


def sid(forecasted_array, observed_array):
    """Returns the ___
    arguments: forecasted array, observed array"""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    first = (observed_array / np.mean(observed_array)) - (forecasted_array / np.mean(forecasted_array))
    second1 = np.log10(observed_array) - np.log10(np.mean(observed_array))
    second2 = np.log10(forecasted_array) - np.log10(np.mean(forecasted_array))
    return np.dot(first, second1 - second2)


def sga(forecasted_array, observed_array):
    """Returns the spectral gradient angle
    arguments: forecasted array, observed array"""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    SGx = observed_array[1:] - observed_array[:observed_array.size - 1]
    SGy = forecasted_array[1:] - forecasted_array[:forecasted_array.size - 1]
    return sa(SGx, SGy)


""" ###################################################################################################################
                        H Metrics: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985)                                       
    ################################################################################################################"""


def h1(forecasted_array, observed_array, type='mean'):
    """H1 Metric: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985).
    arguments: forecasted array, observed array, type where the three types are mean, absolute, and rmhe."""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    h = (forecasted_array - observed_array) / observed_array
    if type == 'mean':
        return h.mean()
    elif type == 'absolute':
        return np.abs(h).mean()
    elif type == 'rmhe':
        return np.sqrt((h ** 2).mean())
    else:
        return 'Please make a valid type selection'


def h2(forecasted_array, observed_array, type='mean'):
    """H2 Metric: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985).
    arguments: forecasted array, observed array, type where the three types are mean, absolute, and rmhe."""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    h = (forecasted_array - observed_array) / forecasted_array
    if type == 'mean':
        return h.mean()
    elif type == 'absolute':
        return np.abs(h).mean()
    elif type == 'rmhe':
        return np.sqrt((h ** 2).mean())
    else:
        return 'Please make a valid type selection'


def h3(forecasted_array, observed_array, type='mean'):
    """H3 Metric: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985).
    arguments: forecasted array, observed array, type where the three types are mean, absolute, and rmhe."""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    h = (forecasted_array - observed_array) / (0.5 * (forecasted_array + observed_array))
    if type == 'mean':
        return h.mean()
    elif type == 'absolute':
        return np.abs(h).mean()
    elif type == 'rmhe':
        return np.sqrt((h ** 2).mean())
    else:
        return 'Please make a valid type selection'


def h4(forecasted_array, observed_array, type='mean'):
    """H4 Metric: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985).
    arguments: forecasted array, observed array, type where the three types are mean, absolute, and rmhe."""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    h = (forecasted_array - observed_array) / np.sqrt(forecasted_array * observed_array)
    if type == 'mean':
        return h.mean()
    elif type == 'absolute':
        return np.abs(h).mean()
    elif type == 'rmhe':
        return np.sqrt((h ** 2).mean())
    else:
        return 'Please make a valid type selection'


def h5(forecasted_array, observed_array, type='mean'):
    """H5 Metric: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985).
    arguments: forecasted array, observed array, type where the three types are mean, absolute, and rmhe."""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    h = (forecasted_array - observed_array) / \
        np.reciprocal(0.5 * (np.reciprocal(observed_array) + np.reciprocal(forecasted_array)))
    if type == 'mean':
        return h.mean()
    elif type == 'absolute':
        return np.abs(h).mean()
    elif type == 'rmhe':
        return np.sqrt((h ** 2).mean())
    else:
        return 'Please make a valid type selection'


def h6(forecasted_array, observed_array, type='mean', k=1):
    """H6 Metric: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985).
    arguments: forecasted array, observed array, type where the three types are mean, absolute, and rmhe."""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    h = (forecasted_array / observed_array - 1) / \
        np.power(0.5 * (1 + np.power(forecasted_array / observed_array, k)), 1 / k)
    if type == 'mean':
        return h.mean()
    elif type == 'absolute':
        return np.abs(h).mean()
    elif type == 'rmhe':
        return np.sqrt((h ** 2).mean())
    else:
        return 'Please make a valid type selection'


def h7(forecasted_array, observed_array, type='mean', k=1):
    """H7 Metric: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985).
    arguments: forecasted array, observed array, type where the three types are mean, absolute, and rmhe."""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    h = (forecasted_array / observed_array - 1) / (forecasted_array / observed_array).min()
    if type == 'mean':
        return h.mean()
    elif type == 'absolute':
        return np.abs(h).mean()
    elif type == 'rmhe':
        return np.sqrt((h ** 2).mean())
    else:
        return 'Please make a valid type selection'


def h8(forecasted_array, observed_array, type='mean', k=1):
    """H8 Metric: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985).
    arguments: forecasted array, observed array, type where the three types are mean, absolute, and rmhe."""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    h = (forecasted_array / observed_array - 1) / (forecasted_array / observed_array).max()
    if type == 'mean':
        return h.mean()
    elif type == 'absolute':
        return np.abs(h).mean()
    elif type == 'rmhe':
        return np.sqrt((h ** 2).mean())
    else:
        return 'Please make a valid type selection'


# def h9(forecasted_array, observed_array, type='mean', k=1):
#     h = (forecasted_array / observed_array - 1) / \
#         np.power(0.5 * (1 + np.power(forecasted_array / observed_array, k)), 1 / k)
#     if type == 'mean':
#         return h.mean()
#     elif type == 'absolute':
#         return np.abs(h).mean()
#     elif type == 'rmhe':
#         return np.sqrt((h**2).mean())
#     else:
#         return 'Please make a valid type selection'


def h10(forecasted_array, observed_array, type='mean', k=1):
    """H10 Metric: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985).
    arguments: forecasted array, observed array, type where the three types are mean, absolute, and rmhe."""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    h = np.log(forecasted_array) - np.log(observed_array)
    if type == 'mean':
        return h.mean()
    elif type == 'absolute':
        return np.abs(h).mean()
    elif type == 'rmhe':
        return np.sqrt((h ** 2).mean())
    else:
        return 'Please make a valid type selection'


def all_metrics(forecasted_array, observed_array):
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

    metrics_array[0] = me(forecasted_array, observed_array)
    warnings.filterwarnings("ignore")
    metrics_array[1] = mae(forecasted_array, observed_array)
    metrics_array[2] = mse(forecasted_array, observed_array)
    metrics_array[3] = ed(forecasted_array, observed_array)
    metrics_array[4] = ned(forecasted_array, observed_array)
    metrics_array[5] = rmse(forecasted_array, observed_array)
    metrics_array[6] = rmsle(forecasted_array, observed_array)
    metrics_array[7] = mase(forecasted_array, observed_array)
    metrics_array[8] = r_squared(forecasted_array, observed_array)
    metrics_array[9] = acc(forecasted_array, observed_array)
    metrics_array[10] = mape(forecasted_array, observed_array)
    metrics_array[11] = mapd(forecasted_array, observed_array)
    metrics_array[12] = smap1(forecasted_array, observed_array)
    metrics_array[13] = smap2(forecasted_array, observed_array)
    metrics_array[14] = d(forecasted_array, observed_array)
    metrics_array[15] = d1(forecasted_array, observed_array)
    metrics_array[16] = dr(forecasted_array, observed_array)
    metrics_array[17] = drel(forecasted_array, observed_array)
    metrics_array[18] = dmod(forecasted_array, observed_array)
    metrics_array[19] = M(forecasted_array, observed_array)
    metrics_array[20] = R(forecasted_array, observed_array)
    metrics_array[21] = NSE(forecasted_array, observed_array)
    metrics_array[22] = NSEmod(forecasted_array, observed_array)
    metrics_array[23] = NSErel(forecasted_array, observed_array)
    metrics_array[24] = E_1(forecasted_array, observed_array)
    metrics_array[25] = sa(forecasted_array, observed_array)
    metrics_array[26] = sc(forecasted_array, observed_array)
    metrics_array[27] = sid(forecasted_array, observed_array)
    metrics_array[28] = sga(forecasted_array, observed_array)
    metrics_array[29] = h1(forecasted_array, observed_array, 'mean')
    metrics_array[30] = h1(forecasted_array, observed_array, 'absolute')
    metrics_array[31] = h1(forecasted_array, observed_array, 'rmhe')
    metrics_array[32] = h2(forecasted_array, observed_array, 'mean')
    metrics_array[33] = h2(forecasted_array, observed_array, 'absolute')
    metrics_array[34] = h2(forecasted_array, observed_array, 'rmhe')
    metrics_array[35] = h3(forecasted_array, observed_array, 'mean')
    metrics_array[36] = h3(forecasted_array, observed_array, 'absolute')
    metrics_array[37] = h3(forecasted_array, observed_array, 'rmhe')
    metrics_array[38] = h4(forecasted_array, observed_array, 'mean')
    metrics_array[39] = h4(forecasted_array, observed_array, 'absolute')
    metrics_array[40] = h4(forecasted_array, observed_array, 'rmhe')
    metrics_array[41] = h5(forecasted_array, observed_array, 'mean')
    metrics_array[42] = h5(forecasted_array, observed_array, 'absolute')
    metrics_array[43] = h5(forecasted_array, observed_array, 'rmhe')
    metrics_array[44] = h6(forecasted_array, observed_array, 'mean')
    metrics_array[45] = h6(forecasted_array, observed_array, 'absolute')
    metrics_array[46] = h6(forecasted_array, observed_array, 'rmhe')
    metrics_array[47] = h7(forecasted_array, observed_array, 'mean')
    metrics_array[48] = h7(forecasted_array, observed_array, 'absolute')
    metrics_array[49] = h7(forecasted_array, observed_array, 'rmhe')
    metrics_array[50] = h8(forecasted_array, observed_array, 'mean')
    metrics_array[51] = h8(forecasted_array, observed_array, 'absolute')
    metrics_array[52] = h8(forecasted_array, observed_array, 'rmhe')
    metrics_array[53] = h10(forecasted_array, observed_array, 'mean')
    metrics_array[54] = h10(forecasted_array, observed_array, 'absolute')
    metrics_array[55] = h10(forecasted_array, observed_array, 'rmhe')
    warnings.filterwarnings("always")

    return pd.DataFrame(np.column_stack([metrics_list, metrics_array]), columns=['Metrics', 'Values'])



def remove_nan(forecasted_array, observed_array):
    """Removes the nan, negative, and inf values in two numpy arrays"""
    warnings.filterwarnings("ignore")
    # logical array of nans and infs for simulated
    simLog = np.logical_and(~np.isnan(forecasted_array), ~np.isinf(forecasted_array))
    # logical array of nans, infs, and < 0 for simulated
    simLog = np.logical_and(simLog, np.greater(forecasted_array, 0))
    # logical array of nans and infs for observed
    obsLog = np.logical_and(~np.isnan(observed_array), ~np.isinf(observed_array))
    # logical array of nans, infs, and < 0 for simulated
    obsLog = np.logical_and(obsLog, np.greater(observed_array, 0))
    # logical array of nans, infs, and <0 for both
    allLog = np.logical_and(simLog, obsLog)
    new_forecasted = forecasted_array[allLog]
    new_observed = observed_array[allLog]
    warnings.filterwarnings("always")
    if new_forecasted.size < forecasted_array.size:
        warnings.warn("One of the arrays contained negative, nan, or inf values and they have been removed.",
                      Warning)
    return new_forecasted, new_observed

