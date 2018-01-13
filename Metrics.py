# python 3.6
# -*- coding: utf-8 -*-
"""
Created on Dec 28 12:16:32 2017

@author: Wade Roberts
"""
import numpy as np
import pandas as pd
import numexpr as ne
import warnings

"""####################################################################################################################
            Please note that all of these metrics require inputs of numpy arrays. This is because numpy
            is considerably faster than using only python, and its arrays are very useful for these metrics. 
                        For help, see the numpy documentation at http://www.numpy.org/.
####################################################################################################################"""


def me(forecasted_array, observed_array):
    """Returns the mean error of two 1 dimensional arrays"""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    return (forecasted_array - observed_array).mean()


def mae(forecasted_array, observed_array):
    """Returns the Mean Absolute Error"""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    return (np.absolute(forecasted_array - observed_array)).mean()


def mse(forecasted_array, observed_array):
    """Returns the Mean Squared Error"""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    return ((forecasted_array - observed_array) ** 2).mean()


def ed(forecasted_array, observed_array):
    """Returns the Euclidean Distance"""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    return np.sqrt(((observed_array - forecasted_array) ** 2).sum())


def ned(forecasted_array, observed_array):
    """Returns the Normalized Euclidean Distance"""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    return np.sqrt(
        (((observed_array / observed_array.mean()) - (forecasted_array / forecasted_array.mean())) ** 2).sum())


def rmse(forecasted_array, observed_array):
    """Returns the Root mean squared error"""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    return np.sqrt(((forecasted_array - observed_array) ** 2).mean())


def rmsle(forecasted_array, observed_array):
    """"Return the Root Mean Square Log Error. Note that to calculate the log values, each value in the observed and
    forecasted array is increased by one unit in order to avoid run-time errors and nan values."""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    return np.sqrt(np.mean(np.power(np.log1p(forecasted_array) - np.log1p(observed_array), 2)))


def mase(forecasted_array, observed_array, m=1):
    """Returns the Mean Absolute Scaled Error, the default period for m (seasonal period) is 1.
    Using the default assumes that the data is non-seasonal"""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    start = m
    end = forecasted_array.size - m
    return mae(forecasted_array, observed_array) / \
           (np.sum(np.abs(observed_array[start:observed_array.size] - observed_array[:end])) / end)


def r_squared(forecasted_array, observed_array):
    """Returns the Coefficient of Determination"""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    return (((observed_array - observed_array.mean()) * (forecasted_array - forecasted_array.mean())).sum()) ** 2 / \
           (((observed_array - observed_array.mean()) ** 2).sum() * (
                   (forecasted_array - forecasted_array.mean()) ** 2).sum())


def acc(forecasted_array, observed_array):
    """Returns the Anomaly Correlation Coefficient."""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    return np.dot((forecasted_array - forecasted_array.mean()), (observed_array - observed_array.mean())) / \
           (np.std(observed_array) * np.std(forecasted_array) * forecasted_array.size)


def mape(forecasted_array, observed_array):
    """Returns the Mean Absolute Percentage Error. The answer is a percentage"""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    return np.mean(np.abs(forecasted_array - observed_array) / np.abs(observed_array)) * 100


def mapd(forecasted_array, observed_array):
    """Returns the Mean Absolute Percentage Deviation."""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    return (np.sum(np.abs(forecasted_array - observed_array))) / np.abs(observed_array.sum())


def smap1(forecasted_array, observed_array):
    """Returns the Symmetric Mean Absolute Percentage Error (1)."""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    return (100 / forecasted_array.size) * np.sum(np.abs(forecasted_array - observed_array) /
                                                  (np.abs(forecasted_array) - np.abs(observed_array)))


def smap2(forecasted_array, observed_array):
    """Returns the Symmetric Mean Absolute Percentage Error (2)."""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    num = np.sum(np.abs(forecasted_array - observed_array))
    den = np.sum(forecasted_array + observed_array)
    return (100 / forecasted_array.size) * (num / den)


def d(forecasted_array, observed_array):
    """Returns the Index of Agreement (d)."""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    return 1 - (np.sum((observed_array - forecasted_array) ** 2) /
                np.sum((np.abs(forecasted_array - np.mean(observed_array)) +
                        np.abs(observed_array - np.mean(observed_array))) ** 2))


def d1(forecasted_array, observed_array):
    """Returns the Index of Agreement (d1)."""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    num = np.sum(np.abs(forecasted_array - observed_array))
    den = np.sum((np.abs(forecasted_array - observed_array.mean()) + np.abs(observed_array - observed_array.mean())))
    return 1 - (num / den)


def dr(forecasted_array, observed_array):
    """Returns the Refined Index of Agreement."""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    if np.abs(forecasted_array - observed_array).sum() <= 2 * np.abs(forecasted_array - forecasted_array.mean()).sum():
        return 1 - (np.abs(forecasted_array - observed_array).sum() /
                    np.abs(forecasted_array - forecasted_array.mean()).sum())
    else:
        return (np.sum(np.abs(observed_array - observed_array.mean())) /
                np.sum(np.abs(forecasted_array - observed_array))) - 1


def drel(forecasted_array, observed_array):
    """Returns the Relative Index of Agreement"""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    return 1 - (np.sum(((observed_array - forecasted_array) / observed_array) ** 2) /
                np.sum(((np.abs(forecasted_array - np.mean(observed_array)) +
                         np.abs(observed_array - np.mean(observed_array))) / np.mean(observed_array)) ** 2))


def dmod(forecasted_array, observed_array, j=1):
    """Returns the modified index of agreement, with j=1 as the default."""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    return (1 - (np.sum((np.abs(observed_array - forecasted_array)) ** j)) /
            np.sum((np.abs(forecasted_array - np.mean(observed_array)) +
                    np.abs(observed_array - np.mean(observed_array))) ** j))


def M(forecasted_array, observed_array):
    """Returns Watterson's M value. Watterson IG. 1996. Non-dimensional measures of climate model performance.
    International Journal of Climatology 16: 379–391."""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    return (2 / np.pi) * np.arcsin(1 - mse(forecasted_array, observed_array) /
                                   (np.std(observed_array) ** 2 + np.std(forecasted_array) ** 2 +
                                    (np.mean(forecasted_array) - np.mean(observed_array)) ** 2))


def R(forecasted_array, observed_array):
    """Returns the Mielke-Berry R value. Mielke PW Jr, Berry KJ. 2001. Permutation Methods: A Distance Function Approach.
    Springer-Verlag: New York; 352."""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    forecasted_array2D = forecasted_array[:, None]
    total = ne.evaluate('sum(abs(forecasted_array2D - observed_array))')
    return 1 - (mae(forecasted_array, observed_array) * forecasted_array.size ** 2 / total)


def E(forecasted_array, observed_array):
    """Returns the Nash-Sutcliffe Efficiency value (Nash JE, Sutcliffe JV. 1970. River flow forecasting through
    conceptual models part I—A discussion of principles. Journal of Hydrology 10(3): 282–290.)"""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    return 1 - (
            np.sum((forecasted_array - observed_array) ** 2) / np.sum((observed_array - observed_array.mean()) ** 2))


def Emod(forecasted_array, observed_array, j=1):
    """Returns the modified Nash-Sutcliffe Efficiency value
    (Krause, P., Boyle, D. P., and Base, F.: Comparison of different efficiency criteria for hydrological model
    assessment, Adv. Geosci., 5, 89-97, 2005
    Legates, D. R., and G. J. McCabe Jr. (1999), Evaluating the Use of "Goodness-of-Fit"
    Measures in Hydrologic and Hydroclimatic Model Validation, Water Resour. Res., 35(1), 233-241)"""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    return 1 - (np.sum(np.abs(observed_array - forecasted_array) ** j) / np.sum(
        np.abs(observed_array - np.mean(observed_array)) ** j))


def Erel(forecasted_array, observed_array):
    """Returns the relative Nash-Sutcliffe Efficiency value
    (Krause, P., Boyle, D. P., and Base, F.: Comparison of different efficiency criteria for hydrological model
    assessment, Adv. Geosci., 5, 89-97, 2005
    Legates, D. R., and G. J. McCabe Jr. (1999), Evaluating the Use of "Goodness-of-Fit"
    Measures in Hydrologic and Hydroclimatic Model Validation, Water Resour. Res., 35(1), 233-241)"""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    return 1 - (np.sum(((observed_array - forecasted_array) / observed_array) ** 2) /
                np.sum(np.abs((observed_array - np.mean(observed_array)) / np.mean(observed_array)) ** 2))


def E_1(forecasted_array, observed_array):
    """Returns the Legate-McCabe index. Legates DR, McCabe GJ Jr. 1999. Evaluating the use of “goodness-of-fit” measures
    in hydrologic and hydroclimatic model validation. Water Resources Research 35(1): 233–241."""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    return 1 - (np.sum(np.abs(observed_array - forecasted_array)) / np.sum(
        np.abs(observed_array - np.mean(observed_array))))


def sa(forecasted_array, observed_array):
    """Returns the spectral angle. Robila, S.A.; Gershman, A. In Spectral matching accuracy in processing hyperspectral
    data, Signals, Circuits and Systems, 2005. ISSCS 2005. International Symposium on, 2005; IEEE: pp 163-166."""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    return np.arccos(np.dot(forecasted_array, observed_array) /
                     (np.linalg.norm(forecasted_array) * np.linalg.norm(observed_array)))


def sc(forecasted_array, observed_array):
    """Returns the spectral Correlation. Robila, S.A.; Gershman, A. In Spectral matching accuracy in processing
    hyperspectral data, Signals, Circuits and Systems, 2005. ISSCS 2005. International Symposium on,
    2005; IEEE: pp 163-166."""
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    return np.arccos(np.dot((observed_array - observed_array.mean()), (forecasted_array - forecasted_array.mean())) /
                     (np.linalg.norm(observed_array - observed_array.mean()) *
                      np.linalg.norm(forecasted_array - forecasted_array.mean())))


def sid(forecasted_array, observed_array):
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    first = (observed_array / np.mean(observed_array)) - (forecasted_array / np.mean(forecasted_array))
    second1 = np.log10(observed_array) - np.log10(np.mean(observed_array))
    second2 = np.log10(forecasted_array) - np.log10(np.mean(forecasted_array))
    return np.dot(first, second1 - second2)


def sga(forecasted_array, observed_array):
    assert len(observed_array) == len(forecasted_array)
    forecasted_array, observed_array = remove_nan(forecasted_array, observed_array)
    SGx = observed_array[1:] - observed_array[:observed_array.size - 1]
    SGy = forecasted_array[1:] - forecasted_array[:forecasted_array.size - 1]
    return sa(SGx, SGy)


""" ###################################################################################################################
                        H Metrics: Methods from Tornqvist L, Vartia P, and Vartia YO. (1985)                                       
    ################################################################################################################"""


def h1(forecasted_array, observed_array, type='mean'):
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
    metrics_array[21] = E(forecasted_array, observed_array)
    metrics_array[22] = Emod(forecasted_array, observed_array)
    metrics_array[23] = Erel(forecasted_array, observed_array)
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

""" ###################################################################################################################
    ##############                         Moving on from the Metrics Section                           ###############
    ################################################################################################################"""


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


def remove_nan(forecasted_array, observed_array):
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
