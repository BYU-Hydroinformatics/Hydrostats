# python 3.6
# -*- coding: utf-8 -*-
"""

The metrics module contains all of the metrics included in hydrostats. Each metric is contained in
function, and every metric has the ability to treat missing values as well as remove zero and
negative values from the timeseries data.

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
           'kge_2012', 'lm_index', 'd1_p', 've', 'sa', 'sc', 'sid', 'sga', 'h1_mhe', 'h1_ahe',
           'h1_rmshe', 'h2_mhe', 'h2_ahe', 'h2_rmshe', 'h3_mhe', 'h3_ahe', 'h3_rmshe', 'h4_mhe',
           'h4_ahe', 'h4_rmshe', 'h5_mhe', 'h5_ahe', 'h5_rmshe', 'h6_mhe', 'h6_ahe', 'h6_rmshe',
           'h7_mhe', 'h7_ahe', 'h7_rmshe', 'h8_mhe', 'h8_ahe', 'h8_rmshe', 'h10_mhe', 'h10_ahe',
           'h10_rmshe', 'g_mean_diff', 'mean_var']


####################################################################################################
#                            General and Hydrological Error Metrics                                #
####################################################################################################


def me(simulated_array, observed_array, replace_nan=None, replace_inf=None,
       remove_neg=False, remove_zero=False):
    """

    Compute the mean error of the simulated and observed data.

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/ME.png?raw=true)

    **Range:** -inf < MAE < inf, data units, closer to zero is better, indicates bias.

    **Notes:** The mean error (ME) measures the difference between the simulated data and the
    observed data. For the mean error, a smaller number indicates a better fit to the original
    data. Note that if the error is in the form of random noise, the mean error will be very small,
    which can skew the accuracy of this metric. ME is cumulative and will be small even if there
    are large positive and negative errors that balance.

    **Parameters**

    simulated_array:  *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when 
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation. If None, when 
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when 
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation. If None, when 
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **Example**

    ```python
    import hydrostats as hs
    import numpy as np

    sim = np.arange(10)
    obs = np.random.rand(10)

    hs.me(sim, obs)
    ```

    **References**

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
    simulated_array, observed_array = remove_values(simulated_array, observed_array,
                                                    replace_nan=replace_nan,
                                                    replace_inf=replace_inf,
                                                    remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    return np.mean(simulated_array - observed_array)


def mae(simulated_array, observed_array, replace_nan=None, replace_inf=None,
        remove_neg=False, remove_zero=False):
    """
    Compute the mean absolute error of the simulated and observed data.

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/MAE.png?raw=true)

    **Notes:** The ME measures the absolute difference between the simulated data and the observed
    data. For the mean abolute error, a smaller number indicates a better fit to the original data.
    Also note that random errors do not cancel. Also referred to as an L1-norm.

    **Range:** 0 ≤ MAE < inf, data units, smaller is better, does not indicate bias.

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when 
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation. If None, when 
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when 
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation. If None, when 
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References**

    - Willmott, Cort J., and Kenji Matsuura. “Advantages of the Mean Absolute Error (MAE) over the
      Root Mean Square Error (RMSE) in Assessing Average Model Performance.” Climate Research 30,
      no. 1 (2005): 79–82.
    - Willmott, Cort J., and Kenji Matsuura. “On the Use of Dimensioned Measures of Error to
      Evaluate the Performance of Spatial Interpolators.” International Journal of Geographical
      Information Science 20, no. 1 (2006): 89–102.
    """

    # Checking and cleaning the data
    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
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

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/MSE.png?raw=true)

    **Range:** 0 ≤ MSE < inf, data units squared, smaller is better, does not indicate bias.

    **Notes:** Random errors do not cancel, highlights larger errors, also referred to as a
    squared L2-norm.

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when 
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation. If None, when 
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when 
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References**
    - Wang, Zhou, and Alan C. Bovik. “Mean Squared Error: Love It or Leave It? A New Look at Signal
      Fidelity Measures.” IEEE Signal Processing Magazine 26, no. 1 (2009): 98–117.
    """

    # Checking and cleaning the data
    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
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

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/MLE.png?raw=true)

    **Range:** -inf < MLE < inf, data units, closer to zero is better, indicates bias.

    **Notes** Same as ME only use log ratios as the error term. Limits the impact of outliers, more
    evenly weights high and low data values.

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when 
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation. If None, when 
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when 
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References**

    - Törnqvist, Leo, Pentti Vartia, and Yrjö O. Vartia. “How Should Relative Changes Be Measured?”
      The American Statistician 39, no. 1 (1985): 43–46.
    """

    # Checking and cleaning the data
    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    sim_log = np.log(simulated_array)
    obs_log = np.log(observed_array)
    return np.mean(sim_log - obs_log)


def male(simulated_array, observed_array, replace_nan=None, replace_inf=None,
         remove_neg=False, remove_zero=False):
    """
    Compute the mean absolute log error of the simulated and observed data.

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/MALE.png?raw=true)

    **Range:** 0 ≤ MALE < inf, data units squared, smaller is better, does not indicate bias.

    **Notes** Same as MAE only use log ratios as the error term. Limits the impact of outliers,
    more evenly weights high and low flows.

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when 
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation. If None, when 
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when 
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References**

    - Törnqvist, Leo, Pentti Vartia, and Yrjö O. Vartia. “How Should Relative Changes Be Measured?”
      The American Statistician 39, no. 1 (1985): 43–46.
    """

    # Checking and cleaning the data
    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    sim_log = np.log(simulated_array)
    obs_log = np.log(observed_array)
    return np.mean(np.abs(sim_log - obs_log))


def msle(simulated_array, observed_array, replace_nan=None, replace_inf=None,
         remove_neg=False, remove_zero=False):
    """
    Compute the mean squared log error of the simulated and observed data.

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/MSLE.png?raw=true)

    **Range:** 0 ≤ MSLE < inf, data units squared, smaller is better, does not indicate bias.

    **Notes** Same as the mean squared error (MSE) only use log ratios as the error term. Limits
    the impact of outliers, more evenly weights high and low values.

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when 
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation. If None, when 
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when 
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References**

    - Törnqvist, Leo, Pentti Vartia, and Yrjö O. Vartia. “How Should Relative Changes Be Measured?”
      The American Statistician 39, no. 1 (1985): 43–46.
    """
    # Checking and cleaning the data
    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    sim_log = np.log(simulated_array)
    obs_log = np.log(observed_array)
    return np.mean((sim_log - obs_log) ** 2)


def mde(simulated_array, observed_array, replace_nan=None, replace_inf=None,
        remove_neg=False, remove_zero=False):
    """
    Compute the median error (MdE) between the simulated and observed data.

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/MdE.png?raw=true)

    **Range** -inf < MdE < inf, closer to zero is better.
    **Notes** This metric indicates bias. It is similar to the mean error (ME), only it takes the
    median rather than the mean. Median measures reduces the impact of outliers.

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation. If None, when
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    """

    # Checking and cleaning the data
    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
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

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/MdAE.png?raw=true)

    **Range** 0 ≤ MdAE < inf, closer to zero is better.

    **Notes** This metric does not indicates bias. Random errors (noise) do not cancel.
    It is similar to the mean absolute error (MAE), only it takes the median rather than
    the mean. Median measures reduces the impact of outliers.

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation. If None, when
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    """

    # Checking and cleaning the data
    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
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

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/MdSE.png?raw=true)

    **Range** 0 ≤ MdSE < inf, closer to zero is better.

    **Notes** This metric does not indicates bias. Random errors (noise) do not cancel.
    It is similar to the mean squared error (MSE), only it takes the median rather than
    the mean. Median measures reduces the impact of outliers.

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation. If None, when
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    """

    # Checking and cleaning the data
    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    return np.median((simulated_array - observed_array)**2)


def ed(simulated_array, observed_array, replace_nan=None, replace_inf=None,
       remove_neg=False, remove_zero=False):
    """
    Compute the Euclidean distance between predicted and observed values in vector space.

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/ED.png?raw=true)

    **Range** 0 ≤ ED < inf, smaller is better.
    **Notes** This metric does not indicate bias. It is also sometimes referred to as the L2-norm.

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when 
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation. If None, when 
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when 
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References**

    - Kennard, M. J., Mackay, S. J., Pusey, B. J., Olden, J. D., & Marsh, N. (2010). Quantifying
      uncertainty in estimation of hydrologic metrics for ecohydrological studies. River Research
      and Applications, 26(2), 137-156.
    """

    # Checking and cleaning the data
    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
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

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/NED.png?raw=true)

    **Range** 0 ≤ NED < inf, smaller is better.
    **Notes** This metric does not indicate bias. It is also sometimes referred to as the squared
    L2-norm.

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when 
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation. If None, when 
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when 
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    - Kennard, M. J., Mackay, S. J., Pusey, B. J., Olden, J. D., & Marsh, N. (2010). Quantifying
      uncertainty in estimation of hydrologic metrics for ecohydrological studies. River Research
      and Applications, 26(2), 137-156.

    """

    # Checking and cleaning the data
    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
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

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/RMSE.png?raw=true)

    **Range** 0 ≤ RMSE < inf, smaller is better.
    **Notes** Random errors do not cancel. This metric will highlights larger errors. It is also
    referred to as an L2-norm.

    Hyndman and Koehler, 2006; Willmott and Matsuura, 2005

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when 
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation. If None, when 
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when 
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**
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
    simulated_array, observed_array = remove_values(
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

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/RMSLE.png?raw=true)

    **Range:** 0 ≤ RMSLE < inf. Smaller is better, and it does not indicate bias.

    **Notes:** Random errors do not cancel while using this metric. This metric also limits the
    impact of outliers by more evenly weighting high and low values. To calculate the log values,
    each value in the observed and simulated array is increased by one unit in order to avoid
    run-time errors and nan values (function np.log1p).

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when 
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation. If None, when 
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when 
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

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
    simulated_array, observed_array = remove_values(simulated_array, observed_array,
                                                    replace_nan=replace_nan,
                                                    replace_inf=replace_inf, remove_neg=remove_neg,
                                                    remove_zero=remove_zero)
    return np.sqrt(np.mean(np.power(np.log1p(simulated_array) - np.log1p(observed_array), 2)))


def nrmse_range(simulated_array, observed_array, replace_nan=None, replace_inf=None,
                remove_neg=False, remove_zero=False):
    """

    Compute the range normalized root mean square error between the simulated and observed data.

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/NRMSE_Range.png?raw=true)

    **Range:** 0 ≤ NRMSE < inf.

    **Notes:** This metric is the RMSE normalized by the range of the observed time series (x).
    Normalizing allows comparison between data sets with different scales. The NRMSErange is the
    most sensitive to outliers of the three normalized rmse metrics.

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when 
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation. If None, when 
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when 
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    - Pontius, R.G., Thontteh, O., Chen, H., 2008. Components of information for multiple
    resolution comparison between maps that share a real variable. Environmental and Ecological
    Statistics 15(2) 111-142.

    """

    # Checking and cleaning the data
    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
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
    """

    Compute the range normalized root mean square error between the simulated and observed data.

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/NRMSE_Mean.png?raw=true)

    **Range:** 0 ≤ NRMSE < inf.

    **Notes:** This metric is the RMSE normalized by the mean of the observed time series (x).
    Normalizing allows comparison between data sets with different scales.

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when 
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation. If None, when 
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when 
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    - Pontius, R.G., Thontteh, O., Chen, H., 2008. Components of information for multiple
    resolution comparison between maps that share a real variable. Environmental and Ecological
    Statistics 15(2) 111-142.

    """

    # Checking and cleaning the data
    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    rmse_value = rmse(simulated_array=simulated_array, observed_array=observed_array)
    obs_mean = np.mean(observed_array)
    return rmse_value / obs_mean


def nrmse_iqr(simulated_array, observed_array, replace_nan=None, replace_inf=None,
              remove_neg=False, remove_zero=False):
    """

    Compute the range normalized root mean square error between the simulated and observed data.

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/NRMSE_IQR.png?raw=true)

    **Range:** 0 ≤ NRMSE < inf.

    **Notes:** This metric is the RMSE normalized by the range of the observed time series (x).
    Normalizing allows comparison between data sets with different scales. The NRMSEquartile is
    the least sensitive to outliers of the three normalized rmse metrics.

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when 
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation. If None, when 
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when 
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    - Pontius, R.G., Thontteh, O., Chen, H., 2008. Components of information for multiple
    resolution comparison between maps that share a real variable. Environmental and Ecological
    Statistics 15(2) 111-142.

    """

    # Checking and cleaning the data
    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    rmse_value = rmse(simulated_array=simulated_array, observed_array=observed_array)
    q1 = np.percentile(observed_array, 25)
    q3 = np.percentile(observed_array, 75)
    iqr = q3 - q1
    return rmse_value / iqr


def irmse(simulated_array, observed_array, replace_nan=None, replace_inf=None,
              remove_neg=False, remove_zero=False):
    """

    Compute the inertial root mean square error (IRMSE) between the simulated and observed data.

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/IRMSE1.png?raw=true)
    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/IRMSE2.png?raw=true)
    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/IRMSE3.png?raw=true)

    **Range:** 0 ≤ IRMSE < inf, lower is better.

    **Notes:** This metric is the RMSE devided by by the standard deviation of the gradient of the
    observed timeseries data. This metric is meant to be help understand the ability of the model
    to predict changes in observation.

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation. If None, when
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    - Daga, M., Deo, M.C., 2009. Alternative data-driven methods to estimate wind from waves by
    inverse modeling. Natural Hazards 49(2) 293-310.

    """

    # Checking and cleaning the data
    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    # Getting the gradient of the observed data
    obs_len = observed_array.size
    obs_grad = observed_array[1:obs_len] - observed_array[obs_len - 1]

    # Standard deviation of the gradient
    obs_grad_std = np.std(obs_grad)

    # Divide RMSE by the standard deviation of the gradient of the observed data
    rmse_value = np.sqrt(np.mean((simulated_array - observed_array) ** 2))
    return rmse_value / obs_grad_std


def mase(simulated_array, observed_array, m=1, replace_nan=None, replace_inf=None,
         remove_neg=False, remove_zero=False):
    """

    Compute the mean absolute scaled error between the simulated and observed data.

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/MASE.png?raw=true)

    **Range:**

    **Notes:**

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when 
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation. If None, when 
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when 
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    - Hyndman, R.J., Koehler, A.B., 2006. Another look at measures of forecast accuracy.
    International Journal of Forecasting 22(4) 679-688.

    """

    # Checking and cleaning the data
    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")

    simulated_array, observed_array = remove_values(
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

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/R_pearson.png?raw=true)

    **Range:** -1 ≤ R (Pearson) ≤ 1. 1 indicates perfect postive correlation, 0 indicates
    complete randomness, -1 indicate perfect negative correlation.

    **Notes:** The pearson r coefficient measures linear correlation. It can be affected negatively
    by outliers.

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when 
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation. If None, when 
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when 
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    """
    # Checking and cleaning the data
    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
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

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/R_spearman.png?raw=true)

    **Range:** -1 ≤ R (Pearson) ≤ 1. 1 indicates perfect postive correlation, 0 indicates
    complete randomness, -1 indicate perfect negative correlation.

    **Notes:** The spearman r coefficient measures the monotomic relation between simulated and
    observed data. Because it uses a nonparametric measure of rank correlation, it is less
    influenced by outliers than the pearson correlation coefficient.

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when 
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation. If None, when 
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when 
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    - Spearman C (1904). "The proof and measurement of association between two things". American
    Journal of Psychology. 15: 72–101. doi:10.2307/1412159

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
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

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/r2.png?raw=true)

    **Range:** 0 ≤ r2 ≤ 1. 1 indicates perfect correlation, 0 indicates complete randomness.

    **Notes:** The Coefficient of Determination measures the linear relation between simulated and
    observed data. Because it is the pearson correlation coefficient squared, it is more heavily
    affected by outliers than the pearson correlation coefficient.

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when 
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation. If None, when 
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when 
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
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

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/ACC.png?raw=true)

    **Range:** -1 ≤ ACC ≤ 1. -1 indicates perfect negative correlation of the variation pattern of
    the anomalies, 0 indicates complete randomness of the variation pattern of the anomalies,
    1 indicates perfect correlation of the variation pattern of the anomalies.

    **Notes:** Measures the correlation between the variation pattern of the simulated data
    compared to the observed data.

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when 
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation. If None, when 
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when 
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    - Langland, Rolf H., and Ryan N. Maue. “Recent Northern Hemisphere Mid-Latitude Medium-Range
    Deterministic Forecast Skill.” Tellus A: Dynamic Meteorology and Oceanography 64,
    no. 1 (2012): 17531.
    - Miyakoda, K., G. D. Hembree, R. F. Strickler, and I. Shulman. “Cumulative Results of Extended
    Forecast Experiments I. Model Performance for Winter Cases.” Monthly Weather Review 100, no. 12
    (1972): 836–55.
    - Murphy, Allan H., and Edward S. Epstein. “Skill Scores and Correlation Coefficients in Model
    Verification.” Monthly Weather Review 117, no. 3 (1989): 572–82.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    a = simulated_array - np.mean(simulated_array)
    b = observed_array - np.mean(observed_array)
    c = np.std(observed_array) * np.std(simulated_array) * simulated_array.size
    return np.dot(a, b / c)


def mape(simulated_array, observed_array, replace_nan=None, replace_inf=None,
         remove_neg=False, remove_zero=False):
    """

    Compute the the mean absolute percentage error (MAPE).

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/MAPE.png?raw=true)

    **Range:** 0% ≤ MAPE ≤ 100%. 100% indicates perfect correlation, 0% indicates complete
    randomness.

    **Notes:**

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when 
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when 
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
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
    """

    Compute the the mean absolute percentage deviation (MAPD).

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/MAPD.png?raw=true)

    **Range:**

    **Notes:**

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when 
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when 
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
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
    """

    Compute the the Mean Arctangent Absolute Percentage Error (MAAPE).

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/MAAPE.png?raw=true)

    **Range:** 0 ≤ MAAPE < π/2, does not indicate bias, smaller is better.

    **Notes:** Represents the mean absolute error as a percentage of the observed values. Handles
    0s in the observed data. This metric is not as biased as MAPE by under-over predictions.

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when 
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when 
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    - Kim, S., Kim, H., 2016. A new metric of absolute percentage error for intermittent demand
    forecasts. International Journal of Forecasting 32(3) 669-679.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
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

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/SMAPE1.png?raw=true)

    **Range:** 0 ≤ SMAPE1 < 200%, does not indicate bias, smaller is better, symmetrical.

    **Notes:** This metric is an adjusted version of the MAPE.

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when 
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when 
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    - Flores, B.E., 1986. A pragmatic view of accuracy measurement in forecasting. Omega 14(2)
    93-98.
    - Goodwin, P., Lawton, R., 1999. On the asymmetry of the symmetric MAPE. International Journal
    of Forecasting 15(4) 405-408.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
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

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/SMAPE2.png?raw=true)

    **Range:** 0 ≤ SMAPE1 < 200%, does not indicate bias, smaller is better, symmetrical.

    **Notes:** This metric is an adjusted version of the MAPE with only positive metric values.

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when 
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when 
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    - Flores, B.E., 1986. A pragmatic view of accuracy measurement in forecasting. Omega 14(2)
    93-98.
    - Goodwin, P., Lawton, R., 1999. On the asymmetry of the symmetric MAPE. International Journal
    of Forecasting 15(4) 405-408.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
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

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/d.png?raw=true)

    **Range:** 0 ≤ d < 1, does not indicate bias, larger is better.

    **Notes:** This metric is a modified approach to the Nash-Sutcliffe Efficiency metric.

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when 
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when 
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    - Legates, D.R., McCabe Jr, G.J., 1999. Evaluating the use of “goodness‐of‐fit” Measures in
    hydrologic and hydroclimatic model validation. Water Resources Research 35(1) 233-241.
    - Willmott, C.J., Robeson, S.M., Matsuura, K., 2012. A refined index of model performance.
    International Journal of Climatology 32(13) 2088-2094.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
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

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/d1.png?raw=true)

    **Range:** 0 ≤ d < 1, does not indicate bias, larger is better.

    **Notes:** This metric is a modified approach to the Nash-Sutcliffe Efficiency metric. Compared
    to the other index of agreement (d) it has a reduced impact of outliers.

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when 
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when 
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    - Willmott, C.J., Robeson, S.M., Matsuura, K., 2012. A refined index of model performance.
    International Journal of Climatology 32(13) 2088-2094.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
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

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/dr1.png?raw=true)
    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/dr2.png?raw=true)
    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/dr3.png?raw=true)

    **Range:** -1 ≤ dr < 1, does not indicate bias, larger is better.

    **Notes:** This metric was created to address issues in the index of agreement and the
    Nash-Sutcliffe efficiency metric. It is easy to interpret.

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when 
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when 
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    - Willmott, C.J., Robeson, S.M., Matsuura, K., 2012. A refined index of model performance.
    International Journal of Climatology 32(13) 2088-2094.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
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
    """

    Compute the the relative index of agreement (drel).

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/drel.png?raw=true)

    **Range:** 0 ≤ drel < 1, does not indicate bias, larger is better.

    **Notes:** Instead of absolute differences, this metric uses relative differences.

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when 
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when 
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    - Krause, P., Boyle, D., Bäse, F., 2005. Comparison of different efficiency criteria for
    hydrological model assessment. Advances in geosciences 5 89-97.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
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

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/dmod.png?raw=true)

    **Range:** 0 ≤ dmod < 1, does not indicate bias, larger is better.

    **Notes:** When j=1, this metric is the same as d1. As j becomes larger, outliers have a larger
    impact on the value.

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when 
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when 
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    - Krause, P., Boyle, D., Bäse, F., 2005. Comparison of different efficiency criteria for
    hydrological model assessment. Advances in geosciences 5 89-97.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
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
    """

    Compute Watterson's M (M).

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/M.png?raw=true)

    **Range:** -1 ≤ M < 1, does not indicate bias, larger is better.

    # **Notes:**

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when 
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when 
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    - Watterson, I.G., 1996. Non‐dimensional measures of climate model performance. International
    Journal of Climatology 16(4) 379-391.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    a = 2 / np.pi
    b = mse(simulated_array, observed_array)
    c = np.std(observed_array) ** 2 + np.std(simulated_array) ** 2
    e = (np.mean(simulated_array) - np.mean(observed_array)) ** 2
    f = c + e
    return a * np.arcsin(1 - (b / f))


def mb_r(simulated_array, observed_array, replace_nan=None, replace_inf=None,
         remove_neg=False, remove_zero=False):
    """

    Compute Mielke-Berry R value (MB R).

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/MB_R.png?raw=true)

    **Range:** 0 ≤ MB R < 1, does not indicate bias, larger is better.

    **Notes:** Compares prediction to probability it arose by chance.

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when 
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when 
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

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
    simulated_array, observed_array = remove_values(
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
    return 1 - (mae(simulated_array, observed_array) * size ** 2 / total)


def nse(simulated_array, observed_array, replace_nan=None, replace_inf=None,
        remove_neg=False, remove_zero=False):
    """

    Compute the Nash-Sutcliffe Efficiency.

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/NSE.png?raw=true)

    **Range:** -inf < NSE < 1, does not indicate bias, larger is better.

    **Notes:** The Nash-Sutcliffe efficiency metric compares prediction values to naive preditions
    (i.e. average value).

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when 
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when 
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

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
    simulated_array, observed_array = remove_values(
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
    """

    Compute the modified Nash-Sutcliffe efficiency (NSE mod).

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/NSEmod.png?raw=true)

    **Range:** -inf < NSE (mod) < 1, does not indicate bias, larger is better.

    **Notes:** The modified Nash-Sutcliffe efficiency metric gives less weight to outliers if j=1,
    or more weight to outliers if j is higher. Generally, j=1.

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when 
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when 
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    - Krause, P., Boyle, D., Bäse, F., 2005. Comparison of different efficiency criteria for
    hydrological model assessment. Advances in geosciences 5 89-97.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
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

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/NSErel.png?raw=true)

    **Range:** -inf < NSE (rel) < 1, does not indicate bias, larger is better.

    **Notes:** The modified Nash-Sutcliffe efficiency metric gives less weight to outliers if j=1,
    or more weight to outliers if j is higher. Generally, j=1.

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when 
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when 
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    - Krause, P., Boyle, D., Bäse, F., 2005. Comparison of different efficiency criteria for
    hydrological model assessment. Advances in geosciences 5 89-97.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
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
    """

    Compute the Kling-Gupta efficiency (2009).

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/KGE_2009_1.png?raw=true)
    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/KGE_2009_2.png?raw=true)
    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/KGE_2009_3.png?raw=true)
    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/KGE_2009_4.png?raw=true)
    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/KGE_2009_5.png?raw=true)

    **Range:** -inf < KGE (2009) < 1, does not indicate bias, larger is better.

    **Notes:** Gupta et al. (2009) created this metric to demonstrate the relative importance of
    the three components of the NSE, which are correlation, bias and variability. This was done
    with hydrologic modeling as the context. This metric is meant to address issues with the NSE.

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    s: *tuple of length three*
    > representing the scaling factors to be used for re-scaling the Pearson product-moment
    correlation coefficient (r), Alpha, and Beta, respectively.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    - Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009). Decomposition of the mean
    squared error and NSE performance criteria: Implications for improving hydrological modelling.
    Journal of Hydrology, 377(1-2), 80-91.


    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
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

    # Relative variability between simulated and observed values
    alpha = sim_sigma / obs_sigma

    if obs_mean != 0 and obs_sigma != 0:
        kge = 1 - np.sqrt((s[0] * (pr - 1)) ** 2 + (s[1] * (alpha - 1)) ** 2 + (s[2] * (beta - 1)) ** 2)
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

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/KGE_2012_1.png?raw=true)
    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/KGE_2012_2.png?raw=true)
    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/KGE_2012_3.png?raw=true)
    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/KGE_2012_4.png?raw=true)
    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/KGE_2012_5.png?raw=true)

    **Range:** -inf < KGE (2012) < 1, does not indicate bias, larger is better.

    **Notes:** The modified version of the KGE (2009). Kling proposed this version to avoid
    cross-correlation between bias and variability ratios.

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    s: *tuple of length three*
    > representing the scaling factors to be used for re-scaling the Pearson product-moment
    correlation coefficient (r), Alpha, and Beta, respectively.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    - Kling, H., Fuchs, M., & Paulin, M. (2012). Runoff conditions in the upper Danube basin under
    an ensemble of climate change scenarios. Journal of Hydrology, 424, 264-277.

    """
    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
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

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/E1p.png?raw=true)

    **Range:** 0 ≤ E1' < 1, does not indicate bias, larger is better.

    **Notes:** The obs_bar_p argument represents a seasonal or other selected average.

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    obs_bar_p: *float*
    > Seasonal or other selected average. If None, the mean of the observed array will be used.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    - Legates, D.R., McCabe Jr, G.J., 1999. Evaluating the use of “goodness‐of‐fit” Measures in
    hydrologic and hydroclimatic model validation. Water Resources Research 35(1) 233-241.
    Lehmann, E.L., Casella, G., 1998. Springer Texts in Statistics. Springer-Verlag, New York.


    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
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
    """

    Compute the Legate-McCabe Index of Agreement.

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/D1p.png?raw=true)

    **Range:** 0 ≤ d1' < 1, does not indicate bias, larger is better.

    **Notes:** The obs_bar_p argument represents a seasonal or other selected average.

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    obs_bar_p: *float*
    > Seasonal or other selected average. If None, the mean of the observed array will be used.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    - Legates, D.R., McCabe Jr, G.J., 1999. Evaluating the use of “goodness‐of‐fit” Measures in
    hydrologic and hydroclimatic model validation. Water Resources Research 35(1) 233-241.
    Lehmann, E.L., Casella, G., 1998. Springer Texts in Statistics. Springer-Verlag, New York.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
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

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/VE.png?raw=true)

    **Range:** 0 ≤ VE < 1 smaller is better, does not indicate bias.

    **Notes:** Represents the error as a percentage of flow.

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    - Criss, R.E., Winston, W.E., 2008. Do Nash values have value? Discussion and alternate
    proposals. Hydrological Processes 22(14) 2723.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
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
    """

    Compute the Spectral Angle (SA).

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/SA.png?raw=true)

    **Range:** -π/2 ≤ SA < π/2, closer to 0 is better.

    **Notes:** The spectral angle metric measures the angle between the two vectors in hyperspace.
    It indicates how well the shape of the two series match – not magnitude.

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    - Robila, S.A., Gershman, A., 2005. Spectral matching accuracy in processing hyperspectral
    data, Signals, Circuits and Systems, 2005. ISSCS 2005. International Symposium on. IEEE,
    pp. 163-166.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
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
    """

    Compute the Spectral Correlation (SC).

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/SC.png?raw=true)

    **Range:** -π/2 ≤ SA < π/2, closer to 0 is better.

    **Notes:** The spectral correlation metric measures the angle between the two vectors in
    hyperspace. It indicates how well the shape of the two series match – not magnitude.

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    - Robila, S.A., Gershman, A., 2005. Spectral matching accuracy in processing hyperspectral
    data, Signals, Circuits and Systems, 2005. ISSCS 2005. International Symposium on. IEEE,
    pp. 163-166.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
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
    """

    Compute the Spectral Information Divergence (SID).

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/SID.png?raw=true)

    **Range:** -π/2 ≤ SID < π/2, closer to 0 is better.

    **Notes:** The spectral information divergence measures the angle between the two vectors in
    hyperspace. It indicates how well the shape of the two series match – not magnitude.

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    - Robila, S.A., Gershman, A., 2005. Spectral matching accuracy in processing hyperspectral
    data, Signals, Circuits and Systems, 2005. ISSCS 2005. International Symposium on. IEEE,
    pp. 163-166.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
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
    """

    Compute the Spectral Gradient Angle (SGA).

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/SGA1.png?raw=true)
    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/SGA2.png?raw=true)
    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/SGA3.png?raw=true)
    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/SGA4.png?raw=true)

    **Range:** -π/2 ≤ SID < π/2, closer to 0 is better.

    **Notes:** The spectral gradient angle measures the angle between the two vectors in
    hyperspace. It indicates how well the shape of the two series match – not magnitude.
    SG is the gradient of the simulated or observed time series.

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    - Robila, S.A., Gershman, A., 2005. Spectral matching accuracy in processing hyperspectral
    data, Signals, Circuits and Systems, 2005. ISSCS 2005. International Symposium on. IEEE,
    pp. 163-166.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
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
    """

    Compute the H1 mean error.

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/H1.png?raw=true)
    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/MHE.png?raw=true)

    **Range:**

    **Notes:**

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    - Tornquist, L., Vartia, P., Vartia, Y.O., 1985. How Should Relative Changes be Measured?
    The American Statistician 43-46.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    h = (simulated_array - observed_array) / observed_array
    return np.mean(h)


def h1_ahe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
           remove_zero=False):
    """

    Compute the H1 absolute error.

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/H1.png?raw=true)
    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/AHE.png?raw=true)

    **Range:**

    **Notes:**

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    - Tornquist, L., Vartia, P., Vartia, Y.O., 1985. How Should Relative Changes be Measured?
    The American Statistician 43-46.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
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
    """

    Compute the H1 root mean square error.

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/H1.png?raw=true)
    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/RMSHE.png?raw=true)

    **Range:**

    **Notes:**

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    - Tornquist, L., Vartia, P., Vartia, Y.O., 1985. How Should Relative Changes be Measured?
    The American Statistician 43-46.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    h = (simulated_array - observed_array) / observed_array
    return np.mean(np.sqrt((h ** 2)))


def h2_mhe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
           remove_zero=False):
    """

    Compute the H2 mean error.

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/H2.png?raw=true)
    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/MHE.png?raw=true)

    **Range:**

    **Notes:**

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    - Tornquist, L., Vartia, P., Vartia, Y.O., 1985. How Should Relative Changes be Measured?
    The American Statistician 43-46.


    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    h = (simulated_array - observed_array) / simulated_array
    return np.mean(h)


def h2_ahe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
           remove_zero=False):
    """

    Compute the H2 mean absolute error.

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/H2.png?raw=true)
    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/AHE.png?raw=true)

    **Range:**

    **Notes:**

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    - Tornquist, L., Vartia, P., Vartia, Y.O., 1985. How Should Relative Changes be Measured?
    The American Statistician 43-46.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
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

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/H1.png?raw=true)
    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/MHE.png?raw=true)

    **Range:**

    **Notes:**

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

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
    simulated_array, observed_array = remove_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    h = (simulated_array - observed_array) / simulated_array
    return np.mean(np.sqrt((h ** 2)))


def h3_mhe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
           remove_zero=False):
    """

    Compute the H3 mean error.

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/H3.png?raw=true)
    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/MHE.png?raw=true)

    **Range:**

    **Notes:**

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    - Tornquist, L., Vartia, P., Vartia, Y.O., 1985. How Should Relative Changes be Measured?
    The American Statistician 43-46.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    h = (simulated_array - observed_array) / (0.5 * (simulated_array + observed_array))
    return np.mean(h)


def h3_ahe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
           remove_zero=False):
    """

    Compute the H3 mean absolute error.

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/H3.png?raw=true)
    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/AHE.png?raw=true)

    **Range:**

    **Notes:**

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    - Tornquist, L., Vartia, P., Vartia, Y.O., 1985. How Should Relative Changes be Measured?
    The American Statistician 43-46.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
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

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/H3.png?raw=true)
    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/RMSHE.png?raw=true)

    **Range:**

    **Notes:**

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    - Tornquist, L., Vartia, P., Vartia, Y.O., 1985. How Should Relative Changes be Measured?
    The American Statistician 43-46.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    h = (simulated_array - observed_array) / (0.5 * (simulated_array + observed_array))
    return np.mean(np.sqrt((h ** 2)))


def h4_mhe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
           remove_zero=False):
    """

    Compute the H4 mean error.

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/H4.png?raw=true)
    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/MHE.png?raw=true)

    **Range:**

    **Notes:**

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    - Tornquist, L., Vartia, P., Vartia, Y.O., 1985. How Should Relative Changes be Measured?
    The American Statistician 43-46.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    h = (simulated_array - observed_array) / np.sqrt(simulated_array * observed_array)
    return np.mean(h)


def h4_ahe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
           remove_zero=False):
    """

    Compute the H4 mean absolute error.

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/H4.png?raw=true)
    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/AHE.png?raw=true)

    **Range:**

    **Notes:**

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    - Tornquist, L., Vartia, P., Vartia, Y.O., 1985. How Should Relative Changes be Measured?
    The American Statistician 43-46.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
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

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/H4.png?raw=true)
    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/RMSHE.png?raw=true)

    **Range:**

    **Notes:**

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    - Tornquist, L., Vartia, P., Vartia, Y.O., 1985. How Should Relative Changes be Measured?
    The American Statistician 43-46.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    h = (simulated_array - observed_array) / np.sqrt(simulated_array * observed_array)
    return np.mean(np.sqrt((h ** 2)))


def h5_mhe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
           remove_zero=False):
    """

    Compute the H5 mean error.

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/H5.png?raw=true)
    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/MHE.png?raw=true)

    **Range:**

    **Notes:**

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    - Tornquist, L., Vartia, P., Vartia, Y.O., 1985. How Should Relative Changes be Measured?
    The American Statistician 43-46.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
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


def h5_ahe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
           remove_zero=False):
    """

    Compute the H5 mean absolute error.

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/H5.png?raw=true)
    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/AHE.png?raw=true)

    **Range:**

    **Notes:**

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    - Tornquist, L., Vartia, P., Vartia, Y.O., 1985. How Should Relative Changes be Measured?
    The American Statistician 43-46.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
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

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/H5.png?raw=true)
    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/RMSHE.png?raw=true)

    **Range:**

    **Notes:**

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    - Tornquist, L., Vartia, P., Vartia, Y.O., 1985. How Should Relative Changes be Measured?
    The American Statistician 43-46.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
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
    return np.mean(np.sqrt((h ** 2)))


def h6_mhe(simulated_array, observed_array, k=1, replace_nan=None, replace_inf=None,
           remove_neg=False, remove_zero=False):
    """

    Compute the H6 mean error.

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/H6.png?raw=true)
    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/MHE.png?raw=true)

    **Range:**

    **Notes:**

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    k: *float or integer*
    > Optional parameter to set the value of k.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    - Tornquist, L., Vartia, P., Vartia, Y.O., 1985. How Should Relative Changes be Measured?
    The American Statistician 43-46.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
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


def h6_ahe(simulated_array, observed_array, k=1, replace_nan=None, replace_inf=None,
           remove_neg=False,
           remove_zero=False):
    """

    Compute the H6 mean absolute error.

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/H6.png?raw=true)
    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/AHE.png?raw=true)

    **Range:**

    **Notes:**

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    k: *float or integer*
    > Optional parameter to set the value of k.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    - Tornquist, L., Vartia, P., Vartia, Y.O., 1985. How Should Relative Changes be Measured?
    The American Statistician 43-46.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
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

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/H6.png?raw=true)
    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/RMSHE.png?raw=true)

    **Range:**

    **Notes:**

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    k: *float or integer*
    > Optional parameter to set the value of k.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    - Tornquist, L., Vartia, P., Vartia, Y.O., 1985. How Should Relative Changes be Measured?
    The American Statistician 43-46.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
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
    return np.mean(np.sqrt((h ** 2)))


def h7_mhe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
           remove_zero=False):
    """

    Compute the H7 mean error.

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/H7.png?raw=true)
    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/MHE.png?raw=true)

    **Range:**

    **Notes:**

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    - Tornquist, L., Vartia, P., Vartia, Y.O., 1985. How Should Relative Changes be Measured?
    The American Statistician 43-46.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    h = (simulated_array / observed_array - 1) / np.min(simulated_array / observed_array)
    return np.mean(h)


def h7_ahe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
           remove_zero=False):
    """

    Compute the H7 mean absolute error.

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/H7.png?raw=true)
    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/AHE.png?raw=true)

    **Range:**

    **Notes:**

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    - Tornquist, L., Vartia, P., Vartia, Y.O., 1985. How Should Relative Changes be Measured?
    The American Statistician 43-46.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
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

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/H7.png?raw=true)
    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/RMSHE.png?raw=true)

    **Range:**

    **Notes:**

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    - Tornquist, L., Vartia, P., Vartia, Y.O., 1985. How Should Relative Changes be Measured?
    The American Statistician 43-46.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    h = (simulated_array / observed_array - 1) / np.min(simulated_array / observed_array)
    return np.mean(np.sqrt((h ** 2)))


def h8_mhe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
           remove_zero=False):
    """

    Compute the H8 mean error.

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/H8.png?raw=true)
    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/MHE.png?raw=true)

    **Range:**

    **Notes:**

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    - Tornquist, L., Vartia, P., Vartia, Y.O., 1985. How Should Relative Changes be Measured?
    The American Statistician 43-46.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    h = (simulated_array / observed_array - 1) / np.max(simulated_array / observed_array)
    return np.mean(h)


def h8_ahe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
           remove_zero=False):
    """

    Compute the H8 mean absolute error.

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/H8.png?raw=true)
    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/AHE.png?raw=true)

    **Range:**

    **Notes:**

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    - Tornquist, L., Vartia, P., Vartia, Y.O., 1985. How Should Relative Changes be Measured?
    The American Statistician 43-46.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
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

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/H8.png?raw=true)
    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/RMSHE.png?raw=true)

    **Range:**

    **Notes:**

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    - Tornquist, L., Vartia, P., Vartia, Y.O., 1985. How Should Relative Changes be Measured?
    The American Statistician 43-46.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

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


def h10_mhe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
            remove_zero=False):
    """

    Compute the H10 mean error.

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/H10.png?raw=true)
    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/MHE.png?raw=true)

    **Range:**

    **Notes:**

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    - Tornquist, L., Vartia, P., Vartia, Y.O., 1985. How Should Relative Changes be Measured?
    The American Statistician 43-46.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    h = np.log1p(simulated_array) - np.log1p(observed_array)
    return np.mean(h)


def h10_ahe(simulated_array, observed_array, replace_nan=None, replace_inf=None, remove_neg=False,
            remove_zero=False):
    """

    Compute the H10 mean absolute error.

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/H10.png?raw=true)
    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/AHE.png?raw=true)

    **Range:**

    **Notes:**

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    - Tornquist, L., Vartia, P., Vartia, Y.O., 1985. How Should Relative Changes be Measured?
    The American Statistician 43-46.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
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

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/H10.png?raw=true)
    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/RMSHE.png?raw=true)

    **Range:**

    **Notes:**

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    - Tornquist, L., Vartia, P., Vartia, Y.O., 1985. How Should Relative Changes be Measured?
    The American Statistician 43-46.

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
        simulated_array,
        observed_array,
        replace_nan=replace_nan,
        replace_inf=replace_inf,
        remove_neg=remove_neg,
        remove_zero=remove_zero
    )

    h = np.log1p(simulated_array) - np.log1p(observed_array)
    return np.mean(np.sqrt((h ** 2)))


###################################################################################################
#                         Statistical Error Metrics for Distribution Testing                      #
###################################################################################################


def g_mean_diff(simulated_array, observed_array, replace_nan=None, replace_inf=None,
                remove_neg=False,
                remove_zero=False):
    """

    Compute the geometric mean difference.

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/GMD.png?raw=true)

    **Range:**

    **Notes:** For the difference of geometric means, the geometric mean is computed for each of
    two samples then their difference is taken.

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    """

    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
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

    ![](https://github.com/waderoberts123/Hydrostats/blob/master/docs/pictures/MV.png?raw=true)

    **Range:**

    **Notes:**

    **Parameters**

    simulated_array: *one dimensional ndarray*
    > An array of simulated data from the time series.

    observed_array: *one dimensional ndarray*
    > An array of observed data from the time series.

    replace_nan: *float, optional*
    > If given, indicates which value to replace NaN values with in the two arrays. If None, when
    a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    replace_inf *float, optional*
    > If given, indicates which value to replace Inf values with in the two arrays. If None, when
    an inf value is found at the i-th position in the observed OR simulated array, the i-th value
    of the observed and simulated array are removed before the computation.

    remove_neg *boolean, optional*
    > If true, negative numbers will be removed from the observed and simulated arrays.

    remove_zero *boolean, optional*
    > If true, zeros will be removed from the observed and simulated arrays.

    **References:**

    """
    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
    simulated_array, observed_array = remove_values(
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
    'H1 - MHE', 'H1 - AHE', 'H1 - RMSHE', 'H2 - MHE', 'H2 - AHE', 'H2 - RMSHE', 'H3 - MHE',
    'H3 - AHE', 'H3 - RMSHE', 'H4 - MHE', 'H4 - AHE', 'H4 - RMSHE', 'H5 - MHE', 'H5 - AHE',
    'H5 - RMSHE', 'H6 - MHE', 'H6 - AHE', 'H6 - RMSHE', 'H7 - MHE', 'H7 - AHE', 'H7 - RMSHE',
    'H8 - MHE', 'H8 - AHE', 'H8 - RMSHE', 'H10 - MHE', 'H10 - AHE', 'H10 - RMSHE',
    'Geometric Mean Difference', 'Mean Variance'
]

metric_abbr = [
    'ME', 'MAE', 'MSE', 'MLE', 'MALE', 'MSLE', 'MdE', 'MdAE', 'MdSE', 'ED', 'NED', 'RMSE', 'RMSLE',
    'NRMSE (Range)', 'NRMSE (Mean)', 'NRMSE (IQR)', 'IRMSE', 'MASE', 'r2', 'R (Pearson)',
    'R (Spearman)', 'ACC', 'MAPE', 'MAPD', 'MAAPE', 'SMAPE1', 'SMAPE2', 'd', 'd1', 'd (Mod.)',
    'd (Rel.)', 'dr', 'M', '(MB) R', 'NSE', 'NSE (Mod.)', 'NSE (Rel.)', 'KGE (2009)', 'KGE (2012)',
    "E1'", "D1'", 'VE', 'SA', 'SC', 'SID', 'SGA', 'H1 (MHE)', 'H1 (AHE)', 'H1 (RMSHE)', 'H2 (MHE)',
    'H2 (AHE)', 'H2 (RMSHE)', 'H3 (MHE)', 'H3 (AHE)', 'H3 (RMSHE)', 'H4 (MHE)', 'H4 (AHE)',
    'H4 (RMSHE)', 'H5 (MHE)', 'H5 (AHE)', 'H5 (RMSHE)', 'H6 (MHE)', 'H6 (AHE)', 'H6 (RMSHE)',
    'H7 (MHE)', 'H7 (AHE)', 'H7 (RMSHE)', 'H8 (MHE)', 'H8 (AHE)', 'H8 (RMSHE)', 'H10 (MHE)',
    'H10 (AHE)', 'H10 (RMSHE)', 'GMD', 'MV'
]

function_list = [
    me, mae, mse, mle, male, msle, mde, mdae, mdse, ed, ned, rmse, rmsle, nrmse_range, nrmse_mean,
    nrmse_iqr, irmse, mase, r_squared, pearson_r, spearman_r, acc, mape, mapd, maape, smape1,
    smape2, d, d1, dmod, drel, dr, watt_m, mb_r, nse, nse_mod, nse_rel, kge_2009, kge_2012,
    lm_index, d1_p, ve, sa, sc, sid, sga, h1_mhe, h1_ahe, h1_rmshe, h2_mhe, h2_ahe, h2_rmshe,
    h3_mhe, h3_ahe, h3_rmshe, h4_mhe, h4_ahe, h4_rmshe, h5_mhe, h5_ahe, h5_rmshe, h6_mhe, h6_ahe,
    h6_rmshe, h7_mhe, h7_ahe, h7_rmshe, h8_mhe, h8_ahe, h8_rmshe, h10_mhe, h10_ahe, h10_rmshe,
    g_mean_diff, mean_var,
]


def remove_values(simulated_array, observed_array, replace_nan=None, replace_inf=None,
                  remove_neg=False,
                  remove_zero=False):
    """Removes the nan, negative, and inf values in two numpy arrays"""
    # Filtering warnings so that user doesn't see them while we remove the nans
    warnings.filterwarnings("ignore")
    # Checking to see if the vectors are the same length
    if len(simulated_array.shape) != 1 or len(observed_array.shape) != 1:
        raise HydrostatsError("One or both of the ndarrays are not 1 dimensional.")
    if simulated_array.size != observed_array.size:
        raise HydrostatsError("The two ndarrays are not the same size.")
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
        warnings.warn(
            "One of the arrays contained negative, nan, or inf values and they have been removed.",
            Warning)
    return simulated_array, observed_array


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

            elif metric == 'H6 - MHE':
                metrics_list.append(h6_mhe(sim_array, obs_array, k=h6_mhe_k,
                                           replace_nan=replace_nan, replace_inf=replace_inf,
                                           remove_neg=remove_neg, remove_zero=remove_zero
                                           ))

            elif metric == 'H6 - AHE':
                metrics_list.append(h6_ahe(sim_array, obs_array, k=h6_ahe_k,
                                           replace_nan=replace_nan, replace_inf=replace_inf,
                                           remove_neg=remove_neg, remove_zero=remove_zero
                                           ))

            elif metric == 'H6 - RMSHE':
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
                metrics_list.append(h6_ahe(sim_array, obs_array, k=h6_ahe_k,
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
    # long_str = ''
    # for i in __all__:
    #     long_str += i + ', '
    # print(long_str)
    # x = np.arange(100) / 20
    #
    # sim = np.sin(x) + 2
    # obs = sim * (((np.random.rand(100) - 0.5) / 10) + 1)
    #
    # plt.plot(x, sim)
    # plt.plot(x, obs)
    #
    # plt.show()
    #
    # print(me(sim, obs))
    # print(irmse(sim, obs))
    # print(rmse(sim, obs))
