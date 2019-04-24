# python 3.6
# -*- coding: utf-8 -*-
"""
The ens_metrics module contains all of the metrics included in hydrostats that measure forecast
skill. Each forecast metric is contained in a function, and every metric has the ability to treat
missing values as well as remove zero and negative values from the timeseries data. Users will be
warned which start dates have been removed in the warnings that display during the function
execution.
"""
from __future__ import division
from hydrostats.metrics import pearson_r
import numpy as np
from numba import jit, prange
import warnings

__all__ = ["ens_me", "ens_mae", "ens_mse", "ens_rmse", "ens_pearson_r", "crps_hersbach",
           "crps_kernel", "ens_crps", "ens_brier", "auroc", "skill_score"]


def ens_me(obs, fcst_ens=None, remove_zero=False, remove_neg=False):
    """Calculate the mean error between observed values and the ensemble mean.

    Parameters
    ----------

    obs: 1D ndarray
        Array of observations for each start date.

    fcst_ens: 2D ndarray
        Array of ensemble forecast of dimension n x M, where n = number of start dates and
        M = number of ensemble members.

    remove_neg: bool
        If True, when a negative value is found at the i-th position in the observed OR ensemble
        array, the i-th value of the observed AND ensemble array are removed before the
        computation.

    remove_zero: bool
        If true, when a zero value is found at the i-th position in the observed OR ensemble
        array, the i-th value of the observed AND ensemble array are removed before the
        computation.

    Returns
    -------
    float
        The mean error between the observed time series data and the ensemble mean.

    Examples
    --------
    >>> import numpy as np
    >>> import hydrostats.ens_metrics as em
    >>> np.random.seed(3849590438)

    Creating an observed 1D array and an ensemble 2D array

    >>> ensemble_array = (np.random.rand(100, 52) + 1) * 100  # 52 Ensembles
    >>> observed_array = (np.random.rand(100) + 1) * 100

    Computing the ME between the ensemble mean and the observed data. Note that because the data is
    random the errors cancel out, leaving a low ME value.

    >>> em.ens_me(obs=observed_array, fcst_ens=ensemble_array)
    -2.5217349574908074

    """
    # Treating data
    obs, fcst_ens = treat_data(obs, fcst_ens, remove_neg=remove_neg, remove_zero=remove_zero)

    fcst_ens_mean = np.mean(fcst_ens, axis=1)

    error = fcst_ens_mean - obs
    return np.mean(error)


def ens_mae(obs, fcst_ens=None, remove_zero=False, remove_neg=False):
    """Calculate the mean absolute error between observed values and the ensemble mean.

    Parameters
    ----------
    obs: 1D ndarray
        Array of observations for each start date.

    fcst_ens: 2D ndarray
        Array of ensemble forecast of dimension n x M, where n = number of start dates and
        M = number of ensemble members.

    remove_neg: bool
        If True, when a negative value is found at the i-th position in the observed OR ensemble
        array, the i-th value of the observed AND ensemble array are removed before the
        computation.

    remove_zero: bool
        If true, when a zero value is found at the i-th position in the observed OR ensemble
        array, the i-th value of the observed AND ensemble array are removed before the
        computation.

    Returns
    -------
    float
        The mean absolute error between the observed time series data and the ensemble mean.

    Examples
    --------
    >>> import numpy as np
    >>> import hydrostats.ens_metrics as em
    >>> np.random.seed(3849590438)

    Creating an observed 1D array and an ensemble 2D array

    >>> ensemble_array = (np.random.rand(100, 52) + 1) * 100  # 52 Ensembles
    >>> observed_array = (np.random.rand(100) + 1) * 100

    Computing the ME between the ensemble mean and the observed data. Note that because the data is
    random the errors cancel out, leaving a low ME value.

    >>> em.ens_mae(obs=observed_array, fcst_ens=ensemble_array)
    26.35428724003365

    """
    # Treating data
    obs, fcst_ens = treat_data(obs, fcst_ens, remove_neg=remove_neg, remove_zero=remove_zero)

    fcst_ens_mean = np.mean(fcst_ens, axis=1)

    error = fcst_ens_mean - obs
    return np.mean(np.abs(error))


def ens_mse(obs, fcst_ens=None, remove_zero=False, remove_neg=False):
    """Calculate the mean squared error between observed values and the ensemble mean.

    Parameters
    ----------
    obs: 1D ndarray
        Array of observations for each start date.

    fcst_ens: 2D ndarray
        Array of ensemble forecast of dimension n x M, where n = number of start dates and
        M = number of ensemble members.

    remove_neg: bool
        If True, when a negative value is found at the i-th position in the observed OR ensemble
        array, the i-th value of the observed AND ensemble array are removed before the
        computation.

    remove_zero: bool
        If true, when a zero value is found at the i-th position in the observed OR ensemble
        array, the i-th value of the observed AND ensemble array are removed before the
        computation.

    Returns
    -------
    float
        The mean error between the observed time series data and the ensemble mean.

    Examples
    --------
    >>> import numpy as np
    >>> import hydrostats.ens_metrics as em
    >>> np.random.seed(3849590438)

    Creating an observed 1D array and an ensemble 2D array

    >>> ensemble_array = (np.random.rand(100, 52) + 1) * 100  # 52 Ensembles
    >>> observed_array = (np.random.rand(100) + 1) * 100

    Computing the MSE between the ensemble mean and the observed data

    >>> em.ens_mse(obs=observed_array, fcst_ens=ensemble_array)
    910.5648405687582

    """
    # Treating data
    obs, fcst_ens = treat_data(obs, fcst_ens, remove_neg=remove_neg, remove_zero=remove_zero)

    fcst_ens_mean = np.mean(fcst_ens, axis=1)

    error = fcst_ens_mean - obs
    return np.mean(error ** 2)


def ens_rmse(obs, fcst_ens=None, remove_zero=False, remove_neg=False):
    """Calculate the root mean squared error between observed values and the ensemble mean.

    Parameters
    ----------
    obs: 1D ndarray
        Array of observations for each start date.

    fcst_ens: 2D ndarray
        Array of ensemble forecast of dimension n x M, where n = number of start dates and
        M = number of ensemble members.

    remove_neg: bool
        If True, when a negative value is found at the i-th position in the observed OR ensemble
        array, the i-th value of the observed AND ensemble array are removed before the
        computation.

    remove_zero: bool
        If true, when a zero value is found at the i-th position in the observed OR ensemble
        array, the i-th value of the observed AND ensemble array are removed before the
        computation.

    Returns
    -------
    float
        The mean error between the observed time series data and the ensemble mean.

    Examples
    --------
    >>> import numpy as np
    >>> import hydrostats.ens_metrics as em
    >>> np.random.seed(3849590438)

    Creating an observed 1D array and an ensemble 2D array

    >>> ensemble_array = (np.random.rand(100, 52) + 1) * 100  # 52 Ensembles
    >>> observed_array = (np.random.rand(100) + 1) * 100

    Computing the MSE between the ensemble mean and the observed data

    >>> em.ens_rmse(obs=observed_array, fcst_ens=ensemble_array)
    30.17556694693172

    """
    # Treating data
    obs, fcst_ens = treat_data(obs, fcst_ens, remove_neg=remove_neg, remove_zero=remove_zero)

    fcst_ens_mean = np.mean(fcst_ens, axis=1)

    error = fcst_ens_mean - obs
    return np.sqrt(np.mean(error ** 2))


def ens_pearson_r(obs, fcst_ens, remove_neg=False, remove_zero=False):
    """Calculate the pearson correlation coefficient between observed values and the ensemble mean.

    Parameters
    ----------
    obs: 1D ndarray
        Array of observations for each start date.

    fcst_ens: 2D ndarray
        Array of ensemble forecast of dimension n x M, where n = number of start dates and
        M = number of ensemble members.

    remove_neg: bool
        If True, when a negative value is found at the i-th position in the observed OR ensemble
        array, the i-th value of the observed AND ensemble array are removed before the
        computation.

    remove_zero: bool
        If true, when a zero value is found at the i-th position in the observed OR ensemble
        array, the i-th value of the observed AND ensemble array are removed before the
        computation.

    Returns
    -------
    float
        The pearson correlation coefficient between the observed time series data and the ensemble
        mean.

    Examples
    --------
    >>> import numpy as np
    >>> import hydrostats.ens_metrics as em
    >>> np.random.seed(3849590438)

    Creating an observed 1D array and an ensemble 2D array

    >>> ensemble_array = (np.random.rand(100, 52) + 1) * 100
    >>> observed_array = (np.random.rand(100) + 1) * 100

    Computing the MSE between the ensemble mean and the observed data

    >>> em.ens_pearson_r(obs=observed_array, fcst_ens=ensemble_array)
    -0.13236871294739733

    """
    # Treating data
    obs, fcst_ens = treat_data(obs, fcst_ens, remove_neg=remove_neg, remove_zero=remove_zero)

    fcst_ens_mean = np.mean(fcst_ens, axis=1)

    return pearson_r(fcst_ens_mean, obs)


def ens_crps(obs, fcst_ens, adj=np.nan, remove_neg=False, remove_zero=False, llvm=True):
    """Calculate the ensemble-adjusted Continuous Ranked Probability Score (CRPS)

    Parameters
    ----------
    obs: 1D ndarray
        array of observations for each start date.

    fcst_ens: 2D ndarray
        Array of ensemble forecast of dimension n x M, where n = number of start dates and
        M = number of ensemble members.

    adj: float or int
        A positive number representing ensemble size for which the scores should be
        adjusted. If np.nan (default) scores will not be adjusted. This value can be ‘np.inf‘, in
        which case the adjusted (or fair) crps values will be calculated as per equation 6 in
        Leutbecher et al. (2018).

    remove_neg: bool
        If True, when a negative value is found at the i-th position in the observed OR ensemble
        array, the i-th value of the observed AND ensemble array are removed before the
        computation.

    remove_zero: bool
        If true, when a zero value is found at the i-th position in the observed OR ensemble
        array, the i-th value of the observed AND ensemble array are removed before the
        computation.

    llvm: bool
        If true (default) the crps will be calculated using the numba module, which uses the LLVM compiler
        infrastructure for enhanced performance. If this is not wanted, then set it to false for a pure python
        implementation.

    Returns
    -------
    dict
        Dictionary contains two keys, crps and crpsMean. The value of crps is a list of the crps
        values. Note that if the ensemble forecast or the observed values contained NaN or inf
        values, or negative or zero values if specified, these start dates will not show up in the
        crps values. The crpsMean value is the arithmatic mean of the crps values.

    Examples
    --------

    >>> import numpy as np
    >>> import hydrostats.ens_metrics as em
    >>> np.random.seed(3849590438)

    Creating an observed 1D array and an ensemble 2D array with all random numbers

    >>> ens_array_random = (np.random.rand(15, 52) + 1) * 100  # 52 Ensembles
    >>> obs_array_random = (np.random.rand(15) + 1) * 100

    Creating an observed 1D array and an ensemble 2D array with noise.

    >>> noise = np.random.normal(scale=1, size=(15, 52))
    >>> x = np.linspace(1, 10, 15)
    >>> observed_array = np.sin(x) + 10
    >>> ensemble_array_noise = (np.ones((15, 52)).T * observed_array).T + noise  # 52 Ensembles

    Computing the crps values between the ensemble mean and the observed data with the
    random data. Note that the crps is relatively high because it is random.

    >>> crps_dictionary_rand = em.ens_crps(obs_array_random, ens_array_random)
    >>> print(crps_dictionary_rand['crps'])
    [ 7.73360237  9.59248626 34.46719655 30.10271075  7.451665   16.07882352
     14.59543529  8.55181637 15.4833089   8.32422363 16.55108154 19.20821296
      8.39452279 12.59949378 27.82543302]
    >>> crps_dictionary_rand['crpsMean']
    15.797334183277709

    Computing the crps values between the ensemble mean and the observed data with noise
    in the ensemble data. Note that the crps values are better because the forecast is closer to
    observed values.

    >>> crps_dictionary_noise = em.ens_crps(obs=observed_array, fcst_ens=ensemble_array_noise)
    >>> print(crps_dictionary_noise['crps'])
    [0.26921152 0.21388687 0.24927151 0.26047667 0.30234843 0.1996493
     0.2779844  0.29478927 0.275383   0.25682693 0.21485236 0.22824711
     0.2813889  0.21264652 0.18141063]
    >>> crps_dictionary_noise['crpsMean']
    0.24789156041214638

    References
    ----------
    - Gneiting, T. and Raftery, A. E. (2007) Strictly proper scoring rules,
      prediction and estimation, J. Amer. Stat. Asoc., 102, 359-378.
    - Leutbecher, M. (2018) Ensemble size: How suboptimal is less than infinity?,
      Q. J. R. Meteorol., Accepted.
    - Ferro CAT, Richardson SR, Weigel AP (2008) On the effect of ensemble size on the discrete and
      continuous ranked probability scores. Meteorological Applications. doi: 10.1002/met.45
    - Stefan Siegert (2017). SpecsVerification: Forecast Verification Routines for
      Ensemble Forecasts of Weather and Climate. R package version 0.5-2.
      https://CRAN.R-project.org/package=SpecsVerification
    """
    # Treating the Data
    obs, fcst_ens = treat_data(obs, fcst_ens, remove_neg=remove_neg, remove_zero=remove_zero)

    rows = obs.size
    cols = fcst_ens.shape[1]

    col_len_array = np.ones(rows) * cols
    sad_ens_half = np.zeros(rows)
    sad_obs = np.zeros(rows)
    crps = np.zeros(rows)

    if llvm:
        crps = numba_crps(
            fcst_ens, obs, rows, cols, col_len_array, sad_ens_half, sad_obs, crps, np.float64(adj)
        )
    else:
        crps = python_crps(
            fcst_ens, obs, rows, cols, col_len_array, sad_ens_half, sad_obs, crps, np.float64(adj)
        )

    # Calc mean crps as simple mean across crps[i]
    crps_mean = np.mean(crps)

    # Output array as a dictionary
    output = {'crps': crps, 'crpsMean': crps_mean}

    return output


@jit(nopython=True, parallel=True)
def numba_crps(ens, obs, rows, cols, col_len_array, sad_ens_half, sad_obs, crps, adj):
    for i in prange(rows):
        the_obs = obs[i]
        the_ens = ens[i, :]
        the_ens = np.sort(the_ens)
        sum_xj = 0.
        sum_jxj = 0.

        j = 0
        while j < cols:
            sad_obs[i] += np.abs(the_ens[j] - the_obs)
            sum_xj += the_ens[j]
            sum_jxj += (j + 1) * the_ens[j]
            j += 1

        sad_ens_half[i] = 2.0 * sum_jxj - (col_len_array[i] + 1) * sum_xj

    if np.isnan(adj):
        for i in range(rows):
            crps[i] = sad_obs[i] / col_len_array[i] - sad_ens_half[i] / \
                      (col_len_array[i] * col_len_array[i])
    elif adj > 1:
        for i in range(rows):
            crps[i] = sad_obs[i] / col_len_array[i] - sad_ens_half[i] / \
                      (col_len_array[i] * (col_len_array[i] - 1)) * (1 - 1 / adj)
    elif adj == 1:
        for i in range(rows):
            crps[i] = sad_obs[i] / col_len_array[i]
    else:
        for i in range(rows):
            crps[i] = np.nan

    return crps


def python_crps(ens, obs, rows, cols, col_len_array, sad_ens_half, sad_obs, crps, adj):
    for i in prange(rows):
        the_obs = obs[i]
        the_ens = ens[i, :]
        the_ens = np.sort(the_ens)
        sum_xj = 0.
        sum_jxj = 0.

        j = 0
        while j < cols:
            sad_obs[i] += np.abs(the_ens[j] - the_obs)
            sum_xj += the_ens[j]
            sum_jxj += (j + 1) * the_ens[j]
            j += 1

        sad_ens_half[i] = 2.0 * sum_jxj - (col_len_array[i] + 1) * sum_xj

    if np.isnan(adj):
        for i in range(rows):
            crps[i] = sad_obs[i] / col_len_array[i] - sad_ens_half[i] / \
                      (col_len_array[i] * col_len_array[i])
    elif adj > 1:
        for i in range(rows):
            crps[i] = sad_obs[i] / col_len_array[i] - sad_ens_half[i] / \
                      (col_len_array[i] * (col_len_array[i] - 1)) * (1 - 1 / adj)
    elif adj == 1:
        for i in range(rows):
            crps[i] = sad_obs[i] / col_len_array[i]
    else:
        for i in range(rows):
            crps[i] = np.nan

    return crps


def crps_hersbach(obs, fcst_ens, remove_neg=False, remove_zero=False):
    """Calculate the the continuous ranked probability score (CRPS) as per equation 25-27 in
    Hersbach et al. (2000).

    It is strongly recommended to use the `hydrostats.ens_metric.ens_crps()` function for the improved performance,
    instead of using this particular implementation. This is meant more for a proof of concept of the algorithm
    presented in the literature.

    Parameters
    ----------
    obs: 1D ndarry
        Array of observations for each start date
    fcst_ens: 2D ndarray
        Array of ensemble forecast of dimension n x M, where n = number of start dates and
        M = number of ensemble members.

    remove_neg: bool
        If True, when a negative value is found at the i-th position in the observed OR ensemble
        array, the i-th value of the observed AND ensemble array are removed before the
        computation.

    remove_zero: bool
        If true, when a zero value is found at the i-th position in the observed OR ensemble
        array, the i-th value of the observed AND ensemble array are removed before the
        computation.

    Returns
    -------
    dict
        Dictionary contains a number of *experimental* outputs including:
            - ["crps"] 1D ndarray of crps values per n start dates.
            - ["crpsMean1"] arithmetic mean of crps values.
            - ["crpsMean2"] mean crps using eqn. 28 in Hersbach (2000).

    Notes
    -----
    **NaN and inf treatment:** If any value in obs or fcst_ens is NaN or inf, then the
    corresponding row in both fcst_ens (for all ensemble members) and in obs will be deleted.

    References
    ----------
    - Hersbach, H. (2000) Decomposition of the Continuous Ranked Porbability Score
      for Ensemble Prediction Systems, Weather and Forecasting, 15, 559-570.

    Examples
    --------

    >>> import numpy as np
    >>> import hydrostats.ens_metrics as em
    >>> np.random.seed(3849590438)

    Creating an observed 1D array and an ensemble 2D array with all random numbers

    >>> ens_array_random = (np.random.rand(15, 52) + 1) * 100
    >>> obs_array_random = (np.random.rand(15) + 1) * 100

    Creating an observed 1D array and an ensemble 2D array with noise.

    >>> noise = np.random.normal(scale=1, size=(15, 52))
    >>> x = np.linspace(1, 10, 15)
    >>> observed_array = np.sin(x) + 10
    >>> ensemble_array_noise = (np.ones((15, 52)).T * observed_array).T + noise

    Computing the Hersbach CRPS values between the ensemble mean and the observed data with the
    random data.

    >>> crps_dictionary_rand = em.crps_hersbach(obs_array_random, ens_array_random)
    >>> print(crps_dictionary_rand['crps'])
    [ 7.73360237  9.59248626 34.46719655 30.10271075  7.451665   16.07882352
     14.59543529  8.55181637 15.4833089   8.32422363 16.55108154 19.20821296
      8.39452279 12.59949378 27.82543302]
    >>> crps_dictionary_rand['crpsMean1']
    15.797334183277723
    >>> crps_dictionary_rand['crpsMean2']
    15.797334183277725

    Computing the Hersbach CRPS values between the ensemble mean and the observed data with noise
    in the ensemble data.

    >>> crps_dictionary_noise = em.crps_hersbach(obs=observed_array, fcst_ens=ensemble_array_noise)
    >>> print(crps_dictionary_noise['crps'])
    [0.26921152 0.21388687 0.24927151 0.26047667 0.30234843 0.1996493
     0.2779844  0.29478927 0.275383   0.25682693 0.21485236 0.22824711
     0.2813889  0.21264652 0.18141063]
    >>> crps_dictionary_noise['crpsMean1']
    0.24789156041214705
    >>> crps_dictionary_noise['crpsMean2']
    0.24789156041214705

    """

    # Treating the Data
    obs, fcst_ens = treat_data(obs, fcst_ens, remove_neg=remove_neg, remove_zero=remove_zero)

    # Set parameters
    n = fcst_ens.shape[0]  # number of forecast start dates
    m = fcst_ens.shape[1]  # number of ensemble members

    # Create vector of pi's
    p = np.linspace(0, m, m + 1)
    pi = p / m

    crps = np.zeros(n)
    # crpsAdj = np.zeros(n)

    # Matrices for alpha and beta in CRPS decomposition
    a_mat = np.zeros(shape=(n, m + 1))
    b_mat = np.zeros(shape=(n, m + 1))

    # Loop fcst start times
    for i in range(n):

        # Initialise vectors for storing output
        a = np.zeros(m - 1)
        b = np.zeros(m - 1)

        # Verifying analysis (or obs)
        xa = obs[i]

        # Ensemble fcst CDF
        x = np.sort(fcst_ens[i, :])

        # Deal with 0 < i < m [So, will loop 50 times for m = 51]
        for j in range(m - 1):

            # Rule 1
            if xa > x[j + 1]:
                a[j] = x[j + 1] - x[j]
                b[j] = 0

            # Rule 2
            if x[j] < xa < x[j + 1]:
                a[j] = xa - x[j]
                b[j] = x[j + 1] - xa

            # Rule 3
            if xa < x[j]:
                a[j] = 0
                b[j] = x[j + 1] - x[j]

        # Deal with outliers for i = 0, and i = m,
        # else a & b are 0 for non-outliers
        if xa < x[0]:
            a1 = 0
            b1 = x[0] - xa
        else:
            a1 = 0
            b1 = 0

        # Upper outlier (rem m-1 is for last member m, but ptyhon is 0-based indexing)
        if xa > x[m - 1]:
            am = xa - x[m - 1]
            bm = 0
        else:
            am = 0
            bm = 0

        # Combine full a & b vectors including outlier
        a = np.insert(a, 0, a1)
        a = np.append(a, am)
        b = np.insert(b, 0, b1)
        b = np.append(b, bm)

        # Populate a_mat and b_mat
        a_mat[i, :] = a
        b_mat[i, :] = b

        # Calc crps for individual start times
        crps[i] = ((a * pi ** 2) + (b * (1 - pi) ** 2)).sum()

    # Calc mean crps as simple mean across crps[i]
    crps_mean_method1 = crps.mean()

    # Calc mean crps across all start times from eqn. 28 in Hersbach (2000)
    abar = np.mean(a_mat, 0)
    bbar = np.mean(b_mat, 0)
    crps_mean_method2 = ((abar * pi ** 2) + (bbar * (1 - pi) ** 2)).sum()

    # Output array as a dictionary
    output = {'crps': crps, 'crpsMean1': crps_mean_method1,
              'crpsMean2': crps_mean_method2}

    return output


def crps_kernel(obs, fcst_ens, remove_neg=False, remove_zero=False):
    """Compute the kernel representation of the continuous ranked probability score (CRPS).

    Calculates the kernel representation of the continuous ranked probability score (CRPS) as per
    equation 3 in Leutbecher et al. (2018) and the adjusted (or fair) crps as per equation 6 in the
    same paper. Note that it was Gneiting and Raftery (2007) who show the kernel representation as
    calculated here is equivalent to the standard definition based on the integral
    over the squared error of the cumulative distribution.

    It is strongly recommended to use the `hydrostats.ens_metric.ens_crps()` function for the improved performance,
    instead of using this particular implementation, this is meant more for a proof of concept and an algorithm
    implementation.

    Parameters
    ----------
    obs: 1D ndarray
        array of observations for each start date.

    fcst_ens: 2D ndarray
        Array of ensemble forecast of dimension n x M, where n = number of start dates and
        M = number of ensemble members.

    remove_neg: bool
        If True, when a negative value is found at the i-th position in the observed OR ensemble
        array, the i-th value of the observed AND ensemble array are removed before the
        computation.

    remove_zero: bool
        If true, when a zero value is found at the i-th position in the observed OR ensemble
        array, the i-th value of the observed AND ensemble array are removed before the
        computation.


    Returns
    -------
    dict
        Dictionary of outputs includes:
            - ["crps"] 1D ndarray with crps values of length n.
            - ["crpsAdjusted"] 1D ndarray with adjusted crps values of length n.
            - ["crpsMean"] Arithmetic mean of crps values as a float.
            - ["crpsAdjustedMean"] Arithmetic mean of adjusted crps values as a float.

    Notes
    -----
    **NaN treatment:** If any start date in obs is NaN, then the corresponding row in fcst
    (for all ensemble members) will also be deleted.

    Examples
    --------

    >>> import numpy as np
    >>> import hydrostats.ens_metrics as em
    >>> np.random.seed(3849590438)

    Creating an observed 1D array and an ensemble 2D array with all random numbers

    >>> ens_array_random = (np.random.rand(15, 52) + 1) * 100
    >>> obs_array_random = (np.random.rand(15) + 1) * 100

    Creating an observed 1D array and an ensemble 2D array with noise.

    >>> noise = np.random.normal(scale=1, size=(15, 52))
    >>> x = np.linspace(1, 10, 15)
    >>> observed_array = np.sin(x) + 10
    >>> ensemble_array_noise = (np.ones((15, 52)).T * observed_array).T + noise

    Computing the Hersbach CRPS values between the ensemble mean and the observed data with the
    random data.

    >>> crps_dictionary_rand = em.crps_kernel(obs_array_random, ens_array_random)
    >>> print(crps_dictionary_rand['crps'])
    [ 7.73360237  9.59248626 34.46719655 30.10271075  7.451665   16.07882352
     14.59543529  8.55181637 15.4833089   8.32422363 16.55108154 19.20821296
      8.39452279 12.59949378 27.82543302]
    >>> print(crps_dictionary_rand['crpsAdjusted'])
    [ 7.43000827  9.29100065 34.14067524 29.76359191  7.14776152 15.75147589
     14.25192856  8.23647876 15.19419171  8.05998301 16.26113448 18.90686679
      8.09725139 12.24021268 27.45673444]
    >>> crps_dictionary_rand['crpsMean']
    15.797334183277723
    >>> crps_dictionary_rand['crpsAdjustedMean']
    15.481953018707593

    Computing the Hersbach CRPS values between the ensemble mean and the observed data with noise
    in the ensemble data.

    >>> crps_dictionary_noise = em.crps_kernel(obs=observed_array, fcst_ens=ensemble_array_noise)
    >>> print(crps_dictionary_noise['crps'])
    [0.26921152 0.21388687 0.24927151 0.26047667 0.30234843 0.1996493
     0.2779844  0.29478927 0.275383   0.25682693 0.21485236 0.22824711
     0.2813889  0.21264652 0.18141063]
    >>> print(crps_dictionary_noise['crpsAdjusted'])
    [0.25850726 0.20482797 0.23908004 0.25032814 0.28996894 0.18961905
     0.2670867  0.2821429  0.2634554  0.24573507 0.20457832 0.21730326
     0.26951946 0.2034818  0.17198615]
    >>> crps_dictionary_noise['crpsMean']
    0.2478915604121471
    >>> crps_dictionary_noise['crpsAdjustedMean']
    0.23717469718824744

    References
    ----------
    - Gneiting, T. and Raftery, A. E. (2007) Strictly proper scoring rules,
      prediction and estimation, J. Amer. Stat. Asoc., 102, 359-378.
    - Leutbecher, M. (2018) Ensemble size: How suboptimal is less than infinity?,
      Q. J. R. Meteorol., Accepted.
    """

    # Treatment of data
    obs, fcst_ens = treat_data(obs, fcst_ens, remove_neg=remove_neg, remove_zero=remove_zero)

    # Set parameters
    n = fcst_ens.shape[0]  # number of forecast start dates
    m = fcst_ens.shape[1]  # number of ensemble members

    # Initialise vectors for storing output
    t1 = np.zeros(n)
    t2 = np.zeros(n)
    crps = np.zeros(n)
    crps_adj = np.zeros(n)

    # Loop through start dates
    for i in range(n):

        t1[i] = abs(fcst_ens[i] - obs[i]).sum()

        # Initialise a vec for storing absolute errors for each ensemble pair
        vec = np.zeros(m)

        # Loop through ensemble members
        for j in range(m):
            vec[j] = abs(fcst_ens[i, j] - np.delete(fcst_ens[i, :], j)).sum()
            # vec[j] = abs(fcst[i, j] - fcst[i, :]).sum()

        t2[i] = vec.sum()

        # First term (t1) is the MAE of the ensemble members; Second term (t2)
        # is ensemble spread in terms of absolute difference between all pairs
        # of members
        crps[i] = (1. / m * t1[i]) - (1. / (2 * (m ** 2)) * t2[i])  # kernel representation of crps
        crps_adj[i] = (1. / m * t1[i]) - (
                1. / (2 * m * (m - 1)) * t2[i])  # kernel representation of adjusted crps

    # Calculate mean crps
    crps_mean = crps.mean()
    crps_adj_mean = crps_adj.mean()

    # Output two arrays as a dictionary
    output = {'crps': crps, 'crpsAdjusted': crps_adj, 'crpsMean': crps_mean,
              'crpsAdjustedMean': crps_adj_mean}

    return output


def ens_brier(fcst_ens=None, obs=None, threshold=None, ens_threshold=None, obs_threshold=None, fcst_ens_bin=None,
              obs_bin=None, adj=None):
    """
    Calculate the ensemble-adjusted Brier Score.

    Range: 0 ≤ Brier ≤ 1, lower is better.

    Parameters
    ----------
    obs: 1D ndarray
        Array of observations for each start date.

    fcst_ens: 2D ndarray
        Array of ensemble forecast of dimension n x M, where n = number of start dates and
        M = number of ensemble members.

    threshold: float
        The threshold for an event (e.g. if the event is a 100 year flood, the streamflow value
        that a 100 year flood would have to exceed.

    ens_threshold: float
        If different threshholds for the ensemble forecast and the observed data is desired, then this parameter can be
        set along with the 'obs_threshold' parameter to set different thresholds.

    obs_threshold: float
        If different threshholds for the ensemble forecast and the observed data is desired, then this parameter can be
        set along with the 'ens_threshold' parameter to set different thresholds.

    fcst_ens_bin: 1D ndarray
        Binary array of observations for each start date. 1 for an event and 0 for a non-event.

    obs_bin: 2D ndarray
        Binary array of ensemble forecast of dimension n x M, where n = number of start dates and
        M = number of ensemble members. 1 for an event and 0 for a non-event.

    adj: float or int
        A positive number representing ensemble size for which the scores should be
        adjusted. If None (default) scores will not be adjusted. This value can be ‘np.inf‘, in
        which case the adjusted (or fair) Brier scores will be calculated.

    Returns
    -------
    1D ndarray
        Array of length with the ensemble-adjusted Brier scores. Length may not equal n if data
        has been removed due to NaN or Inf values.

    Notes
    -----
    **NaN and inf treatment:** If any value in obs or fcst_ens is NaN or inf, then the
    corresponding row in both fcst_ens (for all ensemble members) and in obs will be deleted.

    Examples
    --------
    >>> import numpy as np
    >>> import hydrostats.ens_metrics as em
    >>> np.random.seed(3849590438)  # For reproducibility

    Creating an observed 1D array and an ensemble 2D array

    >>> ensemble_array = (np.random.rand(15, 52) + 1) * 100  # 52 Ensembles
    >>> observed_array = (np.random.rand(15) + 1) * 100

    Computing the ensemble-adjusted Brier score between the ensemble mean and the observed data.

    >>> print(em.ens_brier(obs=observed_array, fcst_ens=ensemble_array, threshold=175))
    [0.08321006 0.05325444 0.53402367 0.45303254 0.02995562 0.08321006
     0.08321006 0.03698225 0.02366864 0.0625     0.04474852 0.71597633
     0.04474852 0.04474852 0.09467456]
    >>> np.mean(em.ens_brier(obs=observed_array, fcst_ens=ensemble_array, threshold=175))
    0.15919625246548325

    When we manually create binary data we get the same result

    >>> ensemble_array_bin = (ensemble_array > 175).astype(np.int)
    >>> observed_array_bin = (observed_array > 175).astype(np.int)
    >>> print(em.ens_brier(obs_bin=observed_array_bin, fcst_ens_bin=ensemble_array_bin))
    [0.08321006 0.05325444 0.53402367 0.45303254 0.02995562 0.08321006
     0.08321006 0.03698225 0.02366864 0.0625     0.04474852 0.71597633
     0.04474852 0.04474852 0.09467456]
    >>> np.mean(em.ens_brier(obs_bin=observed_array_bin, fcst_ens_bin=ensemble_array_bin))
    0.15919625246548325

    References
    ----------
    - Ferro CAT, Richardson SR, Weigel AP (2008) On the effect of ensemble size on the discrete and
      continuous ranked probability scores. Meteorological Applications. doi: 10.1002/met.45
    - Stefan Siegert (2017). SpecsVerification: Forecast Verification Routines for
      Ensemble Forecasts of Weather and Climate. R package version 0.5-2.
      https://CRAN.R-project.org/package=SpecsVerification
    """

    # User supplied the binary matrices
    if obs_bin is not None and fcst_ens_bin is not None and fcst_ens is None and obs is None \
            and threshold is None and ens_threshold is None and obs_threshold is None:

        pass

    # User supplied normal matrices with a threshold value to apply to each of them
    elif obs_bin is None and fcst_ens_bin is None and fcst_ens is not None and obs is not None \
            and threshold is not None and ens_threshold is None and obs_threshold is None:

        # Convert the observed data and forecast data to binary data
        obs_bin = (obs > threshold).astype(np.int)
        fcst_ens_bin = (fcst_ens > threshold).astype(np.int)

    # User supplied normal matrices with different thresholds for the forecast ensemble and the observed data
    elif obs_bin is None and fcst_ens_bin is None and fcst_ens is not None and obs is not None \
            and threshold is None and ens_threshold is not None and obs_threshold is not None:

        # Convert the observed data and forecast data to binary data
        obs_bin = (obs > obs_threshold).astype(np.int)
        fcst_ens_bin = (fcst_ens > ens_threshold).astype(np.int)

    else:
        raise RuntimeError(" You must either supply fcst_ens, obs, and threshold (or obs_threshold and ens_threshold "
                           "if there are different thresholds) or you must supply fcst_ens_bin and obs_bin.")

    # Treat missing data and warn users of columns being removed
    obs_bin, fcst_ens_bin = treat_data(obs_bin, fcst_ens_bin, remove_neg=False,
                                       remove_zero=False)

    # Count number of ensemble members that predict the event
    i = np.sum(fcst_ens_bin, axis=1)

    # Calculate ensemble size
    num_cols = fcst_ens_bin.shape[1]

    # No correction for ensemble size is performed if True
    if adj is None:
        adj = num_cols

    # calculate ensemble-adjusted brier scores
    br = (i / num_cols - obs_bin) ** 2 - i * (num_cols - i) / num_cols / \
         (num_cols - 1) * (1 / num_cols - 1 / adj)

    # return the vector of brier scores
    return br


def auroc(fcst_ens=None, obs=None, threshold=None, ens_threshold=None, obs_threshold=None,
          fcst_ens_bin=None, obs_bin=None):
    """
    Calculates Area Under the Relative Operating Characteristic curve (AUROC)
    for a forecast and its verifying binary observation, and estimates the variance of the AUROC

    Range: 0 ≤ AUROC ≤ 1, Higher is better.

    Parameters
    ----------
    obs: 1D ndarray
        Array of observations for each start date.

    fcst_ens: 2D ndarray
        Array of ensemble forecast of dimension n x M, where n = number of start dates and
        M = number of ensemble members.

    threshold: float
        The threshold for an event (e.g. if the event is a 100 year flood, the streamflow value
        that a 100 year flood would have to exceed.

    ens_threshold: float
        If different threshholds for the ensemble forecast and the observed data is desired, then this parameter can be
        set along with the 'obs_threshold' parameter to set different thresholds.

    obs_threshold: float
        If different threshholds for the ensemble forecast and the observed data is desired, then this parameter can be
        set along with the 'ens_threshold' parameter to set different thresholds.

    fcst_ens_bin: 1D ndarray
        Binary array of observations for each start date. 1 for an event and 0 for a non-event.

    obs_bin: 2D ndarray
        Binary array of ensemble forecast of dimension n x M, where n = number of start dates and
        M = number of ensemble members. 1 for an event and 0 for a non-event.

    Notes
    -----
    **NaN and inf treatment:** If any value in obs or fcst_ens is NaN or inf, then the
    corresponding row in both fcst_ens (for all ensemble members) and in obs will be deleted. A
    warning will be shown that informs the user of the rows that have been removed.

    Returns
    -------
    1D ndarray
        An array of two elements, the AUROC and the estimated variance, respectively.

    Examples
    --------

    >>> import numpy as np
    >>> import hydrostats.ens_metrics as em
    >>> np.random.seed(3849590438)

    Creating an observed 1D array and an ensemble 2D array with all random numbers

    >>> ens_array_random = (np.random.rand(100, 52) + 1) * 100
    >>> obs_array_random = (np.random.rand(100) + 1) * 100

    Creating an observed 1D array and an ensemble 2D array with noise.

    >>> noise = np.random.normal(scale=1, size=(100, 52))
    >>> x = np.linspace(1, 10, 100)
    >>> observed_array = np.sin(x) + 10
    >>> ensemble_array_noise = (np.ones((100, 52)).T * observed_array).T + noise

    Calculating the ROC with random values. Note that the area under the curve is close to 0.5
    because the data is random.

    >>> print(em.auroc(obs=obs_array_random, fcst_ens=ens_array_random, threshold=175))
    [0.45560516 0.06406262]

    Calculating the ROC with noise in the forecast values. Note that the ROC value is high because
    the forecast is more accurate.

    >>> print(em.auroc(obs=observed_array, fcst_ens=ensemble_array_noise, threshold=10))
    [0.99137931 0.00566026]

    References
    ----------
    - DeLong et al (1988): Comparing the Areas under Two or More Correlated Receiver Operating
      Characteristic Curves: A Nonparametric Approach. Biometrics. doi: 10.2307/2531595
    - Sun and Xu (2014): Fast Implementation of DeLong's Algorithm for Comparing the Areas Under
      Correlated Receiver Operating Characteristic Curves. IEEE Sign Proc Let 21(11).
      doi: 10.1109/LSP.2014.2337313
    - Stefan Siegert (2017). SpecsVerification: Forecast Verification Routines for Ensemble
      Forecasts of Weather and Climate. R package version 0.5-2.
      https://CRAN.R-project.org/package=SpecsVerification

    """
    # User supplied the binary matrices
    if obs_bin is not None and fcst_ens_bin is not None and fcst_ens is None and obs is None \
            and threshold is None and ens_threshold is None and obs_threshold is None:

        pass

    # User supplied normal matrices with a threshold value to apply to each of them
    elif obs_bin is None and fcst_ens_bin is None and fcst_ens is not None and obs is not None \
            and threshold is not None and ens_threshold is None and obs_threshold is None:

        # Convert the observed data and forecast data to binary data
        obs_bin = (obs > threshold).astype(np.int)
        fcst_ens_bin = (fcst_ens > threshold).astype(np.int)

    # User supplied normal matrices with different thresholds for the forecast ensemble and the observed data
    elif obs_bin is None and fcst_ens_bin is None and fcst_ens is not None and obs is not None \
            and threshold is None and ens_threshold is not None and obs_threshold is not None:

        # Convert the observed data and forecast data to binary data
        obs_bin = (obs > obs_threshold).astype(np.int)
        fcst_ens_bin = (fcst_ens > ens_threshold).astype(np.int)

    else:
        raise RuntimeError(" You must either supply fcst_ens, obs, and threshold (or obs_threshold and ens_threshold "
                           "if there are different thresholds) or you must supply fcst_ens_bin and obs_bin.")

    obs_bin, fcst_ens_bin = treat_data(obs_bin, fcst_ens_bin, remove_neg=False, remove_zero=False)

    if np.all(fcst_ens_bin == 0) or np.all(fcst_ens_bin == 1) or np.all(obs_bin == 0) or np.all(obs_bin == 1):
        raise RuntimeError("Both arrays need at least one event and one non-event, otherwise, "
                           "division by zero will occur!")

    ens_forecast_means = np.mean(fcst_ens_bin, axis=1)

    results = auroc_numba(ens_forecast_means, obs_bin)

    return results


@jit(nopython=True)
def auroc_numba(fcst, obs):
    num_start_dates = obs.size

    i_ord = fcst.argsort()

    sum_v = 0.
    sum_v2 = 0.
    sum_w = 0.
    sum_w2 = 0.
    n = 0
    m = 0
    i = 0

    x = 1
    y = 0

    while True:
        nn = mm = 0
        while x > y:
            j = i_ord[i]
            if obs[j]:
                mm += 1
            else:
                nn += 1
            if i == num_start_dates - 1:
                break
            jp1 = i_ord[i + 1]
            if fcst[j] != fcst[jp1]:
                break
            i += 1
        sum_w += nn * (m + mm / 2.0)
        sum_w2 += nn * (m + mm / 2.0) * (m + mm / 2.0)
        sum_v += mm * (n + nn / 2.0)
        sum_v2 += mm * (n + nn / 2.0) * (n + nn / 2.0)
        n += nn
        m += mm
        i += 1
        if i >= num_start_dates:
            break

    theta = sum_v / (m * n)
    v = sum_v2 / ((m - 1) * n * n) - sum_v * sum_v / (m * (m - 1) * n * n)
    w = sum_w2 / ((n - 1) * m * m) - sum_w * sum_w / (n * (n - 1) * m * m)

    sd_auc = np.sqrt(v / m + w / n)

    return np.array([theta, sd_auc])


def skill_score(scores, bench_scores, perf_score, eff_sample_size=None, remove_nan_inf=False):
    """Calculate the skill score of the given function.

    Parameters
    ----------

    scores: float or ndarray
        The verification scores, or the mean of the verification scores in an ndarray (float).

    bench_scores: float or ndarray
        The reference or benchmark verification scores, or the mean of the benchmark scores (float).

    perf_score: int or float
        The perfect score of the score, typically 1 or 0.

    eff_sample_size: float
        The effective sample size of the data to be used when estimating the sampling uncertainty. Default is None,
        which will set the eff_sample_size to the length of scores.

    remove_nan_inf: bool
        If True, removes NaN and Inf values in the scores if they exist, pairwise. If False (default), the function
        will raise an exception.

    Returns
    -------
    dict
        Dictionary containing: {"skillScore": Float, the skill score, "standardDeviation": Float, the estimated standard
        deviation of the skill score} If the scores and bench scores given were floats, the standard deviation will be
        NaN.

    References
    ----------
    - Stefan Siegert (2017). SpecsVerification: Forecast Verification Routines for Ensemble Forecasts of Weather and
      Climate. R package version 0.5-2. https://CRAN.R-project.org/package=SpecsVerification

    Examples
    --------

    """
    # Check data
    assert np.isfinite(perf_score), 'The perfect score is not finite.'
    if eff_sample_size is not None:
        assert eff_sample_size > 0 and np.isfinite(eff_sample_size), 'The effective sample size must be finite and ' \
                                                                     'greater than 0.'

    if isinstance(scores, np.ndarray) and isinstance(bench_scores, np.ndarray):
        assert scores.size == bench_scores.size, 'The scores and benchmark scores are not the same length'

        # Making a copy to avoid altering original scores
        scores_copy = np.copy(scores)
        bench_scores_copy = np.copy(bench_scores)

        # Removing NaN and Inf if requested
        if remove_nan_inf:

            all_treatment_array = np.ones(scores_copy.size, dtype=bool)

            if np.any(np.isnan(scores_copy)) or np.any(np.isnan(bench_scores_copy)):
                nan_indices_fcst = ~np.isnan(scores_copy)
                nan_indices_obs = ~np.isnan(bench_scores_copy)
                all_nan_indices = np.logical_and(nan_indices_fcst, nan_indices_obs)
                all_treatment_array = np.logical_and(all_treatment_array, all_nan_indices)

                warnings.warn(
                    "Row(s) {} contained NaN values and the row(s) have been removed for the calculation (Rows are "
                    "zero indexed).".format(np.where(~all_nan_indices)[0]),
                    UserWarning
                )

            if np.any(np.isinf(scores_copy)) or np.any(np.isinf(bench_scores_copy)):
                inf_indices_fcst = ~(np.isinf(scores_copy))
                inf_indices_obs = ~np.isinf(bench_scores_copy)
                all_inf_indices = np.logical_and(inf_indices_fcst, inf_indices_obs)
                all_treatment_array = np.logical_and(all_treatment_array, all_inf_indices)

                warnings.warn(
                    "Row(s) {} contained Inf or -Inf values and the row(s) have been removed for the calculation (Rows "
                    "are zero indexed).".format(np.where(~all_inf_indices)[0]),
                    UserWarning
                )

            scores_copy = scores_copy[all_treatment_array]
            bench_scores_copy = bench_scores_copy[all_treatment_array]

        else:  # If User didn't want to remove NaN and Inf
            if np.any(np.isnan(scores_copy)) or np.any(np.isnan(bench_scores_copy)):
                nan_indices_fcst = ~np.isnan(scores_copy)
                nan_indices_obs = ~np.isnan(bench_scores_copy)
                all_nan_indices = np.logical_and(nan_indices_fcst, nan_indices_obs)

                raise RuntimeError("Row(s) {} contained NaN values "
                                   "(Rows are zero indexed).".format(np.where(~all_nan_indices)[0]))

            if np.any(np.isinf(scores_copy)) or np.any(np.isinf(bench_scores_copy)):
                inf_indices_fcst = ~(np.isinf(scores_copy))
                inf_indices_obs = ~np.isinf(bench_scores_copy)
                all_inf_indices = np.logical_and(inf_indices_fcst, inf_indices_obs)

                raise RuntimeError("Row(s) {} contained Inf or -Inf values "
                                   "(Rows are zero indexed).".format(np.where(~all_inf_indices)[0]))

        # Handle effective sample size
        if eff_sample_size is None:
            eff_sample_size = scores.size

        # calculate mean scores, shift by score.perf
        score = np.mean(scores_copy) - perf_score
        bench_score = np.mean(bench_scores_copy) - perf_score

        if bench_score == 0.0:
            skillscore = np.nan
            skillscore_sigma = np.nan
            warnings.warn("The difference between the perfect score and benchmark score is zero, setting the skill"
                          " score value and standard deviation to NaN.")
        else:
            # calculate skill score
            skillscore = 1 - score / bench_score

            # calculate auxiliary quantities
            var_score = np.var(scores_copy, ddof=1)
            var_bench_score = np.var(bench_scores_copy, ddof=1)
            cov_score = np.cov(scores_copy, bench_scores_copy)[0, 1]

            # Calculate skill score standard deviation by error propagation
            def sqrt_na(z):
                if z < 0:
                    z = np.nan

                return np.sqrt(z)

            sqrt_na_val = sqrt_na(
                var_score / bench_score ** 2 + var_bench_score * score ** 2 / bench_score ** 4 - 2 * cov_score *
                score / bench_score ** 3
            )

            skillscore_sigma = (1 / np.sqrt(eff_sample_size)) * sqrt_na_val

            # Set skillscore_sigma to NaN if not finite
            if not np.isfinite(skillscore_sigma):
                skillscore_sigma = np.nan

    elif isinstance(scores, float) and isinstance(bench_scores, float):

        # shift mean scores by perfect score
        score = scores - perf_score
        bench_score = bench_scores - perf_score

        if bench_score == 0.0:
            skillscore = np.nan
            warnings.warn("The difference between the perfect score and benchmark score is zero, setting the skill"
                          " score value to NaN.")
        else:
            # calculate skill score
            skillscore = 1 - score / bench_score

        skillscore_sigma = np.nan

    else:
        raise RuntimeError("The scores and benchmark_scores must either both be ndarrays or both be floats.")

    return_dict = {
        'skillScore': skillscore,
        'standardDeviation': skillscore_sigma
    }

    return return_dict


def treat_data(obs, fcst_ens, remove_zero, remove_neg):
    assert obs.ndim == 1, "obs is not a 1D numpy array."
    assert fcst_ens.ndim == 2, "fcst_ens is not a 2D numpy array."
    assert obs.size == fcst_ens[:, 0].size, "obs and fcst_ens do not have the same amount " \
                                            "of start dates."

    # Give user warning, but let run, if eith obs or fcst are all zeros
    if obs.sum() == 0 or fcst_ens.sum() == 0:
        warnings.warn("All zero values in either 'obs' or 'fcst', "
                      "function might run, but check if data OK.")

    all_treatment_array = np.ones(obs.size, dtype=bool)

    # Treat missing data in obs and fcst_ens, rows in fcst_ens or obs that contain nan values
    if np.any(np.isnan(obs)) or np.any(np.isnan(fcst_ens)):
        nan_indices_fcst = ~(np.any(np.isnan(fcst_ens), axis=1))
        nan_indices_obs = ~np.isnan(obs)
        all_nan_indices = np.logical_and(nan_indices_fcst, nan_indices_obs)
        all_treatment_array = np.logical_and(all_treatment_array, all_nan_indices)

        warnings.warn("Row(s) {} contained NaN values and the row(s) have been "
                      "removed (zero indexed).".format(np.where(~all_nan_indices)[0]))

    if np.any(np.isinf(obs)) or np.any(np.isinf(fcst_ens)):
        inf_indices_fcst = ~(np.any(np.isinf(fcst_ens), axis=1))
        inf_indices_obs = ~np.isinf(obs)
        all_inf_indices = np.logical_and(inf_indices_fcst, inf_indices_obs)
        all_treatment_array = np.logical_and(all_treatment_array, all_inf_indices)

        warnings.warn("Row(s) {} contained Inf or -Inf values and the row(s) have been "
                      "removed (zero indexed).".format(np.where(~all_inf_indices)[0]))

    # Treat zero data in obs and fcst_ens, rows in fcst_ens or obs that contain zero values
    if remove_zero:
        if (obs == 0).any() or (fcst_ens == 0).any():
            zero_indices_fcst = ~(np.any(fcst_ens == 0, axis=1))
            zero_indices_obs = ~(obs == 0)
            all_zero_indices = np.logical_and(zero_indices_fcst, zero_indices_obs)
            all_treatment_array = np.logical_and(all_treatment_array, all_zero_indices)

            warnings.warn("Row(s) {} contained zero values and the row(s) have been "
                          "removed (zero indexed).".format(np.where(~all_zero_indices)[0]))

    # Treat negative data in obs and fcst_ens, rows in fcst_ens or obs that contain negative values
    # warnings.filterwarnings("ignore")  # Ignore Runtime warnings for comparison
    if remove_neg:
        with np.errstate(invalid='ignore'):
            obs_bool = obs < 0
            fcst_ens_bool = fcst_ens < 0
        if obs_bool.any() or fcst_ens_bool.any():
            neg_indices_fcst = ~(np.any(fcst_ens_bool, axis=1))
            neg_indices_obs = ~obs_bool
            all_neg_indices = np.logical_and(neg_indices_fcst, neg_indices_obs)
            all_treatment_array = np.logical_and(all_treatment_array, all_neg_indices)

            # warnings.filterwarnings("always")  # Turn warnings back on

            warnings.warn("Row(s) {} contained negative values and the row(s) have been "
                          "removed (zero indexed).".format(np.where(~all_neg_indices)[0]))
        else:
            pass  # warnings.filterwarnings("always")  # Turn warnings back on
    else:
        pass  # warnings.filterwarnings("always")  # Turn warnings back on

    obs = obs[all_treatment_array]
    fcst_ens = fcst_ens[all_treatment_array, :]

    return obs, fcst_ens


if __name__ == "__main__":
    pass
