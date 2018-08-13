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
from hydrostats.metrics import pearson_r, HydrostatsError
import numpy as np
from numba import jit, prange
import warnings

__all__ = ["ens_me", "ens_mae", "ens_mse", "ens_rmse", "ens_pearson_r", "crps_hersbach",
           "crps_kernel", "ens_crps", "ens_brier", "auroc"]

# TODO: Should there be an error instead of a warning if the observed or forecast values are all 0?
# TODO: Should we allow users to select if they want to calculate two means for efficiency?


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
    >>> import hydrostats as hs
    >>> np.random.seed(3849590438)

    Creating an observed 1D array and an ensemble 2D array

    >>> ensemble_array = (np.random.rand(100, 52) + 1) * 100
    >>> observed_array = (np.random.rand(100) + 1) * 100

    Computing the ME between the ensemble mean and the observed data. Note that because the data is
    random the errors cancel out, leaving a low ME value.

    >>> hs.ens_me(obs=observed_array, fcst_ens=ensemble_array)
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
    >>> import hydrostats as hs
    >>> np.random.seed(3849590438)

    Creating an observed 1D array and an ensemble 2D array

    >>> ensemble_array = (np.random.rand(100, 52) + 1) * 100
    >>> observed_array = (np.random.rand(100) + 1) * 100

    Computing the ME between the ensemble mean and the observed data. Note that because the data is
    random the errors cancel out, leaving a low ME value.

    >>> hs.ens_mae(obs=observed_array, fcst_ens=ensemble_array)
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
    >>> import hydrostats as hs
    >>> np.random.seed(3849590438)

    Creating an observed 1D array and an ensemble 2D array

    >>> ensemble_array = (np.random.rand(100, 52) + 1) * 100
    >>> observed_array = (np.random.rand(100) + 1) * 100

    Computing the MSE between the ensemble mean and the observed data

    >>> hs.ens_mse(obs=observed_array, fcst_ens=ensemble_array)
    910.5648405687582

    """
    # Treating data
    obs, fcst_ens = treat_data(obs, fcst_ens, remove_neg=remove_neg, remove_zero=remove_zero)

    fcst_ens_mean = np.mean(fcst_ens, axis=1)

    error = fcst_ens_mean - obs
    return np.mean(error**2)


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
    >>> import hydrostats as hs
    >>> np.random.seed(3849590438)

    Creating an observed 1D array and an ensemble 2D array

    >>> ensemble_array = (np.random.rand(100, 52) + 1) * 100
    >>> observed_array = (np.random.rand(100) + 1) * 100

    Computing the MSE between the ensemble mean and the observed data

    >>> hs.ens_rmse(obs=observed_array, fcst_ens=ensemble_array)
    30.17556694693172

    """
    # Treating data
    obs, fcst_ens = treat_data(obs, fcst_ens, remove_neg=remove_neg, remove_zero=remove_zero)

    fcst_ens_mean = np.mean(fcst_ens, axis=1)

    error = fcst_ens_mean - obs
    return np.sqrt(np.mean(error**2))


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
    >>> import hydrostats as hs
    >>> np.random.seed(3849590438)

    Creating an observed 1D array and an ensemble 2D array

    >>> ensemble_array = (np.random.rand(100, 52) + 1) * 100
    >>> observed_array = (np.random.rand(100) + 1) * 100

    Computing the MSE between the ensemble mean and the observed data

    >>> hs.ens_pearson_r(obs=observed_array, fcst_ens=ensemble_array)
    -0.13236871294739733

    """
    # Treating data
    obs, fcst_ens = treat_data(obs, fcst_ens, remove_neg=remove_neg, remove_zero=remove_zero)

    fcst_ens_mean = np.mean(fcst_ens, axis=1)

    return pearson_r(fcst_ens_mean, obs)


def ens_crps(obs, fcst_ens, adj=np.nan, remove_neg=False, remove_zero=False):
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

    crps = numba_crps(
        fcst_ens, obs, rows, cols, col_len_array, sad_ens_half, sad_obs, crps, np.float64(adj)
    )

    # Calc mean crps as simple mean across crps[i]
    crps_mean = np.mean(crps)

    # Output array as a dictionary
    output = {'crps': crps, 'crpsMean': crps_mean}

    return output


@jit("f8[:](f8[:,:], f8[:], i4, i4, f8[:], f8[:], f8[:], f8[:], f8)",
        nopython=True, parallel=True)
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


def crps_hersbach(obs, fcst_ens, remove_neg=False, remove_zero=False):
    """Calculate the the continuous ranked probability score (CRPS) as per equation 25-27 in
    Hersbach et al. (2000)

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
    >>> import hydrostats as hs
    >>> np.random.seed(3849590438)

    Creating an observed 1D array and an ensemble 2D array with all random numbers

    >>> ens_array_random = (np.random.rand(100, 52) + 1) * 100
    >>> obs_array_random = (np.random.rand(100) + 1) * 100

    Creating an observed 1D array and an ensemble 2D array with noise.

    >>> noise = np.random.normal(scale=1, size=(100, 52))
    >>> x = np.linspace(1, 10, 100)
    >>> observed_array = np.sin(x) + 10
    >>> ensemble_array_noise = (np.ones((100, 52)).T * observed_array).T + noise

    Computing the Hersbach CRPS values between the ensemble mean and the observed data with the
    random data.

    >>> crps_dictionary_rand = hs.crps_hersbach(obs_array_random, ens_array_random)
    >>> crps_dictionary_rand['crps']
    array([30.02122432, 17.51513486, 10.88716977, ... 18.69424376, 12.6309656 ,  8.55439875])
    >>> crps_dictionary_rand['crpsMean1']
    17.7355079815025
    >>> crps_dictionary_rand['crpsMean2']
    17.735507981502497

    Computing the Hersbach CRPS values between the ensemble mean and the observed data with noise
    in the ensemble data.

    >>> crps_dictionary_noise = hs.crps_hersbach(obs=observed_array, fcst_ens=ensemble_array_noise)
    >>> crps_dictionary_noise['crps']
    array([0.22264673, 0.25639179, 0.29489375, ... 0.20510253, 0.28350378, 0.22158528])
    >>> crps_dictionary_noise['crpsMean1']
    0.24473649776272008
    >>> crps_dictionary_noise['crpsMean2']
    0.2447364977627201

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
            - ["crps_mean"] Arithmetic mean of crps values as a float.
            - ["crpsAdjustedMean"] Arithmetic mean of adjusted crps values as a float.

    Notes
    -----
    **NaN treatment:** If any start date in obs is NaN, then the corresponding row in fcst
    (for all ensemble members) will also be deleted.

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

    print(np.mean(t1))

    # Calculate mean crps
    crps_mean = crps.mean()
    crps_adj_mean = crps_adj.mean()

    # Output two arrays as a dictionary
    output = {'crps': crps, 'crpsAdjusted': crps_adj, 'crpsMean': crps_mean,
              'crpsAdjustedMean': crps_adj_mean}

    return output


def ens_brier(fcst_ens, obs, adj=None):
    """Calculate the ensemble-adjusted Brier Score.

    Parameters
    ----------
    obs: 1D ndarray
        Array of observations for each start date.

    fcst_ens: 2D ndarray
        Array of ensemble forecast of dimension n x M, where n = number of start dates and
        M = number of ensemble members.

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


    References
    ----------
    - Ferro CAT, Richardson SR, Weigel AP (2008) On the effect of ensemble size on the discrete and
      continuous ranked probability scores. Meteorological Applications. doi: 10.1002/met.45
    - Stefan Siegert (2017). SpecsVerification: Forecast Verification Routines for
      Ensemble Forecasts of Weather and Climate. R package version 0.5-2.
      https://CRAN.R-project.org/package=SpecsVerification
    """
    # Treat missing data and warn users of columns being removed
    obs, fcst_ens = treat_data(obs, fcst_ens, remove_neg=False, remove_zero=False)

    # Count number of ensemble members that predict the event
    i = np.sum(fcst_ens, axis=1)

    # Calculate ensemble size
    num_cols = fcst_ens.shape[1]

    # No correction for ensemble size is performed if True
    if adj is None:
        adj = num_cols

    # calculate ensemble-adjusted brier scores
    br = (i / num_cols - obs) ** 2 - i * (num_cols - i) / num_cols / \
         (num_cols - 1) * (1 / num_cols - 1 / adj)

    # return the vector of brier scores
    return br


def auroc(fcst_ens, obs, replace_nan=False, replace_inf=False):
    """Calculates Area Under the Relative Operating characteristic Curve (AUROC) for a forecast and
    its verifying binary observation, and estimates the variance of the AUROC

    Parameters
    ----------
    fcst_ens: 2D ndarray
        Binary ensemble forecast must be given (0 for non-occurrence, 1 for occurrence of the
        event).

    obs: 1D ndarray
        Array of binary observations (0 for non-occurrence, 1 for occurrence of the event).

    Notes
    -----
    **NaN and inf treatment:** If any value in obs or fcst_ens is NaN or inf, then the
    corresponding row in both fcst_ens (for all ensemble members) and in obs will be deleted. A
    warning will be shown that informs the user of the rows that have been removed.

    Returns
    -------
    1D ndarray
        An array of two elements, the AUROC and the estimated variance, respectively.
    """
    obs, fcst = treat_data(obs, fcst_ens, remove_neg=False, remove_zero=False)

    if np.all(fcst == 0) or np.all(fcst == 1) or np.all(obs == 0) or np.all(obs == 1):
        raise HydrostatsError("Both arrays need at least one event and one non-event, otherwise, "
                              "division by zero will occur!")

    ens_forecast_means = np.mean(fcst_ens, axis=1)

    results = auroc_numba(ens_forecast_means, obs)

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


def treat_data(obs, fcst_ens, remove_zero, remove_neg):
    assert obs.ndim == 1, "obs is not a 1D numpy array."
    assert fcst_ens.ndim == 2, "fcst_ens is not a 2D numpy array."
    assert obs.size == fcst_ens[:, 0].size, "obs and fcst_ens do not have the same amount " \
                                            "of start dates."

    # Give user warning, but let run, if eith obs or fcst are all zeros
    if obs.sum() == 0 or fcst_ens.sum() == 0:
        warnings.warn("All zero values in either 'obs' or 'fcst', "
                      "function might run, but check if data OK!")

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
    if remove_neg:
        if (obs < 0).any() or (fcst_ens < 0).any():
            neg_indices_fcst = ~(np.any(fcst_ens < 0, axis=1))
            neg_indices_obs = ~(obs < 0)
            all_neg_indices = np.logical_and(neg_indices_fcst, neg_indices_obs)
            all_treatment_array = np.logical_and(all_treatment_array, all_neg_indices)

            warnings.warn("Row(s) {} contained negative values and the row(s) have been "
                          "removed (zero indexed).".format(np.where(~all_neg_indices)[0]))
    obs = obs[all_treatment_array]
    fcst_ens = fcst_ens[all_treatment_array, :]

    return obs, fcst_ens


if __name__ == "__main__":
    pass
    # obs = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0,
    #                 1], dtype=np.int8)
    # ens = np.genfromtxt(r"C:\Users\wadear\Desktop\binary_fcst.csv",
    #                     delimiter=",", dtype=np.int8)
    # ens = ens[:, 1:]
    #
    # print(auroc(ens, obs))

    # auroc(fcst_rand, obs_rand)

    # forecast_URL = r'https://raw.githubusercontent.com/waderoberts123/Hydrostats/master' \
    #                r'/Sample_data/Forecast_Skill/south_asia_historical_20170809_01-51.csv'
    # observed_URL = r'https://github.com/waderoberts123/Hydrostats/raw/master/Sample_data/' \
    #                r'Forecast_Skill/West_Rapti_Kusum_River_Discharge_2017-08-05_2017-08-15_' \
    #                r'Hourly.csv'
    #
    # ensemble_df = pd.read_csv(forecast_URL, index_col=0)
    # hydrologic_df = pd.read_csv(observed_URL, index_col=0)
    #
    # # Converting ensemble DF index to datetime
    # ensemble_df.index = pd.to_datetime(ensemble_df.index)
    # time_values = ensemble_df.index
    #
    # # Cleaning up the observed_data
    # hydrologic_df = hydrologic_df.dropna()
    # hydrologic_df.index = pd.to_datetime(hydrologic_df.index)
    # new_index = pd.date_range(hydrologic_df.index[0], hydrologic_df.index[-1], freq='1H')
    # hydrologic_df = hydrologic_df.reindex(new_index)
    # hydrologic_df = hydrologic_df.interpolate('pchip')
    # hydrologic_df = hydrologic_df.reindex(time_values).dropna()
    #
    # # Merging the data
    # merged_df = pd.DataFrame.join(hydrologic_df, ensemble_df)
    # # merged_df.to_csv('merged_ensemble_df.csv')
    #
    # obs_array = merged_df.iloc[:, 0].values
    # fcst_ens_matrix = merged_df.iloc[:, 1:].values
    #
    # np.random.seed(3849590438)
    #
    # ens_array_random = (np.random.rand(10000, 52) + 1) * 1000
    # obs_array_random = (np.random.rand(10000) + 1) * 1000
    #
    # print('ME')
    # print(ens_me(obs=obs_array_random, fcst_ens=ens_array_random))
    # print('MAE')
    # print(ens_mae(obs=obs_array_random, fcst_ens=ens_array_random))
    # print('RMSE')
    # print(ens_rmse(obs=obs_array_random, fcst_ens=ens_array_random))
    # print("Corr")
    # print(ens_pearson_r(obs=obs_array_random, fcst_ens=ens_array_random))
    # print('CRPS')
    # print(ens_crps(obs_array_random, ens_array_random, adj=np.inf))
    # print(crps_kernel(obs_array_random, ens_array_random))
    # noise = np.random.normal(scale=1, size=(100, 51))
    # x = np.linspace(1, 10, 100)
    # obs = np.sin(x) + 10
    # sim = (np.ones((100, 51)).T * obs).T + noise

    # print(obs)
    # print(np.mean(sim, axis=1))
    #
    # plt.plot(x, obs)
    # plt.plot(x, np.mean(sim, axis=1))
    # plt.show()
