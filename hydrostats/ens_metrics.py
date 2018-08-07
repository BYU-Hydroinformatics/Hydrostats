# python 3.6
# -*- coding: utf-8 -*-
"""

The ens_metrics module contains all of the metrics included in hydrostats that measure forecast
skill. Each forecast metric is contained in a function, and every metric has the ability to treat
missing values as well as remove zero and negative values from the timeseries data.

"""
from __future__ import division
# from hydrostats.metrics import list_of_metrics, metric_names, metric_abbr, remove_values, \
#     HydrostatsError
import numpy as np
import warnings

__all__ = ["ens_me", "crpsHersbach", "crpsKernel"]

# TODO: Should there be an error instead of a warning if the observed or forecast values are all 0?


def ens_me(obs, fcst_ens=None, remove_zero=False, remove_neg=False):
    """Calculate the mean error between observed values and the ensamble mean.

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
        The mean error between the observed time series data and the ensamble mean.

    """
    # Treating data
    obs, fcst_ens = treat_data(obs, fcst_ens, remove_neg=remove_neg, remove_zero=remove_zero)

    fcst_ens_mean = np.mean(fcst_ens, axis=1)
    print(fcst_ens_mean)

    return np.mean(fcst_ens_mean - obs)


def crpsHersbach(obs, fcst):
    """Calculate the the continuous ranked probability score (CRPS) as per equation 25-27 in
    Hersbach et al. (2000)

    Parameters
    ----------
    obs: 1D ndarry
        Array of observations for each start date
    fcst: 2D ndarray
        Array of ensemble forecast of dimension n x M, where n = number of start dates and
        M = number of ensemble members.

    Returns
    -------
    dict
        Dictionary contains a number of *experimental* outputs including:
            - ["crps"] 1D ndarray of crps values per n start dates.
            - ["crpsMean1"] arithmetic mean of crps values.
            - ["crpsMean2"] mean crps using eqn. 28 in Hersbach (2000).

    Notes
    -----
    **NaN treatment:** If any start date in obs is NaN, then the corresponding row in fcst (for all
    ensemble members) will also be deleted.

    References
    ----------
    - Hersbach, H. (2000) Decomposition of the Continuous Ranked Porbability Score
      for Ensemble Prediction Systems, Weather and Forecasting, 15, 559-570.
    """

    # Make sure obs and fcst have same number of start dates & correct dims
    assert obs.ndim == 1, "crpsHersbach() failed as obs not 1d-array!"
    assert fcst.ndim == 2, "crpsHersbach() failed as fcst not 2d-array!"
    assert len(obs) == len(fcst[:, 0]), "crpsHersbach() failed due to " \
                                        "different length start dates " \
                                        "between obs and fcst!"

    # Give user warning, but let run, if eith obs or fcst are all zeros
    if obs.sum() == 0 or fcst.sum() == 0:
        warnings.warn("All zero values in either 'obs' or 'fcst', "
                      "crpsHersbach() will run, but check if data OK!")

    # Treat missing data in obs, by deleting respective row in fcst, then in obs
    if np.isnan(obs).any():
        fcst = np.delete(fcst, np.argwhere(np.isnan(obs)), axis=0)
        obs = np.delete(obs, np.argwhere(np.isnan(obs)))

    # Set parameters
    n = fcst.shape[0]  # number of forecast start dates
    m = fcst.shape[1]  # number of ensemble members

    # Create vector of pi's
    p = np.linspace(0, m, m + 1)
    pi = p / m

    crps = np.zeros(n)
    # crpsAdj = np.zeros(n)

    # Matrices for alpha and beta in CRPS decomposition
    aMat = np.zeros(shape=(n, m + 1))
    bMat = np.zeros(shape=(n, m + 1))

    # Loop fcst start times
    for i in range(n):

        # Initialise vectors for storing output
        a = np.zeros(m - 1)
        b = np.zeros(m - 1)

        # Verifying analysis (or obs)
        xa = obs[i]

        # Ensemble fcst CDF
        x = np.sort(fcst[i, :])

        # Deal with 0 < i < m [So, will loop 50 times for m = 51]
        for j in range(m - 1):

            # Rule 1
            if xa > x[j + 1]:
                a[j] = x[j + 1] - x[j]
                b[j] = 0

            # Rule 2
            if xa > x[j] and xa < x[j + 1]:
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

        # Populate aMat and bMat
        aMat[i, :] = a
        bMat[i, :] = b

        # Calc crps for individual start times
        crps[i] = ((a * pi ** 2) + (b * (1 - pi) ** 2)).sum()

    # Calc mean crps as simple mean across crps[i]
    crpsMean_method1 = crps.mean()

    # Calc mean crps across all start times from eqn. 28 in Hersbach (2000)
    abar = np.mean(aMat, 0)
    bbar = np.mean(bMat, 0)
    crpsMean_method2 = ((abar * pi ** 2) + (bbar * (1 - pi) ** 2)).sum()

    # Output array as a dictionary
    output = {'crps': crps, 'crpsMean1': crpsMean_method1,
              'crpsMean2': crpsMean_method2}

    return output


def crpsKernel(obs, fcst):
    """Compute the kernel representation of the continuous ranked probability score (CRPS).

    Calculates the kernel representation of the continuous ranked probability score (CRPS) as per
    equation 3 in Leutbecher et al. (2018) and the adjusted (or fair) crps as per equation 6 in the
    same paper. Note that it was Gneiting and Raftery (2007) who show the kernel representation as
    calculated here is equivalent to the standard definition based on the integral
    over the squared error of the cumulative distribution.

    Parameters
    ----------
    obs: 1D ndarray
        Arrar of observations for each start date.

    fcst: 2D ndarray
        Array of ensemble forecast of dimension n x M, where n = number of start dates and
        M = number of ensemble members.

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

    # Make sure obs and fcst have same number of start dates & correct dims
    assert obs.ndim == 1, "crpsKernel() failed as obs not 1D-array!"
    assert fcst.ndim == 2, "crpsKernel() failed as fcst not 2D-array!"
    assert len(obs) == len(fcst[:, 0]), "crpsKernel() failed due to different " \
                                        "length start dates between obs and fcst!"

    # Give user warning, but let run, if eith obs or fcst are all zeros
    if obs.sum() == 0 or fcst.sum() == 0:
        warnings.warn("All zero values in either 'obs' or 'fcst', crpskernel() "
                      "will run, but check if data OK!")

    # Treat missing data in obs, by deleting respective row in fcst, then in obs
    if np.isnan(obs).any():
        fcst = np.delete(fcst, np.argwhere(np.isnan(obs)), axis=0)
        obs = np.delete(obs, np.argwhere(np.isnan(obs)))

    # Set parameters
    n = fcst.shape[0]  # number of forecast start dates
    m = fcst.shape[1]  # number of ensemble members

    # Initialise vectors for storing output
    t1 = np.zeros(n)
    t2 = np.zeros(n)
    crps = np.zeros(n)
    crps_adj = np.zeros(n)

    # Loop through start dates
    for i in range(n):

        t1[i] = abs(fcst[i] - obs[i]).sum()

        # Initialise a vec for storing absolute errors for each ensemble pair
        vec = np.zeros(m)

        # Loop through ensemble members
        for j in range(m):
            vec[j] = abs(fcst[i, j] - np.delete(fcst[i, :], j)).sum()
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


def treat_data(obs, fcst_ens, remove_zero, remove_neg):
    """Check the data to make sure it makes sense.

    Parameters
    ----------
    obs: 1D ndarray
        Arrar of observations for each start date.

    fcst: 2D ndarray
        Array of ensemble forecast of dimension n x M, where n = number of start dates and
        M = number of ensemble members.

    remove_zero: bool
        If True, zeros will be removed at the i-th value from both the observed time series data
        and the ensamble data at the i-th position.

    remove_neg: bool
        If True, negative values will be removed at the i-th value from both the observed time
        series data and the ensamble data at the i-th position.

    Returns
    -------
    tuple of ndarrays
        Returns the treated observed data and the ensemble data.
    """
    assert obs.ndim == 1, "obs is not a 1D numpy array."
    assert fcst_ens.ndim == 2, "fcst_ens is not a 2D numpy array."
    assert obs.size == fcst_ens[:, 0].size, "obs and fcst_ens do not have the same amount " \
                                            "of start dates."

    # Give user warning, but let run, if eith obs or fcst are all zeros
    if obs.sum() == 0 or fcst_ens.sum() == 0:
        warnings.warn("All zero values in either 'obs' or 'fcst', "
                      "crpsHersbach() will run, but check if data OK!")

    # Treat missing data in obs and fcst_ens, rows in fcst_ens or obs that contain nan values
    if np.isnan(obs).any() or np.isnan(fcst_ens).any():
        nan_indices_fcst = ~(np.any(np.isnan(fcst_ens), axis=1))
        nan_indices_obs = ~np.isnan(obs)
        all_nan_indices = np.logical_and(nan_indices_fcst, nan_indices_obs)
        obs = obs[all_nan_indices]
        fcst_ens = fcst_ens[all_nan_indices, :]

        warnings.warn("The observed data contained NaN values and they have been removed.")

    # Treat zero data in obs and fcst_ens, rows in fcst_ens or obs that contain zero values
    if remove_zero:
        if (obs == 0).any() or (fcst_ens == 0).any():
            zero_indices_fcst = ~(np.any(fcst_ens == 0, axis=1))
            zero_indices_obs = ~(obs == 0)
            all_zero_indices = np.logical_and(zero_indices_fcst, zero_indices_obs)
            obs = obs[all_zero_indices]
            fcst_ens = fcst_ens[all_zero_indices, :]

            warnings.warn("The observed data contained zero values and they have been removed.")

    # Treat negative data in obs and fcst_ens, rows in fcst_ens or obs that contain negative values
    if remove_neg:
        if (obs < 0).any() or (fcst_ens < 0).any():
            neg_indices_fcst = ~(np.any(fcst_ens < 0, axis=1))
            neg_indices_obs = ~(obs < 0)
            all_neg_indices = np.logical_and(neg_indices_fcst, neg_indices_obs)
            obs = obs[all_neg_indices]
            fcst_ens = fcst_ens[all_neg_indices, :]

        warnings.warn("The observed data contained negative values and they have been removed.")

    return obs, fcst_ens


if __name__ == "__main__":
    import pandas as pd

    forecast_URL = r'https://raw.githubusercontent.com/waderoberts123/Hydrostats/master' \
                   r'/Sample_data/Forecast_Skill/south_asia_historical_20170809_01-51.csv'
    observed_URL = r'https://github.com/waderoberts123/Hydrostats/raw/master/Sample_data/' \
                   r'Forecast_Skill/West_Rapti_Kusum_River_Discharge_2017-08-05_2017-08-15_' \
                   r'Hourly.csv'

    ensamble_df = pd.read_csv(forecast_URL, index_col=0)
    hydrologic_df = pd.read_csv(observed_URL, index_col=0)

    # Converting Ensamble DF index to datetime
    ensamble_df.index = pd.to_datetime(ensamble_df.index)
    time_values = ensamble_df.index

    # Cleaning up the observed_data
    hydrologic_df = hydrologic_df.dropna()
    hydrologic_df.index = pd.to_datetime(hydrologic_df.index)
    new_index = pd.date_range(hydrologic_df.index[0], hydrologic_df.index[-1], freq='1H')
    hydrologic_df = hydrologic_df.reindex(new_index)
    hydrologic_df = hydrologic_df.interpolate('pchip')
    hydrologic_df = hydrologic_df.reindex(time_values).dropna()

    # Merging the data
    merged_df = pd.DataFrame.join(hydrologic_df, ensamble_df)
    # merged_df.to_csv('merged_ensamble_df.csv')

    obs = merged_df.iloc[:, 0].values
    fcst_ens = merged_df.iloc[:, 1:].values

    print(ens_me(obs, fcst_ens))
