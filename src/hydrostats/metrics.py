"""Contains all the metrics included in the HydroErr package.

These metrics measure hydrologic skill. Each metric is contained in function, and every metric can
treat missing values as well as remove zero and negative values from the timeseries data.
"""

from collections.abc import Sequence

from HydroErr import (
    acc,
    d,
    d1,
    d1_p,
    dmod,
    dr,
    drel,
    ed,
    g_mean_diff,
    h1_mahe,
    h1_mhe,
    h1_rmshe,
    h2_mahe,
    h2_mhe,
    h2_rmshe,
    h3_mahe,
    h3_mhe,
    h3_rmshe,
    h4_mahe,
    h4_mhe,
    h4_rmshe,
    h5_mahe,
    h5_mhe,
    h5_rmshe,
    h6_mahe,
    h6_mhe,
    h6_rmshe,
    h7_mahe,
    h7_mhe,
    h7_rmshe,
    h8_mahe,
    h8_mhe,
    h8_rmshe,
    h10_mahe,
    h10_mhe,
    h10_rmshe,
    irmse,
    kge_2009,
    kge_2012,
    lm_index,
    maape,
    mae,
    male,
    mapd,
    mape,
    mase,
    mb_r,
    mdae,
    mde,
    mdse,
    me,
    mean_var,
    mle,
    mse,
    msle,
    ned,
    nrmse_iqr,
    nrmse_mean,
    nrmse_range,
    nse,
    nse_mod,
    nse_rel,
    pearson_r,
    r_squared,
    rmse,
    rmsle,
    sa,
    sc,
    sga,
    sid,
    smape1,
    smape2,
    spearman_r,
    ve,
    watt_m,
)
from HydroErr.HydroErr import function_list, metric_abbr, metric_names

from hydrostats.typing_aliases import InputArray

__all__: list[str] = [
    "acc",
    "d",
    "d1",
    "d1_p",
    "dmod",
    "dr",
    "drel",
    "ed",
    "g_mean_diff",
    "h1_mahe",
    "h1_mhe",
    "h1_rmshe",
    "h2_mahe",
    "h2_mhe",
    "h2_rmshe",
    "h3_mahe",
    "h3_mhe",
    "h3_rmshe",
    "h4_mahe",
    "h4_mhe",
    "h4_rmshe",
    "h5_mahe",
    "h5_mhe",
    "h5_rmshe",
    "h6_mahe",
    "h6_mhe",
    "h6_rmshe",
    "h7_mahe",
    "h7_mhe",
    "h7_rmshe",
    "h8_mahe",
    "h8_mhe",
    "h8_rmshe",
    "h10_mahe",
    "h10_mhe",
    "h10_rmshe",
    "irmse",
    "kge_2009",
    "kge_2012",
    "list_of_metrics",
    "lm_index",
    "maape",
    "mae",
    "male",
    "mapd",
    "mape",
    "mase",
    "mb_r",
    "mdae",
    "mde",
    "mdse",
    "me",
    "mean_var",
    "mle",
    "mse",
    "msle",
    "ned",
    "nrmse_iqr",
    "nrmse_mean",
    "nrmse_range",
    "nse",
    "nse_mod",
    "nse_rel",
    "pearson_r",
    "r_squared",
    "rmse",
    "rmsle",
    "sa",
    "sc",
    "sga",
    "sid",
    "smape1",
    "smape2",
    "spearman_r",
    "ve",
    "watt_m",
]


def list_of_metrics(
    metrics: Sequence[str],
    sim_array: InputArray,
    obs_array: InputArray,
    abbr: bool = False,
    mase_m: int = 1,
    dmod_j: float = 1,
    nse_mod_j: float = 1,
    h6_mhe_k: float = 1,
    h6_ahe_k: float = 1,
    h6_rmshe_k: float = 1,
    d1_p_obs_bar_p: float | None = None,
    lm_x_obs_bar_p: float | None = None,
    kge2009_s: tuple[float, float, float] = (1, 1, 1),
    kge2012_s: tuple[float, float, float] = (1, 1, 1),
    replace_nan: float | None = None,
    replace_inf: float | None = None,
    remove_neg: bool = False,
    remove_zero: bool = False,
) -> list[float]:
    """Compute multiple hydrologic skill metrics for paired simulated and observed series.

    Given a list of metric names or abbreviations, this function computes each metric for the
    provided simulated and observed arrays, applying optional preprocessing to handle NaN/Inf
    values and to remove zeros or negatives. It supports passing parameters for specific metrics
    (e.g., ``m`` for MASE, ``j`` for modified metrics, ``s`` tuples for KGE variants).

    Parameters
    ----------
    metrics
        A sequence of metric identifiers to compute. These must correspond to either the full
        metric names in ``HydroErr.HydroErr.metric_names`` or their abbreviations in
        ``HydroErr.HydroErr.metric_abbr`` depending on the value of the ``abbr`` parameter.

    sim_array
        1-D array of simulated values.

    obs_array
        1-D array of observed values. Must be the same size as ``sim_array``.

    abbr
        If ``True``, interpret ``metrics`` as abbreviations (e.g., ``"MASE"``, ``"KGE (2012)"``).
        If ``False`` (default), interpret ``metrics`` as full names (e.g. ``"Mean Absolute Scaled
        Error"``, ``"Kling-Gupta Efficiency (2012)"``).

    mase_m
        If given, indicates the seasonal period m for ``MASE``. If not given, the default is 1.

    dmod_j : float, optional
        Exponent parameter for ``Modified Index of Agreement`` (``dmod``). Default is ``1``. A
        higher j gives more emphasis to outliers

    nse_mod_j : float, optional
        Exponent parameter for ``Modified Nash-Sutcliffe Efficiency`` (``nse_mod``). Default is
        ``1``. A higher j gives more emphasis to outliers.

    h6_mhe_k : float, optional
        Parameter ``k`` for ``Mean H6 Error`` (``h6_mhe``). Default is ``1``.

    h6_ahe_k : float, optional
        Parameter ``k`` for ``Mean Absolute H6 Error`` (``h6_mahe``). Default is ``1``.

    h6_rmshe_k : float, optional
        Parameter ``k`` for ``Root Mean Square H6 Error`` (``h6_rmshe``). Default is ``1``.

    d1_p_obs_bar_p : float or None, optional
        Seasonal or other selected average for ``Legate-McCabe Index of Agreement`` (``d1_p``).
        If ``None`` (default), the mean of the observed array will be used.

    lm_x_obs_bar_p : float or None, optional
        Seasonal or other selected average for ``Legate-McCabe Efficiency Index`` (``lm_index``).
        If ``None`` (default), the mean of the observed array will be used.

    kge2009_s : tuple[float, float, float], optional
        Weights ``s`` tuple for ``KGE (2009)`` in order of ``(r, alpha, beta)``. Default is
        ``(1, 1, 1)``.

    kge2012_s : tuple[float, float, float], optional
        Weights ``s`` tuple for ``KGE (2012)`` in order of ``(r, gamma, beta)``. Default is
        ``(1, 1, 1)``.

    replace_nan : float or None, optional
        If given, indicates which value to replace NaN values with in the two arrays. If None, when
        a NaN value is found at the i-th position in the observed OR simulated array, the i-th value
        of the observed and simulated array are removed before the computation.

    replace_inf : float or None, optional
        If given, indicates which value to replace Inf values with in the two arrays. If None, when
        an inf value is found at the i-th position in the observed OR simulated array, the i-th
        value of the observed and simulated array are removed before the computation.

    remove_neg : bool, optional
        If True, when a negative value is found at the i-th position in the observed OR simulated
        array, the i-th value of the observed AND simulated array are removed before the
        computation.

    remove_zero : bool, optional
        If true, when a zero value is found at the i-th position in the observed OR simulated
        array, the i-th value of the observed AND simulated array are removed before the
        computation.

    Returns
    -------
    A list of metric values in the same order as provided in ``metrics``.

    Raises
    ------
    ValueError
        If either ``sim_array`` or ``obs_array`` is not one-dimensional.

    ValueError
        If ``sim_array`` and ``obs_array`` do not have the same size.

    Examples
    --------
    >>> import numpy as np
    >>> import hydrostats.metrics as he
    >>> sim = np.array([5, 7, 9, 2, 4.5, 6.7])
    >>> obs = np.array([4.7, 6, 10, 2.5, 4, 6.8])
    >>> he.list_of_metrics(["Mean Absolute Scaled Error", "Modified Index of Agreement"], sim, obs)
    [he.mase(sim, obs, m=1), he.dmod(sim, obs, j=1)]

    You can also use abbreviations by setting ``abbr=True`` and pass parameters for certain metrics:

    >>> he.list_of_metrics(
    ...     ["MASE", "KGE (2009)", "KGE (2012)"],
    ...     sim,
    ...     obs,
    ...     abbr=True,
    ...     mase_m=3,
    ...     kge2009_s=(1.2, 0.8, 0.6),
    ...     kge2012_s=(1.4, 0.7, 0.9),
    ... )
    [
        he.mase(sim, obs, m=3),
        he.kge_2009(sim, obs, s=(1.2, 0.8, 0.6)),
        he.kge_2012(sim, obs, s=(1.4, 0.7, 0.9))
    ]

    """
    metrics_list = []

    if not abbr:
        for metric in metrics:
            if metric == "Mean Absolute Scaled Error":
                metrics_list.append(
                    mase(
                        sim_array,
                        obs_array,
                        m=mase_m,
                        replace_nan=replace_nan,
                        replace_inf=replace_inf,
                        remove_neg=remove_neg,
                        remove_zero=remove_zero,
                    )
                )

            elif metric == "Modified Index of Agreement":
                metrics_list.append(
                    dmod(
                        sim_array,
                        obs_array,
                        j=dmod_j,
                        replace_nan=replace_nan,
                        replace_inf=replace_inf,
                        remove_neg=remove_neg,
                        remove_zero=remove_zero,
                    )
                )

            elif metric == "Modified Nash-Sutcliffe Efficiency":
                metrics_list.append(
                    nse_mod(
                        sim_array,
                        obs_array,
                        j=nse_mod_j,
                        replace_nan=replace_nan,
                        replace_inf=replace_inf,
                        remove_neg=remove_neg,
                        remove_zero=remove_zero,
                    )
                )

            elif metric == "Legate-McCabe Efficiency Index":
                metrics_list.append(
                    lm_index(
                        sim_array,
                        obs_array,
                        obs_bar_p=lm_x_obs_bar_p,
                        replace_nan=replace_nan,
                        replace_inf=replace_inf,
                        remove_neg=remove_neg,
                        remove_zero=remove_zero,
                    )
                )

            elif metric == "Mean H6 Error":
                metrics_list.append(
                    h6_mhe(
                        sim_array,
                        obs_array,
                        k=h6_mhe_k,
                        replace_nan=replace_nan,
                        replace_inf=replace_inf,
                        remove_neg=remove_neg,
                        remove_zero=remove_zero,
                    )
                )

            elif metric == "Mean Absolute H6 Error":
                metrics_list.append(
                    h6_mahe(
                        sim_array,
                        obs_array,
                        k=h6_ahe_k,
                        replace_nan=replace_nan,
                        replace_inf=replace_inf,
                        remove_neg=remove_neg,
                        remove_zero=remove_zero,
                    )
                )

            elif metric == "Root Mean Square H6 Error":
                metrics_list.append(
                    h6_rmshe(
                        sim_array,
                        obs_array,
                        k=h6_rmshe_k,
                        replace_nan=replace_nan,
                        replace_inf=replace_inf,
                        remove_neg=remove_neg,
                        remove_zero=remove_zero,
                    )
                )

            elif metric == "Legate-McCabe Index of Agreement":
                metrics_list.append(
                    d1_p(
                        sim_array,
                        obs_array,
                        obs_bar_p=d1_p_obs_bar_p,
                        replace_nan=replace_nan,
                        replace_inf=replace_inf,
                        remove_neg=remove_neg,
                        remove_zero=remove_zero,
                    )
                )
            elif metric == "Kling-Gupta Efficiency (2009)":
                metrics_list.append(
                    kge_2009(
                        sim_array,
                        obs_array,
                        s=kge2009_s,
                        replace_nan=replace_nan,
                        replace_inf=replace_inf,
                        remove_neg=remove_neg,
                        remove_zero=remove_zero,
                    )
                )
            elif metric == "Kling-Gupta Efficiency (2012)":
                metrics_list.append(
                    kge_2012(
                        sim_array,
                        obs_array,
                        s=kge2012_s,
                        replace_nan=replace_nan,
                        replace_inf=replace_inf,
                        remove_neg=remove_neg,
                        remove_zero=remove_zero,
                    )
                )
            else:
                index = metric_names.index(metric)
                metric_func = function_list[index]
                metrics_list.append(
                    metric_func(
                        sim_array,
                        obs_array,
                        replace_nan=replace_nan,
                        replace_inf=replace_inf,
                        remove_neg=remove_neg,
                        remove_zero=remove_zero,
                    )
                )

    else:
        for metric in metrics:
            if metric == "MASE":
                metrics_list.append(
                    mase(
                        sim_array,
                        obs_array,
                        m=mase_m,
                        replace_nan=replace_nan,
                        replace_inf=replace_inf,
                        remove_neg=remove_neg,
                        remove_zero=remove_zero,
                    )
                )

            elif metric == "d (Mod.)":
                metrics_list.append(
                    dmod(
                        sim_array,
                        obs_array,
                        j=dmod_j,
                        replace_nan=replace_nan,
                        replace_inf=replace_inf,
                        remove_neg=remove_neg,
                        remove_zero=remove_zero,
                    )
                )

            elif metric == "NSE (Mod.)":
                metrics_list.append(
                    nse_mod(
                        sim_array,
                        obs_array,
                        j=nse_mod_j,
                        replace_nan=replace_nan,
                        replace_inf=replace_inf,
                        remove_neg=remove_neg,
                        remove_zero=remove_zero,
                    )
                )

            elif metric == "E1'":
                metrics_list.append(
                    lm_index(
                        sim_array,
                        obs_array,
                        obs_bar_p=lm_x_obs_bar_p,
                        replace_nan=replace_nan,
                        replace_inf=replace_inf,
                        remove_neg=remove_neg,
                        remove_zero=remove_zero,
                    )
                )

            elif metric == "H6 (MHE)":
                metrics_list.append(
                    h6_mhe(
                        sim_array,
                        obs_array,
                        k=h6_mhe_k,
                        replace_nan=replace_nan,
                        replace_inf=replace_inf,
                        remove_neg=remove_neg,
                        remove_zero=remove_zero,
                    )
                )

            elif metric == "H6 (AHE)":
                metrics_list.append(
                    h6_mahe(
                        sim_array,
                        obs_array,
                        k=h6_ahe_k,
                        replace_nan=replace_nan,
                        replace_inf=replace_inf,
                        remove_neg=remove_neg,
                        remove_zero=remove_zero,
                    )
                )

            elif metric == "H6 (RMSHE)":
                metrics_list.append(
                    h6_rmshe(
                        sim_array,
                        obs_array,
                        k=h6_rmshe_k,
                        replace_nan=replace_nan,
                        replace_inf=replace_inf,
                        remove_neg=remove_neg,
                        remove_zero=remove_zero,
                    )
                )

            elif metric == "D1'":
                metrics_list.append(
                    d1_p(
                        sim_array,
                        obs_array,
                        obs_bar_p=d1_p_obs_bar_p,
                        replace_nan=replace_nan,
                        replace_inf=replace_inf,
                        remove_neg=remove_neg,
                        remove_zero=remove_zero,
                    )
                )
            elif metric == "KGE (2009)":
                metrics_list.append(
                    kge_2009(
                        sim_array,
                        obs_array,
                        s=kge2009_s,
                        replace_nan=replace_nan,
                        replace_inf=replace_inf,
                        remove_neg=remove_neg,
                        remove_zero=remove_zero,
                    )
                )
            elif metric == "KGE (2012)":
                metrics_list.append(
                    kge_2012(
                        sim_array,
                        obs_array,
                        s=kge2012_s,
                        replace_nan=replace_nan,
                        replace_inf=replace_inf,
                        remove_neg=remove_neg,
                        remove_zero=remove_zero,
                    )
                )
            else:
                index = metric_abbr.index(metric)
                metric_func = function_list[index]
                metrics_list.append(
                    metric_func(
                        sim_array,
                        obs_array,
                        replace_nan=replace_nan,
                        replace_inf=replace_inf,
                        remove_neg=remove_neg,
                        remove_zero=remove_zero,
                    )
                )
    return metrics_list
