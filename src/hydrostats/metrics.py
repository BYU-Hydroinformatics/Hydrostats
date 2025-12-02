"""
The metrics module contains all of the metrics included in the HydroErr package. These metrics measure hydrologic skill.
Each metric is contained in function, and every metric has the ability to treat missing values as
well as remove zero and negative values from the timeseries data.
"""

from typing import Iterable, List, Sequence, Tuple, Union

import numpy as np
from HydroErr import *
from HydroErr.HydroErr import metric_names, metric_abbr, function_list


def list_of_metrics(
    metrics: Sequence[str],
    sim_array: np.ndarray,
    obs_array: np.ndarray,
    abbr: bool = False,
    mase_m: int = 1,
    dmod_j: Union[int, float] = 1,
    nse_mod_j: Union[int, float] = 1,
    h6_mhe_k: Union[int, float] = 1,
    h6_ahe_k: Union[int, float] = 1,
    h6_rmshe_k: Union[int, float] = 1,
    d1_p_obs_bar_p: Union[int, float, None] = None,
    lm_x_obs_bar_p: Union[int, float, None] = None,
    kge2009_s: Tuple[float, float, float] = (1, 1, 1),
    kge2012_s: Tuple[float, float, float] = (1, 1, 1),
    replace_nan: Union[float, None] = None,
    replace_inf: Union[float, None] = None,
    remove_neg: bool = False,
    remove_zero: bool = False,
) -> List[float]:
    if sim_array.ndim != 1 or obs_array.ndim != 1:
        raise RuntimeError("One or both of the ndarrays are not 1 dimensional.")
    if sim_array.size != obs_array.size:
        raise RuntimeError("The two ndarrays are not the same size.")

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
