import numpy as np
import pytest
from numpy.typing import NDArray

import hydrostats.metrics as he


@pytest.fixture
def sim() -> NDArray[np.floating]:
    return np.array([5, 7, 9, 2, 4.5, 6.7])


@pytest.fixture
def obs() -> NDArray[np.floating]:
    return np.array([4.7, 6, 10, 2.5, 4, 6.8])


@pytest.fixture
def sim_bad_data() -> NDArray[np.floating]:
    return np.array([6, np.nan, 100, np.inf, 200, -np.inf, 300, 0, 400, -0.1, 5, 7, 9, 2, 4.5, 6.7])


@pytest.fixture
def obs_bad_data() -> NDArray[np.floating]:
    return np.array(
        [np.nan, 100, np.inf, 200, -np.inf, 300, 0, 400, -0.1, 500, 4.7, 6, 10, 2.5, 4, 6.8]
    )


def test_list_of_metrics(sim: NDArray[np.floating], obs: NDArray[np.floating]) -> None:
    expected_list = [func(sim, obs) for func in he.function_list]

    test_list_without_abbr = he.list_of_metrics(he.metric_names, sim, obs)
    test_list_with_abbr = he.list_of_metrics(he.metric_abbr, sim, obs, abbr=True)

    assert expected_list == test_list_with_abbr
    assert expected_list == test_list_without_abbr

    mase_m = 3
    d_mod_j = 3
    nse_mod_j = 3
    h6_mhe_k = 1
    h6_ahe_k = 1
    h6_rmshe_k = 1
    d1_p_obs_bar_p = 5
    lm_x_obs_bar_p = 5
    kge2009_s = (1.2, 0.8, 0.6)
    kge2012_s = (1.4, 0.7, 0.9)

    expected_list_with_params = [
        he.mase(sim, obs, m=mase_m),
        he.dmod(sim, obs, j=d_mod_j),
        he.nse_mod(sim, obs, j=nse_mod_j),
        he.lm_index(sim, obs, obs_bar_p=lm_x_obs_bar_p),
        he.h6_mhe(sim, obs, k=h6_mhe_k),
        he.h6_mahe(sim, obs, k=h6_ahe_k),
        he.h6_rmshe(sim, obs, k=h6_rmshe_k),
        he.d1_p(sim, obs, obs_bar_p=d1_p_obs_bar_p),
        he.kge_2009(sim, obs, s=kge2009_s),
        he.kge_2012(sim, obs, s=kge2012_s),
    ]

    list_of_metric_names = [
        "Mean Absolute Scaled Error",
        "Modified Index of Agreement",
        "Modified Nash-Sutcliffe Efficiency",
        "Legate-McCabe Efficiency Index",
        "Mean H6 Error",
        "Mean Absolute H6 Error",
        "Root Mean Square H6 Error",
        "Legate-McCabe Index of Agreement",
        "Kling-Gupta Efficiency (2009)",
        "Kling-Gupta Efficiency (2012)",
    ]
    list_of_metric_abbr = [
        "MASE",
        "d (Mod.)",
        "NSE (Mod.)",
        "E1'",
        "H6 (MHE)",
        "H6 (AHE)",
        "H6 (RMSHE)",
        "D1'",
        "KGE (2009)",
        "KGE (2012)",
    ]

    test_list_without_abbr_params = he.list_of_metrics(
        metrics=list_of_metric_names,
        sim_array=sim,
        obs_array=obs,
        mase_m=mase_m,
        dmod_j=d_mod_j,
        nse_mod_j=nse_mod_j,
        h6_mhe_k=h6_mhe_k,
        h6_ahe_k=h6_ahe_k,
        h6_rmshe_k=h6_rmshe_k,
        d1_p_obs_bar_p=d1_p_obs_bar_p,
        lm_x_obs_bar_p=lm_x_obs_bar_p,
        kge2009_s=kge2009_s,
        kge2012_s=kge2012_s,
    )

    test_list_with_abbr_params = he.list_of_metrics(
        metrics=list_of_metric_abbr,
        sim_array=sim,
        obs_array=obs,
        abbr=True,
        mase_m=mase_m,
        dmod_j=d_mod_j,
        nse_mod_j=nse_mod_j,
        h6_mhe_k=h6_mhe_k,
        h6_ahe_k=h6_ahe_k,
        h6_rmshe_k=h6_rmshe_k,
        d1_p_obs_bar_p=d1_p_obs_bar_p,
        lm_x_obs_bar_p=lm_x_obs_bar_p,
        kge2009_s=kge2009_s,
        kge2012_s=kge2012_s,
    )

    assert expected_list_with_params == test_list_without_abbr_params
    assert expected_list_with_params == test_list_with_abbr_params


def test_list_of_metrics_raises_for_bad_inputs(
    sim: NDArray[np.floating], obs: NDArray[np.floating]
) -> None:
    unequal_length_sim = np.array([5, 7, 9, 2, 4.5])
    with pytest.raises(ValueError, match=r"^The two ndarrays are not the same size."):
        he.list_of_metrics(he.metric_names, unequal_length_sim, obs)

    unequal_length_obs = np.array([4.7, 6, 10, 2.5, 4])
    with pytest.raises(ValueError, match=r"^The two ndarrays are not the same size."):
        he.list_of_metrics(he.metric_names, sim, unequal_length_obs)

    unequal_dim_sim = np.array([[5, 7], [9, 2], [4.5, 6.7]])
    with pytest.raises(ValueError, match=r"The simulated array is not one dimensional."):
        he.list_of_metrics(he.metric_names, unequal_dim_sim, obs)

    unequal_dim_obs = np.array([[4.7, 6], [10, 2.5], [4, 6.8]])
    with pytest.raises(ValueError, match=r"The observed array is not one dimensional."):
        he.list_of_metrics(he.metric_names, sim, unequal_dim_obs)
