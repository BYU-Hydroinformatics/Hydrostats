from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

import hydrostats.ens_metrics as em


@pytest.fixture(scope="module")
def ensemble_array(files_for_tests: Path) -> NDArray[np.float64]:
    return np.load(files_for_tests / "ensemble_array.npz")["arr_0.npy"]


@pytest.fixture(scope="module")
def observed_array(files_for_tests: Path) -> NDArray[np.float64]:
    return np.load(files_for_tests / "observed_array.npz")["arr_0.npy"]


@pytest.fixture(scope="module")
def ensemble_array_bad(files_for_tests: Path) -> NDArray[np.float64]:
    return np.load(files_for_tests / "ensemble_array_bad_data.npz")["arr_0.npy"]


@pytest.fixture(scope="module")
def observed_array_bad(files_for_tests: Path) -> NDArray[np.float64]:
    return np.load(files_for_tests / "observed_array_bad_data.npz")["arr_0.npy"]


@pytest.fixture(scope="module")
def ens_bin(files_for_tests: Path) -> NDArray[np.int64]:
    return np.load(files_for_tests / "ens_bin.npy")


@pytest.fixture(scope="module")
def obs_bin(files_for_tests: Path) -> NDArray[np.int64]:
    return np.load(files_for_tests / "obs_bin.npy")


def test_ens_me(
    ensemble_array: NDArray[np.float64],
    observed_array: NDArray[np.float64],
    ensemble_array_bad: NDArray[np.float64],
    observed_array_bad: NDArray[np.float64],
) -> None:
    expected_value = -2.5217349574908074
    test_value = em.ens_me(obs=observed_array, fcst_ens=ensemble_array)
    assert np.isclose(expected_value, test_value)

    expected_value_bad_data = em.ens_me(obs=observed_array[8:], fcst_ens=ensemble_array[8:, :])
    test_value_bad_data = em.ens_me(
        obs=observed_array_bad,
        fcst_ens=ensemble_array_bad,
        remove_zero=True,
        remove_neg=True,
    )
    assert np.isclose(expected_value_bad_data, test_value_bad_data)


def test_ens_mae(
    ensemble_array: NDArray[np.float64],
    observed_array: NDArray[np.float64],
    ensemble_array_bad: NDArray[np.float64],
    observed_array_bad: NDArray[np.float64],
) -> None:
    expected_value = 26.35428724003365
    test_value = em.ens_mae(obs=observed_array, fcst_ens=ensemble_array)
    assert np.isclose(expected_value, test_value)

    expected_value_bad_data = em.ens_mae(obs=observed_array[8:], fcst_ens=ensemble_array[8:, :])
    test_value_bad_data = em.ens_mae(
        obs=observed_array_bad,
        fcst_ens=ensemble_array_bad,
        remove_zero=True,
        remove_neg=True,
    )
    assert np.isclose(expected_value_bad_data, test_value_bad_data)


def test_ens_mse(
    ensemble_array: NDArray[np.float64],
    observed_array: NDArray[np.float64],
    ensemble_array_bad: NDArray[np.float64],
    observed_array_bad: NDArray[np.float64],
) -> None:
    expected_value = 910.5648405687582
    test_value = em.ens_mse(obs=observed_array, fcst_ens=ensemble_array)
    assert np.isclose(expected_value, test_value)

    expected_value_bad_data = em.ens_mse(obs=observed_array[8:], fcst_ens=ensemble_array[8:, :])
    test_value_bad_data = em.ens_mse(
        obs=observed_array_bad,
        fcst_ens=ensemble_array_bad,
        remove_zero=True,
        remove_neg=True,
    )
    assert np.isclose(expected_value_bad_data, test_value_bad_data)


def test_ens_rmse(
    ensemble_array: NDArray[np.float64],
    observed_array: NDArray[np.float64],
    ensemble_array_bad: NDArray[np.float64],
    observed_array_bad: NDArray[np.float64],
) -> None:
    expected_value = 30.17556694693172
    test_value = em.ens_rmse(obs=observed_array, fcst_ens=ensemble_array)
    assert np.isclose(expected_value, test_value)

    expected_value_bad_data = em.ens_rmse(obs=observed_array[8:], fcst_ens=ensemble_array[8:, :])
    test_value_bad_data = em.ens_rmse(
        obs=observed_array_bad,
        fcst_ens=ensemble_array_bad,
        remove_zero=True,
        remove_neg=True,
    )
    assert np.isclose(expected_value_bad_data, test_value_bad_data)


def test_ens_crps(
    files_for_tests: Path, ensemble_array: NDArray[np.float64], observed_array: NDArray[np.float64]
) -> None:
    expected_crps = np.load(files_for_tests / "expected_crps.npy")
    expected_mean_crps = 17.735507981502494

    crps_numba = em.ens_crps(obs=observed_array, fcst_ens=ensemble_array)["crps"]
    crps_python = em.ens_crps(obs=observed_array, fcst_ens=ensemble_array, llvm=False)["crps"]

    assert np.all(np.isclose(expected_crps, crps_numba))
    assert np.all(np.isclose(expected_crps, crps_python))

    crps_mean_numba = em.ens_crps(obs=observed_array, fcst_ens=ensemble_array)["crpsMean"]
    crps_mean_python = em.ens_crps(obs=observed_array, fcst_ens=ensemble_array, llvm=False)[
        "crpsMean"
    ]
    assert np.isclose(expected_mean_crps, crps_mean_numba)
    assert np.isclose(expected_mean_crps, crps_mean_python)


def test_ens_pearson_r(
    ensemble_array: NDArray[np.float64],
    observed_array: NDArray[np.float64],
    ensemble_array_bad: NDArray[np.float64],
    observed_array_bad: NDArray[np.float64],
) -> None:
    expected_pearson_r = -0.13236871294739733
    test_pearson_r = em.ens_pearson_r(obs=observed_array, fcst_ens=ensemble_array)

    assert np.isclose(expected_pearson_r, test_pearson_r)

    expected_pearson_r_bad_data = em.ens_pearson_r(
        obs=observed_array[8:], fcst_ens=ensemble_array[8:, :]
    )
    test_pearson_r_bad_data = em.ens_pearson_r(
        obs=observed_array_bad,
        fcst_ens=ensemble_array_bad,
        remove_zero=True,
        remove_neg=True,
    )

    assert np.isclose(expected_pearson_r_bad_data, test_pearson_r_bad_data)


def test_crps_hersbach(
    files_for_tests: Path, ensemble_array: NDArray[np.float64], observed_array: NDArray[np.float64]
) -> None:
    expected_crps = np.load(files_for_tests / "expected_crps.npy")
    expected_mean_crps = 17.735507981502494
    crps_dictionary_test = em.crps_hersbach(obs=observed_array, fcst_ens=ensemble_array)

    assert np.all(np.isclose(expected_crps, crps_dictionary_test["crps"]))
    assert np.all(np.isclose(expected_mean_crps, crps_dictionary_test["crpsMean1"]))
    assert np.all(np.isclose(expected_mean_crps, crps_dictionary_test["crpsMean2"]))


def test_crps_kernel(
    files_for_tests: Path, ensemble_array: NDArray[np.float64], observed_array: NDArray[np.float64]
) -> None:
    expected_crps = np.load(files_for_tests / "expected_crps.npy")
    expected_mean_crps = 17.735507981502494
    crps_dictionary_test = em.crps_kernel(obs=observed_array, fcst_ens=ensemble_array)
    assert np.all(np.isclose(expected_crps, crps_dictionary_test["crps"]))
    assert np.all(np.isclose(expected_mean_crps, crps_dictionary_test["crpsMean"]))


def test_ens_brier(
    files_for_tests: Path,
    ensemble_array: NDArray[np.float64],
    observed_array: NDArray[np.float64],
    ens_bin: NDArray[np.int64],
    obs_bin: NDArray[np.int64],
) -> None:
    expected_scores_bin = np.load(files_for_tests / "expected_brier_bin.npy")
    expected_mean_score_bin = 0.26351701183431947
    brier_scores_test_bin = em.ens_brier(fcst_ens_bin=ens_bin, obs_bin=obs_bin)
    np.testing.assert_allclose(expected_scores_bin, brier_scores_test_bin)
    np.testing.assert_almost_equal(expected_mean_score_bin, brier_scores_test_bin.mean())

    expected_scores = np.load(files_for_tests / "expected_brier.npy")
    expected_mean_score = 0.17164571005917162
    brier_scores_test = em.ens_brier(ensemble_array, observed_array, 180)
    np.testing.assert_allclose(expected_scores, brier_scores_test)
    assert expected_mean_score == pytest.approx(brier_scores_test.mean())

    expected_scores_diff_thresh = np.load(files_for_tests / "expected_brier_diff_thresh.npy")
    expected_scores_diff_thresh_mean = 0.181926775147929
    brier_scores_test_diff_thresh = em.ens_brier(
        fcst_ens=ensemble_array,
        obs=observed_array,
        obs_threshold=180,
        ens_threshold=170,
    )
    np.testing.assert_allclose(expected_scores_diff_thresh, brier_scores_test_diff_thresh)
    assert expected_scores_diff_thresh_mean == pytest.approx(brier_scores_test_diff_thresh.mean())


def test_auroc(
    ensemble_array: NDArray[np.float64],
    observed_array: NDArray[np.float64],
    ens_bin: NDArray[np.int64],
    obs_bin: NDArray[np.int64],
) -> None:
    auroc_expected = np.array([0.45599759, 0.07259804])
    auroc_expected_bin = np.array([0.43596949, 0.05864427])
    auroc_expected_diff_thresh = np.array([0.3812537673297167, 0.06451017097609267])

    auroc_test = em.auroc(fcst_ens=ensemble_array, obs=observed_array, threshold=180)
    auroc_test_diff_thresh = em.auroc(
        fcst_ens=ensemble_array,
        obs=observed_array,
        obs_threshold=180,
        ens_threshold=170,
    )
    auroc_test_bin = em.auroc(fcst_ens_bin=ens_bin, obs_bin=obs_bin)

    np.testing.assert_allclose(auroc_expected, auroc_test)
    np.testing.assert_allclose(auroc_expected_diff_thresh, auroc_test_diff_thresh)
    np.testing.assert_allclose(auroc_expected_bin, auroc_test_bin)


def test_skill_score() -> None:
    expected_skill_score = 0.5714285714285713
    expected_std = 0.04713063421956128

    skill_score_test = em.skill_score(np.array([0.1, 0.2, 0.15]), np.array([0.3, 0.4, 0.35]), 0)

    np.testing.assert_almost_equal(expected_skill_score, skill_score_test["skillScore"])
    np.testing.assert_almost_equal(expected_std, skill_score_test["standardDeviation"])

    nan_skill_score = em.skill_score(np.array([0.1, 0.2, 0.15]), np.array([0.0, 0.0, 0.0]), 0)

    assert np.isnan(nan_skill_score["skillScore"])
    assert np.isnan(nan_skill_score["standardDeviation"])


def test_skill_score_floats() -> None:
    expected_skill_score = 1 / 3
    test_skill_score = em.skill_score(0.8, 0.7, 1)

    np.testing.assert_almost_equal(expected_skill_score, test_skill_score["skillScore"])
    assert np.isnan(test_skill_score["standardDeviation"])

    nan_skill_score = em.skill_score(0.0, 0.0, 0)
    assert np.isnan(nan_skill_score["skillScore"])
    assert np.isnan(nan_skill_score["standardDeviation"])


@pytest.mark.parametrize("func", [em.ens_me, em.ens_mae, em.ens_mse, em.ens_rmse, em.ens_pearson_r])
def test_ens_metric_invalid_reference(
    func: Callable, observed_array: NDArray[np.float64], ensemble_array: NDArray[np.float64]
) -> None:
    with pytest.raises(ValueError, match="Reference series is not understood"):
        func(obs=observed_array, fcst_ens=ensemble_array, reference="invalid")


def test_treat_data_invalid_obs_dim(ensemble_array: NDArray[np.float64]) -> None:
    obs = np.ones((10, 2))
    with pytest.raises(ValueError, match="obs is not a 1D numpy array"):
        em.treat_data(obs, ensemble_array, remove_zero=False, remove_neg=False)


def test_treat_data_invalid_fcst_ens_dim(observed_array: NDArray[np.float64]) -> None:
    fcst_ens = np.ones((10,))
    with pytest.raises(ValueError, match="fcst_ens is not a 2D numpy array"):
        em.treat_data(observed_array, fcst_ens, remove_zero=False, remove_neg=False)


def test_treat_data_length_mismatch(observed_array: NDArray[np.float64]) -> None:
    fcst_ens = np.ones((5, 2))
    with pytest.raises(
        ValueError, match="obs and fcst_ens do not have the same amount of start dates"
    ):
        em.treat_data(observed_array, fcst_ens, remove_zero=False, remove_neg=False)


def test_skill_score_nonfinite_perf_score() -> None:
    arr = np.array([0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="The perfect score is not finite"):
        em.skill_score(arr, arr, np.nan)


def test_skill_score_invalid_eff_sample_size() -> None:
    arr = np.array([0.1, 0.2, 0.3])
    with pytest.raises(
        ValueError, match="The effective sample size must be finite and greater than 0"
    ):
        em.skill_score(arr, arr, 0, eff_sample_size=-1)
    with pytest.raises(
        ValueError, match="The effective sample size must be finite and greater than 0"
    ):
        em.skill_score(arr, arr, 0, eff_sample_size=np.nan)


def test_skill_score_length_mismatch() -> None:
    arr1 = np.array([0.1, 0.2, 0.3])
    arr2 = np.array([0.1, 0.2])
    with pytest.raises(ValueError, match="The scores and benchmark scores are not the same length"):
        em.skill_score(arr1, arr2, 0)
