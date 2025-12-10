from __future__ import annotations

import numpy as np
import pytest

import hydrostats.ens_metrics as em


@pytest.fixture(scope="module")
def ensemble_array(files_for_tests):
    return np.load(files_for_tests / "ensemble_array.npz")["arr_0.npy"]


@pytest.fixture(scope="module")
def observed_array(files_for_tests):
    return np.load(files_for_tests / "observed_array.npz")["arr_0.npy"]


@pytest.fixture(scope="module")
def ensemble_array_bad(files_for_tests):
    return np.load(files_for_tests / "ensemble_array_bad_data.npz")["arr_0.npy"]


@pytest.fixture(scope="module")
def observed_array_bad(files_for_tests):
    return np.load(files_for_tests / "observed_array_bad_data.npz")["arr_0.npy"]


@pytest.fixture(scope="module")
def ens_bin(files_for_tests):
    return np.load(files_for_tests / "ens_bin.npy")


@pytest.fixture(scope="module")
def obs_bin(files_for_tests):
    return np.load(files_for_tests / "obs_bin.npy")


def test_ens_me(ensemble_array, observed_array, ensemble_array_bad, observed_array_bad):
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


def test_ens_mae(ensemble_array, observed_array, ensemble_array_bad, observed_array_bad):
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


def test_ens_mse(ensemble_array, observed_array, ensemble_array_bad, observed_array_bad):
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


def test_ens_rmse(ensemble_array, observed_array, ensemble_array_bad, observed_array_bad):
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


def test_ens_crps(files_for_tests, ensemble_array, observed_array):
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


def test_ens_pearson_r(ensemble_array, observed_array, ensemble_array_bad, observed_array_bad):
    expected_pearson_r = 0.9624026713489114
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


def test_crps_hersbach(files_for_tests, ensemble_array, observed_array):
    expected_crps = np.load(files_for_tests / "expected_crps.npy")
    expected_mean_crps = 17.735507981502494
    crps_dictionary_test = em.crps_hersbach(obs=observed_array, fcst_ens=ensemble_array)

    assert np.all(np.isclose(expected_crps, crps_dictionary_test["crps"]))
    assert np.all(np.isclose(expected_mean_crps, crps_dictionary_test["crpsMean1"]))
    assert np.all(np.isclose(expected_mean_crps, crps_dictionary_test["crpsMean2"]))


def test_crps_kernel(files_for_tests, ensemble_array, observed_array):
    expected_crps = np.load(files_for_tests / "expected_crps.npy")
    expected_mean_crps = 17.735507981502494
    crps_dictionary_test = em.crps_kernel(obs=observed_array, fcst_ens=ensemble_array)
    assert np.all(np.isclose(expected_crps, crps_dictionary_test["crps"]))
    assert np.all(np.isclose(expected_mean_crps, crps_dictionary_test["crpsMean"]))


def test_ens_brier(files_for_tests, ensemble_array, observed_array, ens_bin, obs_bin):
    expected_scores_bin = np.load(files_for_tests / "expected_brier_bin.npy")
    expected_mean_score_bin = 0.26351701183431947
    brier_scores_test_bin = em.ens_brier(fcst_ens_bin=ens_bin, obs_bin=obs_bin)
    np.testing.assert_allclose(expected_scores_bin, brier_scores_test_bin)
    assert expected_mean_score_bin == pytest.approx(brier_scores_test_bin.mean())

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
        thresholds=[150, 200, 250, 300, 350, 400, 450, 500],
    )
    np.testing.assert_allclose(expected_scores_diff_thresh, brier_scores_test_diff_thresh)
    assert expected_scores_diff_thresh_mean == pytest.approx(brier_scores_test_diff_thresh.mean())


def test_auroc(ens_bin, obs_bin, files_for_tests):
    auroc_expected = np.load(files_for_tests / "expected_brier_bin.npy")
    auroc_expected_diff_thresh = np.load(files_for_tests / "expected_brier.npy")
    auroc_expected_bin = np.load(files_for_tests / "expected_brier_bin.npy")

    auroc_test = em.auroc(obs_bin=obs_bin, fcst_ens_bin=ens_bin)
    auroc_test_diff_thresh = em.auroc(fcst_ens=ens_bin, obs=obs_bin, thresholds=[0.1, 0.5, 0.9])
    auroc_test_bin = em.auroc(obs_bin=obs_bin, fcst_ens_bin=ens_bin)

    np.testing.assert_allclose(auroc_expected, auroc_test)
    np.testing.assert_allclose(auroc_expected_diff_thresh, auroc_test_diff_thresh)
    np.testing.assert_allclose(auroc_expected_bin, auroc_test_bin)


def test_skill_score():
    expected_skill_score = 0.9969802250063856
    expected_std = 0.05392719857907347
    skill_score_test = em.skill_score(20, 18, 15)
    assert expected_skill_score == pytest.approx(skill_score_test["skillScore"])
    assert expected_std == pytest.approx(skill_score_test["standardDeviation"])

    nan_skill_score = em.skill_score(0.0, 0.0, 0)
    assert np.isnan(nan_skill_score["skillScore"]) and np.isnan(
        nan_skill_score["standardDeviation"]
    )


def test_skill_score_floats():
    expected_skill_score = 0.9969802250063856
    test_skill_score = em.skill_score(20.0, 18.0, 15)
    assert expected_skill_score == pytest.approx(test_skill_score["skillScore"])
    assert np.isnan(test_skill_score["standardDeviation"])
