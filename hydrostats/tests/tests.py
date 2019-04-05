import sys
import os

package_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if package_path not in sys.path:
    sys.path.insert(0, package_path)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import hydrostats.metrics as he
import hydrostats.ens_metrics as em
import hydrostats.analyze as ha
import hydrostats.data as hd
import hydrostats.visual as hv
import matplotlib.image as mpimg
import unittest
import numpy as np
import pandas as pd

try:
    from io import BytesIO
except ImportError:
    from BytesIO import BytesIO  # For python 2.7 compatibility


class MetricsTests(unittest.TestCase):
    """
    Tests the metrics included in HydroErr package to make sure that they are working correctly.
    """

    def setUp(self):
        self.sim = np.array([5, 7, 9, 2, 4.5, 6.7])
        self.obs = np.array([4.7, 6, 10, 2.5, 4, 6.8])
        self.sim_bad_data = np.array([6, np.nan, 100, np.inf, 200, -np.inf, 300, 0, 400, -0.1, 5, 7, 9, 2, 4.5, 6.7])
        self.obs_bad_data = np.array(
            [np.nan, 100, np.inf, 200, -np.inf, 300, 0, 400, -0.1, 500, 4.7, 6, 10, 2.5, 4, 6.8])

    def test_list_of_metrics(self):
        expected_list = []

        for metric_func in he.function_list:
            expected_list.append(metric_func(self.sim, self.obs))

        test_list_without_abbr = he.list_of_metrics(he.metric_names, self.sim, self.obs)
        test_list_with_abbr = he.list_of_metrics(he.metric_abbr, self.sim, self.obs, abbr=True)

        self.assertEqual(expected_list, test_list_with_abbr)
        self.assertEqual(expected_list, test_list_without_abbr)

        # Testing all of the functions with additional parameters
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
            he.mase(self.sim, self.obs, m=mase_m), he.dmod(self.sim, self.obs, j=d_mod_j),
            he.nse_mod(self.sim, self.obs, j=nse_mod_j), he.lm_index(self.sim, self.obs, obs_bar_p=lm_x_obs_bar_p),
            he.h6_mhe(self.sim, self.obs, k=h6_mhe_k), he.h6_mahe(self.sim, self.obs, k=h6_ahe_k),
            he.h6_rmshe(self.sim, self.obs, k=h6_rmshe_k), he.d1_p(self.sim, self.obs, obs_bar_p=d1_p_obs_bar_p),
            he.kge_2009(self.sim, self.obs, s=kge2009_s), he.kge_2012(self.sim, self.obs, s=kge2012_s)
        ]

        list_of_metric_names = [
            'Mean Absolute Scaled Error', 'Modified Index of Agreement', 'Modified Nash-Sutcliffe Efficiency',
            'Legate-McCabe Efficiency Index', 'Mean H6 Error', 'Mean Absolute H6 Error', 'Root Mean Square H6 Error',
            'Legate-McCabe Index of Agreement', 'Kling-Gupta Efficiency (2009)', 'Kling-Gupta Efficiency (2012)'
        ]
        list_of_metric_abbr = [
            'MASE', 'd (Mod.)', 'NSE (Mod.)', "E1'", 'H6 (MHE)', 'H6 (AHE)', 'H6 (RMSHE)', "D1'", 'KGE (2009)',
            'KGE (2012)'
        ]

        test_list_without_abbr_params = he.list_of_metrics(
            list_of_metric_names, self.sim, self.obs, mase_m=mase_m, dmod_j=d_mod_j, nse_mod_j=nse_mod_j,
            h6_mhe_k=h6_mhe_k, h6_ahe_k=h6_ahe_k, h6_rmshe_k=h6_rmshe_k, d1_p_obs_bar_p=d1_p_obs_bar_p,
            lm_x_obs_bar_p=lm_x_obs_bar_p, kge2009_s=kge2009_s, kge2012_s=kge2012_s
        )
        test_list_with_abbr_params = he.list_of_metrics(
            list_of_metric_abbr, self.sim, self.obs, abbr=True, mase_m=mase_m, dmod_j=d_mod_j, nse_mod_j=nse_mod_j,
            h6_mhe_k=h6_mhe_k, h6_ahe_k=h6_ahe_k, h6_rmshe_k=h6_rmshe_k, d1_p_obs_bar_p=d1_p_obs_bar_p,
            lm_x_obs_bar_p=lm_x_obs_bar_p, kge2009_s=kge2009_s, kge2012_s=kge2012_s
        )

        self.assertEqual(expected_list_with_params, test_list_without_abbr_params)
        self.assertEqual(expected_list_with_params, test_list_with_abbr_params)

        # Testing the exceptions from bad input
        two_dim_sim = np.array([[1], [2], [3], [4]])
        unequal_length_sim = np.array([1, 2, 3, 4, 5])
        obs = np.array([4, 3, 2, 1])

        with self.assertRaises(Exception) as context:
            he.list_of_metrics(list_of_metric_names, two_dim_sim, obs)

        self.assertTrue("One or both of the ndarrays are not 1 dimensional." in context.exception.args[0])
        self.assertIsInstance(context.exception, RuntimeError)

        with self.assertRaises(Exception) as context:
            he.list_of_metrics(list_of_metric_names, unequal_length_sim, obs)

        self.assertTrue("The two ndarrays are not the same size." in context.exception.args[0])
        self.assertIsInstance(context.exception, RuntimeError)

    def tearDown(self):
        del self.sim
        del self.obs


class EnsMetricsTests(unittest.TestCase):

    def setUp(self):
        self.ensemble_array = np.load(r"Files_for_tests/ensemble_array.npz")["arr_0.npy"]
        self.observed_array = np.load(r"Files_for_tests/observed_array.npz")["arr_0.npy"]

        self.ens_bin = np.load(r"Files_for_tests/ens_bin.npy")
        self.obs_bin = np.load(r"Files_for_tests/obs_bin.npy")

        self.ensemble_array_bad_data = np.load(r"Files_for_tests/ensemble_array_bad_data.npz")["arr_0.npy"]
        self.observed_array_bad_data = np.load(r"Files_for_tests/observed_array_bad_data.npz")["arr_0.npy"]

    def test_ens_me(self):
        expected_value = -2.5217349574908074
        test_value = em.ens_me(obs=self.observed_array, fcst_ens=self.ensemble_array)

        self.assertTrue(np.isclose(expected_value, test_value))

        expected_value_bad_data = em.ens_me(obs=self.observed_array[8:], fcst_ens=self.ensemble_array[8:, :])
        test_value_bad_data = em.ens_me(obs=self.observed_array_bad_data, fcst_ens=self.ensemble_array_bad_data,
                                        remove_zero=True, remove_neg=True)

        self.assertTrue(np.isclose(expected_value_bad_data, test_value_bad_data))

    def test_ens_mae(self):
        expected_value = 26.35428724003365
        test_value = em.ens_mae(obs=self.observed_array, fcst_ens=self.ensemble_array)

        self.assertTrue(np.isclose(expected_value, test_value))

        expected_value_bad_data = em.ens_mae(obs=self.observed_array[8:], fcst_ens=self.ensemble_array[8:, :])
        test_value_bad_data = em.ens_mae(obs=self.observed_array_bad_data, fcst_ens=self.ensemble_array_bad_data,
                                         remove_zero=True, remove_neg=True)

        self.assertTrue(np.isclose(expected_value_bad_data, test_value_bad_data))

    def test_ens_mse(self):
        expected_value = 910.5648405687582
        test_value = em.ens_mse(obs=self.observed_array, fcst_ens=self.ensemble_array)

        self.assertTrue(np.isclose(expected_value, test_value))

        expected_value_bad_data = em.ens_mse(obs=self.observed_array[8:], fcst_ens=self.ensemble_array[8:, :])
        test_value_bad_data = em.ens_mse(obs=self.observed_array_bad_data, fcst_ens=self.ensemble_array_bad_data,
                                         remove_zero=True, remove_neg=True)

        self.assertTrue(np.isclose(expected_value_bad_data, test_value_bad_data))

    def test_ens_rmse(self):
        expected_value = 30.17556694693172
        test_value = em.ens_rmse(obs=self.observed_array, fcst_ens=self.ensemble_array)

        self.assertTrue(np.isclose(expected_value, test_value))

        expected_value_bad_data = em.ens_rmse(obs=self.observed_array[8:], fcst_ens=self.ensemble_array[8:, :])
        test_value_bad_data = em.ens_rmse(obs=self.observed_array_bad_data, fcst_ens=self.ensemble_array_bad_data,
                                          remove_zero=True, remove_neg=True)

        self.assertTrue(np.isclose(expected_value_bad_data, test_value_bad_data))

    def test_ens_crps(self):
        expected_crps = np.load("Files_for_tests/expected_crps.npy")
        expected_mean_crps = 17.735507981502494

        crps_numba = em.ens_crps(obs=self.observed_array, fcst_ens=self.ensemble_array)["crps"]
        crps_python = em.ens_crps(obs=self.observed_array, fcst_ens=self.ensemble_array, llvm=False)["crps"]

        self.assertTrue(np.all(np.isclose(expected_crps, crps_numba)))
        self.assertTrue(np.all(np.isclose(expected_crps, crps_python)))

        crps_mean_numba = em.ens_crps(obs=self.observed_array, fcst_ens=self.ensemble_array)["crpsMean"]
        crps_mean_python = em.ens_crps(obs=self.observed_array, fcst_ens=self.ensemble_array, llvm=False)["crpsMean"]

        self.assertTrue(np.isclose(expected_mean_crps, crps_mean_numba))
        self.assertTrue(np.isclose(expected_mean_crps, crps_mean_python))

        expected_value_bad_data = em.ens_crps(
            obs=self.observed_array[8:], fcst_ens=self.ensemble_array[8:, :]
        )
        test_value_bad_data = em.ens_crps(
            obs=self.observed_array_bad_data, fcst_ens=self.ensemble_array_bad_data, remove_zero=True, remove_neg=True
        )

        self.assertTrue(np.all(np.isclose(expected_value_bad_data["crps"], test_value_bad_data["crps"])))
        self.assertEqual(expected_value_bad_data["crpsMean"], test_value_bad_data["crpsMean"])

    def test_ens_pearson_r(self):
        expected_pearson_r = -0.13236871294739733
        test_pearson_r = em.ens_pearson_r(obs=self.observed_array, fcst_ens=self.ensemble_array)

        self.assertTrue(np.isclose(expected_pearson_r, test_pearson_r))

        expected_pearson_r_bad_data = em.ens_pearson_r(obs=self.observed_array[8:], fcst_ens=self.ensemble_array[8:, :])
        test_pearson_r_bad_data = em.ens_pearson_r(obs=self.observed_array_bad_data,
                                                   fcst_ens=self.ensemble_array_bad_data,
                                                   remove_zero=True, remove_neg=True)

        self.assertTrue(np.isclose(expected_pearson_r_bad_data, test_pearson_r_bad_data))

    def test_crps_hersbach(self):
        expected_crps = np.load("Files_for_tests/expected_crps.npy")
        expected_mean_crps = 17.735507981502494

        crps_dictionary_test = em.crps_hersbach(obs=self.observed_array, fcst_ens=self.ensemble_array)

        self.assertTrue(np.all(np.isclose(expected_crps, crps_dictionary_test["crps"])))
        self.assertTrue(np.all(np.isclose(expected_mean_crps, crps_dictionary_test["crpsMean1"])))
        self.assertTrue(np.all(np.isclose(expected_mean_crps, crps_dictionary_test["crpsMean2"])))

    def test_crps_kernel(self):
        expected_crps = np.load("Files_for_tests/expected_crps.npy")
        expected_mean_crps = 17.735507981502494

        crps_dictionary_test = em.crps_kernel(obs=self.observed_array, fcst_ens=self.ensemble_array)

        self.assertTrue(np.all(np.isclose(expected_crps, crps_dictionary_test["crps"])))
        self.assertTrue(np.all(np.isclose(expected_mean_crps, crps_dictionary_test["crpsMean"])))

    def test_ens_brier(self):
        # With the binary array
        expected_scores_bin = np.load("Files_for_tests/expected_brier_bin.npy")
        expected_mean_score_bin = 0.26351701183431947

        brier_scores_test_bin = em.ens_brier(fcst_ens_bin=self.ens_bin, obs_bin=self.obs_bin)

        np.testing.assert_allclose(expected_scores_bin, brier_scores_test_bin)
        self.assertAlmostEqual(expected_mean_score_bin, brier_scores_test_bin.mean())

        # With the data arrays
        expected_scores = np.load("Files_for_tests/expected_brier.npy")
        expected_mean_score = 0.17164571005917162

        brier_scores_test = em.ens_brier(self.ensemble_array, self.observed_array, 180)

        np.testing.assert_allclose(expected_scores, brier_scores_test)
        self.assertAlmostEqual(expected_mean_score, brier_scores_test.mean())

    def test_auroc(self):
        auroc_expected = np.array([0.45599759, 0.07259804])
        auroc_expected_bin = np.array([0.43596949, 0.05864427])

        auroc_test = em.auroc(fcst_ens=self.ensemble_array, obs=self.observed_array, threshold=180)
        auroc_test_bin = em.auroc(fcst_ens_bin=self.ens_bin, obs_bin=self.obs_bin)

        np.testing.assert_allclose(auroc_expected, auroc_test)
        np.testing.assert_allclose(auroc_expected_bin, auroc_test_bin)

    def test_skill_score(self):
        expected_skill_score = 0.5714285714285713
        expected_std = 0.04713063421956128

        skill_score_test = em.skill_score(np.array([0.1, 0.2, 0.15]), np.array([0.3, 0.4, 0.35]), 0)

        self.assertAlmostEqual(expected_skill_score, skill_score_test["skillScore"])
        self.assertAlmostEqual(expected_std, skill_score_test["standardDeviation"])

        nan_skill_score = em.skill_score(np.array([0.1, 0.2, 0.15]), np.array([0.0, 0.0, 0.0]), 0)

        self.assertTrue(np.isnan(nan_skill_score["skillScore"]))
        self.assertTrue(np.isnan(nan_skill_score["standardDeviation"]))

    def test_skill_score_floats(self):
        expected_skill_score = 1 / 3
        test_skill_score = em.skill_score(0.8, 0.7, 1)

        self.assertAlmostEqual(expected_skill_score, test_skill_score["skillScore"])
        self.assertTrue(np.isnan(test_skill_score["standardDeviation"]))

        nan_skill_score = em.skill_score(0., 0., 0)
        self.assertTrue(np.isnan(nan_skill_score["skillScore"]))
        self.assertTrue(np.isnan(nan_skill_score["standardDeviation"]))

    def tearDown(self):
        del self.ensemble_array
        del self.observed_array

        del self.ens_bin
        del self.obs_bin

        del self.ensemble_array_bad_data
        del self.observed_array_bad_data


class AnalysisTests(unittest.TestCase):

    def setUp(self):
        # Reading the merged data from pickle
        self.merged_df = pd.read_pickle("Files_for_tests/merged_df.pkl")

    def test_make_table(self):
        my_metrics = ['MAE', 'r2', 'NSE', 'KGE (2012)']
        seasonal = [['01-01', '03-31'], ['04-01', '06-30'], ['07-01', '09-30'], ['10-01', '12-31']]
        # Using the Function
        table = ha.make_table(self.merged_df, my_metrics, seasonal, remove_neg=True, remove_zero=True)

        # Calculating manually to test
        metric_functions = [he.mae, he.r_squared, he.nse, he.kge_2012]

        season0 = self.merged_df
        season1 = hd.seasonal_period(self.merged_df, daily_period=('01-01', '03-31'))
        season2 = hd.seasonal_period(self.merged_df, daily_period=('04-01', '06-30'))
        season3 = hd.seasonal_period(self.merged_df, daily_period=('07-01', '09-30'))
        season4 = hd.seasonal_period(self.merged_df, daily_period=('10-01', '12-31'))

        all_seasons = [season0, season1, season2, season3, season4]

        test_list = []

        for season in all_seasons:
            temp_list = []
            for metric in metric_functions:
                sim = season.iloc[:, 0].values
                obs = season.iloc[:, 1].values
                temp_list.append(metric(sim, obs, remove_neg=True, remove_zero=True))
            test_list.append(temp_list)

        test_table = pd.DataFrame(
            test_list,
            index=['Full Time Series', 'January-01:March-31', 'April-01:June-30',
                   'July-01:September-30', 'October-01:December-31'],
            columns=['MAE', 'r2', 'NSE', 'KGE (2012)']
        )

        self.assertIsNone(pd.testing.assert_frame_equal(test_table, table))

    def test_lag_analysis(self):
        # Running the lag analysis
        time_lag_df, summary_df = ha.time_lag(self.merged_df, metrics=['ME', 'r2', 'RMSE', 'KGE (2012)', 'NSE'])

        time_lag_df_original = pd.read_csv(os.path.join(os.getcwd(), 'Comparison_Files', 'time_lag_df.csv'),
                                           index_col=0)

        summary_df_original = pd.read_csv(os.path.join(os.getcwd(), 'Comparison_Files', 'summary_df.csv'), index_col=0)

        self.assertIsNone(pd.testing.assert_frame_equal(time_lag_df, time_lag_df_original))
        self.assertIsNone(pd.testing.assert_frame_equal(summary_df, summary_df_original))

    def tearDown(self):
        del self.merged_df


class VisualTests(unittest.TestCase):

    def setUp(self):
        self.merged_df = pd.read_pickle("Files_for_tests/merged_df.pkl")

    def test_plot_full1(self):
        # Creating Test Image
        hv.plot(merged_data_df=self.merged_df,
                title='Hydrograph of Entire Time Series',
                linestyles=['r-', 'k-'],
                legend=('SFPT', 'GLOFAS'),
                labels=['Datetime', 'Streamflow (cfs)'],
                metrics=['ME', 'NSE', 'SA'],
                grid=True)
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_test = mpimg.imread(buf)
        buf.close()

        # Reading Original Image
        img_original = mpimg.imread(r'baseline_images/plot_tests/plot_full1.png')

        # Comparing
        self.assertTrue(np.all(np.isclose(img_test, img_original)))

    def test_plot_seasonal(self):
        daily_avg_df = hd.daily_average(df=self.merged_df)
        daily_std_error = hd.daily_std_error(merged_data=self.merged_df)

        # Creating test image array
        hv.plot(merged_data_df=daily_avg_df,
                title='Daily Average Streamflow (Standard Error)',
                legend=('SFPT', 'GLOFAS'),
                x_season=True,
                labels=['Datetime', 'Streamflow (csm)'],
                linestyles=['r-', 'k-'],
                fig_size=(14, 8),
                ebars=daily_std_error,
                ecolor=('r', 'k'),
                tight_xlim=True
                )
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_test = mpimg.imread(buf)
        buf.close()

        # Reading Original Image
        img_original = mpimg.imread(r'baseline_images/plot_tests/plot_seasonal.png')

        # Comparing
        self.assertTrue(np.all(np.isclose(img_test, img_original)))

    def test_hist_df(self):
        # Creating test image array
        hv.hist(merged_data_df=self.merged_df,
                num_bins=100,
                title='Histogram of Streamflows',
                legend=('SFPT', 'GLOFAS'),
                labels=('Bins', 'Frequency'),
                grid=True)
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_test = mpimg.imread(buf)
        buf.close()

        # Reading original image
        img_original = mpimg.imread(os.path.join(os.getcwd(), 'baseline_images', 'plot_tests', 'hist1.png'))

        # Comparing the images
        self.assertTrue(np.all(np.isclose(img_test, img_original)))

    def test_hist_arrays(self):
        # Creating test image array
        sim_array = self.merged_df.iloc[:, 0].values
        obs_array = self.merged_df.iloc[:, 1].values

        hv.hist(sim_array=sim_array,
                obs_array=obs_array,
                num_bins=100,
                title='Histogram of Streamflows',
                legend=('SFPT', 'GLOFAS'),
                labels=('Bins', 'Frequency'),
                grid=True)
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_test = mpimg.imread(buf)
        buf.close()

        # Reading original image
        img_original = mpimg.imread(os.path.join(os.getcwd(), 'baseline_images', 'plot_tests', 'hist1.png'))

        # Comparing the images
        self.assertTrue(np.all(np.isclose(img_test, img_original)))

    def test_hist_znorm(self):
        # Creating test image array
        # noinspection PyTypeChecker
        hv.hist(merged_data_df=self.merged_df,
                num_bins=100,
                title='Histogram of Streamflows',
                labels=('Bins', 'Frequency'),
                grid=True,
                z_norm=True,
                legend=None,
                prob_dens=True)
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_test = mpimg.imread(buf)
        buf.close()

        # Reading original image
        img_original = mpimg.imread(os.path.join(os.getcwd(), 'baseline_images', 'plot_tests', 'hist_znorm.png'))

        # Comparing the images
        self.assertTrue(np.all(np.isclose(img_test, img_original)))

    def test_hist_error(self):
        df = self.merged_df
        sim_array = df.iloc[:, 0].values
        with self.assertRaises(RuntimeError):
            hv.hist(merged_data_df=self.merged_df, sim_array=sim_array)

    def test_scatter(self):
        # Creating test image array
        hv.scatter(merged_data_df=self.merged_df, grid=True, title='Scatter Plot (Normal Scale)',
                   labels=('SFPT', 'GLOFAS'), best_fit=True)
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_test = mpimg.imread(buf)
        buf.close()

        # Reading original image
        img_original = mpimg.imread(os.path.join(os.getcwd(), 'baseline_images', 'plot_tests', 'scatter.png'))

        # Comparing the images
        self.assertTrue(np.all(np.isclose(img_test, img_original)))

    def test_scatterlog(self):
        sim_array = self.merged_df.iloc[:, 0].values
        obs_array = self.merged_df.iloc[:, 1].values

        # Creating test image array
        hv.scatter(sim_array=sim_array, obs_array=obs_array, grid=True, title='Scatter Plot (Log-Log Scale)',
                   labels=('SFPT', 'GLOFAS'), line45=True, metrics=['ME', 'KGE (2012)'], log_scale=True)
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_test = mpimg.imread(buf)
        buf.close()

        # Reading original image
        img_original = mpimg.imread(os.path.join(os.getcwd(), 'baseline_images', 'plot_tests', 'scatterlog.png'))

        # Comparing the images
        self.assertTrue(np.all(np.isclose(img_test, img_original)))

    def test_qq_plot(self):
        # Creating test image array
        hv.qqplot(merged_data_df=self.merged_df, title='Quantile-Quantile Plot of Data',
                  xlabel='SFPT Data Quantiles', ylabel='GLOFAS Data Quantiles', legend=True,
                  figsize=(8, 6))
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_test = mpimg.imread(buf)
        buf.close()

        # Reading original image
        img_original = mpimg.imread(os.path.join(os.getcwd(), 'baseline_images', 'plot_tests', 'qqplot.png'))

        # Comparing the images
        self.assertTrue(np.all(np.isclose(img_test, img_original)))

    def test_qq_plot2(self):
        # Creating test image array
        sim_array = self.merged_df.iloc[:, 0].values
        obs_array = self.merged_df.iloc[:, 1].values
        hv.qqplot(sim_array=sim_array, obs_array=obs_array, title='Quantile-Quantile Plot of Data',
                  xlabel='SFPT Data Quantiles', ylabel='GLOFAS Data Quantiles', figsize=(8, 6))
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_test = mpimg.imread(buf)
        buf.close()

        # Reading original image
        img_original = mpimg.imread(os.path.join(os.getcwd(), 'baseline_images', 'plot_tests', 'qqplot2.png'))

        # Comparing the images
        self.assertTrue(np.all(np.isclose(img_test, img_original)))

    def tearDown(self):
        del self.merged_df


class DataTests(unittest.TestCase):

    def setUp(self):
        self.merged_df = pd.read_pickle("Files_for_tests/merged_df.pkl")

    def test_julian_to_gregorian(self):
        julian_dates = np.array([2444239.5, 2444239.5416666665, 2444239.5833333335, 2444239.625,
                                 2444239.6666666665, 2444239.7083333335, 2444239.75,
                                 2444239.7916666665, 2444239.8333333335, 2444239.875])
        expected_dates = pd.date_range('1980-01-01', periods=10, freq='H')

        test_df = pd.DataFrame(data=np.random.rand(10, 2),  # Random data in the columns
                               columns=("Simulated Data", "Observed Data"),
                               index=julian_dates)

        test_df_gregorian = hd.julian_to_gregorian(test_df, frequency="H")
        self.assertTrue(np.all(test_df_gregorian.index == expected_dates))

        hd.julian_to_gregorian(test_df, inplace=True, frequency="H")  # Change index values inplace
        self.assertTrue(np.all(test_df.index == expected_dates))

    def test_merge_data(self):

        # TODO: Finish these tests

        random_data = np.random.rand(31)

        # Scenario 1 from docs
        sim_dates = pd.date_range('1980-01-01', '1980-01-31')
        obs_dates = pd.date_range('1980-01-15', '1980-02-16')
        expected_merged_dates = pd.date_range('1980-01-15', '1980-01-31')

        # sim_df =

        # merged_df_1 = hd.merge_data(sim_df=)

    def test_daily_average(self):
        original_df = pd.read_csv(r'Comparison_Files/daily_average.csv', index_col=0)
        original_df.index = original_df.index.astype(np.object)

        test_df = hd.daily_average(self.merged_df)

        self.assertIsNone(pd.testing.assert_frame_equal(original_df, test_df))

    def test_daily_std_dev(self):
        original_df = pd.read_csv(r'Comparison_Files/daily_std_dev.csv', index_col=0)
        original_df.index = original_df.index.astype(np.object)

        test_df = hd.daily_std_dev(self.merged_df)

        self.assertIsNone(pd.testing.assert_frame_equal(original_df, test_df))

    def test_daily_std_error(self):
        original_df = pd.read_csv(r'Comparison_Files/daily_std_error.csv', index_col=0)
        original_df.index = original_df.index.astype(np.object)

        test_df = hd.daily_std_error(self.merged_df)

        self.assertIsNone(pd.testing.assert_frame_equal(original_df, test_df))

    def test_monthly_average(self):
        original_df = pd.read_csv(r'Comparison_Files/monthly_average.csv', index_col=0)
        original_df.index = np.array(['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
                                     dtype=np.object)

        test_df = hd.monthly_average(self.merged_df)

        self.assertIsNone(pd.testing.assert_frame_equal(original_df, test_df))

    def test_monthly_std_dev(self):
        original_df = pd.read_csv(r'Comparison_Files/monthly_std_dev.csv', index_col=0)
        original_df.index = np.array(['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
                                     dtype=np.object)

        test_df = hd.monthly_std_dev(self.merged_df)

        self.assertIsNone(pd.testing.assert_frame_equal(original_df, test_df))

    def test_monthly_std_error(self):
        original_df = pd.read_csv(r'Comparison_Files/monthly_std_error.csv', index_col=0)
        original_df.index = np.array(['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
                                     dtype=np.object)

        test_df = hd.monthly_std_error(self.merged_df)

        self.assertIsNone(pd.testing.assert_frame_equal(original_df, test_df))

    def test_remove_nan_df(self):
        data = np.random.rand(15, 2)
        data[0, 0] = data[1, 1] = np.nan
        data[2, 0] = data[3, 1] = np.inf
        data[4, 0] = data[5, 1] = 0
        data[6, 0] = data[7, 1] = -0.1

        test_df = hd.remove_nan_df(pd.DataFrame(data=data, index=pd.date_range('1980-01-01', periods=15)))
        original_df = pd.DataFrame(data=data[8:, :], index=pd.date_range('1980-01-09', periods=7))

        self.assertIsNone(pd.testing.assert_frame_equal(original_df, test_df))

    def tearDown(self):
        del self.merged_df


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])

    unittest.TextTestRunner(verbosity=2).run(suite)
