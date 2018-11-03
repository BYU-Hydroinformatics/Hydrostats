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


# TODO: Finish tests on ens_metrics, data, and visual


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
        np.random.seed(3849590438)

        self.ensemble_array = (np.random.rand(100, 52) + 1) * 100  # 52 Ensembles
        self.observed_array = (np.random.rand(100) + 1) * 100

        self.ens_bin = (self.ensemble_array > 75).astype(np.int)
        self.obs_bin = (self.observed_array > 75).astype(np.int)

        self.ensemble_array_bad_data = np.copy(self.ensemble_array)
        self.observed_array_bad_data = np.copy(self.observed_array)

        # Creating bad data to test functions
        self.ensemble_array_bad_data[0, 0] = np.nan
        self.observed_array_bad_data[1] = np.nan
        self.ensemble_array_bad_data[2, 0] = np.inf
        self.observed_array_bad_data[3] = np.inf
        self.ensemble_array_bad_data[4, 0] = 0.
        self.observed_array_bad_data[5] = 0.
        self.ensemble_array_bad_data[6, 0] = -0.1
        self.observed_array_bad_data[7] = -0.1

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

    def tearDown(self):
        np.random.seed(seed=None)


class AnalysisTests(unittest.TestCase):

    def setUp(self):
        pd.options.display.max_columns = 100

        # Defining the URLs of the datasets
        sfpt_url = r'https://github.com/waderoberts123/Hydrostats/raw/master/Sample_data/sfpt_data/magdalena' \
                   r'-calamar_interim_data.csv '
        glofas_url = r'https://github.com/waderoberts123/Hydrostats/raw/master/Sample_data/GLOFAS_Data/magdalena' \
                     r'-calamar_ECMWF_data.csv '
        # Merging the data
        self.merged_df = hd.merge_data(sfpt_url, glofas_url, column_names=('SFPT', 'GLOFAS'))

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
        pass


class VisualTests(unittest.TestCase):

    def setUp(self):
        sfpt_url = r'https://github.com/waderoberts123/Hydrostats/raw/master/Sample_data/sfpt_data/' \
                   r'magdalena-calamar_interim_data.csv'
        glofas_url = r'https://github.com/waderoberts123/Hydrostats/raw/master/Sample_data/GLOFAS_Data/' \
                     r'magdalena-calamar_ECMWF_data.csv'
        self.merged_df = hd.merge_data(sfpt_url, glofas_url, column_names=('SFPT', 'GLOFAS'))

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
        img_original = mpimg.imread(os.path.join(os.getcwd(), 'baseline_images', 'plot_tests', 'plot_full1.png'))

        # Comparing
        self.assertTrue(np.all(np.isclose(img_test, img_original)))

    def test_plot_seasonal(self):
        daily_avg_df = hd.daily_average(merged_data=self.merged_df)
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
        img_original = mpimg.imread(os.path.join(os.getcwd(), 'baseline_images', 'plot_tests', 'plot_seasonal.png'))

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
        sfpt_url = r'https://github.com/waderoberts123/Hydrostats/raw/master/Sample_data/sfpt_data/' \
                   r'magdalena-calamar_interim_data.csv'
        glofas_url = r'https://github.com/waderoberts123/Hydrostats/raw/master/Sample_data/GLOFAS_Data/' \
                     r'magdalena-calamar_ECMWF_data.csv'
        self.merged_df = hd.merge_data(sfpt_url, glofas_url, column_names=('Streamflow Prediction Tool', 'GLOFAS'))

    def test_daily_average(self):
        original_df = pd.read_csv(os.path.join(os.getcwd(), 'Comparison_Files', 'daily_average.csv'), index_col=0)
        original_df.index = original_df.index.astype(np.object)

        test_df = hd.daily_average(self.merged_df)

        self.assertIsNone(pd.testing.assert_frame_equal(original_df, test_df))

    def test_daily_std_dev(self):
        original_df = pd.read_csv(os.path.join(os.getcwd(), 'Comparison_Files', 'daily_std_dev.csv'), index_col=0)
        original_df.index = original_df.index.astype(np.object)

        test_df = hd.daily_std_dev(self.merged_df)

        self.assertIsNone(pd.testing.assert_frame_equal(original_df, test_df))

    def test_daily_std_error(self):
        original_df = pd.read_csv(os.path.join(os.getcwd(), 'Comparison_Files', 'daily_std_error.csv'), index_col=0)
        original_df.index = original_df.index.astype(np.object)

        test_df = hd.daily_std_error(self.merged_df)

        self.assertIsNone(pd.testing.assert_frame_equal(original_df, test_df))

    def test_monthly_average(self):
        original_df = pd.read_csv(os.path.join(os.getcwd(), 'Comparison_Files', 'monthly_average.csv'), index_col=0)
        original_df.index = np.array(['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
                                     dtype=np.object)

        test_df = hd.monthly_average(self.merged_df)

        self.assertIsNone(pd.testing.assert_frame_equal(original_df, test_df))

    def test_monthly_std_dev(self):
        original_df = pd.read_csv(os.path.join(os.getcwd(), 'Comparison_Files', 'monthly_std_dev.csv'), index_col=0)
        original_df.index = np.array(['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
                                     dtype=np.object)

        test_df = hd.monthly_std_dev(self.merged_df)

        self.assertIsNone(pd.testing.assert_frame_equal(original_df, test_df))

    def test_monthly_std_error(self):
        original_df = pd.read_csv(os.path.join(os.getcwd(), 'Comparison_Files', 'monthly_std_error.csv'), index_col=0)
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
        pass


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])

    unittest.TextTestRunner(verbosity=2).run(suite)
