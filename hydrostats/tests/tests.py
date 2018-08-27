import sys

if '/home/wade/GitHub/Hydrostats' not in sys.path:
    sys.path.insert(0, '/home/wade/GitHub/Hydrostats')
if '/opt/pycharm-2018.2.2/helpers/pycharm_matplotlib_backend' not in sys.path:
    sys.path.insert(0, '/opt/pycharm-2018.2.2/helpers/pycharm_matplotlib_backend')

import hydrostats.HydroErr as he
import hydrostats.ens_metrics as em
import hydrostats.analyze as ha
import hydrostats.data as hd
import unittest
import doctest
import numpy as np
import pandas as pd
import warnings


class HelperFunctionsTests(unittest.TestCase):

    def test_treat_values_remove(self):
        a = np.random.rand(30, 2)
        a[0, 0] = np.nan
        a[1, 1] = np.nan
        a[2, 0] = np.inf
        a[3, 1] = np.inf
        a[4, 0] = 0
        a[5, 1] = 0
        a[6, 0] = -1
        a[7, 1] = -1

        sim = a[:, 0]
        obs = a[:, 1]

        sim_treated, obs_treated = he.treat_values(sim, obs, remove_zero=True, remove_neg=True)

        # Tests
        self.assertIsNone(np.testing.assert_equal(sim_treated, a[8:, 0]),
                          "Treat values function did not work properly when removing values from "
                          "the simulated data.")
        self.assertIsNone(np.testing.assert_equal(obs_treated, a[8:, 1]),
                          "Treat values function did not work properly when removing values from "
                          "the observed data.")

        with warnings.catch_warnings(record=True) as w:
            # Trigger a warning.
            he.treat_values(sim, obs, remove_zero=True, remove_neg=True)
            # Verify some things
            self.assertTrue(len(w) == 4)
            self.assertTrue(issubclass(w[0].category, UserWarning))
            self.assertTrue(issubclass(w[1].category, UserWarning))
            self.assertTrue(issubclass(w[2].category, UserWarning))
            self.assertTrue(issubclass(w[3].category, UserWarning))
            self.assertTrue("Row(s) [0 1] contained NaN values and the row(s) have been removed "
                            "(Rows are zero indexed)." in str(w[0].message))
            self.assertTrue("Row(s) [2 3] contained Inf or -Inf values and the row(s) have been "
                            "removed (Rows are zero indexed)." in str(w[1].message))
            self.assertTrue("Row(s) [4 5] contained zero values and the row(s) have been removed "
                            "(Rows are zero indexed)." in str(w[2].message))
            self.assertTrue("Row(s) [6 7] contained negative values and the row(s) have been "
                            "removed (Rows are zero indexed)." in str(w[3].message))

    def test_treat_values_replace(self):
        a = np.random.rand(30, 2)
        a[0, 0] = np.nan
        a[1, 1] = np.nan
        a[2, 0] = np.inf
        a[3, 1] = np.inf

        sim = a[:, 0]
        obs = a[:, 1]

        sim_treated, obs_treated = he.treat_values(sim, obs, replace_nan=32, replace_inf=1000)

        sim_new = np.copy(sim)
        obs_new = np.copy(obs)

        sim_new[0] = 32
        sim_new[2] = 1000
        obs_new[1] = 32
        obs_new[3] = 1000

        self.assertIsNone(np.testing.assert_equal(sim_treated, sim_new),
                          "Treat values function did not work properly when replacing values from "
                          "the simulated data.")
        self.assertIsNone(np.testing.assert_equal(obs_treated, obs_new),
                          "Treat values function did not work properly when replacing values from "
                          "the observed data.")

        with warnings.catch_warnings(record=True) as w:
            # Trigger a warning.
            he.treat_values(sim, obs, replace_nan=32, replace_inf=1000)
            # Verify some things
            self.assertTrue(len(w) == 2)
            self.assertTrue(issubclass(w[0].category, UserWarning))
            self.assertTrue(issubclass(w[1].category, UserWarning))
            self.assertTrue("Elements(s) [0] contained NaN values in the simulated array and "
                            "elements(s) [1] contained NaN values in the observed array and have "
                            "been replaced (Elements are zero indexed)." in str(w[0].message))
            self.assertTrue("Elements(s) [2] contained NaN values in the simulated array and "
                            "elements(s) [3] contained NaN values in the observed array and have "
                            "been replaced (Elements are zero indexed)." in str(w[1].message))


class AnalysisTests(unittest.TestCase):

    def setUp(self):
        pd.options.display.max_columns = 100

        # Defining the URLs of the datasets
        sfpt_url = r'https://github.com/waderoberts123/Hydrostats/raw/master/Sample_data/sfpt_data/magdalena' \
                   r'-calamar_interim_data.csv '
        glofas_url = r'https://github.com/waderoberts123/Hydrostats/raw/master/Sample_data/GLOFAS_Data/magdalena' \
                     r'-calamar_ECMWF_data.csv '
        # Merging the data
        self.merged_df = hd.merge_data(sfpt_url, glofas_url, column_names=['SFPT', 'GLOFAS'])

    def test_make_table(self):
        my_metrics = ['MAE', 'r2', 'NSE', 'KGE (2012)']
        seasonal = [['01-01', '03-31'], ['04-01', '06-30'], ['07-01', '09-30'], ['10-01', '12-31']]
        # Using the Function
        table = ha.make_table(self.merged_df, my_metrics, seasonal, remove_neg=True, remove_zero=True)

        # Calculating manually to test
        metric_functions = [he.mae, he.r_squared, he.nse, he.kge_2012]

        season0 = self.merged_df
        season1 = hd.seasonal_period(self.merged_df, daily_period=['01-01', '03-31'])
        season2 = hd.seasonal_period(self.merged_df, daily_period=['04-01', '06-30'])
        season3 = hd.seasonal_period(self.merged_df, daily_period=['07-01', '09-30'])
        season4 = hd.seasonal_period(self.merged_df, daily_period=['10-01', '12-31'])

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

        time_lag_df_original = pd.read_csv(r'/home/wade/GitHub/Hydrostats/hydrostats/tests/Comparison_Files/'
                                           r'time_lag_df.csv', index_col=0)

        summary_df_original = pd.read_csv(r'/home/wade/GitHub/Hydrostats/hydrostats/tests/Comparison_Files/'
                                          r'summary_df.csv', index_col=0)

        self.assertIsNone(pd.testing.assert_frame_equal(time_lag_df, time_lag_df_original))
        self.assertIsNone(pd.testing.assert_frame_equal(summary_df, summary_df_original))

    def tearDown(self):
        pass


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])

    suite.addTest(doctest.DocTestSuite(he))
    suite.addTest(doctest.DocTestSuite(em))

    unittest.TextTestRunner(verbosity=2).run(suite)
