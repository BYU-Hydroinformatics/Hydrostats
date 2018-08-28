import sys
import os

package_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if package_path not in sys.path:
    sys.path.insert(0, package_path)

import hydrostats.metrics as he
import hydrostats.ens_metrics as em
import hydrostats.analyze as ha
import hydrostats.data as hd
import hydrostats.visual as hv
import matplotlib.image as mpimg
import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

try:
    from io import BytesIO
except ImportError:
    from BytesIO import BytesIO


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

    def test_me(self):
        expected_value = 0.03333333333333336
        test_value = he.me(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.me(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_mae(self):
        expected_value = 0.5666666666666665
        test_value = he.mae(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.mae(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_mse(self):
        expected_value = 0.4333333333333333
        test_value = he.mse(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.mse(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_mle(self):
        expected_value = 0.002961767058151136
        test_value = he.mle(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.mle(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_male(self):
        expected_value = 0.09041652188064823
        test_value = he.male(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.male(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_msle(self):
        expected_value = 0.010426437593600514
        test_value = he.msle(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.msle(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_mde(self):
        expected_value = 0.10000000000000009
        test_value = he.mde(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.mde(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_mdae(self):
        expected_value = 0.5
        test_value = he.mdae(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.mdae(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_mdse(self):
        expected_value = 0.25
        test_value = he.mdse(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.mdse(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_ed(self):
        expected_value = 1.6124515496597098
        test_value = he.ed(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.ed(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_ned(self):
        expected_value = 0.28491828688422466
        test_value = he.ned(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.ned(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_rmse(self):
        expected_value = 0.6582805886043833
        test_value = he.rmse(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.rmse(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_rmsle(self):
        expected_value = 0.10210992896677833
        test_value = he.rmsle(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.rmsle(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_nrmse_range(self):
        expected_value = 0.08777074514725111
        test_value = he.nrmse_range(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.nrmse_range(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_nrmse_mean(self):
        expected_value = 0.11616716269489116
        test_value = he.nrmse_mean(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.nrmse_mean(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_nrmse_iqr(self):
        expected_value = 0.27145591282654985
        test_value = he.nrmse_iqr(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.nrmse_iqr(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_irmse(self):
        expected_value = 0.14438269394140332
        test_value = he.irmse(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.irmse(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_mase(self):
        expected_value = 0.1656920077972709
        test_value = he.mase(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.mase(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_r_squared(self):
        expected_value = 0.9246652089263256
        test_value = he.r_squared(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.r_squared(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_pearson_r(self):
        expected_value = 0.9615951377405804
        test_value = he.pearson_r(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.pearson_r(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_spearman_r(self):
        expected_value = 0.942857142857143
        test_value = he.spearman_r(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.spearman_r(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_acc(self):
        expected_value = 0.8013292814504837
        test_value = he.acc(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.acc(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_mape(self):
        expected_value = 11.170038937560838
        test_value = he.mape(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.mape(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_mapd(self):
        expected_value = 0.09999999999999998
        test_value = he.mapd(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.mapd(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_maape(self):
        expected_value = 0.11083600320216158
        test_value = he.maape(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.maape(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_smape1(self):
        expected_value = 5.630408980871249
        test_value = he.smape1(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.smape1(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_smape2(self):
        expected_value = 11.260817961742498
        test_value = he.smape2(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.smape2(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_d(self):
        expected_value = 0.9789712067292139
        test_value = he.d(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.d(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_d1(self):
        expected_value = 0.8508771929824561
        test_value = he.d1(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.d1(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_dmod(self):
        expected_value = 0.8508771929824561
        test_value = he.dmod(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.dmod(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_drel(self):
        expected_value = 0.9746276298212023
        test_value = he.drel(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.drel(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_dr(self):
        expected_value = 0.853448275862069
        test_value = he.dr(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.dr(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_watt_m(self):
        expected_value = 0.832713182570339
        test_value = he.watt_m(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.watt_m(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_mb_r(self):
        expected_value = 0.7843551797040169
        test_value = he.mb_r(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.mb_r(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_nse(self):
        expected_value = 0.923333988598388
        test_value = he.nse(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.nse(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_nse_mod(self):
        expected_value = 0.706896551724138
        test_value = he.nse_mod(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.nse_mod(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_nse_rel(self):
        expected_value = 0.9074983335293921
        test_value = he.nse_rel(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.nse_rel(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_kge_2009(self):
        expected_value = 0.9181073779138655
        test_value = he.kge_2009(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.kge_2009(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_kge_2012(self):
        expected_value = 0.9132923608280753
        test_value = he.kge_2012(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.kge_2012(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_lm_index(self):
        expected_value = 0.706896551724138
        test_value = he.lm_index(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.lm_index(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_d1_p(self):
        expected_value = 0.8508771929824561
        test_value = he.d1_p(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.d1_p(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_ve(self):
        expected_value = 0.9
        test_value = he.ve(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.ve(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_sa(self):
        expected_value = 0.10732665576112205
        test_value = he.sa(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.sa(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_sc(self):
        expected_value = 0.27804040550591774
        test_value = he.sc(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.sc(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_sid(self):
        expected_value = 0.03429918932223696
        test_value = he.sid(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.sid(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_sga(self):
        expected_value = 0.2645366651790464
        test_value = he.sga(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.sga(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_h1_mhe(self):
        expected_value = 0.006798428591294671
        test_value = he.h1_mhe(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.h1_mhe(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_h1_mahe(self):
        expected_value = 0.11170038937560837
        test_value = he.h1_mahe(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.h1_mahe(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_h1_rmshe(self):
        expected_value = 0.1276017779995636
        test_value = he.h1_rmshe(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.h1_rmshe(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_h2_mhe(self):
        expected_value = -0.010344705046197581
        test_value = he.h2_mhe(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.h2_mhe(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_h2_mahe(self):
        expected_value = 0.11500078970228221
        test_value = he.h2_mahe(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.h2_mahe(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_h2_rmshe(self):
        expected_value = 0.13627318643885672
        test_value = he.h2_rmshe(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.h2_rmshe(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_h3_mhe(self):
        expected_value = -0.001491885359832964
        test_value = he.h3_mhe(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.h3_mhe(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_h3_mahe(self):
        expected_value = 0.11260817961742497
        test_value = he.h3_mahe(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.h3_mahe(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_h3_rmshe(self):
        expected_value = 0.13039562916009131
        test_value = he.h3_rmshe(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.h3_rmshe(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_h4_mhe(self):
        expected_value = -0.0016319199045327479
        test_value = he.h4_mhe(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.h4_mhe(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_h4_mahe(self):
        expected_value = 0.11297850488299188
        test_value = he.h4_mahe(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.h4_mahe(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_h4_rmshe(self):
        expected_value = 0.1309317900186668
        test_value = he.h4_rmshe(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.h4_rmshe(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_h5_mhe(self):
        expected_value = -0.0017731382274514507
        test_value = he.h5_mhe(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.h5_mhe(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_h5_mahe(self):
        expected_value = 0.11335058953894532
        test_value = he.h5_mahe(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.h5_mahe(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_h5_rmshe(self):
        expected_value = 0.13147134893754783
        test_value = he.h5_rmshe(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.h5_rmshe(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_h6_mhe(self):
        expected_value = -0.001491885359832948
        test_value = he.h6_mhe(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.h6_mhe(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_h6_mahe(self):
        expected_value = 0.11260817961742496
        test_value = he.h6_mahe(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.h6_mahe(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_h6_rmshe(self):
        expected_value = 0.1303956291600913
        test_value = he.h6_rmshe(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.h6_rmshe(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_h7_mhe(self):
        expected_value = 0.008498035739118379
        test_value = he.h7_mhe(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.h7_mhe(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_h7_mahe(self):
        expected_value = 0.13962548671951047
        test_value = he.h7_mahe(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.h7_mahe(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_h7_rmshe(self):
        expected_value = 0.1595022224994545
        test_value = he.h7_rmshe(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.h7_rmshe(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_h8_mhe(self):
        expected_value = 0.00582722450682403
        test_value = he.h8_mhe(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.h8_mhe(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_h8_mahe(self):
        expected_value = 0.09574319089337861
        test_value = he.h8_mahe(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.h8_mahe(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_h8_rmshe(self):
        expected_value = 0.1093729525710545
        test_value = he.h8_rmshe(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.h8_rmshe(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_h10_mhe(self):
        expected_value = 0.002961767058151136
        test_value = he.h10_mhe(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.h10_mhe(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_h10_mahe(self):
        expected_value = 0.09041652188064823
        test_value = he.h10_mahe(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.h10_mahe(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_h10_rmshe(self):
        expected_value = 0.10210992896677833
        test_value = he.h10_rmshe(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.h10_rmshe(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_g_mean_diff(self):
        expected_value = 0.9924930879953174
        test_value = he.g_mean_diff(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.g_mean_diff(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

    def test_mean_var(self):
        expected_value = 0.010417665529493766
        test_value = he.mean_var(self.sim, self.obs)
        self.assertTrue(expected_value, test_value)

        test_value_bad_data = he.mean_var(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(expected_value, test_value_bad_data)

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

    def test_ens_me(self):
        expected_value = -2.5217349574908074
        test_value = em.ens_me(obs=self.observed_array, fcst_ens=self.ensemble_array)

        self.assertTrue(np.isclose(expected_value, test_value))

    def test_ens_mae(self):
        expected_value = 26.35428724003365
        test_value = em.ens_mae(obs=self.observed_array, fcst_ens=self.ensemble_array)

        self.assertTrue(np.isclose(expected_value, test_value))

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
        self.merged_df = hd.merge_data(sfpt_url, glofas_url, column_names=['SFPT', 'GLOFAS'])

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
        img_original = mpimg.imread(os.path.join(os.getcwd(), 'Comparison_Files', 'plot_full1.png'))

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
        img_original = mpimg.imread(os.path.join(os.getcwd(), 'Comparison_Files', 'plot_seasonal.png'))

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
        img_original = mpimg.imread(os.path.join(os.getcwd(), 'Comparison_Files', 'hist1.png'))

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
        img_original = mpimg.imread(os.path.join(os.getcwd(), 'Comparison_Files', 'hist1.png'))

        # Comparing the images
        self.assertTrue(np.all(np.isclose(img_test, img_original)))

    def test_hist_znorm(self):
        # Creating test image array
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
        img_original = mpimg.imread(os.path.join(os.getcwd(), 'Comparison_Files', 'hist_znorm.png'))

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
        img_original = mpimg.imread(os.path.join(os.getcwd(), 'Comparison_Files', 'scatter.png'))

        # Comparing the images
        self.assertTrue(np.all(np.isclose(img_test, img_original)))

    def test_scatterlog(self):
        sim_array = self.merged_df.iloc[:, 0].values
        obs_array = self.merged_df.iloc[:, 1].values

        # Creating test image array
        hv.scatter(sim_array=sim_array, obs_array=obs_array, grid=True, title='Scatter Plot (Log-Log Scale)',
                   labels=('SFPT', 'GLOFAS'), line45=True, metrics=['ME', 'KGE (2012)'])
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_test = mpimg.imread(buf)
        buf.close()

        # Reading original image
        img_original = mpimg.imread(os.path.join(os.getcwd(), 'Comparison_Files', 'scatterlog.png'))

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
        img_original = mpimg.imread(os.path.join(os.getcwd(), 'Comparison_Files', 'qqplot.png'))

        # Comparing the images
        self.assertTrue(np.all(np.isclose(img_test, img_original)))

    def tearDown(self):
        del self.merged_df


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])

    unittest.TextTestRunner(verbosity=2).run(suite)
