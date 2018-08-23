import hydrostats.HydroErr as he
import hydrostats.ens_metrics as em
import hydrostats.analyze as ha
import unittest as test
import doctest
import numpy as np
import warnings


class MetricsTests(test.TestCase):

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


if __name__ == "__main__":
    suite = test.TestLoader().loadTestsFromTestCase(MetricsTests)
    suite.addTest(doctest.DocTestSuite(he))
    suite.addTest(doctest.DocTestSuite(em))
    suite.addTest(doctest.DocTestSuite(ha))
    test.TextTestRunner(verbosity=2).run(suite)
