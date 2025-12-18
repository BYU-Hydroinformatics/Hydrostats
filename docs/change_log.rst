Release Notes
=============

This is the list of changes to Hydrostats between each release. For full details, see the commit logs at
https://github.com/BYU-Hydroinformatics/Hydrostats.

Version 2.0.0rc1
^^^^^^^^^^^^^^^^

Breaking Changes:

- Drop support for Python 2.7, 3.6 and 3.7. Minimum Python version is now 3.10.
- Raise a ValueError instead of a RuntimeError for bad input in the following functions:
   - analyze.time_lag
   - metrics.list_of_metrics
   - ens_metrics.ens_me
   - ens_metrics.ens_mae
   - ens_metrics.ens_mse
   - ens_metrics.ens_rmse
   - ens_metrics.ens_pearson_r
   - ens_metrics.skill_score
   - ens_metrics.treat_data

Other Changes:

- Add type hints throughout the codebase for better developer experience.
- More modern documentation theme (Furo) for better readability.
- Add ruff as a linter to the development workflow for improved code quality.

Version 0.78
^^^^^^^^^^^^
- Added the ability to use different thresholds for the ensemble forecast for the observed and ensemble forecast data in
  the hydrostats.ens_metrics.auroc() and hydrostats.ens_metrics.ens_brier() methods.
- Changes to documentation to reflect the addition of the .name and .abbr properties to metrics from the HydroErr
  package.

Version 0.77
^^^^^^^^^^^^
- Added a new rolling average feature to the hydrostats.data.daily_average(). Set rolling=True as a parameter to use the
  defaults, or specify arguments from the pandas.DataFrame.rolling() method for a custom rolling average.
- Minor changes and more coverage.

Version 0.76
^^^^^^^^^^^^
- Moved the documentation to new location at https://hydrostats.readthedocs.io/

Version 0.75
^^^^^^^^^^^^
- Minor bug fixes and changes

Version 0.74
^^^^^^^^^^^^

- Added support for parsing julian dates with the new hydrostats.data.julian_to_gregorian() function
- Added support for parsing files with julian dates in the hydrostats.data.merge_data() function.
- Added example code in the github repository, in the "Examples" directory.
