.. hydrostats documentation master file, created by
   sphinx-quickstart on Thu Jul 19 13:18:58 2018. * :ref:`modindex`


Hydrostats Documentation
==========================
Hydrostats is a library of tools and functions for users working with time series data, with some tools specific to the
field of hydrology (hence **Hydro**\stats). All of the tools contained in Hydrostats are built using a few different
python libraries including numpy, scipy, pandas, matplotlib, and numba. It is meant to provide a high-level interface
for users to be able to perform common tasks regarding time series analysis.

Hydrostats contains tools for for preprocessing data, visualizing data, calculating error metrics on observed and
predicted time series, and forecast validation. It contains over 70 error metrics, with many metrics specific to the field of
hydrology.

See the examples folder in this repository for a Jupyter notebook highlighting some of the main features of Hydrostats.

.. image:: https://img.shields.io/badge/powered%20by-BYU%20HydroInformatics-blue.svg
    :target: http://worldwater.byu.edu/
.. image:: https://travis-ci.org/BYU-Hydroinformatics/Hydrostats.svg?branch=master
    :target: https://travis-ci.org/BYU-Hydroinformatics/Hydrostats
.. image:: https://codecov.io/gh/BYU-Hydroinformatics/Hydrostats/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/BYU-Hydroinformatics/Hydrostats

Contents
========

.. toctree::
   :maxdepth: 1

   Installation
   Preprocessing
   Visualization
   Metrics
   ens_metrics
   Analysis
   ref_table
   plot_types
   timezones
   matplotlib_linestyles
   change_log

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`