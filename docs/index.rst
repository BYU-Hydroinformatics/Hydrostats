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

.. image:: https://github.com/BYU-Hydroinformatics/Hydrostats/actions/workflows/ci.yml/badge.svg?branch=master
    :target: https://github.com/BYU-Hydroinformatics/Hydrostats/actions/workflows/ci.yml?query=branch:master
.. image:: https://img.shields.io/pypi/v/hydrostats.svg
    :target: https://pypi.python.org/pypi/hydrostats
.. image:: https://img.shields.io/pypi/l/hydrostats.svg
    :target: https://github.com/BYU-Hydroinformatics/Hydrostats/blob/master/LICENSE.txt
.. image:: https://img.shields.io/pypi/pyversions/hydrostats.svg
    :target: https://pypi.python.org/pypi/hydrostats
.. image:: https://img.shields.io/badge/powered%20by-BYU%20HydroInformatics-blue.svg
    :target: http://worldwater.byu.edu/

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
   contribute

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
