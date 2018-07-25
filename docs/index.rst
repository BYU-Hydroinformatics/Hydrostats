.. hydrostats documentation master file, created by
   sphinx-quickstart on Thu Jul 19 13:18:58 2018. * :ref:`modindex`


Welcome to hydrostats's documentation!
======================================
Hydrostats is a library of tools and functions for users working with timeseries data. It has tools
for preprocessing data, visualizing data, and for calculating error metrics on observed and
predicted time series. It has over 60 error metrics, with many metrics specific to the field of
hydrology.

.. toctree::
   :maxdepth: 2


Installation
============
Hydrostats is freely available on the Python Package index repository (PyPI). It can be installed
with the following command using either virtualenv or Anaconda.

.. code-block:: shell

   pip install hydrostats

When installing hydrostats on Mac OSX operating system, you may get the following error when trying
to run python scripts using hydrostats in an IDE such as PyCharm.

.. code-block:: shell

   **RuntimeError**: Python is not installed as a framework. The Mac OS X backend will not be able
   to function correctly if Python is not installed as a framework. See the Python documentation for
   more information on installing Python as a framework on Mac OS X. Please either reinstall Python
   as a framework, or try one of the other backends.

If this happens, you will need to manually change the backend for use in your IDE. This can be done
by running the following commands in the terminal:

.. code-block:: shell

   sudo apt install nano # If you have not already installed it
   vi ~/.matplotlib/matplotlibrc # Create a file called matplotlibrc

Add the following code to the file you created (hit i to insert text in vi editor):

.. code-block:: shell

   backend: TkAgg

Then close the text document with esc, :wq.

For more information about backends, read here_

.. _here: https://matplotlib.org/faq/usage_faq.html#what-is-a-backend

hydrostats package
==================

The hydrostats package is divided into different modules that contain different functions for users
to analyze time series data. All of the modules are listed below, with their respective functions.
Each function has a link to it's respective documentation page.

.. automodapi:: hydrostats.analyze

.. automodapi:: hydrostats.data

.. automodapi:: hydrostats.metrics

.. automodapi:: hydrostats.visual
