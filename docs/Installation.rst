Installation
============

Hydrostats is available on the Python Package Index (PyPI) and Conda Forge. It can be installed
using various package managers.

Using pip
---------

Install using pip::

    pip install hydrostats

Using uv
--------

Install using uv (recommended for faster installs)::

    uv add hydrostats

Using conda
-----------

Install via conda-forge::

    conda install -c conda-forge hydrostats


macOS Matplotlib Backend Issue
-------------------------------

On macOS, you may encounter this error::

   **RuntimeError**: Python is not installed as a framework. The Mac OS X backend will not be able
   to function correctly if Python is not installed as a framework.

To fix this, configure a different matplotlib backend by creating a configuration file::

   mkdir -p ~/.matplotlib
   echo "backend: TkAgg" > ~/.matplotlib/matplotlibrc

Alternatively, you can use any supported backend (TkAgg, Qt5Agg, etc.).

For more information about matplotlib backends, see the `matplotlib documentation`_.

.. _matplotlib documentation: https://matplotlib.org/stable/users/explain/figure/backends.html
