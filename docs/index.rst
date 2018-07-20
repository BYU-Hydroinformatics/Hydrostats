.. hydrostats documentation master file, created by
   sphinx-quickstart on Thu Jul 19 13:18:58 2018. * :ref:`modindex`


Welcome to hydrostats's documentation!
======================================
Hydrostats is a library of tools and functions for users working with timeseries data. It has tools
for preprocessing data,

.. toctree::
   :maxdepth: 2


Installation
============
Hydrostats is freely available on the Python Package index repository (PyPI). It can be installed with the
following command using either virtualenv or Anaconda.

.. code-block:: python

   pip install hydrostats

.. code-block:: error

   **RuntimeError**: Python is not installed as a framework. The Mac OS X backend will not be able
   to function correctly if Python is not installed as a framework. See the Python documentation for
   more information on installing Python as a framework on Mac OS X. Please either reinstall Python
   as a framework, or try one of the other backends.

hydrostats package
==================

.. automodapi:: hydrostats.analyze

.. automodapi:: hydrostats.data

.. automodapi:: hydrostats.metrics

.. automodapi:: hydrostats.test

.. automodapi:: hydrostats.visual
