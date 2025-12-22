"""A library of tools and functions for users working with time series data, with some tools
specific to the field of hydrology.

All the tools contained in Hydrostats are built using a few different python libraries, including
numpy, scipy, pandas, matplotlib, and numba. It is meant to provide a high-level interface for users
to be able to perform common tasks regarding time series analysis.

Hydrostats contains tools for preprocessing data, visualizing data, calculating error metrics on
observed and predicted time series, and forecast validation. It contains over 70 error metrics, with
many metrics specific to the field of hydrology.

For full documentation, see https://hydrostats.readthedocs.io/en/stable/
"""  # noqa: D205

from hydrostats.analyze import *  # noqa: F403
from hydrostats.metrics import *  # noqa: F403

__version__ = "1.0.0"
