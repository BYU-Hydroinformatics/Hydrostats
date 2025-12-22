"""Typing aliases."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

InputArray = NDArray[np.floating | np.integer] | Sequence[int | float]
FloatArray = NDArray[np.floating | np.integer]
