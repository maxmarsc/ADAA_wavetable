import numpy as np
from math import floor
from numba import njit


@njit
def compute_harmonic_factors(fund: float, sr: float, max: int = -1) -> np.ndarray[int]:
    nyquist = sr / 2.0
    return np.arange(2, floor(nyquist / fund) + 1)[:max]
