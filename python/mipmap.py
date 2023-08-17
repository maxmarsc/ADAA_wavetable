from dataclasses import dataclass
from numba import njit
import numpy as np
import soxr

from typing import List

@dataclass
class MipMapDetails:
    min_pow: int
    max_pow: int

    def generate_scale(self, samplerate: int):
        return np.array([2**idx/samplerate for idx in range(self.min_pow, self.max_pow +1)], dtype=np.float32)


@njit
def mipmap_size(min_pow: int, max_pow: int):
    return max_pow - min_pow + 2

@njit
def mipmap_scale(max_size:int, samplerate: float, num: int) -> np.ndarray[float]:
    # start = samplerate / max_size * 1.3 # Empiric
    start = samplerate / max_size * 1.3 # Empiric
    freqs = np.array([start * 2**i for i in range(num)])
    print("Frequency mipmap scale :", freqs)
    return freqs / samplerate
    # return np.array([2**idx/samplerate for idx in range(min_pow, max_pow +1)], dtype=np.float32)


@njit
def find_mipmap_index(phase_diff: float, scale: np.ndarray[float]) -> int:
    return np.searchsorted(scale, phase_diff)
    # for i, threshold in enumerate(scale):
    #     if phase_diff >= threshold:
    #         return i
    # return len(scale)

def compute_mipmap_waveform(waveform: np.ndarray[float], count: int) -> List[np.ndarray[float]]:
    return [
        soxr.resample(waveform, 2**i, 1, quality="VHQ") for i in range(count)
    ]

