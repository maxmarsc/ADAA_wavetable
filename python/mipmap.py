from dataclasses import dataclass
from numba import njit
import numpy as np
import soxr
import logging

from typing import List, Tuple

@dataclass
class MipMapDetails:
    min_pow: int
    max_pow: int

    def generate_scale(self, samplerate: int):
        return np.array([2**idx/samplerate for idx in range(self.min_pow, self.max_pow +1)], dtype=np.float32)

@njit
def mipmap_size(min_pow: int, max_pow: int):
    return max_pow - min_pow + 2

def mipmap_scale(max_size:int, samplerate: float, num: int) -> np.ndarray[float]:
    start = samplerate / max_size * 2 # Empiric
    freqs = np.array([start * 2**i for i in range(num)])
    logging.info("Frequency mipmap scale : {}".format(freqs))
    logging.info("Phase mipmap scale : {}".format(freqs / samplerate))
    return freqs / samplerate


@njit
def find_mipmap_index(phase_diff: float, scale: np.ndarray[float]) -> int:
    return np.searchsorted(scale, phase_diff)

@njit
def find_mipmap_xfading_indexes(phase_diff: float, scale: np.ndarray[float]) -> Tuple[int, float, int, float]:
    THRESHOLD_FACTOR = 0.98
    mipmap_idx = np.searchsorted(scale, phase_diff)

    # Reached last index, can't cross fade
    if mipmap_idx == scale.shape[0]:
        return (mipmap_idx, 1.0, None, 0.0)
    
    threshold = scale[mipmap_idx] * (1.0 + THRESHOLD_FACTOR) / 2
    
    if phase_diff < threshold:
        # Below threshold, we don't crossfade
        return (mipmap_idx, 1.0, mipmap_idx + 1, 0.0)
    else:
        # Above threshold, starting crossfade
        a = 1.0 / (scale[mipmap_idx] - threshold)
        b = - threshold * a
        factor_next = a * phase_diff + b
        factor_crt = 1.0 - factor_next
        return (mipmap_idx, factor_crt, mipmap_idx+1, factor_next) 



def compute_mipmap_waveform(waveform: np.ndarray[float], count: int) -> List[np.ndarray[float]]:
    return [
        soxr.resample(waveform, 2**i, 1, quality="VHQ") for i in range(count)
    ]

if __name__ == "__main__":
    scale = mipmap_scale(2048, 44100, 5)

    # a = find_mipmap_index_xfading(scale[0], scale)
    # for i in range(5):
    #     print("{} {}".format(i, find_mipmap_index_xfading(scale[i], scale)))
    # print("\n\n===")
    for factor in (1.0, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.999):
        b = find_mipmap_xfading_indexes(scale[0]*factor, scale)
        print(factor, b)