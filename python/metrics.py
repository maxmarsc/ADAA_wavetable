import numpy as np
from math import ceil, floor
from numba import njit
from typing import Tuple, List


@njit
def find_peak_bin_from_freq(
    freq_hint: float, ps: np.ndarray[float], sr: float, search_width_hz=10
) -> int:
    """Find the bin of the highest point of a frequency peak

    Args:
        freq_hint (float): The expected frequency of the peak
        ps (np.ndarray[float]): The periodigram
        sr (float): samplerate
        search_width_hz (int, optional): Width of the search area for the peak. Defaults to 10.

    Returns:
        int: The index of the bin
    """
    num_bins = ps.shape[0]
    hint_idx = floor(2.0 * freq_hint / sr * (num_bins - 1))
    search_width = ceil(search_width_hz / (sr / 2.0) * (num_bins - 1))
    left_limit = hint_idx - search_width
    search_center = search_width
    if left_limit < 0:
        search_center += left_limit
        left_limit = 0
    right_limit = min(num_bins, hint_idx + search_width + 1)
    search_slice = ps[left_limit:right_limit]

    return (
        find_nearest_peak_around(search_slice, search_center, search_width)
        + hint_idx
        - search_width
    )


@njit
def is_peak(slice_of_3: np.ndarray[float]) -> bool:
    return slice_of_3[1] > slice_of_3[0] and slice_of_3[1] > slice_of_3[2]


@njit
def find_nearest_peak_around(
    ps_slice: np.ndarray[float], center: int, search_width: int
) -> int:
    """Find the nearest peak to the center

    Args:
        ps_slice (np.ndarray[float]): The slice in which to search
        center (int): The index of the center of the search
        search_width (int): The width of the search

    Returns:
        int: The index of the bin
    """
    center = ps_slice.shape[0] // 2
    for i in range(0, search_width):
        # right side first
        right_idx = center + i
        if right_idx + 1 < ps_slice.shape[0] and is_peak(
            ps_slice[right_idx - 1 : right_idx + 2]
        ):
            return right_idx

        # left side then
        left_idx = center - i
        if left_idx - 1 >= 0 and is_peak(ps_slice[left_idx - 1 : left_idx + 2]):
            return left_idx

    # fallback
    return np.argmax(ps_slice)


@njit
def find_peak_bins(
    peak_idx: int, ps: np.ndarray[float], search_width=1000
) -> np.ndarray[int]:
    """Given a peak, this will find all the bins of the peak

    Args:
        peak_idx (int): the index of the highest point of the peak
        ps (np.ndarray[float]): the periodigram
        search_width (int, optional): The width (in number of bins) for the search. Defaults to 1000.

    Returns:
        np.ndarray[int]: An array of all the bins of the peak, ordered
    """
    peak_bins = []
    peak = ps[peak_idx]

    # Find values before peak
    crt_val = peak
    for i in range(search_width):
        new_bin = peak_idx - i
        if new_bin < 0:
            break
        new_val = ps[new_bin]
        if new_val > crt_val:
            break
        peak_bins.append(new_bin)
        crt_val = new_val

    # Find values after peak
    crt_val = peak
    for i in range(search_width):
        new_bin = peak_idx + i
        if new_bin >= ps.shape[0]:
            break
        new_val = ps[new_bin]
        if new_val > crt_val:
            break
        peak_bins.append(new_bin)
        crt_val = new_val

    return np.array(peak_bins)


@njit
def compute_harmonic_factors(fund: float, sr: float) -> np.ndarray[int]:
    nyquist = sr / 2.0
    return np.arange(2, floor(nyquist / fund) + 1)


def snr(fundamental: float, psd: np.ndarray[float], sr: float) -> float:
    """SNR computation method. Tailored for saw wave (it will include all the harmonics)

    Unlike matlab's SNR method, this does not assume the input is a sinusoïd

    Args:
        fundamental (float): The fundamental frequency of the signal
        psd (np.ndarray[float]): The periodigram of the signal
        sr (float): The samplerate of the signal

    Returns:
        float: SNR in dB
    """
    harmonics_factors = compute_harmonic_factors(fundamental, sr)

    fund_bin = find_peak_bin_from_freq(fundamental, psd, sr)
    fund_peak_bins = find_peak_bins(fund_bin, psd)

    harmonics_bins = []
    for harmonic_factor in harmonics_factors:
        harmonic = harmonic_factor * fundamental
        harmonic_bin = find_peak_bin_from_freq(harmonic, psd, sr)
        harmonics_bins.extend(find_peak_bins(harmonic_bin, psd))
    harmonics_bins = np.array(harmonics_bins, dtype=int)
    signal_bins = np.concatenate((fund_peak_bins, harmonics_bins))

    signal_power = np.sum(psd[signal_bins])

    noise = np.delete(psd, signal_bins)
    noise_power = np.sum(noise[1:])

    return 10 * np.log10(signal_power / noise_power)


def matlab_snr(fundamental: float, psd: np.ndarray[float], sr: float) -> float:
    """Matlab-like SNR computation method.

    Like matlab's SNR method, this does assume the input is a sinusoïd. By doing
    so it exclude all the harmonics from the signal part. Harmonics are also excluded
    from the noise part

    Args:
        fundamental (float): The fundamental frequency of the signal
        psd (np.ndarray[float]): The periodigram of the signal
        sr (float): The samplerate of the signal

    Returns:
        float: SNR in dB
    """
    harmonics_factors = compute_harmonic_factors(fundamental, sr)

    fund_bin = find_peak_bin_from_freq(fundamental, psd, sr)

    fund_peak_bins = find_peak_bins(fund_bin, psd)

    harmonics_bins = []
    for harmonic_factor in harmonics_factors:
        harmonic = harmonic_factor * fundamental
        harmonic_bin = find_peak_bin_from_freq(harmonic, psd, sr)
        harmonics_bins.extend(find_peak_bins(harmonic_bin, psd))
    harmonics_bins = np.array(harmonics_bins, dtype=int)

    fund_power = np.sum(psd[fund_peak_bins[0] : fund_peak_bins[-1]])

    to_remove = np.concatenate((fund_peak_bins, harmonics_bins))
    noise = np.delete(psd, to_remove)
    noise_power = np.sum(noise[1:])

    return 10 * np.log10(fund_power / noise_power)


def sinad(
    fundamental: float,
    mag_clean: np.ndarray[float],
    mag_noised: np.ndarray[float],
    sr: float,
    max_harmonics: int,
) -> float:
    """SINAD computation method. Tailored for saw wave (it will include all the harmonics)

    The signals will be normalized according to the energy in the fundamental peak

    Args:
        fundamental (float): The fundamental frequency of the signal
        mag_clean (np.ndarray[float]): The magnitudes of the clean signal (no aliasing)
        mag_noised (np.ndarray[float]): The magnitudes of the noisy signal (aliased)
        sr (float): The samplerate
        max_harmonics (int): The maximum number of harmonics to include in the signal part

    Returns:
        float: The SINAD in dB
    """
    # Normalize according to the magnitude of the fundamental :
    fund_bin = find_peak_bin_from_freq(fundamental, mag_noised, sr)
    fund_peak_bins = find_peak_bins(fund_bin, mag_noised)

    mag_sum_noised = np.sum(mag_noised[fund_peak_bins[0] : fund_peak_bins[-1]])
    mag_sum_clean = np.sum(mag_clean[fund_peak_bins[0] : fund_peak_bins[-1]])
    mag_ratio = mag_sum_noised / mag_sum_clean
    mag_noised *= mag_ratio
    mag_noise = mag_noised - mag_clean

    # Compute the PSD after normalization
    psd_clean = np.square(mag_clean)
    psd_noise = np.square(mag_noise)
    psd_noised = np.square(mag_noised)

    # compute and find the harmonics
    harmonics_factors = compute_harmonic_factors(fundamental, sr)[:max_harmonics]
    harmonics_bins = []
    for harmonic_factor in harmonics_factors:
        harmonic = harmonic_factor * fundamental
        harmonic_bin = find_peak_bin_from_freq(harmonic, psd_clean, sr)
        harmonics_bins.extend(find_peak_bins(harmonic_bin, psd_noised))
    harmonics_bins = np.array(harmonics_bins, dtype=int)
    signal_bins = np.concatenate((fund_peak_bins, harmonics_bins))
    signal_power = np.sum(psd_clean[signal_bins])

    nad = np.delete(psd_noise, fund_peak_bins)
    nad_power = np.sum(nad[1:])

    return 10 * np.log10((signal_power + nad_power) / nad_power)
