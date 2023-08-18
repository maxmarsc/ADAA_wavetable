
from collections import defaultdict
import numpy as np
from math import exp, floor
from cmath import exp as cexp
from scipy.signal import welch, butter, cheby2, residue, freqs, zpk2tf, windows, spectrogram
import matplotlib.pyplot as plt
import matplotlib
import soxr
import soundfile as sf
import csv
from pathlib import Path
import argparse
from dataclasses import dataclass
from multiprocessing import Pool
import logging
from numba import jit, njit

from typing import Tuple, List, Dict
from numba.typed import List as NumbaList


from bl_waveform import bl_sawtooth
from decimator import Decimator17, Decimator9
from waveform import FileWaveform, NaiveWaveform
from mipmap import *

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

WAVEFORM_LEN = 4096
SAMPLERATE = 44100
BUTTERWORTH_CTF = 0.45 * SAMPLERATE
CHEBY_CTF = 0.61 * SAMPLERATE
DURATION_S = 1.0
NUM_PROCESS = 19
# CSV_OUTPUT = "benchmark.csv"

matplotlib.use('TkAgg')
# logging.info("Starting matlab")
# MATLAB = matlab.engine.start_matlab()
# logging.info("matlab started")

@dataclass
class MipMapAdaaCache:
    m_mipmap: List[np.ndarray[float]]
    q_mipmap : List[np.ndarray[float]]
    m_diff_mipmap: List[np.ndarray[float]]
    q_diff_mipmap : List[np.ndarray[float]]
    mipmap_scale: np.ndarray[float]

@dataclass
class AdaaCache:
    m: np.ndarray[float]
    q : np.ndarray[float]
    m_diff: np.ndarray[float]
    q_diff : np.ndarray[float]

@dataclass
class NaiveCache:
    waveform: np.ndarray[float]

@njit
def compute_naive_saw(frames: int) -> np.ndarray:
    phase = 0.0
    waveform = np.zeros(frames)
    step = 1.0/frames

    for i in range(frames):
        waveform[i] = 2.0 * phase - 1
        phase = (phase + step) % 1.0

    return waveform


def compute_naive_sin(frames: int) -> np.ndarray:
    phase = np.linspace(0, 2 * np.pi, frames, endpoint=False)
    return np.sin(phase)

@njit
def noteToFreq(note: int) -> float:
    a = 440 #frequency of A (coomon value is 440Hz)
    return (a / 32) * (2 ** ((note - 9) / 12))

NAIVE_SAW = compute_naive_saw(WAVEFORM_LEN)
NAIVE_SAW_X = np.linspace(0, 1, WAVEFORM_LEN + 1, endpoint=True)


def butter_coeffs(order, ctf, samplerate) -> Tuple[np.ndarray[complex], np.ndarray[complex]]:
    """
    Computes butterworth filter coeffs like in the matlab code
    """
    ang_freq = 2 * np.pi * ctf / samplerate
    (z, p, k) = butter(order, ang_freq, output="zpk", analog=True)
    (b, a) = zpk2tf(z, p, k)
    (r, p, _) =  residue(b, a)
    return (r, p, (b,a))

def cheby_coeffs(order, ctf, attn, samplerate) -> Tuple[np.ndarray[complex], np.ndarray[complex]]:
    """
    Computes chebyshev type 2 filter coeffs like in the matlab code
    """
    ang_freq = 2 * np.pi * ctf /samplerate
    (z, p, k) = cheby2(order, attn, ang_freq, output="zpk", analog=True)
    (b, a) = zpk2tf(z, p, k)
    (r, p, _) =  residue(b, a)
    return (r, p, (b,a))

@njit
def compute_m(x0: float, x1: float, y0: float, y1: float) -> float:
    return (y1 - y0) / (x1 - x0)

@njit
def compute_q(x0: float, x1: float, y0: float, y1: float) -> float:
    return (y0 * (x1 - x0) - x0 * (y1 - y0)) / (x1 - x0)

@njit
def compute_m_q_vectors(waveform: np.ndarray):
    """
    Compute the m & q vectors needed by the paper algorithm. In the paper they have waveformq they know the shape of in 
    advance. This function allows you to compute m & q for any kind of waveform

    Notes that when the slope is too big (threshold is empiric), this replicate the previous m & q to mimic the
    m & q definitions found in the paper (ie : in the paper they do have non-linearities in m & q).
    """
    size = waveform.shape[0]
    # slope_thrsd = size / 2
    m = np.zeros(size)
    q = np.zeros(size)
    X = np.linspace(0, 1, waveform.shape[0] + 1)

    # idx_to_estimate = []

    for i in range(size - 1):
        y0 = waveform[i]
        y1 = waveform[i+1]
        x0 = X[i]
        x1 = X[i+1]
        m_i = compute_m(x0, x1, y0, y1)
        q_i = compute_q(x0, x1, y0, y1)

        # if abs(m_i) > slope_thrsd:
        #     print("Exceeded threshold at idx {} of waveform {}".format(i, waveform.shape[0]))
        #     idx_to_estimate.append(i)
        #     continue

        m[i] = m_i
        q[i] = q_i

    m[-1] = np.single(compute_m(np.double(X[-2]), np.double(X[-1]), np.double(waveform[-1]), np.double(waveform[0])))
    q[-1] = np.single(compute_q(np.double(X[-2]), np.double(X[-1]), np.double(waveform[-1]), np.double(waveform[0])))
    # if abs(m[-1]) > slope_thrsd:
    #     j = size-2
    #     while j in idx_to_estimate:
    #         j -= 1
    #     m[-1] = m[j]
    #     q[-1] = q[j]

    # for i in idx_to_estimate:
    #     m[i] = m[i-1]
    #     q[i] = q[i-1]

    
    return (m, q)

@njit
def binary_search_down(x : np.ndarray, x0: float, j_min: int, j_max: int) -> int:
    """
    return i as x_i < x_0 < x_(i+1) && j_min <= i <= j_max
    """
    if x0 < x[0]:
        return -1           # Should it be -1 ? 0 in matlab so it's weird
    elif x0 >= x[j_max]:
        return j_max - 1
    else:
        i_mid = floor((j_min + j_max)/2)

        if x0 < x[i_mid]:
            j_max = i_mid
        elif x0 == x[i_mid]:
            return i_mid
        else:
            j_min = i_mid
        
        if j_max - j_min > 1:
            return binary_search_down(x, x0, j_min, j_max)
        else:
            return j_min

@njit
def binary_search_up(x : np.ndarray, x0: float, j_min: int, j_max: int):
    """
    return i as x_i > x_0 > x_(i+1) && j_min <= i <= j_max
    """
    if x0 >= x[-1]:
        return x.shape[0]
    elif x0 <= x[0]:
        return 0
    else:
        i_mid = floor((j_min + j_max)/2)

        if x0 < x[i_mid]:
            j_max = i_mid
        elif x0 == x[i_mid]:
            return i_mid

        if j_max - j_min > 1:
            return binary_search_up(x, x0, j_min, j_max)
        else:
            return j_max

@njit
def process_naive_linear(waveform, x_values):
    """
    Linear interpolation algorithm
    """
    y = np.zeros(x_values.shape[0])
    waveform_len = waveform.shape[0]

    for (i, x) in enumerate(x_values):
        if i == 0:
            continue

        x_red = x % 1.0
        relative_idx = x_red * waveform_len

        prev_idx = floor(relative_idx)
        next_idx = (prev_idx + 1) % waveform_len

        if relative_idx == prev_idx:
            y[i] = waveform[prev_idx]
        else:
            a = (waveform[next_idx] - waveform[prev_idx]) / (next_idx - prev_idx)
            b = waveform[prev_idx] - prev_idx * a
            y[i] = a * relative_idx + b  
    
    return y

def snr(noised_signal_fft, perfect_signal_fft):
    magnitude_noise = np.abs(noised_signal_fft) - np.abs(perfect_signal_fft)
    magnitude_noise[magnitude_noise< 0.0] = 0.0
    noise_rms = np.sqrt(np.mean(magnitude_noise**2))
    signal_rms =  np.sqrt(np.mean(np.abs(perfect_signal_fft)**2))
    return 10*np.log10(signal_rms/noise_rms)

# @njit
def normalized_fft(time_signal, padding: int=0) -> np.ndarray:
    signal_len = time_signal.shape[0]

    # Pad with zeros
    padding_len = signal_len + padding * 2
    if padding != 0:
        padded_signal = np.zeros(padding_len)
        padded_signal[padding:-padding] = time_signal
        time_signal = padded_signal

    window = np.blackman(padding_len)
    fft = np.fft.rfft(time_signal * window)
    return fft / np.max(np.abs(fft))

def downsample_x2_decim17(y_2x: np.ndarray[float]) -> np.ndarray[float]:
    """
    2x decimate with a 17 residuals coefficient decimator
    """
    new_size = floor(y_2x.shape[0] / 2)
    y = np.zeros(new_size)
    decimator = Decimator17()

    for i in range(new_size):
        y[i] = decimator.process(y_2x[i*2], y_2x[i*2 + 1])

    return y

def downsample_x2_decim9(y_2x: np.ndarray[float]) -> np.ndarray[float]:
    """
    2x decimate with a 9 residuals coefficient decimator
    """
    new_size = floor(y_2x.shape[0] / 2)
    y = np.zeros(new_size)
    decimator = Decimator9()

    for i in range(new_size):
        y[i] = decimator.process(y_2x[i*2], y_2x[i*2 + 1])

    return y

def downsample_x4_decim9_17(y_4x: np.ndarray[float]) -> np.ndarray[float]:
    """
    4x decimate with two consecutives x2 decimators, first a 9 then a 17
    """
    new_size = floor(y_4x.shape[0] / 4)
    y = np.zeros(new_size)
    decimator_a = Decimator9()
    decimator_b = Decimator17()

    for i in range(new_size):
        a = decimator_a.process(y_4x[i*4], y_4x[i*4+1])
        b = decimator_a.process(y_4x[i*4+2], y_4x[i*4+3])
        y[i] = decimator_b.process(a, b)

    return y

@njit
def mq_from_waveform(waveform: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (m, q) =  compute_m_q_vectors(waveform)
    m_diff = np.zeros(m.shape[0])
    q_diff = np.zeros(q.shape[0])
    
    for i in range(m.shape[0] - 1):
        m_diff[i] = m[i+1] - m[i]
        q_diff[i] = q[i+1] - q[i]
    m_diff[-1] = m[0] - m[-1]
    q_diff[-1] = q[0] - q[-1] - m[0]

    return (m, q, m_diff, q_diff)

def mipmap_mq_from_waveform(waveforms: List[np.ndarray[float]]) -> Tuple[List[np.ndarray[float]], List[np.ndarray[float]], List[np.ndarray[float]], List[np.ndarray[float]]] :
    # ret = (list() for _ in range(4))
    m_list = []
    q_list = []
    m_diff_list = []
    q_diff_list = []

    for waveform in waveforms:
        m, q, m_diff, q_diff = mq_from_waveform(waveform)
        m_list.append(m)
        q_list.append(q)
        m_diff_list.append(m_diff)
        q_diff_list.append(q_diff)

    return (m_list, q_list, m_diff_list, q_diff_list)


def compute_thd(noised_signal, clean_signal) -> float:
    noised_fft = normalized_fft(noised_signal)
    clean_fft = normalized_fft(clean_signal)

    return snr(noised_fft, clean_fft)

# @njit
def compute_sinad(noised_signal: np.ndarray, clean_signal: np.ndarray, fundamental: float) -> float:
    assert(noised_signal.shape == clean_signal.shape)
    noised_fft = normalized_fft(noised_signal)
    clean_fft = normalized_fft(clean_signal)

    fft_size = noised_fft.shape[0]
    bin_size = SAMPLERATE / 2 / (fft_size - 1)

    # Expected bin for the fundamental frequency
    expected_fundamental_bin = int(np.round(fundamental / bin_size))

    # Search around the expected bin for the true peak
    search_range = 5  # 5 bins on either side
    fundamental_bin = expected_fundamental_bin + np.argmax(np.abs(clean_fft[expected_fundamental_bin - search_range:expected_fundamental_bin + search_range + 1])) - search_range

    # Check fundamental frequency
    val = np.argmax(np.abs(clean_fft))
    assert(val == fundamental_bin)

    # Compute Harmonic Distortion
    harmonics = [i * fundamental_bin for i in range(1, fft_size // fundamental_bin)]
    hd_power = np.sum([(np.abs(noised_fft[h]) - np.abs(clean_fft[h]))**2 for h in harmonics])
    
    # Compute Noise
    noise_bins = [i for i in range(len(noised_fft)//2) if i not in harmonics and i != fundamental_bin]
    noise_power = np.sum([(np.abs(noised_fft[n]) - np.abs(clean_fft[n]))**2 for n in noise_bins])

    # Compute SINAD
    signal_power = np.abs(clean_fft[fundamental_bin])**2
    sinad_value = signal_power / (hd_power + noise_power)
    sinad_db = 10 * np.log10(sinad_value)

    return sinad_db

@njit
def fast_compute_sinad(noised_fft: np.ndarray, clean_fft: np.ndarray, fundamental: float) -> float:
    assert(noised_fft.shape == clean_fft.shape)

    fft_size = noised_fft.shape[0]
    bin_size = SAMPLERATE / 2 / (fft_size - 1)

    # Expected bin for the fundamental frequency
    expected_fundamental_bin = int(np.round(fundamental / bin_size))

    # Search around the expected bin for the true peak
    search_range = 5  # 5 bins on either side
    fundamental_bin = expected_fundamental_bin + np.argmax(np.abs(clean_fft[expected_fundamental_bin - search_range:expected_fundamental_bin + search_range + 1])) - search_range

    # Check fundamental frequency
    val = np.argmax(np.abs(clean_fft))
    assert(val == fundamental_bin)

    # Compute Harmonic Distortion
    harmonics = [i * fundamental_bin for i in range(1, fft_size // fundamental_bin)]
    hd_power = np.sum(np.array([(np.abs(noised_fft[h]) - np.abs(clean_fft[h]))**2 for h in harmonics]))
    
    # Compute Noise
    noise_bins = [i for i in range(len(noised_fft)//2) if i not in harmonics and i != fundamental_bin]
    noise_power = np.sum(np.array([(np.abs(noised_fft[n]) - np.abs(clean_fft[n]))**2 for n in noise_bins]))

    # Compute SINAD
    signal_power = np.abs(clean_fft[fundamental_bin])**2
    sinad_value = signal_power / (hd_power + noise_power)
    sinad_db = 10 * np.log10(sinad_value)

    return sinad_db

from enum import Enum
class Algorithm(Enum):
    ADAA_BUTTERWORTH = 1
    ADAA_CHEBYSHEV_TYPE2 = 2
    NAIVE = 3

def process_adaa(x: np.ndarray, cache: AdaaCache, ftype: Algorithm,
                 forder:int, os_factor: int) -> Tuple[np.ndarray, str]:
    sr = SAMPLERATE * os_factor
    waveform_len = cache.m.shape[0]
    X = np.linspace(0, 1, waveform_len + 1, endpoint=True)
    


    assert(ftype != Algorithm.NAIVE)
    if ftype == Algorithm.ADAA_BUTTERWORTH:
        (r, p, _) = butter_coeffs(forder, BUTTERWORTH_CTF, sr)
        fname = "BT"
    else:
        (r, p, _) = cheby_coeffs(forder, CHEBY_CTF, 60, sr)
        fname = "CH"

    y = np.zeros(x.shape[0])

    for order in range(0, forder, 2):
        ri = r[order]
        zi = p[order]
        y += process_fwd(x, ri, zi, X, cache.m, cache.q, cache.m_diff, cache.q_diff)

    if os_factor == 2:
        ds_size = int(y.shape[0] / os_factor)
        y_ds = np.zeros((ds_size,))
        decimator = Decimator9()
        for i in range(ds_size):
            y_ds[i] = decimator.process(y[i*2], y[i*2 + 1])
        y = y_ds
    elif os_factor != 1:
        # Downsampling
        y = soxr.resample(
            y,
            SAMPLERATE * os_factor,
            SAMPLERATE,
            quality="HQ"
        )
    # if os_factor != 1:
    #     # Downsampling
    #     y = soxr.resample(
    #         y,
    #         SAMPLERATE * os_factor,
    #         SAMPLERATE,
    #         quality="HQ"
    #     )


    name = "ADAA {} order {}".format(fname, forder)
    if os_factor != 1:
        name += "OVSx{}".format(os_factor)

    return (y, name)

def process_adaa_mipmap(x: List[np.ndarray], cache: MipMapAdaaCache, 
                        ftype: Algorithm, forder:int, os_factor: int) -> Tuple[np.ndarray, str]:
    sr = SAMPLERATE * os_factor
    X =[np.linspace(0, 1, vec.shape[0] + 1, endpoint=True) for vec in cache.m_mipmap]


    assert(ftype != Algorithm.NAIVE)
    if ftype == Algorithm.ADAA_BUTTERWORTH:
        (r, p, _) = butter_coeffs(forder, BUTTERWORTH_CTF, sr)
        fname = "BT"
    else:
        (r, p, _) = cheby_coeffs(forder, CHEBY_CTF, 60, sr)
        fname = "CH"

    y = np.zeros(x.shape[0])

    for order in range(0, forder, 2):
        ri = r[order]
        zi = p[order]
        y += process_fwd_mipmap(x, ri, zi, X, cache.m_mipmap, cache.q_mipmap, cache.m_diff_mipmap, cache.q_diff_mipmap, cache.mipmap_scale)

    if os_factor == 2:
        ds_size = int(y.shape[0] / os_factor)
        y_ds = np.zeros((ds_size,))
        decimator = Decimator9()
        for i in range(ds_size):
            y_ds[i] = decimator.process(y[i*2], y[i*2 + 1])
        y = y_ds
    elif os_factor != 1:
        # Downsampling
        y = soxr.resample(
            y,
            SAMPLERATE * os_factor,
            SAMPLERATE,
            quality="HQ"
        )
    # if os_factor != 1:
    #     # Downsampling
    #     y = soxr.resample(
    #         y,
    #         SAMPLERATE * os_factor,
    #         SAMPLERATE,
    #         quality="HQ"
    #     )


    name = "ADAA {} order {}".format(fname, forder)
    if os_factor != 1:
        name += "OVSx{}".format(os_factor)

    return (y, name)


def process_naive(x: np.ndarray, waveform: np.ndarray, os_factor: int) -> Tuple[np.ndarray, str]:
    # Waveform generation
    y = process_naive_linear(waveform, x)

    if os_factor != 1:
        # Downsampling
        y = soxr.resample(
            y,
            SAMPLERATE * os_factor,
            SAMPLERATE,
            quality="HQ"
        )

    name = "naive"
    if os_factor != 1:
        name += "OVSx{}".format(os_factor)
    
    return (y, name)

# @njit
def generate_sweep_phase(f1, f2, t, fs):
    # Calculate the number of samples
    n = int(t * fs)

    freqs = np.linspace(f1, f2, n-1)
    # start = np.log2(f1)
    # stop = np.log2(f2)
    # freqs = np.logspace(start, stop, num=n-1, base=2)
    phases = np.zeros(n)

    phase = 0
    for (i, freq) in enumerate(freqs):
        step = freq / fs
        phase += step
        phases[i+1] = phase

    return phases



@dataclass
class AlgorithmDetails:
    algorithm: Algorithm
    oversampling : int
    forder: int
    mipmap: bool = False
    waveform_len: int = 4096 // 2

    @property
    def name(self) -> str:
        name = ""
        if self.algorithm is Algorithm.NAIVE:
            name += "naive"
        else:
            name += "ADAA"
            if self.mipmap:
                name += "_mipmap"

            if self.algorithm is Algorithm.ADAA_BUTTERWORTH:
                name += "_BT"
            else:
                name += "_CH"
            name += "_order_{}".format(self.forder)
        
        if self.oversampling > 1:
            name += "_OVSx{}".format(self.oversampling)
        
        name += "_w{}".format(self.waveform_len)
        return name
    
    def name_with_freq(self, freq: int) -> str:
        return self.name + "_{}Hz".format(freq)

def routine(details: AlgorithmDetails, x, cache, freq: int = None) -> Tuple[str, int, np.ndarray[float]]:
    if freq is None:
        name = details.name
    else:
        name = details.name_with_freq(freq)
    logging.info("{} : started".format(name))
    if details.algorithm is Algorithm.NAIVE:
        generated = process_naive(x, cache.waveform, details.oversampling)[0]
    elif not details.mipmap:
        generated = process_adaa(x, cache, details.algorithm, details.forder, os_factor=details.oversampling)[0]
    else:
        generated = process_adaa_mipmap(x, cache, details.algorithm, details.forder, os_factor=details.oversampling)[0]


    logging.info("{} : end".format(name))
    return [name, freq, generated]


def plot_psd(time_signals : Dict[str, np.ndarray[float]]):
    fig, axs = plt.subplots(len(time_signals) + 1)

    # Plot all psd
    for i, (name, signal) in enumerate(time_signals.items()):
        axs[i].psd(signal, label=name, Fs=SAMPLERATE, NFFT=4096)
        axs[-1].plot(signal, label=name)
    
    for ax in axs:
        ax.grid(True, which="both")
        ax.legend()

    plt.show()

def plot_specgram(time_signals: Dict[str, np.ndarray[float]]):
    fig, axs = plt.subplots(len(time_signals))

    for i, (name, data) in enumerate(time_signals.items()):
        axs[i].specgram(data, NFFT=512, noverlap=256, vmin=-80)
        axs[i].set_title(name)
        axs[i].set_ylabel('Frequency [Hz]')
        axs[i].set_xlabel('Time [s]')
        axs[i].legend()

    plt.show()


@njit
def process_fwd_mipmap(x, B, beta: complex, X_mipmap: List[np.ndarray[float]], 
                       m_mipmap: List[np.ndarray[float]], q_mipmap: List[np.ndarray[float]], 
                       m_diff_mipmap: List[np.ndarray[float]], q_diff_mipmap: List[np.ndarray[float]], 
                       mipmap_scale: np.ndarray[float]):
    """
    This is a simplified version of the process method translated from matlab, more suited to real time use :

     - Assuming the playback will only goes forward (no reverse playing), I removed the conditionnal branching on x_diff

     - I replaced the formulas using ever-growing indexes and phase with equivalent ones using only "reduced" variables:
        1. (prev_x - T * floor(j_min/ waveform_frames)) became prev_x_red_bar
        2. (x[n] - T * floor((j_max+1)/waveform_frames)) is equivalent to x_red
        3. (x[n] - T * floor((i)/waveform_frames)) became x_red_bar

    see process() for the original "translation" from matlab code
    """
    y = np.zeros(x.shape[0])

    # Period - should be 1
    # assert(all(phases[-1] == 1.0 for phases in X))
    for phases in X_mipmap:
        assert(phases[-1] == 1.0)


    # waveform_frames = m.shape[0]     # aka k

    expbeta = cexp(beta)

    # Initial condition
    prev_x = x[0]
    prev_cpx_y: complex = 0
    prev_x_diff = 0

    # Setting j indexs and some reduced values
    x_red = prev_x % 1.0
    x_diff = x[1] - x[0]
    prev_mipmap_idx = find_mipmap_index(x_diff, mipmap_scale)
    j_red = binary_search_down(X_mipmap[prev_mipmap_idx], x_red, 0, X_mipmap[prev_mipmap_idx].shape[0] - 1)

    for n in range(1, x.shape[0]):
        # loop init
        x_diff = x[n] - prev_x
        assert(x_diff >= 0)
        mipmap_idx = find_mipmap_index(x_diff, mipmap_scale)
        waveform_frames = m_mipmap[mipmap_idx].shape[0]     # aka k
        prev_x_red_bar = x_red + (x_red == 0.0)     # To replace (prev_x - T * floor(j_min/ waveform_frames))
        
        if mipmap_idx == prev_mipmap_idx:
            prev_j_red = j_red + int(np.sign(x_red - X_mipmap[mipmap_idx][j_red]))
        else:
            j_red = binary_search_down(X_mipmap[mipmap_idx], x_red, 0, X_mipmap[mipmap_idx].shape[0] - 1)
            prev_j_red = j_red + int(np.sign(x_red - X_mipmap[mipmap_idx][j_red]))

        
        x_red = x[n] % 1.0

        # playback going forward
        j_red = binary_search_down(X_mipmap[mipmap_idx], x_red, 0, X_mipmap[mipmap_idx].shape[0] - 1)
        j_max_p_red = j_red
        j_min_red = (prev_j_red - 1) % waveform_frames


        prev_x_red_bar = prev_x % 1.0
        prev_x_red_bar += (prev_x_red_bar == 0.0)

        I = expbeta\
                * (m_mipmap[mipmap_idx][j_min_red] * x_diff + beta * (m_mipmap[mipmap_idx][j_min_red] * prev_x_red_bar + q_mipmap[mipmap_idx][j_min_red]))\
                - m_mipmap[mipmap_idx][j_max_p_red] * x_diff\
                - beta * (m_mipmap[mipmap_idx][j_max_p_red] * x_red + q_mipmap[mipmap_idx][j_max_p_red])

        I_sum = 0
        born_sup = j_max_p_red + waveform_frames * (j_min_red > j_max_p_red)
        for i in range(j_min_red, born_sup):
            i_red = i % waveform_frames
            x_red_bar = x[n] % 1.0
            x_red_bar = x_red_bar + (x_red_bar < X_mipmap[mipmap_idx][i_red])

            I_sum += cexp(beta * (x_red_bar - X_mipmap[mipmap_idx][i_red + 1])/x_diff)\
                        * (beta * q_diff_mipmap[mipmap_idx][i_red] + m_diff_mipmap[mipmap_idx][i_red] * (x_diff + beta * X_mipmap[mipmap_idx][i_red + 1]))

        I = (I + np.sign(x_diff) * I_sum) / (beta**2)

        # See formula (10)
        y_cpx: complex = expbeta * prev_cpx_y + 2 * B * I
        y[n] = y_cpx.real

        prev_x = x[n]

        prev_cpx_y = y_cpx
        prev_x_diff = x_diff
        prev_mipmap_idx = mipmap_idx

    return y


@njit
def process_fwd(x, B, beta: complex, X, m, q, m_diff, q_diff):
    """
    This is a simplified version of the process method translated from matlab, more suited to real time use :

     - Assuming the playback will only goes forward (no reverse playing), I removed the conditionnal branching on x_diff

     - I replaced the formulas using ever-growing indexes and phase with equivalent ones using only "reduced" variables:
        1. (prev_x - T * floor(j_min/ waveform_frames)) became prev_x_red_bar
        2. (x[n] - T * floor((j_max+1)/waveform_frames)) is equivalent to x_red
        3. (x[n] - T * floor((i)/waveform_frames)) became x_red_bar

    see process() for the original "translation" from matlab code
    """
    y = np.zeros(x.shape[0])

    # Period - should be 1
    assert(X[-1] == 1.0)

    waveform_frames = m.shape[0]     # aka k

    expbeta = cexp(beta)

    # Initial condition
    prev_x = x[0]
    prev_cpx_y: complex = 0
    prev_x_diff = 0

    # Setting j indexs and some reduced values
    x_red = prev_x % 1.0
    j_red = binary_search_down(X, x_red, 0, X.shape[0] - 1)

    for n in range(1, x.shape[0]):
        # loop init
        x_diff = x[n] - prev_x
        assert(x_diff >= 0)
        prev_x_red_bar = x_red + (x_red == 0.0)     # To replace (prev_x - T * floor(j_min/ waveform_frames))
        prev_j_red = j_red % waveform_frames

        # TODO: No idea ?
        if (x_diff >= 0 and prev_x_diff >=0) or (x_diff < 0 and prev_x_diff <= 0):
            # If on the same slop as the previous iteration
            prev_j_red = j_red + int(np.sign(x_red - X[j_red]))
            # Is used to avoid computing a new j_min using the binary search, because
            # if j_min == j_max then the I sum is zero so its corresponds to the case
            # where x_n and x_n+1 are in the same interval
        
        x_red = x[n] % 1.0

        # playback going forward
        j_red = binary_search_down(X, x_red, 0, X.shape[0] - 1)
        j_max_p_red = j_red
        j_min_red = (prev_j_red - 1) % waveform_frames


        prev_x_red_bar = prev_x % 1.0
        prev_x_red_bar += (prev_x_red_bar == 0.0)

        I = expbeta\
                * (m[j_min_red] * x_diff + beta * (m[j_min_red] * prev_x_red_bar + q[j_min_red]))\
                - m[j_max_p_red] * x_diff\
                - beta * (m[j_max_p_red] * x_red + q[j_max_p_red])

        I_sum = 0
        born_sup = j_max_p_red + waveform_frames * (j_min_red > j_max_p_red)
        for i in range(j_min_red, born_sup):
            i_red = i % waveform_frames
            x_red_bar = x[n] % 1.0
            x_red_bar = x_red_bar + (x_red_bar < X[i_red])

            I_sum += cexp(beta * (x_red_bar - X[i_red + 1])/x_diff)\
                        * (beta * q_diff[i_red] + m_diff[i_red] * (x_diff + beta * X[i_red + 1]))

        I = (I + np.sign(x_diff) * I_sum) / (beta**2)

        # See formula (10)
        y_cpx: complex = expbeta * prev_cpx_y + 2 * B * I
        y[n] = y_cpx.real

        prev_x = x[n]
        prev_cpx_y = y_cpx
        prev_x_diff = x_diff

    return y

def process_bi(x, B, beta: complex, X, m, q, m_diff, q_diff):
    """
    Direct translation from the matlab algorithm to python. Be aware matlab arrays starts at 1 so I had to make
    a few changes


    This code contains A LOT of annotations with commented out stuff. This is because I want to have a written trace
    of the adaptation I had to make to write the simplified process_fwd() version.

    Notice that this code does support reverse playback whereas process_fwd() does not
    """
    y = np.zeros(x.shape[0])
    abc = np.ndarray(x.shape[0])

    # Period - should be 1
    assert(X[-1] == 1.0)
    T = 1.0

    waveform_frames = m.shape[0]     # aka k

    expbeta = cexp(beta)

    # # Testing
    # alt_j_max_p_red = 0
    # alt_j_min_red = 0

    # Initial condition
    prev_x = x[0]
    prev_cpx_y: complex = 0
    prev_x_diff = 0

    # Setting j indexs and some reduced values
    x_red = prev_x % 1.0
    j_red = binary_search_down(X, x_red, 0, X.shape[0] - 1)
    j = waveform_frames * floor(prev_x / 1.0) + j_red - 1

    for n in range(1, x.shape[0]):
        # loop init
        x_diff = x[n] - prev_x
        # prev_x_red_bar = x_red + (x_red == 0.0)     # To replace (prev_x - T * floor(j_min/ waveform_frames))
        prev_j = j
        # prev_j_red = j_red % waveform_frames

        # TODO: No idea ?
        if (x_diff >= 0 and prev_x_diff >=0) or (x_diff < 0 and prev_x_diff <= 0):
            # If on the same slop as the previous iteration
            prev_j = j + int(np.sign(x_red - X[j_red]))
            # prev_j_red = j_red + int(np.sign(x_red - X[j_red]))
            # Is used to avoid computing a new j_min using the binary search, because
            # if j_min == j_max then the I sum is zero so its corresponds to the case
            # where x_n and x_n+1 are in the same interval
        
        x_red = x[n] % 1.0

        # Should be differentiated upstream to avoid if on each sample
        if x_diff >= 0:
            # playback going forward
            j_red = binary_search_down(X, x_red, 0, X.shape[0] - 1)
            # alt_j_max_p_red = j_red
            # alt_j_min_red = (prev_j_red - 1) % waveform_frames

            j = waveform_frames * floor(x[n] / 1.0) + j_red - 1
            j_min = prev_j
            j_max = j
        else:
            # playback going backward
            j_red = binary_search_up(X, x_red, 0, X.shape[0] - 1)
            j = waveform_frames * floor(x[n] / 1.0) + j_red - 1
            j_min = j
            j_max = prev_j

        j_min_red = j_min % waveform_frames
        j_max_p_red = (j_max + 1) % waveform_frames


        # prev_x_red_bar = prev_x % 1.0
        # prev_x_red_bar += (prev_x_red_bar == 0.0)

        # Could be differentiated upstream to avoid if on each sample
        if x_diff >= 0:
            ## OG version
            I = expbeta\
                    * (m[j_min_red] * x_diff + beta * (m[j_min_red] * (prev_x - T * floor(j_min/ waveform_frames)) + q[j_min_red]))\
                    - m[j_max_p_red] * x_diff\
                    - beta * (m[j_max_p_red] * (x[n] - T * floor((j_max+1)/waveform_frames)) + q[j_max_p_red])

            ### j_min/j_max independant version
            # I = expbeta\
            #         * (m[j_min_red] * x_diff + beta * (m[j_min_red] * prev_x_red_bar + q[j_min_red]))\
            #         - m[j_max_p_red] * x_diff\
            #         - beta * (m[j_max_p_red] * x_red + q[j_max_p_red])
        else:
            I = expbeta\
                    * (m[j_max_p_red] * x_diff + beta * (m[j_max_p_red] * (prev_x - T * floor((j_max+1)/waveform_frames)) + q[j_max_p_red]))\
                    - m[j_min_red] * x_diff\
                    - beta * (m[j_min_red] * (x[n] - T * floor(j_min/waveform_frames)) + q[j_min_red])

        I_sum = 0

        for i in range(j_min, j_max + 1):         #OG Version
        # for i in range(j_min_red, j_max_p_red):
            i_red = i % waveform_frames
            I_sum += cexp(beta * (x[n] - X[i_red + 1] - T * floor((i)/waveform_frames))/x_diff)\
                        * (beta * q_diff[i_red] + m_diff[i_red] * (x_diff + beta * X[i_red + 1]))

            # x_red_bar = x[n] % 1.0
            # x_red_bar = x_red_bar + (x_red_bar < X[i_red])
            # I_sum += cexp(beta * (x_red_bar - X[i_red + 1])/x_diff)\
            #             * (beta * q_diff[i_red] + m_diff[i_red] * (x_diff + beta * X[i_red + 1]))    

        I = (I + np.sign(x_diff) * I_sum) / (beta**2)

        # See formula (10)
        y_cpx: complex = expbeta * prev_cpx_y + 2 * B * I
        y[n] = y_cpx.real
        abc[n] = I_sum


        prev_x = x[n]
        prev_cpx_y = y_cpx
        prev_x_diff = x_diff

    return y



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script with optional arguments")

     # Define the mode argument as a choice between "psd", "metrics", and "sweep"
    parser.add_argument("mode", choices=["psd", "metrics", "sweep"],
                        help="Choose a mode: psd, metrics, or sweep")

    # Define the export argument as a choice between "snr", "sinad", and "both"
    parser.add_argument("--export", choices=["snr", "sinad", "both", "none"], default="both",
                        help="Choose what to export: snr, sinad, or both (default)")
    parser.add_argument("--export-dir", type=Path, default=Path.cwd())
    parser.add_argument("--export-audio", action="store_true")
    parser.add_argument("--no-log", action="store_true", help="Disable console logging")


    args = parser.parse_args()

    if args.mode != "metrics":
        args.export = "none"

    import matlab.engine
    # future_engine = matlab.engine.start_matlab(background=True)
    
    # FREQS = [197, 397, 597, 997, 1599, 2173, 3003, 3997]
    # FREQS = np.int32(np.logspace(start=5, stop=14, num=200, base=2))
    # FREQS = np.int32(np.linspace(start=2**5, stop=2**14, num=200))
    # FREQS = [noteToFreq(i) for i in range(21, 109)]
    FREQS = [noteToFreq(i) for i in range(128)]
    # FREQS = range(4000, 4200)
    # FREQS = np.linspace(60, )
    # FREQS = [1010]

    # FREQS = [3997]
    # for i in range(4, 16):
    #     FREQS.append(2**i -1)
    #     FREQS.append(2**i)
    #     FREQS.append(2**i + 1)


    # FREQS = [(2**i - 1, 2**i, 2**  +1) for i in range(4, 15)]
    # print(FREQS)
    ALGOS_OPTIONS = [
        # AlgorithmDetails(Algorithm.NAIVE, 1, 0),
        # AlgorithmDetails(Algorithm.NAIVE, 4, 0),
        AlgorithmDetails(Algorithm.NAIVE, 8, 0),
        # AlgorithmDetails(Algorithm.ADAA_BUTTERWORTH, 1, 2),
        AlgorithmDetails(Algorithm.ADAA_BUTTERWORTH, 1, 2, mipmap=True),
        # AlgorithmDetails(Algorithm.ADAA_BUTTERWORTH, 2, 2, mipmap=True),
        # AlgorithmDetails(Algorithm.ADAA_BUTTERWORTH, 1, 2, mipmap=False, waveform_len=4096),
        # AlgorithmDetails(Algorithm.ADAA_BUTTERWORTH, 2, 2),
        # AlgorithmDetails(Algorithm.ADAA_BUTTERWORTH, 1, 2, mipmap=False, waveform_len=1024),
        # AlgorithmDetails(Algorithm.ADAA_BUTTERWORTH, 1, 2, mipmap=False, waveform_len=512),
        # AlgorithmDetails(Algorithm.ADAA_BUTTERWORTH, 1, 2, mipmap=False, waveform_len=256),
        # AlgorithmDetails(Algorithm.ADAA_BUTTERWORTH, 1, 2, mipmap=False, waveform_len=128),
        # AlgorithmDetails(Algorithm.ADAA_BUTTERWORTH, 1, 2, mipmap=False, waveform_len=64),
        # AlgorithmDetails(Algorithm.ADAA_BUTTERWORTH, 1, 2, mipmap=False, waveform_len=32),
        # AlgorithmDetails(Algorithm.ADAA_BUTTERWORTH, 1, 4),
        # AlgorithmDetails(Algorithm.ADAA_CHEBYSHEV_TYPE2, 1, 10, mipmap=False, waveform_len=4096),
        # AlgorithmDetails(Algorithm.ADAA_CHEBYSHEV_TYPE2, 1, 10),
        AlgorithmDetails(Algorithm.ADAA_CHEBYSHEV_TYPE2, 1, 10, mipmap=True),
        # AlgorithmDetails(Algorithm.ADAA_CHEBYSHEV_TYPE2, 2, 10, mipmap=True),
        # AlgorithmDetails(Algorithm.ADAA_CHEBYSHEV_TYPE2, 2, 8),
        # AlgorithmDetails(Algorithm.ADAA_CHEBYSHEV_TYPE2, 1, 10, mipmap=False, waveform_len=1024),
        # AlgorithmDetails(Algorithm.ADAA_CHEBYSHEV_TYPE2, 1, 10, mipmap=False, waveform_len=512),
        # AlgorithmDetails(Algorithm.ADAA_CHEBYSHEV_TYPE2, 1, 10, mipmap=False, waveform_len=256),
        # AlgorithmDetails(Algorithm.ADAA_CHEBYSHEV_TYPE2, 1, 10, mipmap=False, waveform_len=128),
        # AlgorithmDetails(Algorithm.ADAA_CHEBYSHEV_TYPE2, 1, 10, mipmap=False, waveform_len=64),
        # AlgorithmDetails(Algorithm.ADAA_CHEBYSHEV_TYPE2, 1, 10, mipmap=False, waveform_len=32),
        # AlgorithmDetails(Algorithm.ADAA_CHEBYSHEV_TYPE2, 1, 10, mipmap=False, waveform_len=16),
        # AlgorithmDetails(Algorithm.ADAA_CHEBYSHEV_TYPE2, 1, 10, mipmap=False, waveform_len=8),
        # AlgorithmDetails(Algorithm.ADAA_BUTTERWORTH, 2, 2),
        # AlgorithmDetails(Algorithm.ADAA_BUTTERWORTH, 2, 4),
        # AlgorithmDetails(Algorithm.ADAA_CHEBYSHEV_TYPE2, 2, 10),
    ]
    sorted_bl = dict()

    # Prepare parallel run

    logging.info("Setting up phase vectors")
    routine_args = []
    mipmap_caches = dict()
    adaa_caches: Dict[int, AdaaCache] = dict()
    naive_caches : Dict[int, NaiveCache] = dict()
    # names = []

    # waveform = NAIVE_SAW
    # waveform = FileWaveform("wavetables/massacre.wav")
    waveform = NaiveWaveform(NaiveWaveform.Type.SAW, 2048, 44100)


    # Init caches
    if any(opt.mipmap and opt.algorithm != Algorithm.NAIVE for opt in ALGOS_OPTIONS):
        # Init mipmap cache
        scale = mipmap_scale(waveform.size, SAMPLERATE, 9)
        mipmap_waveforms = compute_mipmap_waveform(waveform.get(), len(scale) + 1)
        (m, q, m_diff, q_diff) = mipmap_mq_from_waveform(mipmap_waveforms)
        mipmap_cache = MipMapAdaaCache(m,q, m_diff, q_diff, scale)

    if any(not opt.mipmap and opt.algorithm != Algorithm.NAIVE for opt in ALGOS_OPTIONS):
        # Init basic adaa cache
        for opt in ALGOS_OPTIONS:
            if opt.waveform_len not in adaa_caches:
                (m, q, m_diff, q_diff) = mq_from_waveform(waveform.get(opt.waveform_len))
                adaa_caches[opt.waveform_len] = AdaaCache(m, q, m_diff, q_diff)
    
    if any(opt.algorithm == Algorithm.NAIVE for opt in ALGOS_OPTIONS):
        # Init naive cache
        for opt in ALGOS_OPTIONS:
            if opt.waveform_len not in naive_caches:
                naive_caches[opt.waveform_len] = NaiveCache(waveform.get(opt.waveform_len))



    if args.mode == "sweep":
        for options in ALGOS_OPTIONS:
            x = generate_sweep_phase(20, SAMPLERATE / 2, DURATION_S, SAMPLERATE * options.oversampling)
            if options.mipmap and options.algorithm != Algorithm.NAIVE:
                cache = mipmap_cache
            elif not options.mipmap and options.algorithm != Algorithm.NAIVE:
                cache = adaa_caches[options.waveform_len]
            else:
                cache = naive_caches[options.waveform_len]
            routine_args.append([options, x, cache])
    else:
        bl_gen_args = [
            [np.linspace(0, DURATION_S, num = int(DURATION_S * SAMPLERATE), endpoint = False), freq] for freq in FREQS
        ]
        with Pool(NUM_PROCESS) as pool:
            bl_results = pool.starmap(bl_sawtooth, bl_gen_args)

        for i, freq in enumerate(FREQS):
            # dist = SAMPLERATE % freq
            # if dist < 3.0 or abs(freq - dist) < 3.0:
            #     print("Skipping {} Hz" .format(freq))
            #     continue

            # while SAMPLERATE % freq == 0:
            #     print("Skipping {} Hz" .format(freq))
            #     freq -= 3
            # if freq in sorted_bl:
            #     print("Duplicate {} Hz".format(freq))
            #     continue

            # sorted_bl[freq] = bl_sawtooth(np.linspace(0, DURATION_S, num = int(DURATION_S * SAMPLERATE), endpoint = False), freq)
            sorted_bl[freq] = bl_results[i]
            for options in ALGOS_OPTIONS:
                if options.mipmap and options.algorithm != Algorithm.NAIVE:
                    cache = mipmap_cache
                elif not options.mipmap and options.algorithm != Algorithm.NAIVE:
                    cache = adaa_caches[options.waveform_len]
                else:
                    cache = naive_caches[options.waveform_len]

                num_frames = int(DURATION_S * SAMPLERATE * options.oversampling)
                x = np.linspace(0.0, DURATION_S*freq, num_frames, endpoint=True)
                routine_args.append([options, x, cache, freq])
    
    # Run generating in parallel
    with Pool(NUM_PROCESS) as pool:
        results = pool.starmap(routine, routine_args)

    # Delete caches to free memory
    del mipmap_caches
    del adaa_caches
    del naive_caches

    if args.mode == "metrics":
        # engine = future_engine.result()
        import matlab.engine
        engine = matlab.engine.start_matlab()

        # Compute SNRs
        logging.info("Computing SNRs")
        for res in results:
            num_harmonics = max(2, floor(SAMPLERATE / 2 / res[1]))
            res.append(engine.snr(res[2], SAMPLERATE, num_harmonics))

        if not args.no_log:
            logging.info("Logging SNR")
            for (name, freq, data, snr_value) in results:
                print("{} SNR : {}".format(name, snr_value))
    elif args.mode == "psd":
        logging.info("Plotting psd")

        signals = {name: signal for name, _, signal in results}
        plot_psd(signals)
    else:
        assert(args.mode == "sweep")
        logging.info("Plotting sweep spectrogram")
        signals = {name: signal for name, _, signal in results}
        plot_specgram(signals)


    if args.export != "none" or args.export_audio:
        args.export_dir.mkdir(parents=True, exist_ok=True)

    if args.export_audio:
        for res in results:
            filename = res[0] + ".wav"
            sf.write(args.export_dir / filename, res[2] * 0.70, SAMPLERATE)

    # Write to CSV
    if args.export in ("snr", "both"):
        assert(args.export_dir is not None)

        logging.info("Exporting SNR to CSV")
        # Sort data for csv
        sorted_results = defaultdict(list)
        for [name, freq, data, snr_value] in results:
            sorted_results[freq].append(snr_value)

        csv_output = args.export_dir / (args.export_dir.name + "_snr.csv")
        with open(csv_output, "w") as csv_file:
            csvwriter = csv.writer(csv_file)
            names = [opt.name for opt in ALGOS_OPTIONS]
            csvwriter.writerow(["frequency", *names])

            for freq, snr_values in sorted_results.items():
                csvwriter.writerow([freq, *snr_values])

        del sorted_results

    if args.export in ("sinad", "both"):
        sorted_sinad = defaultdict(list)
        sinad_args = [
            [normalized_fft(data), normalized_fft(sorted_bl[freq]), freq] for (_, freq, data, _) in results
        ]
        logging.info("Computing SINADs")
        with Pool(8) as pool:
            sinad_values = pool.starmap(fast_compute_sinad, sinad_args)

        for i, freq in enumerate(FREQS):
            sorted_sinad[freq].append(sinad_values[i])
        # for [name, freq, data, snr_value] in results:
        #     noised_fft = normalized_fft(data)
        #     clean_fft = normalized_fft(sorted_bl[freq])
        #     sorted_sinad[freq].append(fast_compute_sinad(noised_fft, clean_fft, freq))

        logging.info("Exporting SINAD to CSV")
        thd_output = args.export_dir / (args.export_dir.name + "_sinad.csv")
        with open(thd_output, "w") as csv_file:
            csvwriter = csv.writer(csv_file)
            names = [opt.name for opt in ALGOS_OPTIONS]
            csvwriter.writerow(["frequency", *names])

            for freq, thd_values in sorted_sinad.items():
                csvwriter.writerow([freq, *thd_values])
