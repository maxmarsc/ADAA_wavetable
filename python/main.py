from collections import defaultdict
import numpy as np
from math import exp, floor, ceil
from cmath import exp as cexp
from scipy.signal import (
    butter,
    cheby2,
    residue,
    zpk2tf,
    decimate,
)
import matplotlib.pyplot as plt
import matplotlib
import soundfile as sf
import csv
from pathlib import Path
import argparse
from dataclasses import dataclass
from multiprocessing import Pool
import logging
from numba import njit
import os
from typing import Tuple, List, Dict

from bl_waveform import bl_sawtooth
from decimator import Decimator17, Decimator9
from waveform import FileWaveform, NaiveWaveform
from mipmap import *

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

SAMPLERATE = 44100
BUTTERWORTH_CTF = 0.45 * SAMPLERATE
CHEBY_CTF = 0.61 * SAMPLERATE
# Watch out for ram usage
DURATION_S = 5.0
# Watch out for ram usage
NUM_PROCESS = min(os.cpu_count(), 20)

matplotlib.use("TkAgg")


@dataclass
class MipMapAdaaCache:
    m_mipmap: List[np.ndarray[float]]
    q_mipmap: List[np.ndarray[float]]
    m_diff_mipmap: List[np.ndarray[float]]
    q_diff_mipmap: List[np.ndarray[float]]
    mipmap_scale: np.ndarray[float]


@dataclass
class AdaaCache:
    m: np.ndarray[float]
    q: np.ndarray[float]
    m_diff: np.ndarray[float]
    q_diff: np.ndarray[float]


@dataclass
class NaiveCache:
    waveform: np.ndarray[float]


@njit
def noteToFreq(note: int) -> float:
    a = 440  # frequency of A (coomon value is 440Hz)
    return (a / 32) * (2 ** ((note - 9) / 12))


def butter_coeffs(
    order, ctf, samplerate
) -> Tuple[np.ndarray[complex], np.ndarray[complex]]:
    """
    Computes butterworth filter coeffs like in the matlab code
    """
    ang_freq = 2 * np.pi * ctf / samplerate
    (z, p, k) = butter(order, ang_freq, output="zpk", analog=True)
    (b, a) = zpk2tf(z, p, k)
    (r, p, _) = residue(b, a)
    return (r, p, (b, a))


def cheby_coeffs(
    order, ctf, attn, samplerate
) -> Tuple[np.ndarray[complex], np.ndarray[complex]]:
    """
    Computes chebyshev type 2 filter coeffs like in the matlab code
    """
    ang_freq = 2 * np.pi * ctf / samplerate
    (z, p, k) = cheby2(order, attn, ang_freq, output="zpk", analog=True)
    (b, a) = zpk2tf(z, p, k)
    (r, p, _) = residue(b, a)
    return (r, p, (b, a))


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

    About the comments in the function :
    Notes that when the slope is too big (threshold is empiric), this replicate the previous m & q to mimic the
    m & q definitions found in the paper (ie : in the paper they do have non-linearities in m & q).
    """
    size = waveform.shape[0]
    # slope_thrsd = size / 64
    # idx_to_estimate = []
    m = np.zeros(size)
    q = np.zeros(size)
    X = np.linspace(0, 1, waveform.shape[0] + 1)

    for i in range(size - 1):
        y0 = waveform[i]
        y1 = waveform[i + 1]
        x0 = X[i]
        x1 = X[i + 1]
        m_i = compute_m(x0, x1, y0, y1)
        q_i = compute_q(x0, x1, y0, y1)

        # if abs(m_i) > slope_thrsd:
        #     # print("Exceeded threshold at idx {} of waveform {}".format(i, waveform.shape[0]))
        #     idx_to_estimate.append(i)
        #     continue

        m[i] = m_i
        q[i] = q_i

    m[-1] = np.single(
        compute_m(
            np.double(X[-2]),
            np.double(X[-1]),
            np.double(waveform[-1]),
            np.double(waveform[0]),
        )
    )
    q[-1] = np.single(
        compute_q(
            np.double(X[-2]),
            np.double(X[-1]),
            np.double(waveform[-1]),
            np.double(waveform[0]),
        )
    )
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
def process_naive_linear(waveform, x_values):
    """
    Linear interpolation algorithm
    """
    y = np.zeros(x_values.shape[0])
    waveform_len = waveform.shape[0]

    for i, x in enumerate(x_values):
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


# @njit
def normalized_fft(time_signal, padding: int = 0) -> np.ndarray:
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


@njit
def mq_from_waveform(
    waveform: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (m, q) = compute_m_q_vectors(waveform)
    m_diff = np.zeros(m.shape[0])
    q_diff = np.zeros(q.shape[0])

    for i in range(m.shape[0] - 1):
        m_diff[i] = m[i + 1] - m[i]
        q_diff[i] = q[i + 1] - q[i]
    m_diff[-1] = m[0] - m[-1]
    q_diff[-1] = q[0] - q[-1] - m[0]

    return (m, q, m_diff, q_diff)


def mipmap_mq_from_waveform(
    waveforms: List[np.ndarray[float]],
) -> Tuple[
    List[np.ndarray[float]],
    List[np.ndarray[float]],
    List[np.ndarray[float]],
    List[np.ndarray[float]],
]:
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


@njit
def fast_compute_sinad(
    noised_fft: np.ndarray, clean_fft: np.ndarray, fundamental: float
) -> float:
    assert noised_fft.shape == clean_fft.shape
    assert noised_fft.shape[0] > 0

    fft_size = noised_fft.shape[0]
    bin_size = SAMPLERATE / 2 / (fft_size - 1)

    # find fundamental frequency
    fundamental_bin = np.argmax(np.abs(clean_fft))

    # normalize so the two signals have the same fundamental power
    norm_ratio = np.abs(clean_fft[fundamental_bin]) / np.abs(
        noised_fft[fundamental_bin]
    )
    clean_fft /= norm_ratio

    # Compute Harmonic Distortion
    harmonics = [i * fundamental_bin for i in range(2, fft_size // fundamental_bin)]
    hd_power = np.sum(
        np.array(
            [(np.abs(noised_fft[h]) - np.abs(clean_fft[h])) ** 2 for h in harmonics]
        )
    )

    # Compute Noise
    noise_bins = [
        i
        for i in range(1, len(noised_fft) // 2)
        if i not in harmonics and i != fundamental_bin
    ]
    noise_power = np.sum(
        np.array(
            [(np.abs(noised_fft[n]) - np.abs(clean_fft[n])) ** 2 for n in noise_bins]
        )
    )

    # Compute SINAD
    signal_power = np.abs(clean_fft[fundamental_bin])
    sinad_value = (signal_power + hd_power + noise_power) / (hd_power + noise_power)
    sinad_db = 10 * np.log10(sinad_value)

    return sinad_db


from enum import Enum


class Algorithm(Enum):
    ADAA_BUTTERWORTH = 1
    ADAA_CHEBYSHEV_TYPE2 = 2
    NAIVE = 3


def process_adaa(
    x: np.ndarray, cache: AdaaCache, ftype: Algorithm, forder: int, os_factor: int
) -> Tuple[np.ndarray, str]:
    sr = SAMPLERATE * os_factor
    waveform_len = cache.m.shape[0]
    X = np.linspace(0, 1, waveform_len + 1, endpoint=True)
    x = np.mod(x, 1.0)

    assert ftype != Algorithm.NAIVE
    if ftype == Algorithm.ADAA_BUTTERWORTH:
        (r, p, _) = butter_coeffs(forder, BUTTERWORTH_CTF, sr)
        fname = "BT"
    else:
        (r, p, _) = cheby_coeffs(forder, CHEBY_CTF, 60, sr)
        fname = "CH"

    # filter_msg = "{} {}\nr : {}\np : {}".format(fname, forder, r, p)
    # logging.info(filter_msg)
    y = np.zeros(x.shape[0])

    for order in range(0, forder, 2):
        ri = r[order]
        zi = p[order]
        y += process_bi_red(x, ri, zi, X, cache.m, cache.q, cache.m_diff, cache.q_diff)

    if os_factor == 2:
        ds_size = int(y.shape[0] / os_factor)
        y_ds = np.zeros((ds_size,))
        decimator = Decimator17()
        for i in range(ds_size):
            y_ds[i] = decimator.process(y[i * 2], y[i * 2 + 1])
        y = y_ds
    elif os_factor != 1:
        # Downsampling
        y = decimate(y, os_factor)

    name = "ADAA {} order {}".format(fname, forder)
    if os_factor != 1:
        name += "OVSx{}".format(os_factor)

    return (y, name)


def process_adaa_mipmap(
    x: List[np.ndarray],
    cache: MipMapAdaaCache,
    ftype: Algorithm,
    forder: int,
    os_factor: int,
) -> Tuple[np.ndarray, str]:
    sr = SAMPLERATE * os_factor
    X = [np.linspace(0, 1, vec.shape[0] + 1, endpoint=True) for vec in cache.m_mipmap]
    x = np.mod(x, 1.0)

    assert ftype != Algorithm.NAIVE
    if ftype == Algorithm.ADAA_BUTTERWORTH:
        (r, p, _) = butter_coeffs(forder, BUTTERWORTH_CTF, sr)
        fname = "BT"
    else:
        (r, p, _) = cheby_coeffs(forder, CHEBY_CTF, 60, sr)
        fname = "CH"

    # filter_msg = "{} {}\nr : {}\np : {}".format(fname, forder, r, p)
    # logging.info(filter_msg)

    y = np.zeros(x.shape[0])

    for order in range(0, forder, 2):
        ri = r[order]
        zi = p[order]
        y += process_bi_mipmap_xfading(
            x,
            ri,
            zi,
            X,
            cache.m_mipmap,
            cache.q_mipmap,
            cache.m_diff_mipmap,
            cache.q_diff_mipmap,
            cache.mipmap_scale,
        )

    if os_factor == 2:
        ds_size = int(y.shape[0] / os_factor)
        y_ds = np.zeros((ds_size,))
        decimator = Decimator9()
        for i in range(ds_size):
            y_ds[i] = decimator.process(y[i * 2], y[i * 2 + 1])
        y = y_ds
    elif os_factor != 1:
        # Downsampling
        y = decimate(y, os_factor)

    name = "ADAA {} order {}".format(fname, forder)
    if os_factor != 1:
        name += "OVSx{}".format(os_factor)

    return (y, name)


def process_naive(
    x: np.ndarray, waveform: np.ndarray, os_factor: int
) -> Tuple[np.ndarray, str]:
    # Waveform generation
    y = process_naive_linear(waveform, x)
    assert np.isnan(y).any() == False

    if os_factor != 1:
        y = decimate(y, os_factor)

    name = "naive"
    if os_factor != 1:
        name += "OVSx{}".format(os_factor)

    return (y, name)


def generate_sweep_phase(f1, f2, t, fs, log_scale=False):
    # Calculate the number of samples
    n = int(t * fs)

    if log_scale:
        start = np.log2(f1)
        stop = np.log2(f2)
        freqs = np.logspace(start, stop, num=n - 1, base=2)
    else:
        freqs = np.linspace(f1, f2, n - 1)

    phases = np.zeros(n)

    phase = 0
    for i, freq in enumerate(freqs):
        step = freq / fs
        phase += step
        phases[i + 1] = phase

    return phases


@dataclass
class AlgorithmDetails:
    algorithm: Algorithm
    oversampling: int
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


def routine(
    details: AlgorithmDetails, x, cache, freq: int = None
) -> Tuple[str, int, np.ndarray[float]]:
    if freq is None:
        name = details.name
    else:
        name = details.name_with_freq(freq)
    logging.info("{} : started".format(name))
    if details.algorithm is Algorithm.NAIVE:
        generated = process_naive(x, cache.waveform, details.oversampling)[0]
    elif not details.mipmap:
        generated = process_adaa(
            x, cache, details.algorithm, details.forder, os_factor=details.oversampling
        )[0]
    else:
        generated = process_adaa_mipmap(
            x, cache, details.algorithm, details.forder, os_factor=details.oversampling
        )[0]

    logging.info("{} : end".format(name))
    return [name, freq, generated]


def plot_psd(time_signals: Dict[str, np.ndarray[float]]):
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
        axs[i].specgram(data, NFFT=512, noverlap=256, vmin=-60, Fs=44100)
        axs[i].set_title(name)
        axs[i].set_ylabel("Frequency [Hz]")
        axs[i].set_xlabel("Time [s]")
        axs[i].legend()

    plt.show()


@njit
def process_bi_mipmap_xfading(
    x,
    B,
    beta: complex,
    X_mipmap: List[np.ndarray[float]],
    m_mipmap: List[np.ndarray[float]],
    q_mipmap: List[np.ndarray[float]],
    m_diff_mipmap: List[np.ndarray[float]],
    q_diff_mipmap: List[np.ndarray[float]],
    mipmap_scale: np.ndarray[float],
):
    """
    Bidirectionnal version of the algorithm, with mipmapping, phase and index reduction

    This is the version most compatible with real-time implementation
    """
    y = np.zeros(x.shape[0])

    # Period - should be 1
    for phases in X_mipmap:
        assert phases[-1] == 1.0

    expbeta = cexp(beta)

    # Initial condition
    prev_x = x[0]
    prev_cpx_y: complex = 0
    prev_x_diff = 0

    # Setting j indexs and some reduced values
    x_red = prev_x % 1.0
    x_diff = x[1] - x[0]
    mipmap_xfade_idxs = find_mipmap_xfading_indexes(abs(x_diff), mipmap_scale)
    prev_mipmap_idx = mipmap_xfade_idxs[0]
    waveform_frames = m_mipmap[prev_mipmap_idx].shape[0]
    if x_diff > 0:
        j_red = floor(x_red * waveform_frames)
    else:
        j_red = ceil(x_red * waveform_frames)

    for n in range(1, x.shape[0]):
        # loop init
        x_diff = x[n] - prev_x
        if x_diff < -0.5:
            x_diff += 1.0
        elif x_diff > 0.5:
            x_diff -= 1.0

        mipmap_idx, weight, mipmap_idx_up, weight_up = find_mipmap_xfading_indexes(
            abs(x_diff), mipmap_scale
        )
        waveform_frames = m_mipmap[mipmap_idx].shape[0]  # aka k

        # Cautious, in this block, x_red still holds the value of the previous iteration
        if mipmap_idx != prev_mipmap_idx:
            # if prev_x_diff >= 0:
            #     ref = floor(x_red * waveform_frames)
            #     j_red = ref
            # else:
            #     ref = ceil(x_red * waveform_frames)
            #     j_red = ref

            if mipmap_idx > prev_mipmap_idx:
                # Going up in frequencies
                # print("Going up from ", j_red)
                j_red = j_red >> (mipmap_idx - prev_mipmap_idx)
            else:
                # Going down in frequencies
                j_red = j_red << (prev_mipmap_idx - mipmap_idx)
                if prev_x_diff >= 0:
                    j_red += X_mipmap[mipmap_idx][j_red + 1] < x_red
                else:
                    j_red -= X_mipmap[mipmap_idx][j_red - 1] > x_red
                # if prev_x_diff >= 0:
                #     j_red = floor(x_red * waveform_frames)
                # else:
                #     j_red = ceil(x_red * waveform_frames)
                # print("Going down from ", j_red)

                # j_red = j_red << (prev_mipmap_idx - mipmap_idx)

        prev_j_red = j_red
        if (x_diff >= 0 and prev_x_diff >= 0) or (x_diff < 0 and prev_x_diff <= 0):
            # If on the same slop as the previous iteration
            prev_j_red = j_red + int(np.sign(x_red - X_mipmap[mipmap_idx][j_red]))

        x_red = x[n] % 1.0

        # j_red = floor(x_red * waveform_frames)
        if x_diff >= 0:
            # playback going forward
            j_red = floor(x_red * waveform_frames)
            jmax = j_red
            jmin = prev_j_red
        else:
            # playback going backward
            j_red = ceil(x_red * waveform_frames)
            jmax = prev_j_red
            jmin = j_red

        jmin_red = (jmin - 1) % waveform_frames
        jmax_p_red = jmax % waveform_frames

        prev_x_red = prev_x % 1.0

        I_crt = compute_I_bi(
            m_mipmap[mipmap_idx],
            q_mipmap[mipmap_idx],
            m_diff_mipmap[mipmap_idx],
            q_diff_mipmap[mipmap_idx],
            X_mipmap[mipmap_idx],
            jmin,
            jmin_red,
            jmax_p_red,
            beta,
            expbeta,
            x_diff,
            prev_x_red,
            x_red,
        )

        if weight_up != 0.0:
            jmin_red_up = jmin_red // 2
            jmax_p_red_up = jmax_p_red // 2

            I_up = compute_I_bi(
                m_mipmap[mipmap_idx_up],
                q_mipmap[mipmap_idx_up],
                m_diff_mipmap[mipmap_idx_up],
                q_diff_mipmap[mipmap_idx_up],
                X_mipmap[mipmap_idx_up],
                jmin,
                jmin_red_up,
                jmax_p_red_up,
                beta,
                expbeta,
                x_diff,
                prev_x_red,
                x_red,
            )
            I_crt = I_crt * weight + weight_up * I_up

        beta_pow2 = beta**2
        y_cpx: complex = expbeta * prev_cpx_y + 2 * B * (I_crt / beta_pow2)
        y[n] = y_cpx.real

        prev_x = x[n]

        prev_cpx_y = y_cpx
        prev_x_diff = x_diff
        prev_mipmap_idx = mipmap_idx

    return y


@njit
def compute_I_bi(
    m,
    q,
    m_diff,
    q_diff,
    X,
    jmin,
    jmin_red,
    jmax_p_red,
    beta,
    expbeta,
    x_diff,
    prev_x_red,
    x_red,
) -> complex:
    frames = m.shape[0]
    prev_x_red_bar = prev_x_red + (prev_x_red == 0.0) * (x_diff > 0)
    x_red_bar = x_red + (x_red == 0.0) * (x_diff < 0)

    if x_diff > 0:
        idx_prev_bound = jmin_red
        idx_next_bound = jmax_p_red
    else:
        idx_prev_bound = jmax_p_red
        idx_next_bound = jmin_red

    I = (
        expbeta
        * (
            m[idx_prev_bound] * x_diff
            + beta * (m[idx_prev_bound] * prev_x_red_bar + q[idx_prev_bound])
        )
        - m[idx_next_bound] * x_diff
        - beta * (m[idx_next_bound] * x_red_bar + q[idx_next_bound])
    )

    I_sum = 0
    born_sup = jmax_p_red + frames * (jmin_red > jmax_p_red)
    if x_diff < 0 and jmin != 0 and jmin_red > jmax_p_red:
        cycle_offset = -1.0
    else:
        cycle_offset = 0.0
    for i in range(jmin_red, born_sup):
        i_red = i % frames
        x_red_bar = x_red + cycle_offset + (i_red > jmax_p_red)

        I_sum += cexp(beta * (x_red_bar - X[i_red + 1]) / x_diff) * (
            beta * q_diff[i_red] + m_diff[i_red] * (x_diff + beta * X[i_red + 1])
        )

    return I + np.sign(x_diff) * I_sum


@njit
def process_bi_red(x, B, beta: complex, X, m, q, m_diff, q_diff):
    """
    Bidirectionnal version of the algorithm, with phase and index reduction,
    without mipmapping
    """
    y = np.zeros(x.shape[0])

    # Period - should be 1
    assert X[-1] == 1.0

    waveform_frames = m.shape[0]  # aka k

    expbeta = cexp(beta)

    # Initial condition
    prev_x = x[0]
    prev_cpx_y: complex = 0
    prev_x_diff = 0

    # Setting j indexs and some reduced values
    x_red = prev_x % 1.0
    # j_red = binary_search_down(X, x_red, 0, X.shape[0] - 1)
    j_red = j_red = floor(x_red * (X.shape[0] - 1))
    __j = waveform_frames * floor(prev_x / 1.0) + j_red - 1
    __jred = j_red

    for n in range(1, x.shape[0]):
        # loop init
        x_diff = x[n] - prev_x
        if x_diff > 0.5:
            x_diff -= 1
        elif x_diff < -0.5:
            x_diff += 1
        # prev_x_red_bar = x_red + (x_red == 0.0)     # To replace (prev_x - T * floor(j_min/ waveform_frames))
        prev_j_red = j_red % waveform_frames
        __prevj = __j

        # TODO: No idea ?
        if (x_diff >= 0 and prev_x_diff >= 0) or (x_diff < 0 and prev_x_diff <= 0):
            # If on the same slop as the previous iteration
            prev_j_red = j_red + int(np.sign(x_red - X[j_red]))
            __prevj = __j + int(np.sign(x_red - X[__jred]))
            # Is used to avoid computing a new j_min using the binary search, because
            # if j_min == j_max then the I sum is zero so its corresponds to the case
            # where x_n and x_n+1 are in the same interval

        x_red = x[n] % 1.0

        # j_red = floor(x_red * waveform_frames)
        if x_diff >= 0:
            # playback going forward
            j_red = floor(x_red * waveform_frames)
            jmax = j_red
            jmin = prev_j_red

            # OG
            __jred = floor(x_red * waveform_frames)
            __j = waveform_frames * floor(x[n] / 1.0) + __jred - 1
            __jmin = __prevj
            __jmax = __j
        else:
            # playback going backward
            j_red = ceil(x_red * waveform_frames)
            jmax = prev_j_red
            jmin = j_red

            # OG
            __jred = ceil(x_red * waveform_frames)
            __j = waveform_frames * floor(x[n] / 1.0) + __jred - 1
            __jmin = __j
            __jmax = __prevj

        j_min_red = (jmin - 1) % waveform_frames
        j_max_p_red = jmax % waveform_frames
        __jminred = __jmin % waveform_frames
        __jmaxpred = (__jmax + 1) % waveform_frames
        assert j_min_red == __jminred
        assert j_max_p_red == __jmaxpred

        prev_x_red_bar = prev_x % 1.0
        prev_x_red_bar += (prev_x_red_bar == 0.0) * (x_diff >= 0)

        # Check the values of prev_x_red_bar
        if x_diff >= 0:
            ref = prev_x - floor(__jmin / waveform_frames)
            assert prev_x_red_bar == ref
        else:
            ref = prev_x - floor((__jmax + 1) / waveform_frames)
            assert prev_x_red_bar == ref

        # Check the values of x_red_bar2
        x_red_bar2 = x_red + (x_red == 0.0) * (x_diff < 0)
        if x_diff >= 0:
            ref = x[n] - floor((__jmax + 1) / waveform_frames)
            assert x_red_bar2 == ref
        else:
            ref = x[n] - floor(__jmin / waveform_frames)
            assert x_red_bar2 == ref

        if x_diff >= 0:
            I = (
                expbeta
                * (
                    m[j_min_red] * x_diff
                    + beta * (m[j_min_red] * prev_x_red_bar + q[j_min_red])
                )
                - m[j_max_p_red] * x_diff
                - beta * (m[j_max_p_red] * x_red_bar2 + q[j_max_p_red])
            )
        else:
            I = (
                expbeta
                * (
                    m[j_max_p_red] * x_diff
                    + beta * (m[j_max_p_red] * prev_x_red_bar + q[j_max_p_red])
                )
                - m[j_min_red] * x_diff
                - beta * (m[j_min_red] * x_red_bar2 + q[j_min_red])
            )

        I_sum = 0
        born_sup = j_max_p_red + waveform_frames * (j_min_red > j_max_p_red)
        if x_diff < 0 and jmin != 0 and j_min_red > j_max_p_red:
            cycle_offset = -1.0
        else:
            cycle_offset = 0.0

        # Checking the sum bounds
        # assert(born_sup - j_min_red == __jmax + 1 - __jmin)
        for i in range(j_min_red, born_sup):
            i_red = i % waveform_frames

            # Sol A :
            # if x_diff > 0:
            #     x_red_bar = x_red + (i_red >= j_min_red)
            # else:
            #     x_red_bar = x_red - (x_red > X[i_red]) * (i_red != j_min_red)
            # x_red_bar += (i == -1)

            # Sol B :
            x_red_bar = x_red + cycle_offset + (i_red > j_max_p_red)

            I_sum += cexp(beta * (x_red_bar - X[i_red + 1]) / x_diff) * (
                beta * q_diff[i_red] + m_diff[i_red] * (x_diff + beta * X[i_red + 1])
            )

        I = (I + np.sign(x_diff) * I_sum) / (beta**2)

        # See formula (10)
        y_cpx: complex = expbeta * prev_cpx_y + 2 * B * I
        y[n] = y_cpx.real

        prev_x = x[n]
        prev_cpx_y = y_cpx
        prev_x_diff = x_diff

    return y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script with optional arguments")

    # Define the mode argument as a choice between "psd", "metrics", and "sweep"
    parser.add_argument(
        "mode",
        choices=["psd", "metrics", "sweep"],
        help="Choose a mode: psd, metrics, or sweep",
    )

    # Define the export argument as a choice between "snr", "sinad", and "both"
    parser.add_argument(
        "--export",
        choices=["snr", "sinad", "both", "none"],
        default="both",
        help="Choose what to export: snr, sinad, or both (default)",
    )
    parser.add_argument("--export-dir", type=Path, default=Path.cwd())
    parser.add_argument(
        "--export-audio", action="store_true", help="Export the generated audio"
    )
    parser.add_argument(
        "--export-phase", action="store_true", help="Export the generated phase vectors"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enabled logging")
    parser.add_argument(
        "-F",
        "--flip",
        action="store_true",
        help="Flip the generated phase vector to test backward playback",
    )

    args = parser.parse_args()

    if args.mode != "metrics":
        args.export = "none"

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    import matlab.engine

    FREQS = np.logspace(start=5, stop=14.4, num=200, base=2)
    # FREQS = [2007]
    # FREQS = [noteToFreq(i) for i in range(128)]

    ALGOS_OPTIONS = [
        AlgorithmDetails(Algorithm.NAIVE, 1, 0),
        # AlgorithmDetails(Algorithm.NAIVE, 2, 0),
        # AlgorithmDetails(Algorithm.NAIVE, 4, 0),
        AlgorithmDetails(Algorithm.NAIVE, 8, 0),
        # AlgorithmDetails(Algorithm.ADAA_BUTTERWORTH, 1, 2),
        AlgorithmDetails(Algorithm.ADAA_BUTTERWORTH, 1, 2, mipmap=True),
        # AlgorithmDetails(Algorithm.ADAA_BUTTERWORTH, 1, 2, mipmap=False, waveform_len=1024),
        # AlgorithmDetails(Algorithm.ADAA_BUTTERWORTH, 1, 2, mipmap=False, waveform_len=512),
        # AlgorithmDetails(Algorithm.ADAA_BUTTERWORTH, 1, 2, mipmap=False, waveform_len=256),
        # AlgorithmDetails(Algorithm.ADAA_BUTTERWORTH, 1, 2, mipmap=False, waveform_len=128),
        # AlgorithmDetails(Algorithm.ADAA_BUTTERWORTH, 1, 2, mipmap=False, waveform_len=64),
        # AlgorithmDetails(Algorithm.ADAA_BUTTERWORTH, 1, 2, mipmap=False, waveform_len=32),
        AlgorithmDetails(Algorithm.ADAA_CHEBYSHEV_TYPE2, 1, 8, mipmap=True),
    ]
    sorted_bl = dict()

    # Prepare parallel run
    logging.info("Generating caches for computation")
    routine_args = []
    mipmap_caches = dict()
    adaa_caches: Dict[int, AdaaCache] = dict()
    naive_caches: Dict[int, NaiveCache] = dict()

    # waveform = FileWaveform("wavetables/massacre.wav")
    waveform = NaiveWaveform(NaiveWaveform.Type.SAW, 2048, 44100)

    if args.export != "none" or args.export_audio or args.export_phase:
        args.export_dir.mkdir(parents=True, exist_ok=True)

    # Init caches
    if any(opt.mipmap and opt.algorithm != Algorithm.NAIVE for opt in ALGOS_OPTIONS):
        # Init mipmap cache
        scale = mipmap_scale(waveform.size, SAMPLERATE, 9)
        mipmap_waveforms = [
            waveform.get(waveform.size / (2**i)) for i in range(len(scale) + 1)
        ]
        (m, q, m_diff, q_diff) = mipmap_mq_from_waveform(mipmap_waveforms)
        mipmap_cache = MipMapAdaaCache(m, q, m_diff, q_diff, scale)

    if any(
        not opt.mipmap and opt.algorithm != Algorithm.NAIVE for opt in ALGOS_OPTIONS
    ):
        # Init basic adaa cache
        for opt in ALGOS_OPTIONS:
            if opt.waveform_len not in adaa_caches:
                (m, q, m_diff, q_diff) = mq_from_waveform(
                    waveform.get(opt.waveform_len)
                )
                adaa_caches[opt.waveform_len] = AdaaCache(m, q, m_diff, q_diff)

    if any(opt.algorithm == Algorithm.NAIVE for opt in ALGOS_OPTIONS):
        # Init naive cache
        for opt in ALGOS_OPTIONS:
            if opt.waveform_len not in naive_caches:
                naive_caches[opt.waveform_len] = NaiveCache(
                    waveform.get(opt.waveform_len)
                )

    if args.mode == "sweep":
        for options in ALGOS_OPTIONS:
            x = generate_sweep_phase(
                20, SAMPLERATE / 2, DURATION_S, SAMPLERATE * options.oversampling
            )
            postfix = ""
            if args.flip:
                x = np.flip(x)
                postfix = "_flipped"

            if args.export_phase:
                filename = options.name + postfix + "_phase.wav"
                x_red = x % 1.0
                sf.write(
                    args.export_dir / filename,
                    x_red,
                    samplerate=SAMPLERATE,
                    subtype="FLOAT",
                )

            if options.mipmap and options.algorithm != Algorithm.NAIVE:
                cache = mipmap_cache
            elif not options.mipmap and options.algorithm != Algorithm.NAIVE:
                cache = adaa_caches[options.waveform_len]
            else:
                cache = naive_caches[options.waveform_len]
            routine_args.append([options, x, cache])
    else:
        if args.mode == "psd" and len(FREQS) > 1:
            logging.error("PSD mode only support a single frequency value")
            exit(1)

        bl_gen_args = [
            [
                np.linspace(
                    0, DURATION_S, num=int(DURATION_S * SAMPLERATE), endpoint=False
                ),
                freq,
            ]
            for freq in FREQS
        ]
        logging.info("Computing band limited versions")
        with Pool(NUM_PROCESS) as pool:
            bl_results = pool.starmap(bl_sawtooth, bl_gen_args)

        for i, freq in enumerate(FREQS):
            sorted_bl[freq] = bl_results[i]
            for options in ALGOS_OPTIONS:
                if options.mipmap and options.algorithm != Algorithm.NAIVE:
                    cache = mipmap_cache
                elif not options.mipmap and options.algorithm != Algorithm.NAIVE:
                    cache = adaa_caches[options.waveform_len]
                else:
                    cache = naive_caches[options.waveform_len]

                num_frames = int(DURATION_S * SAMPLERATE * options.oversampling)
                x = np.linspace(0.0, DURATION_S * freq, num_frames, endpoint=True)

                postfix = ""
                if args.flip:
                    x = np.flip(x)
                    postfix = "_flipped"

                if args.export_phase:
                    filename = options.name + postfix + "_phase.wav"
                    x_red = x % 1.0
                    sf.write(
                        args.export_dir / filename,
                        x_red,
                        samplerate=SAMPLERATE,
                        subtype="FLOAT",
                    )
                # x = np.flip(x)
                # x = generate_sweep_phase(200, SAMPLERATE / 2, DURATION_S, SAMPLERATE * options.oversampling)
                routine_args.append([options, x, cache, freq])

    # Run generating in parallel
    logging.info(
        "Computing naive and ADAA iterations using {} process".format(NUM_PROCESS)
    )
    with Pool(NUM_PROCESS) as pool:
        results = pool.starmap(routine, routine_args)
    logging.info("Computation ended, cleaning caches")

    # Delete caches to free memory
    del mipmap_caches
    del adaa_caches
    del naive_caches

    if args.mode == "metrics":
        import matlab.engine

        engine = matlab.engine.start_matlab()

        # Compute SNRs
        logging.info("Computing SNRs")
        for res in results:
            num_harmonics = max(2, floor(SAMPLERATE / 2 / res[1]))
            res.append(engine.snr(res[2], SAMPLERATE, num_harmonics))

    elif args.mode == "psd":
        logging.info("Plotting psd")
        signals = {name: signal for name, _, signal in results}
        plot_psd(signals)

    else:
        assert args.mode == "sweep"
        logging.info("Plotting sweep spectrogram")
        signals = {name: signal for name, _, signal in results}
        plot_specgram(signals)

    if args.export_audio:
        for res in results:
            filename = res[0] + ".wav"
            sf.write(
                args.export_dir / filename, res[2] * 0.50, SAMPLERATE, subtype="FLOAT"
            )

    # Write to CSV
    if args.export in ("snr", "both"):
        assert args.export_dir is not None

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
            [normalized_fft(data), normalized_fft(sorted_bl[freq]), freq]
            for (_, freq, data, _) in results
        ]
        logging.info("Computing SINADs")
        sinad_values = [fast_compute_sinad(*args) for args in sinad_args]

        for i, (_, freq, _, _) in enumerate(results):
            sorted_sinad[freq].append(sinad_values[i])

        logging.info("Exporting SINAD to CSV")
        sinad_output = args.export_dir / (args.export_dir.name + "_sinad.csv")
        with open(sinad_output, "w") as csv_file:
            csvwriter = csv.writer(csv_file)
            names = [opt.name for opt in ALGOS_OPTIONS]
            csvwriter.writerow(["frequency", *names])

            for freq, sinad_values in sorted_sinad.items():
                csvwriter.writerow([freq, *sinad_values])
