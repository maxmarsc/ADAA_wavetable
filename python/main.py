
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

from bl_waveform import bl_sawtooth
from decimator import Decimator17, Decimator9

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

WAVEFORM_LEN = 4096
SAMPLERATE = 44100
BUTTERWORTH_CTF = 0.45 * SAMPLERATE
CHEBY_CTF = 0.61 * SAMPLERATE
DURATION_S = 1.0
CSV_OUTPUT = "benchmark.csv"

matplotlib.use('TkAgg')
# logging.info("Starting matlab")
# MATLAB = matlab.engine.start_matlab()
# logging.info("matlab started")


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

def noteToFreq(note):
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

def compute_m(x0, x1, y0, y1):
    return (y1 - y0) / (x1 - x0)

def compute_q(x0, x1, y0, y1):
    return (y0 * (x1 - x0) - x0 * (y1 - y0)) / (x1 - x0)

def compute_m_q_vectors(waveform, X):
    """
    Compute the m & q vectors needed by the paper algorithm. In the paper they have waveformq they know the shape of in 
    advance. This function allows you to compute m & q for any kind of waveform

    Notes that when the slope is too big (threshold is empiric), this replicate the previous m & q to mimic the
    m & q definitions found in the paper (ie : in the paper they do have non-linearities in m & q).
    """
    size = waveform.shape[0]
    slope_thrsd = size / 2
    m = np.zeros(size)
    q = np.zeros(size)

    idx_to_estimate = []

    for i in range(size - 1):
        y0 = waveform[i]
        y1 = waveform[i+1]
        x0 = X[i]
        x1 = X[i+1]
        m_i = compute_m(x0, x1, y0, y1)
        q_i = compute_q(x0, x1, y0, y1)

        if abs(m_i) > slope_thrsd:
            idx_to_estimate.append(i)
            continue

        m[i] = m_i
        q[i] = q_i

    m[-1] = np.single(compute_m(np.double(X[-2]), np.double(X[-1]), np.double(waveform[-1]), np.double(waveform[0])))
    q[-1] = np.single(compute_q(np.double(X[-2]), np.double(X[-1]), np.double(waveform[-1]), np.double(waveform[0])))
    if abs(m[-1]) > slope_thrsd:
        j = size-2
        while j in idx_to_estimate:
            j -= 1
        m[-1] = m[j]
        q[-1] = q[j]

    for i in idx_to_estimate:
        m[i] = m[i-1]
        q[i] = q[i-1]

    
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

    phase = 0.0
    prev_x = 0.0

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
            # a = (waveform[next_idx] - waveform[prev_idx]) / (next_idx - prev_x)
            # b = (waveform[next_idx] * (next_idx - prev_x) - prev_x * (waveform[next_idx] - waveform[prev_idx])) / (next_idx - prev_x)
            # y[i] = a * x + b

        prev_x = x
    
    return y

def snr(noised_signal_fft, perfect_signal_fft):
    magnitude_noise = np.abs(noised_signal_fft) - np.abs(perfect_signal_fft)
    magnitude_noise[magnitude_noise< 0.0] = 0.0
    noise_rms = np.sqrt(np.mean(magnitude_noise**2))
    signal_rms =  np.sqrt(np.mean(np.abs(perfect_signal_fft)**2))
    return 10*np.log10(signal_rms/noise_rms)

def normalized_fft(time_signal):
    # signal_len = 4096
    signal_len = time_signal.shape[0]
    window = windows.blackman(signal_len)
    fft = np.fft.rfft(time_signal[:signal_len] * window)
    # fft = np.fft.rfft(time_signal)
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

def mq_from_waveform(waveform: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(0, 1, waveform.shape[0] + 1, endpoint=True)

    (m, q) =  compute_m_q_vectors(waveform, x)
    m_diff = np.zeros(m.shape[0])
    q_diff = np.zeros(q.shape[0])
    
    for i in range(m.shape[0] - 1):
        m_diff[i] = m[i+1] - m[i]
        q_diff[i] = q[i+1] - q[i]
    m_diff[-1] = m[0] - m[-1]
    q_diff[-1] = q[0] - q[-1] - m[0] * x[-1]

    return (m, q, m_diff, q_diff)

def compute_snr(noised_signal, clean_signal) -> float:
    noised_fft = normalized_fft(noised_signal)
    clean_fft = normalized_fft(clean_signal)

    return snr(noised_fft, clean_fft)

# def compute_snr_matlab(signal, samplerate, engine) -> float:
#     value = engine.snr(signal, samplerate)
#     return float(value)

from enum import Enum
class Algorithm(Enum):
    ADAA_BUTTERWORTH = 1
    ADAA_CHEBYSHEV_TYPE2 = 2
    NAIVE = 3

def process_adaa(x: np.ndarray, m: np.ndarray, q:np.ndarray, m_diff: np.ndarray,
                 q_diff: np.ndarray, ftype: Algorithm,
                 forder:int, os_factor: int) -> Tuple[np.ndarray, str]:
    sr = SAMPLERATE * os_factor
    X = np.linspace(0, 1, WAVEFORM_LEN + 1, endpoint=True)


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
        y += process_fwd(x, ri, zi, X, m, q, m_diff, q_diff)

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

@njit
def generate_sweep_phase(f1, f2, t, fs):
    # Calculate the number of samples
    n = int(t * fs)

    freqs = np.linspace(f1, f2, n-1, endpoint=True)
    phases = np.zeros(n)

    phase = 0
    for (i, freq) in enumerate(freqs):
        step = freq / fs
        phase += step
        phases[i+1] = phase

    return phases

    # # Calculate the time values
    # T = np.linspace(0, t, n, endpoint=False)

    # # Calculate the exponential rate (k)
    # k = t * (f2 - f1) / np.log(f2 / f1)

    # # Calculate the initial frequency (L)
    # L = t / np.log(f2 / f1)

    # # Generate the phase values of the sine sweep
    # phase = k * (np.exp(T / L) - 1)

    # return phase

# def write_snr_to_cs
# def main2():
    # num_frames_total = int(DURATION_S * SAMPLERATE)*
# def work_at_freq(freqs: List[float], csv :bool, ovs : int = 1, write: bool = False):
#     num_frames = int(DURATION_S * SAMPLERATE)
#     for freq in freqs:
#         x = np.linspace(0.0, num_frames*freq / SAMPLERATE, num_frames, endpoint=True)
#         (m, q, m_diff, q_diff) = mq_from_waveform(waveform) 

@dataclass
class AlgorithmDetails:
    algorithm: Algorithm
    oversampling : int
    forder: int

    @property
    def name(self) -> str:
        name = ""
        if self.algorithm is Algorithm.NAIVE:
            name += "naive"
        else:
            name += "ADAA"
            if self.algorithm is Algorithm.ADAA_BUTTERWORTH:
                name += "_BT"
            else:
                name += "_CH"
            name += "_order_{}".format(self.forder)
        
        if self.oversampling > 1:
            name += "_OVSx{}".format(self.oversampling)
        
        return name
    
    def name_with_freq(self, freq: int) -> str:
        return self.name + "_{}Hz".format(freq)

# def main():


def routine(details: AlgorithmDetails, x, freq: int) -> Tuple[str, int, np.ndarray[float]]:
    waveform = NAIVE_SAW
    name = details.name_with_freq(freq)
    # print("Computing {}".format(name))
    logging.info("{} : started".format(name))
    if details.algorithm is Algorithm.NAIVE:
        generated = process_naive(x, waveform, details.oversampling)[0]
    else:
        (m, q, m_diff, q_diff) = mq_from_waveform(waveform)
        generated = process_adaa(x, m, q, m_diff, q_diff, details.algorithm, details.forder, os_factor=details.oversampling)[0]

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

    

def process_for_freq(play_freq:float, log: bool, to_csv: bool):
    # play_freq = 200

    num_frames_total = int(DURATION_S * SAMPLERATE)
    num_frames_total_x2 = int(DURATION_S * SAMPLERATE * 2)
    num_frames_total_x4 = int(DURATION_S * SAMPLERATE * 4)
    num_frames_total_x8 = int(DURATION_S * SAMPLERATE * 8)

    # ======== Creating phase vectors ========
    """
    In the algorithm, phase is not 2*pi cycling but 1.0 cycling, so every 1.0
    the waveform itself
    """
    # x = np.linspace(0.0, num_frames_total*play_freq / SAMPLERATE, num_frames_total, endpoint=True)
    # x_x2 = np.linspace(0.0, num_frames_total*play_freq / SAMPLERATE, num_frames_total_x2, endpoint=True)
    # x_x4 = np.linspace(0.0, num_frames_total*play_freq / SAMPLERATE, num_frames_total_x4, endpoint=True)
    # x_x8 = np.linspace(0.0, num_frames_total*play_freq / SAMPLERATE, num_frames_total_x8, endpoint=True)

    x = np.linspace(0.0, num_frames_total*play_freq / SAMPLERATE, num_frames_total, endpoint=True)
    x_sweep = generate_sweep_phase(1, SAMPLERATE / 2, DURATION_S, SAMPLERATE)
    x_sweep_x8 = generate_sweep_phase(1, SAMPLERATE / 2, DURATION_S, SAMPLERATE * 8)


    x_x2 = np.linspace(0.0, num_frames_total*play_freq / SAMPLERATE, num_frames_total_x2, endpoint=True)
    x_x4 = np.linspace(0.0, num_frames_total*play_freq / SAMPLERATE, num_frames_total_x4, endpoint=True)
    x_x8 = np.linspace(0.0, num_frames_total*play_freq / SAMPLERATE, num_frames_total_x8, endpoint=True)

    y_bl = bl_sawtooth(np.linspace(0, DURATION_S, num = num_frames_total, endpoint = False), play_freq)

    # waveform = compute_naive_sin(WAVEFORM_LEN)
    waveform = NAIVE_SAW

    (m, q, m_diff, q_diff) = mq_from_waveform(waveform)

    # print("Phase sweep\n{}\n{}".format(x_sweep[:8], x_sweep[-8:]))

    y_naive = process_naive(x, waveform, 1)
    y_naive_x8 = process_naive(x_x8, waveform, 8)
    y_bt2 = process_adaa(x, m, q, m_diff, q_diff, Algorithm.ADAA_BUTTERWORTH, 2, 1)
    y_ch10 = process_adaa(x, m, q, m_diff, q_diff, Algorithm.ADAA_CHEBYSHEV_TYPE2, 10, 1)



    # fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    gain = 0.75
    # # print(y_naive_name.replace(" ", "_") + ".wav")
    sf.write(y_naive[1].replace(" ", "_") + "_{}Hz".format(play_freq) + ".wav", y_naive[0] * gain, SAMPLERATE)
    # # print(y_naive_x8_name.replace(" ", "_") + ".wav")
    sf.write(y_naive_x8[1].replace(" ", "_") + "_{}Hz".format(play_freq) + ".wav", y_naive_x8[0] * gain, SAMPLERATE)
    # # print(y_bt2_name.replace(" ", "_") + ".wav")
    sf.write(y_bt2[1].replace(" ", "_") + "_{}Hz".format(play_freq) + ".wav", y_bt2[0] * gain, SAMPLERATE)
    # # print(y_ch10_name.replace(" ", "_") + ".wav")
    sf.write(y_ch10[1].replace(" ", "_") + "_{}Hz".format(play_freq) + ".wav", y_ch10[0] * gain, SAMPLERATE)



    # flatten the axes array, to make it easier to iterate over
    # axs = axs.flatten()

    # for i, y in enumerate([y_naive, y_naive_x8, y_bt2, y_ch10]):
    #     # f, t, Sxx = spectrogram(y[0], SAMPLERATE, nfft=2048)
    #     # axs[i].pcolormesh(t, f, Sxx, shading='gouraud')
    #     # axs[i].ylabel('Frequency [Hz]')
    #     # axs[i].xlabel('Time [sec]')

    #     # Plot the spectrogram on the i-th subplot

    #     # pcm = axs[i].pcolormesh(t, f, 10 * np.log10(Sxx), shading='auto', cmap='inferno')
    #     axs[i].specgram(y[0], NFFT=2028, noverlap=1024)
    #     # fig.colorbar(pcm, ax=axs[i], label='Intensity [dB]')
    #     axs[i].set_ylabel('Frequency [Hz]')
    #     axs[i].set_xlabel('Time [s]')
    #     axs[i].set_title(y[1])
    #     axs[i].legend()



    # axs[0].plot(x, "green")
    # axs[0].plot(x_200, "blue")
    # axs[1].plot(x_s)

    # axs[1].set_yscale('log')

    # f, t, Sxx = signal.spectrogram(x, fs)
    # plt.figure(figsize=(8,10))
    # plt.pcolormesh(t, f, Sxx, shading='gouraud')
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')


    # for ax in axs:
    #     # ax.grid(True, which="both")
    #     ax.legend()
        # ax.set_yscale('log')

    # fig.tight_layout()
    # plt.show()

    # y2_bt = np.zeros(num_frames_total)
    # y2_bt_og = np.zeros(num_frames_total)
    # y4_bt = np.zeros(num_frames_total)
    # y2_ch = np.zeros(num_frames_total)
    # y4_ch = np.zeros(num_frames_total)
    # y10_ch = np.zeros(num_frames_total)
    # y10_ch_og = np.zeros(num_frames_total)
    # y2_ch_x2 = np.zeros(num_frames_total_x2)
    # y4_ch_x2 = np.zeros(num_frames_total_x2)
    # y4_bt_og = np.zeros(num_frames_total)
    # y2_bt_x2 = np.zeros(num_frames_total_x2)
    # y4_bt_x2 = np.zeros(num_frames_total_x2)
    # y_naive_x4 = np.zeros(num_frames_total_x4)
    # y_naive_x8 = np.zeros(num_frames_total_x8)

    # y_ch10_matlab, _ = sf.read("matlab/octave_cheby_test.wav")
    # y_naive_x4_cpp_decimated, _ = sf.read("naive_linear_f1000_decimated.wav")
    # y_naive_x2_cpp_decimated_leg, _ = sf.read("naive_linear_f1000__decimated_legacy.wav")
    # y_naive_x2_cpp_decimated_simd, _ = sf.read("naive_linear_f1000__decimated_simd.wav")


    # ======== M & Q computation (see formula 19) ========
    # (m, q) =  compute_m_q_vectors(NAIVE_SAW, NAIVE_SAW_X)
    # m_diff = np.zeros(m.shape[0])
    # q_diff = np.zeros(q.shape[0])
    
    # for i in range(m.shape[0] - 1):
    #     m_diff[i] = m[i+1] - m[i]
    #     q_diff[i] = q[i+1] - q[i]
    # m_diff[-1] = m[0] - m[-1]
    # q_diff[-1] = q[0] - q[-1] - m[0] * NAIVE_SAW_X[-1]


    # y2_bt_2 = process_adaa(x, m, q, m_diff, q_diff, FilterType.BUTTERWORTH, 2, 1)
    # y4_bt_2 = process_adaa(x, m, q, m_diff, q_diff, FilterType.BUTTERWORTH, 4, 1)
    # y10_ch_2 = process_adaa(x, m, q, m_diff, q_diff, FilterType.CHEBYSHEV_TYPE2, 10, 1)

    # fig, axs = plt.subplots(5)
    # for ax in axs:
    #     ax.grid(True, which="both")
    #     ax.legend()

    # plt.show()
    



    # ======== Computing filter coeffs ========
    # (r2_bt, p2_bt, _) = butter_coeffs(2, BUTTERWORTH_CTF, SAMPLERATE)
    # (r2_bt_x2, p2_bt_x2, _) = butter_coeffs(2, BUTTERWORTH_CTF, SAMPLERATE*2)
    # (r4_bt, p4_bt, _) = butter_coeffs(4, BUTTERWORTH_CTF, SAMPLERATE)
    # (r4_bt_x2, p4_bt_x2, _) = butter_coeffs(4, BUTTERWORTH_CTF, SAMPLERATE*2)
    # (r2_ch, p2_ch, _) = cheby_coeffs(2, 0.6* SAMPLERATE, 60, SAMPLERATE)
    # (r2_ch_x2, p2_ch_x2, _) = cheby_coeffs(2, 0.6* SAMPLERATE, 60, SAMPLERATE * 2)
    # (r4_ch, p4_ch, _) = cheby_coeffs(4, 0.6* SAMPLERATE, 60, SAMPLERATE)
    # (r4_ch_x2, p4_ch_x2, _) = cheby_coeffs(4, 0.6* SAMPLERATE, 60, SAMPLERATE * 2)
    # (r10_ch, p10_ch, _) = cheby_coeffs(10, 0.6* SAMPLERATE, 60, SAMPLERATE)

    # ======== 2nd order processing ========
    # for order in range(0, 2, 2):
    #     # Butterworth
    #     ri_bt = r2_bt[order]
    #     zi_bt = p2_bt[order]
    #     y2_bt += process_fwd(x, ri_bt, zi_bt, NAIVE_SAW_X, m, q, m_diff, q_diff)
    #     # y2_bt_og += process(x, ri_bt, zi_bt, NAIVE_SAW_X, m, q, m_diff, q_diff)

    #     # Chebyshev
    #     ri_ch = r2_ch[order]
    #     zi_ch = p2_ch[order]
    #     y2_ch += process_fwd(x, ri_ch, zi_ch, NAIVE_SAW_X, m, q, m_diff, q_diff)

    #     # Butterworth x2 oversampling
    #     ri_bt_x2 = r2_bt_x2[order]
    #     zi_bt_x2 = p2_bt_x2[order]
    #     y2_bt_x2 += process_fwd(x_x2, ri_bt_x2, zi_bt_x2, NAIVE_SAW_X, m, q, m_diff, q_diff)

    #     # Chebyshev x2 oversampling
    #     ri_ch_x2 = r2_ch_x2[order]
    #     zi_ch_x2 = p2_ch_x2[order]
    #     y2_ch_x2 += process_fwd(x_x2, ri_ch_x2, zi_ch_x2, NAIVE_SAW_X, m, q, m_diff, q_diff)
    

    # ======== 4th order processing ========
    # for order in range(0, 4, 2):
    #     #Butterworth
    #     ri_bt = r4_bt[order]
    #     zi_bt = p4_bt[order]
    #     y4_bt += process_fwd(x, ri_bt, zi_bt, NAIVE_SAW_X, m, q, m_diff, q_diff)
    #     # y4_bt_og += process(x, ri_bt, zi_bt, NAIVE_SAW_X, m, q, m_diff, q_diff)

    #     # Chebyshev
    #     ri_ch = r4_ch[order]
    #     zi_ch = p4_ch[order]
    #     y4_ch += process_fwd(x, ri_ch, zi_ch, NAIVE_SAW_X, m, q, m_diff, q_diff)

    #     # Butterworth 2x oversampling
    #     ri_bt_x2 = r4_bt_x2[order]
    #     zi_bt_x2 = p4_bt_x2[order]
    #     y4_bt_x2 += process_fwd(x_x2, ri_bt_x2, zi_bt_x2, NAIVE_SAW_X, m, q, m_diff, q_diff)

    #     # Chebyshev 2x oversampling
    #     ri_ch_x2 = r4_ch_x2[order]
    #     zi_ch_x2 = p4_ch_x2[order]
    #     y4_ch_x2 += process_fwd(x_x2, ri_ch_x2, zi_ch_x2, NAIVE_SAW_X, m, q, m_diff, q_diff)


    # ======== 10th order processing ========
    # for order in range(0, 10, 2):
    #     # Chebyshev
    #     ri_ch = r10_ch[order]
    #     zi_ch = p10_ch[order]
    #     y10_ch +=  process_fwd(x, ri_ch, zi_ch, NAIVE_SAW_X, m, q, m_diff, q_diff)
    #     y10_ch_og += process(x, ri_ch, zi_ch, NAIVE_SAW_X, m, q, m_diff, q_diff)


    # ======== Linear interpolation processing ========
    # y_naive = process_naive_linear(NAIVE_SAW, x)
    # y_naive_x2 = process_naive_linear(NAIVE_SAW, x_x2)
    # y_naive_x4 = process_naive_linear(NAIVE_SAW, x_x4)
    # y_naive_x8 = process_naive_linear(NAIVE_SAW, x_x8)
    # y_bl = bl_sawtooth(np.linspace(0, DURATION_S, num = num_frames_total, endpoint = False), play_freq)


    # ======== Downsampling the oversampled outputs ========
    # Downsample the upsampled output
    # y2_bt_ds = soxr.resample(
    #     y2_bt_x2,
    #     88200,
    #     44100,
    #     quality="HQ"
    # )
    # y4_bt_ds = soxr.resample(
    #     y4_bt_x2,
    #     88200,
    #     44100,
    #     quality="HQ"
    # )
    # y2_ch_ds = soxr.resample(
    #     y2_ch_x2,
    #     88200,
    #     44100,
    #     quality="HQ"
    # )
    # y4_ch_ds = soxr.resample(
    #     y4_ch_x2,
    #     88200,
    #     44100,
    #     quality="HQ"
    # )
    # y_naive_x2_ds = soxr.resample(
    #     y_naive_x2,
    #     44100 * 2,
    #     44100,
    #     quality="HQ"
    # )
    # y_naive_x2_ds2 = downsample_x2_decim9(y_naive_x2*0.75)      # comparing SOXR to basic decimator
    # y_naive_x4_ds = soxr.resample(
    #     y_naive_x4,
    #     44100 * 4,
    #     44100,
    #     quality="HQ"
    # )
    # y_naive_x4_ds2 = downsample_x4_decim9_17(y_naive_x4)
    # y_naive_x8_ds = soxr.resample(
    #     y_naive_x8,
    #     44100 * 8,
    #     44100,
    #     quality="HQ"
    # )


    # ======== Computing the FFTs for SNR computation ========
    # y_bl_fft = normalized_fft(y_bl)
    # y_naive_fft = normalized_fft(y_naive)
    # y2_bt_fft = normalized_fft(y2_bt)
    # y4_bt_fft = normalized_fft(y4_bt)
    # y2_ch_fft = normalized_fft(y2_ch)
    # y4_ch_fft = normalized_fft(y4_ch)
    # y10_ch_fft = normalized_fft(y10_ch)
    # y10_ch_fft_matlab = normalized_fft(y_ch10_matlab)
    # y2_ch_x2_fft = normalized_fft(y2_ch_ds)
    # y4_ch_x2_fft = normalized_fft(y4_ch_ds)
    # y4_bt_fft = normalized_fft(y4_bt)
    # y2_bt_ds_fft = normalized_fft(y2_bt_ds)
    # y4_bt_ds_fft = normalized_fft(y4_bt_ds)
    # y_naive_x2_fft = normalized_fft(y_naive_x2_ds)
    # y_naive_x4_fft = normalized_fft(y_naive_x4_ds)
    # y_naive_x8_fft = normalized_fft(y_naive_x8_ds)
    # y_naive_x2_fft2 = normalized_fft(y_naive_x2_ds2)
    # y_naive_x4_fft2 = normalized_fft(y_naive_x4_ds2)
    # y4_bt_og_fft = normalized_fft(y4_bt_og)
    # y2_bt_og_fft = normalized_fft(y2_bt_og)
    # y10_ch_fft_og = normalized_fft(y10_ch_og)
    # y_naive_x4_fft3 = normalized_fft(y_naive_x4_cpp_decimated)
    # y_naive_x2_cpp_leg_fft = normalized_fft(y_naive_x2_cpp_decimated_leg)
    # y_naive_x2_cpp_simd_fft = normalized_fft(y_naive_x2_cpp_decimated_simd)

    # y2_bt_2_fft = normalized_fft(y2_bt_2)
    # y4_bt_2_fft = normalized_fft(y4_bt_2)
    # y10_ch_2_fft = normalized_fft(y10_ch_2)



    if log:
        # ======== Computing SNR - should be done over different play_freq, not just one ========
        """
        I"m not 100% sure about the way I compute SNR, don't hesitate to take a look
        """
        # print("Play frequency : {} Hz".format(play_freq))
        # print("SNR band-limited : ", snr(y_bl_fft, y_bl_fft))
        # print("SNR naive : ", snr(y_naive_fft, y_bl_fft))

        # print("SNR ADAA BT order 2 : ", snr(y2_bt_fft, y_bl_fft))
        # print("SNR ADAA BT order 4 : ", snr(y4_bt_fft, y_bl_fft))
        # print("SNR ADAA BT order 2 v2 : ", snr(y2_bt_2_fft, y_bl_fft))
        # print("SNR ADAA BT order 4 v2 : ", snr(y4_bt_2_fft, y_bl_fft))
        # print("SNR ADAA BT order 2 OVSx2 : ", snr(y2_bt_ds_fft, y_bl_fft))
        # print("SNR ADAA BT order 4 OVSx2 : ", snr(y4_bt_ds_fft, y_bl_fft))

        # print("SNR ADAA CH order 2 : ", snr(y2_ch_fft, y_bl_fft))
        # print("SNR ADAA CH order 4 : ", snr(y4_ch_fft, y_bl_fft))
        # print("SNR ADAA CH order 10 : ", snr(y10_ch_fft, y_bl_fft))
        # print("SNR ADAA CH order 10 v2 : ", snr(y10_ch_2_fft, y_bl_fft))
        # print("SNR ADAA CH order 10 (matlab) : ", snr(y10_ch_fft_matlab, y_bl_fft))
        # print("SNR ADAA CH order 10 (OG) : ", snr(y10_ch_fft_og, y_bl_fft))
        # print("SNR ADAA CH order 2 OVSx2 : ", snr(y2_ch_x2_fft, y_bl_fft))
        # print("SNR ADAA CH order 4 OVSx2 : ", snr(y4_ch_x2_fft, y_bl_fft))

        # print("SNR naive OVSx2 : ", snr(y_naive_x2_fft, y_bl_fft))
        # print("SNR naive OVSx4 : ", snr(y_naive_x4_fft, y_bl_fft))
        # print("SNR naive OVSx8 : ", snr(y_naive_x8_fft, y_bl_fft))
        # print("SNR naive OVSx2 decimator9 : ", snr(y_naive_x2_fft2, y_bl_fft))
        # print("SNR naive OVSx4 decimator9_17: ", snr(y_naive_x4_fft2, y_bl_fft))
        # print("SNR naive OVSx4 NE10: ", snr(y_naive_x4_fft3, y_bl_fft))
        # print("SNR naive OVSx2 Legacy: ", snr(y_naive_x2_cpp_leg_fft, y_bl_fft))
        # print("SNR naive OVSx4 SIMD: ", snr(y_naive_x2_cpp_simd_fft, y_bl_fft))


        # fig, axs = plt.subplots(5)
        # sample_offset = 0

        # ======== Displaying PSD ========
        """
        Because I test too many different processing, I cannot print them all at once, it's unreadable.
        That's why many are commented out
        """
        # axs[0].psd(y_naive[sample_offset:], Fs=SAMPLERATE, NFFT=4096, color="red", label="naive-linear")
        # axs[0].psd(y2_bt[sample_offset:], Fs=SAMPLERATE, NFFT=4096, color="blue", label="ADAA-butterworth-2")
        # axs[0].psd(y4_bt[sample_offset:], Fs=SAMPLERATE, NFFT=4096, color="blue", label="ADAA-butterworth-4")
        # axs[2].psd(y2_ch[sample_offset:], Fs=SAMPLERATE, NFFT=4096, color="green", label="ADAA-chebyshev-2")
        # axs[0].psd(y_naive_x4_ds2, Fs=SAMPLERATE, NFFT=4096, color="black", label="Decimator 4")
        # axs[2].psd(y_naive_x2_ds2, Fs=SAMPLERATE, NFFT=4096, color="black", label="Decimator 2")
        # axs[3].psd(y4_ch[sample_offset:], Fs=SAMPLERATE, NFFT=4096, color="black", label="ADAA-chebyshev-4")
        # axs[4].psd(y2_ch_ds[sample_offset:], Fs=SAMPLERATE, NFFT=4096, color="purple", label="ADAA-chebyshev-2 x2")
        # axs[5].psd(y4_ch_ds[sample_offset:], Fs=SAMPLERATE, NFFT=4096, color="orange", label="ADAA-chebyshev-4 x2")
        # axs[2].psd(y2_ds[sample_offset:], Fs=SAMPLERATE, NFFT=4096, color="green", label="ADAA-butterworth-2  x2")
        # axs[3].psd(y4[sample_offset:], Fs=SAMPLERATE, NFFT=4096, color="black", label="ADAA-butterworth-2")
        # axs[2].psd(y_naive_x4_ds[sample_offset:], Fs=SAMPLERATE, NFFT=4096, color="green", label="naive x4")
        # axs[0].psd(y10_ch_og[sample_offset:], Fs=SAMPLERATE, NFFT=4096, color="red", label="ADAA-cheby-10 (OG)")
        # axs[2].psd(y_ch10_matlab, Fs=SAMPLERATE, NFFT=4096, color="green", label="ADAA-cheby-10 (matlab)")
        # axs[3].psd(y10_ch[sample_offset:], Fs=SAMPLERATE, NFFT=4096, color="black", label="ADAA-cheby-10")
        # axs[5].psd(y10_ch_og, Fs=SAMPLERATE, NFFT=4096, color="red", label="cheby 10 [OG]")
        # axs[2].psd(y_naive_x4_ds[sample_offset:], Fs=SAMPLERATE, NFFT=4096, color="purple", label="naive x4")
        # axs[4].psd(y_naive_x2_ds[sample_offset:], Fs=SAMPLERATE, NFFT=4096, color="purple", label="naive x2")
        # axs[1].psd(y_naive_x8_ds[sample_offset:], Fs=SAMPLERATE, NFFT=4096, color="purple", label="naive x8")
        # axs[6].psd(y2_ds[sample_offset:], Fs=SAMPLERATE, NFFT=4096, color="pink", label="ADAA-butterworth-2-OVS")
        # axs[7].psd(y4_ds[sample_offset:], Fs=SAMPLERATE, NFFT=4096, color="orange", label="ADAA-butterworth-4-OVS")
        # axs[5].psd(y_bl[sample_offset:], Fs=SAMPLERATE, NFFT=4096, color="purple", label="band limited")



        # ======== Displaying waveform, usefull to debug algorithm and identify latency ========
        # axs[4].plot(x[sample_offset:], y_naive[sample_offset:], 'red', label="naive-linear")
        # axs[4].plot(x[sample_offset:], y2_bt[sample_offset:], 'b', label="ADAA-butterworth-2")
        # axs[4].plot(x, y_naive_x2_ds2, "r", label="decimator x2")
        # axs[4].plot(x, y_naive_x4_ds2, "green", label="decimator x4")
        # axs[4].plot(x, y_naive_x4_ds, "green", label="naive x4")
        # axs[4].plot(x, y_naive_x4_cpp_decimated, "purple", label="NE10")
        # axs[4].plot(x[sample_offset:], y4_bt[sample_offset:], 'blue', label="ADAA-butterworth-4")
        # axs[4].plot(x[sample_offset:], y_ch10_matlab[sample_offset:], 'blue', label="ADAA-butterworth-10 (matlab)")
        # axs[4].plot(x[sample_offset:], y10_ch[sample_offset:], 'black', label="ADAA-butterworth-10")
        # axs[4].plot(x[sample_offset:], y4_bt_ds[sample_offset:], 'purple', label="ADAA-butterworth-4-OVS")
        # axs[4].plot(x[sample_offset:], y_naive[sample_offset:], 'green', label="naive-linear")
        # axs[4].plot(x[sample_offset:], y_bl[sample_offset:], 'r', label="band_limited")
        # axs[4].plot(x[sample_offset:], y2_bt_og[sample_offset:], 'purple', label="[OG] ADAA-butterworth-2")

        # for ax in axs:
        #     ax.grid(True, which="both")
        #     ax.legend()

        # plt.show()
    
    # if to_csv:
    #     with open(CSV_OUTPUT, "a") as csv_file:
    #         csvwriter = csv.writer(csv_file)
    #         values = [ 
    #             snr(y_naive_fft, y_bl_fft),
    #             snr(y2_bt_fft, y_bl_fft),
    #             snr(y4_bt_fft, y_bl_fft),
    #             snr(y2_bt_ds_fft, y_bl_fft),
    #             snr(y4_bt_ds_fft, y_bl_fft),
    #             snr(y2_ch_fft, y_bl_fft),
    #             snr(y4_ch_fft, y_bl_fft),
    #             snr(y10_ch_fft, y_bl_fft),
    #             snr(y10_ch_fft_og, y_bl_fft),
    #             snr(y2_ch_x2_fft, y_bl_fft),
    #             snr(y4_ch_x2_fft, y_bl_fft),
    #             snr(y_naive_x2_fft, y_bl_fft),
    #             snr(y_naive_x4_fft, y_bl_fft),
    #             snr(y_naive_x8_fft, y_bl_fft),
    #             snr(y_naive_x2_fft2, y_bl_fft)
    #         ]
    #         csvwriter.writerow(values)





def mod_bar(x, k):
    m = x % k
    return m + k * (1 - np.sign(m))

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

     # Define the mode argument as a choice between "psd", "snr", and "sweep"
    parser.add_argument("mode", choices=["psd", "snr", "sweep"],
                        help="Choose a mode: psd, snr, or sweep")

    # Define the export argument as a choice between "snr", "thd", and "both"
    parser.add_argument("--export", choices=["snr", "thd", "both", "none"], default="both",
                        help="Choose what to export: snr, thd, or both (default)")
    parser.add_argument("--export-dir", type=Path, default=Path.cwd())
    parser.add_argument("--export-audio", action="store_true")
    parser.add_argument("--no-log", action="store_true", help="Disable console logging")


    # parser.add_argument("destination", nargs="?", default=Path.cwd(), type=Path,
    #                     help="Output destination")
    # parser.add_argument("--csv", action="store_true", help="Enable output in CSV format")

    args = parser.parse_args()

    import matlab.engine
    future_engine = matlab.engine.start_matlab(background=True)
    
    # FREQS = [197, 397, 597, 997, 1599, 2173, 3003, 3997]
    # FREQS = np.int32(np.logspace(start=5, stop=14, num=100, base=2))
    # FREQS = np.int32(np.linspace(start=32, stop=3500, num=200))
    FREQS = [noteToFreq(i) for i in range(21, 109)]
    # FREQS = range(3643, 3733)

    # FREQS = np.linspace(60, )
    # FREQS = [120]
    # print(FREQS)
    ALGOS_OPTIONS = [
        AlgorithmDetails(Algorithm.NAIVE, 1, 0),
        # AlgorithmDetails(Algorithm.NAIVE, 4, 0),
        AlgorithmDetails(Algorithm.NAIVE, 8, 0),
        AlgorithmDetails(Algorithm.ADAA_BUTTERWORTH, 1, 2),
        # AlgorithmDetails(Algorithm.ADAA_BUTTERWORTH, 1, 4),
        AlgorithmDetails(Algorithm.ADAA_CHEBYSHEV_TYPE2, 1, 10),
        AlgorithmDetails(Algorithm.ADAA_BUTTERWORTH, 2, 2),
        # AlgorithmDetails(Algorithm.ADAA_BUTTERWORTH, 2, 4),
        AlgorithmDetails(Algorithm.ADAA_CHEBYSHEV_TYPE2, 2, 10),
    ]
    sorted_bl = dict()


    # OVS_FACTORS = [1, 2, 4, 8]

    # Prepare parallel run
    logging.info("Setting up phase vectors")
    routine_args = []
    # names = []
    for freq in FREQS:
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

        sorted_bl[freq] = bl_sawtooth(np.linspace(0, DURATION_S, num = int(DURATION_S * SAMPLERATE), endpoint = False), freq)
        for options in ALGOS_OPTIONS:
            num_frames = int(DURATION_S * SAMPLERATE * options.oversampling)
            x = np.linspace(0.0, DURATION_S*freq, num_frames, endpoint=True)
            routine_args.append([options, x, freq])
            # names.append(options.name() + "_{}Hz".format(freq))
    
    # Run generating in parallel
    with Pool(19) as pool:
        results = pool.starmap(routine, routine_args)


    # names = [res[0] for res in results]

    if args.mode == "snr":
        engine = future_engine.result()

        # Compute SNRs
        logging.info("Computing SNRs")
        for res in results:
            num_harmonics = max(2, floor(SAMPLERATE / 2 / res[1]))
            # print(res[0], res[2].shape)
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
        # TODO: sweep
        pass

    if args.export != "none" or args.export_audio:
        args.export_dir.mkdir(parents=True, exist_ok=True)

    if args.export_audio:
        for [name, freq, data, snr_value] in results:
            filename = name + ".wav"
            sf.write(args.export_dir / filename, data, SAMPLERATE)

    # Write to CSV
    if args.export in ("snr", "both"):
        assert(args.export_dir is not None)

        logging.info("Exporting SNR to CSV")
        # Sort data for csv
        sorted_results = defaultdict(list)
        for [name, freq, data, snr_value] in results:
            sorted_results[freq].append(snr_value)

        csv_output = args.export_dir / CSV_OUTPUT
        with open(csv_output, "w") as csv_file:
            csvwriter = csv.writer(csv_file)
            names = [opt.name for opt in ALGOS_OPTIONS]
            csvwriter.writerow(["frequency", *names])

            for freq, snr_values in sorted_results.items():
                csvwriter.writerow([freq, *snr_values])

    if args.export in ("thd", "both"):
        sorted_thd = defaultdict(list)
        for [name, freq, data, snr_value] in results:
            sorted_thd[freq].append(compute_snr(data, sorted_bl[freq]))

        thd_output = args.export_dir / "thd.csv"
        with open(thd_output, "w") as csv_file:
            csvwriter = csv.writer(csv_file)
            names = [opt.name for opt in ALGOS_OPTIONS]
            csvwriter.writerow(["frequency", *names])

            for freq, thd_values in sorted_thd.items():
                csvwriter.writerow([freq, *thd_values])

        

    

    


    # with open(CSV_OUTPUT, "w") as csv_file:
    #     csvwriter = csv.writer(csv_file)
    #     names = [
    #         "naive",
    #         "ADAA BT order 2",
    #         "ADAA BT order 4",
    #         "ADAA BT order 2 OVSx2",
    #         "ADAA BT order 4 OVSx2",
    #         "ADAA CH order 2",
    #         "ADAA CH order 4",
    #         "ADAA CH order 10",
    #         "ADAA CH order 10 (OG)",
    #         "ADAA CH order 2 OVSx2",
    #         "ADAA CH order 4 OVSx2",
    #         "naive OVSx2",
    #         "naive OVSx4",
    #         "naive OVSx8",
    #         "naive OVSx4 decimator9_17"
    #     ]
    #     csvwriter.writerow(names)

    # for (i, freq) in enumerate(np.logspace(5, 14, num=50, base=2.0)):
    #     print("Calling main with {}:{}".format(i, freq))
    #     main(freq, False, True)

    # for freq in [200, 600, 1000, 4000]:
    #     main(play_freq=freq)
    # main()