
import numpy as np
from math import exp, floor
from cmath import exp as cexp
from scipy.signal import welch, butter, cheby2, residue, freqs, zpk2tf
import matplotlib.pyplot as plt
import matplotlib
import soxr
# import soundfile as sf

from typing import Tuple, List

from bl_waveform import bl_sawtooth
from decimator import Decimator17, Decimator9

WAVEFORM_LEN = 2048
SAMPLERATE = 44100
BUTTERWORTH_CTF = 0.45 * SAMPLERATE
DURATION_S = 0.1


matplotlib.use('TkAgg')

def compute_naive_saw(frames: int) -> np.ndarray:
    phase = 0.0
    waveform = np.zeros(frames)
    step = 1.0/frames

    for i in range(frames):
        waveform[i] = 2.0 * phase - 1
        phase = (phase + step) % 1.0

    return waveform


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
            a = (waveform[next_idx] - waveform[prev_idx]) / (next_idx - prev_x)
            b = (waveform[next_idx] * (next_idx - prev_x) - prev_x * (waveform[next_idx] - waveform[prev_idx])) / (next_idx - prev_x)
            y[i] = a * x + b

        prev_x = x
    
    return y

def snr(noised_signal_fft, perfect_signal_fft):
    magnitude_noise = np.abs(noised_signal_fft) - np.abs(perfect_signal_fft)
    noise_rms = np.sqrt(np.mean(magnitude_noise**2))
    signal_rms =  np.sqrt(np.mean(np.abs(perfect_signal_fft)**2))
    return 10*np.log10(signal_rms/noise_rms)

def normalized_fft(time_signal):
    fft = np.fft.rfft(time_signal, n=4096)
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


def main():
    play_freq = 1000

    num_frames_total = int(DURATION_S * SAMPLERATE)
    num_frames_total_x2 = int(DURATION_S * SAMPLERATE * 2)
    num_frames_total_x4 = int(DURATION_S * SAMPLERATE * 4)
    num_frames_total_x8 = int(DURATION_S * SAMPLERATE * 8)

    # ======== Creating phase vectors ========
    """
    In the algorithm, phase is not 2*pi cycling but 1.0 cycling, so every 1.0
    the waveform itself
    """
    x = np.linspace(0.0, num_frames_total*play_freq / SAMPLERATE, num_frames_total, endpoint=True)
    x_x2 = np.linspace(0.0, num_frames_total*play_freq / SAMPLERATE, num_frames_total_x2, endpoint=True)
    x_x4 = np.linspace(0.0, num_frames_total*play_freq / SAMPLERATE, num_frames_total_x4, endpoint=True)
    x_x8 = np.linspace(0.0, num_frames_total*play_freq / SAMPLERATE, num_frames_total_x8, endpoint=True)
    y2_bt = np.zeros(num_frames_total)
    y2_bt_og = np.zeros(num_frames_total)
    y4_bt = np.zeros(num_frames_total)
    y2_ch = np.zeros(num_frames_total)
    y4_ch = np.zeros(num_frames_total)
    y10_ch = np.zeros(num_frames_total)
    y10_ch_og = np.zeros(num_frames_total)
    y2_ch_x2 = np.zeros(num_frames_total_x2)
    y4_ch_x2 = np.zeros(num_frames_total_x2)
    y4_bt_og = np.zeros(num_frames_total)
    y2_bt_x2 = np.zeros(num_frames_total_x2)
    y4_bt_x2 = np.zeros(num_frames_total_x2)
    y_naive_x4 = np.zeros(num_frames_total_x4)
    y_naive_x8 = np.zeros(num_frames_total_x8)

    # y_ch10_matlab, _ = sf.read("matlab/octave_cheby_test.wav")
    # y_naive_x4_cpp_decimated, _ = sf.read("naive_linear_f1000_decimated.wav")
    # y_naive_x2_cpp_decimated_leg, _ = sf.read("naive_linear_f1000__decimated_legacy.wav")
    # y_naive_x2_cpp_decimated_simd, _ = sf.read("naive_linear_f1000__decimated_simd.wav")

    # ======== M & Q computation (see formula 19) ========
    (m, q) =  compute_m_q_vectors(NAIVE_SAW, NAIVE_SAW_X)
    m_diff = np.zeros(m.shape[0])
    q_diff = np.zeros(q.shape[0])
    
    for i in range(m.shape[0] - 1):
        m_diff[i] = m[i+1] - m[i]
        q_diff[i] = q[i+1] - q[i]
    m_diff[-1] = m[0] - m[-1]
    q_diff[-1] = q[0] - q[-1] - m[0] * NAIVE_SAW_X[-1]


    # ======== Computing filter coeffs ========
    (r2_bt, p2_bt, _) = butter_coeffs(2, BUTTERWORTH_CTF, SAMPLERATE)
    (r2_bt_x2, p2_bt_x2, _) = butter_coeffs(2, BUTTERWORTH_CTF, SAMPLERATE*2)
    (r4_bt, p4_bt, _) = butter_coeffs(4, BUTTERWORTH_CTF, SAMPLERATE)
    (r4_bt_x2, p4_bt_x2, _) = butter_coeffs(4, BUTTERWORTH_CTF, SAMPLERATE*2)
    (r2_ch, p2_ch, _) = cheby_coeffs(2, 0.6* SAMPLERATE, 60, SAMPLERATE)
    (r2_ch_x2, p2_ch_x2, _) = cheby_coeffs(2, 0.6* SAMPLERATE, 60, SAMPLERATE * 2)
    (r4_ch, p4_ch, _) = cheby_coeffs(4, 0.6* SAMPLERATE, 60, SAMPLERATE)
    (r4_ch_x2, p4_ch_x2, _) = cheby_coeffs(4, 0.6* SAMPLERATE, 60, SAMPLERATE * 2)
    (r10_ch, p10_ch, _) = cheby_coeffs(10, 0.6* SAMPLERATE, 60, SAMPLERATE)

    # ======== 2nd order processing ========
    for order in range(0, 2, 2):
        # Butterworth
        ri_bt = r2_bt[order]
        zi_bt = p2_bt[order]
        y2_bt += process_fwd(x, ri_bt, zi_bt, NAIVE_SAW_X, m, q, m_diff, q_diff)
        # y2_bt_og += process(x, ri_bt, zi_bt, NAIVE_SAW_X, m, q, m_diff, q_diff)

        # Chebyshev
        ri_ch = r2_ch[order]
        zi_ch = p2_ch[order]
        y2_ch += process_fwd(x, ri_ch, zi_ch, NAIVE_SAW_X, m, q, m_diff, q_diff)

        # Butterworth x2 oversampling
        ri_bt_x2 = r2_bt_x2[order]
        zi_bt_x2 = p2_bt_x2[order]
        y2_bt_x2 += process_fwd(x_x2, ri_bt_x2, zi_bt_x2, NAIVE_SAW_X, m, q, m_diff, q_diff)

        # Chebyshev x2 oversampling
        ri_ch_x2 = r2_ch_x2[order]
        zi_ch_x2 = p2_ch_x2[order]
        y2_ch_x2 += process_fwd(x_x2, ri_ch_x2, zi_ch_x2, NAIVE_SAW_X, m, q, m_diff, q_diff)
    

    # ======== 4th order processing ========
    for order in range(0, 4, 2):
        #Butterworth
        ri_bt = r4_bt[order]
        zi_bt = p4_bt[order]
        y4_bt += process_fwd(x, ri_bt, zi_bt, NAIVE_SAW_X, m, q, m_diff, q_diff)
        # y4_bt_og += process(x, ri_bt, zi_bt, NAIVE_SAW_X, m, q, m_diff, q_diff)

        # Chebyshev
        ri_ch = r4_ch[order]
        zi_ch = p4_ch[order]
        y4_ch += process_fwd(x, ri_ch, zi_ch, NAIVE_SAW_X, m, q, m_diff, q_diff)

        # Butterworth 2x oversampling
        ri_bt_x2 = r4_bt_x2[order]
        zi_bt_x2 = p4_bt_x2[order]
        y4_bt_x2 += process_fwd(x_x2, ri_bt_x2, zi_bt_x2, NAIVE_SAW_X, m, q, m_diff, q_diff)

        # Chebyshev 2x oversampling
        ri_ch_x2 = r4_ch_x2[order]
        zi_ch_x2 = p4_ch_x2[order]
        y4_ch_x2 += process_fwd(x_x2, ri_ch_x2, zi_ch_x2, NAIVE_SAW_X, m, q, m_diff, q_diff)


    # ======== 10th order processing ========
    for order in range(0, 10, 2):
        # Chebyshev
        ri_ch = r10_ch[order]
        zi_ch = p10_ch[order]
        y10_ch +=  process_fwd(x, ri_ch, zi_ch, NAIVE_SAW_X, m, q, m_diff, q_diff)
        # y10_ch_og += process(x, ri_ch, zi_ch, NAIVE_SAW_X, m, q, m_diff, q_diff)


    # ======== Linear interpolation processing ========
    y_naive = process_naive_linear(NAIVE_SAW, x)
    y_naive_x2 = process_naive_linear(NAIVE_SAW, x_x2)
    y_naive_x4 = process_naive_linear(NAIVE_SAW, x_x4)
    y_naive_x8 = process_naive_linear(NAIVE_SAW, x_x8)
    y_bl = bl_sawtooth(np.linspace(0, DURATION_S, num = num_frames_total, endpoint = False), play_freq)


    # ======== Downsampling the oversampled outputs ========
    # Downsample the upsampled output
    y2_bt_ds = soxr.resample(
        y2_bt_x2,
        88200,
        44100,
        quality="HQ"
    )
    y4_bt_ds = soxr.resample(
        y4_bt_x2,
        88200,
        44100,
        quality="HQ"
    )
    y2_ch_ds = soxr.resample(
        y2_ch_x2,
        88200,
        44100,
        quality="HQ"
    )
    y4_ch_ds = soxr.resample(
        y4_ch_x2,
        88200,
        44100,
        quality="HQ"
    )
    y_naive_x2_ds = soxr.resample(
        y_naive_x2,
        44100 * 2,
        44100,
        quality="HQ"
    )
    y_naive_x2_ds2 = downsample_x2_decim9(y_naive_x2*0.75)      # comparing SOXR to basic decimator
    y_naive_x4_ds = soxr.resample(
        y_naive_x4,
        44100 * 4,
        44100,
        quality="HQ"
    )
    y_naive_x4_ds2 = downsample_x4_decim9_17(y_naive_x4)
    y_naive_x8_ds = soxr.resample(
        y_naive_x8,
        44100 * 8,
        44100,
        quality="HQ"
    )


    # ======== Computing the FFTs for SNR computation ========
    y_bl_fft = normalized_fft(y_bl)
    y_naive_fft = normalized_fft(y_naive)
    y2_bt_fft = normalized_fft(y2_bt)
    y4_bt_fft = normalized_fft(y4_bt)
    y2_ch_fft = normalized_fft(y2_ch)
    y4_ch_fft = normalized_fft(y4_ch)
    y10_ch_fft = normalized_fft(y10_ch)
    y2_ch_x2_fft = normalized_fft(y2_ch_ds)
    y4_ch_x2_fft = normalized_fft(y4_ch_ds)
    y4_bt_fft = normalized_fft(y4_bt)
    y2_bt_ds_fft = normalized_fft(y2_bt_ds)
    y4_bt_ds_fft = normalized_fft(y4_bt_ds)
    y_naive_x2_fft = normalized_fft(y_naive_x2_ds)
    y_naive_x4_fft = normalized_fft(y_naive_x4_ds)
    y_naive_x8_fft = normalized_fft(y_naive_x8_ds)
    y_naive_x2_fft2 = normalized_fft(y_naive_x2_ds2)
    y_naive_x4_fft2 = normalized_fft(y_naive_x4_ds2)
    # y4_bt_og_fft = normalized_fft(y4_bt_og)
    # y2_bt_og_fft = normalized_fft(y2_bt_og)
    # y10_ch_fft_og = normalized_fft(y10_ch_og)
    # y_naive_x4_fft3 = normalized_fft(y_naive_x4_cpp_decimated)
    # y_naive_x2_cpp_leg_fft = normalized_fft(y_naive_x2_cpp_decimated_leg)
    # y_naive_x2_cpp_simd_fft = normalized_fft(y_naive_x2_cpp_decimated_simd)


    # ======== Computing SNR - should be done over different play_freq, not just one ========
    """
    I"m not 100% sure about the way I compute SNR, don't hesitate to take a look
    """
    print("Play frequency : {} Hz".format(play_freq))
    print("SNR band-limited : ", snr(y_bl_fft, y_bl_fft))
    print("SNR naive : ", snr(y_naive_fft, y_bl_fft))

    print("SNR ADAA BT order 2 : ", snr(y2_bt_fft, y_bl_fft))
    print("SNR ADAA BT order 4 : ", snr(y4_bt_fft, y_bl_fft))
    print("SNR ADAA BT order 2 OVSx2 : ", snr(y2_bt_ds_fft, y_bl_fft))
    print("SNR ADAA BT order 4 OVSx2 : ", snr(y4_bt_ds_fft, y_bl_fft))

    print("SNR ADAA CH order 2 : ", snr(y2_ch_fft, y_bl_fft))
    print("SNR ADAA CH order 4 : ", snr(y4_ch_fft, y_bl_fft))
    print("SNR ADAA CH order 10 : ", snr(y10_ch_fft, y_bl_fft))
    print("SNR ADAA CH order 2 OVSx2 : ", snr(y2_ch_x2_fft, y_bl_fft))
    print("SNR ADAA CH order 4 OVSx2 : ", snr(y4_ch_x2_fft, y_bl_fft))

    print("SNR naive OVSx2 : ", snr(y_naive_x2_fft, y_bl_fft))
    print("SNR naive OVSx4 : ", snr(y_naive_x4_fft, y_bl_fft))
    print("SNR naive OVSx8 : ", snr(y_naive_x8_fft, y_bl_fft))
    print("SNR naive OVSx2 decimator9 : ", snr(y_naive_x2_fft2, y_bl_fft))
    print("SNR naive OVSx4 decimator9_17: ", snr(y_naive_x4_fft2, y_bl_fft))
    # print("SNR naive OVSx4 NE10: ", snr(y_naive_x4_fft3, y_bl_fft))
    # print("SNR naive OVSx2 Legacy: ", snr(y_naive_x2_cpp_leg_fft, y_bl_fft))
    # print("SNR naive OVSx4 SIMD: ", snr(y_naive_x2_cpp_simd_fft, y_bl_fft))


    fig, axs = plt.subplots(5)
    sample_offset = 0

    # ======== Displaying PSD ========
    """
    Because I test too many different processing, I cannot print them all at once, it's unreadable.
    That's why many are commented out
    """
    axs[0].psd(y_naive[sample_offset:], Fs=SAMPLERATE, NFFT=4096, color="red", label="naive-linear")
    # axs[0].psd(y2_bt[sample_offset:], Fs=SAMPLERATE, NFFT=4096, color="blue", label="ADAA-butterworth-2")
    axs[1].psd(y4_bt[sample_offset:], Fs=SAMPLERATE, NFFT=4096, color="blue", label="ADAA-butterworth-4")
    # axs[2].psd(y2_ch[sample_offset:], Fs=SAMPLERATE, NFFT=4096, color="green", label="ADAA-chebyshev-2")
    # axs[0].psd(y_naive_x4_ds2, Fs=SAMPLERATE, NFFT=4096, color="black", label="Decimator 4")
    # axs[2].psd(y_naive_x2_ds2, Fs=SAMPLERATE, NFFT=4096, color="black", label="Decimator 2")
    # axs[3].psd(y4_ch[sample_offset:], Fs=SAMPLERATE, NFFT=4096, color="black", label="ADAA-chebyshev-4")
    # axs[4].psd(y2_ch_ds[sample_offset:], Fs=SAMPLERATE, NFFT=4096, color="purple", label="ADAA-chebyshev-2 x2")
    # axs[5].psd(y4_ch_ds[sample_offset:], Fs=SAMPLERATE, NFFT=4096, color="orange", label="ADAA-chebyshev-4 x2")
    # axs[2].psd(y2_ds[sample_offset:], Fs=SAMPLERATE, NFFT=4096, color="green", label="ADAA-butterworth-2  x2")
    # axs[3].psd(y4[sample_offset:], Fs=SAMPLERATE, NFFT=4096, color="black", label="ADAA-butterworth-2")
    axs[2].psd(y_naive_x4_ds[sample_offset:], Fs=SAMPLERATE, NFFT=4096, color="green", label="naive x4")
    axs[3].psd(y10_ch[sample_offset:], Fs=SAMPLERATE, NFFT=4096, color="black", label="ADAA-cheby-10")
    # axs[5].psd(y10_ch_og, Fs=SAMPLERATE, NFFT=4096, color="red", label="cheby 10 [OG]")
    # axs[5].psd(y_naive_x8_ds[sample_offset:], Fs=SAMPLERATE, NFFT=4096, color="purple", label="naive x8")
    # axs[6].psd(y2_ds[sample_offset:], Fs=SAMPLERATE, NFFT=4096, color="pink", label="ADAA-butterworth-2-OVS")
    # axs[7].psd(y4_ds[sample_offset:], Fs=SAMPLERATE, NFFT=4096, color="orange", label="ADAA-butterworth-4-OVS")
    # axs[5].psd(y_bl[sample_offset:], Fs=SAMPLERATE, NFFT=4096, color="purple", label="band limited")



    # ======== Displaying waveform, usefull to debug algorithm and identify latency ========
    axs[4].plot(x[sample_offset:], y_naive[sample_offset:], 'red', label="naive-linear")
    # axs[4].plot(x[sample_offset:], y2_bt[sample_offset:], 'b', label="ADAA-butterworth-2")
    # axs[4].plot(x, y_naive_x2_ds2, "r", label="decimator x2")
    # axs[4].plot(x, y_naive_x4_ds2, "green", label="decimator x4")
    axs[4].plot(x, y_naive_x4_ds, "green", label="naive x4")
    # axs[4].plot(x, y_naive_x4_cpp_decimated, "purple", label="NE10")
    axs[4].plot(x[sample_offset:], y4_bt[sample_offset:], 'blue', label="ADAA-butterworth-4")
    axs[4].plot(x[sample_offset:], y10_ch[sample_offset:], 'black', label="ADAA-butterworth-4")
    # axs[4].plot(x[sample_offset:], y4_bt_ds[sample_offset:], 'purple', label="ADAA-butterworth-4-OVS")
    # axs[4].plot(x[sample_offset:], y_naive[sample_offset:], 'green', label="naive-linear")
    # axs[4].plot(x[sample_offset:], y_bl[sample_offset:], 'r', label="band_limited")
    # axs[4].plot(x[sample_offset:], y2_bt_og[sample_offset:], 'purple', label="[OG] ADAA-butterworth-2")

    for ax in axs:
        ax.grid(True, which="both")
        ax.legend()

    plt.show()


def mod_bar(x, k):
    m = x % k
    return m + k * (1 - np.sign(m))

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

def process(x, B, beta: complex, X, m, q, m_diff, q_diff):
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
    main()