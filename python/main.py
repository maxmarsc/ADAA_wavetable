
import numpy as np
from math import exp, floor
from cmath import exp as cexp
from scipy.signal import welch
# from pathlib import Path
# import sys
import matplotlib.pyplot as plt
import matplotlib
import soxr

from bl_waveform import bl_sawtooth

WAVEFORM_LEN = 1024
SAMPLERATE = 44100
BUTTERWORTH_CTF = 0.45 * SAMPLERATE
DURATION_S = 0.1


BUTTER2_COEFFS = np.array([
    [0.0000 - 1.999274j, 0.0000 + 1.999274j],                   # residuals
    [-1.999274 + 1.999274j, -1.999274 - 1.999274j,]                 # poles
])

BUTTER4_COEFFS = np.array([
    [1.306104 - 3.153214j, 1.306104 + 3.153214j, -1.306104 + 0.541006j, -1.306104 - 0.541006j],  # residuals
    [-2.612208 + 1.082012j, -2.612208 - 1.082012j, -1.082012 + 2.612208j, -1.082012 - 2.612208j] # poles
])

BUTTERWORTH_COEFFS = {
    44100 : {
        2 : [
            [0.0000 - 1.999274j, 0.0000 + 1.999274j],                   # residuals
            [-1.999274 + 1.999274j, -1.999274 - 1.999274j]             # poles
        ],
        4 : [
            [1.306104 - 3.153214j, 1.306104 + 3.153214j, -1.306104 + 0.541006j, -1.306104 - 0.541006j],  # residuals
            [-2.612208 + 1.082012j, -2.612208 - 1.082012j, -1.082012 + 2.612208j, -1.082012 - 2.612208j] # poles
        ]
    },
    88200 : {
        2 : [
            [0.000000 -0.999649j, 0.000000 +0.999649j],         #r
            [-0.999649 +0.999649j, 0.999649 -0.999649j]         #p
        ],
        4 : [
            [0.653052 -1.576607j, 0.653052 +1.576607j, -0.653052 +0.270503j, -0.653052 -0.270503j],      #r
            [-1.306104 +0.541006j, -1.306104 -0.541006j, -0.541006 +1.306104j, -0.541006 -1.306104j]         #p
        ]
    }
}

CHEBY2_COEFFS = {
    "44100" : {
        2 : [
            [-0.000121 -0.121141j, -0.000121 +0.121141j],
            [-0.121141 +0.121263j, +0.121141 -0.121263j]
        ],
        4 : [
            [0.493702 -1.282896j, 0.493702 +1.282896j,-0.495186 +0.201183j, -0.495186 -0.201183j],
            [-1.068581 +0.462868j, -1.068581 -0.462868j, -0.415498 +1.048988j, -0.415498 -1.048988j]
        ]
    },
    "88200" : {
        2 : [
            [-0.000061 -0.060571j, -0.000061 +0.060571j],
            [-0.060571 +0.060631j, 0.060571 -0.060631j]
        ],
        4 : [
            [0.246851 -0.641448j, 0.246851 +0.641448j, -0.247593 +0.100591j, -0.247593 -0.100591j],
            [-0.534291 +0.231434j, -0.534291 -0.231434j, -0.207749 +0.524494j, -0.207749 -0.524494j]
        ]
    }
}


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


def compute_m(x0, x1, y0, y1):
    return (y1 - y0) / (x1 - x0)

def compute_q(x0, x1, y0, y1):
    return (y0 * (x1 - x0) - x0 * (y1 - y0)) / (x1 - x0)

def compute_m_q_vectors(waveform, X):
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

def main():
    play_freq = 6000

    num_frames_total = int(DURATION_S * SAMPLERATE)
    num_frames_total_x2 = int(DURATION_S * SAMPLERATE * 2)
    num_frames_total_x4 = int(DURATION_S * SAMPLERATE * 4)
    num_frames_total_x8 = int(DURATION_S * SAMPLERATE * 8)

    x = np.linspace(0.0, num_frames_total*play_freq / SAMPLERATE, num_frames_total, endpoint=True)
    x_x2 = np.linspace(0.0, num_frames_total*play_freq / SAMPLERATE, num_frames_total_x2, endpoint=True)
    x_x4 = np.linspace(0.0, num_frames_total*play_freq / SAMPLERATE, num_frames_total_x4, endpoint=True)
    x_x8 = np.linspace(0.0, num_frames_total*play_freq / SAMPLERATE, num_frames_total_x8, endpoint=True)
    y2 = np.zeros(num_frames_total)
    y2_og = np.zeros(num_frames_total)
    y4 = np.zeros(num_frames_total)
    y4_og = np.zeros(num_frames_total)
    y2_up = np.zeros(num_frames_total_x2)
    y4_up = np.zeros(num_frames_total_x2)
    y_naive_x4 = np.zeros(num_frames_total_x4)
    y_naive_x8 = np.zeros(num_frames_total_x8)

    amplitude_offset = 0

    # Precomputing m & q diffs
    (m, q) =  compute_m_q_vectors(NAIVE_SAW - amplitude_offset, NAIVE_SAW_X)
    m_diff = np.zeros(m.shape[0])
    q_diff = np.zeros(q.shape[0])
    
    for i in range(m.shape[0] - 1):
        m_diff[i] = m[i+1] - m[i]
        q_diff[i] = q[i+1] - q[i]
    m_diff[-1] = m[0] - m[-1]
    q_diff[-1] = q[0] - q[-1] - m[0] * NAIVE_SAW_X[-1]
    # q_diff[-1] = q[0] - q[-1]

    abc = None

    for order in range(0, 2, 2):
        ri = BUTTERWORTH_COEFFS[44100][2][0][order]
        zi = BUTTERWORTH_COEFFS[44100][2][1][order]
        y2 += process_fwd(x, ri, zi, NAIVE_SAW_X, m, q, m_diff, q_diff)
        y2_og += process(x, ri, zi, NAIVE_SAW_X, m, q, m_diff, q_diff)

        ri_up = BUTTERWORTH_COEFFS[88200][2][0][order]
        zi_up = BUTTERWORTH_COEFFS[88200][2][1][order]
        y2_up += process_fwd(x_x2, ri_up, zi_up, NAIVE_SAW_X, m, q, m_diff, q_diff)
    
    for order in range(0, 4, 2):
        ri = BUTTERWORTH_COEFFS[44100][4][0][order]
        zi = BUTTERWORTH_COEFFS[44100][4][1][order]
        y4 += process_fwd(x, ri, zi, NAIVE_SAW_X, m, q, m_diff, q_diff)
        y4_og += process(x, ri, zi, NAIVE_SAW_X, m, q, m_diff, q_diff)

        ri_up = BUTTERWORTH_COEFFS[88200][4][0][order]
        zi_up = BUTTERWORTH_COEFFS[88200][4][1][order]
        y4_up += process_fwd(x_x2, ri_up, zi_up, NAIVE_SAW_X, m, q, m_diff, q_diff)

    y2 += amplitude_offset
    # y2_og += amplitude_offset
    y4 += amplitude_offset
    y4_up +=  amplitude_offset
    y2_up += amplitude_offset

    y_naive = process_naive_linear(NAIVE_SAW, x)
    y_naive_x2 = process_naive_linear(NAIVE_SAW, x_x2)
    y_naive_x4 = process_naive_linear(NAIVE_SAW, x_x4)
    y_naive_x8 = process_naive_linear(NAIVE_SAW, x_x8)
    y_bl = bl_sawtooth(np.linspace(0, DURATION_S, num = num_frames_total, endpoint = False), play_freq)


    # to_resample = np.zeros(y2_up.shape[0] + 128)
    # to_resample[64:-64] = y2_up

    # Downsample the upsampled output
    y2_ds = soxr.resample(
        y2_up,
        88200,
        44100,
        quality="HQ"
    )
    y4_ds = soxr.resample(
        y4_up,
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
    y_naive_x4_ds = soxr.resample(
        y_naive_x4,
        44100 * 4,
        44100,
        quality="HQ"
    )
    y_naive_x8_ds = soxr.resample(
        y_naive_x8,
        44100 * 8,
        44100,
        quality="HQ"
    )

    FFT_SIZE = 4096

    y_bl_fft = normalized_fft(y_bl)
    y_naive_fft = normalized_fft(y_naive)
    y2_fft = normalized_fft(y2)
    y4_fft = normalized_fft(y4)
    y2_og_fft = normalized_fft(y2_og)
    y4_fft = normalized_fft(y4)
    y4_og_fft = normalized_fft(y4_og)
    y2_ds_fft = normalized_fft(y2_ds)
    y4_ds_fft = normalized_fft(y4_ds)
    y_naive_x2_fft = normalized_fft(y_naive_x2_ds)
    y_naive_x4_fft = normalized_fft(y_naive_x4_ds)
    y_naive_x8_fft = normalized_fft(y_naive_x8_ds)

    print("Play frequency : {} Hz".format(play_freq))
    print("SNR band-limited : ", snr(y_bl_fft, y_bl_fft))
    print("SNR naive : ", snr(y_naive_fft, y_bl_fft))
    print("SNR ADAA order 2 : ", snr(y2_fft, y_bl_fft))
    print("SNR ADAA order 4 : ", snr(y4_fft, y_bl_fft))
    print("SNR [OG] ADAA order 2 : ", snr(y2_og_fft, y_bl_fft))
    print("SNR [OG] ADAA order 4 : ", snr(y4_og_fft, y_bl_fft))
    print("SNR ADAA order 2 OVSx2 : ", snr(y2_ds_fft, y_bl_fft))
    print("SNR ADAA order 4 OVSx2 : ", snr(y4_ds_fft, y_bl_fft))
    print("SNR naive OVSx2 : ", snr(y_naive_x2_fft, y_bl_fft))
    print("SNR naive OVSx4 : ", snr(y_naive_x4_fft, y_bl_fft))
    print("SNR naive OVSx8 : ", snr(y_naive_x8_fft, y_bl_fft))


    # print("SNR naive : ", naive_snr
    # , bt2_snr, bt4_snr, bt2_ovs_snr, bt4_ovs_snr)

    # def signalPower(x):
    #     return np.sqrt(np.mean(x**2))
    
    # def SNR(clean, dirty):
    #     noise = dirty - clean

    #     powS = signalPower(clean)
    #     powN = signalPower(noise)
    #     return 10*np.log10(powS/powN)

    # print("SNR bl : {}".format(SNR(y_bl, y_bl)))
    # print("SNR naive : {}".format(SNR(y_bl, y_naive)))
    # print("SNR bt2 : {}".format(SNR(y_bl, y2)))
    # print("SNR bt4 : {}".format(SNR(y_bl, y4)))
    # print("SNR bt2 OVSx2: {}".format(SNR(y_bl, y2_ds)))
    # print("SNR bt4 OVSx2 : {}".format(SNR(y_bl, y4_ds)))
    # print("SNR naive OVSx4 : {}".format(SNR(y_bl, y_naive_x4_ds)))
    # print("SNR naive OVSx8 : {}".format(SNR(y_bl, y_naive_x8_ds)))

    fig, axs = plt.subplots(8)
    sample_offset = 0

    # axs[0].plot(10* np.log10(np.abs(y_naive_fft - y_bl_fft) +  np.finfo(float).eps), color="r", label="naive-linear")
    # axs[0].plot(10* np.log10(np.abs(y2_fft - y_bl_fft) +  np.finfo(float).eps), color="b", label="ADAA-butterworth-2")
    # axs[0].plot(10* np.log10(np.abs(y4_ds_fft - y_bl_fft) +  np.finfo(float).eps), color="purple", label="ADAA-butterworth-4-OVS")

    # axs[4].psd(y_naive - y_bl, Fs=SAMPLERATE, NFFT=4096, color="red", label="ADAA-butterworth-2")
    # axs[0].psd(y2 - y_bl, Fs=SAMPLERATE, NFFT=4096, color="b", label="ADAA-butterworth-2")
    # axs[1].psd(y4 - y_bl, Fs=SAMPLERATE, NFFT=4096, color="black", label="ADAA-butterworth-4")
    # axs[2].psd(y2_ds - y_bl, Fs=SAMPLERATE, NFFT=4096, color="green", label="ADAA-butterworth-2-OVS")
    # axs[3].psd(y4_ds - y_bl, Fs=SAMPLERATE, NFFT=4096, color="purple", label="ADAA-butterworth-4-OVS")
    # axs[4].psd(y_bl[3:], Fs=SAMPLERATE, NFFT=4096, color="purple", label="band limited")

    axs[0].psd(y_naive[sample_offset:], Fs=SAMPLERATE, NFFT=4096, color="red", label="naive-linear")
    axs[1].psd(y2[sample_offset:], Fs=SAMPLERATE, NFFT=4096, color="b", label="ADAA-butterworth-2")
    axs[2].psd(y2_ds[sample_offset:], Fs=SAMPLERATE, NFFT=4096, color="green", label="ADAA-butterworth-2  x2")
    axs[3].psd(y4[sample_offset:], Fs=SAMPLERATE, NFFT=4096, color="black", label="ADAA-butterworth-2")
    axs[4].psd(y_naive_x4_ds[sample_offset:], Fs=SAMPLERATE, NFFT=4096, color="green", label="naive x4")
    axs[5].psd(y_naive_x8_ds[sample_offset:], Fs=SAMPLERATE, NFFT=4096, color="purple", label="naive x8")
    axs[6].psd(y2_ds[sample_offset:], Fs=SAMPLERATE, NFFT=4096, color="pink", label="ADAA-butterworth-2-OVS")
    axs[7].psd(y4_ds[sample_offset:], Fs=SAMPLERATE, NFFT=4096, color="orange", label="ADAA-butterworth-4-OVS")
    # axs[5].psd(y_bl[sample_offset:], Fs=SAMPLERATE, NFFT=4096, color="purple", label="band limited")

    # for ax_idx in (0, 1, 2, 3, 4):
    #     axs[ax_idx].minorticks_on()
    #     axs[ax_idx].grid(which="minor", linestyle=":")

    # axs[5].plot(x, y2_og - y2)
    # axs[6].plot(x[sample_offset:], y_naive[sample_offset:] - y_bl, 'red', label="naive-linear")
    # # axs[6].plot(x[sample_offset:], y_bl[sample_offset:], 'r', label="band_limited")
    # axs[6].plot(x[sample_offset:], y2[sample_offset:] - y_bl, 'b', label="ADAA-butterworth-2")
    # axs[6].plot(x[sample_offset:], y4_ds[sample_offset:] - y_bl, 'purple', label="ADAA-butterworth-4-OVS")
    # axs[6].plot(x[sample_offset:], y_naive[sample_offset:], 'green', label="naive-linear")
    # axs[6].plot(x[sample_offset:], y_bl[sample_offset:], 'r', label="band_limited")
    # axs[6].plot(x[sample_offset:], y2[sample_offset:], 'b', label="ADAA-butterworth-2")
    # axs[5].plot(x[sample_offset:], y2_og[sample_offset:], 'purple', label="[OG] ADAA-butterworth-2")
    # axs[6].plot(x[sample_offset:], y4[sample_offset:], 'black', label="ADAA-butterworth-4")
    # axs[6].plot(x[sample_offset:], y2_ds[sample_offset:], 'green', label="ADAA-butterworth-2-OVS")
    # axs[6].plot(x[sample_offset:], y4_ds[sample_offset:], 'purple', label="ADAA-butterworth-4-OVS")


    plt.grid(True, which="both")
    plt.legend()
    plt.show()


def mod_bar(x, k):
    m = x % k
    return m + k * (1 - np.sign(m))

def process_fwd(x, B, beta: complex, X, m, q, m_diff, q_diff):
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

        # TODO: Shouldn't it be normal modulo to deal with matlab 1 offset ?
        j_min_red = j_min % waveform_frames
        j_max_p_red = (j_max + 1) % waveform_frames

        # if j_min_red != alt_j_min_red or j_max_p_red != alt_j_max_p_red:
        #     pouet = 0

        # prev_x_red_bar = prev_x % 1.0
        # prev_x_red_bar += (prev_x_red_bar == 0.0)

        # if (prev_x - T * floor(j_min/ waveform_frames)) != prev_x_red_bar:
        #     a = prev_x - T * floor(j_min/ waveform_frames) ## ~ x[n-1] % 1.0
        #     b = (X[j_min_red - 1], X[j_min_red], X[j_min_red+1])

        # if x_red != (x[n] - T * floor((j_max+1)/waveform_frames)):
        #     a = (x[n] - T * floor((j_max+1)/waveform_frames))
        #     b = 0

        # Should be differentiated upstream to avoid if on each sample
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
                    * (m[j_max_p_red] * x_diff + beta * (m[j_max_p_red] * (prev_x - 1.0 * floor((j_max+1)/waveform_frames)) + q[j_max_p_red]))\
                    - m[j_min_red] * x_diff\
                    - beta * (m[j_min_red] * (x[n] - 1.0 * floor(j_min/waveform_frames)) + q[j_min_red])

        I_sum = 0

        # if j_max + 1 - j_min != j_max_p_red - j_min_red:
        #     a = j_max + 1 - j_min
        #     b = j_max_p_red - j_min_red
        #     c = "afaf"

        for i in range(j_min, j_max + 1):         #OG Version
        # for i in range(j_min_red, j_max_p_red):
            i_red = i % waveform_frames
            x_red_bar = x[n] % 1.0
            x_red_bar = x_red_bar + (x_red_bar < X[i_red])

            # if x_red_bar != x[n] - T * floor((i)/waveform_frames):
            #     a = x[n] - T * floor((i)/waveform_frames)
            #     b = 0

            # if i_red != alt_i % waveform_frames:
            #     a = alt_i % waveform_frames
            #     b = 0

            # I_sum += cexp(beta * (x_red_bar - X[i_red + 1])/x_diff)\
            #             * (beta * q_diff[i_red] + m_diff[i_red] * (x_diff + beta * X[i_red + 1]))
            # OG Version
            I_sum += cexp(beta * (x[n] - X[i_red + 1] - T * floor((i)/waveform_frames))/x_diff)\
                        * (beta * q_diff[i_red] + m_diff[i_red] * (x_diff + beta * X[i_red + 1]))
    

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