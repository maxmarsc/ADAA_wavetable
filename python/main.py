
import numpy as np
from math import exp, floor
from cmath import exp as cexp
from scipy.signal import welch
# from pathlib import Path
# import sys
import matplotlib.pyplot as plt
import matplotlib

WAVEFORM_LEN = 2048
SAMPLERATE = 44100
BUTTERWORTH_CTF = 0.45 * SAMPLERATE
DURATION_S = 0.1


BUTTER2_COEFFS = np.array([
    [0.0000 - 2.2164j, 0.0000 + 2.2164j],                   # residuals
    [-2.2164 + 2.2164j, -2.2164 - 2.2164j,]                 # poles
])

# NAIVE_SAW = np.zeros(WAVEFORM_LEN)
# NAIVE_SAW[:WAVEFORM_LEN//2] = np.linspace(0, 1, WAVEFORM_LEN//2, endpoint=True)
# NAIVE_SAW[WAVEFORM_LEN//2:] = np.linspace(-1, 0, WAVEFORM_LEN//2, endpoint=False)

matplotlib.use('TkAgg')

def compute_naive_saw(frames: int) -> np.ndarray:
    phase = 0.5
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
    m = np.zeros(size)
    q = np.zeros(size)

    m[-1] = compute_m(X[-2], X[-1], waveform[-1], waveform[0])
    q[-1] = compute_m(X[-2], X[-1], waveform[-1], waveform[0])

    for i in range(size - 1):
        m[i] = compute_m(X[i], X[i+1], waveform[i], waveform[i+1])
        q[i] = compute_q(X[i], X[i+1], waveform[i], waveform[i+1])
    
    return (m, q)


def binary_search_down(x : np.ndarray, x0: float, j_min: int, j_max: int) -> int:
    """
    return i as x_i < x_0 < x_(i+1) && j_min <= i <= j_max
    """
    if x0 <= x[0]:
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

        if relative_idx == prev_idx:
            y[i] = waveform[prev_idx]
        else:
            a = (waveform[prev_idx + 1] - waveform[prev_idx]) / (x - prev_x)
            b = (waveform[prev_idx +1] * (x - prev_x) - prev_x * (waveform[prev_idx + 1] - waveform[prev_idx])) / (x - prev_x)
            y[i] = a * x + b

        prev_x = x
    
    return y




def main():
    orders = len(BUTTER2_COEFFS[0])
    play_freq = 1000

    num_frames_total = int(DURATION_S * SAMPLERATE)
    x = np.linspace(0.0, num_frames_total*play_freq / SAMPLERATE, num_frames_total, endpoint=False)
    y = np.zeros(num_frames_total)
    (m, q) =  compute_m_q_vectors(NAIVE_SAW, NAIVE_SAW_X)

    
    # Precomputing diffs
    m_diff = np.zeros(m.shape[0])
    q_diff = np.zeros(q.shape[0])
    
    for i in range(m.shape[0] - 1):
        m_diff[i] = m[i+1] - m[i]
        q_diff[i] = q[i+1] - q[i]
    m_diff[-1] = m[0] - m[-1]
    q_diff[-1] = q[0] - q[-1] - m[0] * NAIVE_SAW_X[-1]

    # print("m : [{},{},{},...,{},{},{}]".format(m[0], m[1], m[2], m[-3], m[-2], m[-1]))
    # print("q : [{},{},{},...,{},{},{}]".format(q[0], q[1], q[2], q[-3], q[-2], q[-1]))
    # print("m_diff : [{},{},{},...,{},{},{}]".format(m_diff[0], m_diff[1], m_diff[2], m_diff[-3], m_diff[-2], m_diff[-1]))
    # print("q_diff : [{},{},{},...,{},{},{}]".format(q_diff[0], q_diff[1], q_diff[2], q_diff[-3], q_diff[-2], q_diff[-1]))
    abc = None


    for order in range(0, orders, 2):
        ri = BUTTER2_COEFFS[0][order]
        zi = BUTTER2_COEFFS[1][order]
        (new_y, new_abc) = process(x, ri, zi, NAIVE_SAW_X, m, q, m_diff, q_diff)
        abc = new_abc
        y +=new_y
        # y += process(x, ri, zi, NAIVE_SAW_X, m, q, m_diff, q_diff)

    y_naive = process_naive_linear(NAIVE_SAW, x)


    # freqs, powers = welch(y, fs=SAMPLERATE)
    # freqs_naive, power_naive = welch(y_naive, fs=SAMPLERATE)
    fig, axs = plt.subplots(3)
    # axs[0].loglog(freqs, powers, 'b')
    # axs[0].hist(np.log2(powers), log=True, bins=freqs)
    axs[0].psd(y, Fs=SAMPLERATE, NFFT=2048, color="b", label="ADAA-IIR-1")
    axs[1].psd(y_naive, Fs=SAMPLERATE, NFFT=2048, color="r", label="naive-linear")

    # axs[0].loglog(freqs_naive, power_naive, 'r')
    # axs[1].hist(np.log2(power_naive), log=True, bins=freqs_naive)
    test = np.zeros(num_frames_total)
    for i in (1, 45, 89, 133, 177):
        test[i] = -1.

    axs[2].plot(x, y, 'b', label="ADAA-IIR-1")
    axs[2].plot(x, y_naive, 'r', label="naive-linear")
    # axs[2].plot(x, abc, 'g')
    # axs[1].plot(x, test, 'green')
    plt.grid(True, which="both")
    plt.legend()
    plt.show()


def mod_bar(x, k):
    m = x % k
    return m + k * (1 - np.sign(m))


def process(x, B, beta: complex, X, m, q, m_diff, q_diff):
    y = np.zeros(x.shape[0])
    abc = np.ndarray(x.shape[0])


    pos_dif_counter=0
    neg_dif_counter=0
    # print("m_diff[0] : ", m_diff[0])
    # print("q_diff[0] : ", q_diff[0])
    # print("m")

    # Period - should be 1
    T = X[-1]

    waveform_frames = m.shape[0]     # aka k

    expbeta = cexp(beta)

    # Initial condition
    prev_x = 0
    prev_cpx_y: complex = 0
    prev_x_diff = 0

    # Setting j indexs and some reduced values
    x_red = prev_x % T
    j_red = binary_search_down(X, x_red, 0, X.shape[0] - 1)
    j = waveform_frames * floor(prev_x / T) + j_red - 1

    for n in range(1, x.shape[0]):
        # loop init
        x_diff = x[n] - prev_x
        prev_j = j

        if (n-1)%44 == 0:
            a = 0

        # TODO: No idea ?
        if (x_diff >= 0 and prev_x_diff >=0) or (x_diff < 0 and prev_x_diff <= 0):
            # If on the same slop as the previous iteration
            prev_j = j + int(np.sign(x_red - X[j_red]))
        
        x_red = x[n] % T
        if x_red == 0.0:
            a = 0

        # Should be differentiated upstream to avoid if on each sample
        if x_diff >= 0:
            # playback going forward
            j_red = binary_search_down(X, x_red, 0, X.shape[0] - 1)
            j = waveform_frames * floor(x[n] / T) + j_red - 1
            j_min = prev_j
            j_max = j
        else:
            # playback going backward
            j_red = binary_search_up(X, x_red, 0, X.shape[0] - 1)
            j = waveform_frames * floor(x[n] / T) + j_red - 1
            j_min = j
            j_max = prev_j

        # TODO: Shouldn't it be normal modulo to deal with matlab 1 offset ?
        # j_min_red = mod_bar(j_min, waveform_frames)
        # j_max_p_red = mod_bar(j_max +1 , waveform_frames)
        j_min_red = j_min % waveform_frames
        j_max_p_red = (j_max + 1) % waveform_frames


        # Should be differentiated upstream to avoid if on each sample
        if x_diff >= 0:
            pos_dif_counter+=1
            I = expbeta\
                    * (m[j_min_red] * x_diff + beta * (m[j_min_red] * (prev_x - T * floor((j_min - 1) / waveform_frames)) + q[j_min_red]))\
                    - m[j_max_p_red] * x_diff\
                    - beta * (m[j_max_p_red] * (x[n] - T * floor(j_max/waveform_frames)) + q[j_max_p_red])
        else:
            neg_dif_counter+=1
            I = expbeta\
                    * (m[j_max_p_red] * x_diff + beta * (m[j_max_p_red] * (prev_x - T * floor(j_max/waveform_frames)) + q[j_max_p_red]))\
                    - m[j_min_red] * x_diff\
                    - beta * (m[j_min_red] * (x[n] - T * floor((j_min - 1)/waveform_frames)) + q[j_min_red])

        I_sum = 0
        # Here j_min and j_max are not used for anything except their offset
        # TODO: check if it makes sense
        for i in range(j_min, j_max + 1):
        # for i in range(j_min_red, j_max_p_red):
            i_red = i % waveform_frames
            if j_max_p_red <= j_min_red:
                break

            I_sum += cexp(beta * (x[n] - X[i_red + 1] - T * floor((i - 1)/waveform_frames))/x_diff)\
                        * (beta * q_diff[i_red] + m_diff[i_red] * (x_diff + beta * X[i_red + 1]))
    

        I = (I + np.sign(x_diff) * I_sum) / (beta**2)

        # See formula (10)
        y_cpx: complex = expbeta * prev_cpx_y + 2 * B * I
        y[n] = y_cpx.real
        abc[n] = I_sum


        prev_x = x[n]
        prev_cpx_y = y_cpx
        prev_x_diff = x_diff


    print("POS_DIF_COUNT : ", pos_dif_counter)
    print("NEG_DIF_COUNT : ", neg_dif_counter)
    return (y, abc)




if __name__ == "__main__":
    main()