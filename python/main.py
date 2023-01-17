
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
    [0.0000 - 1.999274j, 0.0000 + 1.999274j],                   # residuals
    [-1.999274 + 1.999274j, -1.999274 - 1.999274j,]                 # poles
])

BUTTER4_COEFFS = np.array([
    [-1.4479 + 0.5998j, -1.4479 - 0.5998j, 1.4479 - 3.4956j, 1.4479 + 3.4956j],  # residuals
    [-1.1995 + 2.8959j, -1.1995 - 2.8959j, -2.8959 + 1.1995j, -2.8959 - 1.1995j] # poles
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
    q[-1] = compute_q(X[-2], X[-1], waveform[-1], waveform[0])

    for i in range(size - 1):
        m[i] = compute_m(X[i], X[i+1], waveform[i], waveform[i+1])
        q[i] = compute_q(X[i], X[i+1], waveform[i], waveform[i+1])
    
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
    x = np.linspace(0.0, num_frames_total*play_freq / SAMPLERATE, num_frames_total + 1, endpoint=True)[1:]
    y = np.zeros(num_frames_total)
    y4 = np.zeros(num_frames_total)
    (m, q) =  compute_m_q_vectors(NAIVE_SAW, NAIVE_SAW_X)

    
    # print("x {} {} {} : {} {} {}".format(x[-3], x[-2], x[-1], x[0], x[1], x[2]))
    # print("X {} {} {} : {} {} {}".format(NAIVE_SAW_X[-3], NAIVE_SAW_X[-2], NAIVE_SAW_X[-1], NAIVE_SAW_X[0], NAIVE_SAW_X[1], NAIVE_SAW_X[2]))

    # Precomputing diffs
    m_diff = np.zeros(m.shape[0])
    q_diff = np.zeros(q.shape[0])
    
    for i in range(m.shape[0] - 1):
        m_diff[i] = m[i+1] - m[i]
        q_diff[i] = q[i+1] - q[i]
    m_diff[-1] = m[0] - m[-1]
    q_diff[-1] = q[0] - q[-1] - m[0] * NAIVE_SAW_X[-1]

    abc = None

    for order in range(0, orders, 2):
        ri = BUTTER2_COEFFS[0][order]
        zi = BUTTER2_COEFFS[1][order]
        (new_y, new_abc) = process(x, ri, zi, NAIVE_SAW_X, m, q, m_diff, q_diff)
        abc = new_abc
        y += new_y
        # y += process(x, ri, zi, NAIVE_SAW_X, m, q, m_diff, q_diff)
    
    # for order in range(0, 4, 2):
    #     ri = BUTTER4_COEFFS[0][order]
    #     zi = BUTTER4_COEFFS[1][order]
    #     (new_y, _) = process(x, ri, zi, NAIVE_SAW_X, m, q, m_diff, q_diff)

    #     y4 += new_y

    # y_naive = process_naive_linear(NAIVE_SAW, x)


    # freqs, powers = welch(y, fs=SAMPLERATE)
    # freqs_naive, power_naive = welch(y_naive, fs=SAMPLERATE)
    fig, axs = plt.subplots(3)
    # axs[0].loglog(freqs, powers, 'b')
    # axs[0].hist(np.log2(powers), log=True, bins=freqs)
    axs[1].psd(y, Fs=SAMPLERATE, NFFT=4096, color="b", label="ADAA-butterworth-2")
    # axs[0].psd(y4, Fs=SAMPLERATE, NFFT=2048, color="black", label="ADAA-butterworth-2")
    # axs[0].psd(y_naive, Fs=SAMPLERATE, NFFT=2048, color="r", label="naive-linear")

    # axs[0].loglog(freqs_naive, power_naive, 'r')
    # axs[1].hist(np.log2(power_naive), log=True, bins=freqs_naive)
    # test = np.zeros(num_frames_total)
    # test[440] = 1.0
    # for i in (1, 45, 89, 133, 177):
    #     test[i] = -1.

    # axs[2].plot(x, y_naive, 'r', label="naive-linear")
    axs[2].plot(x, y, 'b', label="ADAA-butterworth-2")
    # axs[2].plot(x, y4, 'black', label="ADAA-butterworth-4")
    # axs[2].plot(x, abc, 'g')
    # axs[2].plot(x, test, 'purple')
    plt.grid(True, which="both")
    plt.legend()
    plt.show()


def mod_bar(x, k):
    m = x % k
    return m + k * (1 - np.sign(m))


def process(x, B, beta: complex, X, m, q, m_diff, q_diff):
    y = np.zeros(x.shape[0])
    abc = np.ndarray(x.shape[0])

    # Period - should be 1
    T = X[-1]

    waveform_frames = m.shape[0]     # aka k

    expbeta = cexp(beta)

    # Initial condition
    prev_x = x[0]
    prev_cpx_y: complex = 0
    prev_x_diff = 0

    # Setting j indexs and some reduced values
    x_red = prev_x % T
    j_red = binary_search_down(X, x_red, 0, X.shape[0] - 1)
    j = waveform_frames * floor(prev_x / T) + j_red - 1

    for n in range(1, x.shape[0]):
        # loop init
        x_diff = x[n] - prev_x
        prev_x_red_bar = x_red + (x_red == 0.0)     # To replace (prev_x - T * floor(j_min/ waveform_frames))
        prev_j = j

        # TODO: No idea ?
        if (x_diff >= 0 and prev_x_diff >=0) or (x_diff < 0 and prev_x_diff <= 0):
            # If on the same slop as the previous iteration
            prev_j = j + int(np.sign(x_red - X[j_red]))
            # Is used to avoid computing a new j_min using the binary search, because
            # if j_min == j_max then the I sum is zero so its corresponds to the case
            # where x_n and x_n+1 are in the same interval
        
        x_red = x[n] % T

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

        prev_x_red_bar = prev_x % 1.0
        prev_x_red_bar += (prev_x_red_bar == 0.0)

        # if (prev_x - T * floor(j_min/ waveform_frames)) != prev_x_red_bar:
        #     a = prev_x - T * floor(j_min/ waveform_frames) ## ~ x[n-1] % 1.0
        #     b = (X[j_min_red - 1], X[j_min_red], X[j_min_red+1])

        # Should be differentiated upstream to avoid if on each sample
        if x_diff >= 0:
            ### OG version
            # I = expbeta\
            #         * (m[j_min_red] * x_diff + beta * (m[j_min_red] * (prev_x - T * floor(j_min/ waveform_frames)) + q[j_min_red]))\
            #         - m[j_max_p_red] * x_diff\
            #         - beta * (m[j_max_p_red] * (x[n] - T * floor((j_max+1)/waveform_frames)) + q[j_max_p_red])

            ### j_min/j_max independant version
            I = expbeta\
                    * (m[j_min_red] * x_diff + beta * (m[j_min_red] * prev_x_red_bar + q[j_min_red]))\
                    - m[j_max_p_red] * x_diff\
                    - beta * (m[j_max_p_red] * x_red + q[j_max_p_red])
        else:
            I = expbeta\
                    * (m[j_max_p_red] * x_diff + beta * (m[j_max_p_red] * (prev_x - T * floor((j_max+1)/waveform_frames)) + q[j_max_p_red]))\
                    - m[j_min_red] * x_diff\
                    - beta * (m[j_min_red] * (x[n] - T * floor(j_min/waveform_frames)) + q[j_min_red])

        I_sum = 0
        # for i in range(j_min, j_max + 1):         OG Version
        for i in range(j_min_red, j_max_p_red):
            i_red = i % waveform_frames
            x_red_bar = x[n] % 1.0
            x_red_bar = x_red_bar + (x_red_bar < X[i_red])

            I_sum += cexp(beta * (x_red_bar - X[i_red + 1])/x_diff)\
                        * (beta * q_diff[i_red] + m_diff[i_red] * (x_diff + beta * X[i_red + 1]))
            # OG Version
            # I_sum += cexp(beta * (x[n] - X[i_red + 1] - T * floor((i)/waveform_frames))/x_diff)\
            #             * (beta * q_diff[i_red] + m_diff[i_red] * (x_diff + beta * X[i_red + 1]))
    

        I = (I + np.sign(x_diff) * I_sum) / (beta**2)

        # See formula (10)
        y_cpx: complex = expbeta * prev_cpx_y + 2 * B * I
        y[n] = y_cpx.real
        abc[n] = I_sum


        prev_x = x[n]
        prev_cpx_y = y_cpx
        prev_x_diff = x_diff

    return (y, abc)




if __name__ == "__main__":
    main()