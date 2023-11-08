from numba import njit
import numpy as np
from math import exp, floor, ceil
from cmath import exp as cexp
from typing import Tuple, List, Dict
from mipmap import *

@njit
def binary_search_down(x : np.ndarray, x0: float, j_min: int, j_max: int) -> int:
    """
    return i as x_i < x_0 < x_(i+1) && j_min <= i <= j_max (or j_max - 1)
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

    FIXME: I think this function is bugged
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
def process_bi(x, B, beta: complex, X, m, q, m_diff, q_diff):
    """
    Direct translation from the matlab algorithm to python. Be aware matlab arrays starts at 1 so I had to make
    a few changes


    This code contains A LOT of annotations with commented out stuff. This is because I want to have a written trace
    of the adaptation I had to make to write the simplified process_fwd() version.

    Notice that this code does support reverse playback whereas process_fwd() does not
    """
    y = np.zeros(x.shape[0])

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
    # j_red = binary_search_down(X, x_red, 0, X.shape[0] - 1)
    j_red = floor(x_red * (X.shape[0] - 1))
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
            # j_red = binary_search_down(X, x_red, 0, X.shape[0] - 1)
            j_red = floor(x_red * waveform_frames)

            j = waveform_frames * floor(x[n] / 1.0) + j_red - 1
            j_min = prev_j
            j_max = j
        else:
            # playback going backward
            # j_red = binary_search_up(X, x_red, 0, X.shape[0] - 1)
            j_red = ceil(x_red * waveform_frames)

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

        if x_diff < 0 and j_min != -1 and j_min_red > j_max_p_red:
            cycle_offset = -1.0
        else:
            cycle_offset = 0.0

        for i in range(j_min, j_max + 1):         #OG Version
            i_red = i % waveform_frames
            ref_bi = x_red + cycle_offset + (i_red > j_max_p_red)

            I_sum += cexp(beta * (x[n] - X[i_red + 1] - T * floor((i)/waveform_frames))/x_diff)\
                        * (beta * q_diff[i_red] + m_diff[i_red] * (x_diff + beta * X[i_red + 1]))

        I = (I + np.sign(x_diff) * I_sum) / (beta**2)

        # See formula (10)
        y_cpx: complex = expbeta * prev_cpx_y + 2 * B * I
        y[n] = y_cpx.real

        prev_x = x[n]
        prev_cpx_y = y_cpx
        prev_x_diff = x_diff

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
    # j_red = binary_search_down(X, x_red, 0, X.shape[0] - 1)
    j_red = floor(x_red * (X.shape[0] - 1))

    for n in range(1, x.shape[0]):
        # loop init
        x_diff = x[n] - prev_x
        assert(x_diff >= 0)
        # prev_x_red_bar = x_red + (x_red == 0.0)     # To replace (prev_x - T * floor(j_min/ waveform_frames))
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
        # j_red = binary_search_down(X, x_red, 0, X.shape[0] - 1)
        j_red = floor(x_red * (X.shape[0] - 1))
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


@njit
def process_fwd_mipmap_xfading(x, B, beta: complex, X_mipmap: List[np.ndarray[float]], 
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
    for phases in X_mipmap:
        assert(phases[-1] == 1.0)

    expbeta = cexp(beta)

    # Initial condition
    prev_x = x[0]
    prev_cpx_y: complex = 0

    # Setting j indexs and some reduced values
    x_red = prev_x % 1.0
    x_diff = x[1] - x[0]
    mipmap_xfade_idxs = find_mipmap_xfading_indexes(x_diff, mipmap_scale)
    prev_mipmap_idx = mipmap_xfade_idxs[0]
    # j_red = binary_search_down(X_mipmap[prev_mipmap_idx], x_red, 0, X_mipmap[prev_mipmap_idx].shape[0] - 1)
    waveform_frames = m_mipmap[prev_mipmap_idx].shape[0]
    j_red = floor(x_red * waveform_frames)

    for n in range(1, x.shape[0]):
        # loop init
        x_diff = x[n] - prev_x
        if x_diff <= 0:
            x_diff += 1.0
        assert(x_diff >= 0)

        mipmap_idx, weight, mipmap_idx_up, weight_up = find_mipmap_xfading_indexes(x_diff, mipmap_scale)
        waveform_frames = m_mipmap[mipmap_idx].shape[0]     # aka k
        prev_x_red_bar = x_red + (x_red == 0.0)     # To replace (prev_x - T * floor(j_min/ waveform_frames))
        
        if mipmap_idx != prev_mipmap_idx:
            if mipmap_idx == prev_mipmap_idx + 1:
                # Going up in frequencies
                j_red = j_red // 2
            else:
                # Going down in frequencies
                j_red = j_red * 2 + (X_mipmap[mipmap_idx][j_red * 2 + 1] < x_red)
        prev_j_red = j_red + int(np.sign(x_red - X_mipmap[mipmap_idx][j_red]))

        
        x_red = x[n] % 1.0

        # playback going forward
        j_red = floor(x_red * waveform_frames)

        j_max_p_red = j_red
        j_min_red = (prev_j_red - 1) % waveform_frames


        prev_x_red_bar = prev_x % 1.0
        prev_x_red_bar += (prev_x_red_bar == 0.0)

        I_crt = compute_I_fwd(m_mipmap[mipmap_idx],
                          q_mipmap[mipmap_idx],
                          m_diff_mipmap[mipmap_idx],
                          q_diff_mipmap[mipmap_idx],
                          X_mipmap[mipmap_idx],
                          j_min_red, j_max_p_red,
                          beta, expbeta, x_diff, prev_x_red_bar, x_red)
        

        if weight_up != 0.0:
            jmin_up = j_min_red // 2
            jmax_up = j_max_p_red // 2

            # # -- Only to make sure I didn't made a mistake in index computation
            # ref_max = binary_search_down(X_mipmap[mipmap_idx_up], x_red, 0, X_mipmap[mipmap_idx_up].shape[0] - 1)
            # prev_x_red = x[n-1] % 1.0
            # ref_prev_j_red = binary_search_down(X_mipmap[mipmap_idx_up], prev_x_red, 0, X_mipmap[mipmap_idx_up].shape[0] - 1)
            # ref_prev_j_red = ref_prev_j_red + int(np.sign(prev_x_red - X_mipmap[mipmap_idx_up][ref_prev_j_red]))
            # ref_min = (ref_prev_j_red -1 ) % m_mipmap[mipmap_idx_up].shape[0]
            # assert(jmin_up == ref_min)
            # assert(jmax_up == ref_max)
            # # ---

            I_up = compute_I_fwd(m_mipmap[mipmap_idx_up],
                               q_mipmap[mipmap_idx_up],
                                m_diff_mipmap[mipmap_idx_up],
                                q_diff_mipmap[mipmap_idx_up],
                                X_mipmap[mipmap_idx_up],
                                jmin_up, jmax_up,
                                beta, expbeta, x_diff, prev_x_red_bar, x_red)
            I_crt = I_crt * weight + weight_up * I_up
            
        y_cpx: complex = expbeta * prev_cpx_y + 2 * B * (I_crt / (beta ** 2))
        # y_cpx: complex = expbeta * prev_cpx_y + 2 * B * I_crt
        y[n] = y_cpx.real


        prev_x = x[n]

        prev_cpx_y = y_cpx
        # prev_x_diff = x_diff # Not required for fwd
        prev_mipmap_idx = mipmap_idx

    return y

@njit
def compute_I_fwd(m, q, m_diff, q_diff, X, jmin, jmax, beta, expbeta, x_diff, prev_x_red_bar, x_red) -> complex:
    frames = m.shape[0]
    I = expbeta\
        * (m[jmin] * x_diff + beta * (m[jmin] * prev_x_red_bar + q[jmin]))\
        - m[jmax] * x_diff\
        - beta * (m[jmax] * x_red + q[jmax])

    I_sum = 0
    born_sup = jmax + frames * (jmin > jmax)
    for i in range(jmin, born_sup):
        i_red = i % frames
        x_red_bar = x_red + (x_red < X[i_red])

        I_sum += cexp(beta * (x_red_bar - X[i_red + 1])/x_diff)\
                    * (beta * q_diff[i_red] + m_diff[i_red] * (x_diff + beta * X[i_red + 1]))
        
    # return (I + np.sign(x_diff) * I_sum) / (beta**2)
    return I + I_sum


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
    # j_red = binary_search_down(X_mipmap[prev_mipmap_idx], x_red, 0, X_mipmap[prev_mipmap_idx].shape[0] - 1)
    j_red = floor(x_red * X_mipmap[prev_mipmap_idx].shape[0] - 1)

    for n in range(1, x.shape[0]):
        # loop init
        x_diff = x[n] - prev_x
        if x_diff <= 0:
            x_diff += 1.0
        assert(x_diff >= 0)
        mipmap_idx = find_mipmap_index(x_diff, mipmap_scale)
        waveform_frames = m_mipmap[mipmap_idx].shape[0]     # aka k
        prev_x_red_bar = x_red + (x_red == 0.0)     # To replace (prev_x - T * floor(j_min/ waveform_frames))
        
        if mipmap_idx == prev_mipmap_idx:
            prev_j_red = j_red + int(np.sign(x_red - X_mipmap[mipmap_idx][j_red]))
        else:
            # j_red = binary_search_down(X_mipmap[mipmap_idx], x_red, 0, X_mipmap[mipmap_idx].shape[0] - 1)
            j_red = floor(x_red * X_mipmap[mipmap_idx].shape[0] - 1)
            prev_j_red = j_red + int(np.sign(x_red - X_mipmap[mipmap_idx][j_red]))

        
        x_red = x[n] % 1.0

        # playback going forward
        # j_red = binary_search_down(X_mipmap[mipmap_idx], x_red, 0, X_mipmap[mipmap_idx].shape[0] - 1)
        j_red = floor(x_red * X_mipmap[mipmap_idx].shape[0] - 1)
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
        # assert(abs(y[n]) < 3.0)


        prev_x = x[n]

        prev_cpx_y = y_cpx
        prev_x_diff = x_diff
        prev_mipmap_idx = mipmap_idx

    return y


@njit
def process_naive_hermite(waveform, x_values):
    """
    Hermite interpolation algorithm

    Based on ADC21 talk by Matt Tytel, who based himself
    on implementation by Laurent de Soras

    https://www.youtube.com/watch?v=qlinVx60778
    """

    y = np.zeros(x_values.shape[0])
    waveform_len = waveform.shape[0]

    for (i, x) in enumerate(x_values):
        x_red = x % 1.0

        relative_idx = x_red * waveform_len
        idx_0 = (floor(relative_idx) - 1) % waveform_len
        idx_1 = floor(relative_idx)
        idx_2 = (floor(relative_idx) + 1 ) % waveform_len
        idx_3 = (floor(relative_idx) + 2 ) % waveform_len

        sample_offset = relative_idx - idx_1

        slope_0 = (waveform[idx_2] - waveform[idx_0]) * 0.5
        slope_1 = (waveform[idx_3] - waveform[idx_1]) * 0.5

        v = waveform[idx_1] - waveform[idx_2]
        w = slope_0 + v
        a = w + v + slope_1
        b_neg = w + a
        stage_1 = a * sample_offset - b_neg
        stage_2 = stage_1 * sample_offset + slope_0
        y[i] = stage_2 * sample_offset + waveform[idx_1]
        # assert(y[i] != np.NaN)

    return y
