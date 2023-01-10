
import numpy as np

WAVEFORM_LEN = 2048
SAMPLERATE = 44100
BUTTERWORTH_CTF = 0.45 * SAMPLERATE
DURATION_S = 0.1


BUTTER2_COEFFS = np.array([
    [0.0000 - 2.2164j, 0.0000 + 2.2164j],                   # residuals
    [-2.2164 + 2.2164j, -2.2164 - 2.2164j,]                 # poles
])

NAIVE_SAW = np.linspace(-1.0, 1.0, WAVEFORM_LEN, endpoint=False)
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



def main():
    orders = len(BUTTER2_COEFFS[0])
    play_freq = 1000

    num_frames_total = int(DURATION_S * SAMPLERATE)
    x = np.linspace(0.0, num_frames_total*play_freq / SAMPLERATE, num_frames_total, endpoint=False)
    y = np.zeros(num_frames_total)
    (m, q) =  compute_m_q_vectors(NAIVE_SAW, NAIVE_SAW_X)

    for order in range(0, orders, 2):
        ri = BUTTER2_COEFFS[0][order]
        zi = BUTTER2_COEFFS[1][order]
        y += process(x, ri, zi, NAIVE_SAW_X, m, q)


def process(x, ri, zi, X, m, q):
    return np.zeros(x.shape[0])




if __name__ == "__main__":
    main()