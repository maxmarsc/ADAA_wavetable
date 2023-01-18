from __future__ import division
from numpy import asarray, zeros, pi, sin, cos, amax, diff, arange, outer

# see https://gist.github.com/endolith/407991

def bl_sawtooth(x, play_freq): # , width=1
    """
    Return a periodic band-limited sawtooth wave with
    period 2*pi which is falling from 0 to 2*pi and rising at
    2*pi (opposite phase relative to a sin)
    Produces the same phase and amplitude as scipy.signal.sawtooth.
    Examples
    --------
    >>> t = linspace(0, 1, num = 1000, endpoint = False)
    >>> f = 5 # Hz
    >>> plot(bl_sawtooth(2 * pi * f * t))
    """
    t = asarray(2 * pi * play_freq * x)

    if abs((t[-1]-t[-2]) - (t[1]-t[0])) > .0000001:
        raise ValueError("Sampling frequency must be constant")

    if t.dtype.char in ['fFdD']:
        ytype = t.dtype.char
    else:
        ytype = 'd'
    y = zeros(t.shape, ytype)

    # Get sampling frequency from timebase
    fs =  1 / (t[1] - t[0])
    #    fs =  1 / amax(diff(t))

    # Sum all multiple sine waves up to the Nyquist frequency

    # TODO: Maybe choose between these based on number of harmonics?

    # Slower, uses less memory
    for h in range(1, int(fs*pi)+1):
        y += 2 / pi * -sin(h * t) / h

    return y