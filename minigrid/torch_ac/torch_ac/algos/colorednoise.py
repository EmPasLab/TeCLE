import numpy as np
from numpy.fft import irfft, rfftfreq


def powerlaw_psd_gaussian(exponent, size, fmin=0, rng=None):


    try:
        size = list(size)
    except TypeError:
        size = [size]


    samples = size[-1]


    f = rfftfreq(samples)


    if 0 <= fmin <= 0.5:
        fmin = max(fmin, 1./samples)
    else:
        raise ValueError("fmin must be chosen between 0 and 0.5.")


    s_scale = f
    ix = np.sum(s_scale < fmin)
    if ix and ix < len(s_scale):
        s_scale[:ix] = s_scale[ix]
    s_scale = s_scale**(-exponent/2.)


    w = s_scale[1:].copy()
    w[-1] *= (1 + (samples % 2)) / 2.
    sigma = 2 * np.sqrt(np.sum(w**2)) / samples


    size[-1] = len(f)


    dims_to_add = len(size) - 1
    s_scale = s_scale[(None,) * dims_to_add + (Ellipsis,)]


    if rng is None:
        rng = np.random.default_rng()
    sr = rng.normal(scale=s_scale, size=size)
    si = rng.normal(scale=s_scale, size=size)


    if not (samples % 2):
        si[..., -1] = 0
        sr[..., -1] *= np.sqrt(2)


    si[..., 0] = 0
    sr[..., 0] *= np.sqrt(2)


    s = sr + 1J * si


    y = irfft(s, n=samples, axis=-1) / sigma

    return y
