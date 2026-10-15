"""
STAGES 3 & 5 — Frequency domain.

Stage 3 (here): analysis. Fourier transform, magnitude spectrum, radial power
profile and energy distribution over frequency.
Stage 5 (further down): Butterworth / Gaussian low-pass filters.

Conventions: all spectra are centered (fftshift), i.e. the DC component sits
at the center of the array. Radii are normalized to the Nyquist frequency:
r=0 -> DC, r=1 -> the highest frequency along the axes.
"""
from functools import lru_cache

import numpy as np

from . import config
from .io_utils import to_uint8


def fft2c(img) -> np.ndarray:
    """2D DFT with a centered (fftshift) result, in float64."""
    return np.fft.fftshift(np.fft.fft2(np.asarray(img, dtype=np.float64)))


def ifft2c(F) -> np.ndarray:
    """Inverse of fft2c; returns the real part (the image is real-valued)."""
    return np.real(np.fft.ifft2(np.fft.ifftshift(F)))


def magnitude(F) -> np.ndarray:
    """Magnitude spectrum |F|."""
    return np.abs(F)


def power(F) -> np.ndarray:
    """Power spectrum |F|²."""
    a = np.abs(F)
    return a * a


def log_spectrum_image(F) -> np.ndarray:
    """Spectrum as log(1+|F|) in uint8; without the logarithm the DC dominates."""
    s = np.log1p(magnitude(F))
    lo, hi = float(s.min()), float(s.max())
    if hi <= lo:
        return np.zeros(s.shape, dtype=np.uint8)
    return to_uint8((s - lo) / (hi - lo) * 255.0)


@lru_cache(maxsize=8)
def _radius_grid_cached(h: int, w: int) -> np.ndarray:
    """
    The radius array for given dimensions. It depends only on (h, w), so it is
    computed once and reused: a full run requests it hundreds of times
    (filters, radial profiles, spectrum energy).

    The array is returned read-only because all callers share it: an accidental
    future modification then fails immediately, instead of silently corrupting
    the results of every subsequent call.
    """
    v = np.arange(h) - h // 2
    u = np.arange(w) - w // 2
    rr, cc = np.meshgrid(v, u, indexing="ij")
    r = np.sqrt(rr ** 2 + cc ** 2) / (min(h, w) / 2.0)
    r.flags.writeable = False
    return r


def radius_grid(shape) -> np.ndarray:
    """Distance from the center (DC), normalized: r = 1 at Nyquist."""
    h, w = shape[:2]
    return _radius_grid_cached(int(h), int(w))


def radial_power_profile(F, nbins=128):
    """
    Mean power per radial ring -> (ring centers, mean power). Mean rather than
    sum, because larger radii contain more coefficients.
    """
    p = power(F)
    r = radius_grid(p.shape)
    rmax = float(r.max())
    idx = np.clip((r / rmax * nbins).astype(int), 0, nbins - 1)
    total = np.bincount(idx.ravel(), weights=p.ravel(), minlength=nbins)
    count = np.bincount(idx.ravel(), minlength=nbins)
    prof = np.divide(total, count, out=np.zeros(nbins), where=count > 0)
    centers = (np.arange(nbins) + 0.5) / nbins * rmax
    return centers, prof


def _drop_dc(p) -> np.ndarray:
    """Zeroes the DC: otherwise 90% of the energy falls within radius ~0.01."""
    q = np.array(p, dtype=np.float64, copy=True)
    h, w = q.shape[:2]
    q[h // 2, w // 2] = 0.0
    return q


def cumulative_energy(F, nbins=512, exclude_dc=True):
    """
    Spectrum energy within radius r -> (radii, fraction in [0,1]).
    The DC is ignored by default (_drop_dc).
    """
    p = _drop_dc(power(F)) if exclude_dc else power(F)
    r = radius_grid(p.shape)
    rmax = float(r.max())
    idx = np.clip((r / rmax * nbins).astype(int), 0, nbins - 1)
    ring = np.bincount(idx.ravel(), weights=p.ravel(), minlength=nbins)
    cum = np.cumsum(ring)
    cum /= cum[-1]
    radii = (np.arange(nbins) + 1) / nbins * rmax
    return radii, cum


def radius_for_energy(F, fraction, nbins=512, exclude_dc=True) -> float:
    """The smallest (normalized) radius that encloses `fraction` of the energy."""
    radii, cum = cumulative_energy(F, nbins=nbins, exclude_dc=exclude_dc)
    i = int(np.searchsorted(cum, fraction))
    return float(radii[min(i, len(radii) - 1)])


def high_freq_energy_ratio(F, r0=None, exclude_dc=True) -> float:
    """Percentage (%) of the energy (without DC) that lies beyond radius r0."""
    r0 = config.HF_RADIUS_FRAC if r0 is None else r0
    p = _drop_dc(power(F)) if exclude_dc else power(F)
    r = radius_grid(p.shape)
    return float(p[r > r0].sum() / p.sum() * 100.0)


def spectrum_report(img) -> dict:
    """All radial quantities of an image together."""
    F = fft2c(img)
    rep = {"hf_pct": high_freq_energy_ratio(F)}
    for lvl in config.ENERGY_LEVELS:
        rep[f"r{int(lvl * 100)}"] = radius_for_energy(F, lvl)
    return rep


def snr_crossover_radius(ref, noisy, nbins=128) -> float:
    """
    The radius where the noise power exceeds the signal power (SNR = 1) — the
    value range for the Stage 5 cutoff frequency.
    Returns NaN if the signal dominates everywhere.
    """
    r, p_sig = radial_power_profile(fft2c(ref), nbins=nbins)
    noise = np.asarray(noisy, dtype=np.float64) - np.asarray(ref, dtype=np.float64)
    _, p_noise = radial_power_profile(fft2c(noise), nbins=nbins)
    above = np.nonzero(p_noise > p_sig)[0]
    return float(r[above[0]]) if above.size else float("nan")


# STAGE 5 — Low-pass filters
#
# Each filter is defined as a transfer function H(u,v) that depends only on the
# radius D. D0 is given on the same normalized scale as Stage 3
# (1 = Nyquist).

def lowpass_ideal(shape, d0) -> np.ndarray:
    """Ideal low-pass: H = 1 for D ≤ D0, otherwise 0."""
    return (radius_grid(shape) <= float(d0)).astype(np.float64)


def lowpass_butterworth(shape, d0, order=None) -> np.ndarray:
    """Butterworth: H = 1 / (1 + (D/D0)^(2n)). At D0 it always gives H = 0.5."""
    n = config.BUTTERWORTH_ORDER if order is None else order
    d = radius_grid(shape) / float(d0)
    return 1.0 / (1.0 + d ** (2 * n))


def lowpass_gaussian(shape, d0) -> np.ndarray:
    """Gaussian: H = exp(-D²/(2·D0²))."""
    d = radius_grid(shape)
    return np.exp(-(d ** 2) / (2.0 * float(d0) ** 2))


def apply_transfer(img, H) -> np.ndarray:
    """Frequency-domain filtering: G = H·F, then the inverse DFT."""
    F = fft2c(img)
    return to_uint8(ifft2c(F * H))


def butterworth_key(n) -> str:
    """The family key of the order-n Butterworth (e.g. n=2 -> "butterworth_n2")."""
    return f"butterworth_n{n}"


def _make_butterworth(n):
    """
    Builder of H(u,v) for a fixed order n. A separate function rather than a
    lambda inside the loop, so that n is bound here and not at call time.
    """
    def builder(shape, d0):
        return lowpass_butterworth(shape, d0, order=n)
    builder.__name__ = f"lowpass_butterworth_n{n}"
    builder.order = n
    return builder


# Each Butterworth order is a separate filter family: its own kernel, its own
# cutoff sweep, its own PSNR/SSIM.
BUTTERWORTH_KEYS = tuple(butterworth_key(n) for n in config.BUTTERWORTH_ORDERS)

LPF_BUILDERS = {"ideal": lowpass_ideal}
LPF_BUILDERS.update({butterworth_key(n): _make_butterworth(n)
                     for n in config.BUTTERWORTH_ORDERS})
LPF_BUILDERS["gaussian"] = lowpass_gaussian

LPF_LABEL = {"ideal": "Ideal LPF"}
LPF_LABEL.update({butterworth_key(n): f"Butterworth n={n}"
                  for n in config.BUTTERWORTH_ORDERS})
LPF_LABEL["gaussian"] = "Gaussian LPF"

# The order of each family (None for non-Butterworth ones) — a table column.
LPF_ORDER = {k: getattr(b, "order", None) for k, b in LPF_BUILDERS.items()}


def lowpass_many(img, family, cutoffs) -> list:
    """
    The same filter for many cutoff frequencies, with only ONE forward DFT.

    The forward transform F does not depend on D0: only the transfer function H
    changes. The Stage 5 and Stage 8 sweeps try 9 cutoff frequencies per family,
    so computing F once instead of nine times saves 8 of the 9 forward
    transforms, with a numerically identical result.
    """
    F = fft2c(img)
    builder = LPF_BUILDERS[family]
    shape = np.asarray(img).shape
    return [to_uint8(ifft2c(F * builder(shape, d0))) for d0 in cutoffs]


def lowpass(img, family, d0) -> np.ndarray:
    """Applies a low-pass filter of the given family."""
    return lowpass_many(img, family, (d0,))[0]


def radial_transfer_profile(H, nbins=128):
    """The profile H(D) versus radius — for the frequency-response plot."""
    r = radius_grid(H.shape)
    rmax = float(r.max())
    idx = np.clip((r / rmax * nbins).astype(int), 0, nbins - 1)
    tot = np.bincount(idx.ravel(), weights=H.ravel(), minlength=nbins)
    cnt = np.bincount(idx.ravel(), minlength=nbins)
    prof = np.divide(tot, cnt, out=np.zeros(nbins), where=cnt > 0)
    return (np.arange(nbins) + 0.5) / nbins * rmax, prof
