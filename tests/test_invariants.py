"""
Invariant checks of the pipeline.

They do not check "correct numbers" — they check properties that MUST hold by
the definition of the methods. If any of them breaks, the report's results are
invalid.

Run (with no extra dependency):
    python tests/test_invariants.py
or, if pytest is available:
    pytest tests/
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, degradation, edges, enhancement, io_utils, metrics  # noqa: E402
from src import frequency_filters as freq                                   # noqa: E402
from src import spatial_filters as spatial                                  # noqa: E402

SHAPE = config.TARGET_SIZE[::-1]        # (height, width)


def _ramp(shape=SHAPE):
    """Image with a linear ramp + vertical stripes: it has both tones and edges."""
    h, w = shape
    x = np.linspace(0, 255, w, dtype=np.float64)
    img = np.tile(x, (h, 1))
    img[:, ::16] = 255.0            # edges, for SSIM / edge detection
    return io_utils.to_uint8(img)


def _flat(value=128, shape=SHAPE):
    """Constant image — no clipping when noise is added."""
    return np.full(shape, value, dtype=np.uint8)


# ---------------------------------------------------------------- io_utils

def test_to_uint8_clips_instead_of_wrapping():
    """uint8 wrap-around would turn 300 into 44 — not acceptable."""
    out = io_utils.to_uint8(np.array([-50.0, -0.4, 0.0, 254.6, 300.0]))
    assert out.dtype == np.uint8
    assert list(out) == [0, 0, 0, 255, 255]


def test_references_follow_the_project_convention():
    """Project-wide convention: grayscale, uint8, TARGET_SIZE."""
    for key, (spec, img) in io_utils.load_all_references().items():
        assert img.dtype == np.uint8, key
        assert img.ndim == 2, key
        assert img.shape == SHAPE, (key, img.shape)


# ----------------------------------------------------------------- metrics

def test_psnr_and_ssim_are_perfect_for_identical_images():
    img = _ramp()
    assert metrics.mse(img, img) == 0.0
    assert metrics.psnr(img, img) == float("inf")
    assert metrics.ssim(img, img) == 1.0


def test_psnr_falls_as_noise_grows():
    """Monotonicity: more noise => lower PSNR."""
    ref = _ramp()
    p = [metrics.psnr(ref, degradation.add_gaussian_noise(
            ref, s, rng=np.random.default_rng(0))) for s in (5, 10, 25, 50)]
    assert p == sorted(p, reverse=True), p


# ------------------------------------------------------------- degradation

def test_gaussian_noise_has_the_requested_sigma():
    """On a constant image there is no clipping, so σ̂ must recover σ."""
    ref = _flat(128)
    for sigma in (10, 25):
        noisy = degradation.add_gaussian_noise(ref, sigma,
                                               rng=np.random.default_rng(1))
        assert abs(degradation.noise_profile(ref, noisy)["sigma_hat"] - sigma) < 0.5


def test_salt_pepper_hits_the_requested_density():
    ref = _flat(128)
    for d in config.SALT_PEPPER_DENSITIES:
        noisy = degradation.add_salt_pepper(ref, d, rng=np.random.default_rng(2))
        changed = degradation.noise_profile(ref, noisy)["changed_pct"] / 100.0
        assert abs(changed - d) < 0.005, (d, changed)
        # Impulse noise: only extreme values, nothing in between.
        assert set(np.unique(noisy[noisy != 128])) <= {0, 255}


def test_degradation_is_reproducible_across_stages():
    """
    The MOST critical invariant of the project: Stages 2–8 each regenerate the
    noise on their own. If two calls do not give an identical image, the PSNRs
    of different stages are compared with each other meaninglessly.
    """
    ref = _ramp()
    first = degradation.degrade_all(ref, image_key="cameraman")
    second = degradation.degrade_all(ref, image_key="cameraman")
    for (s1, a), (s2, b) in zip(first, second):
        assert s1["key"] == s2["key"]
        assert np.array_equal(a, b), s1["key"]

    # Different image => different noise realization.
    other = degradation.degrade_all(ref, image_key="moon")
    assert not np.array_equal(first[0][1], other[0][1])


# ------------------------------------------------------- frequency filters

def test_radius_grid_is_cached_and_read_only():
    a = freq.radius_grid(SHAPE)
    b = freq.radius_grid(SHAPE)
    assert a is b, "the radius array must be reused"
    assert not a.flags.writeable, "it is shared — it must be read-only"


def test_radius_grid_normalisation():
    """r = 0 at DC and r = 1 at Nyquist along the axes."""
    r = freq.radius_grid(SHAPE)
    h, w = SHAPE
    assert r[h // 2, w // 2] == 0.0
    assert np.isclose(r[h // 2, 0], 1.0)


def test_lowpass_transfer_functions_at_the_cutoff():
    """
    The identity of each filter, at radius D = D0:
      ideal        H = 1   (the cutoff is "<=")
      Butterworth  H = 0.5 for EVERY order n — by definition
      Gaussian     H = exp(-1/2)
    """
    d0 = 0.25                                   # falls exactly on pixels
    at_d0 = np.isclose(freq.radius_grid(SHAPE), d0)
    assert at_d0.any(), "D0 must coincide with actual radii"

    assert np.allclose(freq.lowpass_ideal(SHAPE, d0)[at_d0], 1.0)
    for n in config.BUTTERWORTH_ORDERS:
        H = freq.lowpass_butterworth(SHAPE, d0, order=n)
        assert np.allclose(H[at_d0], 0.5), n
    assert np.allclose(freq.lowpass_gaussian(SHAPE, d0)[at_d0], np.exp(-0.5))


def test_ideal_lowpass_is_strictly_binary():
    H = freq.lowpass_ideal(SHAPE, 0.2)
    assert set(np.unique(H)) == {0.0, 1.0}


def test_transfer_functions_pass_dc_and_decay():
    """All LPFs leave the DC untouched and never amplify (0 <= H <= 1)."""
    h, w = SHAPE
    for fam, builder in freq.LPF_BUILDERS.items():
        H = builder(SHAPE, 0.2)
        assert np.isclose(H[h // 2, w // 2], 1.0), fam
        assert H.min() >= 0.0 and H.max() <= 1.0, fam


def test_lowpass_many_matches_lowpass_exactly():
    """
    The optimized path (one DFT for all cutoffs) must give THE SAME result as
    the individual calls — otherwise the report's results change silently.
    """
    img = degradation.add_gaussian_noise(_ramp(), 25, rng=np.random.default_rng(3))
    for fam in freq.LPF_BUILDERS:
        batch = freq.lowpass_many(img, fam, config.LPF_CUTOFFS)
        for d0, got in zip(config.LPF_CUTOFFS, batch):
            assert np.array_equal(got, freq.lowpass(img, fam, d0)), (fam, d0)


def test_cumulative_energy_is_a_proper_distribution():
    radii, cum = freq.cumulative_energy(freq.fft2c(_ramp()))
    assert np.all(np.diff(cum) >= -1e-12), "it must be non-decreasing"
    assert np.isclose(cum[-1], 1.0)
    assert np.all(np.diff(radii) > 0)


def test_radius_for_energy_is_monotone_in_the_fraction():
    F = freq.fft2c(_ramp())
    r = [freq.radius_for_energy(F, lvl) for lvl in config.ENERGY_LEVELS]
    assert r == sorted(r), r


# ---------------------------------------------------------- spatial filters

def test_smoothing_reduces_sharpness():
    """Every smoothing lowers the variance of the Laplacian."""
    ref = _ramp()
    assert spatial.sharpness(spatial.gaussian(ref, 9)) < spatial.sharpness(ref)
    assert spatial.sharpness(spatial.median(ref, 9)) < spatial.sharpness(ref)


def test_median_beats_gaussian_on_salt_and_pepper():
    """
    The key conclusion of Stage 4: impulse noise is the job of the order
    statistic filter, not of linear convolution.

    It is checked on the REAL reference images and not on the synthetic
    _ramp(): the ramp has 1-pixel-wide details (the stripes at 255), i.e.
    exactly what the median filter is designed to erase. There the median loses
    (−1.6 dB) for a reason unrelated to the noise, while on the reference
    images it wins by +7 to +15 dB.
    """
    for key, (spec, ref) in io_utils.load_all_references().items():
        noisy = degradation.add_salt_pepper(ref, 0.10,
                                            rng=np.random.default_rng(4))
        assert metrics.psnr(ref, spatial.median(noisy, 3)) > \
               metrics.psnr(ref, spatial.gaussian(noisy, 3)), key


# ------------------------------------------------------------- enhancement

def test_entropy_bounds():
    """0 bits for a constant image, 8 bits for uniform use of all 256 levels."""
    assert enhancement.entropy(_flat(200)) == 0.0
    uniform = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
    assert np.isclose(enhancement.entropy(uniform), 8.0)


def test_used_levels_pct():
    assert np.isclose(enhancement.used_levels_pct(_flat(7)), 100.0 / 256.0)


def test_contrast_stretch_opens_up_the_range():
    """An image squeezed into [100,140] must be spread over all of [0,255]."""
    narrow = io_utils.to_uint8(np.linspace(100, 140, SHAPE[0] * SHAPE[1])
                               .reshape(SHAPE))
    out = enhancement.contrast_stretch(narrow)
    assert out.min() == 0 and out.max() == 255
    assert enhancement.profile(out)["std"] > enhancement.profile(narrow)["std"]


def test_histogram_equalisation_transfer_curve_is_a_cdf():
    curve = enhancement.transfer_curve(_ramp())
    assert np.all(np.diff(curve) >= 0), "the CDF never decreases"
    assert np.isclose(curve[-1], 255.0)


# ------------------------------------------------------------------- edges

def test_edge_metrics_are_perfect_against_themselves():
    m = edges.sobel_edges(_ramp())
    em = edges.edge_metrics(m, m)
    assert np.isclose(em["precision"], 1.0)
    assert np.isclose(em["recall"], 1.0)
    assert np.isclose(em["f1"], 1.0)


def test_edge_metrics_punish_an_empty_map():
    m = edges.sobel_edges(_ramp())
    em = edges.edge_metrics(m, np.zeros_like(m))
    assert em["precision"] == 0.0 and em["f1"] == 0.0


def test_sobel_percentile_threshold_selects_the_right_fraction():
    """Threshold at the p-th percentile => about (100-p)% of pixels are edges."""
    e = edges.sobel_edges(_ramp(), percentile=92.0)
    assert abs(float((e > 0).mean() * 100.0) - 8.0) < 1.0


def test_sobel_axes_respond_to_perpendicular_edges():
    """Gx responds to VERTICAL edges, Gy to HORIZONTAL ones — not the reverse."""
    vertical = np.zeros(SHAPE, dtype=np.uint8)
    vertical[:, SHAPE[1] // 2:] = 255            # a single vertical edge
    gx, gy = edges.sobel_components(vertical)
    assert np.abs(gx).max() > 0
    assert np.abs(gy).max() == 0


def _run_standalone() -> int:
    """Runs all test_* without pytest, so no dependency is needed."""
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except Exception as exc:                      # noqa: BLE001
            failed.append((name, exc))
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
