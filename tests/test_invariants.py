"""
Έλεγχοι αναλλοίωτων μεγεθών του pipeline (ΕΥ631).

Δεν ελέγχουν «σωστά νούμερα» — ελέγχουν ιδιότητες που ΠΡΕΠΕΙ να ισχύουν από τον
ορισμό των μεθόδων. Αν κάποια σπάσει, τα αποτελέσματα της αναφοράς είναι άκυρα.

Εκτέλεση (χωρίς καμία επιπλέον εξάρτηση):
    python tests/test_invariants.py
ή, αν υπάρχει pytest:
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
    """Εικόνα με γραμμική ράμπα + κατακόρυφες λωρίδες: έχει και τόνους και ακμές."""
    h, w = shape
    x = np.linspace(0, 255, w, dtype=np.float64)
    img = np.tile(x, (h, 1))
    img[:, ::16] = 255.0            # ακμές, για SSIM / ανίχνευση ακμών
    return io_utils.to_uint8(img)


def _flat(value=128, shape=SHAPE):
    """Σταθερή εικόνα — χωρίς clipping όταν προστίθεται θόρυβος."""
    return np.full(shape, value, dtype=np.uint8)


# ---------------------------------------------------------------- io_utils

def test_to_uint8_clips_instead_of_wrapping():
    """Το wrap-around του uint8 θα έκανε το 300 να γίνει 44 — απαγορευτικό."""
    out = io_utils.to_uint8(np.array([-50.0, -0.4, 0.0, 254.6, 300.0]))
    assert out.dtype == np.uint8
    assert list(out) == [0, 0, 0, 255, 255]


def test_references_follow_the_project_convention():
    """Σύμβαση όλου του project: grayscale, uint8, TARGET_SIZE."""
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
    """Μονοτονία: περισσότερος θόρυβος => χαμηλότερο PSNR."""
    ref = _ramp()
    p = [metrics.psnr(ref, degradation.add_gaussian_noise(
            ref, s, rng=np.random.default_rng(0))) for s in (5, 10, 25, 50)]
    assert p == sorted(p, reverse=True), p


# ------------------------------------------------------------- degradation

def test_gaussian_noise_has_the_requested_sigma():
    """Σε σταθερή εικόνα δεν υπάρχει clipping, οπότε το σ̂ πρέπει να βρει το σ."""
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
        # Κρουστικός θόρυβος: μόνο ακραίες τιμές, τίποτα ενδιάμεσο.
        assert set(np.unique(noisy[noisy != 128])) <= {0, 255}


def test_degradation_is_reproducible_across_stages():
    """
    Το ΠΙΟ κρίσιμο αναλλοίωτο του project: τα Στάδια 2–8 ξαναφτιάχνουν τον θόρυβο
    το καθένα μόνο του. Αν δύο κλήσεις δεν δώσουν πανομοιότυπη εικόνα, τα PSNR
    διαφορετικών σταδίων συγκρίνονται μεταξύ τους χωρίς νόημα.
    """
    ref = _ramp()
    first = degradation.degrade_all(ref, image_key="cameraman")
    second = degradation.degrade_all(ref, image_key="cameraman")
    for (s1, a), (s2, b) in zip(first, second):
        assert s1["key"] == s2["key"]
        assert np.array_equal(a, b), s1["key"]

    # Διαφορετική εικόνα => διαφορετική πραγματοποίηση θορύβου.
    other = degradation.degrade_all(ref, image_key="moon")
    assert not np.array_equal(first[0][1], other[0][1])


# ------------------------------------------------------- frequency filters

def test_radius_grid_is_cached_and_read_only():
    a = freq.radius_grid(SHAPE)
    b = freq.radius_grid(SHAPE)
    assert a is b, "ο πίνακας ακτίνων πρέπει να επαναχρησιμοποιείται"
    assert not a.flags.writeable, "μοιράζεται — πρέπει να είναι read-only"


def test_radius_grid_normalisation():
    """r = 0 στη DC και r = 1 στο Nyquist κατά μήκος των αξόνων."""
    r = freq.radius_grid(SHAPE)
    h, w = SHAPE
    assert r[h // 2, w // 2] == 0.0
    assert np.isclose(r[h // 2, 0], 1.0)


def test_lowpass_transfer_functions_at_the_cutoff():
    """
    Η ταυτότητα κάθε φίλτρου, στην ακτίνα D = D0:
      ιδανικό      H = 1   (η αποκοπή είναι «<=»)
      Butterworth  H = 0.5 για ΚΑΘΕ τάξη n — εξ ορισμού
      γκαουσιανό   H = exp(-1/2)
    """
    d0 = 0.25                                   # πέφτει ακριβώς πάνω σε pixel
    at_d0 = np.isclose(freq.radius_grid(SHAPE), d0)
    assert at_d0.any(), "το D0 πρέπει να συμπίπτει με πραγματικές ακτίνες"

    assert np.allclose(freq.lowpass_ideal(SHAPE, d0)[at_d0], 1.0)
    for n in config.BUTTERWORTH_ORDERS:
        H = freq.lowpass_butterworth(SHAPE, d0, order=n)
        assert np.allclose(H[at_d0], 0.5), n
    assert np.allclose(freq.lowpass_gaussian(SHAPE, d0)[at_d0], np.exp(-0.5))


def test_ideal_lowpass_is_strictly_binary():
    H = freq.lowpass_ideal(SHAPE, 0.2)
    assert set(np.unique(H)) == {0.0, 1.0}


def test_transfer_functions_pass_dc_and_decay():
    """Όλα τα ΧΠΦ αφήνουν τη DC ανέπαφη και δεν ενισχύουν ποτέ (0 <= H <= 1)."""
    h, w = SHAPE
    for fam, builder in freq.LPF_BUILDERS.items():
        H = builder(SHAPE, 0.2)
        assert np.isclose(H[h // 2, w // 2], 1.0), fam
        assert H.min() >= 0.0 and H.max() <= 1.0, fam


def test_lowpass_many_matches_lowpass_exactly():
    """
    Ο βελτιστοποιημένος δρόμος (ένας DFT για όλες τις αποκοπές) οφείλει να δίνει
    ΤΟ ΙΔΙΟ αποτέλεσμα με τις μεμονωμένες κλήσεις — αλλιώς τα αποτελέσματα της
    αναφοράς αλλάζουν σιωπηλά.
    """
    img = degradation.add_gaussian_noise(_ramp(), 25, rng=np.random.default_rng(3))
    for fam in freq.LPF_BUILDERS:
        batch = freq.lowpass_many(img, fam, config.LPF_CUTOFFS)
        for d0, got in zip(config.LPF_CUTOFFS, batch):
            assert np.array_equal(got, freq.lowpass(img, fam, d0)), (fam, d0)


def test_cumulative_energy_is_a_proper_distribution():
    radii, cum = freq.cumulative_energy(freq.fft2c(_ramp()))
    assert np.all(np.diff(cum) >= -1e-12), "πρέπει να είναι αύξουσα"
    assert np.isclose(cum[-1], 1.0)
    assert np.all(np.diff(radii) > 0)


def test_radius_for_energy_is_monotone_in_the_fraction():
    F = freq.fft2c(_ramp())
    r = [freq.radius_for_energy(F, lvl) for lvl in config.ENERGY_LEVELS]
    assert r == sorted(r), r


# ---------------------------------------------------------- spatial filters

def test_smoothing_reduces_sharpness():
    """Κάθε εξομάλυνση μειώνει τη διασπορά του Λαπλασιανού."""
    ref = _ramp()
    assert spatial.sharpness(spatial.gaussian(ref, 9)) < spatial.sharpness(ref)
    assert spatial.sharpness(spatial.median(ref, 9)) < spatial.sharpness(ref)


def test_median_beats_gaussian_on_salt_and_pepper():
    """
    Το βασικό συμπέρασμα του Σταδίου 4: ο κρουστικός θόρυβος είναι δουλειά του
    φίλτρου τάξης, όχι της γραμμικής συνέλιξης.

    Ελέγχεται στις ΠΡΑΓΜΑΤΙΚΕΣ εικόνες αναφοράς και όχι στη συνθετική _ramp():
    η ράμπα έχει λεπτομέρειες πλάτους 1 pixel (οι λωρίδες στο 255), δηλαδή
    ακριβώς ό,τι σχεδιάστηκε να σβήνει το φίλτρο διαμέσου. Εκεί το median χάνει
    (−1.6 dB) για λόγο άσχετο με τον θόρυβο, ενώ στις εικόνες αναφοράς κερδίζει
    κατά +7 έως +15 dB.
    """
    for key, (spec, ref) in io_utils.load_all_references().items():
        noisy = degradation.add_salt_pepper(ref, 0.10,
                                            rng=np.random.default_rng(4))
        assert metrics.psnr(ref, spatial.median(noisy, 3)) > \
               metrics.psnr(ref, spatial.gaussian(noisy, 3)), key


# ------------------------------------------------------------- enhancement

def test_entropy_bounds():
    """0 bits για σταθερή εικόνα, 8 bits για ομοιόμορφη χρήση των 256 επιπέδων."""
    assert enhancement.entropy(_flat(200)) == 0.0
    uniform = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
    assert np.isclose(enhancement.entropy(uniform), 8.0)


def test_used_levels_pct():
    assert np.isclose(enhancement.used_levels_pct(_flat(7)), 100.0 / 256.0)


def test_contrast_stretch_opens_up_the_range():
    """Μια εικόνα στριμωγμένη στο [100,140] πρέπει να απλωθεί σε όλο το [0,255]."""
    narrow = io_utils.to_uint8(np.linspace(100, 140, SHAPE[0] * SHAPE[1])
                               .reshape(SHAPE))
    out = enhancement.contrast_stretch(narrow)
    assert out.min() == 0 and out.max() == 255
    assert enhancement.profile(out)["std"] > enhancement.profile(narrow)["std"]


def test_histogram_equalisation_transfer_curve_is_a_cdf():
    curve = enhancement.transfer_curve(_ramp())
    assert np.all(np.diff(curve) >= 0), "η CDF δεν μειώνεται ποτέ"
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
    """Κατώφλι στο p-οστό εκατοστημόριο => περίπου (100-p)% των pixel ακμές."""
    e = edges.sobel_edges(_ramp(), percentile=92.0)
    assert abs(float((e > 0).mean() * 100.0) - 8.0) < 1.0


def test_sobel_axes_respond_to_perpendicular_edges():
    """Gx αποκρίνεται σε ΚΑΤΑΚΟΡΥΦΕΣ ακμές, Gy σε ΟΡΙΖΟΝΤΙΕΣ — όχι το αντίστροφο."""
    vertical = np.zeros(SHAPE, dtype=np.uint8)
    vertical[:, SHAPE[1] // 2:] = 255            # μία κατακόρυφη ακμή
    gx, gy = edges.sobel_components(vertical)
    assert np.abs(gx).max() > 0
    assert np.abs(gy).max() == 0


def _run_standalone() -> int:
    """Τρέχει όλα τα test_* χωρίς pytest, ώστε να μη χρειάζεται εξάρτηση."""
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except Exception as exc:                      # noqa: BLE001
            failed.append((name, exc))
            print(f"  ΣΦΑΛΜΑ {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} πέρασαν")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
