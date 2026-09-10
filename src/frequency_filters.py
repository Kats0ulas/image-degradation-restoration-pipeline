"""
ΣΤΑΔΙΑ 3 & 5 — Πεδίο συχνοτήτων.

Στάδιο 3 (εδώ): ανάλυση. Μετασχηματισμός Fourier, φάσμα πλάτους, ακτινικό
προφίλ ισχύος και κατανομή ενέργειας ανά συχνότητα.
Στάδιο 5 (αργότερα): χαμηλοπερατά φίλτρα Butterworth / Gaussian.

Συμβάσεις: όλα τα φάσματα είναι κεντραρισμένα (fftshift), δηλαδή η DC συνιστώσα
βρίσκεται στο κέντρο του πίνακα. Οι ακτίνες κανονικοποιούνται στη συχνότητα
Nyquist: r=0 -> DC, r=1 -> η υψηλότερη συχνότητα κατά τους άξονες.
"""
from functools import lru_cache

import numpy as np

from . import config
from .io_utils import to_uint8


def fft2c(img) -> np.ndarray:
    """2D DFT με κεντραρισμένο (fftshift) αποτέλεσμα, σε float64."""
    return np.fft.fftshift(np.fft.fft2(np.asarray(img, dtype=np.float64)))


def ifft2c(F) -> np.ndarray:
    """Αντίστροφος του fft2c· επιστρέφει το πραγματικό μέρος (η εικόνα είναι real)."""
    return np.real(np.fft.ifft2(np.fft.ifftshift(F)))


def magnitude(F) -> np.ndarray:
    """Φάσμα πλάτους |F|."""
    return np.abs(F)


def power(F) -> np.ndarray:
    """Φάσμα ισχύος |F|²."""
    a = np.abs(F)
    return a * a


def log_spectrum_image(F) -> np.ndarray:
    """Φάσμα ως log(1+|F|) σε uint8· χωρίς τον λογάριθμο κυριαρχεί η DC."""
    s = np.log1p(magnitude(F))
    lo, hi = float(s.min()), float(s.max())
    if hi <= lo:
        return np.zeros(s.shape, dtype=np.uint8)
    return to_uint8((s - lo) / (hi - lo) * 255.0)


@lru_cache(maxsize=8)
def _radius_grid_cached(h: int, w: int) -> np.ndarray:
    """
    Ο πίνακας ακτίνων για δοσμένες διαστάσεις. Εξαρτάται μόνο από το (h, w),
    οπότε υπολογίζεται μία φορά και επαναχρησιμοποιείται: σε μια πλήρη εκτέλεση
    ζητείται εκατοντάδες φορές (φίλτρα, ακτινικά προφίλ, ενέργεια φάσματος).

    Ο πίνακας επιστρέφεται read-only, επειδή τον μοιράζονται όλοι οι καλούντες:
    έτσι μια μελλοντική κατά λάθος τροποποίηση χτυπάει αμέσως, αντί να αλλοιώσει
    σιωπηλά τα αποτελέσματα κάθε επόμενης κλήσης.
    """
    v = np.arange(h) - h // 2
    u = np.arange(w) - w // 2
    rr, cc = np.meshgrid(v, u, indexing="ij")
    r = np.sqrt(rr ** 2 + cc ** 2) / (min(h, w) / 2.0)
    r.flags.writeable = False
    return r


def radius_grid(shape) -> np.ndarray:
    """Απόσταση από το κέντρο (DC), κανονικοποιημένη: r = 1 στο Nyquist."""
    h, w = shape[:2]
    return _radius_grid_cached(int(h), int(w))


def radial_power_profile(F, nbins=128):
    """
    Μέση ισχύς ανά ακτινικό δακτύλιο -> (κέντρα δακτυλίων, μέση ισχύς). Μέση
    και όχι άθροισμα, γιατί οι μεγάλες ακτίνες έχουν περισσότερους συντελεστές.
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
    """Μηδενίζει τη DC: αλλιώς το 90% της ενέργειας πέφτει σε ακτίνα ~0.01."""
    q = np.array(p, dtype=np.float64, copy=True)
    h, w = q.shape[:2]
    q[h // 2, w // 2] = 0.0
    return q


def cumulative_energy(F, nbins=512, exclude_dc=True):
    """
    Ενέργεια του φάσματος εντός ακτίνας r -> (ακτίνες, κλάσμα στο [0,1]).
    Η DC αγνοείται εξ ορισμού (_drop_dc).
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
    """Η μικρότερη (κανονικοποιημένη) ακτίνα που περικλείει `fraction` της ενέργειας."""
    radii, cum = cumulative_energy(F, nbins=nbins, exclude_dc=exclude_dc)
    i = int(np.searchsorted(cum, fraction))
    return float(radii[min(i, len(radii) - 1)])


def high_freq_energy_ratio(F, r0=None, exclude_dc=True) -> float:
    """Ποσοστό (%) της ενέργειας (χωρίς DC) που βρίσκεται πέρα από την ακτίνα r0."""
    r0 = config.HF_RADIUS_FRAC if r0 is None else r0
    p = _drop_dc(power(F)) if exclude_dc else power(F)
    r = radius_grid(p.shape)
    return float(p[r > r0].sum() / p.sum() * 100.0)


def spectrum_report(img) -> dict:
    """Όλα τα ακτινικά μεγέθη μιας εικόνας μαζί."""
    F = fft2c(img)
    rep = {"hf_pct": high_freq_energy_ratio(F)}
    for lvl in config.ENERGY_LEVELS:
        rep[f"r{int(lvl * 100)}"] = radius_for_energy(F, lvl)
    return rep


def snr_crossover_radius(ref, noisy, nbins=128) -> float:
    """
    Η ακτίνα όπου η ισχύς του θορύβου ξεπερνά την ισχύ του σήματος (SNR = 1) —
    η περιοχή τιμών για τη συχνότητα αποκοπής του Σταδίου 5.
    Επιστρέφει NaN αν το σήμα υπερισχύει παντού.
    """
    r, p_sig = radial_power_profile(fft2c(ref), nbins=nbins)
    noise = np.asarray(noisy, dtype=np.float64) - np.asarray(ref, dtype=np.float64)
    _, p_noise = radial_power_profile(fft2c(noise), nbins=nbins)
    above = np.nonzero(p_noise > p_sig)[0]
    return float(r[above[0]]) if above.size else float("nan")


# ΣΤΑΔΙΟ 5 — Χαμηλοπερατά φίλτρα
#
# Κάθε φίλτρο ορίζεται ως συνάρτηση μεταφοράς H(u,v) που εξαρτάται μόνο από την
# ακτίνα D. Το D0 δίνεται στην ίδια κανονικοποιημένη κλίμακα με το Στάδιο 3
# (1 = Nyquist).

def lowpass_ideal(shape, d0) -> np.ndarray:
    """Ιδανικό χαμηλοπερατό: H = 1 για D ≤ D0, αλλιώς 0."""
    return (radius_grid(shape) <= float(d0)).astype(np.float64)


def lowpass_butterworth(shape, d0, order=None) -> np.ndarray:
    """Butterworth: H = 1 / (1 + (D/D0)^(2n)). Στο D0 δίνει πάντα H = 0.5."""
    n = config.BUTTERWORTH_ORDER if order is None else order
    d = radius_grid(shape) / float(d0)
    return 1.0 / (1.0 + d ** (2 * n))


def lowpass_gaussian(shape, d0) -> np.ndarray:
    """Γκαουσιανό: H = exp(-D²/(2·D0²))."""
    d = radius_grid(shape)
    return np.exp(-(d ** 2) / (2.0 * float(d0) ** 2))


def apply_transfer(img, H) -> np.ndarray:
    """Φιλτράρισμα στο πεδίο συχνοτήτων: G = H·F, μετά αντίστροφος DFT."""
    F = fft2c(img)
    return to_uint8(ifft2c(F * H))


def butterworth_key(n) -> str:
    """Το κλειδί οικογένειας του Butterworth τάξης n (π.χ. n=2 -> "butterworth_n2")."""
    return f"butterworth_n{n}"


def _make_butterworth(n):
    """
    Κατασκευαστής H(u,v) για σταθερή τάξη n. Ξεχωριστή συνάρτηση και όχι lambda
    μέσα στον βρόχο, ώστε το n να δεσμεύεται εδώ και όχι τη στιγμή της κλήσης.
    """
    def builder(shape, d0):
        return lowpass_butterworth(shape, d0, order=n)
    builder.__name__ = f"lowpass_butterworth_n{n}"
    builder.order = n
    return builder


# Κάθε τάξη του Butterworth είναι ξεχωριστή οικογένεια φίλτρων: δικός της
# πυρήνας, δική της σάρωση συχνότητας αποκοπής, δικά της PSNR/SSIM.
BUTTERWORTH_KEYS = tuple(butterworth_key(n) for n in config.BUTTERWORTH_ORDERS)

LPF_BUILDERS = {"ideal": lowpass_ideal}
LPF_BUILDERS.update({butterworth_key(n): _make_butterworth(n)
                     for n in config.BUTTERWORTH_ORDERS})
LPF_BUILDERS["gaussian"] = lowpass_gaussian

LPF_LABEL = {"ideal": "Ιδανικό ΧΠΦ"}
LPF_LABEL.update({butterworth_key(n): f"Butterworth n={n}"
                  for n in config.BUTTERWORTH_ORDERS})
LPF_LABEL["gaussian"] = "Γκαουσιανό ΧΠΦ"

# Η τάξη κάθε οικογένειας (None για όσες δεν είναι Butterworth) — στήλη πίνακα.
LPF_ORDER = {k: getattr(b, "order", None) for k, b in LPF_BUILDERS.items()}


def lowpass_many(img, family, cutoffs) -> list:
    """
    Το ίδιο φίλτρο για πολλές συχνότητες αποκοπής, με ΕΝΑΝ μόνο ευθύ DFT.

    Ο ευθύς μετασχηματισμός F δεν εξαρτάται από το D0: αλλάζει μόνο η συνάρτηση
    μεταφοράς H. Η σάρωση των Σταδίων 5 και 8 δοκιμάζει 9 συχνότητες αποκοπής
    ανά οικογένεια, οπότε ο υπολογισμός του F μία φορά αντί για εννιά γλιτώνει
    8 στους 9 ευθείς μετασχηματισμούς, με αριθμητικά ταυτόσημο αποτέλεσμα.
    """
    F = fft2c(img)
    builder = LPF_BUILDERS[family]
    shape = np.asarray(img).shape
    return [to_uint8(ifft2c(F * builder(shape, d0))) for d0 in cutoffs]


def lowpass(img, family, d0) -> np.ndarray:
    """Εφαρμογή χαμηλοπερατού φίλτρου της δοσμένης οικογένειας."""
    return lowpass_many(img, family, (d0,))[0]


def radial_transfer_profile(H, nbins=128):
    """Το προφίλ H(D) ως προς την ακτίνα — για το γράφημα των αποκρίσεων."""
    r = radius_grid(H.shape)
    rmax = float(r.max())
    idx = np.clip((r / rmax * nbins).astype(int), 0, nbins - 1)
    tot = np.bincount(idx.ravel(), weights=H.ravel(), minlength=nbins)
    cnt = np.bincount(idx.ravel(), minlength=nbins)
    prof = np.divide(tot, cnt, out=np.zeros(nbins), where=cnt > 0)
    return (np.arange(nbins) + 0.5) / nbins * rmax, prof
