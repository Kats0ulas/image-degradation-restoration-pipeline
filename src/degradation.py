"""
ΣΤΑΔΙΟ 2 — Τεχνητή υποβάθμιση των εικόνων αναφοράς.

Δύο μοντέλα θορύβου:
  • Προσθετικός Gaussian:  g(x,y) = f(x,y) + n(x,y),  n ~ N(0, σ²)
  • Κρουστικός Salt-and-Pepper: ποσοστό των pixel γίνεται 0 ή 255

Κάθε παραλλαγή παίρνει δική της γεννήτρια τυχαίων αριθμών, σπαρμένη από το
config.RANDOM_SEED και το όνομα της παραλλαγής, ώστε τα αποτελέσματα να μην
εξαρτώνται από τη σειρά εκτέλεσης των σταδίων.
"""
import numpy as np

from . import config
from .io_utils import to_uint8


def rng_for(*tokens) -> np.random.Generator:
    """Γεννήτρια με seed από το RANDOM_SEED και το κείμενο των tokens."""
    label = "|".join(str(t) for t in tokens)
    extra = int.from_bytes(label.encode("utf-8"), "little") % (2 ** 32)
    return np.random.default_rng([config.RANDOM_SEED, extra])


def add_gaussian_noise(img, sigma, mean=0.0, rng=None) -> np.ndarray:
    """Προσθετικός Gaussian θόρυβος με τυπική απόκλιση `sigma` σε επίπεδα γκρι."""
    rng = rng or np.random.default_rng(config.RANDOM_SEED)
    f = np.asarray(img, dtype=np.float64)
    noise = rng.normal(loc=mean, scale=float(sigma), size=f.shape)
    return to_uint8(f + noise)


def add_salt_pepper(img, density, salt_ratio=0.5, rng=None) -> np.ndarray:
    """Κρουστικός θόρυβος: `density` των pixel γίνονται 255 ή 0 (κατά `salt_ratio`)."""
    rng = rng or np.random.default_rng(config.RANDOM_SEED)
    out = np.asarray(img, dtype=np.uint8).copy()
    r = rng.random(out.shape)
    salt = r < density * salt_ratio
    pepper = (r >= density * salt_ratio) & (r < density)
    out[salt] = 255
    out[pepper] = 0
    return out


def variant_specs():
    """Οι υποβαθμίσεις που εφαρμόζονται σε κάθε εικόνα αναφοράς."""
    specs = []
    for s in config.GAUSS_SIGMAS:
        specs.append({
            "key": f"gauss_s{s}",
            "kind": "gaussian",
            "param": float(s),
            "label": f"Gaussian θόρυβος (σ={s})",
            "short": f"Gauss σ={s}",
        })
    for d in config.SALT_PEPPER_DENSITIES:
        specs.append({
            "key": f"sp_d{int(round(d * 100)):02d}",
            "kind": "salt_pepper",
            "param": float(d),
            "label": f"Salt & Pepper (d={d:.0%})",
            "short": f"S&P d={d:.0%}",
        })
    return specs


def apply_variant(img, spec, image_key="") -> np.ndarray:
    """Εφαρμόζει μία παραλλαγή υποβάθμισης με αναπαραγώγιμη τυχαιότητα."""
    rng = rng_for(image_key, spec["key"])
    if spec["kind"] == "gaussian":
        return add_gaussian_noise(img, spec["param"], rng=rng)
    if spec["kind"] == "salt_pepper":
        return add_salt_pepper(img, spec["param"], rng=rng)
    raise ValueError(f"Άγνωστος τύπος θορύβου: {spec['kind']}")


def degrade_all(img, image_key="") -> list:
    """[(spec, noisy_image), ...] για όλες τις παραλλαγές του σταδίου."""
    return [(s, apply_variant(img, s, image_key)) for s in variant_specs()]


def noise_profile(ref, noisy) -> dict:
    """
    Μετρήσεις πάνω στη διαφορά noisy - ref:
      changed_pct : ποσοστό pixel που άλλαξαν τιμή
      sigma_hat   : τυπική απόκλιση της διαφοράς
      max_abs     : μέγιστη απόλυτη μεταβολή
    """
    d = np.asarray(noisy, dtype=np.float64) - np.asarray(ref, dtype=np.float64)
    return {
        "changed_pct": float((d != 0).mean() * 100.0),
        "sigma_hat": float(d.std()),
        "max_abs": float(np.abs(d).max()),
    }
