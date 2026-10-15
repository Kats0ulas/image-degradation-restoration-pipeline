"""
STAGE 2 — Artificial degradation of the reference images.

Two noise models:
  • Additive Gaussian:  g(x,y) = f(x,y) + n(x,y),  n ~ N(0, σ²)
  • Impulse Salt-and-Pepper: a fraction of the pixels becomes 0 or 255

Each variant gets its own random number generator, seeded from
config.RANDOM_SEED and the variant name, so that the results do not depend on
the order in which the stages run.
"""
import numpy as np

from . import config
from .io_utils import to_uint8


def rng_for(*tokens) -> np.random.Generator:
    """Generator seeded from RANDOM_SEED and the text of the tokens."""
    label = "|".join(str(t) for t in tokens)
    extra = int.from_bytes(label.encode("utf-8"), "little") % (2 ** 32)
    return np.random.default_rng([config.RANDOM_SEED, extra])


def add_gaussian_noise(img, sigma, mean=0.0, rng=None) -> np.ndarray:
    """Additive Gaussian noise with standard deviation `sigma` in gray levels."""
    rng = rng or np.random.default_rng(config.RANDOM_SEED)
    f = np.asarray(img, dtype=np.float64)
    noise = rng.normal(loc=mean, scale=float(sigma), size=f.shape)
    return to_uint8(f + noise)


def add_salt_pepper(img, density, salt_ratio=0.5, rng=None) -> np.ndarray:
    """Impulse noise: a `density` fraction of pixels become 255 or 0 (per `salt_ratio`)."""
    rng = rng or np.random.default_rng(config.RANDOM_SEED)
    out = np.asarray(img, dtype=np.uint8).copy()
    r = rng.random(out.shape)
    salt = r < density * salt_ratio
    pepper = (r >= density * salt_ratio) & (r < density)
    out[salt] = 255
    out[pepper] = 0
    return out


def variant_specs():
    """The degradations applied to each reference image."""
    specs = []
    for s in config.GAUSS_SIGMAS:
        specs.append({
            "key": f"gauss_s{s}",
            "kind": "gaussian",
            "param": float(s),
            "label": f"Gaussian noise (σ={s})",
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
    """Applies one degradation variant with reproducible randomness."""
    rng = rng_for(image_key, spec["key"])
    if spec["kind"] == "gaussian":
        return add_gaussian_noise(img, spec["param"], rng=rng)
    if spec["kind"] == "salt_pepper":
        return add_salt_pepper(img, spec["param"], rng=rng)
    raise ValueError(f"Unknown noise type: {spec['kind']}")


def degrade_all(img, image_key="") -> list:
    """[(spec, noisy_image), ...] for all variants of the stage."""
    return [(s, apply_variant(img, s, image_key)) for s in variant_specs()]


def noise_profile(ref, noisy) -> dict:
    """
    Measurements on the difference noisy - ref:
      changed_pct : percentage of pixels whose value changed
      sigma_hat   : standard deviation of the difference
      max_abs     : maximum absolute change
    """
    d = np.asarray(noisy, dtype=np.float64) - np.asarray(ref, dtype=np.float64)
    return {
        "changed_pct": float((d != 0).mean() * 100.0),
        "sigma_hat": float(d.std()),
        "max_abs": float(np.abs(d).max()),
    }
