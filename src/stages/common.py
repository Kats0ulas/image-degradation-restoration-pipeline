"""Helper functions shared by more than one stage."""
from .. import metrics
from .. import spatial_filters as spatial


def best_spatial_restoration(ref, noisy):
    """The best Stage 4 restoration for the given degradation."""
    best = None
    for fspec, filt in spatial.apply_all(noisy):
        m = metrics.evaluate(ref, filt)
        if best is None or m["psnr"] > best[2]["psnr"]:
            best = (fspec, filt, m)
    return best
