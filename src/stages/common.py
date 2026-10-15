"""Βοηθητικές συναρτήσεις που μοιράζονται περισσότερα από ένα στάδια."""
from .. import metrics
from .. import spatial_filters as spatial


def best_spatial_restoration(ref, noisy):
    """Η καλύτερη αποκατάσταση του Σταδίου 4 για τη συγκεκριμένη υποβάθμιση."""
    best = None
    for fspec, filt in spatial.apply_all(noisy):
        m = metrics.evaluate(ref, filt)
        if best is None or m["psnr"] > best[2]["psnr"]:
            best = (fspec, filt, m)
    return best
