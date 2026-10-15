"""ΣΤΑΔΙΟ 3 — Ανάλυση στο πεδίο συχνοτήτων (FFT)"""
import numpy as np

from .. import config, degradation, io_utils, tables, viz
from .. import frequency_filters as freq


def run():
    print("\n=== ΣΤΑΔΙΟ 3: Ανάλυση πεδίου συχνοτήτων ===")
    out = io_utils.ensure_dir(config.STAGE_DIRS[3])

    refs = io_utils.load_all_references()
    rows = []

    for key, (spec, ref) in refs.items():
        print(f"\n  [{spec['label']}]")
        # Η αναφορά και οι υποβαθμισμένες εκδοχές της, στην ίδια λίστα.
        versions = [("Αναφορά", "ref", ref)]
        versions += [(v["short"], v["key"], n)
                     for v, n in degradation.degrade_all(ref, image_key=key)]

        spec_items, noise_spec_items = [], []
        prof_curves, cum_curves = [], []

        for name, vkey, img in versions:
            F = freq.fft2c(img)

            simg = freq.log_spectrum_image(F)
            io_utils.save_image(simg, out / f"{key}_spectrum_{vkey}.png")
            spec_items.append((simg, name))

            # Ακτινικά μεγέθη και ακτίνα SNR=1 (χρησιμοποιείται στο Στάδιο 5)
            rep = freq.spectrum_report(img)
            rcross = "—" if name == "Αναφορά" else f"{freq.snr_crossover_radius(ref, img):.3f}"
            rows.append([spec["label"], name, rep["hf_pct"],
                         rep["r90"], rep["r95"], rep["r99"], rcross])
            print(f"    {name:<12s} ενέργεια σε r>{config.HF_RADIUS_FRAC:.2f}: "
                  f"{rep['hf_pct']:6.2f}%   r90={rep['r90']:.3f}  "
                  f"r95={rep['r95']:.3f}  r99={rep['r99']:.3f}  r(SNR=1)={rcross}")

            r, p = freq.radial_power_profile(F)
            prof_curves.append((r, p, name))
            rc, cum = freq.cumulative_energy(F)
            cum_curves.append((rc, cum * 100.0, name))

            if name != "Αναφορά":
                d = img.astype(np.float64) - ref.astype(np.float64)
                noise_spec_items.append((freq.log_spectrum_image(freq.fft2c(d)),
                                         f"φάσμα θορύβου — {name}"))

        viz.plot_grid(spec_items, out / f"{key}_spectra.png", ncols=len(spec_items),
                      suptitle=f"Φάσμα πλάτους log(1+|F|) — {spec['label']}")
        viz.plot_grid(noise_spec_items, out / f"{key}_noise_spectra.png",
                      ncols=len(noise_spec_items),
                      suptitle=f"Φάσμα του θορύβου (g−f) — {spec['label']}")
        viz.plot_curves(prof_curves, out / f"{key}_radial_power.png",
                        xlabel="Κανονικοποιημένη ακτινική συχνότητα (1 = Nyquist)",
                        ylabel="Μέση ισχύς |F|² ανά δακτύλιο",
                        title=f"Ακτινικό προφίλ ισχύος — {spec['label']}",
                        logy=True, xlim=(0, 1.0))
        viz.plot_curves(cum_curves, out / f"{key}_cumulative_energy.png",
                        xlabel="Κανονικοποιημένη ακτινική συχνότητα (1 = Nyquist)",
                        ylabel="Αθροιστική ενέργεια (%) — χωρίς DC",
                        title=f"Συγκέντρωση ενέργειας ανά συχνότητα — {spec['label']}",
                        xlim=(0, 1.0),
                        hlines=[(l * 100.0, f"{l:.0%}") for l in config.ENERGY_LEVELS])

    headers = ["Εικόνα", "Εκδοχή", f"Ενέργεια σε r>{config.HF_RADIUS_FRAC:.2f} (%)",
               "r(90% ενέργειας)", "r(95%)", "r(99%)", "r(SNR=1)"]
    tables.save(headers, rows, out / "spectrum_metrics",
                title="Κατανομή ενέργειας στο πεδίο συχνοτήτων (χωρίς DC, "
                      "ακτίνες κανονικοποιημένες στο Nyquist)")
    print("\n" + tables.render(headers, rows))
    print(f"\n  → αποτελέσματα: {out}")
