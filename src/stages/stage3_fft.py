"""STAGE 3 — Frequency-domain analysis (FFT)"""
import numpy as np

from .. import config, degradation, io_utils, tables, viz
from .. import frequency_filters as freq


def run():
    print("\n=== STAGE 3: Frequency-domain analysis ===")
    out = io_utils.ensure_dir(config.STAGE_DIRS[3])

    refs = io_utils.load_all_references()
    rows = []

    for key, (spec, ref) in refs.items():
        print(f"\n  [{spec['label']}]")
        # The reference and its degraded versions, in the same list.
        versions = [("Reference", "ref", ref)]
        versions += [(v["short"], v["key"], n)
                     for v, n in degradation.degrade_all(ref, image_key=key)]

        spec_items, noise_spec_items = [], []
        prof_curves, cum_curves = [], []

        for name, vkey, img in versions:
            F = freq.fft2c(img)

            simg = freq.log_spectrum_image(F)
            io_utils.save_image(simg, out / f"{key}_spectrum_{vkey}.png")
            spec_items.append((simg, name))

            # Radial quantities and the SNR=1 radius (used in Stage 5)
            rep = freq.spectrum_report(img)
            rcross = "—" if name == "Reference" else f"{freq.snr_crossover_radius(ref, img):.3f}"
            rows.append([spec["label"], name, rep["hf_pct"],
                         rep["r90"], rep["r95"], rep["r99"], rcross])
            print(f"    {name:<12s} energy at r>{config.HF_RADIUS_FRAC:.2f}: "
                  f"{rep['hf_pct']:6.2f}%   r90={rep['r90']:.3f}  "
                  f"r95={rep['r95']:.3f}  r99={rep['r99']:.3f}  r(SNR=1)={rcross}")

            r, p = freq.radial_power_profile(F)
            prof_curves.append((r, p, name))
            rc, cum = freq.cumulative_energy(F)
            cum_curves.append((rc, cum * 100.0, name))

            if name != "Reference":
                d = img.astype(np.float64) - ref.astype(np.float64)
                noise_spec_items.append((freq.log_spectrum_image(freq.fft2c(d)),
                                         f"noise spectrum — {name}"))

        viz.plot_grid(spec_items, out / f"{key}_spectra.png", ncols=len(spec_items),
                      suptitle=f"Magnitude spectrum log(1+|F|) — {spec['label']}")
        viz.plot_grid(noise_spec_items, out / f"{key}_noise_spectra.png",
                      ncols=len(noise_spec_items),
                      suptitle=f"Spectrum of the noise (g−f) — {spec['label']}")
        viz.plot_curves(prof_curves, out / f"{key}_radial_power.png",
                        xlabel="Normalized radial frequency (1 = Nyquist)",
                        ylabel="Mean power |F|² per ring",
                        title=f"Radial power profile — {spec['label']}",
                        logy=True, xlim=(0, 1.0))
        viz.plot_curves(cum_curves, out / f"{key}_cumulative_energy.png",
                        xlabel="Normalized radial frequency (1 = Nyquist)",
                        ylabel="Cumulative energy (%) — without DC",
                        title=f"Energy concentration over frequency — {spec['label']}",
                        xlim=(0, 1.0),
                        hlines=[(l * 100.0, f"{l:.0%}") for l in config.ENERGY_LEVELS])

    headers = ["Image", "Version", f"Energy at r>{config.HF_RADIUS_FRAC:.2f} (%)",
               "r(90% energy)", "r(95%)", "r(99%)", "r(SNR=1)"]
    tables.save(headers, rows, out / "spectrum_metrics",
                title="Energy distribution in the frequency domain (without DC, "
                      "radii normalized to Nyquist)")
    print("\n" + tables.render(headers, rows))
    print(f"\n  → results: {out}")
