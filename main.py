"""
Τελική Εργασία ΕΥ631 — Ψηφιακή Επεξεργασία Εικόνας
Αλυσίδα Υποβάθμισης → Αποκατάστασης → Βελτίωσης ψηφιακής εικόνας.

Εκτέλεση:
    python main.py            # όλα τα στάδια
    python main.py --stage 1  # μόνο ένα στάδιο
    python main.py --stage 1 2

Κάθε στάδιο γράφει τα αποτελέσματά του στο results/stageN_.../
"""
import argparse

import cv2
import numpy as np

from src import config, degradation, edges, enhancement, io_utils, metrics, tables, viz
from src import frequency_filters as freq
from src import spatial_filters as spatial


# ΣΤΑΔΙΟ 1 — Επιλογή και προετοιμασία εικόνων αναφοράς (ground truth)
def image_stats(img):
    """
    Περιγραφικά μέτρα κάθε εικόνας αναφοράς:
      mean/std : θέση και άπλωμα του ιστογράμματος (αντίθεση)
      mean|∇|  : μέση ένταση κλίσης (Sobel)
      edge %   : ποσοστό pixel που ο Canny χαρακτηρίζει ακμή
    """
    f = img.astype(np.float64)
    gx = cv2.Sobel(f, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    edges = cv2.Canny(img, 100, 200)
    return {
        "mean": float(f.mean()),
        "std": float(f.std()),
        "min": int(img.min()),
        "max": int(img.max()),
        "mean_grad": float(grad.mean()),
        "edge_pct": float((edges > 0).mean() * 100.0),
    }


def run_stage1():
    print("\n=== ΣΤΑΔΙΟ 1: Εικόνες αναφοράς ===")
    out = io_utils.ensure_dir(config.STAGE_DIRS[1])
    io_utils.ensure_dir(config.IMAGES_DIR)

    refs = io_utils.load_all_references()
    rows = []
    grid_items = []

    for key, (spec, img) in refs.items():
        io_utils.save_image(img, config.IMAGES_DIR / f"{key}.png")
        io_utils.save_image(img, out / f"ref_{key}.png")

        viz.plot_image_with_hist(img, spec["label"], out / f"ref_{key}_hist.png")

        s = image_stats(img)
        rows.append([spec["label"], f"{img.shape[1]}x{img.shape[0]}",
                     s["mean"], s["std"], f"{s['min']}–{s['max']}",
                     s["mean_grad"], s["edge_pct"]])
        grid_items.append((img, spec["label"]))
        print(f"  {key:10s} {img.shape}  mean={s['mean']:.1f} std={s['std']:.1f} "
              f"mean|grad|={s['mean_grad']:.1f} edges={s['edge_pct']:.2f}%")

    viz.plot_grid(grid_items, out / "references.png", ncols=len(grid_items),
                  suptitle="Εικόνες αναφοράς (ground truth)")
    viz.plot_hist_row(grid_items, out / "references_hist.png",
                      suptitle="Ιστογράμματα εικόνων αναφοράς")

    headers = ["Εικόνα", "Διαστάσεις", "Μέση τιμή", "Τυπ. απόκλιση",
               "Εύρος", "Μέσο |∇| (Sobel)", "Ακμές Canny (%)"]
    tables.save(headers, rows, out / "reference_stats", title="Χαρακτηριστικά εικόνων αναφοράς")
    print("\n" + tables.render(headers, rows))
    print(f"\n  → αποτελέσματα: {out}")


# ΣΤΑΔΙΟ 2 — Τεχνητή υποβάθμιση (Gaussian & Salt-and-Pepper θόρυβος)
def run_stage2():
    print("\n=== ΣΤΑΔΙΟ 2: Υποβάθμιση εικόνων ===")
    out = io_utils.ensure_dir(config.STAGE_DIRS[2])

    refs = io_utils.load_all_references()
    rows = []

    for key, (spec, ref) in refs.items():
        print(f"\n  [{spec['label']}]")
        grid_items = [(ref, "Αναφορά (χωρίς θόρυβο)")]
        hist_items = [(ref, "Αναφορά")]
        noise_items = []

        for vspec, noisy in degradation.degrade_all(ref, image_key=key):
            # Η υποβαθμισμένη εικόνα — είσοδος για τα Στάδια 3–5
            io_utils.save_image(noisy, out / f"{key}_{vspec['key']}.png")

            prof = degradation.noise_profile(ref, noisy)
            m = metrics.evaluate(ref, noisy)
            rows.append([spec["label"], vspec["short"], prof["changed_pct"],
                         prof["sigma_hat"], prof["max_abs"],
                         m["mse"], m["psnr"], m["ssim"]])
            print(f"    {vspec['short']:<12s} pixels που άλλαξαν={prof['changed_pct']:6.2f}%  "
                  f"σ̂={prof['sigma_hat']:6.2f}  PSNR={m['psnr']:6.2f} dB  SSIM={m['ssim']:.4f}")

            title = f"{vspec['short']}\nPSNR={m['psnr']:.2f} dB · SSIM={m['ssim']:.3f}"
            grid_items.append((noisy, title))
            hist_items.append((noisy, vspec["short"]))

            # Χάρτης θορύβου: |g - f| ενισχυμένος ×4
            diff = np.abs(noisy.astype(np.float64) - ref.astype(np.float64))
            noise_items.append((io_utils.to_uint8(diff * 4.0),
                                f"|θόρυβος| ×4 — {vspec['short']}"))

        viz.plot_grid(grid_items, out / f"{key}_grid.png", ncols=len(grid_items),
                      suptitle=f"Υποβάθμιση — {spec['label']}")
        viz.plot_hist_row(hist_items, out / f"{key}_hist.png",
                          suptitle=f"Ιστογράμματα μετά την υποβάθμιση — {spec['label']}")
        viz.plot_grid(noise_items, out / f"{key}_noise_maps.png", ncols=len(noise_items),
                      suptitle=f"Χάρτες θορύβου — {spec['label']}")

    headers = ["Εικόνα", "Υποβάθμιση", "Pixel που άλλαξαν (%)", "σ̂ θορύβου",
               "max|Δ|", "MSE", "PSNR (dB)", "SSIM"]
    tables.save(headers, rows, out / "degradation_metrics",
                title="Ποσοτικοποίηση της υποβάθμισης ως προς την εικόνα αναφοράς")
    print("\n" + tables.render(headers, rows))
    print(f"\n  → αποτελέσματα: {out}")


# ΣΤΑΔΙΟ 3 — Ανάλυση στο πεδίο συχνοτήτων (FFT)
def run_stage3():
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


# ΣΤΑΔΙΟ 4 — Αποθορυβοποίηση στο χωρικό πεδίο
def run_stage4():
    print("\n=== ΣΤΑΔΙΟ 4: Αποθορυβοποίηση στο χωρικό πεδίο ===")
    out = io_utils.ensure_dir(config.STAGE_DIRS[4])

    refs = io_utils.load_all_references()
    sweep_rows, best_rows = [], []

    for key, (spec, ref) in refs.items():
        ref_sharp = spatial.sharpness(ref)
        print(f"\n  [{spec['label']}]  ευκρίνεια αναφοράς (var∇²) = {ref_sharp:.1f}")

        for vspec, noisy in degradation.degrade_all(ref, image_key=key):
            base = metrics.evaluate(ref, noisy)
            print(f"\n    {vspec['short']}  (πριν: PSNR={base['psnr']:.2f} dB, "
                  f"SSIM={base['ssim']:.4f})")

            results = []          # (fspec, filtered, metrics, sharpness%)
            for fspec, filt in spatial.apply_all(noisy):
                m = metrics.evaluate(ref, filt)
                sh = spatial.sharpness(filt) / ref_sharp * 100.0
                results.append((fspec, filt, m, sh))
                sweep_rows.append([spec["label"], vspec["short"], fspec["short"],
                                   base["psnr"], m["psnr"], m["psnr"] - base["psnr"],
                                   m["ssim"], m["ssim"] - base["ssim"], sh])

            # Καλύτερη παράμετρος ανά οικογένεια φίλτρου (κριτήριο: PSNR),
            # ώστε κάθε οικογένεια να κρίνεται στο δικό της βέλτιστο.
            grid_items = [(ref, "Αναφορά"),
                          (noisy, f"Θορυβώδης — {vspec['short']}\n"
                                  f"PSNR={base['psnr']:.2f} dB · SSIM={base['ssim']:.3f}")]
            for family in ("gaussian", "median", "bilateral"):
                fam = [r for r in results if r[0]["family"] == family]
                bspec, bimg, bm, bsh = max(fam, key=lambda r: r[2]["psnr"])
                io_utils.save_image(bimg, out / f"{key}_{vspec['key']}_{bspec['key']}.png")
                best_rows.append([spec["label"], vspec["short"],
                                  spatial.FAMILY_LABEL[family], bspec["short"],
                                  base["psnr"], bm["psnr"], bm["psnr"] - base["psnr"],
                                  bm["ssim"], bm["ssim"] - base["ssim"], bsh])
                grid_items.append((bimg, f"{bspec['short']}\n"
                                         f"PSNR={bm['psnr']:.2f} dB · SSIM={bm['ssim']:.3f}\n"
                                         f"ευκρίνεια {bsh:.0f}%"))
                print(f"      {spatial.FAMILY_LABEL[family]:<28s} καλύτερο: {bspec['short']:<22s} "
                      f"PSNR={bm['psnr']:6.2f} ({bm['psnr'] - base['psnr']:+5.2f}) "
                      f"SSIM={bm['ssim']:.4f} ευκρίνεια={bsh:5.1f}%")

            viz.plot_grid(grid_items, out / f"{key}_{vspec['key']}_compare.png",
                          ncols=len(grid_items),
                          suptitle=f"Χωρικά φίλτρα — {spec['label']} — {vspec['short']}")

            curves = []
            for family in ("gaussian", "median"):
                fam = sorted((r for r in results if r[0]["family"] == family),
                             key=lambda r: r[0]["size"])
                curves.append(([r[0]["size"] for r in fam],
                               [r[2]["psnr"] for r in fam],
                               spatial.FAMILY_LABEL[family]))
            viz.plot_curves(curves, out / f"{key}_{vspec['key']}_psnr_vs_ksize.png",
                            xlabel="Μέγεθος πυρήνα k (k×k)", ylabel="PSNR (dB)",
                            title=f"Επίδραση μεγέθους πυρήνα — {spec['label']} — {vspec['short']}",
                            hlines=[(base["psnr"], "χωρίς φιλτράρισμα")])

    sweep_headers = ["Εικόνα", "Υποβάθμιση", "Φίλτρο", "PSNR πριν", "PSNR μετά",
                     "ΔPSNR", "SSIM μετά", "ΔSSIM", "Ευκρίνεια (% αναφοράς)"]
    tables.save(sweep_headers, sweep_rows, out / "spatial_sweep",
                title="Πλήρης σάρωση παραμέτρων των χωρικών φίλτρων")

    best_headers = ["Εικόνα", "Υποβάθμιση", "Οικογένεια", "Καλύτερη παράμετρος",
                    "PSNR πριν", "PSNR μετά", "ΔPSNR", "SSIM μετά", "ΔSSIM",
                    "Ευκρίνεια (% αναφοράς)"]
    tables.save(best_headers, best_rows, out / "spatial_best",
                title="Καλύτερη παράμετρος ανά οικογένεια φίλτρου (κριτήριο: PSNR)")
    print("\n" + tables.render(best_headers, best_rows))
    print(f"\n  → αποτελέσματα: {out}")


# ΣΤΑΔΙΟ 5 — Αποθορυβοποίηση στο πεδίο συχνοτήτων (χαμηλοπερατά φίλτρα)
def _lpf_gallery(out):
    """
    Οι συναρτήσεις μεταφοράς H(u,v) όλων των φίλτρων — μία εικόνα πυρήνα ανά
    φίλτρο (ξεχωριστή για κάθε τάξη n) — μαζί με τις ακτινικές αποκρίσεις τους.
    """
    shape = config.TARGET_SIZE[::-1]
    d0 = 0.20
    masks, curves = [], []
    for fam, builder in freq.LPF_BUILDERS.items():
        H = builder(shape, d0)
        masks.append((io_utils.to_uint8(H * 255.0), f"{freq.LPF_LABEL[fam]}  D0={d0}"))
        r, prof = freq.radial_transfer_profile(H)
        curves.append((r, prof, freq.LPF_LABEL[fam]))
    viz.plot_grid(masks, out / "lpf_transfer_masks.png",
                  ncols=min(len(masks), 3),
                  suptitle="Συναρτήσεις μεταφοράς χαμηλοπερατών φίλτρων")
    viz.plot_curves(curves, out / "lpf_transfer_profiles.png",
                    xlabel="Κανονικοποιημένη ακτινική συχνότητα D (1 = Nyquist)",
                    ylabel="H(D)", title="Ακτινική απόκριση των φίλτρων (D0 = 0.20)",
                    xlim=(0, 0.6), vlines=[(d0, "D0")])


def _ringing_demo(ref, label, out, key):
    """
    Φαινόμενο Gibbs: ίδια συχνότητα αποκοπής, όλο και πιο απότομη μετάβαση.
    Εφαρμόζεται στην καθαρή εικόνα, ώστε το ringing να μην μπερδεύεται με θόρυβο.
    """
    d0 = 0.08
    items = [(ref, "Αναφορά")]
    for n in config.BUTTERWORTH_ORDERS_DEMO:
        H = freq.lowpass_butterworth(ref.shape, d0, order=n)
        items.append((freq.apply_transfer(ref, H), f"Butterworth n={n}"))
    items.append((freq.lowpass(ref, "ideal", d0), "Ιδανικό (n→∞)"))
    viz.plot_grid(items, out / f"{key}_ringing_demo.png", ncols=len(items),
                  suptitle=f"Φαινόμενο Gibbs: όσο πιο απότομη η αποκοπή, τόσο πιο "
                           f"έντονο το ringing (D0={d0}) — {label}")


def run_stage5():
    print("\n=== ΣΤΑΔΙΟ 5: Αποθορυβοποίηση στο πεδίο συχνοτήτων ===")
    out = io_utils.ensure_dir(config.STAGE_DIRS[5])
    _lpf_gallery(out)

    refs = io_utils.load_all_references()
    sweep_rows, best_rows = [], []

    for key, (spec, ref) in refs.items():
        ref_sharp = spatial.sharpness(ref)
        _ringing_demo(ref, spec["label"], out, key)
        print(f"\n  [{spec['label']}]")

        for vspec, noisy in degradation.degrade_all(ref, image_key=key):
            base = metrics.evaluate(ref, noisy)
            # Η πρόβλεψη του Σταδίου 3 για τη συχνότητα αποκοπής.
            r_snr1 = freq.snr_crossover_radius(ref, noisy)
            print(f"\n    {vspec['short']}  (πριν: PSNR={base['psnr']:.2f} dB) "
                  f"— πρόβλεψη Σταδίου 3: r(SNR=1)={r_snr1:.3f}")

            results = {}
            for fam in freq.LPF_BUILDERS:
                res = []
                for d0 in config.LPF_CUTOFFS:
                    filt = freq.lowpass(noisy, fam, d0)
                    m = metrics.evaluate(ref, filt)
                    sh = spatial.sharpness(filt) / ref_sharp * 100.0
                    res.append((d0, filt, m, sh))
                    sweep_rows.append([spec["label"], vspec["short"], freq.LPF_LABEL[fam],
                                       freq.LPF_ORDER[fam], d0, base["psnr"], m["psnr"],
                                       m["psnr"] - base["psnr"], m["ssim"],
                                       m["ssim"] - base["ssim"], sh])
                results[fam] = res

            # Καλύτερη συχνότητα αποκοπής ανά οικογένεια
            grid_items = [(ref, "Αναφορά"),
                          (noisy, f"Θορυβώδης — {vspec['short']}\n"
                                  f"PSNR={base['psnr']:.2f} dB · SSIM={base['ssim']:.3f}")]
            for fam, res in results.items():
                d0, bimg, bm, bsh = max(res, key=lambda r: r[2]["psnr"])
                io_utils.save_image(bimg, out / f"{key}_{vspec['key']}_{fam}_d{d0:.2f}.png")
                best_rows.append([spec["label"], vspec["short"], freq.LPF_LABEL[fam],
                                  freq.LPF_ORDER[fam], d0, r_snr1, base["psnr"],
                                  bm["psnr"], bm["psnr"] - base["psnr"], bm["ssim"],
                                  bm["ssim"] - base["ssim"], bsh])
                grid_items.append((bimg, f"{freq.LPF_LABEL[fam]}  D0={d0}\n"
                                         f"PSNR={bm['psnr']:.2f} dB · SSIM={bm['ssim']:.3f}\n"
                                         f"ευκρίνεια {bsh:.0f}%"))
                print(f"      {freq.LPF_LABEL[fam]:<24s} βέλτιστο D0={d0:.2f}  "
                      f"PSNR={bm['psnr']:6.2f} ({bm['psnr'] - base['psnr']:+5.2f})  "
                      f"SSIM={bm['ssim']:.4f}  ευκρίνεια={bsh:5.1f}%")

            viz.plot_grid(grid_items, out / f"{key}_{vspec['key']}_compare.png",
                          ncols=min(len(grid_items), 4),
                          suptitle=f"Χαμηλοπερατά φίλτρα — {spec['label']} — {vspec['short']}")

            # PSNR συναρτήσει του D0, με σημειωμένο το r(SNR=1) του Σταδίου 3
            curves = [([d for d, _, _, _ in res], [m["psnr"] for _, _, m, _ in res],
                       freq.LPF_LABEL[fam]) for fam, res in results.items()]
            viz.plot_curves(curves, out / f"{key}_{vspec['key']}_psnr_vs_cutoff.png",
                            xlabel="Συχνότητα αποκοπής D0 (1 = Nyquist)",
                            ylabel="PSNR (dB)", markers=True,
                            title=f"Επιλογή συχνότητας αποκοπής — {spec['label']} — {vspec['short']}",
                            hlines=[(base["psnr"], "χωρίς φιλτράρισμα")],
                            vlines=[(r_snr1, "r(SNR=1) — Στάδιο 3")])

    sweep_headers = ["Εικόνα", "Υποβάθμιση", "Φίλτρο", "Τάξη n", "D0",
                     "PSNR πριν", "PSNR μετά", "ΔPSNR", "SSIM μετά", "ΔSSIM",
                     "Ευκρίνεια (% αναφοράς)"]
    tables.save(sweep_headers, sweep_rows, out / "frequency_sweep",
                title="Πλήρης σάρωση συχνοτήτων αποκοπής")

    best_headers = ["Εικόνα", "Υποβάθμιση", "Φίλτρο", "Τάξη n", "Βέλτιστο D0",
                    "r(SNR=1)", "PSNR πριν", "PSNR μετά", "ΔPSNR", "SSIM μετά",
                    "ΔSSIM", "Ευκρίνεια (% αναφοράς)"]
    tables.save(best_headers, best_rows, out / "frequency_best",
                title="Βέλτιστη συχνότητα αποκοπής ανά φίλτρο, δίπλα στην πρόβλεψη του Σταδίου 3")
    print("\n" + tables.render(best_headers, best_rows))
    print(f"\n  → αποτελέσματα: {out}")


# ΣΤΑΔΙΟ 6 — Βελτίωση εικόνας (ενίσχυση αντίθεσης)
def _best_spatial_restoration(ref, noisy):
    """Η καλύτερη αποκατάσταση του Σταδίου 4 για τη συγκεκριμένη υποβάθμιση."""
    best = None
    for fspec, filt in spatial.apply_all(noisy):
        m = metrics.evaluate(ref, filt)
        if best is None or m["psnr"] > best[2]["psnr"]:
            best = (fspec, filt, m)
    return best


def run_stage6():
    print("\n=== ΣΤΑΔΙΟ 6: Βελτίωση εικόνας ===")
    out = io_utils.ensure_dir(config.STAGE_DIRS[6])

    refs = io_utils.load_all_references()
    rows, chain_rows = [], []

    for key, (spec, ref) in refs.items():
        print(f"\n  [{spec['label']}]")

        # Μέρος Α: βελτίωση της καθαρής εικόνας
        base = enhancement.profile(ref)
        rows.append([spec["label"], "Αναφορά (χωρίς επεξεργασία)", base["mean"],
                     base["std"], base["entropy"], base["used_pct"], 1.0])
        print(f"    {'Αναφορά':<28s} std={base['std']:6.2f}  εντροπία={base['entropy']:.3f}  "
              f"επίπεδα σε χρήση={base['used_pct']:5.1f}%")

        grid_items = [(ref, "Αναφορά")]
        hist_items = [(ref, "Αναφορά")]
        curve_items = [(np.arange(256), enhancement.transfer_curve(ref),
                        "CDF αναφοράς → T(r) της HE")]

        for mspec in enhancement.method_specs():
            eimg = mspec["fn"](ref)
            io_utils.save_image(eimg, out / f"{key}_{mspec['key']}.png")
            p = enhancement.profile(eimg)
            ss = metrics.ssim(ref, eimg)
            rows.append([spec["label"], mspec["short"], p["mean"], p["std"],
                         p["entropy"], p["used_pct"], ss])
            print(f"    {mspec['short']:<28s} std={p['std']:6.2f}  εντροπία={p['entropy']:.3f}  "
                  f"επίπεδα σε χρήση={p['used_pct']:5.1f}%  SSIM={ss:.4f}")
            title = (f"{mspec['short']}\nstd={p['std']:.1f} · H={p['entropy']:.2f} bits\n"
                     f"επίπεδα {p['used_pct']:.0f}%")
            grid_items.append((eimg, title))
            if mspec["key"] in ("he", "clahe_c2_t8"):
                hist_items.append((eimg, mspec["short"]))

        viz.plot_grid(grid_items, out / f"{key}_methods.png", ncols=4,
                      figsize_per=(3.3, 4.1),
                      suptitle=f"Τεχνικές βελτίωσης — {spec['label']}")
        viz.plot_hist_row(hist_items, out / f"{key}_hist.png",
                          suptitle=f"Ιστογράμματα πριν/μετά τη βελτίωση — {spec['label']}")
        # Η συνάρτηση μετασχηματισμού s = T(r) της HE
        viz.plot_curves(curve_items + [(np.arange(256), np.arange(256), "ταυτοτική (καμία αλλαγή)")],
                        out / f"{key}_transfer.png",
                        xlabel="Είσοδος r (επίπεδο γκρι)", ylabel="Έξοδος s = T(r)",
                        title=f"Συνάρτηση μετασχηματισμού της ισοστάθμισης — {spec['label']}",
                        xlim=(0, 255))

        # Μέρος Β: η σειρά μέσα στην αλυσίδα
        # Ίδια βελτίωση, εφαρμοσμένη (α) στη θορυβώδη και (β) στην αποκατεστημένη.
        vspec = next(v for v in degradation.variant_specs()
                     if v["key"] == config.ENHANCE_CHAIN_VARIANT)
        noisy = degradation.apply_variant(ref, vspec, image_key=key)
        rspec, restored, rm = _best_spatial_restoration(ref, noisy)
        print(f"\n    Αλυσίδα με {vspec['short']} — καλύτερη αποκατάσταση: "
              f"{rspec['short']} (PSNR={rm['psnr']:.2f} dB)")

        chain = [("Θορυβώδης", noisy), (f"Αποκατεστημένη ({rspec['short']})", restored)]
        chain_items = [(ref, "Αναφορά")]
        for stage_name, img in chain:
            for mkey, mlabel in (("raw", "χωρίς βελτίωση"), ("he", "+ HE"),
                                 ("clahe", "+ CLAHE clip=2")):
                if mkey == "raw":
                    eimg = img
                elif mkey == "he":
                    eimg = enhancement.histogram_equalization(img)
                else:
                    eimg = enhancement.clahe(img, 2.0, (8, 8))
                p = enhancement.profile(eimg)
                m = metrics.evaluate(ref, eimg)
                sh = spatial.sharpness(eimg) / spatial.sharpness(ref) * 100.0
                chain_rows.append([spec["label"], stage_name, mlabel, p["std"],
                                   p["entropy"], m["psnr"], m["ssim"], sh])
                io_utils.save_image(
                    eimg, out / f"{key}_chain_{'noisy' if stage_name == 'Θορυβώδης' else 'restored'}_{mkey}.png")
                chain_items.append((eimg, f"{stage_name}\n{mlabel}\n"
                                          f"PSNR={m['psnr']:.2f} · SSIM={m['ssim']:.3f}"))
                print(f"      {stage_name:<32s} {mlabel:<16s} std={p['std']:6.2f}  "
                      f"PSNR={m['psnr']:6.2f}  SSIM={m['ssim']:.4f}  ευκρίνεια={sh:6.1f}%")

        viz.plot_grid(chain_items, out / f"{key}_chain.png", ncols=4,
                      figsize_per=(3.4, 4.3),
                      suptitle=f"Σειρά στην αλυσίδα: βελτίωση ΠΡΙΝ vs ΜΕΤΑ την "
                               f"αποκατάσταση — {spec['label']} — {vspec['short']}")

    headers = ["Εικόνα", "Τεχνική", "Μέση τιμή", "Τυπ. απόκλιση (αντίθεση)",
               "Εντροπία (bits)", "Επίπεδα σε χρήση (%)", "SSIM ως προς αναφορά"]
    tables.save(headers, rows, out / "enhancement_metrics",
                title="Βελτίωση αντίθεσης στις καθαρές εικόνες αναφοράς")
    print("\n" + tables.render(headers, rows))

    chain_headers = ["Εικόνα", "Στάδιο αλυσίδας", "Βελτίωση", "Τυπ. απόκλιση",
                     "Εντροπία (bits)", "PSNR (dB)", "SSIM", "Ευκρίνεια (% αναφοράς)"]
    tables.save(chain_headers, chain_rows, out / "enhancement_chain",
                title="Επίδραση της σειράς: βελτίωση πριν vs μετά την αποκατάσταση")
    print("\n" + tables.render(chain_headers, chain_rows))
    print(f"\n  → αποτελέσματα: {out}")


# ΣΤΑΔΙΟ 7 — Ανίχνευση ακμών
def _canny_default():
    """Το μεσαίο ζεύγος κατωφλίων, που χρησιμοποιείται σε όλες τις συγκρίσεις."""
    return config.CANNY_THRESHOLDS[len(config.CANNY_THRESHOLDS) // 2]


def _abs_norm(a):
    """Απόλυτη τιμή πίνακα, κανονικοποιημένη σε uint8 για προβολή."""
    m = max(float(np.abs(a).max()), 1.0)
    return io_utils.to_uint8(np.abs(a) / m * 255.0)


def _sobel_panel(img, img_label, path, suptitle):
    """
    Ο τελεστής Sobel στις τρεις μορφές του — οριζόντιος (Gx), κατακόρυφος (Gy)
    και συνδυασμένος |∇f| — συν τον δυαδικό χάρτη από κατωφλίωση του |∇f|.
    """
    gx, gy = edges.sobel_components(img)
    viz.plot_grid(
        [(img, img_label),
         (_abs_norm(gx), "|Gx| — οριζόντιος τελεστής\n(αποκρίνεται σε ΚΑΤΑΚΟΡΥΦΕΣ ακμές)"),
         (_abs_norm(gy), "|Gy| — κατακόρυφος τελεστής\n(αποκρίνεται σε ΟΡΙΖΟΝΤΙΕΣ ακμές)"),
         (edges.sobel_magnitude(img), "Συνδυασμένος: |∇f| = √(Gx²+Gy²)"),
         (edges.sobel_edges(img),
          f"Συνδυασμένος, κατωφλιωμένος\n(ισχυρότερο "
          f"{100 - config.SOBEL_EDGE_PERCENTILE:.0f}% των pixel)")],
        path, ncols=3, suptitle=suptitle)


def _threshold_study(img, ref_canny_maps, ref_sobel, lead_item, path, suptitle,
                     rows, prefix):
    """
    Σάρωση των κατωφλίων υστέρησης του Canny με precision / recall / F1. Κάθε
    ρύθμιση κρίνεται ως προς τον ίδιο ανιχνευτή πάνω στην εικόνα αναφοράς.
    """
    items = [lead_item]
    for (tlo, thi), rmap in ref_canny_maps.items():
        e = edges.canny(img, tlo, thi)
        em = edges.edge_metrics(rmap, e)
        rows.append(prefix + [f"Canny ({tlo}, {thi})", em["edge_pct"],
                              em["precision"], em["recall"], em["f1"]])
        items.append((e, f"Canny ({tlo}, {thi})\nακμές {em['edge_pct']:.2f}%\n"
                         f"P={em['precision']:.2f} R={em['recall']:.2f}\n"
                         f"F1={em['f1']:.3f}"))
    e = edges.sobel_edges(img)
    em = edges.edge_metrics(ref_sobel, e)
    rows.append(prefix + [f"Sobel (ισχυρότερο {100 - config.SOBEL_EDGE_PERCENTILE:.0f}%)",
                          em["edge_pct"], em["precision"], em["recall"], em["f1"]])
    items.append((e, f"Sobel κατωφλιωμένο\nακμές {em['edge_pct']:.2f}%\n"
                     f"P={em['precision']:.2f} R={em['recall']:.2f}\n"
                     f"F1={em['f1']:.3f}"))
    viz.plot_grid(items, path, ncols=len(items), figsize_per=(3.3, 4.1),
                  suptitle=suptitle)


def run_stage7():
    print("\n=== ΣΤΑΔΙΟ 7: Ανίχνευση ακμών ===")
    out = io_utils.ensure_dir(config.STAGE_DIRS[7])
    lo, hi = _canny_default()

    refs = io_utils.load_all_references()
    thr_rows, restored_rows, chain_rows = [], [], []

    for key, (spec, ref) in refs.items():
        print(f"\n  [{spec['label']}]")

        # Μέρος Α: οι τελεστές στην καθαρή εικόνα (ground truth)
        # Οι χάρτες ακμών της αναφοράς είναι το μέτρο σύγκρισης για όλα τα άλλα.
        ref_sobel = edges.sobel_edges(ref)
        ref_canny_maps = {(tlo, thi): edges.canny(ref, tlo, thi)
                          for tlo, thi in config.CANNY_THRESHOLDS}
        ref_canny = ref_canny_maps[(lo, hi)]
        ref_canny_auto = edges.canny_auto(ref)

        io_utils.save_image(edges.sobel_magnitude(ref), out / f"{key}_sobel_mag.png")
        io_utils.save_image(ref_sobel, out / f"{key}_sobel_edges.png")
        io_utils.save_image(ref_canny, out / f"{key}_canny_edges.png")

        _sobel_panel(ref, "Αναφορά", out / f"{key}_operators.png",
                     f"Τελεστής Sobel: οριζόντιος, κατακόρυφος και συνδυασμένος — "
                     f"{spec['label']}")

        # Κατώφλια υστέρησης στην καθαρή εικόνα. Εδώ δεν έχει νόημα
        # precision/recall (η εικόνα είναι η αναφορά), οπότε μετράμε το ποσοστό
        # ακμών.
        thr_items = [(ref, "Αναφορά")]
        for tlo, thi in config.CANNY_THRESHOLDS:
            e = ref_canny_maps[(tlo, thi)]
            pct = float((e > 0).mean() * 100.0)
            thr_rows.append([spec["label"], f"Canny ({tlo}, {thi})", pct])
            thr_items.append((e, f"Canny ({tlo}, {thi})\nακμές {pct:.2f}%"))
        thr_rows.append([spec["label"],
                         f"Sobel (ισχυρότερο {100 - config.SOBEL_EDGE_PERCENTILE:.0f}%)",
                         float((ref_sobel > 0).mean() * 100.0)])
        thr_items.append((ref_sobel, f"Sobel κατωφλιωμένο\nακμές "
                                     f"{float((ref_sobel > 0).mean() * 100.0):.2f}%"))
        viz.plot_grid(thr_items, out / f"{key}_canny_thresholds.png", ncols=len(thr_items),
                      suptitle=f"Επίδραση των κατωφλίων υστέρησης στην καθαρή εικόνα — "
                               f"{spec['label']}")

        # Μέρος Β: οι ίδιοι τελεστές στην αποκατεστημένη εικόνα
        # Αποκατεστημένη = έξοδος του καλύτερου χωρικού φίλτρου του Σταδίου 4.
        for vspec, noisy in degradation.degrade_all(ref, image_key=key):
            rspec, restored, rm = _best_spatial_restoration(ref, noisy)
            print(f"\n    {vspec['short']}  (αποκατάσταση: {rspec['short']}, "
                  f"PSNR={rm['psnr']:.2f} dB)")

            _sobel_panel(
                restored, f"Αποκατεστημένη\n{rspec['short']}",
                out / f"{key}_{vspec['key']}_restored_sobel.png",
                f"Sobel στην αποκατεστημένη εικόνα — {spec['label']} — "
                f"{vspec['short']} (αποκατάσταση: {rspec['short']})")

            _threshold_study(
                restored, ref_canny_maps, ref_sobel,
                (restored, f"Αποκατεστημένη\n{rspec['short']}\nPSNR={rm['psnr']:.2f} dB"),
                out / f"{key}_{vspec['key']}_restored_thresholds.png",
                f"Επιλογή κατωφλίων στην αποκατεστημένη εικόνα — {spec['label']} — "
                f"{vspec['short']}",
                restored_rows,
                [spec["label"], vspec["short"], rspec["short"]])

            for r in restored_rows[-(len(config.CANNY_THRESHOLDS) + 1):]:
                print(f"      {r[3]:<28s} ακμές={r[4]:5.2f}%  precision={r[5]:.3f}  "
                      f"recall={r[6]:.3f}  F1={r[7]:.3f}")

            # Μέρος Γ: ίδιοι ανιχνευτές σε θορυβώδη vs αποκατεστημένη
            grid_items = [(ref_canny, "Αναφορά — Canny")]
            for src_name, src_img in (("Θορυβώδης", noisy),
                                      ("Αποκατεστημένη", restored)):
                for det_name, det_fn, ref_map in (
                        ("Sobel", edges.sobel_edges, ref_sobel),
                        ("Canny", lambda im: edges.canny(im, lo, hi), ref_canny),
                        ("Canny προσαρμοστικό", edges.canny_auto, ref_canny_auto)):
                    emap = det_fn(src_img)
                    em = edges.edge_metrics(ref_map, emap)
                    chain_rows.append([spec["label"], vspec["short"], src_name,
                                       det_name, em["edge_pct"], em["precision"],
                                       em["recall"], em["f1"]])
                    if det_name == "Canny":
                        io_utils.save_image(
                            emap, out / f"{key}_{vspec['key']}_"
                                        f"{'noisy' if src_name == 'Θορυβώδης' else 'restored'}_canny.png")
                        grid_items.append((emap, f"{src_name} — Canny\nF1={em['f1']:.3f} "
                                                 f"(P={em['precision']:.2f}, R={em['recall']:.2f})"))

            viz.plot_grid(grid_items, out / f"{key}_{vspec['key']}_edges.png", ncols=3,
                          figsize_per=(3.3, 3.9),
                          suptitle=f"Ακμές πριν και μετά την αποκατάσταση — "
                                   f"{spec['label']} — {vspec['short']} "
                                   f"(αποκατάσταση: {rspec['short']})")

    thr_headers = ["Εικόνα", "Ανιχνευτής / κατώφλια", "Pixel που χαρακτηρίστηκαν ακμή (%)"]
    tables.save(thr_headers, thr_rows, out / "edge_thresholds",
                title="Επίδραση των κατωφλίων στην καθαρή εικόνα αναφοράς")
    print("\n" + tables.render(thr_headers, thr_rows))

    rest_headers = ["Εικόνα", "Υποβάθμιση", "Αποκατάσταση", "Ανιχνευτής / κατώφλια",
                    "Ακμές (%)", "Precision", "Recall", "F1"]
    tables.save(rest_headers, restored_rows, out / "edge_thresholds_restored",
                title="Ανίχνευση ακμών στην αποκατεστημένη εικόνα: επίδραση της "
                      "επιλογής παραμέτρων")
    print("\n" + tables.render(rest_headers, restored_rows))

    chain_headers = ["Εικόνα", "Υποβάθμιση", "Είσοδος", "Ανιχνευτής",
                     "Ακμές (%)", "Precision", "Recall", "F1"]
    tables.save(chain_headers, chain_rows, out / "edge_quality",
                title="Ποιότητα ανίχνευσης ακμών ως προς τις ακμές της εικόνας αναφοράς")
    print("\n" + tables.render(chain_headers, chain_rows))
    print(f"\n  → αποτελέσματα: {out}")


# ΣΤΑΔΙΟ 8 — Συγκεντρωτική αξιολόγηση όλης της αλυσίδας
def _all_restorations(ref, noisy):
    """
    Όλες οι μέθοδοι αποκατάστασης των Σταδίων 4 και 5 σε κοινή λίστα, ώστε τα
    δύο πεδία να συγκριθούν με τα ίδια κριτήρια.
    """
    res = []
    for fspec, filt in spatial.apply_all(noisy):
        res.append({"domain": "Χωρικό", "family": spatial.FAMILY_LABEL[fspec["family"]],
                    "name": fspec["short"], "image": filt})
    for fam in freq.LPF_BUILDERS:
        for d0 in config.LPF_CUTOFFS:
            res.append({"domain": "Συχνοτήτων", "family": freq.LPF_LABEL[fam],
                        "name": f"{freq.LPF_LABEL[fam]} D0={d0:g}",
                        "image": freq.lowpass(noisy, fam, d0)})
    for r in res:
        r.update(metrics.evaluate(ref, r["image"]))
    return res


def run_stage8():
    print("\n=== ΣΤΑΔΙΟ 8: Συγκεντρωτική αξιολόγηση ===")
    out = io_utils.ensure_dir(config.STAGE_DIRS[8])

    refs = io_utils.load_all_references()
    overall_rows, family_rows, chain_rows, extra_rows = [], [], [], []
    # Για το ραβδόγραμμα: PSNR του καλύτερου εκπροσώπου κάθε οικογένειας.
    case_labels, family_best = [], {}

    for key, (spec, ref) in refs.items():
        ref_sharp = spatial.sharpness(ref)
        ref_edges_map = edges.sobel_edges(ref)
        pipeline_items = []
        print(f"\n  [{spec['label']}]")

        for vspec, noisy in degradation.degrade_all(ref, image_key=key):
            base = metrics.evaluate(ref, noisy)
            res = _all_restorations(ref, noisy)
            case = f"{key[:4]}\n{vspec['short']}"
            case_labels.append(case)

            # Νικητής ανά οικογένεια και συνολικός νικητής
            fams = {}
            for r in res:
                b = fams.get(r["family"])
                if b is None or r["psnr"] > b["psnr"]:
                    fams[r["family"]] = r
            for fam_name, r in fams.items():
                family_best.setdefault(fam_name, {})[case] = r["psnr"]
                family_rows.append([spec["label"], vspec["short"], r["domain"], fam_name,
                                    r["name"], base["psnr"], r["psnr"],
                                    r["psnr"] - base["psnr"], r["ssim"]])

            winner = max(res, key=lambda r: r["psnr"])
            winner_ssim = max(res, key=lambda r: r["ssim"])
            best_spatial = max((r for r in res if r["domain"] == "Χωρικό"),
                               key=lambda r: r["psnr"])
            best_freq = max((r for r in res if r["domain"] == "Συχνοτήτων"),
                            key=lambda r: r["psnr"])

            overall_rows.append([
                spec["label"], vspec["short"], base["psnr"], base["ssim"],
                winner["name"], winner["psnr"], winner["psnr"] - base["psnr"],
                winner["ssim"], winner_ssim["name"], winner_ssim["ssim"],
                best_spatial["psnr"], best_freq["psnr"],
                best_spatial["psnr"] - best_freq["psnr"],
            ])
            print(f"    {vspec['short']:<12s} νικητής: {winner['name']:<26s} "
                  f"PSNR={winner['psnr']:6.2f} ({winner['psnr'] - base['psnr']:+6.2f})  "
                  f"| χωρικό {best_spatial['psnr']:6.2f} vs συχνοτήτων {best_freq['psnr']:6.2f} "
                  f"({best_spatial['psnr'] - best_freq['psnr']:+5.2f} dB)")

            # PSNR και SSIM σε κάθε βήμα της αλυσίδας: υποβαθμισμένη ->
            # καλύτερο χωρικό -> καλύτερο συχνοτικό -> μετά τη βελτίωση.
            restored = winner["image"]
            clip, tiles = config.FINAL_ENHANCEMENT
            enhanced = enhancement.clahe(restored, clip, tiles)
            em = metrics.evaluate(ref, enhanced)
            chain_rows.append([
                spec["label"], vspec["short"],
                base["psnr"], base["ssim"],
                best_spatial["name"], best_spatial["psnr"], best_spatial["ssim"],
                best_freq["name"], best_freq["psnr"], best_freq["ssim"],
                f"CLAHE clip={clip:g}", em["psnr"], em["ssim"],
            ])

            # Διαγνωστικά πέρα από PSNR/SSIM: ευκρίνεια, εντροπία, F1 ακμών
            e_noisy = edges.sobel_edges(noisy)
            e_rest = edges.sobel_edges(restored)
            f1_noisy = edges.edge_metrics(ref_edges_map, e_noisy)["f1"]
            f1_rest = edges.edge_metrics(ref_edges_map, e_rest)["f1"]
            extra_rows.append([
                spec["label"], vspec["short"], winner["name"],
                winner["psnr"], winner["ssim"],
                spatial.sharpness(restored) / ref_sharp * 100.0,
                enhancement.entropy(noisy), enhancement.entropy(restored),
                enhancement.entropy(enhanced),
                enhancement.used_levels_pct(restored),
                enhancement.used_levels_pct(enhanced),
                f1_noisy, f1_rest,
            ])
            io_utils.save_image(restored, out / f"{key}_{vspec['key']}_best.png")
            io_utils.save_image(enhanced, out / f"{key}_{vspec['key']}_enhanced.png")

            if vspec["key"] in ("gauss_s25", "sp_d10"):
                pipeline_items += [
                    (ref, "1. Αναφορά"),
                    (noisy, f"2. Υποβάθμιση\n{vspec['short']}\nPSNR={base['psnr']:.1f} dB"),
                    (restored, f"3. Αποκατάσταση\n{winner['name']}\nPSNR={winner['psnr']:.1f} dB"),
                    (enhanced, f"4. Βελτίωση\nCLAHE clip=2\nH={enhancement.entropy(enhanced):.2f} bits"),
                    (e_noisy, f"5α. Ακμές ΧΩΡΙΣ\nαποκατάσταση\nF1={f1_noisy:.3f}"),
                    (e_rest, f"5β. Ακμές ΜΕΤΑ την\nαποκατάσταση\nF1={f1_rest:.3f}"),
                ]

        viz.plot_grid(pipeline_items, out / f"{key}_pipeline.png", ncols=6,
                      figsize_per=(3.1, 4.0),
                      suptitle=f"Πλήρης αλυσίδα επεξεργασίας — {spec['label']}")

    order = (["Gaussian (γραμμικό)", "Median (τάξης)", "Bilateral (διατήρηση ακμών)",
              "Ιδανικό ΧΠΦ"]
             + [freq.LPF_LABEL[k] for k in freq.BUTTERWORTH_KEYS]
             + ["Γκαουσιανό ΧΠΦ"])
    series = [(fam, [family_best[fam][c] for c in case_labels])
              for fam in order if fam in family_best]
    viz.plot_bars(case_labels, series, out / "family_comparison.png",
                  ylabel="PSNR στο βέλτιστο κάθε οικογένειας (dB)",
                  title="Σύγκριση όλων των μεθόδων αποκατάστασης ανά περίπτωση",
                  ymin=18.0)

    oh = ["Εικόνα", "Υποβάθμιση", "PSNR πριν", "SSIM πριν", "Νικητής (PSNR)",
          "PSNR", "ΔPSNR", "SSIM νικητή", "Νικητής (SSIM)", "Καλύτερο SSIM",
          "Καλύτερο χωρικό", "Καλύτερο συχνοτήτων", "Διαφορά πεδίων (dB)"]
    tables.save(oh, overall_rows, out / "overall_best",
                title="Συνολικά καλύτερη μέθοδος αποκατάστασης ανά περίπτωση")
    print("\n" + tables.render(oh, overall_rows))

    fh = ["Εικόνα", "Υποβάθμιση", "Πεδίο", "Οικογένεια", "Καλύτερη παράμετρος",
          "PSNR πριν", "PSNR μετά", "ΔPSNR", "SSIM"]
    tables.save(fh, family_rows, out / "family_best",
                title="Καλύτερος εκπρόσωπος κάθε οικογένειας, χωρικό και συχνοτικό πεδίο μαζί")

    # Συγκεντρωτικός πίνακας του Σταδίου 8: PSNR και SSIM σε κάθε κύριο βήμα.
    ch = ["Εικόνα", "Υποβάθμιση",
          "1. Υποβαθμισμένη — PSNR", "1. Υποβαθμισμένη — SSIM",
          "2. Καλύτερο χωρικό φίλτρο", "2. PSNR", "2. SSIM",
          "3. Καλύτερο συχνοτικό φίλτρο", "3. PSNR", "3. SSIM",
          "4. Βελτίωση", "4. PSNR", "4. SSIM"]
    tables.save(ch, chain_rows, out / "pipeline_summary",
                title="Συγκεντρωτική αξιολόγηση: υποβαθμισμένη → καλύτερο χωρικό "
                      "φίλτρο → καλύτερο συχνοτικό φίλτρο → μετά τη βελτίωση")
    print("\n" + tables.render(ch, chain_rows))

    print("\n  Στήλη 4: το PSNR πέφτει μετά τη βελτίωση, καθώς η CLAHE αλλάζει"
          "\n  σκόπιμα την κατανομή των εντάσεων. Το κέρδος της φαίνεται στον"
          "\n  επόμενο πίνακα (εντροπία, επίπεδα γκρι σε χρήση).")

    xh = ["Εικόνα", "Υποβάθμιση", "Καλύτερη αποκατάσταση", "PSNR", "SSIM",
          "Ευκρίνεια (% αναφοράς)", "Εντροπία υποβαθμισμένης",
          "Εντροπία αποκατεστημένης", "Εντροπία μετά τη βελτίωση",
          "Επίπεδα σε χρήση πριν τη βελτίωση (%)",
          "Επίπεδα σε χρήση μετά τη βελτίωση (%)",
          "F1 ακμών χωρίς αποκατάσταση", "F1 ακμών μετά"]
    tables.save(xh, extra_rows, out / "pipeline_diagnostics",
                title="Διαγνωστικά της αλυσίδας: ευκρίνεια, εντροπία και ποιότητα ακμών")
    print("\n" + tables.render(xh, extra_rows))
    print(f"\n  → αποτελέσματα: {out}")


STAGES = {
    1: run_stage1,
    2: run_stage2,
    3: run_stage3,
    4: run_stage4,
    5: run_stage5,
    6: run_stage6,
    7: run_stage7,
    8: run_stage8,
}


def main():
    ap = argparse.ArgumentParser(description="ΕΥ631 — αλυσίδα επεξεργασίας εικόνας")
    ap.add_argument("--stage", nargs="*", type=int, default=None,
                    help="ποια στάδια να τρέξουν (π.χ. --stage 1 2). Χωρίς αυτό: όλα.")
    args = ap.parse_args()

    todo = sorted(STAGES) if not args.stage else args.stage
    for s in todo:
        if s not in STAGES:
            print(f"[!] Το στάδιο {s} δεν υπάρχει.")
            continue
        STAGES[s]()


if __name__ == "__main__":
    main()
