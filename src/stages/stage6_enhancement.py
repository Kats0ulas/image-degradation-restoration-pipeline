"""ΣΤΑΔΙΟ 6 — Βελτίωση εικόνας (ενίσχυση αντίθεσης)"""
import numpy as np

from .. import config, degradation, enhancement, io_utils, metrics, tables, viz
from .. import spatial_filters as spatial
from . import common


def run():
    print("\n=== ΣΤΑΔΙΟ 6: Βελτίωση εικόνας ===")
    out = io_utils.ensure_dir(config.STAGE_DIRS[6])

    refs = io_utils.load_all_references()
    rows, chain_rows, denoised_rows = [], [], []

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
        rspec, restored, rm = common.best_spatial_restoration(ref, noisy)
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

        # Μέρος Γ: όλες οι τεχνικές πάνω στην ΑΠΟΚΑΤΕΣΤΗΜΕΝΗ (αποθορυβοποιημένη)
        # εικόνα — το ζητούμενο του σταδίου. Η σάρωση του clipLimit και τα
        # ιστογράμματα πριν/μετά γίνονται εδώ, όχι μόνο στην καθαρή εικόνα.
        p0 = enhancement.profile(restored)
        denoised_rows.append([spec["label"], vspec["short"], rspec["short"],
                              "Χωρίς βελτίωση", p0["mean"], p0["std"],
                              p0["entropy"], p0["used_pct"], rm["psnr"], rm["ssim"]])
        print(f"\n    Βελτίωση της αποκατεστημένης ({rspec['short']}):")
        print(f"      {'Χωρίς βελτίωση':<28s} std={p0['std']:6.2f}  "
              f"εντροπία={p0['entropy']:.3f}  επίπεδα σε χρήση={p0['used_pct']:5.1f}%")

        den_items = [(restored, f"Αποκατεστημένη\n{rspec['short']}\n"
                                f"std={p0['std']:.1f} · H={p0['entropy']:.2f} bits")]
        den_hist = [(restored, "Αποκατεστημένη (πριν)")]
        for mspec in enhancement.method_specs():
            eimg = mspec["fn"](restored)
            io_utils.save_image(eimg, out / f"{key}_restored_{mspec['key']}.png")
            p = enhancement.profile(eimg)
            m = metrics.evaluate(ref, eimg)
            denoised_rows.append([spec["label"], vspec["short"], rspec["short"],
                                  mspec["short"], p["mean"], p["std"],
                                  p["entropy"], p["used_pct"], m["psnr"], m["ssim"]])
            print(f"      {mspec['short']:<28s} std={p['std']:6.2f}  "
                  f"εντροπία={p['entropy']:.3f}  επίπεδα σε χρήση={p['used_pct']:5.1f}%")
            den_items.append((eimg, f"{mspec['short']}\nstd={p['std']:.1f} · "
                                    f"H={p['entropy']:.2f} bits\nεπίπεδα {p['used_pct']:.0f}%"))
            if mspec["key"] != "stretch":      # η εκφώνηση ζητά HE και CLAHE
                den_hist.append((eimg, mspec["short"]))

        viz.plot_grid(den_items, out / f"{key}_restored_methods.png", ncols=4,
                      figsize_per=(3.3, 4.1),
                      suptitle=f"Βελτίωση της αποθορυβοποιημένης εικόνας — "
                               f"{spec['label']} — {vspec['short']} → {rspec['short']}")
        viz.plot_hist_row(den_hist, out / f"{key}_restored_hist.png",
                          suptitle=f"Ιστογράμματα πριν/μετά τη βελτίωση της "
                                   f"αποθορυβοποιημένης εικόνας — {spec['label']}")

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

    den_headers = ["Εικόνα", "Υποβάθμιση", "Αποκατάσταση", "Τεχνική βελτίωσης",
                   "Μέση τιμή", "Τυπ. απόκλιση (αντίθεση)", "Εντροπία (bits)",
                   "Επίπεδα σε χρήση (%)", "PSNR (dB)", "SSIM"]
    tables.save(den_headers, denoised_rows, out / "enhancement_denoised",
                title="Βελτίωση των αποθορυβοποιημένων εικόνων "
                      "(HE και CLAHE για κάθε clipLimit)")
    print("\n" + tables.render(den_headers, denoised_rows))
    print(f"\n  → αποτελέσματα: {out}")
