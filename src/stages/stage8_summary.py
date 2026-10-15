"""ΣΤΑΔΙΟ 8 — Συγκεντρωτική αξιολόγηση όλης της αλυσίδας"""

from .. import config, degradation, edges, enhancement, io_utils, metrics, tables, viz
from .. import frequency_filters as freq
from .. import spatial_filters as spatial


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
        # Ένας ευθύς DFT για όλη τη σάρωση αποκοπών της οικογένειας.
        for d0, filt in zip(config.LPF_CUTOFFS,
                            freq.lowpass_many(noisy, fam, config.LPF_CUTOFFS)):
            res.append({"domain": "Συχνοτήτων", "family": freq.LPF_LABEL[fam],
                        "name": f"{freq.LPF_LABEL[fam]} D0={d0:g}",
                        "image": filt})
    for r in res:
        r.update(metrics.evaluate(ref, r["image"]))
    return res


def run():
    print("\n=== ΣΤΑΔΙΟ 8: Συγκεντρωτική αξιολόγηση ===")
    out = io_utils.ensure_dir(config.STAGE_DIRS[8])

    refs = io_utils.load_all_references()
    overall_rows, family_rows, chain_rows, extra_rows = [], [], [], []
    # Για το ραβδόγραμμα: PSNR του καλύτερου εκπροσώπου κάθε οικογένειας.
    case_labels, family_best = [], {}
    # Ποιες περιπτώσεις μπαίνουν στην εικόνα της πλήρους αλυσίδας: η ΙΣΧΥΡΟΤΕΡΗ
    # υποβάθμιση κάθε τύπου θορύβου, επιλεγμένη με κανόνα και όχι με σταθερά
    # κλειδιά. Με σταθερά κλειδιά, μια αλλαγή στο config άφηνε το pipeline_items
    # άδειο και το plot_grid έσκαγε. (Με το τρέχον config: gauss_s25, sp_d10.)
    showcase = {s["kind"]: s["key"]
                for s in sorted(degradation.variant_specs(),
                                key=lambda s: s["param"])}

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

            if vspec["key"] in showcase.values():
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
