"""ΣΤΑΔΙΟ 7 — Ανίχνευση ακμών"""
import numpy as np

from .. import config, degradation, edges, io_utils, tables, viz
from . import common


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


def run():
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
            rspec, restored, rm = common.best_spatial_restoration(ref, noisy)
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
