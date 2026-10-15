"""STAGE 7 — Edge detection"""
import numpy as np

from .. import config, degradation, edges, io_utils, tables, viz
from . import common


def _canny_default():
    """The middle threshold pair, used in all comparisons."""
    return config.CANNY_THRESHOLDS[len(config.CANNY_THRESHOLDS) // 2]


def _abs_norm(a):
    """Absolute value of an array, normalized to uint8 for display."""
    m = max(float(np.abs(a).max()), 1.0)
    return io_utils.to_uint8(np.abs(a) / m * 255.0)


def _sobel_panel(img, img_label, path, suptitle):
    """
    The Sobel operator in its three forms — horizontal (Gx), vertical (Gy)
    and combined |∇f| — plus the binary map from thresholding |∇f|.
    """
    gx, gy = edges.sobel_components(img)
    viz.plot_grid(
        [(img, img_label),
         (_abs_norm(gx), "|Gx| — horizontal operator\n(responds to VERTICAL edges)"),
         (_abs_norm(gy), "|Gy| — vertical operator\n(responds to HORIZONTAL edges)"),
         (edges.sobel_magnitude(img), "Combined: |∇f| = √(Gx²+Gy²)"),
         (edges.sobel_edges(img),
          f"Combined, thresholded\n(strongest "
          f"{100 - config.SOBEL_EDGE_PERCENTILE:.0f}% of pixels)")],
        path, ncols=3, suptitle=suptitle)


def _threshold_study(img, ref_canny_maps, ref_sobel, lead_item, path, suptitle,
                     rows, prefix):
    """
    Sweep of the Canny hysteresis thresholds with precision / recall / F1. Each
    setting is judged against the same detector on the reference image.
    """
    items = [lead_item]
    for (tlo, thi), rmap in ref_canny_maps.items():
        e = edges.canny(img, tlo, thi)
        em = edges.edge_metrics(rmap, e)
        rows.append(prefix + [f"Canny ({tlo}, {thi})", em["edge_pct"],
                              em["precision"], em["recall"], em["f1"]])
        items.append((e, f"Canny ({tlo}, {thi})\nedges {em['edge_pct']:.2f}%\n"
                         f"P={em['precision']:.2f} R={em['recall']:.2f}\n"
                         f"F1={em['f1']:.3f}"))
    e = edges.sobel_edges(img)
    em = edges.edge_metrics(ref_sobel, e)
    rows.append(prefix + [f"Sobel (strongest {100 - config.SOBEL_EDGE_PERCENTILE:.0f}%)",
                          em["edge_pct"], em["precision"], em["recall"], em["f1"]])
    items.append((e, f"Sobel thresholded\nedges {em['edge_pct']:.2f}%\n"
                     f"P={em['precision']:.2f} R={em['recall']:.2f}\n"
                     f"F1={em['f1']:.3f}"))
    viz.plot_grid(items, path, ncols=len(items), figsize_per=(3.3, 4.1),
                  suptitle=suptitle)


def run():
    print("\n=== STAGE 7: Edge detection ===")
    out = io_utils.ensure_dir(config.STAGE_DIRS[7])
    lo, hi = _canny_default()

    refs = io_utils.load_all_references()
    thr_rows, restored_rows, chain_rows = [], [], []

    for key, (spec, ref) in refs.items():
        print(f"\n  [{spec['label']}]")

        # Part A: the operators on the clean image (ground truth)
        # The reference edge maps are the yardstick for everything else.
        ref_sobel = edges.sobel_edges(ref)
        ref_canny_maps = {(tlo, thi): edges.canny(ref, tlo, thi)
                          for tlo, thi in config.CANNY_THRESHOLDS}
        ref_canny = ref_canny_maps[(lo, hi)]
        ref_canny_auto = edges.canny_auto(ref)

        io_utils.save_image(edges.sobel_magnitude(ref), out / f"{key}_sobel_mag.png")
        io_utils.save_image(ref_sobel, out / f"{key}_sobel_edges.png")
        io_utils.save_image(ref_canny, out / f"{key}_canny_edges.png")

        _sobel_panel(ref, "Reference", out / f"{key}_operators.png",
                     f"Sobel operator: horizontal, vertical and combined — "
                     f"{spec['label']}")

        # Hysteresis thresholds on the clean image. Precision/recall is
        # meaningless here (the image is the reference), so we measure the
        # edge percentage.
        thr_items = [(ref, "Reference")]
        for tlo, thi in config.CANNY_THRESHOLDS:
            e = ref_canny_maps[(tlo, thi)]
            pct = float((e > 0).mean() * 100.0)
            thr_rows.append([spec["label"], f"Canny ({tlo}, {thi})", pct])
            thr_items.append((e, f"Canny ({tlo}, {thi})\nedges {pct:.2f}%"))
        thr_rows.append([spec["label"],
                         f"Sobel (strongest {100 - config.SOBEL_EDGE_PERCENTILE:.0f}%)",
                         float((ref_sobel > 0).mean() * 100.0)])
        thr_items.append((ref_sobel, f"Sobel thresholded\nedges "
                                     f"{float((ref_sobel > 0).mean() * 100.0):.2f}%"))
        viz.plot_grid(thr_items, out / f"{key}_canny_thresholds.png", ncols=len(thr_items),
                      suptitle=f"Effect of the hysteresis thresholds on the clean image — "
                               f"{spec['label']}")

        # Part B: the same operators on the restored image
        # Restored = output of the best Stage 4 spatial filter.
        for vspec, noisy in degradation.degrade_all(ref, image_key=key):
            rspec, restored, rm = common.best_spatial_restoration(ref, noisy)
            print(f"\n    {vspec['short']}  (restoration: {rspec['short']}, "
                  f"PSNR={rm['psnr']:.2f} dB)")

            _sobel_panel(
                restored, f"Restored\n{rspec['short']}",
                out / f"{key}_{vspec['key']}_restored_sobel.png",
                f"Sobel on the restored image — {spec['label']} — "
                f"{vspec['short']} (restoration: {rspec['short']})")

            _threshold_study(
                restored, ref_canny_maps, ref_sobel,
                (restored, f"Restored\n{rspec['short']}\nPSNR={rm['psnr']:.2f} dB"),
                out / f"{key}_{vspec['key']}_restored_thresholds.png",
                f"Threshold selection on the restored image — {spec['label']} — "
                f"{vspec['short']}",
                restored_rows,
                [spec["label"], vspec["short"], rspec["short"]])

            for r in restored_rows[-(len(config.CANNY_THRESHOLDS) + 1):]:
                print(f"      {r[3]:<28s} edges={r[4]:5.2f}%  precision={r[5]:.3f}  "
                      f"recall={r[6]:.3f}  F1={r[7]:.3f}")

            # Part C: the same detectors on noisy vs restored
            grid_items = [(ref_canny, "Reference — Canny")]
            for src_name, src_img in (("Noisy", noisy),
                                      ("Restored", restored)):
                for det_name, det_fn, ref_map in (
                        ("Sobel", edges.sobel_edges, ref_sobel),
                        ("Canny", lambda im: edges.canny(im, lo, hi), ref_canny),
                        ("Adaptive Canny", edges.canny_auto, ref_canny_auto)):
                    emap = det_fn(src_img)
                    em = edges.edge_metrics(ref_map, emap)
                    chain_rows.append([spec["label"], vspec["short"], src_name,
                                       det_name, em["edge_pct"], em["precision"],
                                       em["recall"], em["f1"]])
                    if det_name == "Canny":
                        io_utils.save_image(
                            emap, out / f"{key}_{vspec['key']}_"
                                        f"{'noisy' if src_name == 'Noisy' else 'restored'}_canny.png")
                        grid_items.append((emap, f"{src_name} — Canny\nF1={em['f1']:.3f} "
                                                 f"(P={em['precision']:.2f}, R={em['recall']:.2f})"))

            viz.plot_grid(grid_items, out / f"{key}_{vspec['key']}_edges.png", ncols=3,
                          figsize_per=(3.3, 3.9),
                          suptitle=f"Edges before and after restoration — "
                                   f"{spec['label']} — {vspec['short']} "
                                   f"(restoration: {rspec['short']})")

    thr_headers = ["Image", "Detector / thresholds", "Pixels classified as edge (%)"]
    tables.save(thr_headers, thr_rows, out / "edge_thresholds",
                title="Effect of the thresholds on the clean reference image")
    print("\n" + tables.render(thr_headers, thr_rows))

    rest_headers = ["Image", "Degradation", "Restoration", "Detector / thresholds",
                    "Edges (%)", "Precision", "Recall", "F1"]
    tables.save(rest_headers, restored_rows, out / "edge_thresholds_restored",
                title="Edge detection on the restored image: effect of the "
                      "parameter choice")
    print("\n" + tables.render(rest_headers, restored_rows))

    chain_headers = ["Image", "Degradation", "Input", "Detector",
                     "Edges (%)", "Precision", "Recall", "F1"]
    tables.save(chain_headers, chain_rows, out / "edge_quality",
                title="Edge detection quality against the edges of the reference image")
    print("\n" + tables.render(chain_headers, chain_rows))
    print(f"\n  → results: {out}")
