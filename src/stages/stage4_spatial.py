"""STAGE 4 — Denoising in the spatial domain"""

from .. import config, degradation, io_utils, metrics, tables, viz
from .. import spatial_filters as spatial


def run():
    print("\n=== STAGE 4: Denoising in the spatial domain ===")
    out = io_utils.ensure_dir(config.STAGE_DIRS[4])

    refs = io_utils.load_all_references()
    sweep_rows, best_rows = [], []

    for key, (spec, ref) in refs.items():
        ref_sharp = spatial.sharpness(ref)
        print(f"\n  [{spec['label']}]  reference sharpness (var∇²) = {ref_sharp:.1f}")

        for vspec, noisy in degradation.degrade_all(ref, image_key=key):
            base = metrics.evaluate(ref, noisy)
            print(f"\n    {vspec['short']}  (before: PSNR={base['psnr']:.2f} dB, "
                  f"SSIM={base['ssim']:.4f})")

            results = []          # (fspec, filtered, metrics, sharpness%)
            for fspec, filt in spatial.apply_all(noisy):
                m = metrics.evaluate(ref, filt)
                sh = spatial.sharpness(filt) / ref_sharp * 100.0
                results.append((fspec, filt, m, sh))
                sweep_rows.append([spec["label"], vspec["short"], fspec["short"],
                                   base["psnr"], m["psnr"], m["psnr"] - base["psnr"],
                                   m["ssim"], m["ssim"] - base["ssim"], sh])

            # Best parameter per filter family (criterion: PSNR), so that each
            # family is judged at its own optimum.
            grid_items = [(ref, "Reference"),
                          (noisy, f"Noisy — {vspec['short']}\n"
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
                                         f"sharpness {bsh:.0f}%"))
                print(f"      {spatial.FAMILY_LABEL[family]:<28s} best: {bspec['short']:<22s} "
                      f"PSNR={bm['psnr']:6.2f} ({bm['psnr'] - base['psnr']:+5.2f}) "
                      f"SSIM={bm['ssim']:.4f} sharpness={bsh:5.1f}%")

            viz.plot_grid(grid_items, out / f"{key}_{vspec['key']}_compare.png",
                          ncols=len(grid_items),
                          suptitle=f"Spatial filters — {spec['label']} — {vspec['short']}")

            curves = []
            for family in ("gaussian", "median"):
                fam = sorted((r for r in results if r[0]["family"] == family),
                             key=lambda r: r[0]["size"])
                curves.append(([r[0]["size"] for r in fam],
                               [r[2]["psnr"] for r in fam],
                               spatial.FAMILY_LABEL[family]))
            viz.plot_curves(curves, out / f"{key}_{vspec['key']}_psnr_vs_ksize.png",
                            xlabel="Kernel size k (k×k)", ylabel="PSNR (dB)",
                            title=f"Effect of kernel size — {spec['label']} — {vspec['short']}",
                            hlines=[(base["psnr"], "no filtering")])

    sweep_headers = ["Image", "Degradation", "Filter", "PSNR before", "PSNR after",
                     "ΔPSNR", "SSIM after", "ΔSSIM", "Sharpness (% of reference)"]
    tables.save(sweep_headers, sweep_rows, out / "spatial_sweep",
                title="Full parameter sweep of the spatial filters")

    best_headers = ["Image", "Degradation", "Family", "Best parameter",
                    "PSNR before", "PSNR after", "ΔPSNR", "SSIM after", "ΔSSIM",
                    "Sharpness (% of reference)"]
    tables.save(best_headers, best_rows, out / "spatial_best",
                title="Best parameter per filter family (criterion: PSNR)")
    print("\n" + tables.render(best_headers, best_rows))
    print(f"\n  → results: {out}")
