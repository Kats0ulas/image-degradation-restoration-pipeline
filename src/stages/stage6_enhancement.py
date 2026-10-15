"""STAGE 6 — Image enhancement (contrast enhancement)"""
import numpy as np

from .. import config, degradation, enhancement, io_utils, metrics, tables, viz
from .. import spatial_filters as spatial
from . import common


def run():
    print("\n=== STAGE 6: Image enhancement ===")
    out = io_utils.ensure_dir(config.STAGE_DIRS[6])

    refs = io_utils.load_all_references()
    rows, chain_rows, denoised_rows = [], [], []

    for key, (spec, ref) in refs.items():
        print(f"\n  [{spec['label']}]")

        # Part A: enhancement of the clean image
        base = enhancement.profile(ref)
        rows.append([spec["label"], "Reference (unprocessed)", base["mean"],
                     base["std"], base["entropy"], base["used_pct"], 1.0])
        print(f"    {'Reference':<28s} std={base['std']:6.2f}  entropy={base['entropy']:.3f}  "
              f"levels in use={base['used_pct']:5.1f}%")

        grid_items = [(ref, "Reference")]
        hist_items = [(ref, "Reference")]
        curve_items = [(np.arange(256), enhancement.transfer_curve(ref),
                        "Reference CDF → HE T(r)")]

        for mspec in enhancement.method_specs():
            eimg = mspec["fn"](ref)
            io_utils.save_image(eimg, out / f"{key}_{mspec['key']}.png")
            p = enhancement.profile(eimg)
            ss = metrics.ssim(ref, eimg)
            rows.append([spec["label"], mspec["short"], p["mean"], p["std"],
                         p["entropy"], p["used_pct"], ss])
            print(f"    {mspec['short']:<28s} std={p['std']:6.2f}  entropy={p['entropy']:.3f}  "
                  f"levels in use={p['used_pct']:5.1f}%  SSIM={ss:.4f}")
            title = (f"{mspec['short']}\nstd={p['std']:.1f} · H={p['entropy']:.2f} bits\n"
                     f"levels {p['used_pct']:.0f}%")
            grid_items.append((eimg, title))
            if mspec["key"] in ("he", "clahe_c2_t8"):
                hist_items.append((eimg, mspec["short"]))

        viz.plot_grid(grid_items, out / f"{key}_methods.png", ncols=4,
                      figsize_per=(3.3, 4.1),
                      suptitle=f"Enhancement techniques — {spec['label']}")
        viz.plot_hist_row(hist_items, out / f"{key}_hist.png",
                          suptitle=f"Histograms before/after enhancement — {spec['label']}")
        # The HE transformation function s = T(r)
        viz.plot_curves(curve_items + [(np.arange(256), np.arange(256), "identity (no change)")],
                        out / f"{key}_transfer.png",
                        xlabel="Input r (gray level)", ylabel="Output s = T(r)",
                        title=f"Equalization transformation function — {spec['label']}",
                        xlim=(0, 255))

        # Part B: the order within the pipeline
        # The same enhancement, applied (a) to the noisy and (b) to the restored image.
        vspec = next(v for v in degradation.variant_specs()
                     if v["key"] == config.ENHANCE_CHAIN_VARIANT)
        noisy = degradation.apply_variant(ref, vspec, image_key=key)
        rspec, restored, rm = common.best_spatial_restoration(ref, noisy)
        print(f"\n    Pipeline with {vspec['short']} — best restoration: "
              f"{rspec['short']} (PSNR={rm['psnr']:.2f} dB)")

        chain = [("Noisy", noisy), (f"Restored ({rspec['short']})", restored)]
        chain_items = [(ref, "Reference")]
        for stage_name, img in chain:
            for mkey, mlabel in (("raw", "no enhancement"), ("he", "+ HE"),
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
                    eimg, out / f"{key}_chain_{'noisy' if stage_name == 'Noisy' else 'restored'}_{mkey}.png")
                chain_items.append((eimg, f"{stage_name}\n{mlabel}\n"
                                          f"PSNR={m['psnr']:.2f} · SSIM={m['ssim']:.3f}"))
                print(f"      {stage_name:<32s} {mlabel:<16s} std={p['std']:6.2f}  "
                      f"PSNR={m['psnr']:6.2f}  SSIM={m['ssim']:.4f}  sharpness={sh:6.1f}%")

        viz.plot_grid(chain_items, out / f"{key}_chain.png", ncols=4,
                      figsize_per=(3.4, 4.3),
                      suptitle=f"Order in the pipeline: enhancement BEFORE vs AFTER "
                               f"restoration — {spec['label']} — {vspec['short']}")

        # Part C: all techniques on the RESTORED (denoised) image — what the
        # stage asks for. The clipLimit sweep and the before/after histograms
        # are done here, not only on the clean image.
        p0 = enhancement.profile(restored)
        denoised_rows.append([spec["label"], vspec["short"], rspec["short"],
                              "No enhancement", p0["mean"], p0["std"],
                              p0["entropy"], p0["used_pct"], rm["psnr"], rm["ssim"]])
        print(f"\n    Enhancement of the restored image ({rspec['short']}):")
        print(f"      {'No enhancement':<28s} std={p0['std']:6.2f}  "
              f"entropy={p0['entropy']:.3f}  levels in use={p0['used_pct']:5.1f}%")

        den_items = [(restored, f"Restored\n{rspec['short']}\n"
                                f"std={p0['std']:.1f} · H={p0['entropy']:.2f} bits")]
        den_hist = [(restored, "Restored (before)")]
        for mspec in enhancement.method_specs():
            eimg = mspec["fn"](restored)
            io_utils.save_image(eimg, out / f"{key}_restored_{mspec['key']}.png")
            p = enhancement.profile(eimg)
            m = metrics.evaluate(ref, eimg)
            denoised_rows.append([spec["label"], vspec["short"], rspec["short"],
                                  mspec["short"], p["mean"], p["std"],
                                  p["entropy"], p["used_pct"], m["psnr"], m["ssim"]])
            print(f"      {mspec['short']:<28s} std={p['std']:6.2f}  "
                  f"entropy={p['entropy']:.3f}  levels in use={p['used_pct']:5.1f}%")
            den_items.append((eimg, f"{mspec['short']}\nstd={p['std']:.1f} · "
                                    f"H={p['entropy']:.2f} bits\nlevels {p['used_pct']:.0f}%"))
            if mspec["key"] != "stretch":      # the assignment asks for HE and CLAHE
                den_hist.append((eimg, mspec["short"]))

        viz.plot_grid(den_items, out / f"{key}_restored_methods.png", ncols=4,
                      figsize_per=(3.3, 4.1),
                      suptitle=f"Enhancement of the denoised image — "
                               f"{spec['label']} — {vspec['short']} → {rspec['short']}")
        viz.plot_hist_row(den_hist, out / f"{key}_restored_hist.png",
                          suptitle=f"Histograms before/after enhancement of the "
                                   f"denoised image — {spec['label']}")

    headers = ["Image", "Technique", "Mean", "Std. dev. (contrast)",
               "Entropy (bits)", "Levels in use (%)", "SSIM vs reference"]
    tables.save(headers, rows, out / "enhancement_metrics",
                title="Contrast enhancement on the clean reference images")
    print("\n" + tables.render(headers, rows))

    chain_headers = ["Image", "Pipeline stage", "Enhancement", "Std. dev.",
                     "Entropy (bits)", "PSNR (dB)", "SSIM", "Sharpness (% of reference)"]
    tables.save(chain_headers, chain_rows, out / "enhancement_chain",
                title="Effect of the order: enhancement before vs after restoration")
    print("\n" + tables.render(chain_headers, chain_rows))

    den_headers = ["Image", "Degradation", "Restoration", "Enhancement technique",
                   "Mean", "Std. dev. (contrast)", "Entropy (bits)",
                   "Levels in use (%)", "PSNR (dB)", "SSIM"]
    tables.save(den_headers, denoised_rows, out / "enhancement_denoised",
                title="Enhancement of the denoised images "
                      "(HE and CLAHE for each clipLimit)")
    print("\n" + tables.render(den_headers, denoised_rows))
    print(f"\n  → results: {out}")
