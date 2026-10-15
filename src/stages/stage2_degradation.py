"""ΣΤΑΔΙΟ 2 — Τεχνητή υποβάθμιση (Gaussian & Salt-and-Pepper θόρυβος)"""
import numpy as np

from .. import config, degradation, io_utils, metrics, tables, viz


def run():
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
