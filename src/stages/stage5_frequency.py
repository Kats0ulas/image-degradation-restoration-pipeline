"""STAGE 5 — Denoising in the frequency domain (low-pass filters)"""

from .. import config, degradation, io_utils, metrics, tables, viz
from .. import frequency_filters as freq
from .. import spatial_filters as spatial


def _lpf_gallery(out):
    """
    The transfer functions H(u,v) of all filters — one kernel image per
    filter (a separate one for each order n) — together with their radial responses.
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
                  suptitle="Transfer functions of the low-pass filters")
    viz.plot_curves(curves, out / "lpf_transfer_profiles.png",
                    xlabel="Normalized radial frequency D (1 = Nyquist)",
                    ylabel="H(D)", title="Radial response of the filters (D0 = 0.20)",
                    xlim=(0, 0.6), vlines=[(d0, "D0")])


def _ringing_demo(ref, label, out, key):
    """
    Gibbs phenomenon: same cutoff frequency, an increasingly steep transition.
    Applied to the clean image, so that ringing is not confused with noise.
    """
    d0 = 0.08
    items = [(ref, "Reference")]
    for n in config.BUTTERWORTH_ORDERS_DEMO:
        H = freq.lowpass_butterworth(ref.shape, d0, order=n)
        items.append((freq.apply_transfer(ref, H), f"Butterworth n={n}"))
    items.append((freq.lowpass(ref, "ideal", d0), "Ideal (n→∞)"))
    viz.plot_grid(items, out / f"{key}_ringing_demo.png", ncols=len(items),
                  suptitle=f"Gibbs phenomenon: the steeper the cutoff, the stronger "
                           f"the ringing (D0={d0}) — {label}")


def run():
    print("\n=== STAGE 5: Denoising in the frequency domain ===")
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
            # The Stage 3 prediction for the cutoff frequency.
            r_snr1 = freq.snr_crossover_radius(ref, noisy)
            print(f"\n    {vspec['short']}  (before: PSNR={base['psnr']:.2f} dB) "
                  f"— Stage 3 prediction: r(SNR=1)={r_snr1:.3f}")

            results = {}
            for fam in freq.LPF_BUILDERS:
                res = []
                # One forward DFT for the whole cutoff sweep of the family.
                for d0, filt in zip(config.LPF_CUTOFFS,
                                    freq.lowpass_many(noisy, fam, config.LPF_CUTOFFS)):
                    m = metrics.evaluate(ref, filt)
                    sh = spatial.sharpness(filt) / ref_sharp * 100.0
                    res.append((d0, filt, m, sh))
                    sweep_rows.append([spec["label"], vspec["short"], freq.LPF_LABEL[fam],
                                       freq.LPF_ORDER[fam], d0, base["psnr"], m["psnr"],
                                       m["psnr"] - base["psnr"], m["ssim"],
                                       m["ssim"] - base["ssim"], sh])
                results[fam] = res

            # Best cutoff frequency per family
            grid_items = [(ref, "Reference"),
                          (noisy, f"Noisy — {vspec['short']}\n"
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
                                         f"sharpness {bsh:.0f}%"))
                print(f"      {freq.LPF_LABEL[fam]:<24s} optimal D0={d0:.2f}  "
                      f"PSNR={bm['psnr']:6.2f} ({bm['psnr'] - base['psnr']:+5.2f})  "
                      f"SSIM={bm['ssim']:.4f}  sharpness={bsh:5.1f}%")

            viz.plot_grid(grid_items, out / f"{key}_{vspec['key']}_compare.png",
                          ncols=min(len(grid_items), 4),
                          suptitle=f"Low-pass filters — {spec['label']} — {vspec['short']}")

            # PSNR as a function of D0, with the Stage 3 r(SNR=1) marked
            curves = [([d for d, _, _, _ in res], [m["psnr"] for _, _, m, _ in res],
                       freq.LPF_LABEL[fam]) for fam, res in results.items()]
            viz.plot_curves(curves, out / f"{key}_{vspec['key']}_psnr_vs_cutoff.png",
                            xlabel="Cutoff frequency D0 (1 = Nyquist)",
                            ylabel="PSNR (dB)", markers=True,
                            title=f"Cutoff frequency selection — {spec['label']} — {vspec['short']}",
                            hlines=[(base["psnr"], "no filtering")],
                            vlines=[(r_snr1, "r(SNR=1) — Stage 3")])

    sweep_headers = ["Image", "Degradation", "Filter", "Order n", "D0",
                     "PSNR before", "PSNR after", "ΔPSNR", "SSIM after", "ΔSSIM",
                     "Sharpness (% of reference)"]
    tables.save(sweep_headers, sweep_rows, out / "frequency_sweep",
                title="Full sweep of cutoff frequencies")

    best_headers = ["Image", "Degradation", "Filter", "Order n", "Optimal D0",
                    "r(SNR=1)", "PSNR before", "PSNR after", "ΔPSNR", "SSIM after",
                    "ΔSSIM", "Sharpness (% of reference)"]
    tables.save(best_headers, best_rows, out / "frequency_best",
                title="Optimal cutoff frequency per filter, next to the Stage 3 prediction")
    print("\n" + tables.render(best_headers, best_rows))
    print(f"\n  → results: {out}")
