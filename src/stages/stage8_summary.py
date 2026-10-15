"""STAGE 8 — Overall evaluation of the whole pipeline"""

from .. import config, degradation, edges, enhancement, io_utils, metrics, tables, viz
from .. import frequency_filters as freq
from .. import spatial_filters as spatial


def _all_restorations(ref, noisy):
    """
    All restoration methods of Stages 4 and 5 in one list, so that the two
    domains are compared by the same criteria.
    """
    res = []
    for fspec, filt in spatial.apply_all(noisy):
        res.append({"domain": "Spatial", "family": spatial.FAMILY_LABEL[fspec["family"]],
                    "name": fspec["short"], "image": filt})
    for fam in freq.LPF_BUILDERS:
        # One forward DFT for the whole cutoff sweep of the family.
        for d0, filt in zip(config.LPF_CUTOFFS,
                            freq.lowpass_many(noisy, fam, config.LPF_CUTOFFS)):
            res.append({"domain": "Frequency", "family": freq.LPF_LABEL[fam],
                        "name": f"{freq.LPF_LABEL[fam]} D0={d0:g}",
                        "image": filt})
    for r in res:
        r.update(metrics.evaluate(ref, r["image"]))
    return res


def run():
    print("\n=== STAGE 8: Overall evaluation ===")
    out = io_utils.ensure_dir(config.STAGE_DIRS[8])

    refs = io_utils.load_all_references()
    overall_rows, family_rows, chain_rows, extra_rows = [], [], [], []
    # For the bar chart: PSNR of the best representative of each family.
    case_labels, family_best = [], {}
    # Which cases go into the full-pipeline figure: the STRONGEST degradation
    # of each noise type, selected by rule rather than by fixed keys. With
    # fixed keys, a config change left pipeline_items empty and plot_grid
    # crashed. (With the current config: gauss_s25, sp_d10.)
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

            # Winner per family and overall winner
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
            best_spatial = max((r for r in res if r["domain"] == "Spatial"),
                               key=lambda r: r["psnr"])
            best_freq = max((r for r in res if r["domain"] == "Frequency"),
                            key=lambda r: r["psnr"])

            overall_rows.append([
                spec["label"], vspec["short"], base["psnr"], base["ssim"],
                winner["name"], winner["psnr"], winner["psnr"] - base["psnr"],
                winner["ssim"], winner_ssim["name"], winner_ssim["ssim"],
                best_spatial["psnr"], best_freq["psnr"],
                best_spatial["psnr"] - best_freq["psnr"],
            ])
            print(f"    {vspec['short']:<12s} winner: {winner['name']:<26s} "
                  f"PSNR={winner['psnr']:6.2f} ({winner['psnr'] - base['psnr']:+6.2f})  "
                  f"| spatial {best_spatial['psnr']:6.2f} vs frequency {best_freq['psnr']:6.2f} "
                  f"({best_spatial['psnr'] - best_freq['psnr']:+5.2f} dB)")

            # PSNR and SSIM at every step of the pipeline: degraded ->
            # best spatial -> best frequency -> after enhancement.
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

            # Diagnostics beyond PSNR/SSIM: sharpness, entropy, edge F1
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
                    (ref, "1. Reference"),
                    (noisy, f"2. Degradation\n{vspec['short']}\nPSNR={base['psnr']:.1f} dB"),
                    (restored, f"3. Restoration\n{winner['name']}\nPSNR={winner['psnr']:.1f} dB"),
                    (enhanced, f"4. Enhancement\nCLAHE clip=2\nH={enhancement.entropy(enhanced):.2f} bits"),
                    (e_noisy, f"5a. Edges WITHOUT\nrestoration\nF1={f1_noisy:.3f}"),
                    (e_rest, f"5b. Edges AFTER\nrestoration\nF1={f1_rest:.3f}"),
                ]

        viz.plot_grid(pipeline_items, out / f"{key}_pipeline.png", ncols=6,
                      figsize_per=(3.1, 4.0),
                      suptitle=f"Full processing pipeline — {spec['label']}")

    order = ([spatial.FAMILY_LABEL[k] for k in ("gaussian", "median", "bilateral")]
             + [freq.LPF_LABEL["ideal"]]
             + [freq.LPF_LABEL[k] for k in freq.BUTTERWORTH_KEYS]
             + [freq.LPF_LABEL["gaussian"]])
    series = [(fam, [family_best[fam][c] for c in case_labels])
              for fam in order if fam in family_best]
    viz.plot_bars(case_labels, series, out / "family_comparison.png",
                  ylabel="PSNR at the optimum of each family (dB)",
                  title="Comparison of all restoration methods per case",
                  ymin=18.0)

    oh = ["Image", "Degradation", "PSNR before", "SSIM before", "Winner (PSNR)",
          "PSNR", "ΔPSNR", "Winner SSIM", "Winner (SSIM)", "Best SSIM",
          "Best spatial", "Best frequency", "Domain difference (dB)"]
    tables.save(oh, overall_rows, out / "overall_best",
                title="Overall best restoration method per case")
    print("\n" + tables.render(oh, overall_rows))

    fh = ["Image", "Degradation", "Domain", "Family", "Best parameter",
          "PSNR before", "PSNR after", "ΔPSNR", "SSIM"]
    tables.save(fh, family_rows, out / "family_best",
                title="Best representative of each family, spatial and frequency domain together")

    # Stage 8 summary table: PSNR and SSIM at each main step.
    ch = ["Image", "Degradation",
          "1. Degraded — PSNR", "1. Degraded — SSIM",
          "2. Best spatial filter", "2. PSNR", "2. SSIM",
          "3. Best frequency filter", "3. PSNR", "3. SSIM",
          "4. Enhancement", "4. PSNR", "4. SSIM"]
    tables.save(ch, chain_rows, out / "pipeline_summary",
                title="Overall evaluation: degraded → best spatial "
                      "filter → best frequency filter → after enhancement")
    print("\n" + tables.render(ch, chain_rows))

    print("\n  Column 4: PSNR drops after enhancement, because CLAHE deliberately"
          "\n  changes the intensity distribution. Its benefit shows in the"
          "\n  next table (entropy, gray levels in use).")

    xh = ["Image", "Degradation", "Best restoration", "PSNR", "SSIM",
          "Sharpness (% of reference)", "Entropy of degraded",
          "Entropy of restored", "Entropy after enhancement",
          "Levels in use before enhancement (%)",
          "Levels in use after enhancement (%)",
          "Edge F1 without restoration", "Edge F1 after"]
    tables.save(xh, extra_rows, out / "pipeline_diagnostics",
                title="Pipeline diagnostics: sharpness, entropy and edge quality")
    print("\n" + tables.render(xh, extra_rows))
    print(f"\n  → results: {out}")
