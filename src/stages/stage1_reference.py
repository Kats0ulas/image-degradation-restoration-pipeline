"""STAGE 1 — Selection and preparation of the reference images (ground truth)"""
import cv2
import numpy as np

from .. import config, io_utils, tables, viz


def image_stats(img):
    """
    Descriptive measures of each reference image:
      mean/std : position and spread of the histogram (contrast)
      mean|∇|  : mean gradient magnitude (Sobel)
      edge %   : percentage of pixels that Canny classifies as edge
    """
    f = img.astype(np.float64)
    gx = cv2.Sobel(f, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    edge_map = cv2.Canny(img, 100, 200)
    return {
        "mean": float(f.mean()),
        "std": float(f.std()),
        "min": int(img.min()),
        "max": int(img.max()),
        "mean_grad": float(grad.mean()),
        "edge_pct": float((edge_map > 0).mean() * 100.0),
    }


def run():
    print("\n=== STAGE 1: Reference images ===")
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
                  suptitle="Reference images (ground truth)")
    viz.plot_hist_row(grid_items, out / "references_hist.png",
                      suptitle="Reference image histograms")

    headers = ["Image", "Dimensions", "Mean", "Std. dev.",
               "Range", "Mean |∇| (Sobel)", "Canny edges (%)"]
    tables.save(headers, rows, out / "reference_stats", title="Reference image characteristics")
    print("\n" + tables.render(headers, rows))
    print(f"\n  → results: {out}")
