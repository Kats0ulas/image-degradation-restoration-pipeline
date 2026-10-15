"""
Plots: histograms, image grids, saving figures.
The "Agg" backend (no window) is used so that it also runs on a server.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config
from .io_utils import ensure_dir

# DejaVu Sans (the matplotlib default) fully covers the math symbols used in
# the labels (σ, Δ, ∇).
plt.rcParams.update({
    "figure.dpi": config.FIG_DPI,
    "savefig.dpi": config.FIG_DPI,
    "font.size": 9,
    "axes.titlesize": 10,
    "figure.autolayout": False,
})


def save_fig(fig, path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def histogram(img) -> np.ndarray:
    """256-bin histogram of an 8-bit grayscale image."""
    return np.bincount(np.asarray(img, dtype=np.uint8).ravel(), minlength=256)


def plot_image_with_hist(img, title, path, hist_color="#3b6ea5"):
    """An image next to its histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(title)
    axes[0].axis("off")

    h = histogram(img)
    axes[1].bar(np.arange(256), h, width=1.0, color=hist_color)
    axes[1].set_xlim(0, 255)
    axes[1].set_title("Intensity histogram")
    axes[1].set_xlabel("Gray level")
    axes[1].set_ylabel("Pixel count")
    fig.tight_layout()
    return save_fig(fig, path)


def plot_grid(items, path, ncols=3, figsize_per=(3.2, 3.4), suptitle=None, cmap="gray"):
    """
    Image grid.  items = list of (image, title).
    """
    items = list(items)
    n = len(items)
    if n == 0:
        # Without this check ncols becomes 0 and a ZeroDivisionError is raised
        # in n/ncols — a message that says nothing about the cause.
        raise ValueError(f"plot_grid: no images were given for {path}")
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (img, title) in zip(axes, items):
        kw = {"cmap": cmap}
        if np.asarray(img).dtype == np.uint8:
            kw.update(vmin=0, vmax=255)
        ax.imshow(img, **kw)
        ax.set_title(title)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    return save_fig(fig, path)


def plot_hist_row(items, path, color="#3b6ea5", suptitle=None):
    """A row of histograms.  items = list of (image, title)."""
    items = list(items)
    n = len(items)
    if n == 0:
        raise ValueError(f"plot_hist_row: no images were given for {path}")
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 3.0), squeeze=False)
    for ax, (img, title) in zip(axes[0], items):
        ax.bar(np.arange(256), histogram(img), width=1.0, color=color)
        ax.set_xlim(0, 255)
        ax.set_title(title)
        ax.set_xlabel("Gray level")
    axes[0][0].set_ylabel("Pixel count")
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    return save_fig(fig, path)


def plot_curves(curves, path, xlabel, ylabel, title=None, logy=False,
                figsize=(7.0, 4.2), xlim=None, hlines=None, vlines=None,
                markers=False):
    """
    Curve plot.  curves = list of (x, y, label).
    logy=True for a logarithmic y axis (power spectra).
    hlines / vlines = lists of (value, label) for reference lines.
    """
    fig, ax = plt.subplots(figsize=figsize)
    for x, y, label in curves:
        ax.plot(x, y, label=label, linewidth=1.4,
                marker="o" if markers else None, markersize=3.5)
    for y, label in (hlines or []):
        ax.axhline(y, color="0.5", linestyle="--", linewidth=0.9)
        ax.annotate(label, xy=(0.995, y), xycoords=("axes fraction", "data"),
                    ha="right", va="bottom", fontsize=7, color="0.35")
    for x, label in (vlines or []):
        ax.axvline(x, color="#b03030", linestyle=":", linewidth=1.1)
        ax.annotate(label, xy=(x, 0.02), xycoords=("data", "axes fraction"),
                    ha="left", va="bottom", fontsize=7, color="#b03030",
                    rotation=90)
    if logy:
        ax.set_yscale("log")
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, linewidth=0.6)
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return save_fig(fig, path)


def plot_bars(categories, series, path, ylabel, title=None, figsize=(9.5, 4.6),
              hline=None, value_fmt="{:.1f}", ymin=None):
    """
    Grouped bar chart.
      categories : x-axis labels (e.g. the degradation cases)
      series     : list of (label, values) — one bar per group
    """
    n_cat = len(categories)
    n_ser = len(series)
    x = np.arange(n_cat)
    width = 0.8 / n_ser
    fig, ax = plt.subplots(figsize=figsize)
    for i, (label, values) in enumerate(series):
        pos = x - 0.4 + width * (i + 0.5)
        bars = ax.bar(pos, values, width, label=label)
        for b, v in zip(bars, values):
            ax.annotate(value_fmt.format(v), xy=(b.get_x() + b.get_width() / 2, v),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6.5)
    if hline is not None:
        ax.axhline(hline, color="0.4", linestyle="--", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)
    if ymin is not None:
        # The axis does not start at 0, otherwise differences of a few dB get lost.
        top = max(max(v) for _, v in series)
        ax.set_ylim(ymin, top + (top - ymin) * 0.16)
    ax.legend(fontsize=8, ncol=min(n_ser, 3), loc="upper center",
              bbox_to_anchor=(0.5, -0.10), frameon=False)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return save_fig(fig, path)
