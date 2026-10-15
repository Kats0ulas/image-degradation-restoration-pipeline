"""
Central project settings: images, noise levels, filter parameters and output
paths. Experiment parameters are changed only from here.
"""
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "images"
RESULTS_DIR = ROOT / "results"

STAGE_DIRS = {
    1: RESULTS_DIR / "stage1_reference",
    2: RESULTS_DIR / "stage2_degradation",
    3: RESULTS_DIR / "stage3_fft",
    4: RESULTS_DIR / "stage4_spatial",
    5: RESULTS_DIR / "stage5_frequency",
    6: RESULTS_DIR / "stage6_enhancement",
    7: RESULTS_DIR / "stage7_edges",
    8: RESULTS_DIR / "stage8_summary",
}

# Reproducibility
# Fixed seed => same noise and same PSNR/SSIM on every run.
RANDOM_SEED = 631

# Stage 1: reference images
# key: file name · loader: skimage.data function · label: plot title
# · note: rationale for the choice
REFERENCE_IMAGES = [
    {
        "key": "cameraman",
        "loader": "camera",
        "label": "Cameraman (strong edges & texture)",
        "note": "Classic test image: sharp edges (coat/sky), fine details "
                "(tripod, buildings) and texture (grass).",
    },
    {
        "key": "moon",
        "loader": "moon",
        "label": "Moon (smooth gradients, low contrast)",
        "note": "Smooth tonal transitions and a narrow histogram — ideal for "
                "showing the effect of histogram equalization / CLAHE.",
    },
]

# All images are normalized to this size (square, convenient for the FFT).
TARGET_SIZE = (512, 512)  # (width, height)

# Stage 2: noise parameters
GAUSS_SIGMAS = (10, 25)          # standard deviation in gray levels [0..255]
SALT_PEPPER_DENSITIES = (0.02, 0.10)

# Stage 3: frequency-domain analysis
# Fractions of the spectrum energy for which we look up the enclosing radius.
ENERGY_LEVELS = (0.90, 0.95, 0.99)
# Radius (normalized to Nyquist) above which we count "high frequencies".
HF_RADIUS_FRAC = 0.25

# Stage 4: spatial filters
# Parameter sweep, to expose the noise / blur trade-off.
#   Gaussian : (ksize, sigma) — sigma=0 => OpenCV derives it from ksize
#   Median   : ksize
#   Bilateral: (d, sigmaColor, sigmaSpace)
GAUSSIAN_PARAMS = ((3, 0), (5, 0), (7, 0), (9, 0))
MEDIAN_PARAMS = (3, 5, 7, 9)
BILATERAL_PARAMS = ((9, 25, 25), (9, 50, 50), (9, 75, 75), (9, 100, 50))

# Stage 5: frequency-domain filters
# Cutoff frequencies D0, normalized to Nyquist (same scale as Stage 3, so they
# compare directly with the r(SNR=1) column).
LPF_CUTOFFS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60, 0.75, 0.90)
# Butterworth orders: n=1 (close to Gaussian), n=2 (the classic choice),
# n=5 (steep transition, close to ideal). Each order is a separate filter,
# with its own cutoff sweep and its own PSNR/SSIM.
BUTTERWORTH_ORDERS = (1, 2, 5)
# The reference order, where a single value is needed.
BUTTERWORTH_ORDER = 2
BUTTERWORTH_ORDERS_DEMO = (1, 2, 5, 20)   # for the Gibbs phenomenon demo

# Stage 6: image enhancement
# Linear stretch: the percentiles mapped to 0 and 255 (not min/max, because a
# single outlier pixel would cancel the stretch).
STRETCH_PERCENTILES = (1.0, 99.0)
# CLAHE: (clipLimit, tileGridSize).
CLAHE_PARAMS = ((2.0, (8, 8)), (3.0, (8, 8)), (4.0, (8, 8)), (2.0, (16, 16)))
# The degradation used in the ordering experiment "enhancement before vs after
# restoration".
ENHANCE_CHAIN_VARIANT = "gauss_s25"

# Stage 7: edge detection
SOBEL_KSIZE = 3
# Binarization threshold of the gradient magnitude as a percentile (not an
# absolute value), so that it does not depend on the image contrast.
SOBEL_EDGE_PERCENTILE = 92.0
# Threshold pairs (low, high) for Canny — hysteresis.
CANNY_THRESHOLDS = ((50, 150), (100, 200), (150, 250))
# Position tolerance (in pixels) when comparing two edge maps.
EDGE_TOLERANCE = 2

# Stage 8: overall evaluation
# The enhancement technique used as the last step of the pipeline. CLAHE clip=2
# rather than HE, because it preserves the image structure without posterization.
FINAL_ENHANCEMENT = (2.0, (8, 8))   # (clipLimit, tileGridSize)

# Plot display
FIG_DPI = 150
