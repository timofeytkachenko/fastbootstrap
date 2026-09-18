"""Constants for fastbootstrap package.

This module centralizes all magic numbers, default values, and configuration
constants used throughout the package.
"""

from typing import Final

# Default bootstrap parameters
DEFAULT_BOOTSTRAP_SAMPLES: Final[int] = 10000
DEFAULT_CONFIDENCE_LEVEL: Final[float] = 0.95
DEFAULT_SEED: Final[int] = 42
DEFAULT_N_JOBS: Final[int] = -1

# Statistical constants
ALPHA_THRESHOLD: Final[float] = 0.05
DEFAULT_QUANTILE: Final[float] = 0.5  # Median
MIN_CONFIDENCE_LEVEL: Final[float] = 0.01
MAX_CONFIDENCE_LEVEL: Final[float] = 0.99

# Visualization constants
DEFAULT_PLOT_WIDTH: Final[int] = 800
DEFAULT_PLOT_HEIGHT: Final[int] = 600
DEFAULT_LINE_WIDTH: Final[float] = 3.0
DEFAULT_BIN_COUNT: Final[int] = 50
SIGNIFICANCE_COLOR: Final[str] = "red"
MEDIAN_COLOR: Final[str] = "black"
NULL_LINE_COLOR: Final[str] = "white"

# Quantile bounds
DEFAULT_QUANTILE_LOWER: Final[float] = 0.01
DEFAULT_QUANTILE_UPPER: Final[float] = 0.99
DEFAULT_QUANTILE_STEPS: Final[int] = 20

# Multiple testing correction methods
CORRECTION_METHODS: Final[tuple[str, ...]] = ("bonferroni", "bh")
DEFAULT_CORRECTION_METHOD: Final[str] = "bh"

# Bootstrap methods
BOOTSTRAP_METHODS: Final[tuple[str, ...]] = (
    "percentile",
    "bca",
    "basic",
    "studentized",
)
DEFAULT_BOOTSTRAP_METHOD: Final[str] = "percentile"

# Numerical constants
EPSILON: Final[float] = 1e-10
MIN_SAMPLE_SIZE: Final[int] = 2
# Advisory only: NOT enforced anywhere (workloads of 1M+ samples are supported
# and benchmarked in the README). Kept for backward compatibility of the
# public API; do not use it to reject inputs.
MAX_BOOTSTRAP_SAMPLES: Final[int] = 100000

# Performance constants
DEFAULT_BATCH_SIZE: Final[int] = 1000
MEMORY_LIMIT_MB: Final[int] = 1000

# Smart batch size thresholds (inclusive right edges, by number of bootstrap samples)
BATCH_SIZE_THRESHOLD_SMALL: Final[int] = 10_000
BATCH_SIZE_THRESHOLD_MEDIUM: Final[int] = 100_000
BATCH_SIZE_THRESHOLD_LARGE: Final[int] = 500_000

# Smart batch size values (retuned from a batch-size sweep on a 16-core M4 Max,
# see README "Batch-Size Sweep"; the pipeline is largely parent-bound, so larger
# batches amortise dispatch overhead while the load-balancing cap keeps small N
# safe on machines with many workers).
BATCH_SIZE_SMALL: Final[int] = 256  # For N <= 10K samples
BATCH_SIZE_MEDIUM: Final[int] = 512  # For 10K < N <= 100K samples
BATCH_SIZE_LARGE: Final[int] = 1000  # For 100K < N <= 500K samples
BATCH_SIZE_MASSIVE: Final[int] = 1000  # For N > 500K samples
# LARGE and MASSIVE currently coincide on purpose: the 16-worker sweep found
# 1000 optimal for every N > 100K. The 500K edge is kept so the two buckets can
# diverge again later without an API change. Retune both, or neither.

# Memory constraint thresholds (GB)
MEMORY_LOW_THRESHOLD: Final[float] = 4.0
MEMORY_MODERATE_THRESHOLD: Final[float] = 8.0

# Sample size threshold for batch size adjustment
LARGE_SAMPLE_THRESHOLD: Final[int] = 100_000

# Smart batch sizing bounds and tuning knobs
MIN_BATCH_FLOOR: Final[int] = 16  # Absolute lower floor for any batch size
MIN_BATCHES_PER_WORKER: Final[int] = 4  # Load-balancing target (>=4 chunks/worker)
SAMPLE_COMPLEXITY_DIVISOR: Final[int] = 2  # Halving factor for large samples
LOW_MEM_BATCH_CAP: Final[int] = 64  # Cap when RAM < MEMORY_LOW_THRESHOLD

# Memory-aware sizing parameters
MEM_FRACTION: Final[float] = 0.25  # Fraction of free RAM budgeted per call
DEFAULT_DTYPE_BYTES: Final[int] = 8  # Bytes per element (float64 default)
BYTES_PER_GB: Final[int] = 1024**3  # GiB-to-bytes conversion factor
INDEX_BYTES: Final[int] = 8  # int64 index array created by Generator.choice

# Default resample width assumed by smart batch sizing when no hint is given
DEFAULT_SAMPLE_SIZE_HINT: Final[int] = 1000

# Lookup tables for bisect-based base-batch selection.
# bisect_left(BATCH_SIZE_EDGES, N) maps N to its bucket index in BATCH_SIZE_BASES
# (edges are inclusive right boundaries; see README heuristic table).
BATCH_SIZE_EDGES: Final[tuple[int, ...]] = (
    BATCH_SIZE_THRESHOLD_SMALL,
    BATCH_SIZE_THRESHOLD_MEDIUM,
    BATCH_SIZE_THRESHOLD_LARGE,
)
BATCH_SIZE_BASES: Final[tuple[int, ...]] = (
    BATCH_SIZE_SMALL,
    BATCH_SIZE_MEDIUM,
    BATCH_SIZE_LARGE,
    BATCH_SIZE_MASSIVE,
)

# BCa method thresholds
JACKKNIFE_PARALLEL_THRESHOLD: Final[int] = (
    1000  # Minimum sample size for parallel jackknife
)

# Error messages
ERROR_MESSAGES: Final[dict[str, str]] = {
    "invalid_confidence_level": "Confidence level must be between 0 and 1",
    "invalid_bootstrap_samples": "Number of bootstrap samples must be a positive integer",
    "invalid_sample_size": f"Sample size must be >= {MIN_SAMPLE_SIZE}",
    "invalid_method": "Invalid method. Choose from: {methods}",
    "empty_array": "Input array cannot be empty",
    "mismatched_arrays": "Input arrays must have compatible shapes",
    "invalid_quantile": "Quantile must be between 0 and 1",
    "division_by_zero": "Division by zero encountered in calculation",
    "insufficient_data": "Insufficient data for reliable bootstrap estimation",
}

# Jupyter notebook styling
JUPYTER_STYLES: Final[dict[str, str]] = {
    "success": '<div class="alert alert-block alert-success">Difference is significant (p-value < 0.05)</div>',
    "warning": '<div class="alert alert-block alert-danger">Difference is non-significant (p-value >= 0.05)</div>',
}
