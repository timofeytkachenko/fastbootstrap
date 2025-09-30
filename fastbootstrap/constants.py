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
MAX_BOOTSTRAP_SAMPLES: Final[int] = 100000

# Performance constants
DEFAULT_BATCH_SIZE: Final[int] = 1000
MEMORY_LIMIT_MB: Final[int] = 1000

# Smart batch size thresholds (based on number of bootstrap samples)
BATCH_SIZE_THRESHOLD_SMALL: Final[int] = 10_000
BATCH_SIZE_THRESHOLD_MEDIUM: Final[int] = 100_000
BATCH_SIZE_THRESHOLD_LARGE: Final[int] = 500_000

# Smart batch size values
BATCH_SIZE_SMALL: Final[int] = 128  # For < 10K samples
BATCH_SIZE_MEDIUM: Final[int] = 256  # For 10K-100K samples
BATCH_SIZE_LARGE: Final[int] = 512  # For 100K-500K samples
BATCH_SIZE_MASSIVE: Final[int] = 1000  # For > 500K samples

# Memory constraint thresholds (GB)
MEMORY_LOW_THRESHOLD: Final[float] = 4.0
MEMORY_MODERATE_THRESHOLD: Final[float] = 8.0

# Sample size threshold for batch size adjustment
LARGE_SAMPLE_THRESHOLD: Final[int] = 100_000

# Error messages
ERROR_MESSAGES: Final[dict[str, str]] = {
    "invalid_confidence_level": "Confidence level must be between 0 and 1",
    "invalid_bootstrap_samples": f"Number of bootstrap samples must be positive and <= {MAX_BOOTSTRAP_SAMPLES}",
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
