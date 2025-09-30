"""FastBootstrap: Fast Python implementation of statistical bootstrap.

This package provides efficient bootstrap methods for statistical analysis
including one-sample and two-sample bootstrap, confidence intervals,
hypothesis testing, and specialized methods like Spotify-style bootstrap.

Key Features
------------
- **Multiple bootstrap methods**: Percentile, BCa, Basic, Studentized, Spotify-style, Poisson
- **High performance**: Parallel processing with joblib, optimized NumPy operations
- **Smart batch sizing**: Intelligent auto-optimization for 5-30% performance gains
- **Comprehensive statistics**: Confidence intervals, p-values, effect sizes, power analysis
- **Flexible API**: Unified interface with method auto-selection
- **Rich visualizations**: Built-in plotting with matplotlib and plotly
- **Production ready**: Extensive error handling, type hints, comprehensive testing

Quick Start
-----------
>>> import fastbootstrap as fb
>>> import numpy as np
>>>
>>> # One-sample bootstrap
>>> sample = np.random.normal(0, 1, 100)
>>> result = fb.bootstrap(sample)
>>> print(f"CI: {result['confidence_interval']}")
>>>
>>> # Two-sample bootstrap with smart batch sizing
>>> control = np.random.normal(0, 1, 100)
>>> treatment = np.random.normal(0.5, 1, 100)
>>> result = fb.bootstrap(control, treatment, batch_size='smart')
>>> print(f"P-value: {result['p_value']:.4f}")

Performance
-----------
- Spotify methods: 30M+ samples/sec (ultra-fast quantile analysis)
- Standard bootstrap: ~65K samples/sec (comprehensive statistics)
- Smart batch sizing: 5-30% speedup for large-scale resampling
- Parallel processing: Automatic multi-core utilization

See Also
--------
- Documentation: https://github.com/timofeytkachenko/fastbootstrap
- Examples: bootstrap_experiment.ipynb
- Performance: README.md benchmarks section
"""

__version__ = "1.8.0"
__author__ = "Timofey Tkachenko"
__email__ = "timofey_tkachenko@pm.me"
__license__ = "MIT"
__url__ = "https://github.com/timofeytkachenko/fastbootstrap"

import numpy as np

# Comparison functions
from .compare_functions import (
    difference,
    difference_of_mean,
    difference_of_median,
    difference_of_std,
    percent_change,
    percent_change_of_mean,
    percent_change_of_median,
    percent_change_of_std,
    percent_difference,
    percent_difference_of_mean,
    percent_difference_of_median,
    percent_difference_of_std,
)

# Constants
from .constants import (
    # Default parameters
    ALPHA_THRESHOLD,
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_N_JOBS,
    DEFAULT_SEED,
    DEFAULT_QUANTILE,
    DEFAULT_BOOTSTRAP_METHOD,
    DEFAULT_CORRECTION_METHOD,
    # Bootstrap and correction methods
    BOOTSTRAP_METHODS,
    CORRECTION_METHODS,
    # Confidence level bounds
    MIN_CONFIDENCE_LEVEL,
    MAX_CONFIDENCE_LEVEL,
    # Numerical constants
    EPSILON,
    MIN_SAMPLE_SIZE,
    MAX_BOOTSTRAP_SAMPLES,
    # Batch size configuration
    DEFAULT_BATCH_SIZE,
    BATCH_SIZE_THRESHOLD_SMALL,
    BATCH_SIZE_THRESHOLD_MEDIUM,
    BATCH_SIZE_THRESHOLD_LARGE,
    BATCH_SIZE_SMALL,
    BATCH_SIZE_MEDIUM,
    BATCH_SIZE_LARGE,
    BATCH_SIZE_MASSIVE,
    # Memory thresholds
    MEMORY_LOW_THRESHOLD,
    MEMORY_MODERATE_THRESHOLD,
    LARGE_SAMPLE_THRESHOLD,
)

# Core statistical functions
from .core import (
    bca_confidence_interval,
    bootstrap_resampling,
    estimate_bin_params,
    estimate_confidence_interval,
    estimate_p_value,
    jackknife_indices,
)

# Exceptions
from .exceptions import (
    BootstrapMethodError,
    FastBootstrapError,
    InsufficientDataError,
    NumericalError,
    PlottingError,
    ValidationError,
)

# Core bootstrap methods
from .methods import (
    bootstrap,
    one_sample_bootstrap,
    poisson_bootstrap,
    spotify_one_sample_bootstrap,
    spotify_two_sample_bootstrap,
    two_sample_bootstrap,
)

# Simulation and power analysis
from .simulation import (
    aa_test_simulation,
    ab_test_simulation,
    power_analysis,
)

# Utility functions
from .utils import (
    display_bootstrap_summary,
    display_markdown_cell_by_significance,
    display_significance_result,
    validate_arrays_compatible,
)

# Visualization functions
from .visualization import (
    bootstrap_plot,
    plot_cdf,
    plot_summary,
    quantile_bootstrap_plot,
)

# Public API - primary interface for users
__all__ = [
    # Version and metadata
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__url__",
    # Main bootstrap functions
    "bootstrap",
    "one_sample_bootstrap",
    "two_sample_bootstrap",
    "spotify_one_sample_bootstrap",
    "spotify_two_sample_bootstrap",
    "poisson_bootstrap",
    # Core statistical functions
    "bootstrap_resampling",
    "estimate_confidence_interval",
    "estimate_p_value",
    "bca_confidence_interval",
    "estimate_bin_params",
    "jackknife_indices",
    # Comparison functions
    "difference",
    "difference_of_mean",
    "difference_of_median",
    "difference_of_std",
    "percent_change",
    "percent_change_of_mean",
    "percent_change_of_median",
    "percent_change_of_std",
    "percent_difference",
    "percent_difference_of_mean",
    "percent_difference_of_median",
    "percent_difference_of_std",
    # Visualization functions
    "bootstrap_plot",
    "plot_summary",
    "plot_cdf",
    "quantile_bootstrap_plot",
    # Simulation and power analysis
    "ab_test_simulation",
    "aa_test_simulation",
    "power_analysis",
    # Utility functions
    "display_significance_result",
    "display_bootstrap_summary",
    "display_markdown_cell_by_significance",
    "validate_arrays_compatible",
    # Configuration functions
    "set_default_bootstrap_samples",
    "set_default_confidence_level",
    "get_config",
    # Constants - Default parameters
    "DEFAULT_BOOTSTRAP_SAMPLES",
    "DEFAULT_CONFIDENCE_LEVEL",
    "DEFAULT_N_JOBS",
    "DEFAULT_SEED",
    "DEFAULT_QUANTILE",
    "DEFAULT_BOOTSTRAP_METHOD",
    "DEFAULT_CORRECTION_METHOD",
    "ALPHA_THRESHOLD",
    # Constants - Methods
    "BOOTSTRAP_METHODS",
    "CORRECTION_METHODS",
    # Constants - Bounds
    "MIN_CONFIDENCE_LEVEL",
    "MAX_CONFIDENCE_LEVEL",
    "MIN_SAMPLE_SIZE",
    "MAX_BOOTSTRAP_SAMPLES",
    # Constants - Numerical
    "EPSILON",
    # Constants - Batch sizing
    "DEFAULT_BATCH_SIZE",
    "BATCH_SIZE_THRESHOLD_SMALL",
    "BATCH_SIZE_THRESHOLD_MEDIUM",
    "BATCH_SIZE_THRESHOLD_LARGE",
    "BATCH_SIZE_SMALL",
    "BATCH_SIZE_MEDIUM",
    "BATCH_SIZE_LARGE",
    "BATCH_SIZE_MASSIVE",
    # Constants - Memory thresholds
    "MEMORY_LOW_THRESHOLD",
    "MEMORY_MODERATE_THRESHOLD",
    "LARGE_SAMPLE_THRESHOLD",
    # Exceptions
    "FastBootstrapError",
    "ValidationError",
    "InsufficientDataError",
    "NumericalError",
    "BootstrapMethodError",
    "PlottingError",
    # NumPy for convenience
    "np",
]


# Module-level configuration functions
def set_default_bootstrap_samples(n_samples: int) -> None:
    """Set the default number of bootstrap samples globally.

    Parameters
    ----------
    n_samples : int
        Number of bootstrap samples to use as default.
        Must be positive integer > 0.

    Raises
    ------
    ValueError
        If n_samples is not positive.

    Notes
    -----
    This affects the default value used in all bootstrap functions.
    Individual function calls can still override this value using the
    `number_of_bootstrap_samples` parameter.

    Time complexity: O(1)
    Space complexity: O(1)

    Examples
    --------
    >>> import fastbootstrap as fb
    >>> fb.set_default_bootstrap_samples(20000)
    >>> # Now all bootstrap calls use 20000 samples by default
    >>> result = fb.bootstrap(sample)  # Uses 20000 samples
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    import fastbootstrap.constants as constants

    constants.DEFAULT_BOOTSTRAP_SAMPLES = n_samples


def set_default_confidence_level(level: float) -> None:
    """Set the default confidence level globally.

    Parameters
    ----------
    level : float
        Confidence level between 0 and 1 (e.g., 0.95 for 95% confidence).

    Raises
    ------
    ValueError
        If level is not between 0 and 1.

    Notes
    -----
    This affects the default value used in all bootstrap functions.
    Individual function calls can still override this value using the
    `bootstrap_conf_level` parameter.

    Common confidence levels:
    - 0.90 (90%)
    - 0.95 (95%, default)
    - 0.99 (99%)

    Time complexity: O(1)
    Space complexity: O(1)

    Examples
    --------
    >>> import fastbootstrap as fb
    >>> fb.set_default_confidence_level(0.99)
    >>> # Now all bootstrap calls use 99% confidence by default
    >>> result = fb.bootstrap(sample)  # Uses 99% CI
    """
    if not (0 < level < 1):
        raise ValueError(f"level must be between 0 and 1, got {level}")

    import fastbootstrap.constants as constants

    constants.DEFAULT_CONFIDENCE_LEVEL = level


def get_config() -> dict[str, int | float]:
    """Get current configuration settings.

    Returns
    -------
    dict[str, int | float]
        Dictionary containing current configuration values:
        - 'default_bootstrap_samples': int
            Default number of bootstrap samples
        - 'default_confidence_level': float
            Default confidence level (0-1)
        - 'default_n_jobs': int
            Default number of parallel jobs (-1 = all cores)
        - 'alpha_threshold': float
            Default significance threshold (typically 0.05)

    Examples
    --------
    >>> import fastbootstrap as fb
    >>> config = fb.get_config()
    >>> print(f"Bootstrap samples: {config['default_bootstrap_samples']}")
    >>> print(f"Confidence level: {config['default_confidence_level']}")
    >>> print(f"Parallel jobs: {config['default_n_jobs']}")
    >>> print(f"Alpha threshold: {config['alpha_threshold']}")

    Notes
    -----
    This function is useful for:
    - Debugging configuration issues
    - Verifying settings after using set_default_* functions
    - Documenting analysis parameters
    - Ensuring reproducibility

    Time complexity: O(1)
    Space complexity: O(1)
    """
    from . import constants

    return {
        "default_bootstrap_samples": constants.DEFAULT_BOOTSTRAP_SAMPLES,
        "default_confidence_level": constants.DEFAULT_CONFIDENCE_LEVEL,
        "default_n_jobs": constants.DEFAULT_N_JOBS,
        "alpha_threshold": constants.ALPHA_THRESHOLD,
    }


# Convenience function for quick version check
def version_info() -> dict[str, str]:
    """Get version and package information.

    Returns
    -------
    dict[str, str]
        Dictionary with version, author, license, and URL.

    Examples
    --------
    >>> import fastbootstrap as fb
    >>> info = fb.version_info()
    >>> print(f"FastBootstrap v{info['version']}")
    """
    return {
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "license": __license__,
        "url": __url__,
    }


__all__.append("version_info")
