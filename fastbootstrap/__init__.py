"""FastBootstrap: Fast Python implementation of statistical bootstrap.

This package provides efficient bootstrap methods for statistical analysis
including one-sample and two-sample bootstrap, confidence intervals,
hypothesis testing, and specialized methods like Spotify-style bootstrap.

Key Features:
- Multiple bootstrap methods (percentile, BCa, basic, studentized)
- Parallel processing for performance
- Comprehensive error handling and validation
- Visualization utilities for results
- Jupyter notebook integration

Examples:
---------
>>> import fastbootstrap as fb
>>> import numpy as np

>>> # One-sample bootstrap
>>> sample = np.random.normal(0, 1, 100)
>>> result = fb.bootstrap(sample)
>>> print(f"Confidence interval: {result['confidence_interval']}")

>>> # Two-sample bootstrap
>>> control = np.random.normal(0, 1, 100)
>>> treatment = np.random.normal(0.5, 1, 100)
>>> result = fb.bootstrap(control, treatment)
>>> print(f"P-value: {result['p_value']:.4f}")
"""

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
    ALPHA_THRESHOLD,
    BOOTSTRAP_METHODS,
    CORRECTION_METHODS,
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_N_JOBS,
    DEFAULT_SEED,
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
    ConvergenceError,
    FastBootstrapError,
    InsufficientDataError,
    MemoryError,
    NumericalError,
    PlottingError,
    ValidationError,
)

# Core bootstrap functions
from .methods import (
    bootstrap,
    one_sample_bootstrap,
    poisson_bootstrap,
    spotify_one_sample_bootstrap,
    spotify_two_sample_bootstrap,
    two_sample_bootstrap,
)

# Simulation functions
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

# Public API - functions that users should primarily use
__all__ = [
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
    # Comparison functions
    "difference_of_mean",
    "percent_difference_of_mean",
    "percent_change_of_mean",
    "difference_of_std",
    "percent_difference_of_std",
    "percent_change_of_std",
    "difference_of_median",
    "percent_difference_of_median",
    "percent_change_of_median",
    "difference",
    "percent_change",
    "percent_difference",
    # Visualization functions
    "bootstrap_plot",
    "plot_summary",
    "quantile_bootstrap_plot",
    # Simulation functions
    "ab_test_simulation",
    "aa_test_simulation",
    "power_analysis",
    "bootstrap_power_analysis",
    # Utility functions
    "display_significance_result",
    "display_bootstrap_summary",
    "calculate_effect_size",
    "interpret_effect_size",
    "create_sample_data",
    "format_p_value",
    # Constants
    "DEFAULT_BOOTSTRAP_SAMPLES",
    "DEFAULT_CONFIDENCE_LEVEL",
    "ALPHA_THRESHOLD",
    "BOOTSTRAP_METHODS",
    # Exceptions
    "FastBootstrapError",
    "ValidationError",
    "InsufficientDataError",
    "NumericalError",
    "BootstrapMethodError",
    # Legacy functions
    "display_markdown_cell_by_significance",
]


# Module-level configuration
def set_default_bootstrap_samples(n_samples: int) -> None:
    """Set the default number of bootstrap samples globally.

    Parameters
    ----------
    n_samples : int
        Number of bootstrap samples to use as default.

    Notes
    -----
    This affects the default value used in bootstrap functions.
    Individual function calls can still override this value.
    """
    import fastbootstrap.constants as constants

    constants.DEFAULT_BOOTSTRAP_SAMPLES = n_samples


def set_default_confidence_level(level: float) -> None:
    """Set the default confidence level globally.

    Parameters
    ----------
    level : float
        Confidence level between 0 and 1.

    Notes
    -----
    This affects the default value used in bootstrap functions.
    Individual function calls can still override this value.
    """
    import fastbootstrap.constants as constants

    constants.DEFAULT_CONFIDENCE_LEVEL = level


def get_config() -> dict:
    """Get current configuration settings.

    Returns
    -------
    dict
        Dictionary containing current configuration values.
    """
    from . import constants

    return {
        "default_bootstrap_samples": constants.DEFAULT_BOOTSTRAP_SAMPLES,
        "default_confidence_level": constants.DEFAULT_CONFIDENCE_LEVEL,
        "default_n_jobs": constants.DEFAULT_N_JOBS,
        "alpha_threshold": constants.ALPHA_THRESHOLD,
    }


# Add configuration functions to __all__
__all__.extend(
    ["set_default_bootstrap_samples", "set_default_confidence_level", "get_config"]
)

# Numpy is already imported above for convenience
__all__.append("np")
