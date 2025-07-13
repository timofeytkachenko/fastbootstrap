"""Backward-compatible bootstrap module.

This module provides backward compatibility for the original bootstrap.py API
while importing from the new refactored modules. All original functions are
available with the same signatures.

Note: This module is maintained for backward compatibility. For new code,
consider using the improved API available in the main fastbootstrap package.
"""

import warnings
from typing import Callable, Dict, Iterator, Optional, Tuple, Union

import numpy as np
from matplotlib.axes import Axes
from scipy.stats import ttest_ind

from .compare_functions import difference, difference_of_mean


# Maintain original function signatures for backward compatibility
def estimate_confidence_interval(
    distribution: np.ndarray, bootstrap_conf_level: float = 0.95
) -> np.ndarray:
    """Estimate the confidence interval of a distribution using quantiles.

    Parameters
    ----------
    distribution : ndarray
        1D array containing the bootstrap distribution or any distribution of interest.
    bootstrap_conf_level : float, optional
        Confidence level, between 0 and 1. Default is 0.95 (95% confidence interval).

    Returns
    -------
    ndarray
        A 1D array [lower_bound, upper_bound] representing the confidence interval.
    """
    from .core import estimate_confidence_interval as _estimate_ci

    return _estimate_ci(distribution, bootstrap_conf_level)


def estimate_p_value(
    bootstrap_difference_distribution: np.ndarray, number_of_bootstrap_samples: int
) -> float:
    """Estimate the two-sided p-value from a bootstrap difference distribution.

    Parameters
    ----------
    bootstrap_difference_distribution : ndarray
        1D array containing the bootstrap difference distribution.
    number_of_bootstrap_samples : int
        Number of bootstrap samples used to generate the distribution.

    Returns
    -------
    float
        The estimated two-sided p-value.
    """
    from .core import estimate_p_value as _estimate_p_value

    return _estimate_p_value(
        bootstrap_difference_distribution, number_of_bootstrap_samples
    )


def estimate_bin_params(sample: np.ndarray) -> Tuple[float, int]:
    """Estimate histogram bin parameters using the Freedman-Diaconis rule.

    Parameters
    ----------
    sample : ndarray
        1D array containing observations.

    Returns
    -------
    tuple
        A tuple (bin_width, bin_count) where:

        - bin_width : float
            Width of each bin calculated using the Freedman-Diaconis rule.
        - bin_count : int
            Number of bins to use for the histogram.
    """
    from .core import estimate_bin_params as _estimate_bin_params

    return _estimate_bin_params(sample)


def jackknife_indices(control: np.ndarray) -> Iterator[np.ndarray]:
    """Generate jackknife indices for leave-one-out resampling.

    Parameters
    ----------
    control : ndarray
        1D array containing the sample data.

    Returns
    -------
    Iterator[ndarray]
        Generator yielding arrays of indices, each with one element removed.
        Each yielded array can be used to select elements for a jackknife sample.
    """
    from .core import jackknife_indices as _jackknife_indices

    return _jackknife_indices(control)


def bca(
    control: np.ndarray,
    bootstrap_distribution: np.ndarray,
    statistic: Callable = np.mean,
    bootstrap_conf_level: float = 0.95,
) -> np.ndarray:
    """Compute the BCa (bias-corrected and accelerated) confidence interval.

    Parameters
    ----------
    control : ndarray
        1D array containing the original sample data.
    bootstrap_distribution : ndarray
        1D array containing the bootstrap distribution of the statistic.
    statistic : callable, optional
        Function that computes the statistic of interest from a sample.
        Default is np.mean.
    bootstrap_conf_level : float, optional
        Confidence level, between 0 and 1. Default is 0.95 (95% confidence interval).

    Returns
    -------
    ndarray
        Array [lower_bound, upper_bound] representing the BCa confidence interval.
    """
    from .core import bca_confidence_interval

    return bca_confidence_interval(
        control, bootstrap_distribution, statistic, bootstrap_conf_level
    )


def bootstrap_resampling(
    sample_function: Callable,
    number_of_bootstrap_samples: int,
    seed: Optional[int] = None,
    n_jobs: int = -1,
) -> np.ndarray:
    """Perform bootstrap resampling with independent random generators per sample.

    Parameters
    ----------
    sample_function : callable
        Function that takes a NumPy Generator and returns a bootstrap statistic.
        Signature: sample_function(generator: np.random.Generator) -> float or array
    number_of_bootstrap_samples : int
        Number of bootstrap samples to generate.
    seed : int, optional
        Seed for reproducibility. If None, a random seed will be used.
    n_jobs : int, optional
        Number of parallel jobs to run. -1 means using all available cores.
        Default is -1.

    Returns
    -------
    ndarray
        Array of bootstrap statistics with shape (number_of_bootstrap_samples,).
    """
    from .core import bootstrap_resampling as _bootstrap_resampling

    return _bootstrap_resampling(
        sample_function, number_of_bootstrap_samples, seed, n_jobs
    )


def bootstrap_plot(
    bootstrap_distribution: np.ndarray,
    bootstrap_confidence_interval: np.ndarray,
    statistic: Optional[Union[str, Callable]] = None,
    title: str = "Bootstrap",
    two_sample_plot: bool = True,
) -> None:
    """Plot a bootstrap distribution with confidence interval markers.

    Parameters
    ----------
    bootstrap_distribution : ndarray
        1D array containing the bootstrap distribution.
    bootstrap_confidence_interval : ndarray
        1D array [lower_bound, upper_bound] representing the confidence interval.
    statistic : str or callable, optional
        Statistic name or function, used for labeling the x-axis.
        If a function is provided, the function name is used (with underscores
        replaced by spaces and words capitalized).
    title : str, optional
        Plot title. Default is 'Bootstrap'.
    two_sample_plot : bool, optional
        If True, adds a vertical line at 0 for two-sample difference visualization.
        Default is True.

    Returns
    -------
    None
        Displays a matplotlib plot and does not return a value.
    """
    from .visualization import bootstrap_plot as _bootstrap_plot

    _bootstrap_plot(
        bootstrap_distribution,
        bootstrap_confidence_interval,
        statistic,
        title,
        two_sample_plot,
    )


def two_sample_bootstrap(
    control: np.ndarray,
    treatment: np.ndarray,
    bootstrap_conf_level: float = 0.95,
    number_of_bootstrap_samples: int = 10000,
    sample_size: Optional[int] = None,
    statistic: Callable = difference_of_mean,
    return_distribution: bool = False,
    seed: Optional[int] = None,
    plot: bool = False,
) -> Dict[str, Union[float, np.ndarray, None]]:
    """Perform two-sample bootstrap analysis for comparing two samples.

    Parameters
    ----------
    control : ndarray
        1D array containing the control sample.
    treatment : ndarray
        1D array containing the treatment sample.
    bootstrap_conf_level : float, optional
        Confidence level, between 0 and 1. Default is 0.95 (95% confidence interval).
    number_of_bootstrap_samples : int, optional
        Number of bootstrap samples to generate. Default is 10000.
    sample_size : int, optional
        Per-sample size for resampling. If None, uses the full length of each sample.
    statistic : callable, optional
        Function that computes the statistic of interest from two samples.
        Should have signature: f(control_sample, treatment_sample) -> float.
        Default is difference_of_mean (computes the difference in means).
    return_distribution : bool, optional
        Whether to include the full bootstrap distribution in the results.
        Default is False.
    seed : int, optional
        Random seed for reproducibility. Default is None.
    plot : bool, optional
        If True, plots the bootstrap distribution. Default is False.

    Returns
    -------
    dict
        Dictionary containing:

        - 'p_value' : float
            The estimated two-sided p-value.
        - 'statistic_value' : float
            Median of the bootstrap difference distribution.
        - 'confidence_interval' : ndarray
            Array [lower_bound, upper_bound] representing the confidence interval.
        - 'distribution' : ndarray or None
            The full bootstrap difference distribution if return_distribution=True,
            otherwise None.
    """
    from .methods import two_sample_bootstrap as _two_sample_bootstrap

    return _two_sample_bootstrap(
        control,
        treatment,
        bootstrap_conf_level,
        number_of_bootstrap_samples,
        sample_size,
        statistic,
        return_distribution,
        seed,
        plot,
    )


def spotify_two_sample_bootstrap(
    control: np.ndarray,
    treatment: np.ndarray,
    number_of_bootstrap_samples: int = 10000,
    sample_size: Optional[int] = None,
    q1: float = 0.5,
    q2: Optional[float] = None,
    statistic: Callable[[np.ndarray, np.ndarray], float] = difference,
    bootstrap_conf_level: float = 0.95,
    return_distribution: bool = False,
    plot: bool = False,
) -> Dict[str, Union[float, np.ndarray, None]]:
    """Perform Spotify-style two-sample bootstrap for quantile comparisons.

    Parameters
    ----------
    control : ndarray
        1D array containing the control sample.
    treatment : ndarray
        1D array containing the treatment sample.
    number_of_bootstrap_samples : int, optional
        Number of bootstrap samples to generate. Default is 10000.
    sample_size : int, optional
        Per-sample size for resampling. If None, uses the full length of each sample.
    q1 : float, optional
        Quantile of interest for the control sample, between 0 and 1.
        Default is 0.5 (median).
    q2 : float, optional
        Quantile of interest for the treatment sample, between 0 and 1.
        If None, uses the same value as q1. Default is None.
    statistic : callable, optional
        Function to compare the quantiles from the two samples.
        Should have signature: f(control_quantile, treatment_quantile) -> float.
        Default is difference (returns control_quantile - treatment_quantile).
    bootstrap_conf_level : float, optional
        Confidence level, between 0 and 1. Default is 0.95 (95% confidence interval).
    return_distribution : bool, optional
        Whether to include the full bootstrap distribution in the results.
        Default is False.
    plot : bool, optional
        If True, plots the bootstrap distribution. Default is False.

    Returns
    -------
    dict
        Dictionary containing:

        - 'p_value' : float
            The estimated two-sided p-value.
        - 'statistic_value' : float
            The difference between the specified quantiles of the original samples.
        - 'confidence_interval' : ndarray
            Array [lower_bound, upper_bound] representing the confidence interval.
        - 'distribution' : ndarray or None
            The full bootstrap difference distribution if return_distribution=True,
            otherwise None.
    """
    from .methods import spotify_two_sample_bootstrap as _spotify_two_sample_bootstrap

    return _spotify_two_sample_bootstrap(
        control,
        treatment,
        number_of_bootstrap_samples,
        sample_size,
        q1,
        q2,
        statistic,
        bootstrap_conf_level,
        return_distribution,
        plot,
    )


def ab_test_simulation(
    control: np.ndarray,
    treatment: np.ndarray,
    number_of_experiments: int = 2000,
    stat_test: Callable = ttest_ind,
    n_jobs: int = -1,
) -> Dict[str, Union[float, np.ndarray]]:
    """Simulate multiple A/B tests with parallel p-value computation.

    Parameters
    ----------
    control : ndarray
        1D array containing the control sample.
    treatment : ndarray
        1D array containing the treatment sample.
    number_of_experiments : int, optional
        Number of simulated experiments to run. Default is 2000.
    stat_test : callable, optional
        Statistical test function that takes two arrays and returns a tuple
        (statistic, p_value). Default is scipy.stats.ttest_ind.
    n_jobs : int, optional
        Number of parallel jobs to run. -1 means using all available cores.
        Default is -1.

    Returns
    -------
    dict
        Dictionary containing:

        - 'p_value' : ndarray
            Array of p-values from all simulated experiments with shape
            (number_of_experiments,).
        - 'test_power' : float
            Proportion of experiments where p < 0.05, representing the test power.
        - 'auc' : float
            Area under the curve of the sorted p-values, a measure of the overall
            distribution of p-values.
    """
    from .simulation import ab_test_simulation as _ab_test_simulation

    result = _ab_test_simulation(
        control, treatment, number_of_experiments, stat_test, n_jobs
    )
    # Maintain backward compatibility with the key name
    result["p_value"] = result.pop("ab_p_values")
    return result


def plot_cdf(p_values: np.ndarray, label: str, ax: Axes, linewidth: float = 3) -> None:
    """Plot the empirical CDF (Cumulative Distribution Function) of p-values.

    Parameters
    ----------
    p_values : ndarray
        1D array of p-values.
    label : str
        Label for the plot legend.
    ax : matplotlib.axes.Axes
        Matplotlib Axes object on which to create the plot.
    linewidth : float, optional
        Line width for the CDF curve. Default is 3.

    Returns
    -------
    None
        Modifies the provided Axes object and does not return a value.
    """
    from .visualization import plot_cdf as _plot_cdf

    _plot_cdf(p_values, label, ax, linewidth)


def plot_summary(aa_p_values: np.ndarray, ab_p_values: np.ndarray) -> None:
    """Create comprehensive summary plots for A/A and A/B test p-values.

    Parameters
    ----------
    aa_p_values : ndarray
        1D array of p-values from an A/A test (null hypothesis).
    ab_p_values : ndarray
        1D array of p-values from an A/B test (alternative hypothesis).

    Returns
    -------
    None
        Displays a matplotlib figure with multiple subplots and does not return a value.
    """
    from .visualization import plot_summary as _plot_summary

    _plot_summary(aa_p_values, ab_p_values)


def quantile_bootstrap_plot(
    control: np.ndarray,
    treatment: np.ndarray,
    n_step: int = 20,
    q1: float = 0.01,
    q2: float = 0.99,
    bootstrap_conf_level: float = 0.95,
    statistic: Callable[[np.ndarray, np.ndarray], np.ndarray] = difference,
    correction: str = "bh",
) -> None:
    """Create an interactive quantile-by-quantile comparison with confidence bands.

    Parameters
    ----------
    control : ndarray
        1D array containing the control sample.
    treatment : ndarray
        1D array containing the treatment sample.
    n_step : int, optional
        Number of quantiles to compare between q1 and q2. Default is 20.
    q1 : float, optional
        Lower quantile bound to start comparison, between 0 and 1. Default is 0.01.
    q2 : float, optional
        Upper quantile bound to end comparison, between 0 and 1. Default is 0.99.
    bootstrap_conf_level : float, optional
        Base confidence level, between 0 and 1. Default is 0.95.
        This will be adjusted for multiple comparisons according to the correction method.
    statistic : callable, optional
        Function to compare quantiles, with signature: f(x, y) -> float.
        Default is difference (returns x - y).
    correction : str, optional
        Method for multiple testing correction: 'bonferroni' or 'bh' (Benjamini-Hochberg).
        Default is 'bh'.

    Returns
    -------
    None
        Displays an interactive Plotly figure and does not return a value.
    """
    from .visualization import quantile_bootstrap_plot as _quantile_bootstrap_plot

    _quantile_bootstrap_plot(
        control, treatment, n_step, q1, q2, bootstrap_conf_level, statistic, correction
    )


def one_sample_bootstrap(
    control: np.ndarray,
    bootstrap_conf_level: float = 0.95,
    number_of_bootstrap_samples: int = 10000,
    sample_size: Optional[int] = None,
    statistic: Callable = np.mean,
    method: str = "percentile",
    return_distribution: bool = False,
    seed: Optional[int] = None,
    plot: bool = False,
) -> Dict[str, Union[float, np.ndarray, None]]:
    """Perform one-sample bootstrap to estimate confidence intervals for a statistic.

    Parameters
    ----------
    control : ndarray
        1D array containing the sample data.
    bootstrap_conf_level : float, optional
        Confidence level, between 0 and 1. Default is 0.95 (95% confidence interval).
    number_of_bootstrap_samples : int, optional
        Number of bootstrap samples to generate. Default is 10000.
    sample_size : int, optional
        Size of each bootstrap sample. If None, uses the full sample size.
    statistic : callable, optional
        Function to compute the statistic of interest, with signature: f(x) -> float.
        Default is np.mean.
    method : str, optional
        Bootstrap method for confidence interval estimation. Options:
        - 'percentile': Simple percentile method
        - 'bca': Bias-corrected and accelerated method
        - 'basic': Basic bootstrap method
        - 'studentized': Studentized bootstrap method
        Default is 'percentile'.
    return_distribution : bool, optional
        Whether to include the full bootstrap distribution in the results.
        Default is False.
    seed : int, optional
        Random seed for reproducibility. Default is None.
    plot : bool, optional
        If True, plots the bootstrap distribution. Default is False.

    Returns
    -------
    dict
        Dictionary containing:

        - 'statistic_value' : float
            Median of the bootstrap distribution.
        - 'confidence_interval' : ndarray
            Array [lower_bound, upper_bound] representing the confidence interval.
        - 'distribution' : ndarray or None
            Full bootstrap distribution if return_distribution=True, otherwise None.
    """
    from .methods import one_sample_bootstrap as _one_sample_bootstrap

    return _one_sample_bootstrap(
        control,
        bootstrap_conf_level,
        number_of_bootstrap_samples,
        sample_size,
        statistic,
        method,
        return_distribution,
        seed,
        plot,
    )


def spotify_one_sample_bootstrap(
    sample: np.ndarray,
    sample_size: Optional[int] = None,
    q: float = 0.5,
    bootstrap_conf_level: float = 0.95,
    number_of_bootstrap_samples: int = 10000,
    return_distribution: bool = False,
    plot=False,
) -> Dict[str, Union[float, np.ndarray, None]]:
    """Perform Spotify-style one-sample bootstrap for quantile estimation.

    Parameters
    ----------
    sample : ndarray
        1D array containing the sample data.
    sample_size : int, optional
        Size of each bootstrap sample. If None, uses the full sample size.
    q : float, optional
        Quantile of interest, between 0 and 1. Default is 0.5 (median).
    bootstrap_conf_level : float, optional
        Confidence level, between 0 and 1. Default is 0.95 (95% confidence interval).
    number_of_bootstrap_samples : int, optional
        Number of bootstrap samples to generate. Default is 10000.
    return_distribution : bool, optional
        Whether to include the full bootstrap distribution in the results.
        Default is False.
    plot : bool, optional
        If True, plots the bootstrap distribution. Default is False.

    Returns
    -------
    dict
        Dictionary containing:

        - 'statistic_value' : float
            The estimated quantile from the bootstrap distribution.
        - 'confidence_interval' : ndarray
            Array [lower_bound, upper_bound] representing the confidence interval.
        - 'distribution' : ndarray or None
            Full bootstrap distribution if return_distribution=True, otherwise None.
    """
    from .methods import spotify_one_sample_bootstrap as _spotify_one_sample_bootstrap

    return _spotify_one_sample_bootstrap(
        sample,
        sample_size,
        q,
        bootstrap_conf_level,
        number_of_bootstrap_samples,
        return_distribution,
        plot,
    )


def poisson_bootstrap(
    control: np.ndarray, treatment: np.ndarray, number_of_bootstrap_samples: int = 10000
) -> float:
    """Perform Poisson bootstrap for comparing aggregated values between two samples.

    Parameters
    ----------
    control : ndarray
        1D array containing the control sample.
    treatment : ndarray
        1D array containing the treatment sample.
    number_of_bootstrap_samples : int, optional
        Number of bootstrap samples to generate. Default is 10000.

    Returns
    -------
    float
        The estimated two-sided p-value.
    """
    from .methods import poisson_bootstrap as _poisson_bootstrap

    return _poisson_bootstrap(control, treatment, number_of_bootstrap_samples)


def bootstrap(
    control: np.ndarray,
    treatment: Optional[np.ndarray] = None,
    bootstrap_conf_level: float = 0.95,
    number_of_bootstrap_samples: int = 10000,
    sample_size: Optional[int] = None,
    statistic: Callable = np.mean,
    method: str = "percentile",  # For one-sample
    spotify_style: bool = False,
    q: Union[float, Tuple[float, float]] = 0.5,  # Single quantile or (q1, q2) pair
    return_distribution: bool = False,
    seed: Optional[int] = None,
    plot: bool = False,
) -> Dict[str, Union[float, np.ndarray, None]]:
    """Unified bootstrap function that automatically selects appropriate method based on inputs.

    Parameters
    ----------
    control : ndarray
        1D array containing the control/first sample.
    treatment : ndarray, optional
        1D array containing the treatment/second sample for two-sample methods.
        If None, one-sample bootstrap is performed on control. Default is None.
    bootstrap_conf_level : float, optional
        Confidence level, between 0 and 1. Default is 0.95 (95% confidence interval).
    number_of_bootstrap_samples : int, optional
        Number of bootstrap samples to generate. Default is 10000.
    sample_size : int, optional
        Size for resampling. If None, uses the full sample size(s).
    statistic : callable, optional
        Function computing the statistic of interest. Default is np.mean.
        For one-sample: f(sample) -> float
        For two-sample: f(control_sample, treatment_sample) -> float
    method : str, optional
        One-sample bootstrap method: 'percentile', 'bca', 'basic', or 'studentized'.
        Only used when treatment is None. Default is 'percentile'.
    spotify_style : bool, optional
        Whether to use Spotify-style bootstrap (quantile-based). Default is False.
    q : float or tuple of float, optional
        Quantile(s) of interest for Spotify-style bootstrap. Default is 0.5 (median).
        For one-sample: single value between 0 and 1
        For two-sample: single value or tuple (q1, q2) for control and treatment
    return_distribution : bool, optional
        Whether to include the full bootstrap distribution. Default is False.
    seed : int, optional
        Random seed for reproducibility. Default is None.
    plot : bool, optional
        Whether to plot the bootstrap distribution. Default is False.

    Returns
    -------
    dict
        Dictionary containing:

        - 'statistic_value' : float
            The primary statistic value (depends on method).
        - 'confidence_interval' : ndarray
            Array [lower_bound, upper_bound] representing the confidence interval.
        - 'distribution' : ndarray or None
            Full bootstrap distribution if return_distribution=True, otherwise None.
        - 'p_value' : float, only for two-sample methods
            The estimated two-sided p-value.

    Notes
    -----
    This function automatically selects the appropriate bootstrap method:
    1. If treatment is None:
       a. If spotify_style is True: spotify_one_sample_bootstrap
       b. Otherwise: one_sample_bootstrap
    2. If treatment is provided:
       a. If spotify_style is True: spotify_two_sample_bootstrap
       b. Otherwise: two_sample_bootstrap
    """
    from .methods import bootstrap as _bootstrap

    return _bootstrap(
        control,
        treatment,
        bootstrap_conf_level,
        number_of_bootstrap_samples,
        sample_size,
        statistic,
        method,
        spotify_style,
        q,
        return_distribution,
        seed,
        plot,
    )


# Show deprecation warning
warnings.warn(
    "Importing from fastbootstrap.bootstrap is deprecated. "
    "Use 'import fastbootstrap as fb' instead for better performance and new features.",
    DeprecationWarning,
    stacklevel=2,
)
