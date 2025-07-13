"""Bootstrap methods for statistical analysis.

This module provides various bootstrap methods including one-sample, two-sample,
Spotify-style bootstrap, and specialized methods like BCa and Poisson bootstrap.
"""

from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from numpy.random import binomial
from scipy.stats import binom

from .compare_functions import difference, difference_of_mean
from .constants import (
    BOOTSTRAP_METHODS,
    DEFAULT_BOOTSTRAP_METHOD,
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_QUANTILE,
    DEFAULT_SEED,
    ERROR_MESSAGES,
)
from .core import (
    _validate_bootstrap_params,
    _validate_sample_array,
    bca_confidence_interval,
    bootstrap_resampling,
    estimate_confidence_interval,
    estimate_p_value,
)
from .exceptions import BootstrapMethodError, ValidationError
from .visualization import bootstrap_plot


def one_sample_bootstrap(
    sample: npt.NDArray[np.floating],
    bootstrap_conf_level: float = DEFAULT_CONFIDENCE_LEVEL,
    number_of_bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    sample_size: Optional[int] = None,
    statistic: Callable[[npt.NDArray[np.floating]], float] = np.mean,
    method: str = DEFAULT_BOOTSTRAP_METHOD,
    return_distribution: bool = False,
    seed: Optional[int] = DEFAULT_SEED,
    plot: bool = False,
) -> Dict[str, Union[float, npt.NDArray[np.floating], None]]:
    """Perform one-sample bootstrap for confidence interval estimation.

    Parameters
    ----------
    sample : ndarray
        1D array containing the sample data.
    bootstrap_conf_level : float, optional
        Confidence level between 0 and 1. Default is 0.95.
    number_of_bootstrap_samples : int, optional
        Number of bootstrap samples to generate. Default is 10000.
    sample_size : int, optional
        Size of each bootstrap sample. If None, uses the original sample size.
    statistic : callable, optional
        Function to compute the statistic of interest. Default is np.mean.
    method : str, optional
        Bootstrap method: 'percentile', 'bca', 'basic', or 'studentized'.
        Default is 'percentile'.
    return_distribution : bool, optional
        Whether to return the full bootstrap distribution. Default is False.
    seed : int, optional
        Random seed for reproducibility. Default is None.
    plot : bool, optional
        Whether to plot the bootstrap distribution. Default is False.

    Returns
    -------
    dict
        Dictionary containing:
        - 'statistic_value' : float
            The estimated statistic value.
        - 'confidence_interval' : ndarray
            Confidence interval [lower_bound, upper_bound].
        - 'distribution' : ndarray or None
            Full bootstrap distribution if return_distribution=True.

    Raises
    ------
    ValidationError
        If inputs are invalid.
    BootstrapMethodError
        If method is not supported.

    Examples
    --------
    >>> sample = np.array([1, 2, 3, 4, 5])
    >>> result = one_sample_bootstrap(sample)
    >>> 'statistic_value' in result and 'confidence_interval' in result
    True

    Notes
    -----
    Time complexity: O(n * b) where n is sample size, b is bootstrap samples.
    Space complexity: O(b).
    """
    # Validate inputs
    _validate_sample_array(sample, "sample")
    _validate_bootstrap_params(
        number_of_bootstrap_samples, bootstrap_conf_level, sample_size
    )

    if method not in BOOTSTRAP_METHODS:
        raise BootstrapMethodError(
            ERROR_MESSAGES["invalid_method"].format(methods=BOOTSTRAP_METHODS),
            method=method,
            available_methods=list(BOOTSTRAP_METHODS),
        )

    # Set sample size
    if sample_size is None:
        sample_size = len(sample)

    # Define bootstrap sample function
    def sample_function(
        generator: np.random.Generator,
    ) -> Union[float, Tuple[float, float]]:
        bootstrap_sample = sample[
            generator.choice(len(sample), size=sample_size, replace=True)
        ]

        if method == "studentized":
            return statistic(bootstrap_sample), bootstrap_sample.std()
        return statistic(bootstrap_sample)

    # Generate bootstrap samples
    bootstrap_stats = bootstrap_resampling(
        sample_function, number_of_bootstrap_samples, seed
    )
    original_stat = statistic(sample)

    # Calculate confidence interval based on method
    if method == "percentile":
        bootstrap_distribution = bootstrap_stats
        confidence_interval = estimate_confidence_interval(
            bootstrap_distribution, bootstrap_conf_level
        )
    elif method == "bca":
        bootstrap_distribution = bootstrap_stats
        confidence_interval = bca_confidence_interval(
            sample, bootstrap_distribution, statistic, bootstrap_conf_level
        )
    elif method == "basic":
        bootstrap_distribution = bootstrap_stats
        percentile_ci = estimate_confidence_interval(
            bootstrap_distribution, bootstrap_conf_level
        )
        # Basic bootstrap: 2 * original_stat - percentile_ci (reversed)
        confidence_interval = np.array(
            [
                2 * original_stat - percentile_ci[1],
                2 * original_stat - percentile_ci[0],
            ]
        )
    elif method == "studentized":
        bootstrap_distribution = bootstrap_stats[:, 0]
        bootstrap_std = bootstrap_stats[:, 1]

        # Studentized bootstrap
        sample_std = sample.std()
        t_stats = (bootstrap_distribution - original_stat) / (
            bootstrap_std / np.sqrt(sample_size)
        )

        t_ci = estimate_confidence_interval(t_stats, bootstrap_conf_level)
        confidence_interval = np.array(
            [
                original_stat - sample_std * t_ci[1] / np.sqrt(sample_size),
                original_stat - sample_std * t_ci[0] / np.sqrt(sample_size),
            ]
        )

        # Use the original bootstrap distribution for plotting
        bootstrap_distribution = bootstrap_stats[:, 0]

    # Calculate statistic value
    statistic_value = np.median(bootstrap_distribution)

    # Plot if requested
    if plot:
        bootstrap_plot(
            bootstrap_distribution,
            confidence_interval,
            statistic,
            two_sample_plot=False,
        )

    return {
        "statistic_value": float(statistic_value),
        "confidence_interval": confidence_interval,
        "distribution": bootstrap_distribution if return_distribution else None,
    }


def two_sample_bootstrap(
    control: npt.NDArray[np.floating],
    treatment: npt.NDArray[np.floating],
    bootstrap_conf_level: float = DEFAULT_CONFIDENCE_LEVEL,
    number_of_bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    sample_size: Optional[int] = None,
    statistic: Callable[
        [npt.NDArray[np.floating], npt.NDArray[np.floating]], float
    ] = difference_of_mean,
    return_distribution: bool = False,
    seed: Optional[int] = DEFAULT_SEED,
    plot: bool = False,
) -> Dict[str, Union[float, npt.NDArray[np.floating], None]]:
    """Perform two-sample bootstrap for comparing two groups.

    Parameters
    ----------
    control : ndarray
        1D array containing the control sample.
    treatment : ndarray
        1D array containing the treatment sample.
    bootstrap_conf_level : float, optional
        Confidence level between 0 and 1. Default is 0.95.
    number_of_bootstrap_samples : int, optional
        Number of bootstrap samples to generate. Default is 10000.
    sample_size : int, optional
        Size for each bootstrap sample. If None, uses original sample sizes.
    statistic : callable, optional
        Function to compute the test statistic. Default is difference_of_mean.
    return_distribution : bool, optional
        Whether to return the full bootstrap distribution. Default is False.
    seed : int, optional
        Random seed for reproducibility. Default is None.
    plot : bool, optional
        Whether to plot the bootstrap distribution. Default is False.

    Returns
    -------
    dict
        Dictionary containing:
        - 'p_value' : float
            Two-sided p-value.
        - 'statistic_value' : float
            Median of the bootstrap distribution.
        - 'confidence_interval' : ndarray
            Confidence interval [lower_bound, upper_bound].
        - 'distribution' : ndarray or None
            Full bootstrap distribution if return_distribution=True.

    Raises
    ------
    ValidationError
        If inputs are invalid.

    Examples
    --------
    >>> control = np.array([1, 2, 3, 4, 5])
    >>> treatment = np.array([2, 3, 4, 5, 6])
    >>> result = two_sample_bootstrap(control, treatment)
    >>> all(key in result for key in ['p_value', 'statistic_value', 'confidence_interval'])
    True

    Notes
    -----
    Time complexity: O((n + m) * b) where n, m are sample sizes, b is bootstrap samples.
    Space complexity: O(b).
    """
    # Validate inputs
    _validate_sample_array(control, "control")
    _validate_sample_array(treatment, "treatment")
    _validate_bootstrap_params(
        number_of_bootstrap_samples, bootstrap_conf_level, sample_size
    )

    # Set sample sizes
    if sample_size is None:
        control_sample_size = len(control)
        treatment_sample_size = len(treatment)
    else:
        control_sample_size = treatment_sample_size = sample_size

    # Define bootstrap sample function
    def sample_function(generator: np.random.Generator) -> float:
        control_sample = control[
            generator.choice(len(control), control_sample_size, replace=True)
        ]
        treatment_sample = treatment[
            generator.choice(len(treatment), treatment_sample_size, replace=True)
        ]
        return statistic(control_sample, treatment_sample)

    # Generate bootstrap distribution
    bootstrap_distribution = bootstrap_resampling(
        sample_function, number_of_bootstrap_samples, seed
    )

    # Calculate statistics
    confidence_interval = estimate_confidence_interval(
        bootstrap_distribution, bootstrap_conf_level
    )
    statistic_value = np.median(bootstrap_distribution)
    p_value = estimate_p_value(bootstrap_distribution, number_of_bootstrap_samples)

    # Plot if requested
    if plot:
        bootstrap_plot(bootstrap_distribution, confidence_interval, statistic)

    return {
        "p_value": float(p_value),
        "statistic_value": float(statistic_value),
        "confidence_interval": confidence_interval,
        "distribution": bootstrap_distribution if return_distribution else None,
    }


def spotify_one_sample_bootstrap(
    sample: npt.NDArray[np.floating],
    sample_size: Optional[int] = None,
    q: float = DEFAULT_QUANTILE,
    bootstrap_conf_level: float = DEFAULT_CONFIDENCE_LEVEL,
    number_of_bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    return_distribution: bool = False,
    plot: bool = False,
) -> Dict[str, Union[float, npt.NDArray[np.floating], None]]:
    """Perform Spotify-style one-sample bootstrap for quantile estimation.

    Parameters
    ----------
    sample : ndarray
        1D array containing the sample data.
    sample_size : int, optional
        Size of each bootstrap sample. If None, uses the original sample size.
    q : float, optional
        Quantile of interest between 0 and 1. Default is 0.5 (median).
    bootstrap_conf_level : float, optional
        Confidence level between 0 and 1. Default is 0.95.
    number_of_bootstrap_samples : int, optional
        Number of bootstrap samples to generate. Default is 10000.
    return_distribution : bool, optional
        Whether to return the full bootstrap distribution. Default is False.
    plot : bool, optional
        Whether to plot the bootstrap distribution. Default is False.

    Returns
    -------
    dict
        Dictionary containing:
        - 'statistic_value' : float
            The estimated quantile value.
        - 'confidence_interval' : ndarray
            Confidence interval [lower_bound, upper_bound].
        - 'distribution' : ndarray or None
            Full bootstrap distribution if return_distribution=True.

    Raises
    ------
    ValidationError
        If inputs are invalid.

    Examples
    --------
    >>> sample = np.array([1, 2, 3, 4, 5])
    >>> result = spotify_one_sample_bootstrap(sample, q=0.5)
    >>> 'statistic_value' in result and 'confidence_interval' in result
    True

    Notes
    -----
    Time complexity: O(n log n + b) where n is sample size, b is bootstrap samples.
    Space complexity: O(n + b).
    """
    # Validate inputs
    _validate_sample_array(sample, "sample")
    _validate_bootstrap_params(
        number_of_bootstrap_samples, bootstrap_conf_level, sample_size
    )

    if not (0 <= q <= 1):
        raise ValidationError(
            ERROR_MESSAGES["invalid_quantile"],
            parameter="q",
            value=q,
        )

    # Set sample size
    if sample_size is None:
        sample_size = len(sample)

    # Sort sample for quantile calculation
    sorted_sample = np.sort(sample)

    # Calculate confidence interval using binomial distribution
    left_quant = (1 - bootstrap_conf_level) / 2
    right_quant = 1 - left_quant

    ci_indices = binom.ppf([left_quant, right_quant], sample_size + 1, q)
    ci_indices = np.clip(ci_indices.astype(int), 0, len(sorted_sample) - 1)

    confidence_interval = sorted_sample[ci_indices]

    # Generate bootstrap distribution using binomial sampling
    quantile_indices = binomial(sample_size + 1, q, number_of_bootstrap_samples)
    quantile_indices = np.clip(quantile_indices, 0, len(sorted_sample) - 1)
    bootstrap_distribution = sorted_sample[quantile_indices]

    # Calculate statistic value
    statistic_value = np.quantile(sample, q)

    # Plot if requested
    if plot:
        bootstrap_plot(
            bootstrap_distribution,
            confidence_interval,
            f"Quantile_{q}",
            two_sample_plot=False,
        )

    return {
        "statistic_value": float(statistic_value),
        "confidence_interval": confidence_interval,
        "distribution": bootstrap_distribution if return_distribution else None,
    }


def spotify_two_sample_bootstrap(
    control: npt.NDArray[np.floating],
    treatment: npt.NDArray[np.floating],
    number_of_bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    sample_size: Optional[int] = None,
    q1: float = DEFAULT_QUANTILE,
    q2: Optional[float] = None,
    statistic: Callable[
        [
            Union[npt.NDArray[np.floating], float],
            Union[npt.NDArray[np.floating], float],
        ],
        Union[npt.NDArray[np.floating], float],
    ] = difference,
    bootstrap_conf_level: float = DEFAULT_CONFIDENCE_LEVEL,
    return_distribution: bool = False,
    plot: bool = False,
) -> Dict[str, Union[float, npt.NDArray[np.floating], None]]:
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
        Size for each bootstrap sample. If None, uses original sample sizes.
    q1 : float, optional
        Quantile of interest for control sample. Default is 0.5.
    q2 : float, optional
        Quantile of interest for treatment sample. If None, uses q1.
    statistic : callable, optional
        Function to compare quantiles. Default is difference.
    bootstrap_conf_level : float, optional
        Confidence level between 0 and 1. Default is 0.95.
    return_distribution : bool, optional
        Whether to return the full bootstrap distribution. Default is False.
    plot : bool, optional
        Whether to plot the bootstrap distribution. Default is False.

    Returns
    -------
    dict
        Dictionary containing:
        - 'p_value' : float
            Two-sided p-value.
        - 'statistic_value' : float
            The difference between quantiles.
        - 'confidence_interval' : ndarray
            Confidence interval [lower_bound, upper_bound].
        - 'distribution' : ndarray or None
            Full bootstrap distribution if return_distribution=True.

    Raises
    ------
    ValidationError
        If inputs are invalid.

    Examples
    --------
    >>> control = np.array([1, 2, 3, 4, 5])
    >>> treatment = np.array([2, 3, 4, 5, 6])
    >>> result = spotify_two_sample_bootstrap(control, treatment)
    >>> all(key in result for key in ['p_value', 'statistic_value', 'confidence_interval'])
    True

    Notes
    -----
    Time complexity: O((n + m) log (n + m) + b) where n, m are sample sizes, b is bootstrap samples.
    Space complexity: O(n + m + b).
    """
    # Validate inputs
    _validate_sample_array(control, "control")
    _validate_sample_array(treatment, "treatment")
    _validate_bootstrap_params(
        number_of_bootstrap_samples, bootstrap_conf_level, sample_size
    )

    if not (0 <= q1 <= 1):
        raise ValidationError(
            ERROR_MESSAGES["invalid_quantile"],
            parameter="q1",
            value=q1,
        )

    # Set q2 if not provided
    if q2 is None:
        q2 = q1

    if not (0 <= q2 <= 1):
        raise ValidationError(
            ERROR_MESSAGES["invalid_quantile"],
            parameter="q2",
            value=q2,
        )

    # Set sample sizes
    if sample_size is None:
        control_sample_size = len(control)
        treatment_sample_size = len(treatment)
    else:
        control_sample_size = treatment_sample_size = sample_size

    # Sort samples for quantile calculation
    sorted_control = np.sort(control)
    sorted_treatment = np.sort(treatment)

    # Generate bootstrap samples using binomial sampling
    control_indices = binomial(control_sample_size + 1, q1, number_of_bootstrap_samples)
    treatment_indices = binomial(
        treatment_sample_size + 1, q2, number_of_bootstrap_samples
    )

    # Clip indices to valid range
    control_indices = np.clip(control_indices, 0, len(sorted_control) - 1)
    treatment_indices = np.clip(treatment_indices, 0, len(sorted_treatment) - 1)

    # Get bootstrap quantile values
    control_quantiles = sorted_control[control_indices]
    treatment_quantiles = sorted_treatment[treatment_indices]

    # Calculate bootstrap distribution
    bootstrap_distribution = statistic(control_quantiles, treatment_quantiles)

    # Calculate original statistic
    control_quantile = np.quantile(control, q1)
    treatment_quantile = np.quantile(treatment, q2)
    statistic_value = statistic(control_quantile, treatment_quantile)

    # Calculate confidence interval and p-value
    confidence_interval = estimate_confidence_interval(
        bootstrap_distribution, bootstrap_conf_level
    )
    p_value = estimate_p_value(bootstrap_distribution, number_of_bootstrap_samples)

    # Plot if requested
    if plot:
        bootstrap_plot(bootstrap_distribution, confidence_interval, statistic)

    return {
        "p_value": float(p_value),
        "statistic_value": float(statistic_value),
        "confidence_interval": confidence_interval,
        "distribution": bootstrap_distribution if return_distribution else None,
    }


def poisson_bootstrap(
    control: npt.NDArray[np.floating],
    treatment: npt.NDArray[np.floating],
    number_of_bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
) -> float:
    """Perform Poisson bootstrap for comparing aggregated values.

    This method is useful for comparing sums or aggregated values between
    two samples using Poisson resampling.

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

    Raises
    ------
    ValidationError
        If inputs are invalid.

    Examples
    --------
    >>> control = np.array([1, 2, 3, 4, 5])
    >>> treatment = np.array([2, 3, 4, 5, 6])
    >>> p_value = poisson_bootstrap(control, treatment)
    >>> 0 <= p_value <= 1
    True

    Notes
    -----
    Time complexity: O(min(n, m) * b) where n, m are sample sizes, b is bootstrap samples.
    Space complexity: O(b).
    """
    # Validate inputs
    _validate_sample_array(control, "control")
    _validate_sample_array(treatment, "treatment")
    _validate_bootstrap_params(number_of_bootstrap_samples, 0.95)

    # Use minimum sample size
    min_sample_size = min(len(control), len(treatment))

    # Initialize bootstrap distributions
    control_distribution = np.zeros(number_of_bootstrap_samples)
    treatment_distribution = np.zeros(number_of_bootstrap_samples)

    # Perform Poisson resampling
    for i in range(min_sample_size):
        weights = np.random.poisson(1, number_of_bootstrap_samples)
        control_distribution += control[i] * weights
        treatment_distribution += treatment[i] * weights

    # Calculate difference distribution
    bootstrap_difference_distribution = treatment_distribution - control_distribution

    # Calculate p-value
    p_value = estimate_p_value(
        bootstrap_difference_distribution, number_of_bootstrap_samples
    )

    return float(p_value)


def bootstrap(
    control: npt.NDArray[np.floating],
    treatment: Optional[npt.NDArray[np.floating]] = None,
    bootstrap_conf_level: float = DEFAULT_CONFIDENCE_LEVEL,
    number_of_bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    sample_size: Optional[int] = None,
    statistic: Callable = np.mean,
    method: str = DEFAULT_BOOTSTRAP_METHOD,
    spotify_style: bool = False,
    q: Union[float, Tuple[float, float]] = DEFAULT_QUANTILE,
    return_distribution: bool = False,
    seed: Optional[int] = DEFAULT_SEED,
    plot: bool = False,
) -> Dict[str, Union[float, npt.NDArray[np.floating], None]]:
    """Unified bootstrap function that automatically selects the appropriate method.

    Parameters
    ----------
    control : ndarray
        1D array containing the control/first sample.
    treatment : ndarray, optional
        1D array containing the treatment/second sample.
        If None, performs one-sample bootstrap.
    bootstrap_conf_level : float, optional
        Confidence level between 0 and 1. Default is 0.95.
    number_of_bootstrap_samples : int, optional
        Number of bootstrap samples to generate. Default is 10000.
    sample_size : int, optional
        Size for resampling. If None, uses original sample size(s).
    statistic : callable, optional
        Function computing the statistic of interest. Default is np.mean.
    method : str, optional
        Bootstrap method for one-sample bootstrap. Default is 'percentile'.
    spotify_style : bool, optional
        Whether to use Spotify-style bootstrap. Default is False.
    q : float or tuple, optional
        Quantile(s) for Spotify-style bootstrap. Default is 0.5.
    return_distribution : bool, optional
        Whether to return the full bootstrap distribution. Default is False.
    seed : int, optional
        Random seed for reproducibility. Default is None.
    plot : bool, optional
        Whether to plot the bootstrap distribution. Default is False.

    Returns
    -------
    dict
        Dictionary containing bootstrap results. Contents depend on the method used.

    Raises
    ------
    ValidationError
        If inputs are invalid.

    Examples
    --------
    >>> # One-sample bootstrap
    >>> sample = np.array([1, 2, 3, 4, 5])
    >>> result = bootstrap(sample)
    >>> 'statistic_value' in result
    True

    >>> # Two-sample bootstrap
    >>> control = np.array([1, 2, 3, 4, 5])
    >>> treatment = np.array([2, 3, 4, 5, 6])
    >>> result = bootstrap(control, treatment)
    >>> 'p_value' in result
    True

    Notes
    -----
    Time complexity: Depends on the selected method.
    Space complexity: Depends on the selected method.
    """
    # Handle quantile parameter for Spotify-style bootstrap
    if spotify_style and isinstance(q, (tuple, list)):
        q1, q2 = q[:2]
    else:
        q1 = q2 = q

    # Select appropriate bootstrap method
    if treatment is None:
        # One-sample bootstrap
        if spotify_style:
            return spotify_one_sample_bootstrap(
                sample=control,
                sample_size=sample_size,
                q=q1,
                bootstrap_conf_level=bootstrap_conf_level,
                number_of_bootstrap_samples=number_of_bootstrap_samples,
                return_distribution=return_distribution,
                plot=plot,
            )
        else:
            return one_sample_bootstrap(
                sample=control,
                bootstrap_conf_level=bootstrap_conf_level,
                number_of_bootstrap_samples=number_of_bootstrap_samples,
                sample_size=sample_size,
                statistic=statistic,
                method=method,
                return_distribution=return_distribution,
                seed=seed,
                plot=plot,
            )
    else:
        # Two-sample bootstrap
        if spotify_style:
            return spotify_two_sample_bootstrap(
                control=control,
                treatment=treatment,
                number_of_bootstrap_samples=number_of_bootstrap_samples,
                sample_size=sample_size,
                q1=q1,
                q2=q2,
                statistic=statistic if statistic != np.mean else difference,
                bootstrap_conf_level=bootstrap_conf_level,
                return_distribution=return_distribution,
                plot=plot,
            )
        else:
            return two_sample_bootstrap(
                control=control,
                treatment=treatment,
                bootstrap_conf_level=bootstrap_conf_level,
                number_of_bootstrap_samples=number_of_bootstrap_samples,
                sample_size=sample_size,
                statistic=statistic if statistic != np.mean else difference_of_mean,
                return_distribution=return_distribution,
                seed=seed,
                plot=plot,
            )
