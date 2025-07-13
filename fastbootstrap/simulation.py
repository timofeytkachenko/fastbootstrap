"""Simulation utilities for A/B testing and statistical analysis.

This module provides functions for simulating A/B tests, calculating statistical
power, and performing related statistical simulations.
"""

from typing import Callable, Dict, Union

import numpy as np
import numpy.typing as npt
from joblib import Parallel, delayed
from scipy.stats import ttest_ind

from .constants import ALPHA_THRESHOLD, DEFAULT_N_JOBS
from .core import _validate_sample_array
from .exceptions import NumericalError, ValidationError
from .visualization import plot_summary


def ab_test_simulation(
    control: npt.NDArray[np.floating],
    treatment: npt.NDArray[np.floating],
    number_of_experiments: int = 2000,
    stat_test: Callable[
        [npt.NDArray[np.floating], npt.NDArray[np.floating]], tuple[float, float]
    ] = ttest_ind,
    n_jobs: int = DEFAULT_N_JOBS,
) -> Dict[str, Union[float, npt.NDArray[np.floating]]]:
    """Simulate multiple A/B tests with parallel p-value computation.

    This function performs multiple bootstrap simulations to estimate the
    statistical power and p-value distribution for A/B testing scenarios.

    Parameters
    ----------
    control : ndarray
        1D array containing the control sample.
    treatment : ndarray
        1D array containing the treatment sample.
    number_of_experiments : int, optional
        Number of simulated experiments to run. Default is 2000.
    stat_test : callable, optional
        Statistical test function that takes two arrays and returns
        (statistic, p_value). Default is scipy.stats.ttest_ind.
    n_jobs : int, optional
        Number of parallel jobs. -1 uses all available cores. Default is -1.

    Returns
    -------
    dict
        Dictionary containing:
        - 'ab_p_values' : ndarray
            Array of p-values from all simulated experiments.
        - 'test_power' : float
            Proportion of experiments with p < 0.05 (statistical power).

    Raises
    ------
    ValidationError
        If inputs are invalid.
    NumericalError
        If simulation fails.

    Examples
    --------
    >>> control = np.random.normal(0, 1, 100)
    >>> treatment = np.random.normal(0.5, 1, 100)
    >>> result = ab_test_simulation(control, treatment, number_of_experiments=100)
    >>> 'test_power' in result and 'ab_p_values' in result
    True

    Notes
    -----
    Time complexity: O(e * (n + m)) where e is experiments, n, m are sample sizes.
    Space complexity: O(e).
    """
    # Validate inputs
    _validate_sample_array(control, "control")
    _validate_sample_array(treatment, "treatment")

    if number_of_experiments <= 0:
        raise ValidationError(
            "Number of experiments must be positive",
            parameter="number_of_experiments",
            value=number_of_experiments,
        )

    try:
        control_size = len(control)
        treatment_size = len(treatment)

        def single_experiment() -> float:
            """Run a single A/B test experiment."""
            # Bootstrap sampling from original distributions
            control_sample = np.random.choice(control, control_size, replace=True)
            treatment_sample = np.random.choice(treatment, treatment_size, replace=True)

            # Perform statistical test
            _, p_value = stat_test(control_sample, treatment_sample)
            return float(p_value)

        # Run experiments in parallel
        ab_p_values = Parallel(n_jobs=n_jobs)(
            delayed(single_experiment)() for _ in range(number_of_experiments)
        )
        ab_p_values = np.array(ab_p_values)

        # Calculate test power (proportion of significant results)
        test_power = float(np.mean(ab_p_values < ALPHA_THRESHOLD))

        return {"ab_p_values": ab_p_values, "test_power": test_power}

    except Exception as e:
        raise NumericalError(
            "Failed to perform A/B test simulation",
            operation="ab_test_simulation",
            values={"n_experiments": number_of_experiments, "n_jobs": n_jobs},
        ) from e


def aa_test_simulation(
    sample: npt.NDArray[np.floating],
    number_of_experiments: int = 2000,
    stat_test: Callable[
        [npt.NDArray[np.floating], npt.NDArray[np.floating]], tuple[float, float]
    ] = ttest_ind,
    n_jobs: int = DEFAULT_N_JOBS,
) -> Dict[str, Union[float, npt.NDArray[np.floating]]]:
    """Simulate multiple A/A tests to validate Type I error rate.

    This function performs A/A test simulations where both groups are drawn
    from the same distribution, used to validate that the Type I error rate
    is properly controlled.

    Parameters
    ----------
    sample : ndarray
        1D array containing the sample data.
    number_of_experiments : int, optional
        Number of simulated experiments to run. Default is 2000.
    stat_test : callable, optional
        Statistical test function that takes two arrays and returns
        (statistic, p_value). Default is scipy.stats.ttest_ind.
    n_jobs : int, optional
        Number of parallel jobs. -1 uses all available cores. Default is -1.

    Returns
    -------
    dict
        Dictionary containing:
        - 'aa_p_values' : ndarray
            Array of p-values from all simulated A/A experiments.
        - 'type_i_error_rate' : float
            Proportion of experiments with p < 0.05 (should be ≈ 0.05).

    Raises
    ------
    ValidationError
        If inputs are invalid.
    NumericalError
        If simulation fails.

    Examples
    --------
    >>> sample = np.random.normal(0, 1, 200)
    >>> result = aa_test_simulation(sample, number_of_experiments=100)
    >>> 'type_i_error_rate' in result and 'aa_p_values' in result
    True

    Notes
    -----
    Time complexity: O(e * n) where e is experiments, n is sample size.
    Space complexity: O(e).
    """
    # Validate inputs
    _validate_sample_array(sample, "sample")

    if number_of_experiments <= 0:
        raise ValidationError(
            "Number of experiments must be positive",
            parameter="number_of_experiments",
            value=number_of_experiments,
        )

    try:
        sample_size = len(sample) // 2  # Split sample in half

        def single_aa_experiment() -> float:
            """Run a single A/A test experiment."""
            # Randomly sample two groups from the same distribution
            shuffled_sample = np.random.permutation(sample)
            group_a = shuffled_sample[:sample_size]
            group_b = shuffled_sample[sample_size : 2 * sample_size]

            # Perform statistical test
            _, p_value = stat_test(group_a, group_b)
            return float(p_value)

        # Run experiments in parallel
        aa_p_values = Parallel(n_jobs=n_jobs)(
            delayed(single_aa_experiment)() for _ in range(number_of_experiments)
        )
        aa_p_values = np.array(aa_p_values)

        # Calculate Type I error rate (should be approximately alpha)
        type_i_error_rate = float(np.mean(aa_p_values < ALPHA_THRESHOLD))

        return {"aa_p_values": aa_p_values, "type_i_error_rate": type_i_error_rate}

    except Exception as e:
        raise NumericalError(
            "Failed to perform A/A test simulation",
            operation="aa_test_simulation",
            values={"n_experiments": number_of_experiments, "n_jobs": n_jobs},
        ) from e


def power_analysis(
    control: npt.NDArray[np.floating],
    treatment: npt.NDArray[np.floating],
    number_of_experiments: int = 2000,
    stat_test: Callable[
        [npt.NDArray[np.floating], npt.NDArray[np.floating]], tuple[float, float]
    ] = ttest_ind,
    n_jobs: int = DEFAULT_N_JOBS,
    plot_results: bool = False,
) -> Dict[str, Union[float, npt.NDArray[np.floating]]]:
    """Perform comprehensive power analysis with both A/A and A/B simulations.

    This function runs both A/A and A/B test simulations to provide a
    comprehensive power analysis including Type I error rate validation
    and statistical power estimation.

    Parameters
    ----------
    control : ndarray
        1D array containing the control sample.
    treatment : ndarray
        1D array containing the treatment sample.
    number_of_experiments : int, optional
        Number of simulated experiments to run. Default is 2000.
    stat_test : callable, optional
        Statistical test function that takes two arrays and returns
        (statistic, p_value). Default is scipy.stats.ttest_ind.
    n_jobs : int, optional
        Number of parallel jobs. -1 uses all available cores. Default is -1.
    plot_results : bool, optional
        Whether to create summary plots. Default is False.

    Returns
    -------
    dict
        Dictionary containing:
        - 'aa_results' : dict
            Results from A/A test simulation.
        - 'ab_results' : dict
            Results from A/B test simulation.
        - 'power_summary' : dict
            Summary statistics including statistical power and Type I error rate.

    Raises
    ------
    ValidationError
        If inputs are invalid.
    NumericalError
        If simulation fails.

    Examples
    --------
    >>> control = np.random.normal(0, 1, 100)
    >>> treatment = np.random.normal(0.5, 1, 100)
    >>> result = power_analysis(control, treatment, number_of_experiments=100)
    >>> 'power_summary' in result
    True

    Notes
    -----
    Time complexity: O(e * (n + m)) where e is experiments, n, m are sample sizes.
    Space complexity: O(e).
    """
    # Validate inputs
    _validate_sample_array(control, "control")
    _validate_sample_array(treatment, "treatment")

    try:
        # Combine samples for A/A testing
        combined_sample = np.concatenate([control, treatment])

        # Run A/A simulation
        aa_results = aa_test_simulation(
            combined_sample, number_of_experiments, stat_test, n_jobs
        )

        # Run A/B simulation
        ab_results = ab_test_simulation(
            control, treatment, number_of_experiments, stat_test, n_jobs
        )

        # Create power summary
        power_summary = {
            "statistical_power": ab_results["test_power"],
            "type_i_error_rate": aa_results["type_i_error_rate"],
            "control_mean": float(np.mean(control)),
            "treatment_mean": float(np.mean(treatment)),
            "control_std": float(np.std(control, ddof=1)),
            "treatment_std": float(np.std(treatment, ddof=1)),
            "sample_size_control": len(control),
            "sample_size_treatment": len(treatment),
        }

        # Create plots if requested
        if plot_results:
            plot_summary(aa_results["aa_p_values"], ab_results["ab_p_values"])

        return {
            "aa_results": aa_results,
            "ab_results": ab_results,
            "power_summary": power_summary,
        }

    except Exception as e:
        raise NumericalError(
            "Failed to perform power analysis",
            operation="power_analysis",
            values={"n_experiments": number_of_experiments},
        ) from e
