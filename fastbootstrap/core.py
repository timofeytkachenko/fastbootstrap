"""Core bootstrap functionality.

This module provides the fundamental bootstrap resampling functions and
core statistical utilities for bootstrap analysis.
"""

import warnings
from functools import lru_cache
from typing import Callable, Optional, Union

import numpy as np
import numpy.typing as npt
import psutil
from joblib import Parallel, delayed
from scipy.stats import norm

from .constants import (
    BATCH_SIZE_LARGE,
    BATCH_SIZE_MASSIVE,
    BATCH_SIZE_MEDIUM,
    BATCH_SIZE_SMALL,
    BATCH_SIZE_THRESHOLD_LARGE,
    BATCH_SIZE_THRESHOLD_MEDIUM,
    BATCH_SIZE_THRESHOLD_SMALL,
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_DTYPE_BYTES,
    DEFAULT_N_JOBS,
    DEFAULT_SEED,
    EPSILON,
    ERROR_MESSAGES,
    JACKKNIFE_PARALLEL_THRESHOLD,
    LARGE_SAMPLE_THRESHOLD,
    LOW_MEM_BATCH_CAP,
    MEM_FRACTION,
    MEMORY_LOW_THRESHOLD,
    MEMORY_MODERATE_THRESHOLD,
    MIN_BATCH_FLOOR,
    MIN_BATCHES_PER_WORKER,
    MIN_SAMPLE_SIZE,
    SAMPLE_COMPLEXITY_DIVISOR,
)
from .exceptions import InsufficientDataError, NumericalError, ValidationError


def _validate_bootstrap_params(
    number_of_bootstrap_samples: int,
    bootstrap_conf_level: float,
    sample_size: Optional[int] = None,
) -> None:
    """Validate bootstrap parameters.

    Parameters
    ----------
    number_of_bootstrap_samples : int
        Number of bootstrap samples.
    bootstrap_conf_level : float
        Confidence level for bootstrap.
    sample_size : int, optional
        Sample size for bootstrap.

    Raises
    ------
    ValidationError
        If parameters are invalid.
    """
    if number_of_bootstrap_samples <= 0:
        raise ValidationError(
            ERROR_MESSAGES["invalid_bootstrap_samples"],
            parameter="number_of_bootstrap_samples",
            value=number_of_bootstrap_samples,
        )

    if not (0 < bootstrap_conf_level < 1):
        raise ValidationError(
            ERROR_MESSAGES["invalid_confidence_level"],
            parameter="bootstrap_conf_level",
            value=bootstrap_conf_level,
        )

    if sample_size is not None and sample_size < MIN_SAMPLE_SIZE:
        raise ValidationError(
            ERROR_MESSAGES["invalid_sample_size"],
            parameter="sample_size",
            value=sample_size,
        )


def _validate_sample_array(array: npt.NDArray[np.floating], name: str) -> None:
    """Validate a sample array.

    Parameters
    ----------
    array : ndarray
        Array to validate.
    name : str
        Name of the array for error messages.

    Raises
    ------
    ValidationError
        If array is invalid.
    InsufficientDataError
        If array is too small.
    """
    if array.size == 0:
        raise ValidationError(ERROR_MESSAGES["empty_array"], parameter=name)

    if array.size < MIN_SAMPLE_SIZE:
        raise InsufficientDataError(
            ERROR_MESSAGES["insufficient_data"],
            sample_size=array.size,
            min_required=MIN_SAMPLE_SIZE,
        )


def estimate_confidence_interval(
    distribution: npt.NDArray[np.floating],
    bootstrap_conf_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> npt.NDArray[np.floating]:
    """Estimate confidence interval from a bootstrap distribution.

    Parameters
    ----------
    distribution : ndarray
        1D array containing the bootstrap distribution.
    bootstrap_conf_level : float, optional
        Confidence level between 0 and 1. Default is 0.95.

    Returns
    -------
    ndarray
        Array [lower_bound, upper_bound] representing the confidence interval.

    Raises
    ------
    ValidationError
        If distribution is empty or confidence level is invalid.
    NumericalError
        If quantile computation fails.

    Examples
    --------
    >>> distribution = np.array([1, 2, 3, 4, 5])
    >>> ci = estimate_confidence_interval(distribution, 0.95)
    >>> len(ci)
    2

    Notes
    -----
    Time complexity: O(n log n) where n is the distribution length.
    Space complexity: O(1).
    """
    _validate_sample_array(distribution, "distribution")
    _validate_bootstrap_params(1, bootstrap_conf_level)

    try:
        left_quant = (1 - bootstrap_conf_level) / 2
        right_quant = 1 - left_quant
        return np.quantile(distribution, [left_quant, right_quant])
    except (ValueError, TypeError) as e:
        raise NumericalError(
            "Failed to compute confidence interval",
            operation="quantile_computation",
            values={"confidence_level": bootstrap_conf_level},
        ) from e


def estimate_p_value(
    bootstrap_difference_distribution: npt.NDArray[np.floating],
    number_of_bootstrap_samples: int,
) -> float:
    """Estimate two-sided p-value from bootstrap difference distribution.

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

    Raises
    ------
    ValidationError
        If distribution is empty or number of samples is invalid.
    NumericalError
        If p-value computation fails.

    Examples
    --------
    >>> distribution = np.array([-1, 0, 1, 2, 3])
    >>> p_value = estimate_p_value(distribution, 5)
    >>> 0 <= p_value <= 1
    True

    Notes
    -----
    Time complexity: O(n) where n is the distribution length.
    Space complexity: O(1).
    """
    _validate_sample_array(
        bootstrap_difference_distribution, "bootstrap_difference_distribution"
    )
    _validate_bootstrap_params(number_of_bootstrap_samples, 0.95)

    try:
        positions = np.sum(bootstrap_difference_distribution < 0, axis=0)
        return float(
            2
            * np.minimum(positions, number_of_bootstrap_samples - positions)
            / number_of_bootstrap_samples
        )
    except (ValueError, TypeError) as e:
        raise NumericalError(
            "Failed to compute p-value",
            operation="p_value_computation",
            values={"sample_count": number_of_bootstrap_samples},
        ) from e


def estimate_bin_params(sample: npt.NDArray[np.floating]) -> tuple[float, int]:
    """Estimate optimal histogram bin parameters using Freedman-Diaconis rule.

    Parameters
    ----------
    sample : ndarray
        1D array containing observations.

    Returns
    -------
    tuple[float, int]
        A tuple (bin_width, bin_count) where:
        - bin_width : float
            Width of each bin using Freedman-Diaconis rule.
        - bin_count : int
            Number of bins for the histogram.

    Raises
    ------
    ValidationError
        If sample is empty.
    NumericalError
        If bin computation fails.

    Examples
    --------
    >>> sample = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> bin_width, bin_count = estimate_bin_params(sample)
    >>> bin_width > 0 and bin_count > 0
    True

    Notes
    -----
    Time complexity: O(n log n) where n is the sample length.
    Space complexity: O(1).
    """
    _validate_sample_array(sample, "sample")

    try:
        q1 = np.quantile(sample, 0.25)
        q3 = np.quantile(sample, 0.75)
        iqr = q3 - q1

        # Freedman-Diaconis rule
        bin_width = (2 * iqr) / (sample.shape[0] ** (1 / 3))

        # Ensure bin_width is not zero
        if bin_width < EPSILON:
            bin_width = (sample.max() - sample.min()) / 10

        bin_count = max(1, int(np.ceil((sample.max() - sample.min()) / bin_width)))

        return float(bin_width), bin_count
    except (ValueError, TypeError) as e:
        raise NumericalError(
            "Failed to compute bin parameters",
            operation="bin_computation",
            values={"sample_size": sample.size},
        ) from e


def _compute_single_jackknife_statistic(
    sample: npt.NDArray[np.floating],
    index_to_remove: int,
    statistic: Callable[[npt.NDArray[np.floating]], float],
) -> float:
    """Compute jackknife statistic for a single leave-one-out sample.

    Helper function for parallel jackknife computation in BCa method.

    Parameters
    ----------
    sample : ndarray
        Original sample data.
    index_to_remove : int
        Index to exclude from the sample.
    statistic : callable
        Function that computes the statistic of interest.

    Returns
    -------
    float
        Statistic computed on leave-one-out sample.

    Notes
    -----
    Time complexity: O(n + s) where n is sample size, s is statistic cost.
    Space complexity: O(n) for the masked array.
    """
    mask = np.ones(len(sample), dtype=bool)
    mask[index_to_remove] = False
    return statistic(sample[mask])


def _compute_jackknife_acceleration(
    sample: npt.NDArray[np.floating],
    statistic: Callable[[npt.NDArray[np.floating]], float],
    n_jobs: int = 1,
    parallel_threshold: int = 1000,
) -> tuple[float, npt.NDArray[np.floating]]:
    """Compute jackknife acceleration parameter for BCa method.

    Efficiently computes jackknife statistics with automatic parallelization
    for large samples. Uses vectorized operations when possible.

    Parameters
    ----------
    sample : ndarray
        Original sample data.
    statistic : callable
        Function that computes the statistic of interest.
    n_jobs : int, optional
        Number of parallel jobs for jackknife computation.
        Only used if sample size >= parallel_threshold. Default is 1.
    parallel_threshold : int, optional
        Minimum sample size to trigger parallel processing. Default is 1000.

    Returns
    -------
    tuple[float, ndarray]
        A tuple (acceleration, jackknife_stats) where:
        - acceleration : float
            BCa acceleration parameter.
        - jackknife_stats : ndarray
            Array of jackknife statistics for all leave-one-out samples.

    Notes
    -----
    **Optimizations:**

    - Small samples (< parallel_threshold): Sequential computation with minimal overhead
    - Large samples (>= parallel_threshold): Parallel computation with joblib
    - Handles degenerate cases (zero variance) gracefully

    **Acceleration Formula:**

    .. math::
        a = \\frac{\\sum_{i=1}^{n} (\\bar{\\theta}_{(\\cdot)} - \\theta_{(i)})^3}
                 {6 [\\sum_{i=1}^{n} (\\bar{\\theta}_{(\\cdot)} - \\theta_{(i)})^2]^{3/2}}

    where :math:`\\theta_{(i)}` is the statistic with i-th observation removed.

    Time complexity: O(n² * s) where n is sample size, s is statistic cost.
    Space complexity: O(n) for jackknife statistics array.
    """
    n = len(sample)

    # Compute jackknife statistics
    if n >= parallel_threshold and n_jobs != 1:
        # Parallel computation for large samples
        parallel_executor = Parallel(n_jobs=n_jobs, prefer="threads")
        jackknife_stats = np.array(
            parallel_executor(
                delayed(_compute_single_jackknife_statistic)(sample, i, statistic)
                for i in range(n)
            )
        )
    else:
        # Sequential computation for small samples (faster due to less overhead)
        jackknife_stats = np.array(
            [
                _compute_single_jackknife_statistic(sample, i, statistic)
                for i in range(n)
            ]
        )

    # Compute acceleration parameter
    jack_mean = np.mean(jackknife_stats)
    centered = jack_mean - jackknife_stats

    numerator = np.sum(centered**3)
    denominator = 6.0 * (np.sum(centered**2) ** 1.5)

    # Handle degenerate case (zero variance in jackknife statistics)
    if abs(denominator) < EPSILON:
        return 0.0, jackknife_stats

    return numerator / denominator, jackknife_stats


def bca_confidence_interval(
    sample: npt.NDArray[np.floating],
    bootstrap_distribution: npt.NDArray[np.floating],
    statistic: Callable[[npt.NDArray[np.floating]], float] = np.mean,
    bootstrap_conf_level: float = DEFAULT_CONFIDENCE_LEVEL,
    n_jobs: int = 1,
    jackknife_parallel_threshold: int = JACKKNIFE_PARALLEL_THRESHOLD,
) -> npt.NDArray[np.floating]:
    """Compute BCa (bias-corrected and accelerated) confidence interval.

    The BCa method adjusts for bias and skewness in the bootstrap distribution,
    providing more accurate coverage than percentile intervals, especially for
    small samples or non-symmetric distributions.

    Parameters
    ----------
    sample : ndarray
        1D array containing the original sample data.
    bootstrap_distribution : ndarray
        1D array containing the bootstrap distribution of the statistic.
    statistic : callable, optional
        Function that computes the statistic of interest.
        Must accept a 1D array and return a scalar.
        Default is np.mean.
    bootstrap_conf_level : float, optional
        Confidence level between 0 and 1. Default is 0.95.
    n_jobs : int, optional
        Number of parallel jobs for jackknife computation.
        Only used if sample size >= jackknife_parallel_threshold.
        -1 uses all available cores. Default is 1 (sequential).
    jackknife_parallel_threshold : int, optional
        Minimum sample size to enable parallel jackknife computation.
        Default is 1000.

    Returns
    -------
    ndarray
        Array [lower_bound, upper_bound] representing the BCa confidence interval.

    Raises
    ------
    ValidationError
        If inputs are invalid or have incompatible shapes.
    NumericalError
        If BCa computation fails due to numerical instability.

    Examples
    --------
    >>> sample = np.array([1, 2, 3, 4, 5])
    >>> bootstrap_dist = np.array([2.5, 3.0, 2.8, 3.2, 2.9])
    >>> ci = bca_confidence_interval(sample, bootstrap_dist)
    >>> len(ci)
    2

    >>> # Large sample with parallel jackknife
    >>> large_sample = np.random.randn(5000)
    >>> bootstrap_dist = np.random.randn(10000)
    >>> ci = bca_confidence_interval(large_sample, bootstrap_dist, n_jobs=-1)
    >>> len(ci)
    2

    Notes
    -----
    **Method Details:**

    The BCa method computes adjusted confidence intervals using:

    1. **Bias Correction (z0)**: Quantifies asymmetry in bootstrap distribution
       relative to the original sample statistic.

    2. **Acceleration (a)**: Measures rate of change of standard error using
       jackknife-after-bootstrap. Accounts for skewness and non-constant variance.

    3. **Adjusted Quantiles**: Transforms nominal quantiles using z0 and a to
       correct for bias and skewness.

    **Corner Cases:**

    - Zero variance in jackknife: Falls back to percentile method (a=0)
    - Infinite z0 (all bootstrap stats on one side): Clamped to ±8 standard deviations
    - Invalid adjusted alphas: Clamped to valid range [0, 1]
    - Degenerate bootstrap distribution: Returns percentile interval with warning

    **Performance:**

    - Small samples (< 1000): Sequential jackknife computation (minimal overhead)
    - Large samples (≥ 1000): Optional parallel jackknife with n_jobs parameter
    - Prefer `threads` backend for jackknife (shared memory access)

    Time complexity: O(n² * s + b log b) where n is sample size, s is statistic cost,
                     b is number of bootstrap samples.
    Space complexity: O(n + b) for jackknife statistics and sorted bootstrap distribution.

    References
    ----------
    .. [1] Efron, B. (1987). "Better bootstrap confidence intervals".
           Journal of the American Statistical Association, 82(397), 171-185.
    .. [2] DiCiccio, T. J., & Efron, B. (1996). "Bootstrap confidence intervals".
           Statistical Science, 11(3), 189-228.
    """
    # Input validation
    _validate_sample_array(sample, "sample")
    _validate_sample_array(bootstrap_distribution, "bootstrap_distribution")
    _validate_bootstrap_params(len(bootstrap_distribution), bootstrap_conf_level)

    try:
        number_of_bootstrap_samples = bootstrap_distribution.shape[0]
        sample_stat = statistic(sample)

        # Confidence interval alphas (nominal quantiles)
        alphas = np.array(
            [(1 - bootstrap_conf_level) / 2, 1 - (1 - bootstrap_conf_level) / 2]
        )

        # Bias correction: proportion of bootstrap stats less than sample stat
        proportion_less = (
            np.sum(bootstrap_distribution < sample_stat) / number_of_bootstrap_samples
        )

        # Handle corner case: all bootstrap stats on one side
        if proportion_less <= EPSILON:
            # All bootstrap stats >= sample stat: severe positive bias
            proportion_less = EPSILON
            warnings.warn(
                "All bootstrap statistics are greater than or equal to sample statistic. "
                "BCa bias correction may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
        elif proportion_less >= (1 - EPSILON):
            # All bootstrap stats <= sample stat: severe negative bias
            proportion_less = 1 - EPSILON
            warnings.warn(
                "All bootstrap statistics are less than or equal to sample statistic. "
                "BCa bias correction may be unreliable.",
                UserWarning,
                stacklevel=2,
            )

        z0 = norm.ppf(proportion_less)

        # Handle corner case: z0 is infinite
        if not np.isfinite(z0):
            warnings.warn(
                "Bias correction value is not finite. Falling back to percentile method.",
                UserWarning,
                stacklevel=2,
            )
            return estimate_confidence_interval(
                bootstrap_distribution, bootstrap_conf_level
            )

        # Compute acceleration using optimized jackknife
        acceleration, _ = _compute_jackknife_acceleration(
            sample, statistic, n_jobs, jackknife_parallel_threshold
        )

        # Compute BCa-adjusted alphas
        z_alphas = z0 + norm.ppf(alphas)

        # Handle corner case: acceleration too large (denominator → 0)
        denominator = 1 - acceleration * z_alphas
        if np.any(np.abs(denominator) < EPSILON):
            warnings.warn(
                "Acceleration parameter causes numerical instability. "
                "Falling back to percentile method.",
                UserWarning,
                stacklevel=2,
            )
            return estimate_confidence_interval(
                bootstrap_distribution, bootstrap_conf_level
            )

        adjusted_alphas = norm.cdf(z0 + z_alphas / denominator)

        # Clamp adjusted alphas to valid range [0, 1]
        adjusted_alphas = np.clip(adjusted_alphas, 0.0, 1.0)

        # Convert alphas to bootstrap distribution indices
        indices = np.round((number_of_bootstrap_samples - 1) * adjusted_alphas).astype(
            int
        )
        indices = np.clip(indices, 0, number_of_bootstrap_samples - 1)

        # Return BCa confidence interval
        sorted_distribution = np.sort(bootstrap_distribution)
        return sorted_distribution[indices]

    except (ValueError, TypeError, ZeroDivisionError) as e:
        raise NumericalError(
            "Failed to compute BCa confidence interval",
            operation="bca_computation",
            values={
                "sample_size": sample.size,
                "bootstrap_samples": len(bootstrap_distribution),
                "confidence_level": bootstrap_conf_level,
            },
        ) from e


def _create_rng_from_seed(seed_sequence: np.random.SeedSequence) -> np.random.Generator:
    """Create a random number generator from a seed sequence.

    Helper function to enable lazy RNG creation in parallel processing,
    reducing memory overhead by avoiding upfront instantiation of all RNGs.

    Parameters
    ----------
    seed_sequence : np.random.SeedSequence
        Seed sequence for RNG initialization.

    Returns
    -------
    np.random.Generator
        Initialized random number generator.

    Notes
    -----
    Time complexity: O(1).
    Space complexity: O(1).
    """
    return np.random.Generator(np.random.PCG64(seed_sequence))


def _execute_bootstrap_sample(
    sample_function: Callable[
        [np.random.Generator], Union[float, npt.NDArray[np.floating]]
    ],
    seed_sequence: np.random.SeedSequence,
) -> Union[float, npt.NDArray[np.floating]]:
    """Execute a single bootstrap sample with lazy RNG creation.

    Combines RNG creation and sampling to minimize memory footprint
    by creating generators only when needed in each parallel worker.

    Parameters
    ----------
    sample_function : callable
        Function that computes bootstrap statistic.
    seed_sequence : np.random.SeedSequence
        Seed for reproducible RNG initialization.

    Returns
    -------
    float or ndarray
        Bootstrap statistic result.

    Notes
    -----
    Time complexity: O(f) where f is sample_function cost.
    Space complexity: O(1) + O(r) where r is result size.
    """
    rng = _create_rng_from_seed(seed_sequence)
    return sample_function(rng)


@lru_cache(maxsize=2)
def _detect_cpu_count(logical: bool = False) -> int:
    """Detect the number of CPU cores, with a safe fallback.

    Cached at module level since hardware topology does not change at runtime.
    Prefers physical cores for CPU-bound parallel workloads; falls back to
    logical cores and finally to ``1`` if detection fails.

    Parameters
    ----------
    logical : bool, optional
        If ``True``, count logical cores (SMT/hyper-threads). If ``False``
        (default), count physical cores.

    Returns
    -------
    int
        Number of CPU cores, guaranteed to be ``>= 1``.
    """
    count = psutil.cpu_count(logical=logical)
    if count is None and not logical:
        count = psutil.cpu_count(logical=True)
    return int(count) if count and count > 0 else 1


def _resolve_n_workers(n_jobs: int) -> int:
    """Resolve effective worker count from a joblib-style ``n_jobs`` value.

    Follows joblib semantics:

    - ``n_jobs == -1`` -> all physical cores
    - ``n_jobs == -k`` (k > 0) -> ``cpu_count + 1 + n_jobs`` (e.g. -2 = all but one)
    - ``n_jobs >= 1`` -> capped at ``cpu_count``
    - ``n_jobs == 0`` -> invalid

    Parameters
    ----------
    n_jobs : int
        joblib-compatible number of jobs.

    Returns
    -------
    int
        Resolved worker count, ``>= 1``.

    Raises
    ------
    ValidationError
        If ``n_jobs`` is ``0`` or otherwise invalid.
    """
    if n_jobs == 0:
        raise ValidationError(
            "n_jobs must be -k (k>=1) or a positive integer, got 0",
            parameter="n_jobs",
            value=n_jobs,
        )

    cpu = _detect_cpu_count(logical=False)

    if n_jobs < 0:
        # joblib convention: -1 == all cores, -2 == all but one, etc.
        return max(1, cpu + 1 + n_jobs)
    return max(1, min(int(n_jobs), cpu))


def _compute_optimal_batch_size(
    number_of_bootstrap_samples: int,
    sample_size: int,
    n_jobs: int,
    available_memory_gb: Optional[float] = None,
    dtype_bytes: int = DEFAULT_DTYPE_BYTES,
) -> int:
    """Compute an optimal joblib batch size for bootstrap parallel processing.

    Selects a batch size that balances dispatch overhead, memory pressure,
    and worker load-balancing. The algorithm picks a workload-scale base from
    the documented heuristic table, applies memory-tier and sample-complexity
    dampening, then clamps the result by a per-call memory budget and a
    load-balancing upper bound (``>= MIN_BATCHES_PER_WORKER`` chunks per worker).

    Parameters
    ----------
    number_of_bootstrap_samples : int
        Total number of bootstrap iterations. Must be positive.
    sample_size : int
        Size of each sample being resampled. Must be ``>= MIN_SAMPLE_SIZE``.
    n_jobs : int
        Number of parallel workers using joblib conventions
        (``-1`` = all cores, ``-k`` = all but ``k-1``, ``>=1`` = explicit count).
    available_memory_gb : float, optional
        Available system memory in GB. Auto-detected via ``psutil`` if ``None``.
    dtype_bytes : int, optional
        Bytes per resampled element, used for the memory-aware cap.
        Defaults to ``DEFAULT_DTYPE_BYTES`` (8, i.e. ``float64``).

    Returns
    -------
    int
        Optimal batch size, ``>= MIN_BATCH_FLOOR`` and ``<=`` the load-balancing cap.

    Raises
    ------
    ValidationError
        If ``number_of_bootstrap_samples``, ``sample_size`` or ``n_jobs`` is invalid.

    Notes
    -----
    Heuristic table (inclusive right edges, see README):

    - ``N <= 10K`` -> 128 (minimise dispatch overhead)
    - ``10K < N <= 100K`` -> 256 (balance speed/memory)
    - ``100K < N <= 500K`` -> 512 (maximise throughput)
    - ``N > 500K`` -> 1000 (optimise memory)

    Memory-tier dampening:

    - ``< MEMORY_LOW_THRESHOLD`` GB -> cap at ``LOW_MEM_BATCH_CAP`` (64)
    - ``< MEMORY_MODERATE_THRESHOLD`` GB -> cap at ``BATCH_SIZE_MEDIUM`` (256)

    Sample-complexity dampening:

    - ``sample_size > LARGE_SAMPLE_THRESHOLD`` -> divide base by
      ``SAMPLE_COMPLEXITY_DIVISOR``, with ``MIN_BATCH_FLOOR`` as the floor.

    Memory-aware cap (new):

    ``mem_cap = floor(MEM_FRACTION * available_bytes
                      / (sample_size * dtype_bytes * n_workers))``

    Load-balancing cap (new, replaces the previously inverted ``min_batch``):

    ``max_batch = max(MIN_BATCH_FLOOR,
                      N // (n_workers * MIN_BATCHES_PER_WORKER))``

    Final value: ``clip(min(base, mem_cap), MIN_BATCH_FLOOR, max_batch)``.

    Time complexity: ``O(1)``. Space complexity: ``O(1)``.

    Examples
    --------
    >>> # Auto-detect memory and compute batch size for a medium workload
    >>> batch_size = _compute_optimal_batch_size(100_000, 1000, -1)
    >>> batch_size >= 16
    True

    >>> # Explicit memory, massive workload, default float64 elements
    >>> bs = _compute_optimal_batch_size(
    ...     1_000_000, 5_000, -1, available_memory_gb=16.0
    ... )
    >>> bs >= 16
    True
    """
    # Input validation: fail loudly on impossible configurations.
    if number_of_bootstrap_samples <= 0:
        raise ValidationError(
            ERROR_MESSAGES["invalid_bootstrap_samples"],
            parameter="number_of_bootstrap_samples",
            value=number_of_bootstrap_samples,
        )
    if sample_size < MIN_SAMPLE_SIZE:
        raise ValidationError(
            ERROR_MESSAGES["invalid_sample_size"],
            parameter="sample_size",
            value=sample_size,
        )
    if dtype_bytes <= 0:
        raise ValidationError(
            "dtype_bytes must be positive",
            parameter="dtype_bytes",
            value=dtype_bytes,
        )

    # Resolve worker count via joblib conventions (also validates n_jobs).
    n_workers = _resolve_n_workers(n_jobs)

    # Detect available memory once; user-supplied values bypass psutil.
    if available_memory_gb is None:
        available_memory_gb = psutil.virtual_memory().available / (1024**3)

    # Workload-scale base: inclusive right edges align with README table.
    if number_of_bootstrap_samples <= BATCH_SIZE_THRESHOLD_SMALL:
        base_batch = BATCH_SIZE_SMALL
    elif number_of_bootstrap_samples <= BATCH_SIZE_THRESHOLD_MEDIUM:
        base_batch = BATCH_SIZE_MEDIUM
    elif number_of_bootstrap_samples <= BATCH_SIZE_THRESHOLD_LARGE:
        base_batch = BATCH_SIZE_LARGE
    else:
        base_batch = BATCH_SIZE_MASSIVE

    # Memory-tier dampening: throttle base on RAM-constrained systems.
    if available_memory_gb < MEMORY_LOW_THRESHOLD:
        base_batch = min(base_batch, LOW_MEM_BATCH_CAP)
    elif available_memory_gb < MEMORY_MODERATE_THRESHOLD:
        base_batch = min(base_batch, BATCH_SIZE_MEDIUM)

    # Sample-complexity dampening: halve base for very wide samples.
    if sample_size > LARGE_SAMPLE_THRESHOLD:
        base_batch = max(MIN_BATCH_FLOOR, base_batch // SAMPLE_COMPLEXITY_DIVISOR)

    # Memory-aware cap: how many resamples fit into MEM_FRACTION of free RAM,
    # split across workers. Guards against batches that would OOM on big arrays.
    available_bytes = available_memory_gb * (1024**3)
    per_sample_cost = max(1, sample_size * dtype_bytes * n_workers)
    mem_cap = max(
        MIN_BATCH_FLOOR, int(MEM_FRACTION * available_bytes // per_sample_cost)
    )

    # Load-balancing cap: ensure at least MIN_BATCHES_PER_WORKER chunks per worker.
    # This is the *upper* bound (was inverted in the previous implementation).
    max_batch = max(
        MIN_BATCH_FLOOR,
        number_of_bootstrap_samples // (n_workers * MIN_BATCHES_PER_WORKER),
    )

    # Final clamp using native min/max (cheaper than np.clip for scalars).
    candidate = min(base_batch, mem_cap, max_batch)
    return max(MIN_BATCH_FLOOR, candidate)


def bootstrap_resampling(
    sample_function: Callable[
        [np.random.Generator], Union[float, npt.NDArray[np.floating]]
    ],
    number_of_bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: Optional[int] = DEFAULT_SEED,
    n_jobs: int = DEFAULT_N_JOBS,
    batch_size: Optional[Union[int, str]] = None,
    sample_size_hint: Optional[int] = None,
) -> npt.NDArray[np.floating]:
    """Perform bootstrap resampling with optimized parallel processing.

    Efficiently generates bootstrap statistics using lazy RNG creation,
    batch processing, and optimized memory management for large-scale datasets.

    Parameters
    ----------
    sample_function : callable
        Function that takes a NumPy Generator and returns a bootstrap statistic.
        Signature: sample_function(generator: np.random.Generator) -> float or ndarray
    number_of_bootstrap_samples : int, optional
        Number of bootstrap samples to generate. Default is 10000.
    seed : int, optional
        Seed for reproducibility. Default is 42.
    n_jobs : int, optional
        Number of parallel jobs. -1 uses all available cores. Default is -1.
    batch_size : int or str, optional
        Number of samples per batch for parallel processing.
        - None or 'auto': Uses joblib's dynamic batch sizing (default).
        - 'smart': Intelligent batch sizing based on workload and system resources.
        - int: Manual batch size specification.
        Default is None.
    sample_size_hint : int, optional
        Hint about sample size for smart batch sizing optimization.
        Only used when batch_size='smart'. Default is None.

    Returns
    -------
    ndarray
        Array of bootstrap statistics with shape (number_of_bootstrap_samples,).

    Raises
    ------
    ValidationError
        If parameters are invalid.
    NumericalError
        If bootstrap computation fails.

    Examples
    --------
    >>> def sample_mean(rng):
    ...     return rng.normal(0, 1, 100).mean()
    >>> results = bootstrap_resampling(sample_mean, 1000)
    >>> len(results)
    1000

    >>> # Smart mode: auto-optimizes based on workload and system resources
    >>> results = bootstrap_resampling(
    ...     sample_mean, 1000000, batch_size='smart', sample_size_hint=1000
    ... )
    >>> len(results)
    1000000

    >>> # Manual mode: explicit batch size control
    >>> results = bootstrap_resampling(sample_mean, 1000000, batch_size=2000)
    >>> len(results)
    1000000

    Notes
    -----
    **Optimizations:**

    1. **Speed**: Uses lazy RNG generation and batch processing to minimize
       overhead. Suitable for datasets with >1M entries.
    2. **Memory**: Avoids upfront RNG list creation, reducing memory by ~O(n).
       Creates generators on-demand in parallel workers.
    3. **Parallelism**: Uses joblib with optimized batch_size and 'processes'
       backend for CPU-bound bootstrap operations.
    4. **Smart Mode**: Automatically selects optimal batch size based on:
       - Number of bootstrap samples (workload scale)
       - Sample size (memory per operation)
       - Available system memory
       - Number of CPU cores

    Time complexity: O(n * f) where n is bootstrap samples, f is sample function cost.
    Space complexity: O(n) for results only, avoiding intermediate storage.
    """
    _validate_bootstrap_params(number_of_bootstrap_samples, 0.95)

    try:
        # Generate seed sequences lazily to avoid storing all RNGs in memory
        base_seed_sequence = np.random.SeedSequence(seed)
        spawned_seeds = base_seed_sequence.spawn(number_of_bootstrap_samples)

        # Determine effective batch size
        effective_batch_size: Union[int, str]
        if batch_size == "smart":
            # Smart mode: compute optimal batch size
            if sample_size_hint is None:
                # Default to medium workload if no hint provided
                sample_size_hint = 1000
            effective_batch_size = _compute_optimal_batch_size(
                number_of_bootstrap_samples,
                sample_size_hint,
                n_jobs,
            )
        elif batch_size is None:
            # Default: use joblib's auto mode
            effective_batch_size = "auto"
        else:
            # Manual specification
            effective_batch_size = batch_size

        # Configure parallel execution with optimized settings
        # - prefer='processes': Better for CPU-bound bootstrap operations
        # - batch_size: Controls memory/speed tradeoff
        # - return_as='generator': Could be used for streaming, but array is more efficient here
        parallel_executor = Parallel(
            n_jobs=n_jobs,
            prefer="processes",
            batch_size=effective_batch_size,
        )

        # Execute bootstrap samples in parallel with lazy RNG creation
        # Seeds are passed directly, RNGs created in workers
        bootstrap_results = parallel_executor(
            delayed(_execute_bootstrap_sample)(sample_function, seed_seq)
            for seed_seq in spawned_seeds
        )

        # Convert to numpy array efficiently (joblib returns list)
        return np.asarray(bootstrap_results, dtype=np.float64)

    except (ValueError, TypeError) as e:
        raise NumericalError(
            "Failed to perform bootstrap resampling",
            operation="bootstrap_resampling",
            values={"n_samples": number_of_bootstrap_samples, "n_jobs": n_jobs},
        ) from e
