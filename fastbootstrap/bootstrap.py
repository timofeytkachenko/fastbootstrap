import warnings
import scipy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from scipy.stats import norm, binom
from numpy.random import binomial
from .compare_functions import difference_of_mean, difference
from scipy.stats import ttest_ind
from typing import Optional, Tuple, Dict, Union, Callable, Iterator
from joblib import Parallel, delayed


def estimate_confidence_interval(
    distribution: np.ndarray, bootstrap_conf_level: float = 0.95
) -> np.ndarray:
    """
    Estimate the confidence interval of a distribution using quantiles.

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
    left_quant = (1 - bootstrap_conf_level) / 2
    right_quant = 1 - (1 - bootstrap_conf_level) / 2
    return np.quantile(distribution, [left_quant, right_quant])


def estimate_p_value(
    bootstrap_difference_distribution: np.ndarray, number_of_bootstrap_samples: int
) -> float:
    """
    Estimate the two-sided p-value from a bootstrap difference distribution.

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
    positions = np.sum(bootstrap_difference_distribution < 0, axis=0)
    return (
        2
        * np.minimum(positions, number_of_bootstrap_samples - positions)
        / number_of_bootstrap_samples
    )


def estimate_bin_params(sample: np.ndarray) -> Tuple[float, int]:
    """
    Estimate histogram bin parameters using the Freedman-Diaconis rule.

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
    q1 = np.quantile(sample, 0.25)
    q3 = np.quantile(sample, 0.75)
    iqr = q3 - q1
    bin_width = (2 * iqr) / (sample.shape[0] ** (1 / 3))
    bin_count = int(np.ceil((sample.max() - sample.min()) / bin_width))
    return bin_width, bin_count


def jackknife_indices(control: np.ndarray) -> Iterator[np.ndarray]:
    """
    Generate jackknife indices for leave-one-out resampling.

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
    base = np.arange(0, len(control))
    return (np.delete(base, i) for i in base)


def bca(
    control: np.ndarray,
    bootstrap_distribution: np.ndarray,
    statistic: Callable = np.mean,
    bootstrap_conf_level: float = 0.95,
) -> np.ndarray:
    """
    Compute the BCa (bias-corrected and accelerated) confidence interval.

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
    number_of_bootstrap_samples = bootstrap_distribution.shape[0]
    sample_stat = statistic(control)

    # alphas for the confidence interval calculation
    alphas = np.array(
        [(1 - bootstrap_conf_level) / 2, 1 - (1 - bootstrap_conf_level) / 2]
    )

    # The bias correction value.
    z0 = norm.ppf(
        np.sum(bootstrap_distribution < sample_stat, axis=0)
        / number_of_bootstrap_samples
    )

    # Statistics of the jackknife distribution
    jackindexes = jackknife_indices(control)
    jstat = [statistic(control[indices]) for indices in jackindexes]
    jmean = np.mean(jstat, axis=0)

    # Acceleration value
    a = np.divide(
        np.sum((jmean - jstat) ** 3, axis=0),
        (6.0 * np.sum((jmean - jstat) ** 2, axis=0) ** 1.5),
    )

    if np.any(np.isnan(a)):
        nanind = np.nonzero(np.isnan(a))
        warnings.warn(
            f"Some acceleration values were undefined. "
            f"This is almost certainly because all values for the statistic were equal. "
            f"Affected confidence intervals will have zero width and may be inaccurate (indexes: {nanind})"
        )

    zs = z0 + norm.ppf(alphas).reshape(alphas.shape + (1,) * z0.ndim)
    avals = norm.cdf(z0 + zs / (1 - a * zs))
    indices = np.round((number_of_bootstrap_samples - 1) * avals)
    indices = np.nan_to_num(indices).astype("int")
    bootstrap_distribution_sorted = np.sort(bootstrap_distribution)
    return bootstrap_distribution_sorted[indices]


def bootstrap_resampling(
    sample_function: Callable,
    number_of_bootstrap_samples: int,
    seed: Optional[int] = None,
    n_jobs: int = -1,
) -> np.ndarray:
    """
    Perform bootstrap resampling with independent random generators per sample.

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
    # Create a base seed sequence to ensure reproducibility if seed is given
    base_seed = np.random.SeedSequence(seed)

    # Spawn a unique SeedSequence for each bootstrap sample to ensure independent random draws
    seeds = base_seed.spawn(number_of_bootstrap_samples)

    # Create a separate Generator for each bootstrap sample
    rngs = [np.random.Generator(np.random.PCG64(s)) for s in seeds]

    # Use joblib.Parallel for concurrent execution
    bootstrap_results = Parallel(n_jobs=n_jobs)(
        delayed(sample_function)(rng) for rng in rngs
    )

    return np.array(bootstrap_results)


def bootstrap_plot(
    bootstrap_distribution: np.ndarray,
    bootstrap_confidence_interval: np.ndarray,
    statistic: Optional[Union[str, Callable]] = None,
    title: str = "Bootstrap",
    two_sample_plot: bool = True,
) -> None:
    """
    Plot a bootstrap distribution with confidence interval markers.

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
    if isinstance(statistic, str):
        xlabel = statistic
    elif hasattr(statistic, "__call__"):
        xlabel = " ".join([i.capitalize() for i in statistic.__name__.split("_")])
    else:
        xlabel = "Stat"

    binwidth, _ = estimate_bin_params(bootstrap_distribution)
    plt.hist(
        bootstrap_distribution,
        bins=np.arange(
            bootstrap_distribution.min(),
            bootstrap_distribution.max() + binwidth,
            binwidth,
        ),
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.axvline(
        x=bootstrap_confidence_interval[0], color="red", linestyle="dashed", linewidth=2
    )
    plt.axvline(
        x=bootstrap_confidence_interval[1], color="red", linestyle="dashed", linewidth=2
    )
    plt.axvline(
        x=np.median(bootstrap_distribution),
        color="black",
        linestyle="dashed",
        linewidth=5,
    )
    if two_sample_plot:
        plt.axvline(x=0, color="white", linewidth=5)
    plt.show()


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
    """
    Perform two-sample bootstrap analysis for comparing two samples.

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

    def sample(generator: np.random.Generator) -> float:
        control_sample = control[
            generator.choice(control.shape[0], control_sample_size, replace=True)
        ]
        treatment_sample = treatment[
            generator.choice(treatment.shape[0], treatment_sample_size, replace=True)
        ]
        return statistic(control_sample, treatment_sample)

    # Set sample sizes
    if sample_size:
        control_sample_size, treatment_sample_size = sample_size, sample_size
    else:
        control_sample_size, treatment_sample_size = (
            control.shape[0],
            treatment.shape[0],
        )

    # Calculate bootstrap distribution
    bootstrap_difference_distribution = bootstrap_resampling(
        sample, number_of_bootstrap_samples, seed
    )

    # Calculate confidence interval and median
    bootstrap_confidence_interval = estimate_confidence_interval(
        bootstrap_difference_distribution, bootstrap_conf_level
    )
    statistic_value = np.median(bootstrap_difference_distribution)

    # Calculate p-value
    p_value = estimate_p_value(
        bootstrap_difference_distribution, number_of_bootstrap_samples
    )

    # Plot if requested
    if plot:
        bootstrap_plot(
            bootstrap_difference_distribution, bootstrap_confidence_interval, statistic
        )

    # Return standardized dictionary
    return {
        "p_value": p_value,
        "statistic_value": statistic_value,  # Renamed from difference_median
        "confidence_interval": bootstrap_confidence_interval,
        "distribution": (
            bootstrap_difference_distribution if return_distribution else None
        ),
    }


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
    """
    Perform Spotify-style two-sample bootstrap for quantile comparisons.

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
    # Set sample sizes
    if sample_size:
        control_sample_size, treatment_sample_size = sample_size, sample_size
    else:
        control_sample_size, treatment_sample_size = (
            control.shape[0],
            treatment.shape[0],
        )

    # Set q2 if not provided
    if q2 is None:
        q2 = q1

    # Sort samples for quantile calculation
    sorted_control = np.sort(control)
    sorted_treatment = np.sort(treatment)

    # Generate bootstrap samples using binomial sampling for quantiles
    treatment_sample_values = sorted_treatment[
        binomial(treatment_sample_size + 1, q2, number_of_bootstrap_samples)
    ]
    control_sample_values = sorted_control[
        binomial(control_sample_size + 1, q1, number_of_bootstrap_samples)
    ]

    # Calculate bootstrap distribution and original statistic
    bootstrap_difference_distribution = statistic(
        control_sample_values, treatment_sample_values
    )
    statistic_value = statistic(  # Renamed from bootstrap_difference
        np.quantile(sorted_control, q1), np.quantile(sorted_treatment, q2)
    )

    # Calculate confidence interval and p-value
    bootstrap_confidence_interval = estimate_confidence_interval(
        bootstrap_difference_distribution, bootstrap_conf_level
    )
    p_value = estimate_p_value(
        bootstrap_difference_distribution, number_of_bootstrap_samples
    )

    # Plot if requested
    if plot:
        bootstrap_plot(
            bootstrap_difference_distribution,
            bootstrap_confidence_interval,
            statistic=statistic,
        )

    # Return standardized dictionary
    return {
        "p_value": p_value,
        "statistic_value": statistic_value,  # Renamed from bootstrap_difference
        "confidence_interval": bootstrap_confidence_interval,
        "distribution": (  # Consistent key name
            bootstrap_difference_distribution if return_distribution else None
        ),
    }


def ab_test_simulation(
    control: np.ndarray,
    treatment: np.ndarray,
    number_of_experiments: int = 2000,
    stat_test: Callable = ttest_ind,
    n_jobs: int = -1,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Simulate multiple A/B tests with parallel p-value computation.

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
    control_size, treatment_size = control.shape[0], treatment.shape[0]

    def experiment() -> float:
        """Run a single experiment, returning a p-value."""
        c_sample = np.random.choice(control, control_size, replace=True)
        t_sample = np.random.choice(treatment, treatment_size, replace=True)
        return stat_test(c_sample, t_sample)[1]

    # Parallelize the experiments
    ab_p_values = Parallel(n_jobs=n_jobs)(
        delayed(experiment)() for _ in range(number_of_experiments)
    )
    ab_p_values = np.array(ab_p_values)

    test_power = np.mean(ab_p_values < 0.05)
    ab_p_values_sorted = np.sort(ab_p_values)
    auc = np.trapz(
        np.arange(ab_p_values_sorted.shape[0]) / ab_p_values_sorted.shape[0],
        ab_p_values_sorted,
    )

    return {
        "ab_p_values": ab_p_values,
        "test_power": test_power,
        "auc": auc,
    }


def plot_cdf(p_values: np.ndarray, label: str, ax: Axes, linewidth: float = 3) -> None:
    """
    Plot the empirical CDF (Cumulative Distribution Function) of p-values.

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
    sorted_p_values = np.sort(p_values)
    position = scipy.stats.rankdata(sorted_p_values, method="ordinal")
    cdf = position / p_values.shape[0]

    sorted_data = np.hstack((sorted_p_values, 1))
    cdf = np.hstack((cdf, 1))

    ax.plot([0, 1], [0, 1], linestyle="--", color="black")
    ax.plot(sorted_data, cdf, label=label, linestyle="solid", linewidth=linewidth)


def plot_summary(aa_p_values: np.ndarray, ab_p_values: np.ndarray) -> None:
    """
    Create comprehensive summary plots for A/A and A/B test p-values.

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
    cdf_h0_title = "Simulated p-value CDFs under H0 (FPR)"
    cdf_h1_title = "Simulated p-value CDFs under H1 (Sensitivity)"
    p_value_h0_title = "Simulated p-values under H0"
    p_value_h1_title = "Simulated p-values under H1"

    test_power = np.mean(ab_p_values < 0.05)

    fig, ax = plt.subplot_mosaic(
        "AB;CD;FF",
        gridspec_kw={"height_ratios": [1, 1, 0.3], "width_ratios": [1, 1]},
        constrained_layout=True,
    )

    ax["A"].set_title(p_value_h0_title, fontsize=10)
    ax["A"].hist(aa_p_values, bins=50, density=True, label=p_value_h0_title, alpha=0.5)
    ax["B"].set_title(p_value_h1_title, fontsize=10)
    ax["B"].hist(ab_p_values, bins=50, density=True, label=p_value_h1_title, alpha=0.5)

    ax["C"].set_title(cdf_h0_title, fontsize=10)
    plot_cdf(aa_p_values, label=cdf_h0_title, ax=ax["C"])

    ax["D"].set_title(cdf_h1_title, fontsize=10)
    ax["D"].axvline(
        0.05, color="black", linestyle="dashed", linewidth=2, label="alpha=0.05"
    )
    plot_cdf(ab_p_values, label=cdf_h1_title, ax=ax["D"])

    ax["F"].set_title("Power", fontsize=10)
    ax["F"].set_xlim(0, 1)
    ax["F"].yaxis.set_tick_params(labelleft=False)
    ax["F"].barh(y=0, width=test_power, label="Power")


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
    """
    Create an interactive quantile-by-quantile comparison with confidence bands.

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
    quantiles_to_compare = np.linspace(q1, q2, n_step)
    statistics, correction_list = list(), list()
    for i, quantile in enumerate(quantiles_to_compare):
        match correction:
            case "bonferroni":
                corr = (1 - bootstrap_conf_level) / n_step
            case "bh":
                corr = ((1 - bootstrap_conf_level) * (i + 1)) / n_step
        correction_list.append(corr)
        # p_value, bootstrap_mean, bootstrap_confidence_interval = (
        stats = spotify_two_sample_bootstrap(
            control,
            treatment,
            q1=quantile,
            q2=quantile,
            statistic=statistic,
            bootstrap_conf_level=1 - corr,
        )
        statistics.append(
            [
                stats["p_value"],
                stats["statistic_value"],
                stats["confidence_interval"][0],
                stats["confidence_interval"][1],
            ]
        )
    statistics = np.array(statistics)

    df = pd.DataFrame(
        {
            "quantile": quantiles_to_compare,
            "p_value": statistics[:, 0],
            "difference": statistics[:, 1],
            "lower_bound": statistics[:, 2],
            "upper_bound": statistics[:, 3],
        }
    )
    df["significance"] = (df["p_value"] < correction_list).astype(str)

    df["ci_upper"] = df["upper_bound"] - df["difference"]
    df["ci_lower"] = df["difference"] - df["lower_bound"]

    fig = go.Figure(
        [
            go.Scatter(
                name="Difference",
                x=df["quantile"],
                y=df["difference"],
                mode="lines",
                line=dict(color="red"),
            ),
            go.Scatter(
                name="Upper Bound",
                x=df["quantile"],
                y=df["upper_bound"],
                mode="lines",
                marker=dict(color="black"),
                line=dict(width=0),
                showlegend=False,
            ),
            go.Scatter(
                name="Lower Bound",
                x=df["quantile"],
                y=df["lower_bound"],
                marker=dict(color="black"),
                line=dict(width=0),
                mode="lines",
                fillcolor="rgba(68, 68, 68, 0.3)",
                fill="tonexty",
                showlegend=False,
            ),
        ]
    )
    fig.add_hline(y=0, line_width=3, line_dash="dash", line_color="black")

    fig.update_traces(
        customdata=df,
        name="Quantile Information",
        hovertemplate="<br>".join(
            [
                "Quantile: %{customdata[0]}",
                "p_value: %{customdata[1]}",
                "Significance: %{customdata[5]}",
                "Difference: %{customdata[2]}",
                "Lower Bound: %{customdata[3]}",
                "Upper Bound: %{customdata[4]}",
            ]
        ),
    )

    fig.show()


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
    """
    Perform one-sample bootstrap to estimate confidence intervals for a statistic.

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

    def sample(generator: np.random.Generator) -> float:
        control_sample = control[
            generator.choice(control.shape[0], size=sample_size, replace=True)
        ]

        # Handle studentized method separately
        if method == "studentized":
            return statistic(control_sample), control_sample.std()
        return statistic(control_sample)

    # Validate method
    if method not in ["bca", "basic", "percentile", "studentized"]:
        raise ValueError(
            "method argument should be one of the following: bca, basic, percentile, studentized"
        )

    # Set sample size
    sample_size = sample_size if sample_size else control.shape[0]

    # Generate bootstrap samples
    bootstrap_stats = bootstrap_resampling(sample, number_of_bootstrap_samples, seed)
    sample_stat = statistic(control)

    # Calculate confidence interval using the specified method
    if method == "bca":
        bootstrap_distribution = bootstrap_stats
        bootstrap_confidence_interval = bca(
            control, bootstrap_distribution, statistic, bootstrap_conf_level
        )
    elif method == "basic":
        bootstrap_distribution = bootstrap_stats
        percentile_bootstrap_confidence_interval = estimate_confidence_interval(
            bootstrap_distribution, bootstrap_conf_level
        )
        bootstrap_confidence_interval = np.array(
            [
                2 * sample_stat - percentile_bootstrap_confidence_interval[1],
                2 * sample_stat - percentile_bootstrap_confidence_interval[0],
            ]
        )
    elif method == "percentile":
        bootstrap_distribution = bootstrap_stats
        bootstrap_confidence_interval = estimate_confidence_interval(
            bootstrap_distribution, bootstrap_conf_level
        )
    elif method == "studentized":
        bootstrap_distribution = bootstrap_stats[:, 0]
        bootstrap_std_distribution = bootstrap_stats[:, 1]
        bootstrap_distribution_std = bootstrap_distribution.std()
        bootstrap_std_errors = bootstrap_std_distribution / np.sqrt(sample_size)
        t_statistics = (bootstrap_distribution - sample_stat) / bootstrap_std_errors
        lower, upper = estimate_confidence_interval(t_statistics, bootstrap_conf_level)
        bootstrap_confidence_interval = np.array(
            [
                sample_stat - bootstrap_distribution_std * upper,
                sample_stat - bootstrap_distribution_std * lower,
            ]
        )

    # Calculate median of the bootstrap distribution
    statistic_value = np.median(bootstrap_distribution)

    # Plot if requested
    if plot:
        bootstrap_plot(
            bootstrap_distribution,
            bootstrap_confidence_interval,
            statistic,
            two_sample_plot=False,
        )

    # Return standardized dictionary
    return {
        "statistic_value": statistic_value,  # Renamed from difference_median
        "confidence_interval": bootstrap_confidence_interval,
        "distribution": (  # Renamed from bootstrap_distribution
            bootstrap_distribution if return_distribution else None
        ),
    }


def spotify_one_sample_bootstrap(
    sample: np.ndarray,
    sample_size: Optional[int] = None,
    q: float = 0.5,
    bootstrap_conf_level: float = 0.95,
    number_of_bootstrap_samples: int = 10000,
    return_distribution: bool = False,
    plot=False,
) -> Dict[str, Union[float, np.ndarray, None]]:
    """
    Perform Spotify-style one-sample bootstrap for quantile estimation.

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
    # Set sample size
    if not sample_size:
        sample_size = sample.shape[0]

    # Sort sample for quantile calculation
    sample_sorted = np.sort(sample)

    # Calculate confidence interval bounds
    left_quant = (1 - bootstrap_conf_level) / 2
    right_quant = 1 - (1 - bootstrap_conf_level) / 2
    ci_indexes = binom.ppf([left_quant, right_quant], sample_size + 1, q)
    bootstrap_confidence_interval = sample_sorted[
        [int(np.floor(ci_indexes[0])), int(np.ceil(ci_indexes[1]))]
    ]

    # Generate bootstrap distribution using binomial sampling
    quantile_indices = binomial(sample_size + 1, q, number_of_bootstrap_samples)
    bootstrap_distribution = sample_sorted[quantile_indices]

    # Calculate the statistic value (quantile)
    statistic_value = np.quantile(
        sample, q
    )  # Use direct quantile instead of bootstrap median

    # Plot if requested
    if plot:
        statistic = f"Quantile_{q}"
        bootstrap_plot(
            bootstrap_distribution,
            bootstrap_confidence_interval,
            statistic,
            two_sample_plot=False,
        )

    # Return standardized dictionary
    return {
        "statistic_value": statistic_value,  # Renamed from bootstrap_quantile
        "confidence_interval": bootstrap_confidence_interval,
        "distribution": (  # Renamed from bootstrap_distribution
            bootstrap_distribution if return_distribution else None
        ),
    }


def poisson_bootstrap(
    control: np.ndarray, treatment: np.ndarray, number_of_bootstrap_samples: int = 10000
) -> float:
    """
    Perform Poisson bootstrap for comparing aggregated values between two samples.

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
    sample_size = np.min([control.shape[0], treatment.shape[0]])
    control_distribution = np.zeros(shape=number_of_bootstrap_samples)
    treatment_distribution = np.zeros(shape=number_of_bootstrap_samples)
    for control_item, treatment_item in zip(
        control[:sample_size], treatment[:sample_size]
    ):
        weights = np.random.poisson(1, number_of_bootstrap_samples)
        control_distribution += control_item * weights
        treatment_distribution += treatment_item * weights

    bootstrap_difference_distribution = treatment_distribution - control_distribution
    p_value = estimate_p_value(
        bootstrap_difference_distribution, number_of_bootstrap_samples
    )
    return p_value


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
    """
    Unified bootstrap function that automatically selects appropriate method based on inputs.

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
    # Handle quantile(s) for Spotify-style bootstrap
    if spotify_style:
        if isinstance(q, tuple) and len(q) == 2:
            q1, q2 = q
        else:
            q1, q2 = q, q

    # Select appropriate bootstrap method based on inputs
    if treatment is None:
        # One-sample bootstrap
        if spotify_style:
            return spotify_one_sample_bootstrap(
                sample=control,
                sample_size=sample_size,
                q=q1,  # Use first quantile
                bootstrap_conf_level=bootstrap_conf_level,
                number_of_bootstrap_samples=number_of_bootstrap_samples,
                return_distribution=return_distribution,
                plot=plot,
            )
        else:
            return one_sample_bootstrap(
                control=control,
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
            result = spotify_two_sample_bootstrap(
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
            result = two_sample_bootstrap(
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

        return result
