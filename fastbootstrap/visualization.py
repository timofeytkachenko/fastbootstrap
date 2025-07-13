"""Visualization utilities for bootstrap analysis.

This module provides plotting functions for visualizing bootstrap distributions,
confidence intervals, and statistical comparisons.
"""

from typing import Callable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.graph_objects as go
import scipy.stats
from matplotlib.axes import Axes

from .compare_functions import difference
from .constants import (
    ALPHA_THRESHOLD,
    CORRECTION_METHODS,
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_CORRECTION_METHOD,
    DEFAULT_LINE_WIDTH,
    DEFAULT_QUANTILE_LOWER,
    DEFAULT_QUANTILE_STEPS,
    DEFAULT_QUANTILE_UPPER,
    MEDIAN_COLOR,
    NULL_LINE_COLOR,
    SIGNIFICANCE_COLOR,
)
from .core import estimate_bin_params
from .exceptions import PlottingError, ValidationError


def _validate_plotting_inputs(
    bootstrap_distribution: npt.NDArray[np.floating],
    confidence_interval: npt.NDArray[np.floating],
) -> None:
    """Validate inputs for plotting functions.

    Parameters
    ----------
    bootstrap_distribution : ndarray
        Bootstrap distribution array.
    confidence_interval : ndarray
        Confidence interval array.

    Raises
    ------
    ValidationError
        If inputs are invalid.
    """
    if bootstrap_distribution.size == 0:
        raise ValidationError("Bootstrap distribution cannot be empty")

    if confidence_interval.size != 2:
        raise ValidationError(
            "Confidence interval must have exactly 2 elements",
            parameter="confidence_interval",
            value=confidence_interval.size,
        )


def bootstrap_plot(
    bootstrap_distribution: npt.NDArray[np.floating],
    bootstrap_confidence_interval: npt.NDArray[np.floating],
    statistic: Optional[Union[str, Callable]] = None,
    title: str = "Bootstrap Distribution",
    two_sample_plot: bool = True,
) -> None:
    """Plot bootstrap distribution with confidence interval markers.

    Parameters
    ----------
    bootstrap_distribution : ndarray
        1D array containing the bootstrap distribution.
    bootstrap_confidence_interval : ndarray
        1D array [lower_bound, upper_bound] representing the confidence interval.
    statistic : str or callable, optional
        Statistic name or function for x-axis labeling.
        If callable, uses the function name with formatting.
    title : str, optional
        Plot title. Default is 'Bootstrap Distribution'.
    two_sample_plot : bool, optional
        If True, adds a vertical line at 0 for two-sample visualization.
        Default is True.

    Raises
    ------
    ValidationError
        If inputs are invalid.
    PlottingError
        If plotting fails.

    Examples
    --------
    >>> import numpy as np
    >>> dist = np.random.normal(0, 1, 1000)
    >>> ci = np.array([-1.96, 1.96])
    >>> bootstrap_plot(dist, ci)

    Notes
    -----
    Time complexity: O(n log n) where n is the distribution length.
    Space complexity: O(n) for histogram computation.
    """
    _validate_plotting_inputs(bootstrap_distribution, bootstrap_confidence_interval)

    try:
        # Determine x-axis label
        if isinstance(statistic, str):
            xlabel = statistic
        elif callable(statistic):
            xlabel = " ".join(
                word.capitalize() for word in statistic.__name__.split("_")
            )
        else:
            xlabel = "Statistic"

        # Compute optimal bin parameters
        bin_width, _ = estimate_bin_params(bootstrap_distribution)

        # Create histogram
        bins = np.arange(
            bootstrap_distribution.min(),
            bootstrap_distribution.max() + bin_width,
            bin_width,
        )

        plt.figure(figsize=(10, 6))
        plt.hist(
            bootstrap_distribution,
            bins=bins,
            alpha=0.7,
            edgecolor="black",
        )

        # Add confidence interval lines
        plt.axvline(
            x=bootstrap_confidence_interval[0],
            color=SIGNIFICANCE_COLOR,
            linestyle="--",
            linewidth=DEFAULT_LINE_WIDTH,
            label=f"CI Lower: {bootstrap_confidence_interval[0]:.3f}",
        )
        plt.axvline(
            x=bootstrap_confidence_interval[1],
            color=SIGNIFICANCE_COLOR,
            linestyle="--",
            linewidth=DEFAULT_LINE_WIDTH,
            label=f"CI Upper: {bootstrap_confidence_interval[1]:.3f}",
        )

        # Add median line
        median_val = np.median(bootstrap_distribution)
        plt.axvline(
            x=median_val,
            color=MEDIAN_COLOR,
            linestyle="-",
            linewidth=DEFAULT_LINE_WIDTH,
            label=f"Median: {median_val:.3f}",
        )

        # Add null hypothesis line for two-sample tests
        if two_sample_plot:
            plt.axvline(
                x=0,
                color=NULL_LINE_COLOR,
                linewidth=DEFAULT_LINE_WIDTH + 2,
                label="Null (H₀: difference = 0)",
            )

        plt.title(title, fontsize=14, fontweight="bold")
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        raise PlottingError(
            "Failed to create bootstrap plot",
            plot_type="bootstrap_histogram",
            backend="matplotlib",
        ) from e


def plot_cdf(
    p_values: npt.NDArray[np.floating],
    label: str,
    ax: Axes,
    linewidth: float = DEFAULT_LINE_WIDTH,
) -> None:
    """Plot empirical CDF of p-values.

    Parameters
    ----------
    p_values : ndarray
        1D array of p-values.
    label : str
        Label for the plot legend.
    ax : matplotlib.axes.Axes
        Matplotlib Axes object for plotting.
    linewidth : float, optional
        Line width for the CDF curve. Default is 3.0.

    Raises
    ------
    ValidationError
        If p_values is empty.
    PlottingError
        If plotting fails.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> p_vals = np.array([0.1, 0.05, 0.8, 0.3, 0.01])
    >>> plot_cdf(p_vals, "Test", ax)

    Notes
    -----
    Time complexity: O(n log n) where n is the length of p_values.
    Space complexity: O(n).
    """
    if p_values.size == 0:
        raise ValidationError("P-values array cannot be empty")

    try:
        sorted_p_values = np.sort(p_values)
        position = scipy.stats.rankdata(sorted_p_values, method="ordinal")
        cdf = position / p_values.size

        # Add endpoint for complete CDF
        sorted_data = np.hstack([sorted_p_values, 1.0])
        cdf_values = np.hstack([cdf, 1.0])

        # Plot diagonal reference line
        ax.plot(
            [0, 1], [0, 1], linestyle="--", color="black", alpha=0.5, label="Uniform"
        )

        # Plot empirical CDF
        ax.plot(
            sorted_data, cdf_values, label=label, linestyle="-", linewidth=linewidth
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("P-value")
        ax.set_ylabel("Cumulative Probability")

    except Exception as e:
        raise PlottingError(
            "Failed to plot CDF",
            plot_type="empirical_cdf",
            backend="matplotlib",
        ) from e


def plot_summary(
    aa_p_values: npt.NDArray[np.floating],
    ab_p_values: npt.NDArray[np.floating],
) -> None:
    """Create comprehensive summary plots for A/A and A/B test p-values.

    Parameters
    ----------
    aa_p_values : ndarray
        1D array of p-values from A/A test (null hypothesis).
    ab_p_values : ndarray
        1D array of p-values from A/B test (alternative hypothesis).

    Raises
    ------
    ValidationError
        If p-value arrays are empty.
    PlottingError
        If plotting fails.

    Examples
    --------
    >>> aa_vals = np.random.uniform(0, 1, 1000)
    >>> ab_vals = np.random.beta(2, 5, 1000)
    >>> plot_summary(aa_vals, ab_vals)

    Notes
    -----
    Time complexity: O(n log n) where n is the length of p-value arrays.
    Space complexity: O(n).
    """
    if aa_p_values.size == 0 or ab_p_values.size == 0:
        raise ValidationError("P-values arrays cannot be empty")

    try:
        # Calculate test power
        test_power = float(np.mean(ab_p_values < ALPHA_THRESHOLD))

        # Create subplot layout
        fig, axes = plt.subplot_mosaic(
            "AB;CD;EE",
            gridspec_kw={"height_ratios": [1, 1, 0.3], "width_ratios": [1, 1]},
            figsize=(12, 10),
        )

        # Plot histograms
        axes["A"].hist(
            aa_p_values,
            bins=50,
            density=True,
            alpha=0.7,
            color="blue",
            label="A/A Test (H₀)",
        )
        axes["A"].set_title("P-values under H₀ (Type I Error)", fontsize=12)
        axes["A"].set_xlabel("P-value")
        axes["A"].set_ylabel("Density")
        axes["A"].legend()

        axes["B"].hist(
            ab_p_values,
            bins=50,
            density=True,
            alpha=0.7,
            color="red",
            label="A/B Test (H₁)",
        )
        axes["B"].set_title("P-values under H₁ (Power)", fontsize=12)
        axes["B"].set_xlabel("P-value")
        axes["B"].set_ylabel("Density")
        axes["B"].legend()

        # Plot CDFs
        axes["C"].set_title("CDF under H₀ (False Positive Rate)", fontsize=12)
        plot_cdf(aa_p_values, "A/A Test", axes["C"])
        axes["C"].legend()

        axes["D"].set_title("CDF under H₁ (Sensitivity)", fontsize=12)
        axes["D"].axvline(
            ALPHA_THRESHOLD,
            color="black",
            linestyle="--",
            linewidth=2,
            label=f"α = {ALPHA_THRESHOLD}",
        )
        plot_cdf(ab_p_values, "A/B Test", axes["D"])
        axes["D"].legend()

        # Power bar chart
        axes["E"].barh(0, test_power, color="green", alpha=0.7, height=0.5)
        axes["E"].set_xlim(0, 1)
        axes["E"].set_title(f"Statistical Power: {test_power:.3f}", fontsize=12)
        axes["E"].set_xlabel("Power")
        axes["E"].set_yticks([])

        plt.tight_layout()
        plt.show()

    except Exception as e:
        raise PlottingError(
            "Failed to create summary plot",
            plot_type="summary_plots",
            backend="matplotlib",
        ) from e


def quantile_bootstrap_plot(
    control: npt.NDArray[np.floating],
    treatment: npt.NDArray[np.floating],
    n_step: int = DEFAULT_QUANTILE_STEPS,
    q1: float = DEFAULT_QUANTILE_LOWER,
    q2: float = DEFAULT_QUANTILE_UPPER,
    bootstrap_conf_level: float = DEFAULT_CONFIDENCE_LEVEL,
    statistic: Callable[
        [npt.NDArray[np.floating], npt.NDArray[np.floating]], npt.NDArray[np.floating]
    ] = difference,
    correction: str = DEFAULT_CORRECTION_METHOD,
) -> None:
    """Create interactive quantile-by-quantile comparison plot.

    Parameters
    ----------
    control : ndarray
        1D array containing the control sample.
    treatment : ndarray
        1D array containing the treatment sample.
    n_step : int, optional
        Number of quantiles to compare. Default is 20.
    q1 : float, optional
        Lower quantile bound. Default is 0.01.
    q2 : float, optional
        Upper quantile bound. Default is 0.99.
    bootstrap_conf_level : float, optional
        Base confidence level. Default is 0.95.
    statistic : callable, optional
        Function to compare quantiles. Default is difference.
    correction : str, optional
        Multiple testing correction method. Default is 'bh'.

    Raises
    ------
    ValidationError
        If inputs are invalid.
    PlottingError
        If plotting fails.

    Examples
    --------
    >>> control = np.random.normal(0, 1, 1000)
    >>> treatment = np.random.normal(0.5, 1, 1000)
    >>> quantile_bootstrap_plot(control, treatment)

    Notes
    -----
    Time complexity: O(n * m) where n is n_step, m is bootstrap samples.
    Space complexity: O(n).
    """
    # Validate inputs
    if control.size == 0 or treatment.size == 0:
        raise ValidationError("Control and treatment arrays cannot be empty")

    if not (0 < q1 < q2 < 1):
        raise ValidationError("Quantile bounds must satisfy 0 < q1 < q2 < 1")

    if correction not in CORRECTION_METHODS:
        raise ValidationError(
            f"Invalid correction method. Choose from: {CORRECTION_METHODS}",
            parameter="correction",
            value=correction,
        )

    try:
        # Import here to avoid circular imports
        from .methods import spotify_two_sample_bootstrap

        quantiles_to_compare = np.linspace(q1, q2, n_step)
        statistics = []
        correction_values = []

        for i, quantile in enumerate(quantiles_to_compare):
            # Calculate correction factor
            if correction == "bonferroni":
                corr_factor = (1 - bootstrap_conf_level) / n_step
            elif correction == "bh":  # Benjamini-Hochberg
                corr_factor = ((1 - bootstrap_conf_level) * (i + 1)) / n_step
            else:
                corr_factor = 1 - bootstrap_conf_level

            correction_values.append(corr_factor)

            # Perform bootstrap analysis
            result = spotify_two_sample_bootstrap(
                control=control,
                treatment=treatment,
                q1=quantile,
                q2=quantile,
                statistic=statistic,
                bootstrap_conf_level=1 - corr_factor,
            )

            statistics.append(
                [
                    result["p_value"],
                    result["statistic_value"],
                    result["confidence_interval"][0],
                    result["confidence_interval"][1],
                ]
            )

        statistics = np.array(statistics)

        # Create DataFrame for plotting
        df = pd.DataFrame(
            {
                "quantile": quantiles_to_compare,
                "p_value": statistics[:, 0],
                "difference": statistics[:, 1],
                "lower_bound": statistics[:, 2],
                "upper_bound": statistics[:, 3],
            }
        )

        # Add significance column
        df["significance"] = (df["p_value"] < correction_values).astype(str)

        # Create interactive plot
        fig = go.Figure()

        # Add difference line
        fig.add_trace(
            go.Scatter(
                x=df["quantile"],
                y=df["difference"],
                mode="lines+markers",
                name="Difference",
                line=dict(color="red", width=2),
            )
        )

        # Add confidence band
        fig.add_trace(
            go.Scatter(
                x=df["quantile"],
                y=df["upper_bound"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df["quantile"],
                y=df["lower_bound"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(68, 68, 68, 0.3)",
                name="Confidence Band",
            )
        )

        # Add horizontal line at zero
        fig.add_hline(y=0, line_width=3, line_dash="dash", line_color="black")

        # Update layout
        fig.update_layout(
            title="Quantile-by-Quantile Bootstrap Comparison",
            xaxis_title="Quantile",
            yaxis_title="Difference",
            hovermode="x unified",
            showlegend=True,
        )

        # Add custom hover data
        fig.update_traces(
            hovertemplate="<br>".join(
                [
                    "Quantile: %{x:.3f}",
                    "Difference: %{y:.3f}",
                    "P-value: %{customdata[0]:.4f}",
                    "Significant: %{customdata[1]}",
                ]
            ),
            customdata=df[["p_value", "significance"]].values,
        )

        fig.show()

    except Exception as e:
        raise PlottingError(
            "Failed to create quantile bootstrap plot",
            plot_type="quantile_comparison",
            backend="plotly",
        ) from e
