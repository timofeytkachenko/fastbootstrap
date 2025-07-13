"""Utility functions for fastbootstrap package.

This module provides utility functions for Jupyter notebook integration,
data validation, and helper functions for bootstrap analysis.
"""

from typing import Optional

import numpy as np
import numpy.typing as npt
from IPython.display import Markdown, display

from .constants import ALPHA_THRESHOLD, JUPYTER_STYLES
from .exceptions import ValidationError


def display_significance_result(
    p_value: float,
    alpha: float = ALPHA_THRESHOLD,
    custom_message: Optional[str] = None,
) -> None:
    """Display significance test results in Jupyter notebook with styled alerts.

    Parameters
    ----------
    p_value : float
        The p-value from the statistical test.
    alpha : float, optional
        The significance threshold. Default is 0.05.
    custom_message : str, optional
        Custom message to display. If None, uses default message.

    Raises
    ------
    ValidationError
        If p_value is not a valid probability or alpha is not between 0 and 1.

    Examples
    --------
    >>> display_significance_result(0.03)  # Will show green success alert
    >>> display_significance_result(0.07)  # Will show red warning alert

    Notes
    -----
    This function is designed for use in Jupyter notebooks and requires
    IPython to be available.
    """
    # Validate inputs
    if not (0 <= p_value <= 1):
        raise ValidationError(
            "P-value must be between 0 and 1",
            parameter="p_value",
            value=p_value,
        )

    if not (0 < alpha < 1):
        raise ValidationError(
            "Alpha must be between 0 and 1",
            parameter="alpha",
            value=alpha,
        )

    # Determine significance and message
    is_significant = p_value < alpha

    if custom_message is None:
        if is_significant:
            message = f"Result is significant (p-value = {p_value:.4f} < α = {alpha})"
        else:
            message = (
                f"Result is not significant (p-value = {p_value:.4f} ≥ α = {alpha})"
            )
    else:
        message = custom_message

    # Select appropriate style
    if is_significant:
        style_template = JUPYTER_STYLES["success"]
    else:
        style_template = JUPYTER_STYLES["warning"]

    # Create styled HTML
    styled_html = style_template.replace("p-value < 0.05", message)
    styled_html = styled_html.replace("p-value > 0.05", message)
    styled_html = styled_html.replace(
        "significant", "significant" if is_significant else "not significant"
    )

    # Display the result
    display(Markdown(styled_html))


def display_bootstrap_summary(
    result: dict,
    title: Optional[str] = None,
    show_distribution: bool = False,
) -> None:
    """Display a formatted summary of bootstrap results in Jupyter notebook.

    Parameters
    ----------
    result : dict
        Dictionary containing bootstrap results with keys like 'p_value',
        'statistic_value', 'confidence_interval', etc.
    title : str, optional
        Title for the summary. If None, uses a default title.
    show_distribution : bool, optional
        Whether to display information about the distribution. Default is False.

    Examples
    --------
    >>> result = {'p_value': 0.03, 'statistic_value': 1.5, 'confidence_interval': [0.1, 2.9]}
    >>> display_bootstrap_summary(result, title="Two-Sample Bootstrap Results")

    Notes
    -----
    This function is designed for use in Jupyter notebooks.
    """
    if title is None:
        title = "Bootstrap Analysis Results"

    # Create summary markdown
    summary_lines = [f"## {title}", ""]

    # Add p-value if available
    if "p_value" in result:
        p_value = result["p_value"]
        summary_lines.extend([f"**P-value:** {p_value:.4f}", ""])

        # Add significance assessment
        if p_value < ALPHA_THRESHOLD:
            summary_lines.append("✅ **Result is statistically significant**")
        else:
            summary_lines.append("❌ **Result is not statistically significant**")
        summary_lines.append("")

    # Add statistic value
    if "statistic_value" in result:
        summary_lines.extend(
            [f"**Statistic Value:** {result['statistic_value']:.4f}", ""]
        )

    # Add confidence interval
    if "confidence_interval" in result:
        ci = result["confidence_interval"]
        summary_lines.extend(
            [f"**Confidence Interval:** [{ci[0]:.4f}, {ci[1]:.4f}]", ""]
        )

    # Add distribution info if requested
    if (
        show_distribution
        and "distribution" in result
        and result["distribution"] is not None
    ):
        dist = result["distribution"]
        summary_lines.extend(
            [
                "### Distribution Statistics",
                f"- **Mean:** {np.mean(dist):.4f}",
                f"- **Std:** {np.std(dist):.4f}",
                f"- **Min:** {np.min(dist):.4f}",
                f"- **Max:** {np.max(dist):.4f}",
                f"- **Samples:** {len(dist)}",
                "",
            ]
        )

    # Display the summary
    display(Markdown("\n".join(summary_lines)))


def validate_arrays_compatible(
    *arrays: npt.NDArray[np.floating],
    min_length: int = 2,
    check_finite: bool = True,
) -> None:
    """Validate that arrays are compatible for bootstrap analysis.

    Parameters
    ----------
    *arrays : ndarray
        Variable number of arrays to validate.
    min_length : int, optional
        Minimum required length for each array. Default is 2.
    check_finite : bool, optional
        Whether to check for finite values. Default is True.

    Raises
    ------
    ValidationError
        If any array fails validation.

    Examples
    --------
    >>> arr1 = np.array([1, 2, 3, 4, 5])
    >>> arr2 = np.array([2, 3, 4, 5, 6])
    >>> validate_arrays_compatible(arr1, arr2)
    """
    for i, array in enumerate(arrays):
        array_name = f"array_{i}"

        # Check if array is empty
        if array.size == 0:
            raise ValidationError(
                f"Array {i} is empty",
                parameter=array_name,
                value=array.size,
            )

        # Check minimum length
        if len(array) < min_length:
            raise ValidationError(
                f"Array {i} has insufficient length",
                parameter=array_name,
                value=len(array),
                expected=f"at least {min_length}",
            )

        # Check for finite values
        if check_finite and not np.all(np.isfinite(array)):
            raise ValidationError(
                f"Array {i} contains non-finite values",
                parameter=array_name,
                value="contains NaN or inf",
            )


def format_p_value(p_value: float, threshold: float = 0.001) -> str:
    """Format p-value for display with appropriate precision.

    Parameters
    ----------
    p_value : float
        The p-value to format.
    threshold : float, optional
        Threshold below which to display as "< threshold". Default is 0.001.

    Returns
    -------
    str
        Formatted p-value string.

    Examples
    --------
    >>> format_p_value(0.045)
    '0.045'
    >>> format_p_value(0.0005)
    '< 0.001'
    >>> format_p_value(0.12345)
    '0.123'
    """
    if p_value < threshold:
        return f"< {threshold}"
    elif p_value < 0.01:
        return f"{p_value:.4f}"
    elif p_value < 0.1:
        return f"{p_value:.3f}"
    else:
        return f"{p_value:.3f}"


def calculate_effect_size(
    control: npt.NDArray[np.floating],
    treatment: npt.NDArray[np.floating],
    method: str = "cohen_d",
) -> float:
    """Calculate effect size between two samples.

    Parameters
    ----------
    control : ndarray
        Control group sample.
    treatment : ndarray
        Treatment group sample.
    method : str, optional
        Method for calculating effect size. Options: 'cohen_d', 'glass_delta'.
        Default is 'cohen_d'.

    Returns
    -------
    float
        The calculated effect size.

    Raises
    ------
    ValidationError
        If method is not supported or arrays are invalid.

    Examples
    --------
    >>> control = np.array([1, 2, 3, 4, 5])
    >>> treatment = np.array([2, 3, 4, 5, 6])
    >>> effect_size = calculate_effect_size(control, treatment)
    >>> isinstance(effect_size, float)
    True
    """
    validate_arrays_compatible(control, treatment)

    if method not in ["cohen_d", "glass_delta"]:
        raise ValidationError(
            f"Unsupported effect size method: {method}",
            parameter="method",
            value=method,
            expected="'cohen_d' or 'glass_delta'",
        )

    mean_diff = np.mean(treatment) - np.mean(control)

    if method == "cohen_d":
        # Cohen's d: pooled standard deviation
        pooled_std = np.sqrt(
            (
                (len(control) - 1) * np.var(control, ddof=1)
                + (len(treatment) - 1) * np.var(treatment, ddof=1)
            )
            / (len(control) + len(treatment) - 2)
        )
        return float(mean_diff / pooled_std)

    elif method == "glass_delta":
        # Glass's delta: control group standard deviation
        control_std = np.std(control, ddof=1)
        if control_std == 0:
            raise ValidationError(
                "Control group standard deviation is zero",
                parameter="control",
                value=control_std,
            )
        return float(mean_diff / control_std)


def interpret_effect_size(effect_size: float, method: str = "cohen_d") -> str:
    """Interpret effect size magnitude according to standard conventions.

    Parameters
    ----------
    effect_size : float
        The effect size value.
    method : str, optional
        The method used to calculate effect size. Default is 'cohen_d'.

    Returns
    -------
    str
        Interpretation of the effect size magnitude.

    Examples
    --------
    >>> interpret_effect_size(0.2)
    'small'
    >>> interpret_effect_size(0.5)
    'medium'
    >>> interpret_effect_size(0.8)
    'large'
    """
    abs_effect = abs(effect_size)

    if method == "cohen_d":
        if abs_effect < 0.2:
            return "negligible"
        elif abs_effect < 0.5:
            return "small"
        elif abs_effect < 0.8:
            return "medium"
        else:
            return "large"

    elif method == "glass_delta":
        # Similar thresholds for Glass's delta
        if abs_effect < 0.2:
            return "negligible"
        elif abs_effect < 0.5:
            return "small"
        elif abs_effect < 0.8:
            return "medium"
        else:
            return "large"

    else:
        return "unknown"


def create_sample_data(
    n_control: int = 100,
    n_treatment: int = 100,
    control_params: tuple[float, float] = (0.0, 1.0),
    treatment_params: tuple[float, float] = (0.5, 1.0),
    distribution: str = "normal",
    seed: Optional[int] = None,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Create sample data for bootstrap analysis demonstrations.

    Parameters
    ----------
    n_control : int, optional
        Sample size for control group. Default is 100.
    n_treatment : int, optional
        Sample size for treatment group. Default is 100.
    control_params : tuple, optional
        Parameters for control distribution (mean, std). Default is (0.0, 1.0).
    treatment_params : tuple, optional
        Parameters for treatment distribution (mean, std). Default is (0.5, 1.0).
    distribution : str, optional
        Distribution type: 'normal', 'exponential', 'uniform'. Default is 'normal'.
    seed : int, optional
        Random seed for reproducibility. Default is None.

    Returns
    -------
    tuple
        Tuple containing (control_sample, treatment_sample).

    Examples
    --------
    >>> control, treatment = create_sample_data(50, 50, seed=42)
    >>> len(control) == 50 and len(treatment) == 50
    True
    """
    if seed is not None:
        np.random.seed(seed)

    if distribution == "normal":
        control = np.random.normal(control_params[0], control_params[1], n_control)
        treatment = np.random.normal(
            treatment_params[0], treatment_params[1], n_treatment
        )
    elif distribution == "exponential":
        control = np.random.exponential(control_params[0], n_control)
        treatment = np.random.exponential(treatment_params[0], n_treatment)
    elif distribution == "uniform":
        control = np.random.uniform(control_params[0], control_params[1], n_control)
        treatment = np.random.uniform(
            treatment_params[0], treatment_params[1], n_treatment
        )
    else:
        raise ValidationError(
            f"Unsupported distribution: {distribution}",
            parameter="distribution",
            value=distribution,
            expected="'normal', 'exponential', or 'uniform'",
        )

    return control, treatment


# Maintain backward compatibility with the original function
def display_markdown_cell_by_significance(p_value: float) -> None:
    """Display significance result (legacy function for backward compatibility).

    Parameters
    ----------
    p_value : float
        The p-value from the statistical test.

    Notes
    -----
    This function is deprecated. Use display_significance_result instead.
    """
    display_significance_result(p_value)
