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
