"""Comparison functions for statistical bootstrap analysis.

This module provides optimized comparison functions for computing differences
between control and treatment groups in bootstrap analysis.
"""

from typing import Union

import numpy as np
import numpy.typing as npt

from .constants import EPSILON, ERROR_MESSAGES
from .exceptions import NumericalError, ValidationError


def _validate_arrays(
    control: npt.NDArray[np.floating], treatment: npt.NDArray[np.floating]
) -> None:
    """Validate input arrays for comparison functions.

    Parameters
    ----------
    control : ndarray
        Control group array.
    treatment : ndarray
        Treatment group array.

    Raises
    ------
    ValidationError
        If arrays are empty or have invalid shapes.
    """
    if control.size == 0:
        raise ValidationError(ERROR_MESSAGES["empty_array"], parameter="control")
    if treatment.size == 0:
        raise ValidationError(ERROR_MESSAGES["empty_array"], parameter="treatment")


def difference_of_mean(
    control: npt.NDArray[np.floating], treatment: npt.NDArray[np.floating]
) -> float:
    """Calculate the difference in means between treatment and control groups.

    Parameters
    ----------
    control : ndarray
        1D array of control group observations.
    treatment : ndarray
        1D array of treatment group observations.

    Returns
    -------
    float
        The difference in means: mean(treatment) - mean(control).

    Raises
    ------
    ValidationError
        If input arrays are empty.
    NumericalError
        If numerical computation fails.

    Examples
    --------
    >>> control = np.array([1, 2, 3, 4, 5])
    >>> treatment = np.array([2, 3, 4, 5, 6])
    >>> difference_of_mean(control, treatment)
    1.0

    Notes
    -----
    Time complexity: O(n + m) where n, m are array lengths.
    Space complexity: O(1).
    """
    _validate_arrays(control, treatment)

    try:
        return float(np.mean(treatment) - np.mean(control))
    except (ValueError, TypeError) as e:
        raise NumericalError(
            "Failed to compute difference of means",
            operation="difference_of_mean",
            values={"control_shape": control.shape, "treatment_shape": treatment.shape},
        ) from e


def percent_difference_of_mean(
    control: npt.NDArray[np.floating], treatment: npt.NDArray[np.floating]
) -> float:
    """Calculate the symmetric percent difference in means.

    This function computes the percent difference using the average of both
    means as the denominator, providing a symmetric result useful when
    statistics might be close to zero.

    Parameters
    ----------
    control : ndarray
        1D array of control group observations.
    treatment : ndarray
        1D array of treatment group observations.

    Returns
    -------
    float
        The symmetric percent difference:
        (mean(treatment) - mean(control)) / ((mean(control) + mean(treatment)) / 2) * 100.

    Raises
    ------
    ValidationError
        If input arrays are empty.
    NumericalError
        If the average of means is too close to zero or computation fails.

    Examples
    --------
    >>> control = np.array([1, 2, 3])
    >>> treatment = np.array([2, 3, 4])
    >>> percent_difference_of_mean(control, treatment)
    40.0

    Notes
    -----
    Time complexity: O(n + m) where n, m are array lengths.
    Space complexity: O(1).
    """
    _validate_arrays(control, treatment)

    try:
        control_mean = np.mean(control)
        treatment_mean = np.mean(treatment)
        avg_mean = (control_mean + treatment_mean) / 2.0

        if abs(avg_mean) < EPSILON:
            raise NumericalError(
                ERROR_MESSAGES["division_by_zero"],
                operation="percent_difference_of_mean",
                values={"control_mean": control_mean, "treatment_mean": treatment_mean},
            )

        return float((treatment_mean - control_mean) / avg_mean * 100.0)
    except (ValueError, TypeError) as e:
        raise NumericalError(
            "Failed to compute percent difference of means",
            operation="percent_difference_of_mean",
        ) from e


def percent_change_of_mean(
    control: npt.NDArray[np.floating], treatment: npt.NDArray[np.floating]
) -> float:
    """Calculate the percent change in means.

    Parameters
    ----------
    control : ndarray
        1D array of control group observations.
    treatment : ndarray
        1D array of treatment group observations.

    Returns
    -------
    float
        The percent change: (mean(treatment) - mean(control)) / |mean(control)| * 100.

    Raises
    ------
    ValidationError
        If input arrays are empty.
    NumericalError
        If control mean is too close to zero or computation fails.

    Examples
    --------
    >>> control = np.array([1, 2, 3])
    >>> treatment = np.array([2, 3, 4])
    >>> percent_change_of_mean(control, treatment)
    50.0

    Notes
    -----
    Time complexity: O(n + m) where n, m are array lengths.
    Space complexity: O(1).
    """
    _validate_arrays(control, treatment)

    try:
        control_mean = np.mean(control)
        treatment_mean = np.mean(treatment)

        if abs(control_mean) < EPSILON:
            raise NumericalError(
                ERROR_MESSAGES["division_by_zero"],
                operation="percent_change_of_mean",
                values={"control_mean": control_mean},
            )

        return float((treatment_mean - control_mean) / abs(control_mean) * 100.0)
    except (ValueError, TypeError) as e:
        raise NumericalError(
            "Failed to compute percent change of means",
            operation="percent_change_of_mean",
        ) from e


def difference_of_std(
    control: npt.NDArray[np.floating], treatment: npt.NDArray[np.floating]
) -> float:
    """Calculate the difference in standard deviations.

    Parameters
    ----------
    control : ndarray
        1D array of control group observations.
    treatment : ndarray
        1D array of treatment group observations.

    Returns
    -------
    float
        The difference in standard deviations: std(treatment) - std(control).

    Raises
    ------
    ValidationError
        If input arrays are empty.
    NumericalError
        If numerical computation fails.

    Examples
    --------
    >>> control = np.array([1, 2, 3, 4, 5])
    >>> treatment = np.array([2, 3, 4, 5, 6])
    >>> difference_of_std(control, treatment)
    0.0

    Notes
    -----
    Time complexity: O(n + m) where n, m are array lengths.
    Space complexity: O(1).
    """
    _validate_arrays(control, treatment)

    try:
        return float(np.std(treatment, ddof=1) - np.std(control, ddof=1))
    except (ValueError, TypeError) as e:
        raise NumericalError(
            "Failed to compute difference of standard deviations",
            operation="difference_of_std",
        ) from e


def percent_difference_of_std(
    control: npt.NDArray[np.floating], treatment: npt.NDArray[np.floating]
) -> float:
    """Calculate the symmetric percent difference in standard deviations.

    Parameters
    ----------
    control : ndarray
        1D array of control group observations.
    treatment : ndarray
        1D array of treatment group observations.

    Returns
    -------
    float
        The symmetric percent difference in standard deviations:
        (std(treatment) - std(control)) / ((std(control) + std(treatment)) / 2) * 100.

    Raises
    ------
    ValidationError
        If input arrays are empty.
    NumericalError
        If the average of standard deviations is too close to zero or computation fails.

    Examples
    --------
    >>> control = np.array([1, 2, 3])
    >>> treatment = np.array([1, 3, 5])
    >>> round(percent_difference_of_std(control, treatment), 2)
    63.64

    Notes
    -----
    Time complexity: O(n + m) where n, m are array lengths.
    Space complexity: O(1).
    """
    _validate_arrays(control, treatment)

    try:
        control_std = np.std(control, ddof=1)
        treatment_std = np.std(treatment, ddof=1)
        avg_std = (control_std + treatment_std) / 2.0

        if abs(avg_std) < EPSILON:
            raise NumericalError(
                ERROR_MESSAGES["division_by_zero"],
                operation="percent_difference_of_std",
                values={"control_std": control_std, "treatment_std": treatment_std},
            )

        return float((treatment_std - control_std) / avg_std * 100.0)
    except (ValueError, TypeError) as e:
        raise NumericalError(
            "Failed to compute percent difference of standard deviations",
            operation="percent_difference_of_std",
        ) from e


def percent_change_of_std(
    control: npt.NDArray[np.floating], treatment: npt.NDArray[np.floating]
) -> float:
    """Calculate the percent change in standard deviations.

    Parameters
    ----------
    control : ndarray
        1D array of control group observations.
    treatment : ndarray
        1D array of treatment group observations.

    Returns
    -------
    float
        The percent change: (std(treatment) - std(control)) / |std(control)| * 100.

    Raises
    ------
    ValidationError
        If input arrays are empty.
    NumericalError
        If control standard deviation is too close to zero or computation fails.

    Examples
    --------
    >>> control = np.array([1, 2, 3])
    >>> treatment = np.array([1, 3, 5])
    >>> round(percent_change_of_std(control, treatment), 2)
    100.0

    Notes
    -----
    Time complexity: O(n + m) where n, m are array lengths.
    Space complexity: O(1).
    """
    _validate_arrays(control, treatment)

    try:
        control_std = np.std(control, ddof=1)
        treatment_std = np.std(treatment, ddof=1)

        if abs(control_std) < EPSILON:
            raise NumericalError(
                ERROR_MESSAGES["division_by_zero"],
                operation="percent_change_of_std",
                values={"control_std": control_std},
            )

        return float((treatment_std - control_std) / abs(control_std) * 100.0)
    except (ValueError, TypeError) as e:
        raise NumericalError(
            "Failed to compute percent change of standard deviations",
            operation="percent_change_of_std",
        ) from e


def difference_of_median(
    control: npt.NDArray[np.floating], treatment: npt.NDArray[np.floating]
) -> float:
    """Calculate the difference in medians.

    Parameters
    ----------
    control : ndarray
        1D array of control group observations.
    treatment : ndarray
        1D array of treatment group observations.

    Returns
    -------
    float
        The difference in medians: median(treatment) - median(control).

    Raises
    ------
    ValidationError
        If input arrays are empty.
    NumericalError
        If numerical computation fails.

    Examples
    --------
    >>> control = np.array([1, 2, 3, 4, 5])
    >>> treatment = np.array([2, 3, 4, 5, 6])
    >>> difference_of_median(control, treatment)
    1.0

    Notes
    -----
    Time complexity: O(n log n + m log m) where n, m are array lengths.
    Space complexity: O(1).
    """
    _validate_arrays(control, treatment)

    try:
        return float(np.median(treatment) - np.median(control))
    except (ValueError, TypeError) as e:
        raise NumericalError(
            "Failed to compute difference of medians", operation="difference_of_median"
        ) from e


def percent_difference_of_median(
    control: npt.NDArray[np.floating], treatment: npt.NDArray[np.floating]
) -> float:
    """Calculate the symmetric percent difference in medians.

    Parameters
    ----------
    control : ndarray
        1D array of control group observations.
    treatment : ndarray
        1D array of treatment group observations.

    Returns
    -------
    float
        The symmetric percent difference in medians:
        (median(treatment) - median(control)) / ((median(control) + median(treatment)) / 2) * 100.

    Raises
    ------
    ValidationError
        If input arrays are empty.
    NumericalError
        If the average of medians is too close to zero or computation fails.

    Examples
    --------
    >>> control = np.array([1, 2, 3])
    >>> treatment = np.array([2, 3, 4])
    >>> percent_difference_of_median(control, treatment)
    40.0

    Notes
    -----
    Time complexity: O(n log n + m log m) where n, m are array lengths.
    Space complexity: O(1).
    """
    _validate_arrays(control, treatment)

    try:
        control_median = np.median(control)
        treatment_median = np.median(treatment)
        avg_median = (control_median + treatment_median) / 2.0

        if abs(avg_median) < EPSILON:
            raise NumericalError(
                ERROR_MESSAGES["division_by_zero"],
                operation="percent_difference_of_median",
                values={
                    "control_median": control_median,
                    "treatment_median": treatment_median,
                },
            )

        return float((treatment_median - control_median) / avg_median * 100.0)
    except (ValueError, TypeError) as e:
        raise NumericalError(
            "Failed to compute percent difference of medians",
            operation="percent_difference_of_median",
        ) from e


def percent_change_of_median(
    control: npt.NDArray[np.floating], treatment: npt.NDArray[np.floating]
) -> float:
    """Calculate the percent change in medians.

    Parameters
    ----------
    control : ndarray
        1D array of control group observations.
    treatment : ndarray
        1D array of treatment group observations.

    Returns
    -------
    float
        The percent change: (median(treatment) - median(control)) / |median(control)| * 100.

    Raises
    ------
    ValidationError
        If input arrays are empty.
    NumericalError
        If control median is too close to zero or computation fails.

    Examples
    --------
    >>> control = np.array([1, 2, 3])
    >>> treatment = np.array([2, 3, 4])
    >>> percent_change_of_median(control, treatment)
    50.0

    Notes
    -----
    Time complexity: O(n log n + m log m) where n, m are array lengths.
    Space complexity: O(1).
    """
    _validate_arrays(control, treatment)

    try:
        control_median = np.median(control)
        treatment_median = np.median(treatment)

        if abs(control_median) < EPSILON:
            raise NumericalError(
                ERROR_MESSAGES["division_by_zero"],
                operation="percent_change_of_median",
                values={"control_median": control_median},
            )

        return float((treatment_median - control_median) / abs(control_median) * 100.0)
    except (ValueError, TypeError) as e:
        raise NumericalError(
            "Failed to compute percent change of medians",
            operation="percent_change_of_median",
        ) from e


def difference(
    control: Union[npt.NDArray[np.floating], float],
    treatment: Union[npt.NDArray[np.floating], float],
) -> Union[npt.NDArray[np.floating], float]:
    """Calculate element-wise difference between treatment and control.

    This function is optimized for use with Spotify-style bootstrap where
    arrays or scalars are passed for quantile comparisons.

    Parameters
    ----------
    control : ndarray or float
        Control group values or single value.
    treatment : ndarray or float
        Treatment group values or single value.

    Returns
    -------
    ndarray or float
        Element-wise difference: treatment - control.

    Raises
    ------
    ValidationError
        If arrays have incompatible shapes.
    NumericalError
        If numerical computation fails.

    Examples
    --------
    >>> control = np.array([1, 2, 3])
    >>> treatment = np.array([2, 3, 4])
    >>> difference(control, treatment)
    array([1., 1., 1.])

    >>> difference(2.0, 3.0)
    1.0

    Notes
    -----
    Time complexity: O(n) where n is the array length.
    Space complexity: O(n) for array output, O(1) for scalar output.
    """
    try:
        result = treatment - control
        if isinstance(result, np.ndarray):
            return result.astype(np.float64)
        return float(result)
    except (ValueError, TypeError) as e:
        raise NumericalError(
            "Failed to compute difference",
            operation="difference",
            values={"control_type": type(control), "treatment_type": type(treatment)},
        ) from e


def percent_change(
    control: Union[npt.NDArray[np.floating], float],
    treatment: Union[npt.NDArray[np.floating], float],
) -> Union[npt.NDArray[np.floating], float]:
    """Calculate element-wise percent change.

    This function is optimized for use with Spotify-style bootstrap.

    Parameters
    ----------
    control : ndarray or float
        Control group values or single value.
    treatment : ndarray or float
        Treatment group values or single value.

    Returns
    -------
    ndarray or float
        Element-wise percent change: (treatment - control) / |control| * 100.

    Raises
    ------
    ValidationError
        If arrays have incompatible shapes.
    NumericalError
        If control contains values too close to zero or computation fails.

    Examples
    --------
    >>> control = np.array([1, 2, 4])
    >>> treatment = np.array([2, 3, 5])
    >>> percent_change(control, treatment)
    array([100.,  50.,  25.])

    >>> percent_change(2.0, 3.0)
    50.0

    Notes
    -----
    Time complexity: O(n) where n is the array length.
    Space complexity: O(n) for array output, O(1) for scalar output.
    """
    try:
        control_abs = np.abs(control)
        if isinstance(control_abs, np.ndarray):
            if np.any(control_abs < EPSILON):
                raise NumericalError(
                    ERROR_MESSAGES["division_by_zero"], operation="percent_change"
                )
        else:
            if abs(control_abs) < EPSILON:
                raise NumericalError(
                    ERROR_MESSAGES["division_by_zero"],
                    operation="percent_change",
                    values={"control": control},
                )

        result = (treatment - control) / control_abs * 100.0
        if isinstance(result, np.ndarray):
            return result.astype(np.float64)
        return float(result)
    except (ValueError, TypeError) as e:
        raise NumericalError(
            "Failed to compute percent change", operation="percent_change"
        ) from e


def percent_difference(
    control: Union[npt.NDArray[np.floating], float],
    treatment: Union[npt.NDArray[np.floating], float],
) -> Union[npt.NDArray[np.floating], float]:
    """Calculate element-wise symmetric percent difference.

    This function provides a symmetric result useful when statistics might
    be close to zero. Optimized for use with Spotify-style bootstrap.

    Parameters
    ----------
    control : ndarray or float
        Control group values or single value.
    treatment : ndarray or float
        Treatment group values or single value.

    Returns
    -------
    ndarray or float
        Element-wise symmetric percent difference:
        (treatment - control) / ((control + treatment) / 2) * 100.

    Raises
    ------
    ValidationError
        If arrays have incompatible shapes.
    NumericalError
        If the average of control and treatment is too close to zero or computation fails.

    Examples
    --------
    >>> control = np.array([1, 2, 3])
    >>> treatment = np.array([2, 3, 4])
    >>> percent_difference(control, treatment)
    array([66.67, 40.  , 28.57])

    >>> percent_difference(2.0, 3.0)
    40.0

    Notes
    -----
    Time complexity: O(n) where n is the array length.
    Space complexity: O(n) for array output, O(1) for scalar output.
    """
    try:
        avg_value = (control + treatment) / 2.0
        if isinstance(avg_value, np.ndarray):
            if np.any(np.abs(avg_value) < EPSILON):
                raise NumericalError(
                    ERROR_MESSAGES["division_by_zero"], operation="percent_difference"
                )
        else:
            if abs(avg_value) < EPSILON:
                raise NumericalError(
                    ERROR_MESSAGES["division_by_zero"],
                    operation="percent_difference",
                    values={"control": control, "treatment": treatment},
                )

        result = (treatment - control) / avg_value * 100.0
        if isinstance(result, np.ndarray):
            return result.astype(np.float64)
        return float(result)
    except (ValueError, TypeError) as e:
        raise NumericalError(
            "Failed to compute percent difference", operation="percent_difference"
        ) from e
