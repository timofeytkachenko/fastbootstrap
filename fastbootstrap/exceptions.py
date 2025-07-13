"""Custom exceptions for fastbootstrap package.

This module defines custom exception classes for better error handling
and debugging throughout the package.
"""

from typing import Any, Optional


class FastBootstrapError(Exception):
    """Base exception class for all fastbootstrap errors."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        """Initialize the exception.

        Parameters
        ----------
        message : str
            The error message.
        details : dict[str, Any], optional
            Additional details about the error.
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        """Return string representation of the exception."""
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} (Details: {details_str})"
        return self.message


class ValidationError(FastBootstrapError):
    """Exception raised for input validation errors."""

    def __init__(
        self,
        message: str,
        parameter: Optional[str] = None,
        value: Optional[Any] = None,
        expected: Optional[str] = None,
    ) -> None:
        """Initialize validation error.

        Parameters
        ----------
        message : str
            The error message.
        parameter : str, optional
            The parameter that caused the validation error.
        value : Any, optional
            The invalid value.
        expected : str, optional
            Description of what was expected.
        """
        details = {}
        if parameter:
            details["parameter"] = parameter
        if value is not None:
            details["value"] = value
        if expected:
            details["expected"] = expected

        super().__init__(message, details)


class InsufficientDataError(FastBootstrapError):
    """Exception raised when there is insufficient data for bootstrap analysis."""

    def __init__(
        self,
        message: str,
        sample_size: Optional[int] = None,
        min_required: Optional[int] = None,
    ) -> None:
        """Initialize insufficient data error.

        Parameters
        ----------
        message : str
            The error message.
        sample_size : int, optional
            The actual sample size.
        min_required : int, optional
            The minimum required sample size.
        """
        details = {}
        if sample_size is not None:
            details["sample_size"] = sample_size
        if min_required is not None:
            details["min_required"] = min_required

        super().__init__(message, details)


class NumericalError(FastBootstrapError):
    """Exception raised for numerical computation errors."""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        values: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize numerical error.

        Parameters
        ----------
        message : str
            The error message.
        operation : str, optional
            The operation that caused the error.
        values : dict[str, Any], optional
            Values involved in the computation.
        """
        details = {}
        if operation:
            details["operation"] = operation
        if values:
            details.update(values)

        super().__init__(message, details)


class BootstrapMethodError(FastBootstrapError):
    """Exception raised for bootstrap method-specific errors."""

    def __init__(
        self,
        message: str,
        method: Optional[str] = None,
        available_methods: Optional[list[str]] = None,
    ) -> None:
        """Initialize bootstrap method error.

        Parameters
        ----------
        message : str
            The error message.
        method : str, optional
            The invalid method.
        available_methods : list[str], optional
            Available methods.
        """
        details = {}
        if method:
            details["method"] = method
        if available_methods:
            details["available_methods"] = available_methods

        super().__init__(message, details)


class ConvergenceError(FastBootstrapError):
    """Exception raised when bootstrap computation doesn't converge."""

    def __init__(
        self,
        message: str,
        iterations: Optional[int] = None,
        tolerance: Optional[float] = None,
    ) -> None:
        """Initialize convergence error.

        Parameters
        ----------
        message : str
            The error message.
        iterations : int, optional
            Number of iterations attempted.
        tolerance : float, optional
            The convergence tolerance.
        """
        details = {}
        if iterations is not None:
            details["iterations"] = iterations
        if tolerance is not None:
            details["tolerance"] = tolerance

        super().__init__(message, details)


class MemoryError(FastBootstrapError):
    """Exception raised when memory requirements exceed limits."""

    def __init__(
        self,
        message: str,
        required_memory: Optional[int] = None,
        available_memory: Optional[int] = None,
    ) -> None:
        """Initialize memory error.

        Parameters
        ----------
        message : str
            The error message.
        required_memory : int, optional
            Required memory in MB.
        available_memory : int, optional
            Available memory in MB.
        """
        details = {}
        if required_memory is not None:
            details["required_memory_mb"] = required_memory
        if available_memory is not None:
            details["available_memory_mb"] = available_memory

        super().__init__(message, details)


class PlottingError(FastBootstrapError):
    """Exception raised for plotting and visualization errors."""

    def __init__(
        self,
        message: str,
        plot_type: Optional[str] = None,
        backend: Optional[str] = None,
    ) -> None:
        """Initialize plotting error.

        Parameters
        ----------
        message : str
            The error message.
        plot_type : str, optional
            The type of plot that failed.
        backend : str, optional
            The plotting backend used.
        """
        details = {}
        if plot_type:
            details["plot_type"] = plot_type
        if backend:
            details["backend"] = backend

        super().__init__(message, details)
