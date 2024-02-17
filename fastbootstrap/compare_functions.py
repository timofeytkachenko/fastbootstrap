import numpy as np


def mean_of_difference(control, treatment):
    """Calculates mean of difference change. A good default.
    Args:
        control: numpy array of control statistics
        treatment: numpy array of treatment statistics
    Returns:
        treatment - control
    """
    return np.mean(treatment - control)


def percent_difference_of_mean(control, treatment):
    """Calculates percent difference of median between control and treatment. Useful when your statistics
        might be close to zero. Provides a symmetric result.
    Args:
        control: numpy array of control statistics
        treatment: numpy array of treatment statistics
    Returns:
        (np.mean(treatment) - np.mean(control)) / ((np.mean(control) + np.mean(treatment)) / 2.0) * 100.0
    """
    return (np.mean(treatment) - np.mean(control)) / ((np.mean(control) + np.mean(treatment)) / 2.0) * 100.0


def difference_of_mean(control, treatment):
    """Calculates difference of mean change.
    Args:
        control: numpy array of control statistics
        treatment: numpy array of treatment statistics
    Returns:
        mean(treatment) - mean(control)
    """
    return np.mean(treatment) - np.mean(control)


def median_of_difference(control, treatment):
    """Calculates median of difference change. A good default.
    Args:
        control: numpy array of control statistics
        treatment: numpy array of treatment statistics
    Returns:
        np.median(treatment - control)
    """
    return np.median(treatment - control)


def difference_of_median(control, treatment):
    """Calculates difference of median change.
    Args:
        control: numpy array of control statistics
        treatment: numpy array of treatment statistics
    Returns:
        median(treatment) - median(control)
    """
    return np.median(treatment) - np.median(control)


def percent_difference_of_median(control, treatment):
    """Calculates percent difference of median between control and treatment. Useful when your statistics
        might be close to zero. Provides a symmetric result.
    Args:
        control: numpy array of control statistics
        treatment: numpy array of treatment statistics
    Returns:
        (treatment - control) / ((control + treatment) / 2.0) * 100.0
    """
    return (np.mean(treatment) - np.mean(control)) / ((np.mean(control) + np.mean(treatment)) / 2.0) * 100.0


def mean_percent_change(control, treatment):
    """Calculates percent change.
    Args:
        control: numpy array of control statistics
        treatment: numpy array of treatment statistics
    Returns:
        (treatment - control) / treatment * 100
    """
    return (np.mean(treatment) - np.mean(control)) * 100.0 / abs(np.mean(treatment))


def difference(control, treatment):
    """Calculates difference change. A good default.
    Args:
        control: numpy array of control statistics
        treatment: numpy array of treatment statistics
    Returns:
        treatment - control
    """
    return treatment - control


def percent_change(control, treatment):
    """Calculates percent change.
    Args:
        control: numpy array of control statistics
        treatment: numpy array of treatment statistics
    Returns:
        (treatment - control) / treatment * 100
    """
    return (treatment - control) * 100.0 / abs(treatment)


def percent_change_of_median(control, treatment):
    """Calculates percent change of median.
    Args:
        control: numpy array of control statistics
        treatment: numpy array of treatment statistics
    Returns:
        (np.median(treatment) - np.median(control)) * 100.0 / abs(np.median(treatment))
    """
    return (np.median(treatment) - np.median(control)) * 100.0 / abs(np.median(treatment))


def percent_change_of_mean(control, treatment):
    """Calculates percent change of mean.
    Args:
        control: numpy array of control statistics
        treatment: numpy array of treatment statistics
    Returns:
        (np.mean(treatment) - np.mean(control)) * 100.0 / abs(np.mean(treatment))
    """
    return (np.mean(treatment) - np.mean(control)) * 100.0 / abs(np.mean(treatment))


def ratio(control, treatment):
    """Calculates ratio between control and treatment.
    Args:
        control: numpy array of control statistics
        treatment: numpy array of treatment statistics
    Returns:
        control / treatment
    """
    return control / treatment


def percent_difference(control, treatment):
    """Calculates ratio between control and treatment. Useful when your statistics
        might be close to zero. Provides a symmetric result.
    Args:
        control: numpy array of control statistics
        treatment: numpy array of treatment statistics
    Returns:
        (treatment - control) / ((control + treatment) / 2.0) * 100.0
    """
    return (treatment - control) / ((control + treatment) / 2.0) * 100.0
