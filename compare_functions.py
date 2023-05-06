import numpy as np


def mean_of_difference(sample_1, sample_2):
    """Calculates mean of difference change. A good default.
    Args:
        sample_1: numpy array of control statistics
        sample_2: numpy array of control statistics
    Returns:
        sample_2 - sample_1
    """
    return np.mean(sample_2 - sample_1)


def difference_of_mean(sample_1, sample_2):
    """Calculates difference of mean change.
    Args:
        sample_1: numpy array of control statistics
        sample_2: numpy array of test statistics
    Returns:
        mean(sample_2) - mean(sample_1)
    """
    return np.mean(sample_2) - np.mean(sample_1)


def median_of_difference(sample_1, sample_2):
    """Calculates median of difference change. A good default.
    Args:
        sample_1: numpy array of control statistics
        sample_2: numpy array of control statistics
    Returns:
        sample_2 - sample_1
    """
    return np.median(sample_2 - sample_1)


def difference_of_median(sample_1, sample_2):
    """Calculates difference of median change.
    Args:
        sample_1: numpy array of control statistics
        sample_2: numpy array of test statistics
    Returns:
        median(sample_2) - median(sample_1)
    """
    return np.median(sample_2) - np.median(sample_1)


def mean_percent_change(sample_1, sample_2):
    """Calculates percent change.
    Args:
        sample_1: numpy array of control statistics
        sample_2: numpy array of test statistics
    Returns:
        (sample_2 - sample_1) / sample_2 * 100
    """
    return (np.mean(sample_2) - np.mean(sample_1)) * 100.0 / abs(np.mean(sample_2))


def difference(sample_1, sample_2):
    """Calculates difference change. A good default.
    Args:
        sample_1: numpy array of control statistics
        sample_2: numpy array of test statistics
    Returns:
        sample_2 - sample_1
    """
    return sample_2 - sample_1


def percent_change(sample_1, sample_2):
    """Calculates percent change.
    Args:
        sample_1: numpy array of control statistics
        sample_2: numpy array of test statistics
    Returns:
        (sample_2 - sample_1) / sample_2 * 100
    """
    return (sample_2 - sample_1) * 100.0 / abs(sample_2)


def ratio(sample_1, sample_2):
    """Calculates ratio between control and control
    Args:
        sample_1: numpy array of control statistics
        sample_2: numpy array of test statistics
    Returns:
        sample_1 / sample_2
    """
    return sample_1 / sample_2


def percent_difference(sample_1, sample_2):
    """Calculates ratio between control and test. Useful when your statistics
        might be close to zero. Provides a symmetric result.
    Args:
        sample_1: numpy array of control statistics
        sample_2: numpy array of test statistics
    Returns:
        (sample_2 - sample_1) / ((sample_1 + sample_2) / 2.0) * 100.0
    """
    return (sample_2 - sample_1) / ((sample_1 + sample_2) / 2.0) * 100.0