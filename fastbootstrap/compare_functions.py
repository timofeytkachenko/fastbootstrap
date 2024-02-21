import numpy as np


def difference_of_mean(control: np.ndarray, treatment: np.ndarray) -> float:
    """Calculates difference of mean change.

    Args:
        control (ndarray): numpy array of control statistics
        treatment (ndarray)t: numpy array of treatment statistics

    Returns:
        mean(treatment) - mean(control)

    """

    return np.mean(treatment) - np.mean(control)


def percent_difference_of_mean(control: np.ndarray, treatment: np.ndarray) -> float:
    """Calculates percent difference of mean between control and treatment.
        Useful when your statistics might be close to zero. Provides a symmetric result.

    Args:
        control (ndarray): numpy array of control statistics
        treatment (ndarray): numpy array of treatment statistics

    Returns:
        (np.mean(treatment) - np.mean(control)) / ((np.mean(control) + np.mean(treatment)) / 2.0) * 100.0

    """

    return (
        (np.mean(treatment) - np.mean(control))
        / ((np.mean(control) + np.mean(treatment)) / 2.0)
        * 100.0
    )


def percent_change_of_mean(control: np.ndarray, treatment: np.ndarray) -> float:
    """Calculates percent change of mean.
    Args:
        control (ndarray): numpy array of control statistics
        treatment (ndarray): numpy array of treatment statistics

    Returns:
        (np.mean(treatment) - np.mean(control)) * 100.0 / abs(np.mean(control))

    """

    return (np.mean(treatment) - np.mean(control)) * 100.0 / abs(np.mean(control))


def difference_of_std(control, treatment):
    """Calculates difference of standard deviation change.

    Args:
        control (ndarray): numpy array of control statistics
        treatment (ndarray): numpy array of treatment statistics

    Returns:
        std(treatment) - std(control)

    """

    return np.std(treatment) - np.std(control)


def percent_difference_of_std(control: np.ndarray, treatment: np.ndarray) -> float:
    """Calculates percent difference of standard deviation between control and treatment.
        Useful when your statistics might be close to zero. Provides a symmetric result.

    Args:
        control (ndarray): numpy array of control statistics
        treatment (ndarray): numpy array of treatment statistics

    Returns:
        (np.std(treatment) - np.std(control)) / ((np.std(control) + np.std(treatment)) / 2.0) * 100.0

    """

    return (
        (np.std(treatment) - np.std(control))
        / ((np.std(control) + np.std(treatment)) / 2.0)
        * 100.0
    )


def percent_change_of_std(control: np.ndarray, treatment: np.ndarray) -> float:
    """Calculates percent change of standard deviation.

    Args:
        control (ndarray): numpy array of control statistics
        treatment (ndarray): numpy array of treatment statistics

    Returns:
        (np.std(treatment) - np.std(control)) * 100.0 / abs(np.std(control))

    """

    return (np.std(treatment) - np.std(control)) * 100.0 / abs(np.std(control))


def difference_of_median(control: np.ndarray, treatment: np.ndarray) -> float:
    """Calculates difference of median change.

    Args:
        control (ndarray): numpy array of control statistics
        treatment (ndarray): numpy array of treatment statistics

    Returns:
        median(treatment) - median(control)

    """

    return np.median(treatment) - np.median(control)


def percent_difference_of_median(control: np.ndarray, treatment: np.ndarray) -> float:
    """Calculates percent difference of median between control and treatment.
        Useful when your statistics might be close to zero. Provides a symmetric result.

    Args:
        control (ndarray): numpy array of control statistics
        treatment (ndarray): numpy array of treatment statistics

    Returns:
        (np.median(treatment) - np.median(control)) / ((np.median(control) + np.median(treatment)) / 2.0) * 100.0

    """

    return (
        (np.median(treatment) - np.median(control))
        / ((np.median(control) + np.median(treatment)) / 2.0)
        * 100.0
    )


def percent_change_of_median(control: np.ndarray, treatment: np.ndarray) -> float:
    """Calculates percent change of median.

    Args:
        control (ndarray): numpy array of control statistics
        treatment (ndarray): numpy array of treatment statistics

    Returns: (np.median(treatment) - np.median(control)) * 100.0 / abs(np.median(control))

    """

    return (np.median(treatment) - np.median(control)) * 100.0 / abs(np.median(control))


def difference(control: np.ndarray, treatment: np.ndarray) -> np.ndarray:
    """Calculates difference change. Useful with Spotify Bootstrap.

    Args:
        control (ndarray): numpy array of control statistics
        treatment (ndarray): numpy array of treatment statistics

    Returns:
        treatment - control

    """

    return treatment - control


def percent_change(control: np.ndarray, treatment: np.ndarray) -> np.ndarray:
    """Calculates percent change. Useful with Spotify Bootstrap.

    Args:
        control (ndarray): numpy array of control statistics
        treatment (ndarray): numpy array of treatment statistics

    Returns:
        (treatment - control) / treatment * 100

    """

    return (treatment - control) * 100.0 / abs(control)


def percent_difference(control: np.ndarray, treatment: np.ndarray) -> np.ndarray:
    """Calculates ratio between control and treatment. Useful when your statistics
        might be close to zero. Provides a symmetric result. Useful with Spotify Bootstrap.

    Args:
        control: numpy array of control statistics
        treatment: numpy array of treatment statistics

    Returns:
        (treatment - control) / ((control + treatment) / 2.0) * 100.0

    """

    return (treatment - control) / ((control + treatment) / 2.0) * 100.0
