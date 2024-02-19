import scipy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from scipy.stats import norm, binom
from numpy.random import binomial
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from .compare_functions import difference_of_mean, difference
from scipy.stats import ttest_ind
from typing import Tuple, Union, Callable

plt.style.use('ggplot')


def estimate_confidence_interval(distribution: np.ndarray, bootstrap_conf_level: float = 0.95) -> np.ndarray:
    """Estimation confidence interval of distribution

    Args:
        distribution (ndarray): 1D array containing distribution
        bootstrap_conf_level (float): Confidence level. Defaults to 0.95

    Returns:
        ndarray: 1D array containing confidence interval

    """

    left_quant = (1 - bootstrap_conf_level) / 2
    right_quant = 1 - (1 - bootstrap_conf_level) / 2
    return np.quantile(distribution, [left_quant, right_quant])


def estimate_p_value(bootstrap_difference_distribution: np.ndarray, number_of_bootstrap_samples: int) -> float:
    """P-value estimation

    Args:
        bootstrap_difference_distribution (ndarray):  1D array containing bootstrap difference distribution
        number_of_bootstrap_samples (int): Number of bootstrap samples

    Returns:
        float: p-value

    """

    positions = np.sum(bootstrap_difference_distribution < 0, axis=0)
    return 2 * np.minimum(positions, number_of_bootstrap_samples - positions) / number_of_bootstrap_samples


def estimate_bin_params(sample: object) -> Tuple[float, int]:
    """Estimation plot bin params

    Args:
        sample (ndarray): 1D array containing observations

    Returns:
        tuple(float, int): Tuple containing bin width and bin count

    """

    q1 = np.quantile(sample, 0.25)
    q3 = np.quantile(sample, 0.75)
    iqr = q3 - q1
    bin_width = (2 * iqr) / (sample.shape[0] ** (1 / 3))
    bin_count = int(np.ceil((sample.max() - sample.min()) / bin_width))
    return bin_width, bin_count


def compute_jackknife_distribution(control: np.ndarray, statistic: Callable = np.mean) -> np.ndarray:
    """Returns jackknife resampled replicates for the given data and statistical function

    Args:
        control (ndarray): 1D array containing control sample
        statistic (function): statistical function to be used for the interval

    Returns:
        ndarray: jackknife distribution

    """

    jack_distribution = []
    for i in range(control.shape[0]):
        jack_sample = np.delete(control, i)
        jack_distribution.append(statistic(jack_sample))
    return np.array(jack_distribution)


def compute_a(jack_distribution: np.ndarray) -> float:
    """Returns the acceleration constant a

    Args:
        jack_distribution (ndarray): 1D array containing jackknife distribution

    Returns:
        float: acceleration constant a

    """
    mean = np.mean(jack_distribution)
    return (1 / 6) * np.divide(np.sum(mean - jack_distribution) ** 3,
                               (np.sum(mean - jack_distribution) ** 2) ** (3 / 2))


def compute_z0(control: np.ndarray, bootstrap_distribution: np.ndarray, statistic: Callable = np.mean) -> float:
    """Computes z0 for given data and statistical function

    Args:
        control (ndarray): 1D array containing control sample
        bootstrap_distribution (ndarray): 1D array containing bootstrap distribution
        statistic (function): statistical function to be used for the interval

    Returns:
        float: z0 value

    """

    s = statistic(control)
    return norm.ppf(np.sum(bootstrap_distribution < s) / bootstrap_distribution.shape[0])


def compute_bca_confidence_interval(control: np.ndarray, bootstrap_distribution: np.ndarray,
                                    bootstrap_conf_level: float = 0.95,
                                    statistic=np.mean) -> np.ndarray:
    """Returns BCa confidence interval for given data at given alpha level

    Args:
        control (ndarray): 1D array containing control sample
        bootstrap_distribution (ndarray): 1D array containing bootstrap distribution
        bootstrap_conf_level (float): confidence level for the interval
        statistic (function): statistical function to be used for the interval

    Returns:
        ndarray: array with lower and upper bounds of the confidence interval

    """

    # Compute bootstrap and jackknife replicates
    jack_distribution = compute_jackknife_distribution(control, statistic)

    # Compute a and z0
    a = compute_a(jack_distribution)
    z0 = compute_z0(control, bootstrap_distribution, statistic)

    # Compute confidence interval indices
    alpha = np.array([(1 - bootstrap_conf_level) / 2, 1 - (1 - bootstrap_conf_level) / 2])
    zs = z0 + norm.ppf(alpha).reshape(alpha.shape + (1,) * z0.ndim)
    a = norm.cdf(z0 + zs / (1 - a * zs))
    indices = np.round((bootstrap_distribution.shape[0] - 1) * a)
    indices = np.nan_to_num(indices).astype('int')

    # Compute confidence interval
    bootstrap_distribution_sorted = np.sort(bootstrap_distribution)
    bootstrap_confidence_interval = bootstrap_distribution_sorted[indices]
    return bootstrap_confidence_interval


def bootstrap_plot(bootstrap_difference_distribution: np.ndarray, bootstrap_confidence_interval: np.ndarray,
                   statistic: Union[str, Callable] = None, title: str = 'Bootstrap',
                   two_sample_plot: bool = True) -> None:
    """Bootstrap distribution plot

    Args:
        bootstrap_difference_distribution (ndarray): 1D array containing bootstrap difference distribution
        bootstrap_confidence_interval (ndarray): 1D array containing bootstrap confidence interval
        statistic (Union[str, Callable], optional): Statistic name or function. Defaults to None.
            If None, then 'Stat' will be used as xlabel; if the statistic is str, then it will be used as xlabel,
            else if statistic is function then formated function name will be used as xlabel
        title (str): Plot title. Defaults to 'Bootstrap'
        two_sample_plot (bool): If True, then two-sample bootstrap plot will be shown,
            it means that zero x-line will be added. Defaults to True

    """

    if isinstance(statistic, str):
        xlabel = statistic
    elif hasattr(statistic, '__call__'):
        xlabel = ' '.join([i.capitalize() for i in statistic.__name__.split('_')])
    else:
        xlabel = 'Stat'

    binwidth, _ = estimate_bin_params(bootstrap_difference_distribution)
    plt.hist(bootstrap_difference_distribution,
             bins=np.arange(bootstrap_difference_distribution.min(), bootstrap_difference_distribution.max() + binwidth,
                            binwidth))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Count')
    plt.axvline(x=bootstrap_confidence_interval[0], color='red', linestyle='dashed', linewidth=2)
    plt.axvline(x=bootstrap_confidence_interval[1], color='red', linestyle='dashed', linewidth=2)
    plt.axvline(x=np.median(bootstrap_difference_distribution), color='black', linestyle='dashed', linewidth=5)
    if two_sample_plot:
        plt.axvline(x=0, color='white', linewidth=5)
    plt.show()


def two_sample_bootstrap(control: np.ndarray, treatment: np.ndarray, bootstrap_conf_level: float = 0.95,
                         number_of_bootstrap_samples: int = 10000,
                         sample_size: int = None,
                         statistic: Callable = difference_of_mean,
                         return_distribution: bool = False,
                         plot: bool = False) -> Tuple[float, float, np.ndarray, np.ndarray] | Tuple[
    float, float, np.ndarray]:
    """Two-sample bootstrap

    Args:
        control (ndarray): 1D array containing control sample
        treatment (ndarray): 1D array containing treatment sample
        bootstrap_conf_level (float): Confidence level
        number_of_bootstrap_samples (int): Number of bootstrap samples
        sample_size (int): Sample size. Defaults to None. If None,
            then control_sample_size and treatment_sample_size
            wiil be equal to control.shape[0] and treatment.shape[0] respectively
        statistic (Callable): Statistic function. Defaults to difference_of_mean.
            Choose statistic function from compare_functions.py
        return_distribution (bool): If True, then bootstrap difference distribution will be returned. Defaults to False
        plot (bool): If True, then bootstrap plot will be shown. Defaults to True

    Returns:
        Tuple[float, float, ndarray, ndarray]: Tuple containing p-value, difference distribution statistic,
            bootstrap confidence interval, bootstrap difference distribution

    """

    def sample():
        control_sample = control[generator.choice(control.shape[0], control_sample_size, replace=True)]
        treatment_sample = treatment[generator.choice(treatment.shape[0], treatment_sample_size, replace=True)]
        return statistic(control_sample, treatment_sample)

    if sample_size:
        control_sample_size, treatment_sample_size = [sample_size] * 2
    else:
        control_sample_size, treatment_sample_size = control.shape[0], treatment.shape[0]

    generator = np.random.Generator(np.random.PCG64())
    pool = ThreadPool(cpu_count())
    bootstrap_difference_distribution = np.array(
        pool.starmap(sample, [() for i in range(number_of_bootstrap_samples)]))
    pool.close()

    bootstrap_confidence_interval = estimate_confidence_interval(bootstrap_difference_distribution,
                                                                 bootstrap_conf_level)
    bootstrap_difference_median = np.median(bootstrap_difference_distribution)
    p_value = estimate_p_value(bootstrap_difference_distribution, number_of_bootstrap_samples)

    if plot:
        bootstrap_plot(bootstrap_difference_distribution, bootstrap_confidence_interval, statistic=statistic)

    if return_distribution:
        return p_value, bootstrap_difference_median, bootstrap_confidence_interval, bootstrap_difference_distribution
    else:
        return p_value, bootstrap_difference_median, bootstrap_confidence_interval


def ctr_bootstrap(control: np.ndarray, treatment: np.ndarray, bootstrap_conf_level: float = 0.95,
                  number_of_bootstrap_samples: int = 10000,
                  sample_size: int = None,
                  return_distribution: bool = False,
                  plot: bool = False) -> Tuple[float, float, np.ndarray, np.ndarray] | Tuple[float, float, np.ndarray]:
    """Two-sample CTR-Bootstrap

    In every sample global CTR will be used to difference calculation:
    global_ctr_control_sample = control_sample.clicks.sum() / control_sample.views.sum()
    global_ctr_treatment_sample = treatment_sample.clicks.sum() / treatment_sample.views.sum()
    crt_difference = global_ctr_treatment_sample - global_ctr_control_sample

    Args:
        control (ndarray): 1D array containing control sample
        treatment (ndarray): 1D array containing treatment sample
        bootstrap_conf_level (float): Confidence level
        number_of_bootstrap_samples (int): Number of bootstrap samples
        sample_size (int): Sample size. Defaults to None. If None,
            then control_sample_size and treatment_sample_size
            wiil be equal to control.shape[0] and treatment.shape[0] respectively
        statistic (Callable): Statistic function. Defaults to difference_of_mean.
            Choose statistic function from compare_functions.py
        return_distribution (bool): If True, then bootstrap difference distribution will be returned. Defaults to False
        plot (bool): If True, then bootstrap plot will be shown. Defaults to True

    Returns:
        Tuple[float, float, ndarray, ndarray]: Tuple containing p-value, difference distribution statistic,
            bootstrap confidence interval, bootstrap difference distribution

    """

    def sample():
        control_sample = control.sample(control_sample_size, replace=True)
        treatment_sample = treatment.sample(treatment_sample_size, replace=True)
        global_ctr_control_sample = control_sample.clicks.sum() / control_sample.views.sum()
        global_ctr_treatment_sample = treatment_sample.clicks.sum() / treatment_sample.views.sum()
        return global_ctr_treatment_sample - global_ctr_control_sample

    if sample_size:
        control_sample_size, treatment_sample_size = [sample_size] * 2
    else:
        control_sample_size, treatment_sample_size = control.shape[0], treatment.shape[0]

    pool = ThreadPool(cpu_count())
    bootstrap_difference_distribution = np.array(
        pool.starmap(sample, [() for i in range(number_of_bootstrap_samples)]))
    pool.close()

    bootstrap_confidence_interval = estimate_confidence_interval(bootstrap_difference_distribution,
                                                                 bootstrap_conf_level)
    bootstrap_difference_median = np.median(bootstrap_difference_distribution)
    p_value = estimate_p_value(bootstrap_difference_distribution, number_of_bootstrap_samples)

    if plot:
        bootstrap_plot(bootstrap_difference_distribution, bootstrap_confidence_interval, statistic='CTR Difference')

    if return_distribution:
        return p_value, bootstrap_difference_median, bootstrap_confidence_interval, bootstrap_difference_distribution
    else:
        return p_value, bootstrap_difference_median, bootstrap_confidence_interval


def spotify_two_sample_bootstrap(control: np.ndarray, treatment: np.ndarray, number_of_bootstrap_samples: int = 10000,
                                 sample_size: int = None, q1: float = 0.5, q2: float = None,
                                 statistic: Callable = difference,
                                 bootstrap_conf_level: float = 0.95,
                                 return_distribution: bool = False,
                                 plot: bool = False) -> Tuple[float, float, np.ndarray, np.ndarray] | Tuple[
    float, float, np.ndarray]:
    """Two-sample Spotify-Bootstrap

    Can be used for difference of quantiles, difference of means, difference of medians, etc.
    Note: Can be used with different quantiles for control and treatment samples

    Mårten Schultzberg and Sebastian Ankargren. “Resampling-free bootstrap inference for quantiles.”
    arXiv e-prints, art. arXiv:2202.10992, (2022). https://arxiv.org/abs/2202.10992

    Args:
        control (ndarray): 1D array containing control sample
        treatment (ndarray): 1D array containing treatment sample
        bootstrap_conf_level (float): Confidence level
        number_of_bootstrap_samples (int): Number of bootstrap samples
        sample_size (int): Sample size. Defaults to None. If None,
            then control_sample_size and treatment_sample_size
            wil be equal to control.shape[0] and treatment.shape[0] respectively
        q1 (float): Quantile of interest for control sample. Defaults to 0.5
        q2 (float): Quantile of interest for treatment sample. Defaults to 0.5
        statistic (Callable): Statistic function. Defaults to difference.
            Choose statistic function from compare_functions.py
        return_distribution (bool): If True, then bootstrap difference distribution will be returned. Defaults to False
        plot (bool): If True, then bootstrap plot will be shown. Defaults to True

    Returns:
        Tuple[float, float, ndarray, ndarray]: Tuple containing p-value, difference distribution statistic,
            bootstrap confidence interval, bootstrap difference distribution

    """

    if sample_size:
        control_sample_size, treatment_sample_size = [sample_size] * 2
    else:
        control_sample_size, treatment_sample_size = control.shape[0], treatment.shape[0]

    if q2 is None:
        q2 = q1

    sorted_control = np.sort(control)
    sorted_treatment = np.sort(treatment)
    treatment_sample_values = sorted_treatment[
        binomial(treatment_sample_size + 1, q2, number_of_bootstrap_samples)]
    control_sample_values = sorted_control[
        binomial(control_sample_size + 1, q1, number_of_bootstrap_samples)]
    bootstrap_difference_distribution = statistic(control_sample_values, treatment_sample_values)
    bootstrap_difference = statistic(np.quantile(sorted_control, q1), np.quantile(sorted_treatment, q2))
    bootstrap_confidence_interval = estimate_confidence_interval(bootstrap_difference_distribution,
                                                                 bootstrap_conf_level)
    p_value = estimate_p_value(bootstrap_difference_distribution, number_of_bootstrap_samples)
    if plot:
        bootstrap_plot(bootstrap_difference_distribution, bootstrap_confidence_interval, statistic=statistic)

    if return_distribution:
        return p_value, bootstrap_difference, bootstrap_confidence_interval, bootstrap_difference_distribution
    else:
        return p_value, bootstrap_difference, bootstrap_confidence_interval


def ab_test_simulation(control: np.ndarray, treatment: np.ndarray, number_of_experiments: int = 2000,
                       stat_test: Callable = ttest_ind) -> Tuple[np.ndarray, float, float]:
    """A/B test simulation

    Args:
        control (ndarray): 1D array containing control sample
        treatment (ndarray): 1D array containing treatment sample
        number_of_experiments (int): Number of experiments. Defaults to 2000
        stat_test (Callable): Statistical test. Defaults to ttest_ind

    Returns:
        Tuple[np.ndarray, float, float]: Tuple containing p-values, test power and AUC
        
    """

    def experiment():
        return stat_test(np.random.choice(control, control_size, replace=True),
                         np.random.choice(treatment, treatment_size, replace=True))[1]

    control_size, treatment_size = control.shape[0], treatment.shape[0]

    pool = ThreadPool(cpu_count())
    ab_p_values = np.array(pool.starmap(experiment, [() for i in range(number_of_experiments)]))
    pool.close()

    test_power = np.mean(ab_p_values < 0.05)
    ab_p_values_sorted = np.sort(ab_p_values)
    auc = np.trapz(np.arange(ab_p_values_sorted.shape[0]) / ab_p_values_sorted.shape[0], ab_p_values_sorted)

    return ab_p_values, test_power, auc


def plot_summary(aa_p_values: np.ndarray, ab_p_values: np.ndarray):
    """Plot summary for A/A and A/B testing

    Args:
        aa_p_values (ndarray): 1D array containing A/A p-values
        treatment (ndarray): 1D array containing A/B p-values

    """

    cdf_h0_title = 'Simulated p-value CDFs under H0 (FPR)'
    cdf_h1_title = 'Simulated p-value CDFs under H1 (Sensitivity)'
    p_value_h0_title = 'Simulated p-values under H0'
    p_value_h1_title = 'Simulated p-values under H1'

    test_power = np.mean(ab_p_values < 0.05)

    fig, ax = plt.subplot_mosaic('AB;CD;FF', gridspec_kw={'height_ratios': [1, 1, 0.3], 'width_ratios': [1, 1]},
                                 constrained_layout=True)

    ax['A'].set_title(p_value_h0_title, fontsize=10)
    ax['A'].hist(aa_p_values, bins=50, density=True, label=p_value_h0_title, alpha=0.5)
    ax['B'].set_title(p_value_h1_title, fontsize=10)
    ax['B'].hist(ab_p_values, bins=50, density=True, label=p_value_h1_title, alpha=0.5)

    ax['C'].set_title(cdf_h0_title, fontsize=10)
    plot_cdf(aa_p_values, label=cdf_h0_title, ax=ax['C'])

    ax['D'].set_title(cdf_h1_title, fontsize=10)
    ax['D'].axvline(0.05, color='black', linestyle='dashed', linewidth=2, label='alpha=0.05')
    plot_cdf(ab_p_values, label=cdf_h1_title, ax=ax['D'])

    ax['F'].set_title('Power', fontsize=10)
    ax['F'].set_xlim(0, 1)
    ax['F'].yaxis.set_tick_params(labelleft=False)
    ax['F'].barh(y=0, width=test_power, label='Power')


def quantile_bootstrap_plot(control: np.ndarray, treatment: np.ndarray, n_step: int = 20, q1: float = 0.01,
                            q2: float = 0.99, bootstrap_conf_level: float = 0.95, statistic: Callable = difference,
                            correction: str = 'bh') -> None:
    """Quantile Bootstrap Plot

    Args:
        control (ndarray): 1D array containing control sample
        treatment (ndarray): 1D array containing treatment sample
        n_step (int): Number of quantiles to compare. Defaults to 20
        q1 (float): Lower quantile. Defaults to 0.025
        q2 (float): Upper quantile. Defaults to 0.975
        bootstrap_conf_level (float): Confidence level. Defaults to 0.95
        statistic (Callable): Statistic function. Defaults to difference.
        correction (str): Correction method. Defaults to 'bh'

    """

    quantiles_to_compare = np.linspace(q1, q2, n_step)
    statistics, correction_list = list(), list()
    for i, quantile in enumerate(quantiles_to_compare):
        match correction:
            case 'bonferroni':
                corr = (1 - bootstrap_conf_level) / n_step
            case 'bh':
                corr = ((1 - bootstrap_conf_level) * (i + 1)) / n_step
        correction_list.append(corr)
        p_value, bootstrap_mean, bootstrap_confidence_interval = spotify_two_sample_bootstrap(control, treatment,
                                                                                              q1=quantile,
                                                                                              q2=quantile,
                                                                                              statistic=statistic,
                                                                                              bootstrap_conf_level=1 - corr)
        statistics.append([p_value, bootstrap_mean, bootstrap_confidence_interval[0], bootstrap_confidence_interval[1]])
    statistics = np.array(statistics)

    df = pd.DataFrame(
        {'quantile': quantiles_to_compare,
         'p_value': statistics[:, 0],
         'difference': statistics[:, 1],
         'lower_bound': statistics[:, 2],
         'upper_bound': statistics[:, 3]}
    )
    df['significance'] = (df['p_value'] < correction_list).astype(str)

    df["ci_upper"] = df["upper_bound"] - df["difference"]
    df["ci_lower"] = df["difference"] - df["lower_bound"]

    fig = go.Figure([
        go.Scatter(
            name='Difference',
            x=df['quantile'],
            y=df['difference'],
            mode='lines',
            line=dict(color='red'),
        ),
        go.Scatter(
            name='Upper Bound',
            x=df['quantile'],
            y=df['upper_bound'],
            mode='lines',
            marker=dict(color="black"),
            line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            name='Lower Bound',
            x=df['quantile'],
            y=df['lower_bound'],
            marker=dict(color="black"),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            showlegend=False
        )
    ])
    fig.add_hline(y=0, line_width=3, line_dash="dash", line_color="black")

    fig.update_traces(
        customdata=df,
        name='Quantile Information',
        hovertemplate="<br>".join([
            "Quantile: %{customdata[0]}",
            "p_value: %{customdata[1]}",
            "Significance: %{customdata[5]}",
            "Difference: %{customdata[2]}",
            "Lower Bound: %{customdata[3]}",
            "Upper Bound: %{customdata[4]}"
        ])
    )

    fig.show()


def plot_cdf(p_values: np.ndarray, label: str, ax: Axes, linewidth: float = 3) -> None:
    """CFF plot function

    Args:
        p_values (ndarray): 1D array containing p-values
        label (str): Label for the plot
        ax (Axes): Axes object to plot on
        linewidth (float): Linewidth for the plot. Defaults to 3

    """

    sorted_p_values = np.sort(p_values)
    position = scipy.stats.rankdata(sorted_p_values, method='ordinal')
    cdf = position / p_values.shape[0]

    sorted_data = np.hstack((sorted_p_values, 1))
    cdf = np.hstack((cdf, 1))

    ax.plot([0, 1], [0, 1], linestyle='--', color='black')
    ax.plot(sorted_data, cdf, label=label, linestyle='solid', linewidth=linewidth)


def one_sample_bootstrap(control: np.ndarray, bootstrap_conf_level: float = 0.95,
                         number_of_bootstrap_samples: int = 10000,
                         sample_size: int = None,
                         statistic: Callable = np.mean,
                         method: str = 'percentile',
                         return_distribution: bool = False,
                         plot: bool = False) -> Tuple[float, np.ndarray, np.ndarray] | Tuple[float, np.ndarray]:
    """One sample bootstrap

    Args:
        control (ndarray): 1D array containing control sample
        bootstrap_conf_level (float): Confidence level
        number_of_bootstrap_samples (int): Number of bootstrap samples
        sample_size (int): Sample size. Defaults to None. If None,
            then control_sample_size and treatment_sample_size
            wiil be equal to control.shape[0] and treatment.shape[0] respectively
        statistic (Callable): Statistic function. Defaults to difference_of_mean.
            Choose statistic function from compare_functions.py
        method (str): Bootstrap method. Defaults to 'percentile'. Choose from 'bca', 'basic', 'percentile', 'studentized'
        return_distribution (bool): If True, then bootstrap difference distribution will be returned. Defaults to False
        plot (bool): If True, then bootstrap plot will be shown. Defaults to True

    Returns:
        Tuple[float, ndarray, ndarray]: Tuple containing difference distribution statistic,
            bootstrap confidence interval, bootstrap difference distribution

    """

    def sample():
        control_sample = control[generator.choice(control.shape[0], size=sample_size, replace=True)]

        match method:
            case 'studentized':
                return statistic(control_sample), control_sample.std()
            case _:
                return statistic(control_sample)

    if method not in ['bca', 'basic', 'percentile', 'studentized']:
        raise ValueError('method argument should be one of the following: bca, basic, percentile, studentized')

    sample_size = sample_size if sample_size else control.shape[0]

    generator = np.random.Generator(np.random.PCG64())
    pool = ThreadPool(cpu_count())
    bootstrap_stats = np.array(
        pool.starmap(sample, [() for i in range(number_of_bootstrap_samples)]))
    pool.close()

    sample_stat = statistic(control)

    match method:
        case 'bca':
            bootstrap_distribution = bootstrap_stats
            bootstrap_confidence_interval = compute_bca_confidence_interval(control, bootstrap_distribution,
                                                                            bootstrap_conf_level=bootstrap_conf_level,
                                                                            statistic=statistic)
        case 'basic':
            bootstrap_distribution = bootstrap_stats
            percentile_bootstrap_confidence_interval = estimate_confidence_interval(bootstrap_distribution,
                                                                                    bootstrap_conf_level)
            bootstrap_confidence_interval = np.array(
                [2 * sample_stat - percentile_bootstrap_confidence_interval[1],
                 2 * sample_stat - percentile_bootstrap_confidence_interval[0]])

        case 'percentile':
            bootstrap_distribution = bootstrap_stats
            bootstrap_confidence_interval = estimate_confidence_interval(bootstrap_distribution,
                                                                         bootstrap_conf_level)
        case 'studentized':
            bootstrap_distribution = bootstrap_stats[:, 0]
            bootstrap_std_distribution = bootstrap_stats[:, 1]
            bootstrap_distribution_std = bootstrap_distribution.std()
            bootstrap_std_errors = bootstrap_std_distribution / np.sqrt(sample_size)
            t_statistics = (bootstrap_distribution - sample_stat) / bootstrap_std_errors
            lower, upper = estimate_confidence_interval(t_statistics, bootstrap_conf_level)
            bootstrap_confidence_interval = np.array(
                [sample_stat - bootstrap_distribution_std * upper, sample_stat - bootstrap_distribution_std * lower])

    bootstrap_difference_median = np.median(bootstrap_distribution)

    if plot:
        bootstrap_plot(bootstrap_distribution, bootstrap_confidence_interval, statistic=statistic,
                       two_sample_plot=False)

    if return_distribution:
        return bootstrap_difference_median, bootstrap_confidence_interval, bootstrap_distribution
    else:
        return bootstrap_difference_median, bootstrap_confidence_interval


def spotify_one_sample_bootstrap(sample: np.ndarray, sample_size: int = None, q: float = 0.5,
                                 bootstrap_conf_level: float = 0.95, number_of_bootstrap_samples: int = 10000,
                                 return_distribution: bool = False,
                                 plot=False) -> Tuple[float, np.ndarray, np.ndarray] | Tuple[float, np.ndarray]:
    """One-sample Spotify-Bootstrap

    Mårten Schultzberg and Sebastian Ankargren. “Resampling-free bootstrap inference for quantiles.”
    arXiv e-prints, art. arXiv:2202.10992, (2022). https://arxiv.org/abs/2202.10992

    Args:
        sample (ndarray): 1D array containing sample
        sample_size (int): sample size. Defaults to None
        q (float): Quantile of interest. Defaults to 0.5
        bootstrap_conf_level (float): Confidence level. Defaults to 0.95
        number_of_bootstrap_samples (int): Number of bootstrap samples. Defaults to 10000
        return_distribution (bool): If True, then bootstrap difference distribution will be returned. Defaults to False
        plot (bool): Plot histogram of bootstrap distribution. Defaults to False

    Returns:
        Tuple[float, ndarray, ndarray]: Tuple containing quantile of interest, bootstrap confidence interval and bootstrap distribution

    """

    if not sample_size:
        sample_size = sample.shape[0]

    sample_sorted = np.sort(sample)
    left_quant = (1 - bootstrap_conf_level) / 2
    right_quant = 1 - (1 - bootstrap_conf_level) / 2
    ci_indexes = binom.ppf([left_quant, right_quant], sample_size + 1, q)
    bootstrap_confidence_interval = sample_sorted[[int(np.floor(ci_indexes[0])), int(np.ceil(ci_indexes[1]))]]
    quantile_indices = binomial(sample_size + 1, q, number_of_bootstrap_samples)
    bootstrap_distribution = sample_sorted[quantile_indices]

    if plot:
        statistic = f'Quantile_{q}'
        bootstrap_plot(bootstrap_distribution, bootstrap_confidence_interval, statistic=statistic,
                       two_sample_plot=False)

    if return_distribution:
        return np.quantile(sample_sorted, q), bootstrap_confidence_interval, bootstrap_distribution
    else:
        return np.quantile(sample_sorted, q), bootstrap_confidence_interval


def poisson_bootstrap(control: np.ndarray, treatment: np.ndarray,
                      number_of_bootstrap_samples: int = 10000) -> float:
    """Simple Poisson Bootstrap

    Args:
        control (ndarray): 1D array containing control sample
        treatment (ndarray): 1D array containing treatment sample
        number_of_bootstrap_samples (int): Number of bootstrap samples. Defaults to 10000

    Returns:
        float: p-value

    """

    sample_size = np.min([control.shape[0], treatment.shape[0]])
    control_distribution = np.zeros(shape=number_of_bootstrap_samples)
    treatment_distribution = np.zeros(shape=number_of_bootstrap_samples)
    for control_item, treatment_item in zip(control[:sample_size], treatment[:sample_size]):
        weights = np.random.poisson(1, number_of_bootstrap_samples)
        control_distribution += control_item * weights
        treatment_distribution += treatment_item * weights

    bootstrap_difference_distribution = treatment_distribution - control_distribution
    p_value = estimate_p_value(bootstrap_difference_distribution, number_of_bootstrap_samples)
    return p_value


def two_sample_tbootstrap(control: np.ndarray, treatment: np.ndarray, bootstrap_conf_level: float = 0.95,
                          number_of_bootstrap_samples: int = 10000,
                          sample_size: int = None,
                          statistic: Callable = difference_of_mean,
                          return_distribution: bool = False,
                          plot: bool = False) -> Tuple[float, float, np.ndarray, np.ndarray] | Tuple[
    float, float, np.ndarray]:
    """Two-sample Studentized t-Bootstrap

    Args:
        control (ndarray): 1D array containing control sample
        treatment (ndarray): 1D array containing treatment sample
        bootstrap_conf_level (float): Confidence level
        number_of_bootstrap_samples (int): Number of bootstrap samples
        sample_size (int): Sample size. Defaults to None. If None,
            then control_sample_size and treatment_sample_size
            wiil be equal to control.shape[0] and treatment.shape[0] respectively
        statistic (Callable): Statistic function. Defaults to difference_of_mean.
            Choose statistic function from compare_functions.py
        return_distribution (bool): If True, then bootstrap difference distribution will be returned. Defaults to False
        plot (bool): If True, then bootstrap plot will be shown. Defaults to True

    Returns:
        Tuple[float, float, ndarray, ndarray]: Tuple containing p-value, difference distribution statistic,
            bootstrap confidence interval, bootstrap difference distribution

    """

    def sample():
        control_sample = control[generator.choice(control.shape[0], sample_size, replace=True)]
        treatment_sample = treatment[generator.choice(treatment.shape[0], sample_size, replace=True)]
        return statistic(control_sample, treatment_sample), np.std(treatment_sample - control_sample)

    if sample_size is None:
        sample_size = np.min([control.shape[0], treatment.shape[0]])

    generator = np.random.Generator(np.random.PCG64())
    pool = ThreadPool(cpu_count())
    bootstrap_stats = np.array(pool.starmap(sample, [() for i in range(number_of_bootstrap_samples)]))
    pool.close()

    bootstrap_difference_distribution = bootstrap_stats[:, 0]
    bootstrap_difference_std_distribution = bootstrap_stats[:, 1]
    sample_stat = statistic(control, treatment)
    bootstrap_difference_distribution_std = bootstrap_difference_distribution.std()
    bootstrap_std_errors = bootstrap_difference_std_distribution / np.sqrt(sample_size)
    t_statistics = (bootstrap_difference_distribution - sample_stat) / bootstrap_std_errors
    lower, upper = estimate_confidence_interval(t_statistics, bootstrap_conf_level)
    bootstrap_confidence_interval = np.array([sample_stat - bootstrap_difference_distribution_std * upper,
                                              sample_stat - bootstrap_difference_distribution_std * lower])
    bootstrap_difference_distribution_median = np.median(bootstrap_difference_distribution)
    p_value = estimate_p_value(bootstrap_difference_distribution, number_of_bootstrap_samples)

    if plot:
        bootstrap_plot(bootstrap_difference_distribution, bootstrap_confidence_interval, statistic=statistic,
                       title='t-Bootstrap')

    if return_distribution:
        return p_value, bootstrap_difference_distribution_median, bootstrap_confidence_interval, bootstrap_difference_distribution
    else:
        return p_value, bootstrap_difference_distribution_median, bootstrap_confidence_interval
