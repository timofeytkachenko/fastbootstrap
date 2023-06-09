import scipy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from scipy.stats import norm, binom
from numpy.random import normal, binomial
from multiprocess import Pool, cpu_count
from compare_functions import difference_of_mean, difference
from scipy.stats import percentileofscore
from scipy.stats import ttest_ind
from tqdm.auto import tqdm
from typing import Tuple, Union, Callable
from inspect import getfullargspec

plt.style.use('ggplot')

colors = {'False': 'red', 'True': 'black'}


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


def bootstrap_plot(bootstrap_difference_distribution: np.ndarray, bootstrap_confidence_interval: np.ndarray,
                   statistic: Union[str, Callable] = None) -> None:
    """Bootstrap distribution plot

    Args:
        bootstrap_difference_distribution (ndarray): 1D array containing bootstrap difference distribution
        bootstrap_confidence_interval (ndarray): 1D array containing bootstrap confidence interval
        statistic (Union[str, Callable], optional): Statistic name or function. Defaults to None.
        If None, then 'Stat' will be used as xlabel; if the statistic is str, then it will be used as xlabel,
        else if statistic is function then formated function name will be used as xlabel

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
    plt.title('Bootstrap')
    plt.xlabel(xlabel)
    plt.ylabel('Count')
    plt.axvline(x=bootstrap_confidence_interval[0], color='red', linestyle='dashed', linewidth=2)
    plt.axvline(x=bootstrap_confidence_interval[1], color='red', linestyle='dashed', linewidth=2)
    plt.axvline(x=bootstrap_difference_distribution.mean(), color='black', linestyle='dashed', linewidth=5)
    plt.axvline(x=0, color='white', linewidth=5)
    plt.show()


def bootstrap(control: np.ndarray, treatment: np.ndarray, bootstrap_conf_level: float = 0.95,
              number_of_bootstrap_samples: int = 10000,
              sample_size: int = None,
              statistic: Callable = difference_of_mean, plot: bool = False, progress_bar: bool = False) -> Tuple[
    float, float, np.ndarray, np.ndarray]:
    """Two-sample bootstrap

    If max([control.shape[0], treatment.shape[0]) > 10000, then multiprocessing will be used

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
        plot (bool): If True, then bootstrap plot will be shown. Defaults to True
        progress_bar (bool): If True, then progress bar will be shown. Defaults to False
    Returns:
        Tuple[float, float, ndarray, ndarray]: Tuple containing p-value, difference distribution statistic,
            bootstrap confidence interval, bootstrap difference distribution

    """

    def sample():
        control_sample = np.random.choice(control, control_sample_size, replace=True)
        treatment_sample = np.random.choice(treatment, treatment_sample_size, replace=True)
        return statistic(control_sample, treatment_sample)

    if sample_size:
        control_sample_size, treatment_sample_size = [sample_size] * 2
    else:
        control_sample_size, treatment_sample_size = control.shape[0], treatment.shape[0]

    if np.max([control.shape[0], treatment.shape[0]]) <= 10000:
        if progress_bar:
            bootstrap_difference_distribution = np.array([sample() for i in tqdm(range(number_of_bootstrap_samples))])
        else:
            bootstrap_difference_distribution = np.array([sample() for i in range(number_of_bootstrap_samples)])
    else:
        pool = Pool(cpu_count())
        bootstrap_difference_distribution = np.array(
            pool.starmap(sample, [() for i in range(number_of_bootstrap_samples)]))
        pool.close()

    bootstrap_confidence_interval = estimate_confidence_interval(bootstrap_difference_distribution,
                                                                 bootstrap_conf_level)
    bootstrap_difference_mean = bootstrap_difference_distribution.mean()
    p_value = estimate_p_value(bootstrap_difference_distribution, number_of_bootstrap_samples)

    if plot:
        bootstrap_plot(bootstrap_difference_distribution, bootstrap_confidence_interval, statistic=statistic)
    return p_value, bootstrap_difference_mean, bootstrap_confidence_interval, bootstrap_difference_distribution


def ctr_bootstrap(control: np.ndarray, treatment: np.ndarray, bootstrap_conf_level: float = 0.95,
                  number_of_bootstrap_samples: int = 10000,
                  sample_size: int = None, plot: bool = False, progress_bar: bool = False) -> Tuple[
    float, float, np.ndarray, np.ndarray]:
    """Two-sample CTR-Bootstrap

        If max([control.shape[0], treatment.shape[0]) > 10000, then multiprocessing will be used.
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
            plot (bool): If True, then bootstrap plot will be shown. Defaults to True
            progress_bar (bool): If True, then progress bar will be shown. Defaults to False
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

    if np.max([control.shape[0], treatment.shape[0]]) <= 10000:
        if progress_bar:
            bootstrap_difference_distribution = np.array([sample() for i in tqdm(range(number_of_bootstrap_samples))])
        else:
            bootstrap_difference_distribution = np.array([sample() for i in range(number_of_bootstrap_samples)])
    else:
        pool = Pool(cpu_count())
        bootstrap_difference_distribution = np.array(
            pool.starmap(sample, [() for i in range(number_of_bootstrap_samples)]))
        pool.close()

    bootstrap_confidence_interval = estimate_confidence_interval(bootstrap_difference_distribution,
                                                                 bootstrap_conf_level)
    bootstrap_difference_mean = bootstrap_difference_distribution.mean()
    p_value = estimate_p_value(bootstrap_difference_distribution, number_of_bootstrap_samples)

    if plot:
        bootstrap_plot(bootstrap_difference_distribution, bootstrap_confidence_interval, statistic='CTR Difference')
    return p_value, bootstrap_difference_mean, bootstrap_confidence_interval, bootstrap_difference_distribution


def spotify_one_sample_bootstrap(sample: np.ndarray, sample_size: int = None, quantile_of_interest: float = 0.5,
                                 bootstrap_conf_level: float = 0.95) -> Tuple[float, np.ndarray]:
    """One-sample Spotify-Bootstrap

    Mårten Schultzberg and Sebastian Ankargren. “Resampling-free bootstrap inference for quantiles.”
    arXiv e-prints, art. arXiv:2202.10992, (2022). https://arxiv.org/abs/2202.10992

    Args:
        sample (ndarray): 1D array containing sample
        sample_size (int): sample size. Defaults to None
        quantile_of_interest (float): Quantile of interest. Defaults to 0.5
        bootstrap_conf_level (float): Confidence level. Defaults to 0.95

    Returns:
        Tuple[float, ndarray]: Tuple containing quantile of interest mean value and bootstrap confidence interval

    """
    if not sample_size:
        sample_size = sample.shape[0]
    sample_sorted = np.sort(sample)
    left_quant = (1 - bootstrap_conf_level) / 2
    right_quant = 1 - (1 - bootstrap_conf_level) / 2
    ci_indexes = binom.ppf([left_quant, right_quant], sample_size + 1, quantile_of_interest)
    bootstrap_confidence_interval = sample_sorted[[int(np.floor(ci_indexes[0])), int(np.ceil(ci_indexes[1]))]]
    return np.quantile(sample_sorted, quantile_of_interest), bootstrap_confidence_interval


def spotify_two_sample_bootstrap(control: np.ndarray, treatment: np.ndarray, number_of_bootstrap_samples: int = 10000,
                                 sample_size: int = None, q1: float = 0.5, q2: float = 0.5,
                                 statistic: Callable = difference,
                                 bootstrap_conf_level: float = 0.95,
                                 plot: bool = False) -> Tuple[float, float, np.ndarray, np.ndarray]:
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
        plot (bool): If True, then bootstrap plot will be shown. Defaults to True
    Returns:
        Tuple[float, float, ndarray, ndarray]: Tuple containing p-value, difference distribution statistic,
            bootstrap confidence interval, bootstrap difference distribution

    """

    if sample_size:
        control_sample_size, treatment_sample_size = [sample_size] * 2
    else:
        control_sample_size, treatment_sample_size = control.shape[0], treatment.shape[0]

    sorted_control = np.sort(control)
    sorted_treatment = np.sort(treatment)
    treatment_sample_values = sorted_treatment[
        binomial(treatment_sample_size + 1, q2, number_of_bootstrap_samples)]
    control_sample_values = sorted_control[
        binomial(control_sample_size + 1, q1, number_of_bootstrap_samples)]
    bootstrap_difference_distribution = statistic(control_sample_values, treatment_sample_values)

    bootstrap_difference_mean = statistic(
        np.quantile(sorted_control, q1), np.quantile(sorted_treatment, q2))

    bootstrap_confidence_interval = estimate_confidence_interval(bootstrap_difference_distribution,
                                                                 bootstrap_conf_level)
    p_value = estimate_p_value(bootstrap_difference_distribution, number_of_bootstrap_samples)
    if plot:
        bootstrap_plot(bootstrap_difference_distribution, bootstrap_confidence_interval, statistic=statistic)

    return p_value, bootstrap_difference_mean, bootstrap_confidence_interval, bootstrap_difference_distribution


def poisson_bootstrap(control: np.ndarray, treatment: np.ndarray, number_of_bootstrap_samples: int = 10000) -> float:
    """Poisson-Bootstrap

    Sample size should be equal for control and treatment samples

    Args:
        control (ndarray): 1D array containing control sample
        treatment (ndarray): 1D array containing treatment sample
        number_of_bootstrap_samples (int): Number of bootstrap samples

    Returns:
        float: p-value

    """

    poisson_bootstraps = scipy.stats.poisson(1).rvs((number_of_bootstrap_samples, control.shape[0])).astype(np.int64)

    values_1 = np.matmul(control, poisson_bootstraps.T)
    values_2 = np.matmul(treatment, poisson_bootstraps.T)

    difference = values_2 - values_1
    p_value = estimate_p_value(difference, number_of_bootstrap_samples)
    return p_value


def ctr_poisson_bootstrap(ctrs_1: np.ndarray, weights_1: np.ndarray, ctrs_2: np.ndarray, weights_2: np.ndarray,
                          number_of_bootstrap_samples: int = 10000) -> float:
    """CTR Poisson-Bootstrap for global CTRs

    Args:
        ctrs_1 (ndarray): 1D array containing control CTRs
        weights_1 (ndarray): 1D array containing control weights
        ctrs_2 (ndarray): 1D array containing treatment CTRs
        weights_2 (ndarray): 1D array containing treatment weights
        number_of_bootstrap_samples (int): Number of bootstrap samples

    Returns:
        float: p-value

    """
    poisson_bootstraps = scipy.stats.poisson(1).rvs(size=(number_of_bootstrap_samples, ctrs_1.shape[0])).astype(
        np.int64)

    values_1 = np.matmul(ctrs_1 * weights_1, poisson_bootstraps.T)
    weights_1 = np.matmul(weights_1, poisson_bootstraps.T)

    values_2 = np.matmul(ctrs_2 * weights_2, poisson_bootstraps.T)
    weights_2 = np.matmul(weights_2, poisson_bootstraps.T)

    difference = values_2 / weights_2 - values_1 / weights_1

    p_value = estimate_p_value(difference, number_of_bootstrap_samples)
    return p_value


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

    pool = Pool(cpu_count())
    ab_p_values = np.array(pool.starmap(experiment, [() for i in range(number_of_experiments)]))
    pool.close()

    test_power = np.mean(ab_p_values < 0.05)
    ab_p_values_sorted = np.sort(ab_p_values)
    auc = np.trapz(np.arange(ab_p_values_sorted.shape[0]) / ab_p_values_sorted.shape[0], ab_p_values_sorted)

    return ab_p_values, test_power, auc


def plot_summary(aa_p_values: np.ndarray, ab_p_values: np.ndarray) -> None:
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
                            q2: float = 0.99, statistic: Callable = difference) -> None:
    """

    Args:
        control (ndarray): 1D array containing control sample
        treatment (ndarray): 1D array containing treatment sample
        n_step (int): Number of quantiles to compare. Defaults to 20
        q1 (float): Lower quantile. Defaults to 0.025
        q2 (float): Upper quantile. Defaults to 0.975
        statistic (Callable): Statistic function. Defaults to difference.

    """
    quantiles_to_compare = np.linspace(q1, q2, n_step)
    statistics = list()
    for quantile in quantiles_to_compare:
        p_value, bootstrap_mean, bootstrap_confidence_interval, _ = spotify_two_sample_bootstrap(control, treatment,
                                                                                                 q1=quantile,
                                                                                                 q2=quantile,
                                                                                                 statistic=statistic)
        statistics.append([p_value, bootstrap_mean, bootstrap_confidence_interval[0], bootstrap_confidence_interval[1]])
    statistics = np.array(statistics)

    df = pd.DataFrame(
        {'quantile': quantiles_to_compare,
         'p_value': statistics[:, 0],
         'difference': statistics[:, 1],
         'lower_bound': statistics[:, 2],
         'upper_bound': statistics[:, 3]}
    )
    df['significance'] = (df['p_value'] < 0.05).astype(str)

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


def intra_user_correlation_aware_weights(clicks_1: np.ndarray, views_1: np.ndarray, views_2: np.ndarray) -> Tuple[
    np.ndarray, np.ndarray]:
    """Calculates weights for UMVUE global ctr estimate for every user in every experiment both in treatment and control groups

    Args:
        clicks_1 (ndarray): clicks of every user from control group in every experiment
        views_1 (ndarray): views of every user from control group in every experiment
        views_2 (ndarray): views of every user from treatment group in every experiment

    Returns:
        Tuple[ndarray, ndarray]: weights for every user in every experiment

    """
    ri = clicks_1 / views_1
    s3 = clicks_1 * (1 - ri) ** 2 + (views_1 - clicks_1) * ri ** 2
    s3 = np.sum(s3, axis=0) / np.sum(views_1 - 1, axis=0)
    rb = np.mean(clicks_1 / views_1, axis=0)
    s2 = clicks_1 * (1 - rb) ** 2 + (views_1 - clicks_1) * rb ** 2
    s2 = np.sum(s2, axis=0) / (np.sum(views_1, axis=0) - 1)
    rho = np.maximum(0, 1 - s3 / s2)
    w_1 = views_1 / (1 + (views_1 - 1) * rho)
    w_2 = views_2 / (1 + (views_2 - 1) * rho)
    return w_1, w_2


def estimate_quantile_of_mean(control: np.ndarray, bootstrap_conf_level: float = 0.95,
                              number_of_bootstrap_samples: int = 10000) -> Tuple[float, np.ndarray]:
    """Estimation quantile of the mean of the control group

    Args:
        control (ndarray):  array of control group data
        bootstrap_conf_level (float): confidence level for bootstrap confidence interval
        number_of_bootstrap_samples (int): number of bootstrap samples

    Returns:
        Tuple[float, np.ndarray]: quantile of the mean of the control group and bootstrap confidence interval

    """

    def sample():
        control_sample = np.random.choice(control, size=control.shape[0], replace=True)
        quantile_of_mean = percentileofscore(control_sample, np.mean(control_sample)) / 100
        return quantile_of_mean

    if control.shape[0] > 10000:
        pool = Pool(cpu_count())
        quantile_array = np.array(pool.starmap(sample,
                                               [() for i in range(number_of_bootstrap_samples)]))
        pool.close()
    else:
        quantile_array = np.array([sample() for i in range(number_of_bootstrap_samples)])

    confidence_interval = estimate_confidence_interval(quantile_array,
                                                       bootstrap_conf_level=bootstrap_conf_level)
    return np.mean(quantile_array), confidence_interval


def one_sample_bootstrap(control: np.ndarray, bootstrap_conf_level: float = 0.95,
                         number_of_bootstrap_samples: int = 10000,
                         sample_size: int = None,
                         statistic: Callable = np.mean, plot: bool = False, progress_bar: bool = False) -> Tuple[
    float, float, np.ndarray, np.ndarray]:
    """One sample bootstrap

    If control.shape[0] > 10000, then multiprocessing will be used

    Args:
        control (ndarray): 1D array containing control sample
        bootstrap_conf_level (float): Confidence level
        number_of_bootstrap_samples (int): Number of bootstrap samples
        sample_size (int): Sample size. Defaults to None. If None,
            then control_sample_size and treatment_sample_size
            wiil be equal to control.shape[0] and treatment.shape[0] respectively
        statistic (Callable): Statistic function. Defaults to difference_of_mean.
            Choose statistic function from compare_functions.py
        plot (bool): If True, then bootstrap plot will be shown. Defaults to True
        progress_bar (bool): If True, then progress bar will be shown. Defaults to False
    Returns:
        Tuple[float, float, ndarray, ndarray]: Tuple containing p-value, difference distribution statistic,
            bootstrap confidence interval, bootstrap difference distribution

    """

    def sample():
        control_sample = np.random.choice(control, control_sample_size, replace=True)
        return statistic(control_sample)

    control_sample_size = sample_size if sample_size else control.shape[0]

    if control.shape[0] <= 10000:
        if progress_bar:
            bootstrap_difference_distribution = np.array([sample() for i in tqdm(range(number_of_bootstrap_samples))])
        else:
            bootstrap_difference_distribution = np.array([sample() for i in range(number_of_bootstrap_samples)])
    else:
        pool = Pool(cpu_count())
        bootstrap_difference_distribution = np.array(
            pool.starmap(sample, [() for i in range(number_of_bootstrap_samples)]))
        pool.close()

    bootstrap_confidence_interval = estimate_confidence_interval(bootstrap_difference_distribution,
                                                                 bootstrap_conf_level)
    bootstrap_difference_mean = bootstrap_difference_distribution.mean()
    p_value = estimate_p_value(bootstrap_difference_distribution, number_of_bootstrap_samples)

    if plot:
        binwidth, _ = estimate_bin_params(bootstrap_difference_distribution)
        plt.hist(bootstrap_difference_distribution,
                 bins=np.arange(bootstrap_difference_distribution.min(),
                                bootstrap_difference_distribution.max() + binwidth,
                                binwidth))
        xlabel = ' '.join([i.capitalize() for i in statistic.__name__.split('_')])
        plt.title('Bootstrap')
        plt.xlabel(xlabel)
        plt.ylabel('Count')
        plt.axvline(x=bootstrap_confidence_interval[0], color='red', linestyle='dashed', linewidth=2)
        plt.axvline(x=bootstrap_confidence_interval[1], color='red', linestyle='dashed', linewidth=2)
        plt.axvline(x=bootstrap_difference_distribution.mean(), color='black', linestyle='dashed', linewidth=5)
        plt.show()
    return p_value, bootstrap_difference_mean, bootstrap_confidence_interval, bootstrap_difference_distribution


def spotify_one_sample_bootstrap(sample: np.ndarray, sample_size: int = None, quantile_of_interest: float = 0.5,
                                 bootstrap_conf_level: float = 0.95, number_of_bootstrap_samples: int = 10000,
                                 plot=False) -> Tuple[float, np.ndarray, np.ndarray]:
    """One-sample Spotify-Bootstrap

    Mårten Schultzberg and Sebastian Ankargren. “Resampling-free bootstrap inference for quantiles.”
    arXiv e-prints, art. arXiv:2202.10992, (2022). https://arxiv.org/abs/2202.10992

    Args:
        sample (ndarray): 1D array containing sample
        sample_size (int): sample size. Defaults to None
        quantile_of_interest (float): Quantile of interest. Defaults to 0.5
        bootstrap_conf_level (float): Confidence level. Defaults to 0.95
        number_of_bootstrap_samples (int): Number of bootstrap samples. Defaults to 10000
        plot (bool): Plot histogram of bootstrap distribution. Defaults to False


    Returns:
        Tuple[float, ndarray, ndarray]: Tuple containing quantile of interest, bootstrap confidence interval and bootstrap distribution

    """
    if not sample_size:
        sample_size = sample.shape[0]
    sample_sorted = np.sort(sample)
    left_quant = (1 - bootstrap_conf_level) / 2
    right_quant = 1 - (1 - bootstrap_conf_level) / 2
    ci_indexes = binom.ppf([left_quant, right_quant], sample_size + 1, quantile_of_interest)
    bootstrap_confidence_interval = sample_sorted[[int(np.floor(ci_indexes[0])), int(np.ceil(ci_indexes[1]))]]
    quantile_indices = binomial(sample_size + 1, quantile_of_interest, number_of_bootstrap_samples)
    bootstrap_distribution = sample_sorted[quantile_indices]

    if plot:
        plt.title('Bootstrap')
        plt.xlabel('q_' + str(quantile_of_interest))
        plt.ylabel('Count')
        plt.hist(bootstrap_distribution, bins=100, density=True)
        plt.axvline(x=bootstrap_confidence_interval[0], color='red', linestyle='dashed', linewidth=2)
        plt.axvline(x=bootstrap_confidence_interval[1], color='red', linestyle='dashed', linewidth=2)
        plt.axvline(x=bootstrap_distribution.mean(), color='black', linestyle='dashed', linewidth=5)
        plt.show()
    return np.quantile(sample_sorted, quantile_of_interest), bootstrap_confidence_interval, bootstrap_distribution
