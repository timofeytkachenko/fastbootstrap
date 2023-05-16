import scipy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.stats import norm, binom
from numpy.random import normal, binomial
from multiprocess import Pool, cpu_count
from compare_functions import difference_of_mean, difference
from scipy.stats import percentileofscore
from tqdm.auto import tqdm
from typing import Tuple, Union, Callable

colors = {'False': 'red', 'True': 'black'}


def estimate_quantile_of_mean(sample: np.ndarray, batch_size_percent: int = 5, bootstrap_conf_level: float = 0.95,
                              shuffle: bool = True) -> float:
    """

    Args:
        sample (ndarray): 1D array containing observations
        batch_size_percent (int): Batch size. Defaults to 5% of sample size
        bootstrap_conf_level (float): bootstrap confidence level. Defaults to 0.95
        shuffle (bool): If True, then sample will be shuffled. Defaults to True

    Returns:
        Tuple[float, ndarray]: Tuple containing mean quantile and quantile confidence interval

    """
    if shuffle:
        np.random.shuffle(sample)

    batch_size = int(np.ceil(sample.shape[0] * batch_size_percent / 100))
    batches = np.array_split(sample, batch_size)
    batch_means = np.array([np.mean(batch) for batch in batches])
    mean = np.median(batch_means)
    mean_quantile_distribution = np.array([percentileofscore(batch, score=mean) / 100 for batch in batches])
    confidence_interval = estimate_confidence_interval(mean_quantile_distribution,
                                                       bootstrap_conf_level=bootstrap_conf_level)
    return np.median(mean_quantile_distribution), confidence_interval


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
              statistic: Callable = difference_of_mean, plot: bool = True) -> Tuple[
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
        bootstrap_difference_distribution = np.array([sample() for i in tqdm(range(number_of_bootstrap_samples))])
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


def ctr_bootstrap(control, treatment, bootstrap_conf_level=0.95, number_of_bootstrap_samples=10000, sample_size=None,
                  plot=True):
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
        bootstrap_difference_distribution = np.array([sample() for i in tqdm(range(number_of_bootstrap_samples))])
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


def spotify_two_sample_bootstrap(control: object, treatment: object, number_of_bootstrap_samples: object = 10000,
                                 sample_size: object = None, quantile_of_interest: object = 0.5,
                                 statistic: object = difference,
                                 bootstrap_conf_level: object = 0.95,
                                 plot: object = True) -> object:
    """Two-sample Spotify-Bootstrap
    
    Mårten Schultzberg and Sebastian Ankargren. “Resampling-free bootstrap inference for quantiles.”
    arXiv e-prints, art. arXiv:2202.10992, (2022). https://arxiv.org/abs/2202.10992

    Args:
        control (ndarray): 1D array containing control sample
        treatment (ndarray): 1D array containing treatment sample
        bootstrap_conf_level (float): Confidence level
        number_of_bootstrap_samples (int): Number of bootstrap samples
        sample_size (int): Sample size. Defaults to None. If None,
            then control_sample_size and treatment_sample_size
            wiil be equal to control.shape[0] and treatment.shape[0] respectively
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
        binomial(treatment_sample_size + 1, quantile_of_interest, number_of_bootstrap_samples)]
    control_sample_values = sorted_control[
        binomial(control_sample_size + 1, quantile_of_interest, number_of_bootstrap_samples)]
    bootstrap_difference_distribution = statistic(control_sample_values, treatment_sample_values)

    bootstrap_difference_mean = statistic(
        np.quantile(sorted_control, quantile_of_interest), np.quantile(sorted_treatment, quantile_of_interest))

    bootstrap_confidence_interval = estimate_confidence_interval(bootstrap_difference_distribution,
                                                                 bootstrap_conf_level)
    p_value = estimate_p_value(bootstrap_difference_distribution, number_of_bootstrap_samples)
    if plot:
        statistic = f'q-{quantile_of_interest} ' + ' '.join([i.capitalize() for i in statistic.__name__.split('_')])
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


def quantile_bootstrap_plot(control: np.ndarray, treatment: np.ndarray, n_step: int = 20, q1: float = 0.01,
                            q2: float = 0.99, plot_type: str = 'line', statistic: Callable = difference) -> None:
    """

    Args:
        control (ndarray): 1D array containing control sample
        treatment (ndarray): 1D array containing treatment sample
        n_step (int): Number of quantiles to compare. Defaults to 20
        q1 (float): Lower quantile. Defaults to 0.025
        q2 (float): Upper quantile. Defaults to 0.975
        plot_type (str): Plot type. Defaults to 'line'. Choose from 'line' or 'bar'
        statistic (Callable): Statistic function. Defaults to difference.

    """
    quantiles_to_compare = np.linspace(q1, q2, n_step)
    statistics = list()
    for quantile in quantiles_to_compare:
        p_value, bootstrap_mean, bootstrap_confidence_interval, _ = spotify_two_sample_bootstrap(control, treatment,
                                                                                                 quantile_of_interest=quantile,
                                                                                                 statistic=statistic,
                                                                                                 plot=False)
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

    if plot_type == 'bar':
        fig = go.Figure(data=[go.Bar(
            x=df["quantile"],
            y=df["difference"],
            marker_color=df['significance'].map(colors),
            customdata=df

        )])

        fig.update_traces(
            error_y={
                "type": "data",
                "symmetric": False,
                "array": df["ci_upper"],
                "arrayminus": df["ci_lower"]
            }
        )

        fig.update_traces(
            name='Quantile Information',
            hovertemplate="<br>".join([
                "Quantile: %customdata[0]",
                "p_value: %{customdata[1]}",
                "Significance: %{customdata[5]}",
                "Difference: %{customdata[2]}",
                "Lower Bound: %{customdata[3]}",
                "Upper Bound: %{customdata[4]}"
            ])
        )

        fig.update_layout(title_text='Quantile Bootstrap Bar Plot')
        fig.show()
    else:
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
        fig.update_layout(
            yaxis_title='Difference',
            title='Quantile Bootstrap Line Plot',
            hovermode="x"
        )
        fig.show()