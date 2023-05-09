import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, binom
from numpy.random import normal, binomial
from multiprocess import Pool, cpu_count
from tqdm.auto import tqdm
from compare_functions import difference_of_mean


def estimate_bootstrap_confidence_interval(bootstrap_difference_distribution, bootstrap_conf_level=0.95):
    left_quant = (1 - bootstrap_conf_level) / 2
    right_quant = 1 - (1 - bootstrap_conf_level) / 2
    bootstrap_confidence_interval = np.quantile(bootstrap_difference_distribution, [left_quant, right_quant]).tolist()
    return bootstrap_confidence_interval


def estimate_p_value(bootstrap_difference_distribution, number_of_bootstrap_samples):
    positions = np.sum(bootstrap_difference_distribution < 0, axis=0)
    return 2 * np.minimum(positions, number_of_bootstrap_samples - positions) / number_of_bootstrap_samples


def estimate_bin_params(sample):
    q1 = np.quantile(sample, 0.25)
    q3 = np.quantile(sample, 0.75)
    iqr = q3 - q1
    bin_width = (2 * iqr) / (sample.shape[0] ** (1 / 3))
    bin_count = int(np.ceil((sample.max() - sample.min()) / bin_width))
    return bin_width, bin_count


def bootstrap_plot(bootstrap_difference_distribution, bootstrap_difference_mean, bootstrap_confidence_interval,
                   statistic=None):
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
    plt.axvline(x=bootstrap_difference_mean, color='black', linestyle='dashed', linewidth=5)
    plt.axvline(x=0, color='white', linewidth=5)
    plt.show()


def bootstrap(control, treatment, bootstrap_conf_level=0.95, number_of_bootstrap_samples=10000, sample_size=None,
              statistic=difference_of_mean, plot=True):
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

    bootstrap_confidence_interval = estimate_bootstrap_confidence_interval(bootstrap_difference_distribution,
                                                                           bootstrap_conf_level)
    bootstrap_difference_mean = bootstrap_difference_distribution.mean()
    p_value = estimate_p_value(bootstrap_difference_distribution, number_of_bootstrap_samples)

    if plot:
        bootstrap_plot(bootstrap_difference_distribution, bootstrap_difference_mean, bootstrap_confidence_interval,
                       statistic)
    return p_value, bootstrap_difference_mean, bootstrap_confidence_interval, bootstrap_difference_distribution


def ctr_bootstrap(control, treatment, bootstrap_conf_level=0.95, number_of_bootstrap_samples=10000, sample_size=None,
                  plot=True):
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

    bootstrap_confidence_interval = estimate_bootstrap_confidence_interval(bootstrap_difference_distribution,
                                                                           bootstrap_conf_level)
    bootstrap_difference_mean = bootstrap_difference_distribution.mean()
    p_value = estimate_p_value(bootstrap_difference_distribution, number_of_bootstrap_samples)

    if plot:
        bootstrap_plot(bootstrap_difference_distribution, bootstrap_difference_mean, bootstrap_confidence_interval,
                       statistic='CTR Difference')
    return p_value, bootstrap_difference_mean, bootstrap_confidence_interval, bootstrap_difference_distribution


def spotify_one_sample_bootstrap(sample, sample_size=None, quantile_of_interest=0.5, bootstrap_conf_level=0.95):
    if not sample_size:
        sample_size = sample.shape[0]
    sample_sorted = np.sort(sample)
    left_quant = (1 - bootstrap_conf_level) / 2
    right_quant = 1 - (1 - bootstrap_conf_level) / 2
    ci_indexes = binom.ppf([left_quant, right_quant], sample_size + 1, quantile_of_interest)
    bootstrap_confidence_interval = sample_sorted[[int(np.floor(ci_indexes[0])), int(np.ceil(ci_indexes[1]))]].tolist()
    return np.quantile(sample_sorted, quantile_of_interest), bootstrap_confidence_interval


def spotify_two_sample_bootstrap(control, treatment, number_of_bootstrap_samples=10000,
                                 sample_size=None, quantile_of_interest=0.5, bootstrap_conf_level=0.95, plot=True):
    if sample_size:
        control_sample_size, treatment_sample_size = [sample_size] * 2
    else:
        control_sample_size, treatment_sample_size = control.shape[0], treatment.shape[0]

    sorted_control = np.sort(control)
    sorted_treatment = np.sort(treatment)
    bootstrap_difference_distribution = sorted_treatment[binomial(treatment_sample_size + 1, quantile_of_interest,
                                                                  number_of_bootstrap_samples)] - sorted_control[
                                            binomial(control_sample_size + 1, quantile_of_interest,
                                                     number_of_bootstrap_samples)]

    bootstrap_difference_mean = np.quantile(sorted_treatment, quantile_of_interest) - np.quantile(sorted_control,
                                                                                                  quantile_of_interest)

    bootstrap_confidence_interval = estimate_bootstrap_confidence_interval(bootstrap_difference_distribution,
                                                                           bootstrap_conf_level)
    p_value = estimate_p_value(bootstrap_difference_distribution, number_of_bootstrap_samples)
    if plot:
        statistic = f'q-{quantile_of_interest} Difference'
        bootstrap_plot(bootstrap_difference_distribution, bootstrap_difference_mean, bootstrap_confidence_interval,
                       statistic=statistic)
    return p_value, bootstrap_difference_mean, bootstrap_confidence_interval, bootstrap_difference_distribution


def poisson_bootstrap(control, treatment, number_of_bootstrap_samples=10000):
    poisson_bootstraps = scipy.stats.poisson(1).rvs((number_of_bootstrap_samples, control.shape[0])).astype(np.int64)

    values_1 = np.matmul(control, poisson_bootstraps.T)
    values_2 = np.matmul(treatment, poisson_bootstraps.T)

    difference = values_2 - values_1

    positions = np.sum(difference < 0, axis=0)

    return 2 * np.minimum(positions, number_of_bootstrap_samples - positions) / number_of_bootstrap_samples
