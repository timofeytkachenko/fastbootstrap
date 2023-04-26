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


def estimate_p_value(bootstrap_difference_mean, bootstrap_difference_std):
    p_1 = norm.cdf(x=0, loc=bootstrap_difference_mean, scale=bootstrap_difference_std)
    p_2 = norm.cdf(x=0, loc=-bootstrap_difference_mean, scale=bootstrap_difference_std)
    return min(p_1, p_2) * 2


def estimate_bin_params(sample):
    q1 = np.quantile(sample, 0.25)
    q3 = np.quantile(sample, 0.75)
    iqr = q3 - q1
    bin_width = (2 * iqr) / (sample.shape[0] ** (1 / 3))
    bin_count = int(np.ceil((sample.max() - sample.min()) / bin_width))
    return bin_width, bin_count


def bootstrap_plot(bootstrap_difference_distribution, bootstrap_confidence_interval, statistic=None):
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
    plt.axvline(x=np.mean(bootstrap_difference_distribution), color='black', linestyle='dashed', linewidth=5)
    plt.axvline(x=0, color='white', linewidth=5)
    plt.show()


def bootstrap(sample_1, sample_2, bootstrap_conf_level=0.95, number_of_bootstrap_samples=10000, sample_size=None,
              statistic=difference_of_mean, plot=True):
    def sample():
        import numpy as np
        samples_1 = np.random.choice(sample_1, sample_size, replace=True)
        samples_2 = np.random.choice(sample_2, sample_size, replace=True)
        return statistic(samples_1, samples_2)

    if not sample_size:
        sample_size = np.max([sample_1.shape[0], sample_2.shape[0]])

    if np.max([sample_1.shape[0], sample_2.shape[0]]) <= 10000:
        bootstrap_difference_distribution = np.zeros(shape=number_of_bootstrap_samples)
        for i in tqdm(range(number_of_bootstrap_samples)):
            samples_1 = np.random.choice(sample_1, sample_size, replace=True)
            samples_2 = np.random.choice(sample_2, sample_size, replace=True)
            bootstrap_difference_distribution[i] = statistic(samples_1, samples_2)
    else:
        pool = Pool(cpu_count())
        bootstrap_difference_distribution = np.array(
            pool.starmap(sample, [() for i in range(number_of_bootstrap_samples)]))
        pool.close()

    bootstrap_confidence_interval = estimate_bootstrap_confidence_interval(bootstrap_difference_distribution,
                                                                           bootstrap_conf_level)
    bootstrap_difference_mean = bootstrap_difference_distribution.mean()
    bootstrap_difference_std = bootstrap_difference_distribution.std()
    p_value = estimate_p_value(bootstrap_difference_mean, bootstrap_difference_std)

    if plot:
        bootstrap_plot(bootstrap_difference_distribution, bootstrap_confidence_interval, statistic)
    return p_value, bootstrap_difference_mean, bootstrap_confidence_interval, bootstrap_difference_distribution


def ctr_bootstrap(control, treatment, bootstrap_conf_level=0.95, number_of_bootstrap_samples=10000, sample_size=None,
                  plot=True):
    def sample():
        import numpy as np
        control_sample = control.sample(sample_size, replace=True)
        treatment_sample = treatment.sample(sample_size, replace=True)
        ctr_control_sample = control_sample.clicks.sum() / control_sample.views.sum()
        ctr_treatment_sample = treatment_sample.clicks.sum() / treatment_sample.views.sum()
        return ctr_treatment_sample - ctr_control_sample

    if not sample_size:
        sample_size = np.max([control.shape[0], treatment.shape[0]])

    if np.max([control.shape[0], treatment.shape[0]]) <= 10000:
        bootstrap_difference_distribution = np.zeros(shape=number_of_bootstrap_samples)
        for i in tqdm(range(number_of_bootstrap_samples)):
            control_sample = control.sample(sample_size, replace=True)
            treatment_sample = treatment.sample(sample_size, replace=True)
            ctr_control_sample = control_sample.clicks.sum() / control_sample.views.sum()
            ctr_treatment_sample = treatment_sample.clicks.sum() / treatment_sample.views.sum()
            bootstrap_difference_distribution[i] = ctr_treatment_sample - ctr_control_sample
    else:
        pool = Pool(cpu_count())
        bootstrap_difference_distribution = np.array(
            pool.starmap(sample, [() for i in range(number_of_bootstrap_samples)]))
        pool.close()

    bootstrap_confidence_interval = estimate_bootstrap_confidence_interval(bootstrap_difference_distribution,
                                                                           bootstrap_conf_level)
    bootstrap_difference_mean = bootstrap_difference_distribution.mean()
    bootstrap_difference_std = bootstrap_difference_distribution.std()
    p_value = estimate_p_value(bootstrap_difference_mean, bootstrap_difference_std)

    if plot:
        bootstrap_plot(bootstrap_difference_distribution, bootstrap_confidence_interval, statistic='CTR Difference')
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


def spotify_two_sample_bootstrap(sample_1, sample_2, number_of_bootstrap_samples=10000,
                                 sample_size=None, quantile_of_interest=0.5, bootstrap_conf_level=0.95, plot=True):
    if sample_size:
        sample_size_1, sample_size_2 = [sample_size] * 2
    else:
        sample_size_1, sample_size_2 = sample_1.shape[0], sample_2.shape[0]

    sorted_sample_1 = np.sort(sample_1)
    sorted_sample_2 = np.sort(sample_2)
    bootstrap_difference_distribution = sorted_sample_2[binomial(sample_size_2 + 1, quantile_of_interest,
                                                                 number_of_bootstrap_samples)] - sorted_sample_1[
                                            binomial(sample_size_1 + 1, quantile_of_interest,
                                                     number_of_bootstrap_samples)]

    bootstrap_difference_mean = np.quantile(sorted_sample_2, quantile_of_interest) - np.quantile(sorted_sample_1,
                                                                                                 quantile_of_interest)
    bootstrap_difference_std = bootstrap_difference_distribution.std()

    bootstrap_confidence_interval = estimate_bootstrap_confidence_interval(bootstrap_difference_distribution,
                                                                           bootstrap_conf_level)
    p_value = estimate_p_value(bootstrap_difference_mean, bootstrap_difference_std)
    if plot:
        statistic = f'q-{quantile_of_interest} Difference'
        bootstrap_plot(bootstrap_difference_distribution, bootstrap_confidence_interval, statistic=statistic)
    return p_value, bootstrap_difference_mean, bootstrap_confidence_interval, bootstrap_difference_distribution
