import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from multiprocess import Pool, cpu_count
from tqdm.auto import tqdm


def estimate_quants(boot_data, bootstrap_conf_level=0.95):
    left_quant = (1 - bootstrap_conf_level) / 2
    right_quant = 1 - (1 - bootstrap_conf_level) / 2
    quants = np.quantile(boot_data, [left_quant, right_quant])
    return quants


def estimate_pvalue(boot_data):
    p_1 = norm.cdf(x=0, loc=boot_data.mean(), scale=boot_data.std())
    p_2 = norm.cdf(x=0, loc=-boot_data.mean(), scale=boot_data.std())
    return min(p_1, p_2) * 2


def naive_bootstrap(sample_1, sample_2, bootstrap_conf_level=0.95, boot_it=10000, boot_len=None, statistic=np.mean):
    if not boot_len:
        boot_len = np.max([sample_1.shape[0], sample_2.shape[0]])

    boot_data = np.zeros(shape=boot_it)
    for i in tqdm(range(boot_it)):
        samples_1 = np.random.choice(sample_1, boot_len, replace=True)
        samples_2 = np.random.choice(sample_2, boot_len, replace=True)
        boot_data[i] = statistic(samples_1 - samples_2)

    quants = estimate_quants(boot_data, bootstrap_conf_level)
    p_value = estimate_pvalue(boot_data)

    return boot_data, quants, p_value


def parallel_bootstrap(sample_1, sample_2, bootstrap_conf_level=0.95, boot_it=10000, boot_len=None, statistic=np.mean):
    def sample():
        import numpy as np
        samples_1 = np.random.choice(sample_1, boot_len, replace=True)
        samples_2 = np.random.choice(sample_2, boot_len, replace=True)
        return statistic(samples_1 - samples_2)

    if not boot_len:
        boot_len = np.max([sample_1.shape[0], sample_2.shape[0]])

    pool = Pool(cpu_count())
    boot_data = np.array(pool.starmap(sample, [() for i in range(boot_it)]))
    pool.close()

    quants = estimate_quants(boot_data, bootstrap_conf_level)
    p_value = estimate_pvalue(boot_data)

    return boot_data, quants, p_value


def estimate_bin_params(sample):
    q1 = np.quantile(sample, 0.25)
    q3 = np.quantile(sample, 0.75)
    iqr = q3 - q1
    bin_width = (2 * iqr) / (sample.shape[0] ** (1 / 3))
    bin_count = int(np.ceil((sample.max() - sample.min()) / bin_width))
    return bin_width, bin_count


def bootstrap_plot(boot_data, quants, statistic=None):
    if isinstance(statistic, str):
        xlabel = statistic
    elif hasattr(statistic, '__call__'):
        xlabel = statistic.__name__[0].upper() + statistic.__name__[1:] + '(Difference)'
    else:
        xlabel = 'Stat(Difference)'

    binwidth, _ = estimate_bin_params(boot_data)
    plt.hist(boot_data, bins=np.arange(boot_data.min(), boot_data.max() + binwidth, binwidth))
    plt.title('Bootstrap')
    plt.xlabel(xlabel)
    plt.ylabel('Count')
    plt.axvline(x=quants[0], color='red', linestyle='dashed', linewidth=2)
    plt.axvline(x=quants[1], color='red', linestyle='dashed', linewidth=2)
    plt.axvline(x=np.mean(boot_data), color='black', linestyle='dashed', linewidth=5)
    plt.axvline(x=0, color='white', linewidth=5)
    plt.show()


def bootstrap(sample_1, sample_2, bootstrap_conf_level=0.95, boot_it=10000, boot_len=None, statistic=np.mean,
              plot=True):
    if np.max([sample_1.shape[0], sample_2.shape[0]]) <= 10000:
        boot_data, quants, p_value = naive_bootstrap(sample_1, sample_2, bootstrap_conf_level, boot_it, boot_len,
                                                     statistic)
    else:
        boot_data, quants, p_value = parallel_bootstrap(sample_1, sample_2, bootstrap_conf_level, boot_it, boot_len,
                                                        statistic)

    if plot:
        bootstrap_plot(boot_data, quants, statistic)
    return boot_data, quants, p_value


def naive_ctr_bootstrap(control, treatment, bootstrap_conf_level=0.95, boot_it=10000, boot_len=None):
    if not boot_len:
        boot_len = np.max([control.shape[0], treatment.shape[0]])

    boot_data = np.zeros(shape=boot_it)
    for i in tqdm(range(boot_it)):
        control_sample = control.sample(boot_len, replace=True)
        treatment_sample = treatment.sample(boot_len, replace=True)
        ctr_control_sample = control_sample.clicks.sum() / control_sample.views.sum()
        ctr_treatment_sample = treatment_sample.clicks.sum() / treatment_sample.views.sum()
        boot_data[i] = ctr_treatment_sample - ctr_control_sample

    quants = estimate_quants(boot_data, bootstrap_conf_level)
    p_value = estimate_pvalue(boot_data)

    return boot_data, quants, p_value


def parallel_ctr_bootstrap(control, treatment, bootstrap_conf_level=0.95, boot_it=10000, boot_len=None):
    def sample():
        import numpy as np
        control_sample = control.sample(boot_len, replace=True)
        treatment_sample = treatment.sample(boot_len, replace=True)
        ctr_control_sample = control_sample.clicks.sum() / control_sample.views.sum()
        ctr_treatment_sample = treatment_sample.clicks.sum() / treatment_sample.views.sum()
        return ctr_treatment_sample - ctr_control_sample

    if not boot_len:
        boot_len = np.max([control.shape[0], treatment.shape[0]])

    pool = Pool(cpu_count())
    boot_data = np.array(pool.starmap(sample, [() for i in range(boot_it)]))
    pool.close()

    quants = estimate_quants(boot_data, bootstrap_conf_level)
    p_value = estimate_pvalue(boot_data)

    return boot_data, quants, p_value


def ctr_bootstrap(sample_1, sample_2, bootstrap_conf_level=0.95, boot_it=10000, boot_len=None, statistic=np.mean,
                  plot=True):
    if np.max([sample_1.shape[0], sample_2.shape[0]]) <= 10000:
        boot_data, quants, p_value = naive_ctr_bootstrap(sample_1, sample_2, bootstrap_conf_level, boot_it, boot_len)
    else:
        boot_data, quants, p_value = parallel_ctr_bootstrap(sample_1, sample_2, bootstrap_conf_level, boot_it, boot_len)

    if plot:
        bootstrap_plot(boot_data, quants, statistic='CTR Difference')
    return boot_data, quants, p_value
