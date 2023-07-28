import imp

import numpy as np
import scipy
from numpy import linalg as LA

from scipy import special
import ErrorMethods as EM



def pre_getTransform(eps, randomized_bins=1024, domain_bins=1024):
    ee = np.exp(eps)
    w = ((eps * ee) - ee + 1) / (2 * ee * (ee - 1 - eps)) * 2
    p = ee / (w * ee + 1)
    q = 1 / (w * ee + 1)

    # report matrix
    m = randomized_bins
    n = domain_bins
    m_cell = (1 + w) / m
    n_cell = 1 / n

    transform = np.ones((m, n)) * q * m_cell
    for i in range(n):
        left_most_v = (i * n_cell)
        right_most_v = ((i + 1) * n_cell)

        ll_bound = int(left_most_v / m_cell)
        lr_bound = int((left_most_v + w) / m_cell)
        rl_bound = int(right_most_v / m_cell)
        rr_bound = int((right_most_v + w) / m_cell)

        ll_v = left_most_v - w / 2
        rl_v = right_most_v - w / 2
        l_p = ((ll_bound + 1) * m_cell - w / 2 - ll_v) * (p - q) + q * m_cell
        r_p = ((rl_bound + 1) * m_cell - w / 2 - rl_v) * (p - q) + q * m_cell
        if rl_bound > ll_bound:
            transform[ll_bound, i] = (l_p - q * m_cell) * (
                    (ll_bound + 1) * m_cell - w / 2 - ll_v) / n_cell * 0.5 + q * m_cell
            transform[ll_bound + 1, i] = p * m_cell - (p * m_cell - r_p) * (
                    rl_v - ((ll_bound + 1) * m_cell - w / 2)) / n_cell * 0.5
        else:
            transform[ll_bound, i] = (l_p + r_p) / 2
            transform[ll_bound + 1, i] = p * m_cell

        lr_v = left_most_v + w / 2
        rr_v = right_most_v + w / 2
        r_p = (rr_v - (rr_bound * m_cell - w / 2)) * (p - q) + q * m_cell
        l_p = (lr_v - (lr_bound * m_cell - w / 2)) * (p - q) + q * m_cell
        if rr_bound > lr_bound:
            if rr_bound < m:
                transform[rr_bound, i] = (r_p - q * m_cell) * (
                        rr_v - (rr_bound * m_cell - w / 2)) / n_cell * 0.5 + q * m_cell

            transform[rr_bound - 1, i] = p * m_cell - (p * m_cell - l_p) * (
                    (rr_bound * m_cell - w / 2) - lr_v) / n_cell * 0.5

        else:
            transform[rr_bound, i] = (l_p + r_p) / 2
            transform[rr_bound - 1, i] = p * m_cell

        if rr_bound - 1 > ll_bound + 2:
            transform[ll_bound + 2: rr_bound - 1, i] = p * m_cell

    return transform


def getSmoothMmatrix(n=1024):
    # smoothing matrix
    smoothing_factor = 2
    binomial_tmp = [scipy.special.binom(smoothing_factor, k) for k in range(smoothing_factor + 1)]
    smoothing_matrix = np.zeros((n, n))
    central_idx = int(len(binomial_tmp) / 2)
    for i in range(int(smoothing_factor / 2)):
        smoothing_matrix[i, : central_idx + i + 1] = binomial_tmp[central_idx - i:]
    for i in range(int(smoothing_factor / 2), n - int(smoothing_factor / 2)):
        smoothing_matrix[i, i - central_idx: i + central_idx + 1] = binomial_tmp
    for i in range(n - int(smoothing_factor / 2), n):
        remain = n - i - 1
        smoothing_matrix[i, i - central_idx + 1:] = binomial_tmp[: central_idx + remain]
    row_sum = np.sum(smoothing_matrix, axis=1)
    smoothing_matrix = (smoothing_matrix.T / row_sum).T
    return smoothing_matrix


def pre_sw(ori_samples, l, h, eps):
    """
        :param ori_samples:原始数据（分桶后数据0~K-1）
        :param l: low 1
        :param h: K
        :param eps: epsilon
        :return:EMS(n, ns_hist, transform, max_iteration, loglikelihood_threshold) * len(ori_samples)
    """
    ee = np.exp(eps)
    w = ((eps * ee) - ee + 1) / (2 * ee * (ee - 1 - eps)) * 2
    p = ee / (w * ee + 1)
    q = 1 / (w * ee + 1)

    samples = (ori_samples - l) / (h - l)
    randoms = np.random.uniform(0, 1, len(samples))
    noisy_samples = np.zeros_like(samples)

    # report
    index = randoms <= (q * samples)
    noisy_samples[index] = randoms[index] / q - w / 2
    index = randoms > (q * samples)
    noisy_samples[index] = (randoms[index] - q * samples[index]) / p + samples[index] - w / 2
    index = randoms > q * samples + p * w
    noisy_samples[index] = (randoms[index] - q * samples[index] - p * w) / q + samples[index] + w / 2

    return noisy_samples


def pre_EMS(noisy_samples, eps, transform, smoothing_matrix, n=1024, randomized_bins=1024, max_iteration=10000,
            loglikelihood_threshold=1e-3):
    """
        :param n: domain_bins=1024
        :return: theta like np.ones(n)
    """
    ee = np.exp(eps)
    w = ((eps * ee) - ee + 1) / (2 * ee * (ee - 1 - eps)) * 2
    ns_hist, _ = np.histogram(noisy_samples, bins=randomized_bins, range=(-w / 2, 1 + w / 2))

    # EMS
    theta = np.ones(n) / float(n)
    theta_old = np.zeros(n)
    r = 0
    sample_size = sum(ns_hist)
    old_logliklihood = 0

    while LA.norm(theta_old - theta, ord=1) > 1 / sample_size and r < max_iteration:
        theta_old = np.copy(theta)
        X_condition = np.matmul(transform, theta_old)

        TMP = transform.T / X_condition

        P = np.copy(np.matmul(TMP, ns_hist))
        P = P * theta_old

        theta = np.copy(P / sum(P))

        # Smoothing step
        theta = np.matmul(smoothing_matrix, theta)
        theta = theta / sum(theta)

        logliklihood = np.inner(ns_hist, np.log(np.matmul(transform, theta)))
        imporve = logliklihood - old_logliklihood

        if r > 1 and abs(imporve) < loglikelihood_threshold:
            # print("stop when", imporve / old_logliklihood, loglikelihood_threshold)
            break

        old_logliklihood = logliklihood

        r += 1
    return theta

def SW_EMS(K, d, ori_data, user_num, eps):
    transform = pre_getTransform(eps, randomized_bins=K, domain_bins=K)
    smoothing_matrix = getSmoothMmatrix(K)
    noisy_data = pre_sw(ori_data, 0, d, eps)
    theta = pre_EMS(noisy_data, eps, transform, smoothing_matrix, n=K, randomized_bins=K, max_iteration=10000,
                    loglikelihood_threshold=1e-3)

    return theta

def count(data, num, d=1):
    c = np.zeros(num)
    for v in data:  # 每个数据
        index = int(v * num / d)
        if index < num:
            c[index] += 1
        else:
            c[num - 1] += 1
    return c


if __name__ == "__main__":

    # income_numerical.npy, 524200, 2308374
    # Retirement_numerical.npy, 59690.74, 178012
    # taxi_pickup_time_numerical.npy, 86399, 2189968

    K = 256

    for file in ['beta_numerical.npy']:

        kl_errors = []

        emd_errors = []

        ks_errors = []

        mean_errors = []

        Quantiles_errors = []

        range1_errors = []

        range4_errors = []

        variance_errors = []

        for eps in [0.1,0.2,0.3,0.4,0.5]:

            kl_error = []
            
            emd_error = []

            ks_error = []

            mean_error = []

            Quantiles_error = []

            range1_error = []

            range4_error = []

            variance_error = []

            for time in range(10):

                if time % 10 == 0:
                    print(file, eps, time)

                samples = np.load(file)

                samples = (samples - np.min(samples)) / (np.max(samples) - np.min(samples))

                theta = SW_EMS(K, 1, samples, len(samples), eps)

                real = count(samples, K) / len(samples)

                # KL
                KL = EM.KL(real, theta)
                kl_error.append(KL)

                # emd
                EMD = EM.EMD_value(theta, real, K)
                emd_error.append(EMD)

                # ks
                KS = EM.KS_value(theta, real, K)
                ks_error.append(KS)

                # mean
                MEAN = EM.Mean_value(theta, real, K)
                mean_error.append(MEAN)

                # Quantiles
                QUANTITLE = EM.Quantiles_value(theta, real, K)
                Quantiles_error.append(QUANTITLE)

                # Range_Query
                RANGEQUERY_1 = EM.Range_Query(theta, real, 0.1, K)
                RANGEQUERY_4 = EM.Range_Query(theta, real, 0.4, K)
                range1_error.append(RANGEQUERY_1)
                range4_error.append(RANGEQUERY_4)

                # Variance
                VARIANCE = EM.Variance_value(samples,theta, real, K)
                variance_error.append(VARIANCE)

            kl_errors.append((np.mean(kl_error)))
            emd_errors.append(np.mean(emd_error))
            ks_errors.append(np.mean(ks_error))
            mean_errors.append(np.mean(mean_error))
            Quantiles_errors.append(np.mean(Quantiles_error))
            range1_errors.append(np.mean(range1_error))
            range4_errors.append(np.mean(range4_error))
            variance_errors.append(np.mean(variance_error))

            print("--------------------------------------")
            print("dataset and epsilon:", file, eps)
            print("SW+EMS estimate KL:", kl_errors[-1])
            print("SW+EMS estimate EMD:", emd_errors[-1])
            print("SW+EMS estimate KS:", ks_errors[-1])
            print("SW+EMS estimate MEAN:", mean_errors[-1])
            print("SW+EMS estimate QUANTITLE:", Quantiles_errors[-1])
            print("SW+EMS estimate RANGEQUERY_1:", range1_errors[-1])
            print("SW+EMS estimate RANGEQUERY_4:", range4_error[-1])
            print("SW+EMS estimate VARIANCE:", variance_errors[-1])
            print("--------------------------------------")

        print("--------------------------------------")
        print("dataset:", file)
        print("SW+EMS estimate KL:", kl_errors)
        print("SW+EMS estimate EMD:", emd_errors)
        print("SW+EMS estimate KS:", ks_errors)
        print("SW+EMS estimate MEAN:", mean_errors)
        print("SW+EMS estimate QUANTITLE:", Quantiles_errors)
        print("SW+EMS estimate RANGEQUERY_1:", range1_errors)
        print("SW+EMS estimate RANGEQUERY_4:", range4_errors)
        print("SW+EMS estimate VARIANCE:", variance_errors)
        print("--------------------------------------")
