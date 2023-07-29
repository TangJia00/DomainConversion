import copy
import csv
import math
import numpy as np
import scipy
import ErrorMethods as EM
import matplotlib.pyplot as plt

from scipy import special
from random import shuffle
from numpy import linalg as LA

f_out = open('result.txt', 'w')


def buck_uniform(n, K, d, old_Q, old_p, old_h):
    new_p = np.zeros(K)
    w = d / K
    i = 0
    sweep = old_Q[i + 1]
    for index in range(K):
        l = index * w
        r = (index + 1) * w
        if sweep >= r:
            new_p[index] += w * old_h[i]
            if sweep == r and i != n - 1:
                i += 1
                sweep = old_Q[i + 1]
            continue
        else:  # sweep<r
            new_p[index] += old_h[i] * (sweep - l)
            i += 1
            sweep = old_Q[i + 1]
            while (sweep < r):
                new_p[index] += old_p[i]
                i += 1
                sweep = old_Q[i + 1]
            if sweep == r:
                new_p[index] += old_p[i]
                if i != n - 1:
                    i += 1
                    sweep = old_Q[i + 1]
                continue
            else:  # sweep>r
                new_p[index] += (r - old_Q[i]) * old_h[i]
    return new_p


def count(data, num, d=1):
    c = np.zeros(num)
    for v in data:
        index = int(v * num / d)
        if index < num:
            c[index] += 1
        else:
            c[num - 1] += 1
    return c


def Find_Quantile(K, data, Quantile):
    result = 0
    right = K
    left = 0
    if data >= Quantile[-1]:
        return K - 1
    while left <= right:
        middle = (left + right) // 2
        if Quantile[middle] <= data < Quantile[middle + 1]:
            result = middle
            break
        elif Quantile[middle + 1] <= data:
            left = middle + 1
        else:
            right = middle - 1
    return result


def getPComplete(h, width, index):
    result = 0
    for i in range(index):
        result += (h[i] * width[i])
    return result


def FMap_0(samples, h, Quantile, K):
    num = len(samples)
    Fx = np.zeros(num)
    wide = 1/K
    for i in range(num):
        Q_index = Find_Quantile(K, samples[i], Quantile)
        Fx[i] = sum(h[0:Q_index])*wide + h[Q_index] * (samples[i] - Quantile[Q_index])
    return Fx


def FMap(samples, h, Quantile, K):
    num = len(samples)
    Fx = np.zeros(num)
    width = [0] * K
    for i in range(K):
        width[i] = Quantile[i + 1] - Quantile[i]
    for i in range(num):
        Q_index = Find_Quantile(K, samples[i], Quantile)
        Fx[i] = getPComplete(h, width, Q_index) + h[Q_index] * (samples[i] - Quantile[Q_index])
    return Fx


def ReFMap(h, Quantile, K):
    datas = [0] * (K + 1)
    for i in range(K):
        datas[i + 1] = (i + 1) / K
    ReF = [0] * (K + 1)
    CumulativeFrequency = [0] * (K + 1)
    width = [0] * K
    for i in range(K):
        width[i] = Quantile[i + 1] - Quantile[i]
    for i in range(K):
        CumulativeFrequency[i + 1] = h[i] * width[i] + CumulativeFrequency[i]

    for i in range(K - 1):
        index = Find_Quantile(K, datas[i + 1], CumulativeFrequency)
        ReF[i + 1] = Quantile[index] + (datas[i + 1] - CumulativeFrequency[index]) / (
                    CumulativeFrequency[index + 1] - CumulativeFrequency[index]) * width[index]
    ReF[K] = 1
    return ReF


def smooth2(h1, k):
    spl = [[h1[i], h1[i], h1[i]] for i in range(len(h1))]
    smoo = []
    for index, item in enumerate(spl):
        if index == len(spl) - 1:
            smoo.extend(item)
            break

        hei = (item[2] - spl[index + 1][0]) / 3

        if hei <= 0:
            spl[index][2] += abs(hei)
            spl[index + 1][0] -= abs(hei)
        else:
            spl[index][2] -= abs(hei)
            spl[index + 1][0] += abs(hei)

        smoo.extend(item)

    return smoo


def get_h(old_Quan,old_h,new_Quan):
    new_h=np.zeros(len(new_Quan)-1)
    k=0
    n=0
    for S in new_Quan[1:]:
        new_h[n] = old_h[k]
        if S==old_Quan[k+1]:
            k+=1
        n+=1
    return new_h


def gather(Q1, h1, Q2, h2):
    Q_S = copy.deepcopy(Q1)
    Q_S.extend(Q2)
    Q_S = list(set(Q_S))
    Q_S = sorted(Q_S)
    n_new_interval = len(Q_S) - 1
    new1_h = get_h(Q1, h1, Q_S)
    new2_h = get_h(Q2, h2, Q_S)
    new_h = [0.5 * new1_h[i] + 0.5 * new2_h[i] for i in range(len(new1_h))]
    new_w = [0] * n_new_interval
    for n in range(n_new_interval):
        new_w[n] = Q_S[n + 1] - Q_S[n]
    new_p = [new_h[i] * new_w[i] for i in range(len(new_h))]
    return Q_S, n_new_interval, new_p, new_h


def smoothing(theta, n):
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
    theta_smooth = np.matmul(smoothing_matrix, theta)
    return theta_smooth


def norm_sub(f,SUM=1):
    n = len(f)
    f = np.array(f)
    while(True):
        index_nega = f<0
        f[index_nega] = 0
        f_sum = np.sum(f)
        x = f_sum - SUM
        index_posi = f>0
        positive_num = np.sum(index_posi)
        y = x / positive_num
        f[index_posi] -= y
        if(np.sum(f<0)==0):
            break
    return f


def Aggregation(F_hat, Map_equadisquantile, theta1, bound1, K):
    f=[]
    for i in range(K):
        lenth1 = bound1[i+1]-bound1[i]
        lenth2 = Map_equadisquantile[i+1]-Map_equadisquantile[i]
        tmpf = theta1[i] * (lenth2 / (lenth1 + lenth2)) + F_hat[i] * (lenth1 / (lenth1 + lenth2))
        f.append(tmpf)
    f = np.array(f)
    return f


def sw(ori_samples, l, h, eps, randomized_bins=1024, domain_bins=1024):
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

    max_iteration = 10000
    loglikelihood_threshold = 1e-3
    ns_hist, _ = np.histogram(noisy_samples, bins=randomized_bins, range=(-w / 2, 1 + w / 2))
    # return EM(n, ns_hist, transform, max_iteration, loglikelihood_threshold) * len(ori_samples)
    # return EMS(n, ns_hist, transform, max_iteration, loglikelihood_threshold) * len(ori_samples)
    return EMS(n, ns_hist, transform, max_iteration, loglikelihood_threshold)


def EMS(n, ns_hist, transform, max_iteration, loglikelihood_threshold):
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

# possess prior knowledge
def scheme0(samples, l, h, eps, randomized_bins=1024, domain_bins=1024):
    # user grouping
    K = randomized_bins
    shuffle(samples)
    sample1 = samples[:int(len(samples) * 0.5)]
    sample2 = samples[int(len(samples) * 0.5):]

    # obtain prior knowledge
    theta1 = count(sample1, K) / len(sample1)
    EquaDisQuantile = [i/K for i in range(K+1)]
    width1 = [1/K] * K
    height1 = [theta1[i]/width1[i] for i in range(K)]

    # collection
    Encoded_Data2 = FMap_0(sample2, height1, EquaDisQuantile, K)
    Encoded_theta2 = sw(Encoded_Data2, l, h, eps, K, K)
    Encoded_width2 = [1/K] * K
    Encoded_height2 = Encoded_theta2 / Encoded_width2
    X = [i / K for i in range(K)]
    Map_equadisquantile2 = FMap_0(EquaDisQuantile, height1, EquaDisQuantile, K)
    Cumulative_distribution2 = FMap_0(Map_equadisquantile2, Encoded_height2, EquaDisQuantile, K)

    theta2 = np.zeros(K)
    for i in range(K):
        theta2[i] = Cumulative_distribution2[i+1] - Cumulative_distribution2[i]

    return theta2

# two round
def scheme1(samples, l, h, eps, randomized_bins=1024, domain_bins=1024):
    # user grouping
    K = randomized_bins
    shuffle(samples)
    sample1 = samples[:int(len(samples) * 0.5)]
    sample2 = samples[int(len(samples) * 0.5):]

    # round 1
    theta1 = sw(sample1, l, h, eps, K, K)
    EquaDisQuantile = [i/K for i in range(K+1)]
    width1 = [1/K] * K
    height1 = [theta1[i]/width1[i] for i in range(K)]

    # round 2
    # s_height1 = smooth2(height1, K)
    # EquaDisQuantile_3K = [i / (3*K) for i in range(3 * K + 1)]
    Encoded_Data2 = FMap_0(sample2, height1, EquaDisQuantile, K)
    # Encoded_real = count(Encoded_Data2, K) / len(Encoded_Data2)
    Encoded_theta2 = sw(Encoded_Data2, l, h, eps, K, K)
    Encoded_width2 = [1/K] * K
    Encoded_height2 = Encoded_theta2 / Encoded_width2
    X = [i / K for i in range(K)]
    Map_equadisquantile2 = FMap_0(EquaDisQuantile, height1, EquaDisQuantile, K)     ##非均匀分位点
    Cumulative_distribution2 = FMap_0(Map_equadisquantile2, Encoded_height2, EquaDisQuantile, K)

    theta2 = np.zeros(K)
    for i in range(K):
        theta2[i] = Cumulative_distribution2[i+1] - Cumulative_distribution2[i]

    return theta2

# 
def scheme2(samples, l, h, eps, randomized_bins=1024, domain_bins=1024):
    # user grouping
    K = randomized_bins
    shuffle(samples)
    sample1 = samples[:int(len(samples) * 0.25)]
    sample2 = samples[int(len(samples) * 0.25):int(len(samples) * 0.5)]
    sample3 = samples[int(len(samples) * 0.5):]

    # round 1
    theta1 = sw(sample1, l, h, eps, K, K)
    EquaDisQuantile = [i/K for i in range(K+1)]
    width1 = [1/K] * K
    height1 = [theta1[i]/width1[i] for i in range(K)]

    # roud 2
    # s_height1 = smooth2(height1, K)
    # EquaDisQuantile_3K = [i / (3*K) for i in range(3 * K + 1)]
    Encoded_Data2 = FMap_0(sample2, height1, EquaDisQuantile, K)
    Encoded_theta2 = sw(Encoded_Data2, l, h, eps, K, K)
    Encoded_width2 = [1/K] * K
    Encoded_height2 = Encoded_theta2 / Encoded_width2
    Map_equadisquantile2 = FMap_0(EquaDisQuantile, height1, EquaDisQuantile, K)
    Cumulative_distribution2 = FMap_0(Map_equadisquantile2, Encoded_height2, EquaDisQuantile, K)

    theta2 = np.zeros(K)
    for i in range(K):
        theta2[i] = Cumulative_distribution2[i+1] - Cumulative_distribution2[i]
    EquaDisQuantile = [i/K for i in range(K+1)]
    width2 = [1/K] * K
    height2 = [theta2[i]/width2[i] for i in range(K)]

    # round 3
    # s_height2 = smooth2(height2, K)  # 得到3k个高度
    # EquaDisQuantile_3K = [i / (3*K) for i in range(3 * K + 1)]
    Encoded_Data3 = FMap_0(sample3, height2, EquaDisQuantile, K)
    Encoded_theta3 = sw(Encoded_Data3, l, h, eps, K, K)
    Encoded_width3 = [1/K] * K
    Encoded_height3 = Encoded_theta3 / Encoded_width3
    Map_equadisquantile3 = FMap_0(EquaDisQuantile, height2, EquaDisQuantile, K)     ##非均匀分位点
    Cumulative_distribution3 = FMap_0(Map_equadisquantile3, Encoded_height3, EquaDisQuantile, K)

    theta3 = np.zeros(K)
    for i in range(K):
        theta3[i] = Cumulative_distribution3[i+1] - Cumulative_distribution3[i]

    return theta3




if __name__ == "__main__":

    # income_numerical.npy, 524200, 2308374
    # Retirement_numerical.npy, 59690.74, 178012
    # taxi_pickup_time_numerical.npy, 86399, 2189968

    K = 256
    X = [i/K for i in range(K)]

    #for file in ['income_numerical.npy', 'Retirement_numerical.npy', 'taxi_pickup_time_numerical.npy','beta_numerical.npy']:
    for file in ['income_numerical.npy']:
        kl1_errors = []
        kl2_errors = []
        kl3_errors = []

        emd1_errors = []
        emd2_errors = []
        emd3_errors = []

        mean1_errors = []
        mean2_errors = []
        mean3_errors = []

        Quantiles1_errors = []
        Quantiles2_errors = []
        Quantiles3_errors = []

        range11_errors = []
        range12_errors = []
        range13_errors = []

        range41_errors = []
        range42_errors = []
        range43_errors = []

        variance1_errors = []
        variance2_errors = []
        variance3_errors = []

        # for eps in [0.1,0.15,0.2,0.25,0.3,0.35,0.4]:
        for eps in [0.1]:
            kl1_error = []
            kl2_error = []
            kl3_error = []

            emd1_error = []
            emd2_error = []
            emd3_error = []

            mean1_error = []
            mean2_error = []
            mean3_error = []

            Quantiles1_error = []
            Quantiles2_error = []
            Quantiles3_error = []

            range11_error = []
            range12_error = []
            range13_error = []

            range41_error = []
            range42_error = []
            range43_error = []

            variance1_error = []
            variance2_error = []
            variance3_error = []

            samples = np.load(file)
            samples = (samples - np.min(samples)) / (np.max(samples) - np.min(samples))  # 值映射到0-1之间

            for time in range(100):  #100

                # if time % 10 == 0:
                #    print(file, eps, time)
                print(file, eps, time)

                real = count(samples, K) / len(samples)

                l = np.min(samples)
                h = np.max(samples)

                theta = sw(samples, l, h, eps, K, K)

                F_hat1 = scheme1(samples, l, h, eps, K, K)

                # F_hat2 = scheme2(samples, l, h, eps, K, K)

                KL1 = EM.KL(real, F_hat1)
                kl1_error.append(KL1)
                # KL2 = EM.KL(real, F_hat2)
                # kl2_error.append(KL2)
                KL3 = EM.KL(real,theta)
                kl3_error.append(KL3)

                # emd
                EMD1 = EM.EMD_value(F_hat1, real, K)
                emd1_error.append(EMD1)
                # EMD2 = EM.EMD_value(F_hat2, real, K)
                # emd2_error.append(EMD2)
                EMD3 = EM.EMD_value(theta, real, K)
                emd3_error.append(EMD3)

                # mean
                MEAN1 = EM.Mean_value(F_hat1, real, K)
                mean1_error.append(MEAN1)
                # MEAN2 = EM.Mean_value(F_hat2, real, K)
                # mean2_error.append(MEAN2)
                MEAN3 = EM.Mean_value(theta, real, K)
                mean3_error.append(MEAN3)

                # Quantiles
                QUANTITLE1 = EM.Quantiles_value(F_hat1, real, K)
                Quantiles1_error.append(QUANTITLE1)
                # QUANTITLE2 = EM.Quantiles_value(F_hat2, real, K)
                # Quantiles2_error.append(QUANTITLE2)
                QUANTITLE3 = EM.Quantiles_value(theta, real, K)
                Quantiles3_error.append(QUANTITLE3)

                # Range_Query
                RANGEQUERY_11 = EM.Range_Query(F_hat1, real, 0.1, K)
                RANGEQUERY_41 = EM.Range_Query(F_hat1, real, 0.4, K)
                range11_error.append(RANGEQUERY_11)
                range41_error.append(RANGEQUERY_41)
                # RANGEQUERY_12 = EM.Range_Query(F_hat2, real, 0.1, K)
                # RANGEQUERY_42 = EM.Range_Query(F_hat2, real, 0.4, K)
                # range12_error.append(RANGEQUERY_12)
                # range42_error.append(RANGEQUERY_42)
                RANGEQUERY_13 = EM.Range_Query(theta, real, 0.1, K)
                RANGEQUERY_43 = EM.Range_Query(theta, real, 0.4, K)
                range13_error.append(RANGEQUERY_13)
                range43_error.append(RANGEQUERY_43)

                # Variance
                VARIANCE1 = EM.Variance_value(F_hat1, real, K)
                variance1_error.append(VARIANCE1)
                # VARIANCE2 = EM.Variance_value(F_hat2, real, K)
                # variance2_error.append(VARIANCE2)
                VARIANCE3 = EM.Variance_value(theta, real, K)
                variance3_error.append(VARIANCE3)

            kl1_errors.append(np.mean(kl1_error))
            # kl2_errors.append(np.mean(kl2_error))
            kl3_errors.append(np.mean(kl3_error))

            emd1_errors.append(np.mean(emd1_error))
            # emd2_errors.append(np.mean(emd2_error))
            emd3_errors.append(np.mean(emd3_error))

            mean1_errors.append(np.mean(mean1_error))
            # mean2_errors.append(np.mean(mean2_error))
            mean3_errors.append(np.mean(mean3_error))

            Quantiles1_errors.append(np.mean(Quantiles1_error))
            # Quantiles2_errors.append(np.mean(Quantiles2_error))
            Quantiles3_errors.append(np.mean(Quantiles3_error))

            range11_errors.append(np.mean(range11_error))
            # range12_errors.append(np.mean(range12_error))
            range13_errors.append(np.mean(range13_error))

            range41_errors.append(np.mean(range41_error))
            # range42_errors.append(np.mean(range42_error))
            range43_errors.append(np.mean(range43_error))

            variance1_errors.append(np.mean(variance1_error))
            # variance2_errors.append(np.mean(variance2_error))
            variance3_errors.append(np.mean(variance3_error))

        print("-------------------------------------------")
        print("dataset:", file)
        print("KL1:", kl1_errors)
        # print("KL2:", kl2_errors)
        print("KL3:", kl3_errors)
        print("EMD1:", emd1_errors)
        # print("EMD2:", emd2_errors)
        print("EMD3:", emd3_errors)
        print("MEAN1:", mean1_errors)
        # print("MEAN2:", mean2_errors)
        print("MEAN3:", mean3_errors)
        print("Quantiles1:", Quantiles1_errors)
        # print("Quantiles2:", Quantiles2_errors)
        print("Quantiles3:", Quantiles3_errors)
        print("range11:", range11_errors)
        # print("range12:", range12_errors)
        print("range13:", range13_errors)
        print("range41:", range41_errors)
        # print("range42:", range42_errors)
        print("range43:", range43_errors)
        print("variance1:", variance1_errors)
        # print("variance2:", variance2_errors)
        print("variance3:", variance3_errors)
        print("-------------------------------------------")

        f_out.write(str(file) + '\n')
        f_out.write(str('kl') + '\n')
        for i in range(len(kl1_errors)):
            f_out.write(str(kl1_errors[i]) + '\t')
        f_out.write('\n')
        for i in range(len(kl2_errors)):
            f_out.write(str(kl2_errors[i]) + '\t')
        f_out.write('\n')
        for i in range(len(kl3_errors)):
            f_out.write(str(kl3_errors[i]) + '\t')
        f_out.write('\n')

        f_out.write(str('emd') + '\n')
        for i in range(len(emd1_errors)):
            f_out.write(str(emd1_errors[i]) + '\t')
        f_out.write('\n')
        for i in range(len(emd2_errors)):
            f_out.write(str(emd2_errors[i]) + '\t')
        f_out.write('\n')
        for i in range(len(emd3_errors)):
            f_out.write(str(emd3_errors[i]) + '\t')
        f_out.write('\n')

        f_out.write(str('mean') + '\n')
        for i in range(len(mean1_errors)):
            f_out.write(str(mean1_errors[i]) + '\t')
        f_out.write('\n')
        for i in range(len(mean2_errors)):
            f_out.write(str(mean2_errors[i]) + '\t')
        f_out.write('\n')
        for i in range(len(mean3_errors)):
            f_out.write(str(mean3_errors[i]) + '\t')
        f_out.write('\n')

        f_out.write(str('Quantiles') + '\n')
        for i in range(len(Quantiles1_errors)):
            f_out.write(str(Quantiles1_errors[i]) + '\t')
        f_out.write('\n')
        for i in range(len(Quantiles2_errors)):
            f_out.write(str(Quantiles2_errors[i]) + '\t')
        f_out.write('\n')
        for i in range(len(Quantiles3_errors)):
            f_out.write(str(Quantiles3_errors[i]) + '\t')
        f_out.write('\n')

        f_out.write(str('range1') + '\n')
        for i in range(len(range11_errors)):
            f_out.write(str(range11_errors[i]) + '\t')
        f_out.write('\n')
        for i in range(len(range12_errors)):
            f_out.write(str(range12_errors[i]) + '\t')
        f_out.write('\n')
        for i in range(len(range13_errors)):
            f_out.write(str(range13_errors[i]) + '\t')
        f_out.write('\n')

        f_out.write(str('range4') + '\n')
        for i in range(len(range41_errors)):
            f_out.write(str(range41_errors[i]) + '\t')
        f_out.write('\n')
        for i in range(len(range42_errors)):
            f_out.write(str(range42_errors[i]) + '\t')
        f_out.write('\n')
        for i in range(len(range43_errors)):
            f_out.write(str(range43_errors[i]) + '\t')
        f_out.write('\n')

        f_out.write(str('variance') + '\n')
        for i in range(len(variance1_errors)):
            f_out.write(str(variance1_errors[i]) + '\t')
        f_out.write('\n')
        for i in range(len(variance2_errors)):
            f_out.write(str(variance2_errors[i]) + '\t')
        f_out.write('\n')
        for i in range(len(variance3_errors)):
            f_out.write(str(variance3_errors[i]) + '\t')
        f_out.write('\n')

    f_out.close()










