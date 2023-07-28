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

f_out = open('log_sw.txt', 'w')


# n个桶转化为等距K个桶
def buck_uniform(n, K, d, old_Q, old_p, old_h):   ##没用
    new_p = np.zeros(K)
    w = d / K
    i = 0  # 旧桶编号
    sweep = old_Q[i + 1]  # 旧桶的右边界
    for index in range(K):  # 每个新桶(等距)
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
    for v in data:  # 每个数据
        index = int(v * num / d)
        if index < num:
            c[index] += 1
        else:
            c[num - 1] += 1
    return c


#计算信噪比倒数之和
def sum_SNR(f,f_hat,num):   ##没用
    print('sum_snr')
    sum=0
    for i in range(num):
        if f[i]!=0:
            sum+=(np.abs(f[i]-f_hat[i])/f[i])
    return sum


#计算KL散度
def KL_value(f, f_hat, num):
    sums = 0
    for i in range(num):
        if f[i] != 0 and f_hat[i] != 0:
            sums += f[i] * (math.log(f[i] / f_hat[i]))
    return sums


def Find_Quantile(K, data, Quantile):     #找data位于哪个桶
    result = 0
    # 每个桶 桶号k∈0~K-1
    right = K
    left = 0
    if data >= Quantile[-1]:    #将1放入最后一个桶里
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


# 完整区间的累积概率
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


def ReFMap(h, Quantile, K):   ##没用
    datas = [0] * (K + 1)
    for i in range(K):
        datas[i + 1] = (i + 1) / K  # 待逆映射还原的分位点
    ReF = [0] * (K + 1)
    CumulativeFrequency = [0] * (K + 1)  # 每个分位点对应一个概率累计值
    width = [0] * K
    for i in range(K):
        width[i] = Quantile[i + 1] - Quantile[i]
    for i in range(K):
        CumulativeFrequency[i + 1] = h[i] * width[i] + CumulativeFrequency[i]  # 第一个为0,计算后面k个端点的概率累加值

    for i in range(K - 1):
        index = Find_Quantile(K, datas[i + 1], CumulativeFrequency)  # 先找到位于第几个区间内
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


#更新桶高到2K-1个区间
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
    # 合并Quantile
    # Q_S = np.concatenate((Q2, Q1))
    # Q_S = np.array(list(set(Q_S)))
    # Q_S = sorted(Q_S)

    Q_S = copy.deepcopy(Q1)
    Q_S.extend(Q2)
    Q_S = list(set(Q_S))
    Q_S = sorted(Q_S)

    # print('Q_S', Q_S)
    # 密度平均
    # 新的区间个数
    n_new_interval = len(Q_S) - 1  # len个分位点（含0，100）
    # print(n_new_interval)
    # 得到每个区间的密度1
    new1_h = get_h(Q1, h1, Q_S)
    # 得到每个区间的密度2
    new2_h = get_h(Q2, h2, Q_S)
    # 取平均
    new_h = [0.3 * new1_h[i] + 0.7 * new2_h[i] for i in range(len(new1_h))]
    # print('取平均new_h', new_h)

    new_w = [0] * n_new_interval
    for n in range(n_new_interval):
        new_w[n] = Q_S[n + 1] - Q_S[n]
    # print("new_w:", new_w, sum(new_w))
    new_p = [new_h[i] * new_w[i] for i in range(len(new_h))]
    # print('new_p', new_p, sum(new_p))
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


def norm_sub(f,SUM=1):    #非负且和为1
    n = len(f)
    f = np.array(f)
    while(True):
        index_nega = f<0
        f[index_nega] = 0  #负值置零
        f_sum = np.sum(f)  #总频率
        x = f_sum - SUM  #总差值
        index_posi = f>0
        positive_num = np.sum(index_posi)
        y = x / positive_num  # 平均差值
        f[index_posi] -= y
        if(np.sum(f<0)==0):  #全正退出
            break
    #print("norm_su后频率",f)
    #print("频率之和",sum(f))
    return f


def Aggregation(F_hat, Map_equadisquantile, theta1, bound1, K):
    f=[]
    for i in range(K):
        lenth1 = bound1[i+1]-bound1[i]
        lenth2 = Map_equadisquantile[i+1]-Map_equadisquantile[i]
        tmpf = theta1[i] * (lenth2 / (lenth1 + lenth2)) + F_hat[i] * (lenth1 / (lenth1 + lenth2))
        # w1 = 0.3
        # tmpf = theta1[i] * w1 + F_hat[i] * (1-w1)
        f.append(tmpf)
    f = np.array(f)
    return f


def sw(ori_samples, l, h, eps, randomized_bins=1024, domain_bins=1024):
    ee = np.exp(eps)
    w = ((eps * ee) - ee + 1) / (2 * ee * (ee - 1 - eps)) * 2  ##2b
    p = ee / (w * ee + 1)
    q = 1 / (w * ee + 1)

    samples = (ori_samples - l) / (h - l)  ##映射到0-1之间
    randoms = np.random.uniform(0, 1, len(samples))  ##randoms是一组随机概率值，模拟每一次扰动过程

    noisy_samples = np.zeros_like(samples)

    # report                                              ##将随机值通过每个v的累积分布函数映射回[-b~1+b]区间得到扰动后的报告值
    index = randoms <= (q * samples)
    noisy_samples[index] = randoms[index] / q - w / 2
    index = randoms > (q * samples)
    noisy_samples[index] = (randoms[index] - q * samples[index]) / p + samples[index] - w / 2
    index = randoms > q * samples + p * w
    noisy_samples[index] = (randoms[index] - q * samples[index] - p * w) / q + samples[index] + w / 2

    # report matrix
    m = randomized_bins  ## m=~d,扰动值的分桶数
    n = domain_bins  ## n=d,估计分布的分桶数
    m_cell = (1 + w) / m  ## 扰动桶宽
    n_cell = 1 / n  ## 估计桶宽

    transform = np.ones((m, n)) * q * m_cell  ##概率转移矩阵M
    for i in range(n):
        left_most_v = (i * n_cell)  ##当前真实桶的左值
        right_most_v = ((i + 1) * n_cell)  ##当前真实桶的右值

        ll_bound = int(left_most_v / m_cell)  ##扰动分布中，真实左值所在桶号
        lr_bound = int((left_most_v + w) / m_cell)  ##+2b
        rl_bound = int(right_most_v / m_cell)
        rr_bound = int((right_most_v + w) / m_cell)  ##+2b

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
    ns_hist, _ = np.histogram(noisy_samples, bins=randomized_bins, range=(-w / 2, 1 + w / 2))  ##扰动值频数直方图
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


def scheme(samples, l, h, eps, randomized_bins=1024, domain_bins=1024):
    # 随机分组
    shuffle(samples)
    # sample1 = samples[:int(len(samples) * 0.1)]
    # sample2 = samples[int(len(samples) * 0.1):int(len(samples) * 0.3)]
    # sample3 = samples[int(len(samples) * 0.3):]
    sample1 = samples
    sample2 = samples
    sample3 = samples

    # 估计频率
    # 第一轮统计
    theta1 = sw(sample1, l, h, eps, K, K)

    # 输出第一轮统计结果
    for i in range(len(theta1)):
        f_out.write(str(theta1[i]) + ':')
    f_out.write('\n')

    # 分位点坐标
    EquaDisQuantile = [i/K for i in range(K+1)]
    # 桶宽
    width1 = [1/K] * K
    # 桶高
    height1 = [theta1[i]/width1[i] for i in range(K)]

    # 第二轮统计
    # 平滑
    s_height1 = smooth2(height1, K)  # 得到3k个高度
    EquaDisQuantile_3K = [i / (3*K) for i in range(3 * K + 1)]

    # 数据映射
    Encoded_Data2 = FMap_0(sample2, s_height1, EquaDisQuantile_3K, 3*K)

    # 扰动
    Encoded_theta2 = sw(Encoded_Data2, l, h, eps, K, K)
    # 映射的每个桶得高、宽、面积
    Encoded_width2 = [1/K] * K
    Encoded_height2 = Encoded_theta2 / Encoded_width2

    # 等距的分位点映射后的分位点
    Map_equadisquantile2 = FMap_0(EquaDisQuantile, s_height1, EquaDisQuantile_3K, 3*K)     ##非均匀分位点
    # 映射后的分位点处的累计分布值
    Cumulative_distribution2 = FMap_0(Map_equadisquantile2, Encoded_height2, EquaDisQuantile, K)

    theta2 = np.zeros(K)
    for i in range(K):
        theta2[i] = Cumulative_distribution2[i+1] - Cumulative_distribution2[i]
    for i in range(len(theta2)):
        f_out.write(str(theta2[i]) + ':')
    f_out.write('\n')

    # 分位点坐标
    EquaDisQuantile = [i/K for i in range(K+1)]
    # 桶宽
    width2 = [1/K] * K
    # 桶高
    height2 = [theta2[i]/width2[i] for i in range(K)]

    # 第三轮统计
    # 平滑
    s_height2 = smooth2(height2, K)  # 得到3k个高度
    EquaDisQuantile_3K = [i / (3*K) for i in range(3 * K + 1)]

    # 数据映射
    Encoded_Data3 = FMap_0(sample3, s_height2, EquaDisQuantile_3K, 3*K)

    # 扰动
    Encoded_theta3 = sw(Encoded_Data3, l, h, eps, K, K)
    # 映射的每个桶得高、宽、面积
    Encoded_width3 = [1/K] * K
    Encoded_height3 = Encoded_theta3 / Encoded_width3

    # 等距的分位点映射后的分位点
    Map_equadisquantile3 = FMap_0(EquaDisQuantile, s_height2, EquaDisQuantile_3K, 3*K)     ##非均匀分位点
    # 映射后的分位点处的累计分布值
    Cumulative_distribution3 = FMap_0(Map_equadisquantile3, Encoded_height3, EquaDisQuantile, K)

    theta3 = np.zeros(K)
    for i in range(K):
        theta3[i] = Cumulative_distribution3[i+1] - Cumulative_distribution3[i]
    for i in range(len(theta3)):
        f_out.write(str(theta3[i]) + ':')
    f_out.write('\n')

    X = [i / K for i in range(K)]
    plt.plot(X, theta1, color='green', linewidth=0.5, linestyle='--', label='round1')
    plt.plot(X, theta2, color='cyan', linewidth=0.5, linestyle=':', label='round2')
    plt.plot(X, theta3, color='violet', linewidth=0.5, linestyle='dashed', label='round3')

    return theta3


# 这里显示调用了plt.figure返回一个fig对象, 为了能够后面保存这个fig对象
# 并且设置了fig的大小和每英寸的分辨率
# 注意: resolution = figsize*dpi（因此这里的savefig的分辨率为1200X900）
fig = plt.figure(figsize=(4,3),dpi=300)
ax = plt.gca() # gca： get current axis


if __name__ == "__main__":

    # income_numerical.npy, 524200, 2308374
    # Retirement_numerical.npy, 59690.74, 178012
    # taxi_pickup_time_numerical.npy, 86399, 2189968

    K = 256
    X = [i/K for i in range(K)]

    #for file in ['income_numerical.npy', 'Retirement_numerical.npy', 'taxi_pickup_time_numerical.npy','beta_numerical.npy']:
    for file in ['beta_numerical.npy']:
        kl_errors = []
        kl1_errors = []
        emd_errors = []

        ks_errors = []

        mean_errors = []

        Quantiles_errors = []

        range1_errors = []

        range4_errors = []

        variance_errors = []

        for eps in [0.1]:
        #for eps in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75]:
        #for eps in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:

            kl_error = []
            kl1_error = []
            emd_error = []

            ks_error = []

            mean_error = []

            Quantiles_error = []

            range1_error = []

            range4_error = []

            variance_error = []

            samples = np.load(file)

            samples = (samples - np.min(samples)) / (np.max(samples) - np.min(samples))  # 值映射到0-1之间

            for time in range(100):  #100

                if time % 10 == 0:
                    print(file, eps, time)

                real = count(samples, K) / len(samples)
                for i in range(len(real)):
                    f_out.write(str(real[i]) + ':')
                f_out.write('\n')

                # 作图
                plt.plot(X, real, color='red', linewidth=0.5, linestyle='-.', label='real')

                l = np.min(samples)
                h = np.max(samples)

                theta = sw(samples, l, h, eps, K, K)
                for i in range(len(theta)):
                    f_out.write(str(theta[i]) + ':')
                f_out. write('\n')

                # 设置线条参数：设置颜色，线宽，线形，标记，标记大小，图例标签等等
                plt.plot(X, theta, color='blue', linewidth=0.5, linestyle='-', label='sw')

                F_hat = scheme(samples, l, h, eps, K, K)

                f_out.close()
                
                # 设置图例（legend）
                # plt.legend(loc='auto', frameon=False) # frameon is flag to draw a frame around the legend
                # Advanced legend
                plt.legend(loc = 'best', prop = {'size':8})

                #KL
                KL1 = EM.KL(real,theta)
                #print("KL散度",KL)
                kl1_error.append(KL1)

                KL = EM.KL(real,F_hat)
                #print("KL散度",KL)
                kl_error.append(KL)

                # emd
                EMD = EM.EMD_value(F_hat, real, K)
                emd_error.append(EMD)

                # ks
                KS = EM.KS_value(F_hat, real, K)
                ks_error.append(KS)

                # mean
                MEAN = EM.Mean_value(F_hat, real, K)
                mean_error.append(MEAN)

                # Quantiles
                #QUANTITLE = EM.Quantiles_value(F_hat, real, K)
                #Quantiles_error.append(QUANTITLE)

                # Range_Query
                RANGEQUERY_1 = EM.Range_Query(F_hat, real, 0.1, K)
                RANGEQUERY_4 = EM.Range_Query(F_hat, real, 0.4, K)
                range1_error.append(RANGEQUERY_1)
                range4_error.append(RANGEQUERY_4)

                # Variance
                VARIANCE = EM.Variance_value(F_hat, real, K)   ##############################有误Variance_value() missing 1 required positional argument: 'K'
                variance_error.append(VARIANCE)

            kl_errors.append(np.mean(kl_error))
            kl1_errors.append(np.mean(kl1_error))
            emd_errors.append(np.mean(emd_error))
            ks_errors.append(np.mean(ks_error))
            mean_errors.append(np.mean(mean_error))
            #Quantiles_errors.append(np.mean(Quantiles_error))
            range1_errors.append(np.mean(range1_error))
            range4_errors.append(np.mean(range4_error))
            variance_errors.append(np.mean(variance_error))



        print("-------------------------------------------")
        print("dataset:", file)
        print("our_solution estimate KL:", kl_errors)
        print("our_solution estimate KL:", kl1_errors)
        print("our_solution estimate EMD:", emd_errors)
        print("dataset and epsilon:", file, eps)
        print("our_solution estimate KS:", ks_errors)
        print("dataset and epsilon:", file, eps)
        print("our_solution estimate MEAN:", mean_errors)
        print("dataset and epsilon:", file, eps)
        print("our_solution estimate RANGE_1:", range1_errors)
        print("dataset and epsilon:", file, eps)
        print("our_solution estimate RANGE_4:", range4_errors)
        print("dataset and epsilon:", file, eps)
        #print("our_solution estimate QUANTILES:", Quantiles_errors)
        print("dataset and epsilon:", file, eps)
        print("our_solution estimate VARIANCE:", variance_errors)
        print("-------------------------------------------")












