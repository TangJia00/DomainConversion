import numpy as np
import numpy as np
from fo.fo_factory import FOFactory
from fo.lh import LH
from queue import Queue
from scipy.linalg import hadamard
import matplotlib.pyplot as plt


class HaarHRR(object):
    def __init__(self):
        self.D = 0
        self.sample_size = 0
        self.branch = 0
        self.l = 0
        # use array to store a B-adic tree
        self.tree = np.zeros(0)
        self.haar_matrix = np.zeros(0)
        self.estimates = np.zeros(0)
        self.eps = 0
        self.p = 0

    def init_haarhrr(self, domain_size):
        self.D = domain_size
        self.l = int(np.log2(self.D))
        self.tree = np.zeros(self.D)
        self.haar_matrix = self.haarMatrix(self.D).T
        # self.p = np.exp(self.eps) / (1 + np.exp(self.eps))
        self.estimates = np.zeros(self.D)

    def haarMatrix(self, n, normalized=False):
        # Allow only size n of power 2
        n = 2 ** np.ceil(np.log2(n))
        if n > 2:
            h = self.haarMatrix(n / 2, normalized)
        else:
            return np.array([[1, 1], [1, -1]])

        # calculate upper haar part
        h_n = np.kron(h, [1, 1])
        # calculate lower haar part
        if normalized:
            h_i = np.sqrt(n / 2) * np.kron(np.eye(len(h)), [1, -1])
        else:
            h_i = np.kron(np.eye(len(h)), [1, -1])
        # combine parts
        h = np.vstack((h_n, h_i))
        return h

    def HRR(self, samples, haar_layer_signs, d):
        hadamard_matrix = hadamard(d)
        div = self.D / d
        n = len(samples)
        # print("d:", d, )
        # print("subset sample size:", len(samples))
        sample_hist, _ = np.histogram(samples, self.D, range=(0, self.D))
        check_sample_hist, _ = np.histogram(samples, d, range=(0, self.D))
        # check_sample_hist /=  len(samples)
        # sample_hist = sample_hist / len(samples)
        mid_results = np.zeros((d, self.D))
        for idx, i_freq in enumerate(sample_hist):
            sampled_dist = np.random.multinomial(i_freq, np.ones(d)/d, size=1)[0]
            non_flips = np.random.binomial(sampled_dist, self.p)
            flips = sampled_dist - non_flips
            mid_results[:, idx] = np.multiply((non_flips - flips ),
                                              hadamard_matrix[int(idx / div), :])
            # print(mid_results[:, idx])

        '''
        sqrt(d) is cancelled in denominator because 
            1. hadamard_matrix has a sqrt(d) in denominator as well
            2. uniformly random sample bit in d bits to flip, so we need to amplify the result by d
         '''
        est_freq = np.matmul(hadamard_matrix, np.sum(mid_results, axis=1)) / (2 * self.p - 1)
        # print(est_freq, sum(est_freq))


        estimates = np.matmul(mid_results, haar_layer_signs)
        # print(estimates.shape)
        estimates = np.matmul(hadamard_matrix, estimates)
        # print(estimates.shape)
        estimates = estimates / (2 * self.p - 1)
        # print(estimates)
        # exit()

        return estimates


    def Haar(self, real_samples):
        self.sample_size = len(real_samples)
        layer_sample_sizes = np.random.multinomial(self.sample_size,
                                                   np.ones(self.l ) / (self.l ),
                                                   size=1)[0]

        sample_pointer = 0
        self.tree[0] = len(real_samples) / np.sqrt(self.D)
        for idx, layer_count in enumerate(layer_sample_sizes):
            layer_nodes = np.exp2(idx)
            subset_samples = np.copy(real_samples[sample_pointer: sample_pointer + layer_count])
            sample_pointer += layer_count

            haar_start = int(np.around(np.exp2(idx) - 1)) + 1
            haar_end = int(np.around(np.exp2(idx + 1) - 1 )) + 1
            haar_layer_signs = np.sum(self.haar_matrix[:, haar_start:haar_end], axis=1).flatten()
            # print("layer signs", haar_layer_signs)
            layer_coefficients = self.HRR(subset_samples, haar_layer_signs, d=haar_end - haar_start)
            # print("h(v):", (self.l - idx), np.exp2((self.l - idx)/2))
            self.tree[haar_start:haar_end] = layer_coefficients / np.exp2((self.l - idx)/2) * len(real_samples) / len(subset_samples)
            # print(self.tree[haar_start:haar_end])

            # real_dist, _ = np.histogram(subset_samples, layer_nodes, range=(0, self.D))
        return

    def estimate(self, real_samples, epsilon):
        self.eps = epsilon
        self.p = np.exp(self.eps) / (1 + np.exp(self.eps))
        self.Haar(real_samples)
        # self.estimates = np.matmul(1 / np.sqrt(self.D) *self.haar_matrix, self.tree)
        # self.estimates = self.estimates / len(real_samples)
        self.estimates = np.matmul(1 / np.sqrt(self.D)* self.haarMatrix(self.D, True).T, self.tree)
        self.estimates = self.estimates / len(real_samples)
        # print('sum:', np.sum(self.estimates))
        # print(self.tree)
        # print(self.haarMatrix(256, True))
        # exit()
        return self.estimates