import csv
import math
import numpy as np
import scipy

from scipy import special
from random import shuffle
from numpy import linalg as LA


def Duchi(samples,eps):
    ne = exp(eps)
    a = (samples*(ne-1)+ne+1)/2.0
    p = a / (ne + 1)
    ce = (ne + 1) / (ne - 1)
    randoms = np.random.uniform(0,1,len(samples))
    noisy_samples = np.zeros_like(samples)
    num = len(samples)
    for i in range(num):
        if randoms[i] <= p:
            bit = 1
            noisy_samples[i] = (bit * 2 - 1) * ce
        else:
            bit = 0
            noisy_samples[i] = (bit * 2 - 1) * ce
    return noisy_samples


def PM(samples, eps):
    z = exp(eps / 2)
    P1 = (samples + 1) / (2 + 2 * z)
	P2 = z / (z + 1)
	P3 = (1 - samples) / (2 + 2 * z)
	C = (z + 1) / (z - 1)
	g1 = (C + 1)*samples / 2 - (C - 1) / 2
	g2 = (C + 1)*samples / 2 + (C - 1) / 2

    randoms = np.random.uniform(0,1,len(samples))
    noisy_samples = np.zeros_like(samples)

    index = randoms < P1
    noisy_samples[index] = -C + randoms[index] * (g1 - (-C))
    index = randoms < P1 + P2
    noisy_samples[index] = (g2 - g1)*randoms[index] + g1
    index = randoms >= P1 + P2
    noisy_samples[index] = (C - g2)*randoms[index] + g2
	return noisy_samples
