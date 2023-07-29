import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
import argparse

def gen_beta_data(sample_size, alpha=5, beta=2):
    samples = rd.beta(alpha, beta, sample_size)
    # selections = []
    # for i in range(len(samples)):
    #     if 0.2 < samples[i]:
    #         selections.append(samples[i])
    plt.hist(samples, 512, range=(0,1))
    plt.show()
    np.save('beta_numerical', selections)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate metric for exp results')
    parser.add_argument('--alpha', type=float, default=5,
                        help='parameter for beta distribution')
    parser.add_argument('--beta', type=float, default=2,
                        help='parameter for beta distribution')
    parser.add_argument('--sample_size', type=int, default=1000000,
                        help='tasks')
    args = parser.parse_args()

    gen_beta_data(args.sample_size, args.alpha, args.beta)