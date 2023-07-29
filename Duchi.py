import numpy as np


class Duchi(object):
    def __init__(self):
        self.eps = 0
        self.domain_size = 0
        self.output = 0

    def init_method(self, eps, domain_size):
        self.eps = eps
        self.domain_size = domain_size
        self.output = (np.exp(eps) + 1) / (np.exp(eps) - 1)
        # print("eps=", self.eps, "output=", self.output)

    def randomize(self, samples):
        sample_size = len(samples)
        proj_samples = samples * 2 / self.domain_size - 1
        probs = (np.exp(self.eps) - 1) / (2 * np.exp(self.eps) + 2) * proj_samples + 1 / 2
        ns = np.zeros(sample_size)
        tmps = np.random.binomial(1, probs)
        # print(tmps)
        ns[tmps == 1] = self.output
        ns[tmps == 0] = - self.output
        # for i, prob in enumerate(probs):
        #     tmp = np.random.binomial(1, prob, 1)[0]
        #     if tmp == 1:
        #         ns[i] = self.output
        #     else:
        #         ns[i] = - self.output
        return ns

    def estimate_mean(self, samples):
        ns = self.randomize(samples)
        # print("proj mean:", np.mean(ns))
        mean = (np.mean(ns) + 1) / 2 * self.domain_size
        # mean = np.mean(ns)
        return mean

    def estimate_var(self, samples):
        first_half = samples[:int(len(samples)/2)]
        second_half = samples[int(len(samples) / 2):]
        ns_first = self.randomize(first_half)
        mean = (np.mean(ns_first) + 1) / 2 * self.domain_size
        ns_second = self.randomize(np.square(second_half - mean)/self.domain_size)
        # print(len(ns_second))
        var = (np.mean(ns_second) + 1) / 2 * self.domain_size * self.domain_size

        return var



if __name__ == "__main__":

    # income_numerical.npy, 524200, 2308374
    # Retirement_numerical.npy, 59690.74, 178012
    # taxi_pickup_time_numerical.npy, 86399, 2189968

    # for file in ['income_numerical.npy', 'Retirement_numerical.npy', 'taxi_pickup_time_numerical.npy']:
    for file in ['beta_numerical.npy']:

        # if file == 'income_numerical.npy':
        #     domain_size = 524200
        # elif file == 'Retirement_numerical.npy':
        #     domain_size = 59690.74
        # else:
        #     domain_size = 86399

        mean_errors = []
        var_errors = []

        for eps in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:

            mean_error = []
            var_error = []

            for time in range(100):

                if time % 10 == 0:
                    print(eps, time)

                samples = np.load(file)

                samples = (samples - np.min(samples)) / np.max(samples)

                pm = Duchi()
                pm.init_method(eps, 1)

                mean = pm.estimate_mean(samples)
                real_mean = np.mean(samples)
                mean_error.append(abs(mean - real_mean))

                var = pm.estimate_var(samples)
                real_var = np.var(samples)
                var_error.append(abs((var - real_var)))

            print("--------------------------------------")
            print("dataset and epsilon:", file, eps)
            # print("mean:", mean_error)
            # print("var:", var_error)
            print("Duchi estimate mean:", np.mean(np.array(mean_error)))
            print("Duchi estimate var:", np.mean(np.array(var_error)))
            print("--------------------------------------")

            mean_errors.append(np.mean(np.array(mean_error)))
            var_errors.append(np.mean(np.array(var_error)))

        print('mean', mean_errors)
        print('var', var_errors)