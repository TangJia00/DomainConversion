import numpy as np

class PM(object):
    def __init__(self):
        self.eps = 0
        self.c = 0
        self.domain_size = 0
        self.p = 0

    def init_method(self, eps, domain_size):
        self.eps = eps
        self.c = (np.exp(eps/2) + 1)/(np.exp(eps/2) - 1)
        self.p = (np.exp(eps) - np.exp(eps/2)) / (2 * np.exp(eps/2) + 2)
        self.domain_size = domain_size
        # print("eps=",self.eps, "C=", self.c)

    def compute_l_r(self, t):
        l = (self.c + 1) / 2 * t - (self.c - 1) / 2
        r = l + self.c - 1
        return l, r

    # def randomize(self, samples):
    #     sample_size = len(samples)
    #     proj_samples = samples * 2 / self.domain_size - 1
    #     ns = np.zeros(sample_size)
    #     x = np.random.uniform(0, 1, sample_size)
    #     bar = np.exp(self.eps/2) / (np.exp(self.eps/2) + 1)
    #     print("bar:", bar)
    #     for i, sample in enumerate(proj_samples):
    #         l, r = self.compute_l_r(sample)
    #         # print(sample, l, r)
    #         if x[i] < bar:
    #             ns[i] = np.random.uniform(l, r)
    #         else:
    #             tmp = np.random.uniform(-self.c / 2 - 1 / 2, self.c / 2 + 1 / 2)
    #             # print("tmp", tmp, (self.c + 1) / 2 * (sample + 1))
    #             if tmp + self.c / 2 + 1 / 2 < (self.c + 1) / 2 * (sample + 1):
    #                 ns[i] = tmp - (self.c - 1)/2
    #             else:
    #                 ns[i] = tmp + (self.c - 1)/2
    #     return ns

    def randomize(self, samples):
        sample_size = len(samples)
        proj_samples = samples * 2 / self.domain_size - 1
        ns = np.zeros(sample_size)
        y = np.random.uniform(0, 1, sample_size)
        bar = np.exp(self.eps/2) / (np.exp(self.eps/2) + 1)
        q = self.p / np.exp(self.eps)
        # print("bar:", bar)
        for i, sample in enumerate(proj_samples):
            l, r = self.compute_l_r(sample)
            if y[i] < (l + self.c) * q:
                ns[i] = (y[i] / q - self.c)
            elif y[i] < (self.c - 1) * self.p + (l + self.c) * q:
                ns[i] = ((y[i] - (l + self.c) * q) / self.p + l)
            else:
                ns[i] = ((y[i] - (l + self.c) * q - (self.c - 1) * self.p) / q + r)
        return ns

    def estimate_mean(self, samples):
        ns = self.randomize(samples)
        # print("proj mean:", np.mean(ns))
        mean = (np.mean(ns) + 1) / 2 * self.domain_size
        # mean = np.mean(ns)
        return mean

    def estimate_var(self, samples):
        first_half = samples[:int(len(samples) / 2)]
        second_half = samples[int(len(samples) / 2):]
        ns_first = self.randomize(first_half)
        mean = (np.mean(ns_first) + 1) / 2 * self.domain_size
        ns_second = self.randomize(np.square(second_half - mean) / self.domain_size)
        # print(len(ns_second))
        var = (np.mean(ns_second) + 1) / 2 * self.domain_size * self.domain_size

        return var


if __name__ == "__main__":

    # income_numerical.npy, 524200, 2308374
    # Retirement_numerical.npy, 59690.74, 178012
    # taxi_pickup_time_numerical.npy, 86399, 2189968

    for file in ['beta_numerical.npy', 'income_numerical.npy', 'Retirement_numerical.npy',
                 'taxi_pickup_time_numerical.npy']:

        mean_errors = []
        var_errors = []

        for eps in [0.1,0.15,0.2,0.25,0.3,0.35,0.4]:

            mean_error = []
            var_error = []

            for time in range(50):

                #if time % 10 == 0:
                print(file, eps, time)

                samples = np.load(file)

                samples = (samples - np.min(samples)) / np.max(samples)

                pm = PM()
                pm.init_method(eps, 1)

                mean = pm.estimate_mean(samples)
                real_mean = np.mean(samples)
                mean_error.append(abs(mean - real_mean))

                var = pm.estimate_var(samples)
                real_var = np.var(samples)
                var_error.append(abs((var - real_var)))

            # print("--------------------------------------")
            # print("dataset and epsilon:", file, eps)
            # print("mean:", mean_error)
            # print("var:", var_error)
            # print("PM estimate mean:", np.mean(np.array(mean_error)))
            # print("PM estimate var:", np.mean(np.array(var_error)))
            # print("--------------------------------------")

            mean_errors.append(np.mean(np.array(mean_error)))
            var_errors.append(np.mean(np.array(var_error)))

        print("--------------------------------------")
        print("mean errors:", mean_errors)
        print("var errors:", var_errors)
        print("--------------------------------------")