import numpy as np
import matplotlib.pyplot as plt
#import pygmo as pg


class inf(object):
    def __init__(self):
        self.xmax = 1
        self.xmin = 0
        self.xarray, self.hsupport = np.linspace(0, 1, 501, retstep=True)
        self.support = np.c_[self.xarray, 1 - self.xarray]
        self.prob_dist = self.posterior(self.xarray)
        # plt.plot(self.xarray, self.prob_dist)
        # plt.show()

    def __call__(self, value = None):

        pdf_value = self.post_dens(x=value, Data=self.Data)
        return pdf_value

    def _update(self, Data):
        self.Data = Data
        self.prob_dist = self.post_dens(x=self.xarray, Data=self.Data)
        # plt.plot(self.xarray, self.prob_dist)
        # plt.show()


    def log_prior(self, a):
        """
        log prior using uniform distribution
        :param a: value of input narray x sigma narray
        :return:  value of logprior
        """
        Lprior = np.zeros(len(a))
        max_ls = self.xmax
        min_ls = self.xmin
        min_condition = np.vstack((a >= min_ls))
        max_condition = np.vstack((a <= max_ls))
        prior = np.product(1.0 * (min_condition & max_condition), axis=1)
        Lprior[prior != 0] = np.log(prior[prior != 0])
        Lprior[prior == 0] = -np.inf
        return Lprior

    def log_likelihood_i(self, data, val):
        m_vector = np.c_[val, 1 - val]
        log_lk = []
        for i in m_vector:
            dU = -1 * np.dot((data[0, :] - data[1, :]), i)
            log_lk.append(-np.log(1 + np.exp(dU)))
        return log_lk

    def log_likelihood(self, Data, val):

        arr_ = range(len(Data))
        l_lklhd = [self.log_likelihood_i(Data[idx], val) for idx in arr_]

        l_lklhd = np.sum(l_lklhd, axis=0)

        return l_lklhd

    def post_dens(self, x, Data=np.array([])):

        unorm_post = self.posterior(self.xarray, Data)

        constant = np.sum(unorm_post * self.hsupport)

        pdf_x = self.posterior(x, Data)

        return pdf_x / constant

    def posterior(self, val, Data=np.array([])):

        if Data.size > 0:
            L_prior = self.log_prior(val)
            L_Like = self.log_likelihood(Data, val)
            post = L_prior + L_Like

        else:
            L_prior = self.log_prior(val)
            post = L_prior

        return np.exp(post)

    def sampler(self, n):
        """

        :param n: number of samples
        :param dist: pdf of distribution. normalised inside the function
        :param domain: discreatised domain
        :return: set of samples
        """
        print("inside sampler")
        assert not len(self.prob_dist) == 1, "Trying to generate samples from scalar. Hint: Insert pdf"
        domain = self.xarray
        dist = self.prob_dist
        dist = dist / np.sum(dist)
        probabilities = dist * (1 / np.sum(dist))
        val = np.random.choice(domain, n, p=probabilities)
        return val
