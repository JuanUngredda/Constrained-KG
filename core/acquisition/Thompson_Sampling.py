# Copyright (c) 2018, Raul Astudillo Marban

import numpy as np
from GPyOpt.experiment_design import initial_design
from GPyOpt.acquisitions.base import AcquisitionBase
from GPyOpt.core.task.cost import constant_cost_withGradients
from GPyOpt.experiment_design import initial_design
from aux_modules.gradient_modules import gradients
from scipy.stats import norm
import scipy
import time
import matplotlib.pyplot as plt

class TS(AcquisitionBase):
    """
    Multi-attribute knowledge gradient acquisition function

    :param model: GPyOpt class of model.
    :param space: GPyOpt class of domain.
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer.
    :param utility: utility function. See utility class for details. 
    """

    analytical_gradient_prediction = False

    def __init__(self, model, space, model_c=None, nz = 5, optimizer=None, cost_withGradients=None, utility=None, true_func=None):
        self.optimizer = optimizer
        self.utility = utility
        self.MCMC = False
        self.nz = nz
        self.counter = 0
        self.current_max_value = np.inf
        self.Z_samples_obj = None
        self.Z_samples_const = None
        self.true_func = true_func
        self.saved_Nx = -10
        self.name = "Constrained_Thompson_Sampling"
        super(TS, self).__init__(model, space, optimizer, model_c, cost_withGradients=cost_withGradients)
        if cost_withGradients == None:
            self.cost_withGradients = constant_cost_withGradients
        else:
            print('LBC acquisition does now make sense with cost. Cost set to constant.')
            self.cost_withGradients = constant_cost_withGradients


    def _compute_acq(self, X ):
        """
        Computes the aquisition function
        
        :param X: set of points at which the acquisition function is evaluated. Should be a 2d array.
        """
        # print("_compute_acq")
        X =np.atleast_2d(X)
        # X = X.astype("int")
        self.seed = int(time.time())
        suggested_sample, marginal_acqX = self._marginal_acq(X)
        return suggested_sample, marginal_acqX


    def _marginal_acq(self, X):
        # print("_marginal_acq")
        """

        """

        n_h=1


        for h in range(n_h):


            X = np.atleast_2d(X)

            mu_f, cov_f = self.model.predict_full_cov(X)
            mu_c, cov_c = self.model_c.predict_full_cov(X)

            np.random.seed(self.seed)
            mu_f = np.array(mu_f).reshape(-1)

            cov_f_jittered = cov_f[0] + np.identity(cov_f[0].shape[0])*1e-3
            Z_f = np.random.multivariate_normal(mu_f, cov_f_jittered , 1)


            Z_c = []
            for i in range(len(mu_c)):
                np.random.seed(self.seed)
                cov_c_jittered = cov_c[i] + np.identity(cov_c[i].shape[0])*1e-3
                Z_c.append(np.random.multivariate_normal(mu_c[i], cov_c_jittered , 1))


            Z_c = np.vstack(Z_c)
            Z_f = np.array(Z_f)
            Samples = np.concatenate((Z_f, Z_c)).T
            int_c = Samples[:, 1:] < 0
            int_c = np.product(int_c, axis=1)
            bool_c = list(map(bool, int_c))

            feasable_f = Samples[bool_c,0]
            feasable_x = X[bool_c,:]

            if len(feasable_f) ==0:
                print("Only infeasable solution")
                idx_minimum_violation = self.minimum_total_violation( X_samples=X, candidate_samples=Samples)

                suggested_sample = X[idx_minimum_violation]
                suggested_sample = np.array(suggested_sample).reshape(-1)
                suggested_sample = np.array(suggested_sample).reshape(1, -1)
                return suggested_sample, Samples[idx_minimum_violation,0]
            else:
                print("Feasable solution found")

                suggested_sample = feasable_x[np.argmax(feasable_f)]
                suggested_sample = np.array(suggested_sample).reshape(-1)
                suggested_sample = np.array(suggested_sample).reshape(1, -1)

                return suggested_sample, np.max(feasable_f)


    def minimum_total_violation(self, X_samples , candidate_samples):
        sum_constraints = np.sum(candidate_samples[:, 1:], axis=1)
        idx_minimum_violation = np.argmin(sum_constraints)

        return idx_minimum_violation


