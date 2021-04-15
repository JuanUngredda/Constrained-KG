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
from pyDOE import *
# from pathos.multiprocessing import ProcessingPool as Pool
# from multiprocessing import Pool
from itertools import permutations, product
import numpy as np
from scipy.linalg import lapack


class nEI(AcquisitionBase):
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

        self.counter = 0
        self.current_max_value = np.inf
        self.Z_samples_obj = None
        self.Z_samples_const = None
        self.Z_cdKG = None
        self.true_func = true_func
        self.saved_Nx = -10
        self.n_base_points = nz
        self.name = "noisy Expected Improvement"
        self.fixed_discretisation = None

        self.Z_obj = np.array([-3.24, -2.64, -1.67, -0.67, 0, 0.67, 1.67, 2.64, 3.24])
        super(nEI, self).__init__(model, space, optimizer, model_c, cost_withGradients=cost_withGradients)
        if cost_withGradients == None:
            self.cost_withGradients = constant_cost_withGradients
        else:
            print('LBC acquisition does now make sense with cost. Cost set to constant.')
            self.cost_withGradients = constant_cost_withGradients

        dimy = 1
        dimc =  self.model_c.output_dim
        self.dim = dimy + dimc

    def _compute_acq(self, X ):
        """
        Computes the aquisition function
        
        :param X: set of points at which the acquisition function is evaluated. Should be a 2d array.
        """
        # print("_compute_acq")

        X =np.atleast_2d(X)
        marginal_acqX = self._marginal_acq(X)
        nEI = np.reshape(marginal_acqX, (X.shape[0],1))
        # print("X", X, "KG", KG)
        return nEI

    def _marginal_acq(self, X):
        # print("_marginal_acq")
        """

        """
        #Initialise marginal acquisition variables
        acqX = np.zeros((X.shape[0], 1))

        #Precompute posterior pariance at vector points X. These are the same through every cKG loop.


        for i in range(0, len(X)):
            x = np.atleast_2d(X[i])

            Expected_value_EI = []
            # print("self.Z_obj[",self.Z_obj)
            for z in range(len(self.Z_obj)):

                Z_obj = self.Z_obj[z]
                EI_val = self.expected_improvement(x, Z_obj)
                Expected_value_EI.append(EI_val)

            # print("mean EI", np.mean(Expected_value_EI))
            acqX[i,:] = np.mean(Expected_value_EI)
        return acqX.reshape(-1)

    def expected_improvement(self, X, Z_obj):
        '''
        Computes the EI at points X based on existing samples X_sample
        and Y_sample using a Gaussian process surrogate model.

        Args:
            X: Points at which EI shall be computed (m x d).
            X_sample: Sample locations (n x d).
            Y_sample: Sample values (n x 1).
            gpr: A GaussianProcessRegressor fitted to samples.
            xi: Exploitation-exploration trade-off parameter.

        Returns:
            Expected improvements at points X.
        '''

        mu = self.model.posterior_mean(X)
        sigma = self.model.posterior_variance(X, noise=False)

        sigma = np.sqrt(sigma).reshape(-1, 1)
        mu = mu.reshape(-1,1)
        self.C = self.model_c.get_Y_values()
        bool_C = np.product(np.concatenate(self.C, axis=1) < 0, axis=1)

        mean_sampled_pts = self.model.posterior_mean(self.model.get_X_values()).reshape(-1)
        var_sampled_pts = self.model.posterior_variance(self.model.get_X_values(), noise=False).reshape(-1)
        objective_sampled_vals = mean_sampled_pts + np.sqrt(var_sampled_pts) * Z_obj
        func_val = objective_sampled_vals.reshape(-1)* bool_C.reshape(-1)

        mu_sample_opt = np.max(func_val)
        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        pf = self.probability_feasibility_multi_gp(X,self.model_c).reshape(-1,1)
        pf = np.array(pf).reshape(-1)
        ei = np.array(ei).reshape(-1)
        return (ei *pf).reshape(-1)

    def probability_feasibility_multi_gp(self, x, model, l=0):
        # print("model",model.output)
        x = np.atleast_2d(x)
        Fz = []
        for m in range(model.output_dim):
            Fz.append(self.probability_feasibility(x, model.output[m], l))
        Fz = np.product(Fz, axis=0)
        return Fz

    def probability_feasibility(self, x, model, l=0):

        mean = model.posterior_mean(x)
        var = model.posterior_variance(x, noise=False)
        std = np.sqrt(var).reshape(-1, 1)

        mean = mean.reshape(-1, 1)

        norm_dist = norm(mean, std)
        Fz = norm_dist.cdf(l)

        return Fz.reshape(-1, 1)

    def _compute_acq_withGradients(self, X):
        """
        """
        # print("_compute_acq_withGradients")

        raise
        X = np.atleast_2d(X)
        # X = X.astype("int")

        # self.update_current_best()
        # self.update_current_best()
        # Compute marginal aquisition function and its gradient for every value of the utility function's parameters samples,

        # if self.Z_samples_obj is None:
        #
        #     np.random.seed(X.shape[0])

        marginal_acqX, marginal_dacq_dX = self._marginal_acq_with_gradient(X)

        acqX = np.reshape(marginal_acqX,(X.shape[0], 1))
        dacq_dX = np.reshape(marginal_dacq_dX , X.shape)
        # print("self.Z_samples_obj", self.Z_samples_obj, "self.Z_samples_const", self.Z_samples_const)
        KG = acqX #-self.current_max_value
        #KG[KG < 0.0] = 0.0
        # print("acqX",acqX, "self.current_max_value",self.current_max_value,"KG",KG,"dacq_dX",dacq_dX)
        # print("self.current_max_value", self.current_max_value)
        # print("KG", KG)
        #print("self.Z_samples_obj ",self.Z_samples_obj ,"self.Z_samples_const",self.Z_samples_const)
        #print("X", X, "acqX", np.array(acqX).reshape(-1), "grad", np.array(dacq_dX).reshape(-1))
        return np.array(KG).reshape(-1), np.array(dacq_dX).reshape(-1)

    def run_inner_func_vals(self,f ):
        X = initial_design('random', self.space, 1000)
        f_val = []
        for x in X:
            x = x.reshape(1,-1)
            f_val.append(np.array(f(x)).reshape(-1))
        print("f_val min max", np.min(f_val), np.max(f_val))
        plt.scatter(X[:,0],X[:,1], c=np.array(f_val).reshape(-1))
        plt.show()

    def update_current_best(self):
        n_observations = self.model.get_X_values().shape[0]
        if n_observations > self.counter:
            print("updating current best..........")
            self.counter = n_observations
            self.current_max_xopt, self.current_max_value = self._compute_current_max()
        assert self.current_max_value.reshape(-1) is not np.inf; "error ocurred updating current best"

    def _compute_current_max(self):
        def current_func(X_inner):
            mu = self.model.posterior_mean(X_inner)[0]
            mu = mu.reshape(-1, 1)
            pf = self.probability_feasibility_multi_gp(X_inner, self.model_c).reshape(-1, 1)
            mu = np.array(mu).reshape(-1)
            pf = np.array(pf).reshape(-1)
            return -(mu * pf).reshape(-1)
        inner_opt_x, inner_opt_val = self.optimizer.optimize_inner_func(f=current_func, f_df=None, num_samples=1000)
        print("inner_opt_x, inner_opt_val",inner_opt_x, inner_opt_val)
        return inner_opt_x,-inner_opt_val
