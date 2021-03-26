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
from pathos.multiprocessing import ProcessingPool as Pool
from itertools import permutations
import numpy as np
from scipy.linalg import lapack


class KG(AcquisitionBase):
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
        self.name = "Constrained_KG"



        super(KG, self).__init__(model, space, optimizer, model_c, cost_withGradients=cost_withGradients)
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

        if self.Z_cdKG is None:
            self.generate_random_vectors()

        marginal_acqX = self._marginal_acq(X)
        KG = np.reshape(marginal_acqX, (X.shape[0],1))
        return KG

    def generate_random_vectors(self, optimize_discretization=True, optimize_random_Z=False):
        print("Setting dynamic parameters")
        self.base_marg_points = 40
        self.n_marginalisation_points = np.array([-3.5, -1.64, -0.67, 0 , 0.67, 1.64, 3.5])
        if optimize_random_Z:
            self.update_current_best()
            #lhd = lhs(self.dim - 1, samples=self.n_marginalisation_points)  # ** dim)
            perm = permutations(self.n_marginalisation_points, self.dim-1)
            self.Z_cdKG = np.array(list(perm))#norm(loc=0, scale=1).ppf(lhd)  # Pseudo-random number generation to compute expectation constrainted discrete kg
            print("perm", self.Z_cdKG.shape)
            perm = permutations(self.n_marginalisation_points, self.dim) # Pseudo-random number generation to compute expectation constrainted discrete kg
            perm = np.array(list(perm))

            index = np.random.choice(range(perm.shape[0]),self.n_base_points)
            # lhd = lhs(self.dim, samples=self.n_base_points)  # ** dim)
            # lhd = norm(loc=0, scale=1).ppf(lhd)  # Pseudo-random number generation to compute expectation

            self.Z_obj = np.array(list(perm))[index, :1]
            self.Z_const = np.array(list(perm))[index, 1:]

            print("self.Z_obj",self.Z_obj)
            print("self.Z_const",self.Z_const)
        self.optimise_discretisation = optimize_discretization


    def _marginal_acq(self, X):
        # print("_marginal_acq")
        """

        """
        #Initialise marginal acquisition variables
        acqX = np.zeros((X.shape[0], 1))

        #Precompute posterior pariance at vector points X. These are the same through every cKG loop.

        varX_obj = self.model.posterior_variance(X, noise=True)
        varX_c = self.model_c.posterior_variance(X, noise=True)

        # print("X",X)
        for i in range(0, len(X)):
            x = np.atleast_2d(X[i])

            #For each x new precompute covariance matrices for

            self.model.partial_precomputation_for_covariance(x)
            self.model.partial_precomputation_for_covariance_gradient(x)

            self.model_c.partial_precomputation_for_covariance(x)
            self.model_c.partial_precomputation_for_covariance_gradient(x)

            # Precompute aux_obj and aux_c for computational efficiency.
            aux_obj = np.reciprocal(varX_obj[:, i])
            aux_c = np.reciprocal(varX_c[:, i])


            #Create discretisation for discrete KG.
            start = time.time()
            if self.optimise_discretisation:
                self.X_Discretisation = self.Discretisation_X(index=i, X=X, aux_obj =aux_obj , aux_c =aux_c ,
                                                        Z_samples_obj=self.Z_samples_obj,
                                                        Z_samples_const=self.Z_samples_const)
                self.optimise_discretisation = False
            stop = time.time()
            # print("time generate disc", stop-start)
            # print("xnew",x,"X_discretisation", self.X_Discretisation, "shape", self.X_Discretisation.shape)
            start = time.time()
            acqX[i,:] = self.discrete_KG(Xd = self.X_Discretisation , xnew = x, Zc=self.Z_cdKG, aux_obj =aux_obj ,
                                                 aux_c =aux_c )

            # print("acqX",acqX)
            stop = time.time()
            # print("time solve KG", stop-start)

        return acqX.reshape(-1)

    def Discretisation_X(self, index, X, aux_obj, aux_c, Z_samples_obj, Z_samples_const):
        """

             """
        i = index
        x = np.atleast_2d(X[i])

        statistics_precision = []
        X_discretisation = np.zeros((self.n_base_points,X.shape[1]))

        # efficiency = 0
        self.new_anchors_flag = True
        for z in range(self.n_base_points):

            Z_obj  = self.Z_obj[z]
            Z_const = self.Z_const[z]

            # inner function of maKG acquisition function.
            def inner_func(X_inner):
                X_inner = np.atleast_2d(X_inner)
                # X_inner = X_inner.astype("int")
                grad_obj = gradients(x_new=x, model=self.model, Z=Z_obj, aux=aux_obj,
                                     X_inner=X_inner)  # , test_samples = self.test_samples)
                mu_xnew = grad_obj.compute_value_mu_xnew(x=X_inner)

                grad_c = gradients(x_new=x, model=self.model_c, Z=Z_const, aux=aux_c,
                                   X_inner=X_inner)  # , test_samples = initial_design('random', self.space, 1000))

                Fz = grad_c.compute_probability_feasibility_multi_gp(x=X_inner, l=0)
                func_val = mu_xnew * Fz #- self.control_variate

                return -func_val.reshape(-1)  # mu_xnew , Fz

            # inner function of maKG acquisition function with its gradient.

            def inner_func_with_gradient(X_inner):
                # print("inner_func_with_gradient")

                X_inner = np.atleast_2d(X_inner)
                # X_inner = X_inner.astype("int")

                grad_obj = gradients(x_new=x, model=self.model, Z=Z_obj, aux=aux_obj, X_inner=X_inner,
                                     precompute_grad=True)
                mu_xnew = grad_obj.compute_value_mu_xnew(x=X_inner)

                grad_mu_xnew = grad_obj.compute_gradient_mu_xnew(x=X_inner)

                grad_c = gradients(x_new=x, model=self.model_c, Z=Z_const, aux=aux_c, X_inner=X_inner,
                                   precompute_grad=True)

                Fz, grad_Fz = grad_c.compute_probability_feasibility_multi_gp(x=X_inner, l=0,
                                                                              gradient_flag=True)
                func_val = np.array(mu_xnew * Fz) #- self.control_variate

                func_grad_val = np.array(mu_xnew).reshape(-1) * grad_Fz.reshape(-1) + Fz.reshape(
                    -1) * grad_mu_xnew.reshape(
                    -1)  # grad_c.product_gradient_rule(func = np.array([np.array(mu_xnew).reshape(-1), Fz.reshape(-1)]), grad = np.array([grad_mu_xnew.reshape(-1) ,grad_Fz.reshape(-1) ]))

                assert ~ np.isnan(func_val); "nans found"

                return -func_val, -func_grad_val

            # self.gradient_sanity_check_2D(inner_func,inner_func_with_gradient)
            start = time.time()
            # print("Zobj", Z_obj, "Zc", Z_const)
            inner_opt_x, inner_opt_val = self.optimizer.optimize_inner_func(f=inner_func,
                                                                            f_df=None, reuse=self.new_anchors_flag)
            stop=time.time()
            print("time inner", stop-start)


            statistics_precision.append(inner_opt_val)
            # if ~ np.any(np.all(X_discretisation==inner_opt_x.reshape(-1),axis=1)):
            X_discretisation[z] = inner_opt_x.reshape(-1)
                # z +=1


            # efficiency+=1
        # print("efficiency", efficiency, "self.n_base_points",self.n_base_points)
        self.new_anchors_flag = False
        # raise
        return X_discretisation

    def discrete_KG(self,Xd, xnew, Zc, aux_obj , aux_c  ):

        xnew = np.atleast_2d(xnew)
        Xd = np.concatenate((Xd, xnew))
        Xd = np.concatenate((Xd, self.current_max_xopt))

        out = []
        for Zc_partition in np.array_split(Zc, 1):

            marginal_KG = np.zeros(Zc_partition.shape[0] )
            grad_c = gradients(x_new=xnew, model=self.model_c, Z=Zc_partition, aux=aux_c,
                               X_inner=Xd)  # , test_samples = initial_design('random', self.space, 1000))
            Fz = grad_c.compute_probability_feasibility_multi_gp(x=Xd, l=0)

            MM = self.model.predict(Xd)[0].reshape(-1)  # move up
            SS_Xd = self.model.posterior_covariance_between_points_partially_precomputed(Xd, xnew)[:, :, 0]
            inv_sd = np.asarray(np.sqrt(aux_obj)).reshape(())

            SS = SS_Xd * inv_sd
            MM = MM.reshape(-1)[:,np.newaxis]
            SS = SS.reshape(-1)[:,np.newaxis]


            self.c_SS = SS * Fz
            self.c_MM = MM * Fz
            self.Fz = Fz
            self.parallel = True
            start = time.time()
            # Finally compute KG!
            if self.parallel:
                pool = Pool(6)
                marginal_KG = pool.map(self.parallel_KG, list(range(Zc_partition.shape[0] )))
            else:
                for z in range(self.n_marginalisation_points):
                    marginal_KG[z] = self.KG(self.c_MM[:,z], self.c_SS[:,z])
            out.append(marginal_KG)
        KG_value = np.mean(out)
        # print("MSE", np.std(out)/len(np.array(out).reshape(-1)))
        stop = time.time()
        # print("TIME", stop-start)
        assert ~np.isnan(KG_value); "KG cant be nan"
        return -KG_value

    def parallel_KG(self, index):
        """
        Takes a set of intercepts and gradients of linear functions and returns
        the average hieght of the max of functions over Gaussain input.

        ARGS
            mu: length n vector, initercepts of linear functions
            sig: length n vector, gradients of linear functions

        RETURNS
            out: scalar value is gaussain expectation of epigraph of lin. funs
        """

        mu = self.c_MM[:,index]
        sig = self.c_SS[:,index]

        mmss = np.vstack((mu,sig)).T

        mmss_unique = np.unique(mmss , axis=0)

        mu = mmss_unique[:,0]
        sig = mmss_unique[:,1]
        O = sig.argsort()

        if np.all(mu==0) and np.all(sig==0):
            return 0

        a,b = self.filter_zeros(mu[O],sig[O])

        n = len(a)

        A = [0]
        C = [-float("inf")]

        while A[-1] < n - 1:
            s = A[-1]
            si = range(s + 1, n)
            Ci = -(a[s] - a[si]) / (b[s] - b[si])

            bestsi = np.argmin(Ci)
            C.append(Ci[bestsi])
            A.append(si[bestsi])

        C.append(float("inf"))
        cdf_C = norm.cdf(C)
        diff_CDF = cdf_C[1:] - cdf_C[:-1]
        pdf_C = norm.pdf(C)
        diff_PDF = pdf_C[1:] - pdf_C[:-1]
        # print(" a[A]*diff_CDF - b[A]*diff_PDF", a[A]*diff_CDF - b[A]*diff_PDF)
        # print("np.max(mu)", np.max(mu))
        out = np.sum(a[A] * diff_CDF - b[A] * diff_PDF) - np.max(mu)

        if out <0:
            # print("out", out)
            out=0
        #assert out >= 0;"KG cannot be negative"
        if np.isnan(out):
            print("mu", mu)
            print("sigma",sig)
            print("out", out)
            raise
        assert ~np.isnan(out);"KG cant be nan"
        return out

    def filter_zeros(self, a ,b):
        if len(np.array(range(len(b)))[b==0])==0:
            index=0
        else:
            index = np.max(np.array(range(len(b)))[b==0])

        return a[index:], b[index:]

    #TODO BOOST WITH NUMBA
    def KG(self, mu, sig):
        """
        Takes a set of intercepts and gradients of linear functions and returns
        the average hieght of the max of functions over Gaussain input.

        ARGS
            mu: length n vector, initercepts of linear functions
            sig: length n vector, gradients of linear functions

        RETURNS
            out: scalar value is gaussain expectation of epigraph of lin. funs
        """

        n = len(mu)
        O = sig.argsort()
        a = mu[O]
        b = sig[O]

        A = [0]
        C = [-float("inf")]
        while A[-1] < n - 1:
            s = A[-1]
            si = range(s + 1, n)
            Ci = -(a[s] - a[si]) / (b[s] - b[si])
            bestsi = np.argmin(Ci)
            C.append(Ci[bestsi])
            A.append(si[bestsi])

        C.append(float("inf"))
        cdf_C = norm.cdf(C)
        diff_CDF = cdf_C[1:] - cdf_C[:-1]
        pdf_C = norm.pdf(C)
        diff_PDF = pdf_C[1:] - pdf_C[:-1]

        out = np.sum(a[A] * diff_CDF - b[A] * diff_PDF) - np.max(mu)

        if out <0:
            # print("out", out)
            out=0
        # assert out >= 0;"KG cannot be negative"
        return out

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
