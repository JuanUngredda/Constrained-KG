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
import itertools
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
        # self.n_base_points = nz
        self.name = "Constrained_KG"
        self.fixed_discretisation = None


        super(KG, self).__init__(model, space, optimizer, model_c, cost_withGradients=cost_withGradients)
        if cost_withGradients == None:
            self.cost_withGradients = constant_cost_withGradients
        else:
            print('LBC acquisition does now make sense with cost. Cost set to constant.')
            self.cost_withGradients = constant_cost_withGradients

        dimy = 1
        dimc =  self.model_c.output_dim
        self.dimc = dimc
        self.dim = dimy + dimc

    def _compute_acq(self, X ):
        """
        Computes the aquisition function
        
        :param X: set of points at which the acquisition function is evaluated. Should be a 2d array.
        """
        # print("_compute_acq")

        X =np.atleast_2d(X)
        marginal_acqX = self._marginal_acq(X)
        KG = np.reshape(marginal_acqX, (X.shape[0],1))
        return KG

    def generate_random_vectors(self, optimize_discretization=True, optimize_random_Z=False, fixed_discretisation=None):

        self.base_marg_points = 40
        self.n_marginalisation_points = np.array([ -2.64, -0.67, 0, 0.67, 2.64])
        self.c_marginalisation_points = np.array([-2.64, 0, 2.64])


        self.n_base_points = len(self.n_marginalisation_points)
        self.optimise_discretisation = optimize_discretization

        if optimize_random_Z:

            lhd = lhs(self.dim - 1, samples=len(self.n_marginalisation_points))  # ** dim)
            constraint_quantiles = norm(loc=0, scale=1).ppf(lhd)
            self.Z_cdKG = constraint_quantiles

            # clist = []
            # for j in range(self.dimc):
            #     clist.append(self.c_marginalisation_points)
            # res = list(itertools.product(*clist))
            # list(res)
            # self.Z_cdKG = np.array(list(res))

            # alllist = []
            # alllist.append(self.n_marginalisation_points)
            # for j in range(self.dimc):
            #     alllist.append(self.c_marginalisation_points)
            # res = list(itertools.product(*alllist))
            # list(res)

            self.Z_obj = np.atleast_2d(self.n_marginalisation_points).T #np.array(list(res))[:, :1]
            self.Z_const = constraint_quantiles #np.array(list(res))[:, 1:]


        if fixed_discretisation is not None:
            perm = product(self.n_marginalisation_points, repeat= self.dim-1)
            self.Z_cdKG = np.array(list(perm)) #Pseudo-random number generation to compute expectation constrainted discrete kg
            self.fixed_discretisation = True
            self.X_Discretisation = fixed_discretisation
        else:
            self.fixed_discretisation = False
            if optimize_discretization==True:
                self.X_Discretisation =None


    def _marginal_acq(self, X):
        # print("_marginal_acq")
        """

        """
        #Initialise marginal acquisition variables


        #Precompute posterior pariance at vector points X. These are the same through every cKG loop.

        varX_obj = self.model.posterior_variance(X, noise=True)
        varX_c = self.model_c.posterior_variance(X, noise=True)

        acqX = np.zeros((X.shape[0], 1))

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
            if self.fixed_discretisation is False:
                if self.optimise_discretisation:
                    print("optimise_discretisation", self.optimise_discretisation)
                    self.X_Discretisation = self.Discretisation_X(index=i, X=X, aux_obj =aux_obj , aux_c =aux_c)
                    print("updated discretisation")
                    self.optimise_discretisation = False

            kg_val = self.discrete_KG(Xd = self.X_Discretisation , xnew = x, Zc=self.Z_cdKG, aux_obj =aux_obj , aux_c =aux_c )
            acqX[i,:] = kg_val
        return acqX.reshape(-1)

    def Discretisation_X(self, index, X, aux_obj, aux_c):
        """

             """
        i = index
        x = np.atleast_2d(X[i])

        statistics_precision = []
        X_discretisation = np.zeros((len(self.Z_obj),X.shape[1]))

        # efficiency = 0
        self.new_anchors_flag = True
        for z in range(len(self.Z_obj)):

            # print("self.Z_obj",self.Z_obj.shape)
            # print("self.Z_const", self.Z_const.shape)
            Z_obj  = self.Z_obj[z]
            Z_const = self.Z_const[z]
            # inner function of maKG acquisition function.

            def inner_func(X_inner):
                X_inner = np.atleast_2d(X_inner)
                # X_inner = X_inner.astype("int")
                grad_obj = gradients(x_new=x, model=self.model, Z=Z_obj, aux=aux_obj,
                                     X_inner=X_inner)  # , test_samples = self.test_samples)
                mu_xnew = grad_obj.compute_value_mu_xnew(x=X_inner)

                Z_const = self.Z_const[z]
                if len(Z_const.shape)==1:
                    Z_const = np.atleast_2d(Z_const)

                grad_c = gradients(x_new=x, model=self.model_c, Z=Z_const, aux=aux_c,
                                   X_inner=X_inner)  # , test_samples = initial_design('random', self.space, 1000))

                Fz = grad_c.compute_probability_feasibility_multi_gp(x=X_inner, l=0)

                func_val = mu_xnew* Fz #- inner_opt_val

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
                print("-func_val, -func_grad_val",-func_val, -func_grad_val)
                return -func_val, func_grad_val


            inner_opt_x, inner_opt_val = self.optimizer.optimize_inner_func(f=inner_func,
                                                                            f_df=None)


            statistics_precision.append(inner_opt_val)
            X_discretisation[z] = inner_opt_x.reshape(-1)

        self.new_anchors_flag = False
        # print("discretisation", X_discretisation)
        return X_discretisation

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

        marginal_acqX, marginal_dacq_dX = self._marginal_acq_with_gradient(X)

        acqX = np.reshape(marginal_acqX,(X.shape[0], 1))
        dacq_dX = np.reshape(marginal_dacq_dX , X.shape)
        KG = acqX
        return np.array(KG).reshape(-1), np.array(dacq_dX).reshape(-1)

    def _marginal_acq_with_gradient(self, X):
        # print("_marginal_acq")
        """

        """
        # Initialise marginal acquisition variables

        # Precompute posterior pariance at vector points X. These are the same through every cKG loop.

        varX_obj = self.model.posterior_variance(X, noise=True)
        varX_c = self.model_c.posterior_variance(X, noise=True)
        acqX = np.zeros((X.shape[0], 1))
        dacq_dX = np.zeros((X.shape[0], X.shape[1], 1))

        for i in range(0, len(X)):
            x = np.atleast_2d(X[i])

            # For each x new precompute covariance matrices for

            self.model.partial_precomputation_for_covariance(x)
            self.model.partial_precomputation_for_covariance_gradient(x)

            self.model_c.partial_precomputation_for_covariance(x)
            self.model_c.partial_precomputation_for_covariance_gradient(x)

            # Precompute aux_obj and aux_c for computational efficiency.

            aux_obj = np.reciprocal(varX_obj[:, i])
            aux_c = np.reciprocal(varX_c[:, i])

            # Create discretisation for discrete KG.
            start = time.time()
            # print("self.fixed_discretisation",self.fixed_discretisation)
            # print("self.optimise_discretisation",self.optimise_discretisation)
            # if test_mode:
            #     self.X_Discretisation = self.Discretisation_X(index=i, X=X, aux_obj=aux_obj, aux_c=aux_c)
            if self.fixed_discretisation is False:
                if self.optimise_discretisation:
                    print("optimise_discretisation", self.optimise_discretisation)
                    self.X_Discretisation = self.Discretisation_X(index=i, X=X, aux_obj=aux_obj, aux_c=aux_c)
                    print("updated discretisation")
                    self.optimise_discretisation = False
            # print("x", x)
            kg_val, kg_grad = self.discrete_KG(Xd=self.X_Discretisation, xnew=x, Zc=self.Z_cdKG, aux_obj=aux_obj,
                                      aux_c=aux_c, grad=True)
            # print("kg",  kg_val)
            acqX[i, :] = kg_val
            dacq_dX[i, :, 0] = kg_grad
        # print("acqX",acqX)

        return acqX.reshape(-1), dacq_dX

    def _plots(self, test_samples, discretisation=None):

        def current_func(X_inner, Z_const, aux_c):
            X_inner = np.atleast_2d(X_inner)
            mu = self.model.posterior_mean(X_inner)[0]
            mu = np.array(mu).reshape(-1)
            grad_c = gradients(x_new=x, model=self.model_c, Z=Z_const, aux=aux_c,
                               X_inner=X_inner)  # , test_samples = initial_design('random', self.space, 1000))

            Fz = grad_c.compute_probability_feasibility_multi_gp(x=X_inner, l=0).reshape(-1)
            return -(mu * Fz).reshape(-1)

        plot_samples = initial_design('random', self.space, 1000)
        test_samples = np.sort(test_samples ,axis=0)
        plot_samples = np.sort(plot_samples,axis=0)
        # print("test_samples",test_samples)
        varX_obj = self.model.posterior_variance(test_samples, noise=True)
        varX_c = self.model_c.posterior_variance(test_samples, noise=True)

        colors = ["magenta", "green", "red", "blue", "orange"]
        Z_obj = np.array([  -2.64, -0.67, 0, 0.67, 2.64]) #np.array([-2.64, -1.64, -0.67, 0, 0.67, 1.64, 2.64])
        Z_const = np.array([-2.64,  0, 2.64])
        for i in range(0, len(test_samples)):
            x = np.atleast_2d(test_samples[i])
            # For each x new precompute covariance matrices for
            self.model.partial_precomputation_for_covariance(x)
            self.model.partial_precomputation_for_covariance_gradient(x)
            self.model_c.partial_precomputation_for_covariance(x)
            self.model_c.partial_precomputation_for_covariance_gradient(x)
            # Precompute aux_obj and aux_c for computational efficiency.
            aux_obj = np.reciprocal(varX_obj[:, i])
            aux_c = np.reciprocal(varX_c[:, i])
            maxgraph = []
            feasibility_graph = []
            posterior_mean_c = []
            posterior_var_c = []
            for zc in range(len(Z_const)):
                individual_maxgraph = []

                grad_c = gradients(x_new=x, model=self.model_c, Z=Z_const[zc], aux=aux_c,
                                   X_inner=plot_samples)  # , test_samples = initial_design('random', self.space, 1000))
                feasibility_graph.append(grad_c.compute_probability_feasibility_multi_gp(x=plot_samples, l=0).reshape(-1))
                posterior_mean_c.append(grad_c.compute_value_mu_xnew(x=plot_samples).reshape(-1))
                posterior_var_c.append(grad_c.compute_posterior_var_x_new(x=plot_samples).reshape(-1))
                for zo in range(len(Z_obj)):
                    def inner_func(X_inner, verbose=False):
                        X_inner = np.atleast_2d(X_inner)
                        # X_inner = X_inner.astype("int")
                        grad_obj = gradients(x_new=x, model=self.model, Z=Z_obj[zo], aux=aux_obj,
                                             X_inner=X_inner)  # , test_samples = self.test_samples)
                        mu_xnew = grad_obj.compute_value_mu_xnew(x=X_inner)

                        grad_c = gradients(x_new=x, model=self.model_c, Z=Z_const[zc], aux=aux_c,
                                           X_inner=X_inner)  # , test_samples = initial_design('random', self.space, 1000))

                        Fz = grad_c.compute_probability_feasibility_multi_gp(x=X_inner, l=0)

                        func_val = mu_xnew * Fz  # - inner_opt_val

                        return -func_val.reshape(-1)

                    def inner_func_with_gradient(X_inner):
                        # print("inner_func_with_gradient")

                        X_inner = np.atleast_2d(X_inner)
                        # X_inner = X_inner.astype("int")

                        grad_obj = gradients(x_new=x, model=self.model, Z=Z_obj[zo], aux=aux_obj, X_inner=X_inner,
                                             precompute_grad=True)
                        mu_xnew = grad_obj.compute_value_mu_xnew(x=X_inner)

                        grad_mu_xnew = grad_obj.compute_gradient_mu_xnew(x=X_inner)

                        grad_c = gradients(x_new=x, model=self.model_c, Z=Z_const[zc], aux=aux_c, X_inner=X_inner,
                                           precompute_grad=True)

                        Fz, grad_Fz = grad_c.compute_probability_feasibility_multi_gp(x=X_inner, l=0,
                                                                                      gradient_flag=True)
                        func_val = np.array(mu_xnew * Fz)  # - self.control_variate

                        func_grad_val = np.array(mu_xnew).reshape(-1) * grad_Fz.reshape(-1) + Fz.reshape(
                            -1) * grad_mu_xnew.reshape(
                            -1)  # grad_c.product_gradient_rule(func = np.array([np.array(mu_xnew).reshape(-1), Fz.reshape(-1)]), grad = np.array([grad_mu_xnew.reshape(-1) ,grad_Fz.reshape(-1) ]))

                        assert ~ np.isnan(func_val);
                        "nans found"
                        print("-func_val, -func_grad_val", -func_val, -func_grad_val)
                        return -func_val, func_grad_val


                    def inner_func_predict(X_inner):
                        X_inner = np.atleast_2d(X_inner)
                        # X_inner = X_inner.astype("int")
                        grad_obj = gradients(x_new=x, model=self.model, Z=Z_obj[zo], aux=aux_obj,
                                             X_inner=X_inner)
                        mu_xnew = self.model.predict(X_inner)[0].reshape(-1)
                        var_xnew = grad_obj.compute_b(xopt=X_inner)

                        grad_c = gradients(x_new=x, model=self.model_c, Z=Z_const[zc], aux=aux_c,
                                           X_inner=X_inner)

                        Fz = grad_c.compute_probability_feasibility_multi_gp(x=X_inner, l=0)

                        func_val = mu_xnew.reshape(-1) * Fz.reshape(-1)  # - inner_opt_val

                        return func_val.reshape(-1), var_xnew.reshape(-1)* (np.array(Fz).reshape(-1))

                    if discretisation is not None:
                        current_test_samples_mean = -current_func(X_inner=plot_samples, Z_const=Z_const[zc],
                                                                  aux_c=aux_c)
                        next_test_samples_mean = -inner_func(plot_samples)
                        mean_n1, varn_n1 = inner_func_predict(plot_samples)

                        current_inner_opt_x, current_inner_opt_val = self._compute_current_max(x, Z_const[zc], aux_c)

                        discretisation_vals = -inner_func(discretisation)
                        next_inner_opt_x = discretisation[np.argmax(discretisation_vals)]
                        # print("func_val2",inner_func(next_inner_opt_x , verbose=True))
                        next_inner_opt_val = -np.max(discretisation_vals )


                        individual_maxgraph.append(-next_inner_opt_val)
                        maxgraph.append(-next_inner_opt_val)
                        plt.plot(plot_samples, mean_n1, color="pink")
                        plt.plot(plot_samples, mean_n1 + 1.96 * varn_n1, color="pink", alpha=0.3)
                        plt.plot(plot_samples, mean_n1 - 1.96 * varn_n1, color="pink", alpha=0.3)
                        plt.fill_between(plot_samples.reshape(-1), mean_n1 - 1.96 * varn_n1,
                                         mean_n1 + 1.96 * varn_n1, color="pink", alpha=0.2)


                        plt.plot(plot_samples, current_test_samples_mean, color="grey")
                        plt.plot(plot_samples, next_test_samples_mean, color="grey", linestyle='dashed')
                        plt.scatter(current_inner_opt_x, current_inner_opt_val, color="red")
                        plt.scatter(next_inner_opt_x, -next_inner_opt_val, color="green")
                        plt.scatter(np.array(discretisation).reshape(-1), np.repeat([0], len(np.array(discretisation).reshape(-1))), color="blue")
                        plt.scatter(x, [0], color="magenta")

                    else:
                        current_test_samples_mean = -current_func(X_inner=plot_samples, Z_const=Z_const[zc], aux_c=aux_c)
                        next_test_samples_mean = -inner_func(plot_samples)
                        mean_n1, varn_n1 = inner_func_predict(plot_samples)

                        current_inner_opt_x, current_inner_opt_val = self._compute_current_max(x, Z_const[zc], aux_c)
                        next_inner_opt_x, next_inner_opt_val = self.optimizer.optimize_inner_func(f=inner_func,
                                                                                        f_df=None)
                        individual_maxgraph.append(-next_inner_opt_val)
                        maxgraph.append(-next_inner_opt_val)
                        plt.plot(plot_samples, mean_n1, color="pink")
                        plt.plot(plot_samples, mean_n1 + 1.95*varn_n1, color="pink", alpha=0.3)
                        plt.plot(plot_samples, mean_n1 - 1.95 * varn_n1, color="pink", alpha=0.3)
                        plt.fill_between(plot_samples.reshape(-1), mean_n1 - 1.95 *varn_n1, mean_n1 + 1.95*varn_n1, color="pink", alpha=0.2)

                        plt.plot(plot_samples, current_test_samples_mean, color = "grey")
                        plt.plot(plot_samples, next_test_samples_mean, color="grey", linestyle='dashed')
                        plt.scatter(current_inner_opt_x, current_inner_opt_val , color="grey")
                        plt.scatter(next_inner_opt_x, -next_inner_opt_val, color="grey")
                        plt.scatter(x, [0], color="magenta")

                plt.show()


            Zo, Zc = np.meshgrid(Z_obj, Z_const)
            maxvals = np.array(maxgraph).reshape(Zo.shape)


            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(Zc,Zo, maxvals)
            ax.set_ylabel('$Z_{y}$')
            ax.set_xlabel('$Z_{c}$')
            ax.set_zlabel('$max_{j} \mu$')
            plt.show()


            # plt.plot(plot_samples, np.array(feasibility_graph).T)
            # plt.title("feasible region")
            # plt.show()
            #
            # plt.plot(plot_samples, np.array(posterior_mean_c).T)
            # plt.plot(plot_samples, np.array(self.model_c.predict(plot_samples)[0]).reshape(-1), color="black")
            #
            # grad_c = gradients(x_new=x, model=self.model_c, Z=Z_const[0], aux=aux_c,
            #                    X_inner=plot_samples)
            #
            # var_xnew = grad_c.compute_b(xopt=plot_samples)
            #
            # plt.plot(plot_samples, np.array(self.model_c.predict(plot_samples)[0]).reshape(-1) + 1.95*np.sqrt(var_xnew), color="black")
            # plt.plot(plot_samples,
            #          np.array(self.model_c.predict(plot_samples)[0]).reshape(-1) - 1.95 * np.sqrt(var_xnew), color="black")
            # plt.title("posterior means constraints")
            # plt.show()
            #
            # plt.title("posterior var constraints")
            # plt.plot(plot_samples, np.array(posterior_var_c).T)
            # plt.show()


        kg_val = []
        plot_kg_samples = initial_design('random', self.space, 500)
        plot_kg_samples  = np.sort(plot_kg_samples , axis=0)
        varX_obj = self.model.posterior_variance(plot_kg_samples, noise=True)
        varX_c = self.model_c.posterior_variance(plot_kg_samples, noise=True)

        for i in range( len(plot_kg_samples)):
            x = np.atleast_2d(plot_kg_samples[i])
            self.model.partial_precomputation_for_covariance(x)
            self.model.partial_precomputation_for_covariance_gradient(x)
            self.model_c.partial_precomputation_for_covariance(x)
            self.model_c.partial_precomputation_for_covariance_gradient(x)

            aux_obj = np.reciprocal(varX_obj[:, i])
            aux_c = np.reciprocal(varX_c[:, i])
            kg_val.append(self.discrete_KG(Xd=plot_kg_samples, xnew=x, Zc=self.Z_cdKG, aux_obj=aux_obj, aux_c=aux_c, grad=False))

        plt.plot(plot_kg_samples.reshape(-1), np.array(kg_val).reshape(-1))
        plt.show()

        plot_kg_samples = plot_kg_samples[np.argmax(np.array(kg_val).reshape(-1))]
        print("arg max kg", plot_kg_samples, "max kg", np.max(kg_val))
        plot_kg_samples = np.sort(plot_kg_samples, axis=0)
        varX_obj = self.model.posterior_variance(plot_kg_samples, noise=True)
        varX_c = self.model_c.posterior_variance(plot_kg_samples, noise=True)

        for i in range(len(plot_kg_samples)):
            x = np.atleast_2d(plot_kg_samples[i])
            self.model.partial_precomputation_for_covariance(x)
            self.model.partial_precomputation_for_covariance_gradient(x)
            self.model_c.partial_precomputation_for_covariance(x)
            self.model_c.partial_precomputation_for_covariance_gradient(x)

            aux_obj = np.reciprocal(varX_obj[:, i])
            aux_c = np.reciprocal(varX_c[:, i])
            kg_val.append(
                self.discrete_KG(Xd=initial_design('random', self.space, 500), xnew=x, Zc=self.Z_cdKG, aux_obj=aux_obj, aux_c=aux_c, grad=False, verbose=False))

    def discrete_KG(self, Xd, xnew, Zc, aux_obj, aux_c, grad=False, verbose=False):
        xnew = np.atleast_2d(xnew)
        Xd = np.concatenate((Xd, xnew))

        self.grad = grad
        out = []
        for Zc_partition in np.array_split(Zc, 1):
            # print("computing gradients")
            grad_c = gradients(x_new=xnew, model=self.model_c, Z=Zc_partition, aux=aux_c,
                               X_inner=Xd)  # , test_samples = initial_design('random', self.space, 1000))
            Fz = grad_c.compute_probability_feasibility_multi_gp(x=Xd, l=0)

            MM = self.model.predict(Xd)[0].reshape(-1)  # move up
            SS_Xd = self.model.posterior_covariance_between_points_partially_precomputed(Xd, xnew)[:, :, 0]
            inv_sd = np.asarray(np.sqrt(aux_obj)).reshape(())

            SS = SS_Xd * inv_sd
            MM = MM.reshape(-1)[:, np.newaxis]
            SS = SS.reshape(-1)[:, np.newaxis]

            self.c_SS = np.abs(SS) * Fz
            self.c_MM = MM * Fz

            self.bases_value = np.max(self.c_MM, axis=0)

            marginal_KG = []
            for idx in list(range(Zc_partition.shape[0])):
                marginal_KG.append(self.parallel_KG(idx, Xd=Xd,verbose=verbose))
            out.append(marginal_KG)
        if verbose:
            print("Zc", np.array_split(Zc, 1))
            print("marginal_KG",out)
            print("KG_value",np.mean(out))
        KG_value = np.mean(out)
        # gradKG_value = np.mean(gradout, axis=?)
        assert ~np.isnan(KG_value);
        "KG cant be nan"
        return KG_value#, gradKG_value

    def parallel_KG(self, index, Xd=None,verbose=False):
        """
        Calculates the linear epigraph, i.e. the boundary of the set of points
        in 2D lying above a collection of straight lines y=a+bx.
        Parameters
        ----------
        a
            Vector of intercepts describing a set of straight lines
        b
            Vector of slopes describing a set of straight lines
        tol
            Minimum slope (in absolute value) different from zero
        Returns
        -------
        KGCB
            average hieght of the epigraph
        grad_a
            dKGCB/db, vector
        grad_b
            dKGCB/db, vector
        """
        a = np.array(self.c_MM[:, index]).reshape(-1)
        b = np.array(self.c_SS[:, index]).reshape(-1)


        assert len(b) > 0, "must provide slopes"
        assert len(a) == len(b), f"#intercepts != #slopes, {len(a)}, {len(b)}"
        # ensure 1D
        a = np.atleast_1d(a.squeeze())
        b = np.atleast_1d(b.squeeze())
        max_a_index = np.argmax(a)

        amax = np.max(a)
        b_amax = b[np.argmax(a)]
        n_elems = len(a)

        if np.all(np.abs(b) < 0.0000000001):
            return 0  # index, 0, np.zeros(a.shape), np.zeros(b.shape)

        # order by ascending b and descending a
        order = np.lexsort((-a, b))
        a = a[order]
        b = b[order]

        a_sorted = a
        b_sorted = b
        # exclude duplicated b (or super duper similar b)
        max1 = np.max(a)
        threshold = (b[-1] - b[0]) * 0.00001
        diff_b = b[1:] - b[:-1]
        keep = diff_b > threshold
        keep = np.concatenate([[True], keep])
        order = order[keep]
        a = a[keep]
        b = b[keep]
        max2 = np.max(a)

        #making sure that the current maximum is not filtered
        if not amax in a:
            a = np.concatenate((a, np.atleast_1d(amax)))
            b = np.concatenate((b, np.atleast_1d(b_amax)))

        order = np.lexsort((-a, b))
        a = a[order]
        b = b[order]

        # initialize
        idz = [0]
        i_last = 0
        x = [-np.inf]

        n_lines = len(a)

        # main loop TODO describe logic
        # TODO not pruning properly, e.g. a=[0,1,2], b=[-1,0,1]
        # returns x=[-inf, -1, -1, inf], shouldn't affect kgcb
        while i_last < n_lines - 1:
            i_mask = np.arange(i_last + 1, n_lines)
            x_mask = -(a[i_last] - a[i_mask]) / (b[i_last] - b[i_mask])

            best_pos = np.argmin(x_mask)
            idz.append(i_mask[best_pos])
            x.append(x_mask[best_pos])

            i_last = idz[-1]

        x.append(np.inf)

        x = np.array(x)
        idz = np.array(idz)

        # found the epigraph, now compute the expectation
        a = a[idz]
        b = b[idz]
        order = order[idz]

        pdf = norm.pdf(x)
        cdf = norm.cdf(x)

        KG = np.sum(a * (cdf[1:] - cdf[:-1]) + b * (pdf[:-1] - pdf[1:]))
        KG -= np.max(a)

        if verbose:
            print("a_sorted ",a_sorted )
            print("b_sorted", b_sorted)
            print("max1", max1)
            print("max2", max2)
            print("current max",np.max(a) )
            print("idz", idz)
            print("a", a)
            print("b", b)
            plt.scatter(Xd.reshape(-1), np.array(self.c_MM[:, index]).reshape(-1))
            plt.plot(np.linspace(0,5,2), np.repeat(np.max(a), 2), color="red")
            plt.show()
            # raise
        # if KG < -1e-5:
        #     print("KG cant be negative")
        #     print("np.sum(a * (cdf[1:] - cdf[:-1]) + b * (pdf[:-1] - pdf[1:]))",
        #           np.sum(a * (cdf[1:] - cdf[:-1]) + b * (pdf[:-1] - pdf[1:])))
        #     print("self.bases_value[index]", np.max(a))
        #     print("KG", KG)
        #     raise

        # KG = np.clip(KG, 0, np.inf)

        if np.isnan(KG):
            print("KG", KG)
            print("self.bases_value[index]", max_a_index)
            print("a", a)
            print("b", b)
            raise

        if self.grad:
            # gradients of KG
            grad_a = np.zeros((n_elems))
            grad_a[max_a_index] = -1
            grad_a[order] = grad_a[order] + cdf[1:] - cdf[:-1]

            grad_b = np.zeros((n_elems))
            grad_b[order] = -pdf[1:] + pdf[:-1]

            return KG , grad_a, grad_b
        else:
            return KG

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

    def _compute_current_max(self,x, Z_const, aux_c):
        def current_func(X_inner):
            X_inner = np.atleast_2d(X_inner)
            mu = self.model.posterior_mean(X_inner)[0]
            mu = np.array(mu).reshape(-1)
            grad_c = gradients(x_new=x, model=self.model_c, Z=Z_const, aux=aux_c,
                               X_inner=X_inner)  # , test_samples = initial_design('random', self.space, 1000))

            Fz = grad_c.compute_probability_feasibility_multi_gp(x=X_inner, l=0).reshape(-1)

            return -(mu * Fz).reshape(-1)
        inner_opt_x, inner_opt_val = self.optimizer.optimize_inner_func(f=current_func, f_df=None, num_samples=1000)
        # print("inner_opt_x, inner_opt_val",inner_opt_x, inner_opt_val)
        return inner_opt_x,-inner_opt_val
