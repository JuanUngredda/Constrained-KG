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

class KG(AcquisitionBase):
    """
    Multi-attribute knowledge gradient acquisition function

    :param model: GPyOpt class of model.
    :param space: GPyOpt class of domain.
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer.
    :param utility: utility function. See utility class for details. 
    """

    analytical_gradient_prediction = False

    def __init__(self, model, space, model_c=None, nz = 5, optimizer=None, cost_withGradients=None, utility=None):
        self.optimizer = optimizer
        self.utility = utility
        self.MCMC = False
        self.nz = nz
        self.counter = 0
        self.current_max_value = np.inf
        super(KG, self).__init__(model, space, optimizer, model_c, cost_withGradients=cost_withGradients)
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

        # self.update_current_best()
          # Number of samples of Z.

        self.Z_samples_obj = np.random.normal(size=self.nz)
        self.Z_samples_const = np.random.normal(size=self.nz)
        # print("self.Z_samples_obj",self.Z_samples_obj,"self.Z_samples_const",self.Z_samples_const)
        marginal_acqX = self._marginal_acq(X)
        acqX = np.reshape(marginal_acqX, (X.shape[0],1))

        # KG =  acqX - self.current_max_value
        # KG[KG < 0.0] = 0.0
        # print("self.current_max_value",  self.current_max_value)
        # print("KG", KG)
        # print("acqX",acqX)
        # print("acqX",acqX)
        return acqX

    def update_current_best(self):
        n_observations = self.model.get_X_values().shape[0]
        if n_observations > self.counter:
            print("updating current best..........")
            self.counter = n_observations
            self.current_max_value = self._compute_current_max()
        assert self.current_max_value.reshape(-1) is not np.inf; "error ocurred updating current best"

    def _compute_current_max(self):
        def current_func(X_inner):
            mu = self.model.posterior_mean(X_inner)[0]
            mu = mu.reshape(-1, 1)
            pf = self.probability_feasibility_multi_gp(X_inner, self.model_c).reshape(-1, 1)
            return -(mu * pf)
        inner_opt_x, inner_opt_val = self.optimizer.optimize(f=current_func, f_df=None, num_samples=100)

        return -inner_opt_val

    def probability_feasibility_multi_gp(self, x, model, l=0):
        # print("model",model.output)
        x = np.atleast_2d(x)
        Fz = []
        for m in range(model.output_dim):
            Fz.append(self.probability_feasibility(x, model.output[m], l))
        Fz = np.product(Fz, axis=0)
        return Fz

    def probability_feasibility(self, x, model, l=0):

        model = model.model

        mean, cov = model.predict(x, full_cov=True)
        var = np.diag(cov).reshape(-1, 1)
        std = np.sqrt(var).reshape(-1, 1)

        mean = mean.reshape(-1, 1)

        norm_dist = norm(mean, std)
        Fz = norm_dist.cdf(l)

        return Fz.reshape(-1, 1)

    def _compute_acq_withGradients(self, X):
        """
        """
        # print("_compute_acq_withGradients")


        X =np.atleast_2d(X)
        # self.update_current_best()
        # Compute marginal aquisition function and its gradient for every value of the utility function's parameters samples,

        marginal_acqX, marginal_dacq_dX = self._marginal_acq_with_gradient(X)

        acqX = np.reshape(marginal_acqX,(X.shape[0], 1))
        dacq_dX = np.reshape(marginal_dacq_dX , X.shape)
        # print("self.Z_samples_obj", self.Z_samples_obj, "self.Z_samples_const", self.Z_samples_const)
        # KG = acqX -self.current_max_value
        # KG[KG < 0.0] = 0.0
        # print("self.current_max_value", self.current_max_value)
        # print("KG", KG)
        # print("X", X, "acqX", np.array(acqX).reshape(-1), "grad", np.array(dacq_dX).reshape(-1))

        return np.array(acqX).reshape(-1), np.array(dacq_dX).reshape(-1)

    def _marginal_acq(self, X):
        # print("_marginal_acq")
        """
        """
        """
        """
        marginal_acqX = np.zeros((X.shape[0],1))
        if self.MCMC:
            n_h = 10 # Number of GP hyperparameters samples.
            gp_hyperparameters_samples_obj = self.model.get_hyperparameters_samples(n_h)
            gp_hyperparameters_samples_const = self.model_c.get_hyperparameters_samples(n_h)
        else:
            n_h = 1
            gp_hyperparameters_samples_obj  = self.model.get_model_parameters()

            gp_hyperparameters_samples_const  = self.model_c.get_model_parameters()
            if len(gp_hyperparameters_samples_const)>1:
                gp_hyperparameters_samples_const = [gp_hyperparameters_samples_const]
        # print("gp_hyperparameters_samples_obj", gp_hyperparameters_samples_obj)
        # print("gp_hyperparameters_samples_const", gp_hyperparameters_samples_const)
        n_z= self.nz # Number of samples of Z.

        Z_samples_obj = self.Z_samples_obj
        Z_samples_const = self.Z_samples_const

        # print("nz", n_z)
        # print("Z_samples_obj",Z_samples_obj)
        # print("Z_samples_const",Z_samples_const)

        for h in range(n_h):

            self.model.set_hyperparameters(gp_hyperparameters_samples_obj[h])
            self.model_c.set_hyperparameters(gp_hyperparameters_samples_const[h])
            # print("after for loop self.model_c.get_model_parameters()[0]",self.model_c.get_model_parameters())
            varX_obj = self.model.posterior_variance(X)
            varX_c = self.model_c.posterior_variance(X)


            for i in range(0,len(X)):
                x = np.atleast_2d(X[i])

                self.model.partial_precomputation_for_covariance(x)
                self.model.partial_precomputation_for_covariance_gradient(x)

                self.model_c.partial_precomputation_for_covariance(x)
                self.model_c.partial_precomputation_for_covariance_gradient(x)

                # Precompute aux1 and aux2 for computational efficiency.
                aux_obj = np.reciprocal(varX_obj[:,i])
                aux_c = np.reciprocal(varX_c[:,i])

                # print("x",x)

                statistics_precision = []
                for z in range(n_z):
                    Z_obj = Z_samples_obj[z]
                    Z_const = Z_samples_const[z]


                    # inner function of maKG acquisition function.
                    def inner_func(X_inner):

                        X_inner = np.atleast_2d(X_inner)

                        grad_obj = gradients(x_new=x, model= self.model, Z = Z_obj, aux=aux_obj, X_inner=X_inner)#, test_samples = self.test_samples)
                        mu_xnew = grad_obj.compute_value_mu_xnew(x=X_inner)


                        grad_c = gradients(x_new=x, model=self.model_c, Z=Z_const, aux=aux_c, X_inner=X_inner)#, test_samples = initial_design('random', self.space, 1000))
                        Fz =  grad_c.compute_probability_feasibility_multi_gp(x = X_inner, l=0)

                        func_val = np.array(mu_xnew * Fz)

                        return -func_val
                    # inner function of maKG acquisition function with its gradient.

                    def inner_func_with_gradient(X_inner):
                        # print("inner_func_with_gradient")


                        X_inner = np.atleast_2d(X_inner)
                        grad_obj = gradients(x_new=x, model= self.model, Z = Z_obj, aux=aux_obj, X_inner=X_inner, precompute_grad =True)
                        mu_xnew = grad_obj.compute_value_mu_xnew(x=X_inner)
                        grad_mu_xnew = grad_obj.compute_gradient_mu_xnew(x = X_inner)


                        grad_c = gradients(x_new=x, model=self.model_c, Z=Z_const, aux=aux_c, X_inner=X_inner, precompute_grad =True)
                        Fz , grad_Fz =  grad_c.compute_probability_feasibility_multi_gp(x = X_inner, l=0, gradient_flag=True)
                        func_val = np.array(mu_xnew *Fz)


                        func_grad_val = grad_c.product_gradient_rule(func = np.array([func_val.reshape(-1), Fz.reshape(-1)]), grad = np.array([grad_mu_xnew.reshape(-1) ,grad_Fz.reshape(-1) ]))
                        return -func_val, -func_grad_val

                    inner_opt_x, inner_opt_val =self.optimizer.optimize_inner_func(f =inner_func, f_df=inner_func_with_gradient)
                    statistics_precision.append(inner_opt_val)
                    marginal_acqX[i, 0] -= inner_opt_val

        marginal_acqX = marginal_acqX/(n_z*n_h)
        return marginal_acqX

    def run_inner_func_vals(self,f ):
        X = initial_design('random', self.space, 1000)
        f_val = []
        for x in X:
            x = x.reshape(1,-1)
            f_val.append(np.array(f(x)).reshape(-1))
        print("f_val min max", np.min(f_val), np.max(f_val))
        plt.scatter(X[:,0],X[:,1], c=np.array(f_val).reshape(-1))
        plt.show()

    def gradient_sanity_check_2D(self,  f, grad_f, delta= 1e-4):

        init_design = initial_design('random', self.space, 1000)

        fixed_dim =1
        variable_dim = 0
        v1 = np.repeat(np.array(init_design[0, fixed_dim]), len(init_design[:, 1])).reshape(-1, 1)
        v2 = init_design[:, 1].reshape(-1, 1)
        X = np.concatenate((v1, v2), axis=1)

        numerical_grad = []
        analytical_grad = []
        func_val = []
        dim = X.shape[1]
        delta_matrix = np.identity(dim)
        for x in X:
            x = x.reshape(1,-1)
            f_val = np.array(f(x)).reshape(-1)
            f_delta = []
            for i in range(dim):
                one_side = np.array(f(x + delta_matrix[i]*delta)).reshape(-1)
                two_side = np.array(f(x - delta_matrix[i]*delta)).reshape(-1)
                f_delta.append(one_side - two_side)

            func_val.append(f_val)
            f_delta = np.array(f_delta).reshape(-1)
            numerical_grad.append(np.array(f_delta/(2*delta)).reshape(-1))
            analytical_grad.append(grad_f(x).reshape(-1))

        func_val = np.array(func_val)
        numerical_grad = np.array(numerical_grad)
        analytical_grad = np.array(analytical_grad)

        dif = np.abs(numerical_grad - analytical_grad)
        print("dif mean", np.mean(dif,axis=0), "dif min", np.min(dif,axis=0), "dif max", np.max(dif,axis=0))

        #PLOTS
        fig, (ax1, ax2, ax3,ax4) = plt.subplots(4)

        ax1.scatter(v2.reshape(-1), np.array(func_val).reshape(-1), label="actual function")
        ax1.legend()
        ax2.scatter(v2.reshape(-1),np.array(numerical_grad[:,variable_dim]).reshape(-1), label="numerical")
        ax2.legend()
        ax3.scatter(v2.reshape(-1),np.array(analytical_grad[:,variable_dim]).reshape(-1), label="analytical")
        ax3.legend()
        ax4.scatter(v2.reshape(-1), dif[:,variable_dim].reshape(-1), label="errors")
        ax4.legend()
        plt.show()

    def _marginal_acq_with_gradient(self, X):

        """
        """
        marginal_acqX = np.zeros((X.shape[0],1))
        marginal_dacq_dX = np.zeros((X.shape[0], X.shape[1], 1))
        if self.MCMC:
            n_h = 10 # Number of GP hyperparameters samples.
            gp_hyperparameters_samples_obj = self.model.get_hyperparameters_samples(n_h)
            gp_hyperparameters_samples_const = self.model_c.get_hyperparameters_samples(n_h)
        else:
            n_h = 1
            gp_hyperparameters_samples_obj  = self.model.get_model_parameters()

            gp_hyperparameters_samples_const  = self.model_c.get_model_parameters()
            if len(gp_hyperparameters_samples_const)>1:
                gp_hyperparameters_samples_const = [gp_hyperparameters_samples_const]

       # Number of samples of Z.
        n_z = self.nz
        Z_samples_obj = self.Z_samples_obj
        Z_samples_const = self.Z_samples_const

        # print("nz", n_z)
        # print("Z_samples_obj", Z_samples_obj)
        # print("Z_samples_const", Z_samples_const)
        for h in range(n_h):

            self.model.set_hyperparameters(gp_hyperparameters_samples_obj[h])
            self.model_c.set_hyperparameters(gp_hyperparameters_samples_const[h])
            # print("after for loop self.model_c.get_model_parameters()[0]",self.model_c.get_model_parameters())

            varX_obj = self.model.posterior_variance(X)
            dvar_obj_dX = self.model.posterior_variance_gradient(X)

            varX_c = self.model_c.posterior_variance(X)
            dvar_c_dX = self.model_c.posterior_variance_gradient(X)


            for i in range(0,len(X)):
                x = np.atleast_2d(X[i])

                self.model.partial_precomputation_for_covariance(x)
                self.model.partial_precomputation_for_covariance_gradient(x)

                self.model_c.partial_precomputation_for_covariance(x)
                self.model_c.partial_precomputation_for_covariance_gradient(x)

                # Precompute aux1 and aux2 for computational efficiency.
                aux_obj = np.reciprocal(varX_obj[:, i])
                aux2_obj = np.square(np.reciprocal(varX_obj[:, i]))

                aux_c = np.reciprocal(varX_c[:, i])
                aux2_c = np.square(np.reciprocal(varX_c[:, i]))

                # print("x",x)

                statistics_precision = []
                for z in range(n_z):
                    Z_obj = Z_samples_obj[z]
                    Z_const = Z_samples_const[z]


                    # inner function of maKG acquisition function.
                    def inner_func(X_inner):
                        X_inner = np.atleast_2d(X_inner)

                        grad_obj = gradients(x_new=x, model= self.model, Z = Z_obj, aux=aux_obj, X_inner=X_inner)
                        mu_xnew = grad_obj.compute_value_mu_xnew(x=X_inner)

                        grad_c = gradients(x_new=x, model=self.model_c, Z=Z_const, aux=aux_c , X_inner=X_inner)
                        Fz =  grad_c.compute_probability_feasibility_multi_gp(x = X_inner, l=0)
                        func_val = np.array(mu_xnew * Fz)

                        return -func_val
                    # inner function of maKG acquisition function with its gradient.
                    def inner_func_with_gradient(X_inner):
                        # print("inner_func_with_gradient")
                        X_inner = np.atleast_2d(X_inner)

                        grad_obj = gradients(x_new=x, model= self.model, Z = Z_obj, aux=aux_obj, X_inner=X_inner, precompute_grad =True)

                        mu_xnew = grad_obj.compute_value_mu_xnew(x=X_inner)
                        grad_mu_xnew = grad_obj.compute_gradient_mu_xnew(x = X_inner)

                        grad_c = gradients(x_new=x, model=self.model_c, Z=Z_const, aux=aux_c, X_inner=X_inner, precompute_grad = True)

                        Fz, grad_Fz = grad_c.compute_probability_feasibility_multi_gp(x=X_inner, l=0,
                                                                                      gradient_flag=True)
                        func_val =  np.array(mu_xnew *Fz)
                        func_grad_val =  grad_c.product_gradient_rule(func = np.array([func_val.reshape(-1), Fz.reshape(-1)]), grad = np.array([grad_mu_xnew.reshape(-1) ,grad_Fz.reshape(-1) ]))

                        return -func_val, -func_grad_val


                    x_opt, opt_val = self.optimizer.optimize_inner_func(f =inner_func, f_df=inner_func_with_gradient)

                    marginal_acqX[i,0] -= opt_val
                    x_opt = np.atleast_2d(x_opt)

                    grad_obj = gradients(x_new=x, model=self.model, Z=Z_obj, xopt =x_opt, aux=aux_obj, aux2=aux2_obj, varX=varX_obj[:,i], dvar_dX=dvar_obj_dX[:,i,:])# , test_samples= initial_design('random', self.space, 1000) )

                    mu_xopt = grad_obj.compute_value_mu_xopt(x=x_opt)
                    grad_mu_xopt = grad_obj.compute_grad_mu_xopt(x_opt)

                    grad_c = gradients(x_new=x, model=self.model_c, Z=Z_const, xopt =x_opt, aux=aux_c, aux2=aux2_c, varX=varX_c[:,i], dvar_dX=dvar_c_dX[:,i,:])#, test_samples= initial_design('random', self.space, 1000))
                    Fz_xopt , grad_Fz_xopt = grad_c.compute_probability_feasibility_multi_gp_xopt(xopt = x_opt, gradient_flag=True)

                    dacq_dX = np.array(mu_xopt).reshape(-1)*np.array(grad_Fz_xopt).reshape(-1) +  np.array(Fz_xopt).reshape(-1) * np.array(grad_mu_xopt).reshape(-1)


                    marginal_dacq_dX[i, :, 0] += dacq_dX

        marginal_acqX = marginal_acqX/(n_h*n_z)
        marginal_dacq_dX = marginal_dacq_dX/(n_h*n_z)
        return marginal_acqX, marginal_dacq_dX





