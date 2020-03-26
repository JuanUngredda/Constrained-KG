# Copyright (c) 2018, Raul Astudillo Marban

import numpy as np
from GPyOpt.experiment_design import initial_design
from GPyOpt.acquisitions.base import AcquisitionBase
from GPyOpt.core.task.cost import constant_cost_withGradients
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

    def __init__(self, model, space, model_c=None, nz = 10, optimizer=None, cost_withGradients=None, utility=None):
        self.optimizer = optimizer
        self.utility = utility
        self.MCMC = False
        self.nz = nz
        super(KG, self).__init__(model, space, optimizer, model_c, cost_withGradients=cost_withGradients)
        if cost_withGradients == None:
            self.cost_withGradients = constant_cost_withGradients
        else:
            print('LBC acquisition does now make sense with cost. Cost set to constant.')
            self.cost_withGradients = constant_cost_withGradients


    def _compute_acq(self, X , marg_acqX_info=False):
        """
        Computes the aquisition function
        
        :param X: set of points at which the acquisition function is evaluated. Should be a 2d array.
        """

        # print("_compute_acq")
        start = time.time()
        # print("X acq", X, "shape", X.shape)
        X =np.atleast_2d(X)

        marginal_acqX = self._marginal_acq(X)
        acqX = np.reshape(marginal_acqX, (X.shape[0],1))
        stop = time.time()
        # print("time _comp_acq", stop-start)
        return acqX

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

        Z_samples_obj = np.random.normal(size=n_z)
        Z_samples_const = np.random.normal(size=n_z)


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

                        mean_obj = self.posterior_mu_x_new(self.model, X_inner, x, Z_obj, aux_obj)
                        mean_const = self.posterior_mu_x_new(self.model_c, X_inner, x, Z_const, aux_c)
                        kernel_const = self.posterior_kernel_x_new(self.model_c, X_inner, x, aux_c,likelihood=True)
                        Fz = self.probability_feasibility_multi_gp(x, self.model_c, mean=mean_const, cov=kernel_const, grad=False, l=0)
                        func_val = mean_obj*Fz
                        return -func_val
                    # inner function of maKG acquisition function with its gradient.

                    def inner_func_with_gradient(X_inner):
                        print("inner_func_with_gradien")
                        X_inner = np.atleast_2d(X_inner)

                        muX_inner = self.model.posterior_mean(X_inner)
                        dmu_dX_inner  = self.model.posterior_mean_gradient(X_inner)
                        cov = self.model.posterior_covariance_between_points_partially_precomputed( X_inner,x)[:,:,0]

                        print("X_inner", X_inner)
                        dcov_dX_inner = self.model.posterior_covariance_gradient_partially_precomputed(X_inner,x)
                        a = muX_inner
                        da_dX_inner = dmu_dX_inner
                        b = np.sqrt(aux_obj * np.square(cov)) #np.sqrt(np.matmul(aux,np.square(cov)))
                        for k in range(X_inner.shape[1]):
                            dcov_dX_inner[:,:,k] = np.multiply(cov,dcov_dX_inner[:,:,k])
                        db_dX_inner  =  np.tensordot(aux_obj,dcov_dX_inner,axes=1)
                        db_dX_inner = np.multiply(np.reciprocal(b),db_dX_inner.T).T
                        func_val = np.reshape(a + b*Z_obj, (len(X_inner),1))
                        func_gradient = np.reshape(da_dX_inner + db_dX_inner*Z_obj, X_inner.shape)

                        mean_const = self.posterior_mu_x_new(self.model_c, X_inner, x, Z_const, aux_c)
                        kernel_const = self.posterior_kernel_x_new(self.model_c, X_inner, x, aux_c, likelihood=True)

                        #TODO: Gradients are not working properly. The gradient is computed just with the current data. The resulting gradient by any reason is always positive or negative.

                        Fz, grad_Fz = self.probability_feasibility_multi_gp( x, self.model_c, mean=mean_const, cov=kernel_const, grad=True,
                                                          l=0)

                        grad_func = self.product_gradient_rule(func = np.array([func_val, Fz]), grad = np.array([func_gradient,grad_Fz ]))

                        return np.array(func_val*Fz), np.array(grad_func), np.array(func_val) ,np.array(func_gradient), np.array(Fz),np.array( grad_Fz )

                    # x_plot = np.linspace(40.55276382,41,20)[:,None]
                    # f_x = np.zeros(x_plot.shape[0])
                    # grad_fx = np.zeros(x_plot.shape[0])
                    # func_val = np.zeros(x_plot.shape[0])
                    # func_grad = np.zeros(x_plot.shape[0])
                    # grad_Fz = np.zeros(x_plot.shape[0])
                    # Fz = np.zeros(x_plot.shape[0])
                    # mean_const = np.zeros(x_plot.shape[0])
                    # for i in range(x_plot.shape[0]):
                    #     f_x[i], grad_fx[i], func_val[i], func_grad[i], Fz[i], grad_Fz[i] = inner_func_with_gradient(x_plot[i])
                    #
                    # print("f_x shape", f_x.shape)
                    # print("grad_fx shape", grad_fx.shape)
                    # print("func_val shape", func_val.shape)
                    # print("func_grad shape",func_grad.shape)
                    # print("x_plot",x_plot.reshape(-1))
                    # print("func_val.reshape(-1)",func_val.reshape(-1))
                    # print("func_grad.reshape(-1)",func_grad.reshape(-1))
                    # print("f_x.reshape(-1)",f_x.reshape(-1))
                    # print("grad_fx.reshape(-1)",grad_fx.reshape(-1))
                    # print("grad_fx.reshape(-1)", grad_fx.reshape(-1))
                    # fig, (ax1, ax2,ax3,ax4, ax5, ax6,ax7) = plt.subplots(7)
                    # ax1.scatter(x_plot.reshape(-1), func_val.reshape(-1))
                    # ax2.scatter(x_plot.reshape(-1), func_grad.reshape(-1))
                    # ax3.scatter(x_plot.reshape(-1), mean_const.reshape(-1))
                    # ax4.scatter(x_plot.reshape(-1), Fz.reshape(-1))
                    # ax5.scatter(x_plot.reshape(-1), grad_Fz.reshape(-1))
                    # ax6.scatter(x_plot.reshape(-1), f_x.reshape(-1))
                    # ax7.scatter(x_plot.reshape(-1), grad_fx.reshape(-1))
                    # plt.show()

                    inner_opt_val =self.optimizer.optimize_inner_func(f =inner_func, f_df=None)[1]
                    statistics_precision.append(inner_opt_val)
                    marginal_acqX[i, 0] -= inner_opt_val

                    # design_plot = initial_design('random', self.space, 1000)
                    # func_val = inner_func(design_plot)
                    # # print("inner_func",func_val)
                    # print("func_val",np.min(func_val), "marginal_acqX[i,0]",inner_opt_val)
                    # fig, axs = plt.subplots(2, 2)
                    # axs[0, 0].set_title('True Function')
                    # axs[0, 0].scatter(design_plot[:, 0], design_plot[:, 1], c=np.array(func_val).reshape(-1))
                    #
                    # plt.show()

                #print("x", x, "marginal_acqX[i,0]",marginal_acqX[i,0]/(n_z*n_h))
            # print("x", x, "mean precision",np.mean(statistics_precision),"std precision", np.std(statistics_precision), "MSE", 1.95*np.std(statistics_precision)/np.sqrt(n_z))
        marginal_acqX = marginal_acqX/(n_z*n_h)
        return marginal_acqX


    # def _compute_acq_withGradients(self, X):
    #     """
    #     """
    #     X =np.atleast_2d(X)
    #     # Compute marginal aquisition function and its gradient for every value of the utility function's parameters samples,
    #     acqX, dacq_dX = self._marginal_acq_with_gradient(X)
    #     acqX = np.reshape(acqX,(X.shape[0], 1))
    #     dacq_dX = np.reshape(dacq_dX, X.shape)
    #     return acqX, dacq_dX

    def product_gradient_rule(self, func, grad):

        func = func + np.array(1e-10)
        recip = np.reciprocal(func)
        prod = np.product(func)
        vect1 = prod*recip
        vect1 = vect1.reshape(-1)
        vect2 = grad.reshape(-1)
        grad = np.dot(vect1, vect2)
        return grad

    def posterior_mu_x_new(self, model, X_inner, x_new, Z, aux):
        muX_inner = model.posterior_mean(X_inner)
        cov = model.posterior_covariance_between_points_partially_precomputed(X_inner, x_new)[:, :, 0]
        func_val = []
        for j in range(muX_inner.shape[0]):
            a = muX_inner[j]
            # print("aux",aux)
            # print("cov",cov)
            # print("aux[j]",aux[j])
            # print("cov[j]",cov[j])
            b = np.sqrt(aux[j] * np.square(cov[j]))
            func_val.append(np.reshape(a + b * Z, (len(X_inner), 1)))
        return func_val

    def posterior_kernel_x_new(self, model, X_inner, x, aux, likelihood=None):
        cov = model.posterior_covariance_between_points_partially_precomputed(X_inner, x)[:, :, 0]
        kernel_current = model.posterior_covariance_between_points(X_inner,X_inner)
        func_val = []
        for j in range(cov.shape[0]):
            b = np.sqrt(aux[j] * np.square(cov[j]))
            if likelihood != None:
                likelihood_noise = model.get_model_parameters()[j][:,-1]
                kernel_current_lik = kernel_current[j] + np.identity(kernel_current[j].shape[0])*likelihood_noise
            b = b[np.newaxis]
            kernel_new_x = kernel_current_lik - np.dot(np.transpose(b),b)
            func_val.append(kernel_new_x)
        return func_val

    def probability_feasibility_multi_gp(self, x, model, mean=None, cov=None, grad=False, l=0):
        # print("model",model.output)
        if grad == False:
            Fz = []
            for m in range(model.output_dim):
                Fz.append(self.probability_feasibility( x, model.output[m], mean[m], cov[m], grad, l))
            Fz = np.product(Fz,axis=0)
            return Fz
        else:
            Fz = []
            grad_Fz = []
            for m in range(model.output_dim):
                # print("model.output[m]",model.output[m])
                # print("mean[m]",mean[m])
                # print("cov[m]",cov[m])
                Fz_aux, grad_Fz_aux = self.probability_feasibility( x, model.output[m], mean[m], cov[m], grad, l)
                Fz.append(Fz_aux)
                grad_Fz.append(grad_Fz_aux)

            # print("np.array(Fz)", np.array(Fz), "grad_Fz", np.array(grad_Fz))
            grad_Fz = self.product_gradient_rule(func = np.array(Fz), grad = np.array(grad_Fz))
            # print("output grad_Fz", grad_Fz)

            Fz = np.product(Fz, axis=0)
            return Fz, grad_Fz

    def probability_feasibility(self, x, model, mean=None, cov=None, grad=False, l=0):

        model = model.model
        # kern = model.kern
        # X = model.X

        if (mean is None) and (cov is None):
            mean = model.posterior_mean(x)
            cov = model.posterior_covariance_between_points_partially_precomputed(x, x)[:, :, 0]

        var = np.diag(cov).reshape(-1, 1)
        std = np.sqrt(var).reshape(-1, 1)

        aux_var = np.reciprocal(var)
        mean = mean.reshape(-1, 1)

        norm = scipy.stats.norm(mean, std)
        fz = norm.pdf(l)
        Fz = norm.cdf(l)

        if grad == True:
            grad_mean , grad_var = model.predictive_gradients(x)
            grad_std = (1/2.0)*grad_var

            # cov = kern.K(X, X) + np.eye(X.shape[0]) * 1e-3
            # L = scipy.linalg.cholesky(cov, lower=True)
            # u = scipy.linalg.solve(L, np.eye(X.shape[0]))
            # Ainv = scipy.linalg.solve(L.T, u)

            dims = range(x.shape[1])
            grad_Fz = []

            for d in dims:
                # K1 = np.diag(np.dot(np.dot(kern.dK_dX(x, X, d), Ainv), kern.dK_dX2(X, x, d)))
                # K2 = np.diag(kern.dK2_dXdX2(x, x, d, d))
                # var_grad = K2 - K1
                # var_grad = var_grad.reshape(-1, 1)
                grd_mean_d = grad_mean[:, d].reshape(-1, 1)
                grd_std_d = grad_std[:, d].reshape(-1, 1)
                # print("grd_mean ",grd_mean_d, "var_grad ",grd_std_d )
                # print("std",std)
                # print("aux_var", aux_var)
                # print("fz",fz)
                grad_Fz.append(fz * aux_var * (mean * grd_std_d - grd_mean_d * std))
            grad_Fz = np.stack(grad_Fz, axis=1)
            return Fz.reshape(-1, 1), grad_Fz[:, :, 0]
        else:
            return Fz.reshape(-1, 1)


    # def _marginal_acq_with_gradient(self, X):
    #
    #     """
    #     """
    #     marginal_acqX = np.zeros((X.shape[0],1))
    #     marginal_dacq_dX =  np.zeros((X.shape[0],X.shape[1],1))
    #     if self.MCMC:
    #         n_h = 10 # Number of GP hyperparameters samples.
    #         gp_hyperparameters_samples_obj = self.model.get_hyperparameters_samples(n_h)
    #         gp_hyperparameters_samples_const = self.model_c.get_hyperparameters_samples(n_h)
    #     else:
    #         n_h = 1
    #         gp_hyperparameters_samples_obj  = self.model.get_model_parameters()[0]
    #         gp_hyperparameters_samples_const  = self.model_c.get_model_parameters()[0]
    #
    #     n_z= 5 # Number of samples of Z.
    #     Z_samples_obj = np.random.normal(size=n_z)
    #     Z_samples_const = np.random.normal(size=n_z)
    #
    #     for h in range(n_h):
    #         self.model.set_hyperparameters(gp_hyperparameters_samples_obj[h])
    #         self.model_c.set_hyperparameters(gp_hyperparameters_samples_const[h])
    #
    #         varX_obj = self.model.posterior_variance(X)
    #         dvar_obj_dX = self.model.posterior_variance_gradient(X)
    #
    #         varX_c = self.model_c.posterior_variance(X)
    #         dvar_c_dX = self.model_c.posterior_variance_gradient(X)
    #
    #         for i in range(0,len(X)):
    #             x = np.atleast_2d(X[i])
    #             self.model.partial_precomputation_for_covariance(x)
    #             self.model.partial_precomputation_for_covariance_gradient(x)
    #
    #             self.model_c.partial_precomputation_for_covariance(x)
    #             self.model_c.partial_precomputation_for_covariance_gradient(x)
    #
    #             # Precompute aux1 and aux2 for computational efficiency.
    #             aux_obj = np.reciprocal(varX_obj[:,i])
    #             aux2_obj = np.square(np.reciprocal(varX_obj[:,i]))
    #
    #             aux_c = np.reciprocal(varX_c[:,i])
    #             aux2_c = np.square(np.reciprocal(varX_c[:,i]))
    #
    #             for z in range(n_z):
    #                 Z_obj = Z_samples_obj[z]
    #                 Z_const = Z_samples_const[z]
    #
    #                 # inner function of maKG acquisition function.
    #                 def inner_func(X_inner):
    #
    #                     X_inner = np.atleast_2d(X_inner)
    #                     mean_obj = self.posterior_mu_x_new(self.model, X_inner, Z_obj, aux_obj)
    #                     mean_const = self.posterior_mu_x_new(self.model_c, X_inner, Z_const, aux_c)
    #                     kernel_const = self.posterior_kernel_x_new(self.model_c, X_inner, x, likelihood=True)
    #
    #                     print("mean_const",mean_const.shape)
    #                     print("kernel_const", kernel_const.shape)
    #
    #                     Fz = self.probability_feasibility(x, self.model_c, mean=mean_const, cov=kernel_const, grad=False, l=0)
    #                     print("Fz",Fz.shape)
    #                     return -(mean_obj*Fz)
    #                 # inner function of maKG acquisition function with its gradient.
    #
    #                 def inner_func_with_gradient(X_inner):
    #
    #                     X_inner = np.atleast_2d(X_inner)
    #                     muX_inner = self.model.posterior_mean(X_inner)
    #                     dmu_dX_inner  = self.model.posterior_mean_gradient(X_inner)
    #                     cov = self.model.posterior_covariance_between_points_partially_precomputed( X_inner,x)[:,:,0]
    #                     dcov_dX_inner = self.model.posterior_covariance_gradient_partially_precomputed(X_inner,x)
    #                     a = muX_inner
    #                     da_dX_inner = dmu_dX_inner
    #                     b = np.sqrt(np.matmul(aux,np.square(cov)))
    #                     for k in range(X_inner.shape[1]):
    #                         dcov_dX_inner[:,:,k] = np.multiply(cov,dcov_dX_inner[:,:,k])
    #                     db_dX_inner  =  np.tensordot(aux,dcov_dX_inner,axes=1)
    #                     db_dX_inner = np.multiply(np.reciprocal(b),db_dX_inner.T).T
    #                     func_val = np.reshape(a + b*Z, (len(X_inner),1))
    #                     func_gradient = np.reshape(da_dX_inner + db_dX_inner*Z, X_inner.shape)
    #
    #                     mean_const = self.posterior_mu_x_new(self.model_c, X_inner, Z_const, aux_c)
    #                     kernel_const = self.posterior_kernel_x_new(self.model_c, X_inner, x, likelihood=True)
    #                     Fz, grad_Fz = self.probability_feasibility( x, self.model_c, mean=mean_const, cov=kernel_const, grad=True,
    #                                                       l=0)
    #
    #                     grad_func = grad_Fz*func_val + Fz*func_gradient
    #
    #                     return -(func_val*Fz), -grad_func
    #
    #                 x_opt, opt_val = self.optimizer.optimize_inner_func(f =inner_func, f_df=inner_func_with_gradient)
    #                # print("opt_val",opt_val)
    #                 marginal_acqX[i,0] -= opt_val
    #                 x_opt = np.atleast_2d(x_opt)
    #                 cov_opt = self.model.posterior_covariance_between_points_partially_precomputed(x_opt,x)[:,0,0]
    #                 dcov_opt_dx = self.model.posterior_covariance_gradient(x,x_opt)[:,0,:]
    #                 b = np.sqrt(np.dot(aux_obj,np.square(cov_opt)))
    #                 marginal_dacq_dX[i,:,0] = 0.5*Z*np.reciprocal(b)*np.matmul(aux2_obj,(2*np.multiply(varX[:,i]*cov_opt,dcov_opt_dx.T) - np.multiply(np.square(cov_opt),dvar_dX[:,i,:].T)).T)
    #
    #     marginal_acqX = marginal_acqX/(n_h*n_z)
    #     marginal_dacq_dX = marginal_dacq_dX/(n_h*n_z)
    #     return marginal_acqX, marginal_dacq_dX





