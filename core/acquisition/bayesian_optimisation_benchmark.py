# Copyright (c) 2018, Raul Astudillo

import GPyOpt
import collections
import numpy as np
import pygmo as pg
import time
import csv
import matplotlib.pyplot as plt
from pyDOE import lhs
import time
from GPyOpt.DM.Decision_Maker import DM
from GPyOpt.DM.inference import inf
from GPyOpt.experiment_design import initial_design
from GPyOpt.util.general import best_value
from GPyOpt.util.duplicate_manager import DuplicateManager
from GPyOpt.core.errors import InvalidConfigError
from GPyOpt.core.task.cost import CostModel
from GPyOpt.optimization.acquisition_optimizer import ContextManager
from scipy.stats import norm
try:
    from GPyOpt.plotting.plots_bo import plot_acquisition, plot_convergence
except:
    pass


class BO(object):
    """
    Runner of the multi-attribute Bayesian optimization loop. This class wraps the optimization loop around the different handlers.
    :param model: GPyOpt model class.
    :param space: GPyOpt space class.
    :param objective: GPyOpt objective class.
    :param acquisition: GPyOpt acquisition class.
    :param evaluator: GPyOpt evaluator class.
    :param X_init: 2d numpy array containing the initial inputs (one per row) of the model.
    :param Y_init: 2d numpy array containing the initial outputs (one per row) of the model.
    :param cost: GPyOpt cost class (default, none).
    :param normalize_Y: whether to normalize the outputs before performing any optimization (default, True).
    :param model_update_interval: interval of collected observations after which the model is updated (default, 1).
    :param de_duplication: GPyOpt DuplicateManager class. Avoids re-evaluating the objective at previous, pending or infeasible locations (default, False).
    """


    def __init__(self, model, model_c,space, objective, constraint, acquisition, evaluator, X_init ,expensive=False,  Y_init=None, C_init=None, cost = None, normalize_Y = False, model_update_interval = 1, true_preference = 0.5):
        self.true_preference = true_preference
        self.model_c = model_c
        self.model = model
        self.space = space
        self.objective = objective
        self.constraint = constraint
        self.acquisition = acquisition
        self.utility = acquisition.utility
        self.evaluator = evaluator
        self.normalize_Y = normalize_Y
        self.model_update_interval = model_update_interval
        self.X = X_init
        self.Y = Y_init
        self.C = C_init
        self.cost = CostModel(cost)
        self.model_parameters_iterations = None
        self.expensive = expensive

    def suggest_next_locations(self, context = None, pending_X = None, ignored_X = None):
        """
        Run a single optimization step and return the next locations to evaluate the objective.
        Number of suggested locations equals to batch_size.

        :param context: fixes specified variables to a particular context (values) for the optimization run (default, None).
        :param pending_X: matrix of input configurations that are in a pending state (i.e., do not have an evaluation yet) (default, None).
        :param ignored_X: matrix of input configurations that the user black-lists, i.e., those configurations will not be suggested again (default, None).
        """
        self.model_parameters_iterations = None
        self.num_acquisitions = 0
        self.context = context
        self._update_model(self.normalization_type)

        suggested_locations = self._compute_next_evaluations(pending_zipped_X = pending_X, ignored_zipped_X = ignored_X)

        return suggested_locations
    
    
    # def _value_so_far(self):
    #     """
    #     Computes E_n[U(f(x_max))|f], where U is the utility function, f is the true underlying ojective function and x_max = argmax E_n[U(f(x))|U]. See
    #     function _marginal_max_value_so_far below.
    #     """
    #
    #     output = 0
    #     support = self.utility.parameter_dist.support
    #     utility_dist = self.utility.parameter_dist.prob_dist
    #
    #     a = np.reshape(self.objective.evaluate(self._marginal_max_value_so_far())[0],(self.objective.output_dim,))
    #
    #     output += self.utility.eval_func(support[i],a)*utility_dist[i]
    #     #print(output)
    #     return output
    #
    #
    # def _marginal_max_value_so_far(self):
    #     """
    #     Computes argmax E_n[U(f(x))|U] (The abuse of notation can be misleading; note that the expectation is with
    #     respect to the posterior distribution on f after n evaluations)
    #     """
    #
    #     def val_func(X):
    #         X = np.atleast_2d(X)
    #         muX = self.model.posterior_mean(X)
    #         return muX
    #
    #     def val_func_with_gradient(X):
    #         X = np.atleast_2d(X)
    #         muX = self.model.posterior_mean(X)
    #         dmu_dX = self.model.posterior_mean_gradient(X)
    #         valX = np.reshape( muX, (X.shape[0],1))
    #         dval_dX =  dmu_dX
    #         return -valX, -dval_dX
    #
    #
    #     argmax = self.acquisition.optimizer.optimize_inner_func(f=val_func, f_df=val_func_with_gradient)[0]
    #     return argmax
    #
    #
    def run_optimization(self, max_iter = 1, max_time = np.inf,  eps = 1e-8, context = None, verbosity=False, evaluations_file = None):
        """
        Runs Bayesian Optimization for a number 'max_iter' of iterations (after the initial exploration data)

        :param max_iter: exploration horizon, or number of acquisitions. If nothing is provided optimizes the current acquisition.
        :param max_time: maximum exploration horizon in seconds.
        :param eps: minimum distance between two consecutive x's to keep running the model.
        :param context: fixes specified variables to a particular context (values) for the optimization run (default, None).
        :param verbosity: flag to print the optimization results after each iteration (default, False).
        :param evaluations_file: filename of the file where the evaluated points and corresponding evaluations are saved (default, None).
        """
        self.verbosity=verbosity
        if self.objective is None:
            raise InvalidConfigError("Cannot run the optimization loop without the objective function")

        # --- Save the options to print and save the results
        self.verbosity = verbosity
        self.evaluations_file = evaluations_file
        self.context = context
    
                
        # --- Setting up stop conditions
        self.eps = eps
        if  (max_iter is None) and (max_time is None):
            self.max_iter = 0
            self.max_time = np.inf
        elif (max_iter is None) and (max_time is not None):
            self.max_iter = np.inf
            self.max_time = max_time
        elif (max_iter is not None) and (max_time is None):
            self.max_iter = max_iter
            self.max_time = np.inf
        else:
            self.max_iter = max_iter
            self.max_time = max_time

        # --- Initial function evaluation and model fitting
        if self.X is not None and self.Y is None:
            self.Y, cost_values = self.objective.evaluate(self.X)
            self.C, cost_values = self.constraint.evaluate(self.X)
            if self.cost.cost_type == 'evaluation_time':
                self.cost.update_cost_model(self.X, cost_values)
    
        #self.model.updateModel(self.X,self.Y)

        # --- Initialize iterations and running time
        self.time_zero = time.time()
        self.cum_time  = 0
        self.num_acquisitions = 0
        self.suggested_sample = self.X
        self.Y_new = self.Y
        self.Opportunity_Cost = []
        value_so_far = []

        # --- Initialize time cost of the evaluations
        print("MAIN LOOP STARTS")
        Opportunity_Cost = []
        self.true_best_stats = {"true_best":[], "mean_gp":[], "std gp":[], "pf":[], "mu_pf":[], "var_pf":[], "residual_noise":[]}
        while (self.max_iter > self.num_acquisitions ):


            self._update_model()
            self.Opportunity_Cost_caller()

            print("maKG optimizer")
            start = time.time()
            self.suggested_sample = self.optimize_final_evaluation() # self._compute_next_evaluations()
            finish = time.time()
            print("time optimisation point X", finish - start)

            if verbosity:
                ####plots
                design_plot = initial_design('random', self.space, 1000)
                ac_f = self.expected_improvement(design_plot)
                Y, _ = self.objective.evaluate(design_plot)
                C, _ = self.constraint.evaluate(design_plot)
                pf = self.probability_feasibility_multi_gp(design_plot, self.model_c).reshape(-1, 1)
                mu_f = self.model.predict(design_plot)[0]


                bool_C = np.product(np.concatenate(C,axis=1)<0,axis=1)
                func_val = Y * bool_C.reshape(-1,1)

                print("self.suggested_sample",self.suggested_sample)
                fig, axs = plt.subplots(2, 2)
                axs[0, 0].set_title('True Function')
                axs[0, 0].scatter(design_plot[:, 0], design_plot[:, 1], c=np.array(func_val).reshape(-1))
                axs[0, 0].scatter(self.X[:, 0], self.X[:, 1], color="red", label="sampled")
                axs[0, 0].scatter(self.suggested_sample[:, 0], self.suggested_sample[:, 1], marker = "x", color="red",
                                  label="suggested")

                axs[0, 1].set_title('approximation Acqu Function')
                axs[0, 1].scatter(design_plot[:, 0], design_plot[:, 1], c=np.array(ac_f).reshape(-1))


                axs[1,0].set_title("convergence")
                axs[1,0].plot(range(len(self.Opportunity_Cost)), np.array(self.Opportunity_Cost).reshape(-1))
                axs[1,0].set_yscale("log")

                axs[1,1].set_title("mu")
                axs[1,1].scatter(design_plot[:, 0], design_plot[:, 1], c=np.array(mu_f).reshape(-1)*np.array(pf).reshape(-1))


                plt.show()

            self.X = np.vstack((self.X,self.suggested_sample))
            # --- Evaluate *f* in X, augment Y and update cost function (if needed)
            self.evaluate_objective()

            print("X", self.X,"Y", self.Y, "C", self.C, "OC", self.Opportunity_Cost)
            # --- Update current evaluation time and function evaluations
            self.cum_time = time.time() - self.time_zero
            self.num_acquisitions += 1


        return self.X, self.Y, self.C , self.Opportunity_Cost
        # --- Print the desired result in files
        #if self.evaluations_file is not None:
            #self.save_evaluations(self.evaluations_file)

        #file = open('test_file.txt','w')
        #np.savetxt('test_file.txt',value_so_far)


    def Opportunity_Cost_caller(self):

        # design_plot = initial_design('random', self.space, 1000)
        # ac_f = self.expected_improvement(design_plot)
        # fig, axs = plt.subplots(2, 2)
        # axs[0, 0].set_title('True Function')
        # axs[0, 0].scatter(design_plot[:, 0], design_plot[:, 1], c=np.array(ac_f).reshape(-1))

        # axs[0, 0].scatter(suggested_sample[:, 0], suggested_sample[:, 1], color="red")
        # plt.show()
        # print("self.suggested_sample",suggested_sample)
        # --- Evaluate *f* in X, augment Y and update cost function (if needed)

        samples = self.X
        print("samples",samples)
        Y = self.model.posterior_mean(samples)
        # print("Y",Y)
        pf = self.probability_feasibility_multi_gp(samples, model=self.model_c)
        func_val = np.array(Y).reshape(-1) * np.array(pf).reshape(-1)

        print("Y", Y, "pf", pf)
        suggested_final_sample = samples[np.argmax(func_val)]
        suggested_final_sample = np.array(suggested_final_sample).reshape(-1)
        suggested_final_sample = np.array(suggested_final_sample).reshape(1, -1)
        print("suggested_final_sample",suggested_final_sample,"max", np.max(func_val))
        Y_true, _ = self.objective.evaluate(suggested_final_sample, true_val=True)
        C_true, _ = self.constraint.evaluate(suggested_final_sample, true_val=True)


        bool_C_true = np.product(np.concatenate(C_true, axis=1) < 0, axis=1)
        func_val_true = Y_true * bool_C_true.reshape(-1, 1)
        print("suggested_final_sample", suggested_final_sample, "func_val_true", func_val_true)

        if self.expensive:
            self.Opportunity_Cost.append(np.array(np.abs(np.max(func_val_true))).reshape(-1))
        else:
            self.true_best_value()
            optimum = np.max(np.abs(self.true_best_stats["true_best"]))
            # print("optimum", optimum)
            self.Opportunity_Cost.append(optimum - np.array(np.abs(np.max(func_val_true))).reshape(-1))
            # print("OC_i", optimum - np.array(np.abs(np.max(func_val_true))).reshape(-1))

    def optimize_final_evaluation(self):


        self.acquisition.optimizer.context_manager = ContextManager(self.space, self.context)
        out = self.acquisition.optimizer.optimize(f=self.expected_improvement, duplicate_manager=None)
        suggested_sample =  self.space.zip_inputs(out[0])
        # suggested_sample = suggested_sample.astype("int")
        print("suggested_sample",suggested_sample)
        return suggested_sample


    def expected_improvement(self, X, offset=0.0):
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
        # X = X.astype("int")
        mu = self.model.posterior_mean(X)
        sigma = self.model.posterior_variance(X, noise=False)

        sigma = np.sqrt(sigma).reshape(-1, 1)
        mu = mu.reshape(-1,1)
        # Needed for noise-based model,
        # otherwise use np.max(Y_sample).
        # See also section 2.4 in [...]
        bool_C = np.product(np.concatenate(self.C, axis=1) < 0, axis=1)
        func_val = self.Y * bool_C.reshape(-1, 1)

        mu_sample_opt = np.max(func_val) - offset

        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        pf = self.probability_feasibility_multi_gp(X,self.model_c).reshape(-1,1)

        return -(ei.reshape(-1) *pf.reshape(-1) )

    def probability_feasibility_multi_gp(self, x, model, mean=None, cov=None, grad=False, l=0):
        # print("model",model.output)
        x = np.atleast_2d(x)
        if grad == False:
            Fz = []
            for m in range(model.output_dim):
                Fz.append(self.probability_feasibility( x, model.output[m], grad, l))
            Fz = np.product(Fz,axis=0)
            return Fz
        else:
            Fz = []
            grad_Fz = []
            for m in range(model.output_dim):
                # print("model.output[m]",model.output[m])
                # print("mean[m]",mean[m])
                # print("cov[m]",cov[m])
                Fz_aux, grad_Fz_aux = self.probability_feasibility( x, model.output[m])
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
        mean = model.posterior_mean(x)
        var = model.posterior_variance(x, noise=False)
        std = np.sqrt(var).reshape(-1, 1)

        aux_var = np.reciprocal(var)
        mean = mean.reshape(-1, 1)

        norm_dist = norm(mean, std)
        fz = norm_dist.pdf(l)
        Fz = norm_dist.cdf(l)

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


    def evaluate_objective(self):
        """
        Evaluates the objective
        """
        print(1)
        print(self.suggested_sample)
        self.Y_new, cost_new = self.objective.evaluate(self.suggested_sample)
        self.C_new, C_cost_new = self.constraint.evaluate(self.suggested_sample)
        self.cost.update_cost_model(self.suggested_sample, cost_new)   
        for j in range(self.objective.output_dim):
            print(self.Y_new[j])
            self.Y[j] = np.vstack((self.Y[j],self.Y_new[j]))

        for k in range(self.constraint.output_dim):
            print(self.C_new[k])
            self.C[k] = np.vstack((self.C[k],self.C_new[k]))



    def compute_current_best(self):
        current_acqX = self.acquisition.current_compute_acq()
        return current_acqX


    def _distance_last_evaluations(self):
        """
        Computes the distance between the last two evaluations.
        """
        return np.sqrt(sum((self.X[self.X.shape[0]-1,:]-self.X[self.X.shape[0]-2,:])**2))


    def _compute_next_evaluations(self, pending_zipped_X=None, ignored_zipped_X=None, re_use=False):

        """
        Computes the location of the new evaluation (optimizes the acquisition in the standard case).
        :param pending_zipped_X: matrix of input configurations that are in a pending state (i.e., do not have an evaluation yet).
        :param ignored_zipped_X: matrix of input configurations that the user black-lists, i.e., those configurations will not be suggested again.
        :return:
        """
        ## --- Update the context if any

        self.acquisition.optimizer.context_manager = ContextManager(self.space, self.context)

        aux_var = self.evaluator.compute_batch(duplicate_manager=None, re_use=re_use)
        ### We zip the value in case there are categorical variables
        return self.space.zip_inputs(aux_var[0])
        #return initial_design('random', self.space, 1)

    def _update_model(self):
        """
        Updates the model (when more than one observation is available) and saves the parameters (if available).
        """
        if (self.num_acquisitions%self.model_update_interval)==0:

            ### --- input that goes into the model (is unziped in case there are categorical variables)
            X_inmodel = self.space.unzip_inputs(self.X)
            Y_inmodel = list(self.Y)
            C_inmodel = list(self.C)

            print("X", X_inmodel,len(X_inmodel) ,"Y",Y_inmodel, len(Y_inmodel))
            self.model.updateModel(X_inmodel, Y_inmodel)
            self.model_c.updateModel(X_inmodel, C_inmodel)
        ### --- Save parameters of the model
        #self._save_model_parameter_values()


    def get_evaluations(self):
        return self.X.copy(), self.Y.copy()

    def true_best_value(self):
        from scipy.optimize import minimize

        X = initial_design('random', self.space, 1000)

        fval = self.func_val(X)

        anchor_point = np.array(X[np.argmin(fval)]).reshape(-1)
        anchor_point = anchor_point.reshape(1, -1)
        print("anchor_point",anchor_point)
        best_design = minimize(self.func_val, anchor_point, method='Nelder-Mead', tol=1e-8).x

        self.true_best_stats["true_best"].append(self.func_val(best_design))
        self.true_best_stats["mean_gp"].append(self.model.posterior_mean(best_design))
        self.true_best_stats["std gp"].append(self.model.posterior_variance(best_design, noise=False))
        self.true_best_stats["pf"].append(self.probability_feasibility_multi_gp(best_design,self.model_c).reshape(-1,1))
        mean = self.model_c.posterior_mean(best_design)
        var = self.model_c.posterior_variance(best_design, noise=False)
        residual_noise = self.model_c.posterior_variance(self.X[1], noise=False)
        self.true_best_stats["mu_pf"].append(mean)
        self.true_best_stats["var_pf"].append(var)
        self.true_best_stats["residual_noise"].append(residual_noise)

        if False:
            fig, axs = plt.subplots(3, 2)
            N = len(np.array(self.true_best_stats["std gp"]).reshape(-1))
            GAP = np.array(np.abs(np.abs(self.true_best_stats["true_best"]).reshape(-1) - np.abs(self.true_best_stats["mean_gp"]).reshape(-1))).reshape(-1)
            print("GAP len", len(GAP))
            print("N",N)
            axs[0, 0].set_title('GAP')
            axs[0, 0].plot(range(N),GAP)
            axs[0, 0].set_yscale("log")

            axs[0, 1].set_title('VAR')
            axs[0, 1].plot(range(N),np.array(self.true_best_stats["std gp"]).reshape(-1))
            axs[0, 1].set_yscale("log")

            axs[1, 0].set_title("PF")
            axs[1, 0].plot(range(N),np.array(self.true_best_stats["pf"]).reshape(-1))

            axs[1, 1].set_title("mu_PF")
            axs[1, 1].plot(range(N),np.abs(np.array(self.true_best_stats["mu_pf"]).reshape(-1)))
            axs[1, 1].set_yscale("log")

            axs[2, 1].set_title("std_PF")
            axs[2, 1].plot(range(N),np.sqrt(np.array(self.true_best_stats["var_pf"]).reshape(-1)))
            axs[2, 1].set_yscale("log")

            axs[2, 0].set_title("Irreducible noise")
            axs[2, 0].plot(range(N), np.sqrt(np.array(self.true_best_stats["residual_noise"]).reshape(-1)))
            axs[2, 0].set_yscale("log")

            plt.show()

    def func_val(self, x):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        Y,_ = self.objective.evaluate(x, true_val=True)
        C,_ = self.constraint.evaluate(x, true_val=True)
        Y = np.array(Y).reshape(-1)
        out = Y.reshape(-1)* np.product(np.concatenate(C, axis=1) < 0, axis=1).reshape(-1)
        out = np.array(out).reshape(-1)
        return -out