# Copyright (c) 2018, Raul Astudillo

import GPyOpt
import collections
import numpy as np
#import pygmo as pg
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
import pandas as pd
import os
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


    def __init__(self, model, model_c,space, objective, constraint, acquisition, evaluator, X_init ,  tag_last_evaluation  =True,expensive=False,Y_init=None, C_init=None, cost = None, normalize_Y = False, model_update_interval = 1, deterministic=True,true_preference = 0.5):
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
        self.deterministic = deterministic
        self.cost = CostModel(cost)
        self.model_parameters_iterations = None
        self.expensive = expensive
        self.tag_last_evaluation = tag_last_evaluation
        try:
            if acquisition.name == "Constrained_Thompson_Sampling":
                self.sample_from_acq = True

            else:
                self.sample_from_acq = False

        except:
            print("name of acquisition function wasnt provided")
            self.sample_from_acq = False


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

        suggested_locations = self._compute_next_evaluations(pending_zipped_X=pending_X, ignored_zipped_X=ignored_X)
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
    def run_optimization(self, max_iter = 1, max_time = np.inf,  eps = 1e-8, context = None, verbosity=False, path = None,KG_dynamic_optimisation=False, evaluations_file = None):
        """
        Runs Bayesian Optimization for a number 'max_iter' of iterations (after the initial exploration data)

        :param max_iter: exploration horizon, or number of acquisitions. If nothing is provided optimizes the current acquisition.
        :param max_time: maximum exploration horizon in seconds.
        :param eps: minimum distance between two consecutive x's to keep running the model.
        :param context: fixes specified variables to a particular context (values) for the optimization run (default, None).
        :param verbosity: flag to print the optimization results after each iteration (default, False).
        :param evaluations_file: filename of the file where the evaluated points and corresponding evaluations are saved (default, None).
        """

        if self.objective is None:
            raise InvalidConfigError("Cannot run the optimization loop without the objective function")

        # --- Save the options to print and save the results
        self.KG_dynamic_optimisation = KG_dynamic_optimisation
        self.verbosity = verbosity
        self.evaluations_file = evaluations_file
        self.context = context
        self.path = path
                
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

 #        self.X = np.array([[ 1.75,       12.75      ],
 # [ 9.25       ,14.25      ],
 # [ 4.75       ,11.25      ],
 # [ 0.25       , 3.75      ],
 # [ 3.25       , 5.25      ],
 # [-4.25       , 6.75      ],
 # [ 6.25       , 8.25      ],
 # [-1.25       , 9.75      ],
 # [-2.75       , 0.75      ],
 # [ 7.75       , 2.25      ],
 # [10.         , 0.46190376],
 # [-4.07672534 ,13.18691809],
 # [ 5.57342965,  0.4728215 ],
 # [-1.47666097 , 7.4558601 ],
 # [-0.58304314 , 4.6502318 ],
 # [-2.99537612, 12.43184346],
 # [-5.        , 15.        ],
 # [-2.61100274, 15.        ],
 # [ 1.82193578,  2.39787769],
 # [-3.87814538, 11.43865536],
 # [ 1.38264341,  1.98608386],
 # [-0.20475798,  1.62051209],
 # [ 9.45116644,  0.54745162],
 # [ 9.32309544,  0.54290219],
 # [ 5.01421171,  0.94878182],
 # [ 6.71028756,  0.38835956],
 # [ 8.17895474,  1.81650718],
 # [ 2.8157139 ,  1.51718192],
 # [ 1.0956769 ,  1.10321236],
 # [-1.45610591,  5.57659565],
 # [ 4.02134336,  0.87608649],
 # [ 1.97188218,  1.13838647],
 # [ 7.90865545,  4.20979887],
 # [ 5.01051958,  0.24428875],
 # [ 1.25780231,  3.25093737],
 # [ 3.28221282,  0.32520072],
 # [ 3.2170687 ,  0.1789976 ],
 # [ 3.01849866,  0.11181718],
 # [ 3.39818085,  0.07663895],
 # [ 3.26688867,  0.05719408],
 # [ 3.26531851,  0.05256111],
 # [ 3.14626724,  0.12634154]] )

        # --- Initial function evaluation and model fitting
        if self.X is not None and self.Y is None:
            self.Y, cost_values = self.objective.evaluate(self.X)
            self.C, cost_values = self.constraint.evaluate(self.X)
            if self.cost.cost_type == 'evaluation_time':
                self.cost.update_cost_model(self.X, cost_values)
        self.n_init = self.X.shape[0]
        #self.model.updateModel(self.X,self.Y)

        # --- Initialize iterations and running time
        self.time_zero = time.time()
        self.cum_time  = 0
        self.num_acquisitions = 0
        self.suggested_sample = self.X
        self.Y_new = self.Y
        self.Opportunity_Cost = []
        self.recommended_value = []
        self.underlying_optimum = []
        value_so_far = []

        # --- Initialize time cost of the evaluations
        print("MAIN LOOP STARTS")
        Opportunity_Cost = []
        self.true_best_stats = {"true_best": [], "mean_gp": [], "std gp": [], "pf": [], "mu_pf": [], "var_pf": [],
                                "residual_noise": []}




        while (self.max_iter > self.num_acquisitions ):


            self._update_model()

            self.optimize_final_evaluation(self.KG_dynamic_optimisation)
            print("maKG optimizer")
            start = time.time()


            self.suggested_sample = self._compute_next_evaluations()

            finish = time.time()

            if verbosity:
                self.verbosity_plot_2D()

            print("self.suggested_sample",self.suggested_sample)
            print("time optimisation point X", finish - start)

            self.X = np.vstack((self.X,self.suggested_sample))
            # --- Evaluate *f* in X, augment Y and update cost function (if needed)
            self.evaluate_objective()

            # --- Update current evaluation time and function evaluations
            self.cum_time = time.time() - self.time_zero
            self.num_acquisitions += 1
            print("optimize_final_evaluation")

            X, Y, C, recommended_val, optimum, Opportunity_cost = self.X, self.Y, self.C , self.recommended_value, self.underlying_optimum, self.Opportunity_Cost
            C_bool = np.product(np.concatenate(C, axis=1) < 0, axis=1)
            data = {}

            print("C", C)
            print("np.array(Opportunity_cost).reshape(-1)", np.array(Opportunity_cost).reshape(-1))
            print("np.array(Y).reshape(-1)", np.array(Y).reshape(-1))
            print("np.array(C_bool).reshape(-1)", np.array(C_bool).reshape(-1))
            data["Opportunity_cost"] = np.concatenate((np.zeros(self.n_init), np.array(Opportunity_cost).reshape(-1)))
            data["Y"] = np.array(Y).reshape(-1)
            data["C_bool"] = np.array(C_bool).reshape(-1)
            data["recommended_val"] = np.concatenate((np.zeros(self.n_init), np.array(recommended_val).reshape(-1)))
            data["optimum"] = np.concatenate((np.zeros(self.n_init), np.array(optimum).reshape(-1)))

            gen_file = pd.DataFrame.from_dict(data)
            folder = "RESULTS"
            subfolder = self.evaluations_file
            cwd = os.getcwd()

            path =self.path
            if os.path.isdir(cwd + "/" + folder + "/" + subfolder) == False:
                os.makedirs(cwd + "/" + folder + "/" + subfolder)
            print("path", path)
            gen_file.to_csv(path_or_buf=path)
            print("self.X, self.Y, self.C , self.Opportunity_Cost",self.X, self.Y, self.C , self.Opportunity_Cost)

        return self.X, self.Y, self.C , self.recommended_value, self.underlying_optimum, self.Opportunity_Cost
        # --- Print the desired result in files
        #if self.evaluations_file is not None:
            #self.save_evaluations(self.evaluations_file)

        #file = open('test_file.txt','w')
        #np.savetxt('test_file.txt',value_so_far)

    def verbosity_plot_1D(self):
        ####plots
        print("generating plots")
        design_plot = np.linspace(0,5,250)[:,None]


        bool_C_sampled = np.product(np.concatenate(self.C, axis=1) < 0, axis=1)

        # precision = []
        # for i in range(20):
        #     kg_f = -self.acquisition._compute_acq(design_plot)
        #     precision.append(np.array(kg_f).reshape(-1))

        # print("mean precision", np.mean(precision, axis=0), "std precision",  np.std(precision, axis=0), "max precision", np.max(precision, axis=0), "min precision",np.min(precision, axis=0))
        ac_f = self.expected_improvement(design_plot)

        Y, _ = self.objective.evaluate(design_plot, true_val=True)
        C, _ = self.constraint.evaluate(design_plot)
        pf = self.probability_feasibility_multi_gp(design_plot, self.model_c).reshape(-1, 1)
        mu_f = self.model_c.predict(design_plot)[0]
        # mu_var = self.model_c.posterior_variance(design_plot, noise=True)

        # plt.plot(design_plot,  np.array(mu_f).reshape(-1))
        #
        # plt.plot(design_plot, np.array(mu_var).reshape(-1))
        #
        # print("min", np.min(np.sqrt(mu_var)))
        # norm_dist = norm(mu_f, np.sqrt(mu_var))
        #
        # Fz = norm_dist.cdf(0)
        # plt.plot(design_plot, np.array(Fz).reshape(-1))
        # plt.show()
        # raise
        bool_C = np.product(np.concatenate(C, axis=1) < 0, axis=1)
        func_val = Y * bool_C.reshape(-1, 1)

        kg_f = -self.acquisition._compute_acq(design_plot)
        design_plot = design_plot.reshape(-1)


        fig, axs = plt.subplots(3, 2)
        axs[0, 0].set_title('True Function')
        axs[0, 0].plot(design_plot, np.array(func_val).reshape(-1))
        axs[0, 0].scatter(self.X, np.array(self.Y).reshape(-1)*bool_C_sampled , color="red")
        suggested_sample_value , _= self.objective.evaluate(self.suggested_sample)
        axs[0, 0].scatter(self.suggested_sample, suggested_sample_value, marker="x", color="magenta")
        axs[0, 0].legend()

        axs[0, 1].set_title('approximation pf Function')
        axs[0, 1].plot(design_plot,  np.array(pf).reshape(-1))
        axs[0, 1].legend()

        axs[1, 0].set_title("approximation mu constraint Function")
        axs[1, 0].plot(design_plot, np.array(mu_f).reshape(-1) )

        axs[1, 0].legend()

        axs[1, 1].set_title("mu pf")
        axs[1, 1].plot(design_plot, np.array(mu_f).reshape(-1) * np.array(pf).reshape(-1))
        axs[1, 1].legend()

        best_kg_val =  -self.acquisition._compute_acq(self.suggested_sample)
        axs[2, 1].set_title('approximation kg Function')
        axs[2, 1].plot(design_plot, np.array(kg_f).reshape(-1))
        axs[2, 1].scatter(self.suggested_sample, np.array(best_kg_val ).reshape(-1), marker="x", color="magenta")
        axs[2, 1].legend()

        axs[2, 0].set_title('Opportunity Cost')
        axs[2, 0].plot(range(len(self.Opportunity_Cost)), self.Opportunity_Cost)
        axs[2, 0].set_yscale("log")

        plt.show()

    def verbosity_plot_2D(self):
        ####plots
        print("generating plots")
        design_plot = initial_design('random', self.space, 400)

        # precision = []
        # for i in range(20):
        self.acquisition.generate_random_vectors(optimize_discretization=False, optimize_random_Z=False, fixed_discretisation=design_plot)
        kg_f = self.acquisition._compute_acq(design_plot)
        print("np.min", np.min(kg_f), "max", np.max(kg_f))

        #     precision.append(np.array(kg_f).reshape(-1))

        # print("mean precision", np.mean(precision, axis=0), "std precision",  np.std(precision, axis=0), "max precision", np.max(precision, axis=0), "min precision",np.min(precision, axis=0))
        # ac_f = self.expected_improvement(design_plot)

        Y, _ = self.objective.evaluate(design_plot)
        C, _ = self.constraint.evaluate(design_plot)
        pf = self.probability_feasibility_multi_gp(design_plot, self.model_c).reshape(-1, 1)
        mu_f = self.model.predict(design_plot)[0]

        bool_C = np.product(np.concatenate(C, axis=1) < 0, axis=1)
        func_val = Y * bool_C.reshape(-1, 1)

        # kg_f = -self.acquisition._compute_acq(design_plot)
        fig, axs = plt.subplots(3, 2)
        axs[0, 0].set_title('True Function')
        axs[0, 0].scatter(design_plot[:, 0], design_plot[:, 1], c=np.array(func_val).reshape(-1))
        axs[0, 0].scatter(self.X[:, 0], self.X[:, 1], color="red", label="sampled")
        #suggested_sample_value = self.objective.evaluate(self.suggested_sample)
        axs[0, 0].scatter(self.suggested_sample[:,0], self.suggested_sample[:,1], marker="x", color="red",
                          label="suggested")
        axs[0, 0].legend()

        axs[0, 1].set_title('approximation Acqu Function')
        # axs[0, 1].scatter(design_plot[:,0],design_plot[:,1], c=np.array(ac_f).reshape(-1))
        # axs[0, 1].legend()
        print("self.suggested_sample",self.suggested_sample)
        import matplotlib
        axs[1, 0].set_title("KG")
        axs[1, 0].scatter(design_plot[:,0],design_plot[:,1],norm=matplotlib.colors.LogNorm(), c= np.array(kg_f).reshape(-1))
        axs[1, 1].scatter(self.suggested_sample[:, 0], self.suggested_sample[:, 1], color="red", label="KG suggested")
        axs[1, 0].legend()

        axs[1, 1].set_title("mu pf")
        axs[1, 1].scatter(design_plot[:,0],design_plot[:,1],c= np.array(mu_f).reshape(-1) * np.array(pf).reshape(-1))
        axs[1,1].scatter(self.suggested_sample[:,0], self.suggested_sample[:,1], color="red", label="KG suggested")
        axs[1,1].scatter(self.final_suggested_sample[:,0], self.final_suggested_sample[:,1], color="magenta", label="Final suggested")
        axs[1, 1].legend()


        #axs[2, 1].legend()
        # import os
        # folder = "IMAGES"
        # subfolder = "new_branin"
        # cwd = os.getcwd()
        # print("cwd", cwd)
        # time_taken = time.time()
        # path = cwd + "/" + folder + "/" + subfolder + '/im_' +str(time_taken) +str(self.X.shape[0]) + '.pdf'
        # if os.path.isdir(cwd + "/" + folder + "/" + subfolder) == False:
        #     os.makedirs(cwd + "/" + folder + "/" + subfolder)
        # plt.savefig(path)
        plt.show()

        print("self.Opportunity_Cost plot", self.Opportunity_Cost)
        plt.title('Opportunity Cost')
        plt.plot(range(len(self.Opportunity_Cost)), self.Opportunity_Cost)
        plt.yscale("log")
        plt.show()

    def optimize_final_evaluation(self, KG_dynamic_optimisation):



        # design_plot = initial_design('random', self.space, 1000)
        # ac_f = self.expected_improvement(design_plot)
        # fig, axs = plt.subplots(2, 2)
        # axs[0, 0].set_title('True Function')
        # axs[0, 0].scatter(design_plot[:, 0], design_plot[:, 1], c=np.array(ac_f).reshape(-1))


        start = time.time()
        if KG_dynamic_optimisation:
            #Saving original variables
            self.original_X = self.X.copy()
            self.original_Y = self.Y.copy()
            self.original_C = self.C.copy()

            self.acquisition.optimizer.context_manager = ContextManager(self.space, self.context)
            out = self.acquisition.optimizer.optimize_inner_func(f=self.expected_improvement, duplicate_manager=None,
                                                                 num_samples=100)

            print("best expected improvement found", out)
            self.suggested_sample = self.space.zip_inputs(out[0])
            self.X = np.vstack((self.X, self.suggested_sample))

            self.evaluate_objective()
            self._update_model()

        self.acquisition.optimizer.context_manager = ContextManager(self.space, self.context)
        out = self.acquisition.optimizer.optimize_inner_func(f=self.aggregated_posterior, duplicate_manager=None,  num_samples=100)
        print("best sample found from Posterior GP", out)

        suggested_final_sample =  self.space.zip_inputs(out[0])

        Y_true, _ = self.objective.evaluate(suggested_final_sample, true_val=True)
        C_true, _ = self.constraint.evaluate(suggested_final_sample, true_val=True)
        # print("C_true", C_true)
        bool_C_true = np.product(np.concatenate(C_true, axis=1) < 0, axis=1)

        if bool_C_true==0:
            samples = self.X
            # print("samples", samples)
            Y = self.model.posterior_mean(samples)
            # print("Y",Y)
            pf = self.probability_feasibility_multi_gp(samples, model=self.model_c)

            # print("pf", pf)
            func_val = np.array(Y).reshape(-1) * np.array(pf).reshape(-1)

            # print("Y", Y, "pf", pf, "func_val", func_val)
            suggested_final_sample = samples[np.argmax(func_val)]
            suggested_final_sample = np.array(suggested_final_sample).reshape(-1)
            suggested_final_sample = np.array(suggested_final_sample).reshape(1, -1)
            # print("suggested_final_sample", suggested_final_sample, "val",np.max(func_val) )
            Y_true, _ = self.objective.evaluate(suggested_final_sample, true_val=True)

            C_true, _ = self.constraint.evaluate(suggested_final_sample, true_val=True)
            # print("C_true", C_true)
            bool_C_true = np.product(np.concatenate(C_true, axis=1) < 0, axis=1)

        func_val_true = Y_true * bool_C_true.reshape(-1, 1)

        # print("func_val_true",func_val_true)
        self.X = self.original_X.copy()
        self.Y = self.original_Y.copy()
        self.C = self.original_C.copy()
        self._update_model()

        self.true_best_value()
        optimum = np.max(self.true_best_stats["true_best"])
        self.recommended_value.append(np.max(func_val_true).reshape(-1))
        self.underlying_optimum.append(optimum.reshape(-1))
        print("OC", optimum - np.max(func_val_true),"optimum", optimum, "func_val recommended", np.max(func_val_true))
        self.Opportunity_Cost.append(optimum - np.max(func_val_true))


    def aggregated_posterior(self, X):
        mu = self.model.posterior_mean(X)
        mu = mu.reshape(-1, 1)
        pf = self.probability_feasibility_multi_gp(X, self.model_c).reshape(-1, 1)
        pf = np.array(pf).reshape(-1)
        mu = np.array(mu).reshape(-1)
        return -(mu * pf).reshape(-1)

    def expected_improvement(self, X):
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

        bool_C = np.product(np.concatenate(self.C, axis=1) < 0, axis=1)
        func_val = self.model.posterior_mean(self.model.get_X_values()).reshape(-1)* bool_C.reshape(-1)
        mu_sample_opt = np.max(func_val)

        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        pf = self.probability_feasibility_multi_gp(X,self.model_c).reshape(-1,1)
        pf = np.array(pf).reshape(-1)
        ei = np.array(ei).reshape(-1)
        return -(ei *pf ).reshape(-1)



    def probability_feasibility_multi_gp(self, x, model, mean=None, cov=None, l=0):
        # print("model",model.output)
        x = np.atleast_2d(x)

        Fz = []
        for m in range(model.output_dim):
            Fz.append(self.probability_feasibility( x, model.output[m], l))
        Fz = np.product(Fz,axis=0)
        return Fz

    def probability_feasibility(self, x, model, l=0):

        model = model.model
        # kern = model.kern
        # X = model.X
        mean = model.posterior_mean(x)
        var = model.posterior_variance(x, noise=False)
        # print("mean",mean,"var",var)
        std = np.sqrt(var).reshape(-1, 1)

        mean = mean.reshape(-1, 1)

        norm_dist = norm(mean, std)


        Fz = norm_dist.cdf(l)

        return Fz.reshape(-1, 1)


    def evaluate_objective(self):
        """
        Evaluates the objective
        """
        print(1)
            # print(self.suggested_sample)
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

        if self.sample_from_acq:
            print("suggest next location given THOMPSON SAMPLING")
            candidate_points= initial_design('latin', self.space, 2000)
            aux_var = self.acquisition._compute_acq(candidate_points)
        else:
            # try:
                # self.acquisition.generate_random_vectors(optimize_discretization=True, optimize_random_Z=True)
            aux_var = self.evaluator.compute_batch(duplicate_manager=None, re_use=re_use, dynamic_optimisation=self.KG_dynamic_optimisation)
            # except:
            #     aux_var = self.evaluator.compute_batch(duplicate_manager=None, re_use=re_use)

        return self.space.zip_inputs(aux_var[0])
        #return initial_design('random', self.space, 1)


    def _update_fantasised_model(self, X, Y, C):

        if (self.num_acquisitions % self.model_update_interval) == 0:
            ### --- input that goes into the model (is unziped in case there are categorical variables)
            X_inmodel = self.space.unzip_inputs(X)
            Y_inmodel = list(Y)
            C_inmodel = list(C)

            self.model.updateModel(X_inmodel, Y_inmodel)
            self.model_c.updateModel(X_inmodel, C_inmodel)

    def _update_model(self):
        """
        Updates the model (when more than one observation is available) and saves the parameters (if available).
        """
        if (self.num_acquisitions%self.model_update_interval)==0:

            ### --- input that goes into the model (is unziped in case there are categorical variables)
            X_inmodel = self.space.unzip_inputs(self.X)
            Y_inmodel = list(self.Y)
            C_inmodel = list(self.C)
            
            self.model.updateModel(X_inmodel, Y_inmodel)
            self.model_c.updateModel(X_inmodel, C_inmodel)
        ### --- Save parameters of the model
        #self._save_model_parameter_values()


    def get_evaluations(self):
        return self.X.copy(), self.Y.copy()

    def true_best_value(self):
        from scipy.optimize import minimize

        X = initial_design('random', self.space, 10000)

        fval = self.func_val(X)

        anchor_point = np.array(X[np.argmin(fval)]).reshape(-1)
        anchor_point = anchor_point.reshape(1, -1)

        best_design = minimize(self.func_val, anchor_point, method='nelder-mead', tol=1e-8).x

        value_best_design  = -self.func_val(best_design)
        print("best design", best_design, "value_best_design", value_best_design)

        self.true_best_stats["true_best"].append(value_best_design)
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