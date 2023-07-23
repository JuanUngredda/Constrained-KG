# Copyright (c) 2018, Raul Astudillo

import os
import time
from datetime import datetime

# import pygmo as pg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

import GPyOpt
from GPyOpt.core.errors import InvalidConfigError
from GPyOpt.core.task.cost import CostModel
from GPyOpt.experiment_design import initial_design
from GPyOpt.optimization.acquisition_optimizer import ContextManager

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

    def __init__(self, model, model_c, space, objective, constraint, acquisition, evaluator,
                 X_init, ls_evaluator=None, ls_acquisition=None, tag_last_evaluation=True, expensive=False,
                 Y_init=None, C_init=None, cost=None, normalize_Y=False, model_update_interval=1,
                 underlying_discretisation=None,
                 deterministic=True, true_preference=0.5):

        self.true_preference = true_preference
        self.model_c = model_c
        self.model = model
        self.space = space
        self.objective = objective
        self.constraint = constraint
        self.acquisition = acquisition
        self.ls_acquisition = ls_acquisition
        self.utility = acquisition.utility
        self.evaluator = evaluator
        self.ls_evaluator = ls_evaluator
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
        self.underlying_discretisation = underlying_discretisation

        try:
            if acquisition.name == "Constrained_Thompson_Sampling":
                self.sample_from_acq = True

            else:
                self.sample_from_acq = False

        except:
            print("name of acquisition function wasnt provided")
            self.sample_from_acq = False

    def suggest_next_locations(self, context=None, pending_X=None, ignored_X=None):
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

    def run_optimization(self, max_iter=1,
                         max_time=np.inf,
                         compute_OC=True,
                         stop_date=None,
                         eps=1e-8,
                         context=None,
                         verbosity=False,
                         path=None,
                         KG_dynamic_optimisation=False,
                         evaluations_file=None,
                         rep=None):
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
        self.compute_OC = compute_OC
        self.KG_dynamic_optimisation = KG_dynamic_optimisation
        self.verbosity = verbosity
        self.evaluations_file = evaluations_file
        self.context = context
        self.path = path

        # --- Setting up stop conditions
        self.eps = eps
        if (max_iter is None) and (max_time is None):
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
        self.n_init = self.X.shape[0]
        # self.model.updateModel(self.X,self.Y)

        # --- Initialize iterations and running time
        self.time_zero = time.time()
        self.cum_time = 0
        self.num_acquisitions = 0
        self.suggested_sample = self.X
        self.Y_new = self.Y
        self.Opportunity_Cost_sampled = []
        self.Opportunity_Cost_GP_mean = []

        self.recommended_value_GP_mean = []
        self.recommended_value_sampled = []
        self.underlying_optimum = []
        value_so_far = []

        # --- Initialize time cost of the evaluations
        print("MAIN LOOP STARTS")
        Opportunity_Cost = []
        self.true_best_stats = {"true_best": [], "mean_gp": [], "std gp": [], "pf": [], "mu_pf": [], "var_pf": [],
                                "residual_noise": []}

        self.stop_date = stop_date  # datetime(2021, 5, 8, 6)
        today = datetime.now()
        while (self.max_iter > self.num_acquisitions) and (self.stop_date > today):
            today = datetime.now()

            self._update_model()

            self.optimize_final_evaluation(self.KG_dynamic_optimisation)
            print("maKG optimizer")
            start = time.time()

            print("X", self.X)
            print("Y", self.Y)
            print("C", np.product(np.array(self.C) < 0, axis=0))
            self.suggested_sample = self._compute_next_evaluations()

            finish = time.time()
            if verbosity:
                print("self.suggested_sample", self.suggested_sample)
                self.acquisition._plots(self.suggested_sample)

            if False:  # verbosity:
                print("self.suggested_sample", self.suggested_sample)
                initial_design = GPyOpt.experiment_design.initial_design('latin', self.space, 10000)
                fvals, _ = self.objective.evaluate(initial_design)
                cvals, _ = self.constraint.evaluate(initial_design)
                cvals = np.hstack(cvals).squeeze()

                cvalsbool = np.array(cvals) < 0

                if len(cvalsbool.shape) > 1:
                    cvalsbool = np.product(cvalsbool, axis=1)

                cvalsbool = np.array(cvalsbool, dtype=bool).reshape(-1)

                fvals = np.array(fvals).reshape(-1)

                plt.title("real surface")
                plt.scatter(initial_design[:, 0][cvalsbool], initial_design[:, 1][cvalsbool], c=fvals[cvalsbool])
                plt.scatter(self.X[:, 0], self.X[:, 1], color="magenta")
                plt.scatter(self.suggested_sample[:, 0], self.suggested_sample[:, 1], color="red", s=30)
                plt.show()

                initial_design = GPyOpt.experiment_design.initial_design('latin', self.space, 1000)
                acq_vals = self.acquisition._compute_acq(initial_design)

                agregated_posterior = self.aggregated_posterior(initial_design)
                plt.title("estimated surface")
                plt.scatter(initial_design[:, 0], initial_design[:, 1], c=-agregated_posterior)
                plt.show()

                _, penalty = self.update_penalty()
                print("penalty", penalty)
                agregated_penalised_posterior = self.agreagated_penalised_posterior(initial_design,
                                                                                    penalisation=penalty)
                plt.title("estimated penalised surface")
                plt.scatter(initial_design[:, 0], initial_design[:, 1], c=-agregated_penalised_posterior)
                plt.show()
                #
                # print("max val", np.max(acq_vals), "min val", np.min(acq_vals))
                # plt.title("acq")
                # plt.scatter(initial_design[:, 0], initial_design[:, 1], c=acq_vals.reshape(-1))
                # plt.show()

                # raise
            print("self.suggested_sample", self.suggested_sample)
            print("time optimisation point X", finish - start)

            self.X = np.vstack((self.X, self.suggested_sample))
            # --- Evaluate *f* in X, augment Y and update cost function (if needed)
            self.evaluate_objective()

            # --- Update current evaluation time and function evaluations
            self.cum_time = time.time() - self.time_zero
            self.num_acquisitions += 1
            print("optimize_final_evaluation")

            C_bool = np.product(np.concatenate(self.C, axis=1) < 0, axis=1)
            data = {}

            data["OC sampled"] = np.concatenate(
                (np.zeros(self.n_init), np.array(self.Opportunity_Cost_sampled).reshape(-1)))
            data["OC GP mean"] = np.concatenate(
                (np.zeros(self.n_init), np.array(self.Opportunity_Cost_GP_mean).reshape(-1)))
            data["Y"] = np.array(self.Y).reshape(-1)
            # data["C"] = np.array(self.C).reshape(-1)
            data["C_bool"] = np.array(C_bool).reshape(-1)
            data["recommended_val_sampled"] = np.concatenate(
                (np.zeros(self.n_init), np.array(self.recommended_value_sampled).reshape(-1)))
            data["recommended_val_GP"] = np.concatenate(
                (np.zeros(self.n_init), np.array(self.recommended_value_GP_mean).reshape(-1)))
            data["optimum"] = np.concatenate((np.zeros(self.n_init), np.array(self.underlying_optimum).reshape(-1)))

            print("1", len(data["OC sampled"]), "2", len(data["OC GP mean"]), "3", len(data["Y"]), "4",
                  len(data["C_bool"]), "5", len(data["recommended_val_sampled"]),
                  "6", len(data["recommended_val_GP"]), "7", len(data["optimum"]))

            print(data)
            gen_file = pd.DataFrame.from_dict(data)
            folder = "RESULTS"
            subfolder = self.evaluations_file
            cwd = os.getcwd()

            path = self.path
            if os.path.isdir(cwd + "/" + folder + "/" + subfolder) == False:
                os.makedirs(cwd + "/" + folder + "/" + subfolder)
            print("path", path)
            gen_file.to_csv(path_or_buf=path)

            np.savetxt(cwd + "/" + folder + "/" + subfolder + "/X_" + str(rep) + ".csv", self.X, delimiter=',')

            print("self.X, self.Y, self.C , OC sampled, OC GP mean", self.X, self.Y, self.C)

            print("OC GP", self.Opportunity_Cost_GP_mean)
            print("OC sampled", self.Opportunity_Cost_sampled)

            discretisation_optimum_val = -np.min(self.func_val(self.underlying_discretisation))
            print("best achievable discretisation value", discretisation_optimum_val)
            if np.isclose(discretisation_optimum_val, np.mean(self.Opportunity_Cost_GP_mean[-3:]), rtol=1e-4, atol=1e-4) & (len(
                    self.Opportunity_Cost_GP_mean) > 3):
                print("Code stopped early")
                break
            overall_time_stop = time.time()

        return self.X, self.Y, self.C, self.recommended_value_sampled, self.underlying_optimum, self.Opportunity_Cost_sampled
        # --- Print the desired result in files
        # if self.evaluations_file is not None:
        # self.save_evaluations(self.evaluations_file)

        # file = open('test_file.txt','w')
        # np.savetxt('test_file.txt',value_so_far)

    def verbosity_plot_1D(self):
        ####plots
        print("generating plots")
        design_plot = np.linspace(0, 5, 1000)[:, None]

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
        mu_f = self.model.predict(design_plot)[0]

        bool_C = np.product(np.concatenate(C, axis=1) < 0, axis=1)
        func_val = Y * bool_C.reshape(-1, 1)

        self.acquisition.generate_random_vectors(optimize_discretization=True, optimize_random_Z=False,
                                                 fixed_discretisation=design_plot)

        best_kg_val = self.acquisition._compute_acq(self.suggested_sample)

        self.acquisition.generate_random_vectors(optimize_discretization=False, optimize_random_Z=False,
                                                 fixed_discretisation=design_plot)

        kg_f = self.acquisition._compute_acq(design_plot)
        design_plot = design_plot.reshape(-1)

        fig, axs = plt.subplots(2, 2)
        axs[0, 0].set_title('True Function')
        axs[0, 0].plot(design_plot, np.array(func_val).reshape(-1))
        axs[0, 0].scatter(self.X, np.array(self.Y).reshape(-1) * bool_C_sampled, color="red")
        axs[0, 0].scatter(self.suggested_sample, np.array([0]), marker="x", color="magenta")
        axs[0, 0].legend()

        axs[1, 0].set_title("mu pf")
        axs[1, 0].plot(design_plot, np.array(mu_f).reshape(-1) * np.array(pf).reshape(-1))
        axs[1, 0].legend()

        axs[1, 1].set_title('approximation kg Function')
        axs[1, 1].plot(design_plot, np.array(kg_f).reshape(-1))
        axs[1, 1].scatter(self.suggested_sample, np.array(best_kg_val).reshape(-1), marker="x", color="magenta")
        axs[1, 1].legend()

        axs[0, 1].set_title('Opportunity Cost')
        axs[0, 1].plot(range(len(self.Opportunity_Cost_sampled)), self.Opportunity_Cost_sampled)
        axs[0, 1].set_yscale("log")
        plt.show()

    def verbosity_plot_2D(self):
        ####plots
        print("generating plots")
        design_plot = initial_design('random', self.space, 400)

        # precision = []
        # for i in range(20):
        # self.acquisition.generate_random_vectors(optimize_discretization=False, optimize_random_Z=False, fixed_discretisation=design_plot)
        # kg_f = self.acquisition._compute_acq(design_plot)
        # print("np.min", np.min(kg_f), "max", np.max(kg_f))

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
        fig, axs = plt.subplots(1, 2)
        axs[0].set_title('True Function')
        axs[0].scatter(design_plot[:, 0], design_plot[:, 1], c=np.array(func_val).reshape(-1))
        axs[0].scatter(self.X[:, 0], self.X[:, 1], color="red", label="sampled")
        # suggested_sample_value = self.objective.evaluate(self.suggested_sample)
        axs[0].scatter(self.suggested_sample[:, 0], self.suggested_sample[:, 1], marker="x", color="red",
                       label="suggested")
        axs[0].legend()

        # axs[0, 1].set_title('approximation Acqu Function')
        # axs[0, 1].scatter(design_plot[:,0],design_plot[:,1], c=np.array(ac_f).reshape(-1))
        # axs[0, 1].legend()
        print("self.suggested_sample", self.suggested_sample)
        # import matplotlib
        # axs[1, 0].set_title("KG")
        # axs[1, 0].scatter(design_plot[:,0],design_plot[:,1],norm=matplotlib.colors.LogNorm(), c= np.array(kg_f).reshape(-1))
        # axs[1, 1].scatter(self.suggested_sample[:, 0], self.suggested_sample[:, 1], color="red", label="KG suggested")
        # axs[1, 0].legend()

        axs[1].set_title("mu pf")
        axs[1].scatter(design_plot[:, 0], design_plot[:, 1], c=np.array(mu_f).reshape(-1) * np.array(pf).reshape(-1))
        axs[1].scatter(self.suggested_sample[:, 0], self.suggested_sample[:, 1], color="red", label="KG suggested")
        # axs[1,1].scatter(self.final_suggested_sample[:,0], self.final_suggested_sample[:,1], color="magenta", label="Final suggested")
        axs[1].legend()

        plt.show()

        print("self.Opportunity_Cost plot", self.Opportunity_Cost)
        plt.title('Opportunity Cost')
        plt.plot(range(len(self.Opportunity_Cost)), self.Opportunity_Cost)
        plt.yscale("log")
        plt.show()

    def optimize_final_evaluation(self, KG_dynamic_optimisation):

        if KG_dynamic_optimisation:
            # Saving original variables
            self.original_X = self.X.copy()
            self.original_Y = self.Y.copy()
            self.original_C = self.C.copy()

            if self.ls_evaluator is not None:
                out = self.ls_evaluator.compute_batch(duplicate_manager=None, re_use=False, dynamic_optimisation=False)
                self.suggested_sample = self.space.zip_inputs(out[0])
                self.X = np.vstack((self.X, self.suggested_sample))

                self.evaluate_objective()
                self._update_model()
            #
            self.acquisition.optimizer.context_manager = ContextManager(self.space, self.context)
            if self.underlying_discretisation is None:
                out = self.acquisition.optimizer.optimize(f=self.aggregated_posterior, duplicate_manager=None,
                                                          additional_anchor_points=self.X[7:, :], num_samples=1000)
            else:
                fX_vals = self.aggregated_posterior(self.underlying_discretisation)
                out = (self.underlying_discretisation[np.argmax(-fX_vals)][None, :], np.min(fX_vals))

            suggested_final_sample_GP_recommended = self.space.zip_inputs(out[0])
            suggested_final_mean_GP_recommended = -out[1]

            Y = self.model.posterior_mean(self.X)
            pf = self.probability_feasibility_multi_gp(self.X, model=self.model_c)
            func_val = np.array(Y).reshape(-1) * np.array(pf).reshape(-1)

            print("np.array(Y).reshape(-1)", np.array(Y).reshape(-1))
            print(" np.array(pf).reshape(-1)", np.array(pf).reshape(-1))
            print("func_val", func_val)

            suggested_final_historical_sample = self.X[np.argmax(func_val)]
            suggested_final_historical_sample = np.array(suggested_final_historical_sample).reshape(1, -1)
            suggested_final_mean_historical_sample = np.array(np.max(func_val)).reshape(-1)

            if suggested_final_mean_GP_recommended.reshape(-1) < np.array(
                    suggested_final_mean_historical_sample).reshape(-1):

                Y_true, _ = self.objective.evaluate(suggested_final_historical_sample, true_val=True)
                C_true, _ = self.constraint.evaluate(suggested_final_historical_sample, true_val=True)

            else:
                Y_true, _ = self.objective.evaluate(suggested_final_sample_GP_recommended, true_val=True)
                C_true, _ = self.constraint.evaluate(suggested_final_sample_GP_recommended, true_val=True)

            bool_C_true = np.product(np.concatenate(C_true, axis=1) < 0, axis=1)
            func_val_true_GP_recommended = Y_true * bool_C_true.reshape(-1, 1)
            func_val_true_GP_mean = func_val_true_GP_recommended

            if self.compute_OC:
                self.true_best_value()
                optimum = np.max(self.true_best_stats["true_best"])
            else:

                optimum = np.array([0]).reshape(-1)
            self.recommended_value_GP_mean.append(np.max(func_val_true_GP_mean).reshape(-1))
            print("best posterior mean")
            print("best sample found from Posterior GP sample", out[0], "best sample historical sample",
                  suggested_final_historical_sample)
            print("best sample found from Posterior GP val", -out[1], "best sample historical val",
                  suggested_final_mean_historical_sample)

            print("OC", np.max(func_val_true_GP_mean), "optimum", optimum, "func_val recommended",
                  np.max(func_val_true_GP_mean))

            self.Opportunity_Cost_GP_mean.append(np.max(func_val_true_GP_mean))

            Y = self.model.posterior_mean(self.X)
            pf = self.probability_feasibility_multi_gp(self.X, model=self.model_c)
            func_val = np.array(Y).reshape(-1) * np.array(pf).reshape(-1)
            suggested_final_historical_sample = self.X[np.argmax(func_val)]
            suggested_final_historical_sample = np.array(suggested_final_historical_sample).reshape(1, -1)
            #
            Y_true, _ = self.objective.evaluate(suggested_final_historical_sample, true_val=True)
            C_true, _ = self.constraint.evaluate(suggested_final_historical_sample, true_val=True)
            #
            bool_C_true = np.product(np.concatenate(C_true, axis=1) < 0, axis=1)
            func_val_true_GP_recommended = Y_true * bool_C_true.reshape(-1, 1)
            func_val_true_sampled = func_val_true_GP_recommended
            #
            if self.compute_OC:
                self.true_best_value()
                optimum = np.max(self.true_best_stats["true_best"])
            else:
                optimum = np.array([0]).reshape(-1)
            self.recommended_value_sampled.append([0])
            self.underlying_optimum.append([optimum])
            self.Opportunity_Cost_sampled.append(np.max(func_val_true_sampled))

            print("sampled OC")
            print("best sample found from Posterior GP sample", out[0], "best sample historical sample",
                  np.max(func_val_true_sampled).reshape(-1))
            print("OC", np.max(func_val_true_sampled), "optimum", optimum, "func_val recommended",
                  np.max(func_val_true_sampled))

            # raise
            self.X = self.original_X.copy()
            self.Y = self.original_Y.copy()
            self.C = self.original_C.copy()
            self._update_model()

            return 0
        else:
            if self.deterministic:
                Y = self.model.posterior_mean(self.X)
                pf = self.probability_feasibility_multi_gp(self.X, model=self.model_c)
                func_val = np.array(Y).reshape(-1) * np.array(pf).reshape(-1)
                suggested_final_historical_sample = self.X[np.argmax(func_val)]
                suggested_final_historical_sample = np.array(suggested_final_historical_sample).reshape(1, -1)

                Y_true, _ = self.objective.evaluate(suggested_final_historical_sample, true_val=True)
                C_true, _ = self.constraint.evaluate(suggested_final_historical_sample, true_val=True)

                bool_C_true = np.product(np.concatenate(C_true, axis=1) < 0, axis=1)
                func_val_true_GP_recommended = Y_true * bool_C_true.reshape(-1, 1)

                func_val_true = func_val_true_GP_recommended

                if self.compute_OC:
                    self.true_best_value()
                    optimum = np.max(self.true_best_stats["true_best"])
                else:
                    optimum = np.array([0]).reshape(-1)
                self.recommended_value_sampled.append(np.max(func_val_true).reshape(-1))
                self.underlying_optimum.append(optimum.reshape(-1))
                self.Opportunity_Cost_sampled.append(np.max(func_val_true))

                self.recommended_value_GP_mean.append(np.max(func_val_true).reshape(-1))
                self.Opportunity_Cost_GP_mean.append(np.max(func_val_true))
                return 0

            else:  # KG_dynamic_optimisation:

                self.acquisition.optimizer.context_manager = ContextManager(self.space, self.context)

                if self.underlying_discretisation is None:
                    out = self.acquisition.optimizer.optimize(f=self.aggregated_posterior, duplicate_manager=None,
                                                              additional_anchor_points=self.X[7:, :], num_samples=1000)
                else:
                    fX_vals = self.aggregated_posterior(self.underlying_discretisation)
                    out = (self.underlying_discretisation[np.argmax(-fX_vals)][None, :], np.min(fX_vals))

                suggested_final_sample_GP_recommended = self.space.zip_inputs(out[0])
                suggested_final_mean_GP_recommended = -out[1]

                Y = self.model.posterior_mean(self.X)
                pf = self.probability_feasibility_multi_gp(self.X, model=self.model_c)
                func_val = np.array(Y).reshape(-1) * np.array(pf).reshape(-1)

                print("np.array(Y).reshape(-1)", np.array(Y).reshape(-1))
                print(" np.array(pf).reshape(-1)", np.array(pf).reshape(-1))
                print("func_val", func_val)

                suggested_final_historical_sample = self.X[np.argmax(func_val)]
                suggested_final_historical_sample = np.array(suggested_final_historical_sample).reshape(1, -1)
                suggested_final_mean_historical_sample = np.array(np.max(func_val)).reshape(-1)

                if suggested_final_mean_GP_recommended.reshape(-1) < np.array(
                        suggested_final_mean_historical_sample).reshape(-1):

                    Y_true, _ = self.objective.evaluate(suggested_final_historical_sample, true_val=True)
                    C_true, _ = self.constraint.evaluate(suggested_final_historical_sample, true_val=True)

                else:
                    Y_true, _ = self.objective.evaluate(suggested_final_sample_GP_recommended, true_val=True)
                    C_true, _ = self.constraint.evaluate(suggested_final_sample_GP_recommended, true_val=True)

                bool_C_true = np.product(np.concatenate(C_true, axis=1) < 0, axis=1)
                func_val_true_GP_recommended = Y_true * bool_C_true.reshape(-1, 1)
                func_val_true_GP_mean = func_val_true_GP_recommended

                if self.compute_OC:
                    self.true_best_value()
                    optimum = np.max(self.true_best_stats["true_best"])
                else:
                    optimum = np.array([0]).reshape(-1)
                self.recommended_value_GP_mean.append(np.max(func_val_true_GP_mean).reshape(-1))
                print("best posterior mean")
                print("best sample found from Posterior GP sample", out[0], "best sample historical sample",
                      suggested_final_historical_sample)
                print("best sample found from Posterior GP val", -out[1], "best sample historical val",
                      suggested_final_mean_historical_sample)

                print("OC", np.max(func_val_true_GP_mean), "optimum", optimum, "func_val recommended",
                      np.max(func_val_true_GP_mean))

                self.Opportunity_Cost_GP_mean.append(np.max(func_val_true_GP_mean))

                # Saving original variables
                # self.original_X = self.X.copy()
                # self.original_Y = self.Y.copy()
                # self.original_C = self.C.copy()
                #
                # out = self.ls_evaluator.compute_batch(duplicate_manager=None, re_use=False, dynamic_optimisation=False)
                # self.suggested_sample = self.space.zip_inputs(out[0])
                # self.X = np.vstack((self.X, self.suggested_sample))
                #
                # self.evaluate_objective()
                # self._update_model()
                #
                Y = self.model.posterior_mean(self.X)
                pf = self.probability_feasibility_multi_gp(self.X, model=self.model_c)
                func_val = np.array(Y).reshape(-1) * np.array(pf).reshape(-1)
                suggested_final_historical_sample = self.X[np.argmax(func_val)]
                suggested_final_historical_sample = np.array(suggested_final_historical_sample).reshape(1, -1)
                #
                Y_true, _ = self.objective.evaluate(suggested_final_historical_sample, true_val=True)
                C_true, _ = self.constraint.evaluate(suggested_final_historical_sample, true_val=True)
                #
                bool_C_true = np.product(np.concatenate(C_true, axis=1) < 0, axis=1)
                func_val_true_GP_recommended = Y_true * bool_C_true.reshape(-1, 1)
                func_val_true_sampled = func_val_true_GP_recommended
                #
                if self.compute_OC:
                    self.true_best_value()
                    print(self.true_best_stats["true_best"])
                    optimum = np.max(self.true_best_stats["true_best"])
                else:
                    optimum = np.array([0]).reshape(-1)
                self.recommended_value_sampled.append([0])
                self.underlying_optimum.append([0])
                self.Opportunity_Cost_sampled.append(np.max(func_val_true_sampled))

            #     print("sampled OC")
            #     print("best sample found from Posterior GP sample",  out[0], "best sample historical sample",np.max(func_val_true_sampled).reshape(-1))
            #     print("OC", optimum - np.max(func_val_true_sampled), "optimum", optimum, "func_val recommended",
            #           np.max(func_val_true_sampled))
            #
            #  # Y_true * bool_C_true.reshape(-1, 1)
            #
            # # print("func_val_true",func_val_true)
            # self.X = self.original_X.copy()
            # self.Y = self.original_Y.copy()
            # self.C = self.original_C.copy()
            # self._update_model()

            return 0

    def aggregated_posterior(self, X):
        mu = self.model.posterior_mean(X)
        pf = self.probability_feasibility_multi_gp(X, self.model_c)
        pf = np.array(pf).reshape(-1)
        mu = np.array(mu).reshape(-1)
        return -(mu * pf).reshape(-1)

    def agreagated_penalised_posterior(self, X, penalisation):
        mu = self.model.posterior_mean(X)
        pf = self.probability_feasibility_multi_gp(X, self.model_c)
        pf = np.array(pf).reshape(-1)
        mu = np.array(mu).reshape(-1)
        feasable_term = (mu * pf).reshape(-1)
        infeasible_term = penalisation.squeeze() * (1 - pf.reshape(-1))
        overall_term = feasable_term + infeasible_term
        return -(overall_term).reshape(-1)

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
        mu = mu.reshape(-1, 1)

        bool_C = np.product(np.concatenate(self.C, axis=1) < 0, axis=1)
        func_val = self.model.posterior_mean(self.model.get_X_values()).reshape(-1) * bool_C.reshape(-1)
        mu_sample_opt = np.max(func_val)

        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        pf = self.probability_feasibility_multi_gp(X, self.model_c).reshape(-1, 1)
        pf = np.array(pf).reshape(-1)
        ei = np.array(ei).reshape(-1)
        return -(ei * pf).reshape(-1)

    def probability_feasibility_multi_gp(self, x, model, mean=None, cov=None, l=0):
        # print("model",model.output)
        x = np.atleast_2d(x)

        Fz = []
        for m in range(model.output_dim):
            Fz.append(self.probability_feasibility(x, model.output[m], l))
        Fz = np.product(Fz, axis=0)
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
            self.Y[j] = np.vstack((self.Y[j], self.Y_new[j]))

        for k in range(self.constraint.output_dim):
            print(self.C_new[k])
            self.C[k] = np.vstack((self.C[k], self.C_new[k]))

    def compute_current_best(self):
        current_acqX = self.acquisition.current_compute_acq()
        return current_acqX

    def _distance_last_evaluations(self):
        """
        Computes the distance between the last two evaluations.
        """
        return np.sqrt(sum((self.X[self.X.shape[0] - 1, :] - self.X[self.X.shape[0] - 2, :]) ** 2))

    def _compute_next_evaluations(self, pending_zipped_X=None, ignored_zipped_X=None, re_use=False):

        """
        Computes the location of the new evaluation (optimizes the acquisition in the standard case).
        :param pending_zipped_X: matrix of input configurations that are in a pending state (i.e., do not have an evaluation yet).
        :param ignored_zipped_X: matrix of input configurations that the user black-lists, i.e., those configurations will not be suggested again.
        :return:
        """
        ## --- Update the context if any

        self.acquisition.optimizer.context_manager = ContextManager(self.space, self.context)
        if self.underlying_discretisation is not None:
            fX_values = self.acquisition._compute_acq(self.underlying_discretisation)
            aux_var = self.underlying_discretisation[np.argmax(fX_values)][None, :]
            return self.space.zip_inputs(aux_var)
        else:
            if self.sample_from_acq:
                print("suggest next location given THOMPSON SAMPLING")
                candidate_points = initial_design('latin', self.space, 2000)
                aux_var = self.acquisition._compute_acq(candidate_points)
            else:
                # try:
                # self.acquisition.generate_random_vectors(optimize_discretization=True, optimize_random_Z=True)
                aux_var = self.evaluator.compute_batch(duplicate_manager=None, re_use=re_use,
                                                       dynamic_optimisation=self.KG_dynamic_optimisation)
                # except:
                #     aux_var = self.evaluator.compute_batch(duplicate_manager=None, re_use=re_use)

        return self.space.zip_inputs(aux_var[0])
        # return initial_design('random', self.space, 1)

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
        if (self.num_acquisitions % self.model_update_interval) == 0:
            ### --- input that goes into the model (is unziped in case there are categorical variables)
            X_inmodel = self.space.unzip_inputs(self.X)
            Y_inmodel = list(self.Y)
            C_inmodel = list(self.C)

            self.model.updateModel(X_inmodel, Y_inmodel)
            self.model_c.updateModel(X_inmodel, C_inmodel)
        ### --- Save parameters of the model
        # self._save_model_parameter_values()

    def get_evaluations(self):
        return self.X.copy(), self.Y.copy()

    def true_best_value(self):
        from scipy.optimize import minimize

        X = initial_design('random', self.space, 10000)

        fval = self.func_val(X)

        anchor_point = np.array(X[np.argmin(fval)]).reshape(-1)
        anchor_point = anchor_point.reshape(1, -1)

        if self.compute_OC:
            best_design = minimize(self.func_val, anchor_point, method='nelder-mead', tol=1e-8).x
            value_best_design = -self.func_val(best_design)
            print("best design", best_design, "value_best_design", value_best_design)
            self.true_best_stats["true_best"].append(value_best_design)
        else:
            self.true_best_stats["true_best"].append(0)

        self.true_best_stats["mean_gp"].append(self.model.posterior_mean(best_design))
        self.true_best_stats["std gp"].append(self.model.posterior_variance(best_design, noise=False))
        self.true_best_stats["pf"].append(
            self.probability_feasibility_multi_gp(best_design, self.model_c).reshape(-1, 1))
        mean = self.model_c.posterior_mean(best_design)
        var = self.model_c.posterior_variance(best_design, noise=False)
        residual_noise = self.model_c.posterior_variance(self.X[1], noise=False)
        self.true_best_stats["mu_pf"].append(mean)
        self.true_best_stats["var_pf"].append(var)
        self.true_best_stats["residual_noise"].append(residual_noise)

        if False:
            fig, axs = plt.subplots(3, 2)
            N = len(np.array(self.true_best_stats["std gp"]).reshape(-1))
            GAP = np.array(np.abs(
                np.abs(self.true_best_stats["true_best"]).reshape(-1) - np.abs(self.true_best_stats["mean_gp"]).reshape(
                    -1))).reshape(-1)
            print("GAP len", len(GAP))
            print("N", N)
            axs[0, 0].set_title('GAP')
            axs[0, 0].plot(range(N), GAP)
            axs[0, 0].set_yscale("log")

            axs[0, 1].set_title('VAR')
            axs[0, 1].plot(range(N), np.array(self.true_best_stats["std gp"]).reshape(-1))
            axs[0, 1].set_yscale("log")

            axs[1, 0].set_title("PF")
            axs[1, 0].plot(range(N), np.array(self.true_best_stats["pf"]).reshape(-1))

            axs[1, 1].set_title("mu_PF")
            axs[1, 1].plot(range(N), np.abs(np.array(self.true_best_stats["mu_pf"]).reshape(-1)))
            axs[1, 1].set_yscale("log")

            axs[2, 1].set_title("std_PF")
            axs[2, 1].plot(range(N), np.sqrt(np.array(self.true_best_stats["var_pf"]).reshape(-1)))
            axs[2, 1].set_yscale("log")

            axs[2, 0].set_title("Irreducible noise")
            axs[2, 0].plot(range(N), np.sqrt(np.array(self.true_best_stats["residual_noise"]).reshape(-1)))
            axs[2, 0].set_yscale("log")

            plt.show()

    def func_val(self, x):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        Y, _ = self.objective.evaluate(x, true_val=True)
        C, _ = self.constraint.evaluate(x, true_val=True)
        Y = np.array(Y).reshape(-1)
        out = Y.reshape(-1) * np.product(np.concatenate(C, axis=1) < 0, axis=1).reshape(-1)
        out = np.array(out).reshape(-1)
        return -out

    def current_func_no_constraint(self, X_inner):
        X_inner = np.atleast_2d(X_inner)
        mu = self.model.posterior_mean(X_inner)[0]
        mu = np.array(mu).reshape(-1)
        return mu

    def update_penalty(self):
        inner_opt_x, inner_opt_val = self.acquisition.optimizer.optimize_inner_func(f=self.current_func_no_constraint,
                                                                                    f_df=None,
                                                                                    num_samples=500)
        inner_opt_x = np.array(inner_opt_x).reshape(-1)
        inner_opt_x = np.atleast_2d(inner_opt_x)

        return inner_opt_x, inner_opt_val
