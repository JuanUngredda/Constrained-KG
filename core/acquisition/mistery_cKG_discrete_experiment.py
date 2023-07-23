import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

import GPyOpt
from GPyOpt.objective_examples.experiments2d import mistery
from Hybrid_discrete_KG_v2 import KG
from bayesian_optimisation import BO
from multi_objective import MultiObjective
from multi_outputGP import multi_outputGP

# ALWAYS check cost in
# --- Function to optimize
print("mistery activate")


def function_caller_mistery_v2(it):
    np.random.seed(it)
    for noise in [1e-06]:
        for num_samples in [50, 100, 500]:
            # func2 = dropwave()
            num_underlying_samples = num_samples
            noise_objective = noise
            noise_constraints = 1e-06
            mistery_f = mistery(sd_obj=np.sqrt(noise_objective), sd_c=np.sqrt(noise_constraints))

            # --- Attributes
            # repeat same objective function to solve a 1 objective problem
            f = MultiObjective([mistery_f.f])
            c = MultiObjective([mistery_f.c])

            # --- Attributes
            # repeat same objective function to solve a 1 objective problem

            # c2 = MultiObjective([test_c2])
            # --- Space
            # define space of variables
            space = GPyOpt.Design_space(space=[{'name': 'var_1', 'type': 'continuous', 'domain': (0, 5)},
                                               {'name': 'var_2', 'type': 'continuous', 'domain': (0,
                                                                                                  5)}])  # GPyOpt.Design_space(space =[{'name': 'var_1', 'type': 'continuous', 'domain': (0,100)}])#

            verbose = False
            if verbose:
                mistery_f_denoised = mistery(sd_obj=0, sd_c=0)
                plot_design = GPyOpt.experiment_design.initial_design('latin', space, 100000)

                fvals = mistery_f_denoised.f(plot_design)  # f.evaluate(plot_design)
                cvals = mistery_f_denoised.c(plot_design)  # c.evaluate(plot_design)
                cvalsbool = np.array(cvals).reshape(-1) < 0
                fvals = np.array(fvals).reshape(-1)
                X_argmax = np.atleast_2d(plot_design[np.argmax(fvals[cvalsbool])])

                max_fvals = []
                for _ in range(100):
                    fvals_noised = mistery_f.f(X_argmax)  # f.evaluate(plot_design)
                    # cvals = mistery_f.c(X_argmax) #c.evaluate(plot_design)
                    # cvalsbool = np.array(cvals).reshape(-1) < 0
                    fvals_noised = np.array(fvals_noised).reshape(-1)
                    max_fvals.append(fvals_noised)

                signal_to_noise_ratio = np.abs(np.mean(max_fvals)) / np.std(max_fvals)
                print("signal_to_noise_ratio", signal_to_noise_ratio)
                # print("max constrained value", np.max(fvals[cvalsbool]))
                # print("max", np.max(fvals), "min", np.min(fvals))
                # print("max", np.max(cvals), "min", np.min(cvals))

                plt.scatter(plot_design[:, 0][cvalsbool], plot_design[:, 1][cvalsbool], c=fvals[cvalsbool])
                plt.show()
                raise

            n_f = 1
            n_c = 1
            model_f = multi_outputGP(output_dim=n_f, noise_var=[noise_objective] * n_f, exact_feval=[True] * n_f)
            model_c = multi_outputGP(output_dim=n_c, noise_var=[noise_constraints] * n_c, exact_feval=[True] * n_c)

            # --- Aquisition optimizer
            # optimizer for inner acquisition function
            type_anchor_points_logic = "max_objective"
            acq_opt = GPyOpt.optimization.AcquisitionOptimizer(optimizer="lbfgs", inner_optimizer='lbfgs', space=space,
                                                               model=model_f, model_c=model_c,
                                                               anchor_points_logic=type_anchor_points_logic)
            #
            # # --- Initial design
            # initial design
            underlying_discrete_space = GPyOpt.experiment_design.initial_design('latin', space, num_underlying_samples)
            initial_design = underlying_discrete_space[
                np.random.choice(np.arange(len(underlying_discrete_space)), size=10, replace=False)]

            nz = 20  # (n_c+1)
            acquisition = KG(model=model_f, model_c=model_c, space=space, nz=nz, optimizer=acq_opt,
                             underlying_discretisation=underlying_discrete_space)

            evaluator = GPyOpt.core.evaluators.Sequential(acquisition)
            bo = BO(model_f, model_c, space, f, c, acquisition, evaluator, initial_design,
                    ls_evaluator=None,  # last_step_evaluator,
                    ls_acquisition=None,  # Last_Step_acq,
                    deterministic=False,
                    underlying_discretisation=underlying_discrete_space)

            stop_date = datetime(2030, 5, 9, 7)  # year month day hour
            max_iter = 100
            # print("Finished Initialization")
            subfolder = "discrete_mistery_cKG_n_obj_" + str(noise_objective) + "_n_c_" + str(noise_constraints)
            folder = "RESULTS"
            cwd = os.getcwd()
            path = cwd + "/" + folder + "/" + subfolder + '/' + str(num_underlying_samples)
            file_name = 'it_' + str(it) + '.csv'
            if not os.path.isdir(path):
                os.makedirs(path)

            if not os.path.isfile(path + "/" + file_name):
                bo.run_optimization(max_iter=max_iter,
                                    verbosity=False,
                                    path=path + "/" + file_name,
                                    stop_date=stop_date,
                                    compute_OC=True,
                                    evaluations_file=subfolder,
                                    KG_dynamic_optimisation=True)

            print("Code Ended")


for i in range(30):
    function_caller_mistery_v2(it=i)
