import numpy as np
import GPyOpt
from GPyOpt.objective_examples.experiments2d import mistery,  test_function_2, new_brannin
import GPy as GPy
from multi_objective import MultiObjective
from multi_outputGP import multi_outputGP
import matplotlib.pyplot as plt
import scipy
from Hybrid_continuous_KG_v2 import KG
from nEI import nEI
from EI import EI
from bayesian_optimisation import BO
import pandas as pd
import os
from datetime import datetime

#ALWAYS check cost in
# --- Function to optimize
print("mistery activate")
def function_caller_mistery_v2(rep):
    # rep = rep + 20
    np.random.seed(rep)

    for noise in [1]:
        # func2 = dropwave()
        noise_objective = noise
        noise_constraints = (0.1) ** 2
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
            X_argmax  = np.atleast_2d(plot_design[np.argmax(fvals[cvalsbool])])

            max_fvals = []
            for _ in range(100):
                fvals_noised = mistery_f.f(X_argmax) #f.evaluate(plot_design)
                # cvals = mistery_f.c(X_argmax) #c.evaluate(plot_design)
                # cvalsbool = np.array(cvals).reshape(-1) < 0
                fvals_noised = np.array(fvals_noised).reshape(-1)
                max_fvals.append(fvals_noised)

            signal_to_noise_ratio = np.abs(np.mean(max_fvals))/np.std(max_fvals)
            print("signal_to_noise_ratio",signal_to_noise_ratio)
            # print("max constrained value", np.max(fvals[cvalsbool]))
            # print("max", np.max(fvals), "min", np.min(fvals))
            # print("max", np.max(cvals), "min", np.min(cvals))

            plt.scatter(plot_design[:,0][cvalsbool],plot_design[:,1][cvalsbool], c=fvals[cvalsbool])
            plt.show()
            raise


        n_f = 1
        n_c = 1
        model_f = multi_outputGP(output_dim=n_f, noise_var=[noise_objective] * n_f, exact_feval=[False] * n_f)
        model_c = multi_outputGP(output_dim=n_c, noise_var=[noise_constraints] * n_c, exact_feval=[False] * n_c)

        # --- Aquisition optimizer
        #optimizer for inner acquisition function
        type_anchor_points_logic = "max_objective"
        acq_opt = GPyOpt.optimization.AcquisitionOptimizer(optimizer="lbfgs",inner_optimizer='lbfgs',space=space, model=model_f, model_c=model_c,anchor_points_logic=type_anchor_points_logic)
        #
        # # --- Initial design
        #initial design
        initial_design = GPyOpt.experiment_design.initial_design('latin', space, 10)

        nz = 20 # (n_c+1)
        acquisition = KG(model=model_f, model_c=model_c , space=space, nz=nz, optimizer = acq_opt)


        # Last_Step_acq = EI(model=model_f, model_c=model_c, space=space, nz=nz, optimizer=acq_opt)
        # last_step_evaluator = GPyOpt.core.evaluators.Sequential(Last_Step_acq)
        #
        evaluator = GPyOpt.core.evaluators.Sequential(acquisition)
        bo = BO(model_f, model_c, space, f, c, acquisition, evaluator, initial_design,
                ls_evaluator=None,#last_step_evaluator,
                ls_acquisition=None,#Last_Step_acq,
                deterministic=False)

        stop_date = datetime(2022, 5, 9, 7)  # year month day hour
        max_iter  = 100
        # print("Finished Initialization")
        subfolder = "mistery_cKG_n_obj_" + str(noise_objective) + "_n_c_" + str(noise_constraints)
        folder = "RESULTS"
        cwd = os.getcwd()
        path =cwd + "/" + folder + "/" + subfolder + '/it_' + str(rep) + '.csv'
        X, Y, C, recommended_val, optimum, Opportunity_cost = bo.run_optimization(max_iter = max_iter,
                                                                                  verbosity=False,
                                                                                  path=path,
                                                                                  stop_date=stop_date,
                                                                                  compute_OC=True,
                                                                                  evaluations_file=subfolder,
                                                                                  KG_dynamic_optimisation=True)

        print("Code Ended")
        print("X",X,"Y",Y, "C", C)
# function_caller_mistery_v2(rep=4)


