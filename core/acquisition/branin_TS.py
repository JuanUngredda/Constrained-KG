import numpy as np
import GPyOpt
from GPyOpt.objective_examples.experiments2d import mistery,  test_function_2, new_brannin
import GPy as GPy
from multi_objective import MultiObjective
from multi_outputGP import multi_outputGP
import matplotlib.pyplot as plt
import scipy
from Thompson_Sampling import TS
from bayesian_optimisation import BO
import pandas as pd
import os

#ALWAYS check cost in
# --- Function to optimize

print("new branin TS activate")
def function_caller_new_brannin_TS(rep):
    rep = rep
    np.random.seed(rep)

    for noise in [1.0]:
        # func2 = dropwave()
        noise_objective = noise
        noise_constraints = (0.1) ** 2
        mistery_f = new_brannin(sd_obj=np.sqrt(noise_objective), sd_c=np.sqrt(noise_constraints))

        # --- Attributes
        # repeat same objective function to solve a 1 objective problem
        f = MultiObjective([mistery_f.f])
        c = MultiObjective([mistery_f.c])

        # --- Attributes
        # repeat same objective function to solve a 1 objective problem

        # c2 = MultiObjective([test_c2])
        # --- Space
        # define space of variables
        space = GPyOpt.Design_space(space=[{'name': 'var_1', 'type': 'continuous', 'domain': (-5, 10)},
                                           {'name': 'var_2', 'type': 'continuous', 'domain': (0, 15)}])
        n_f = 1
        n_c = 1
        model_f = multi_outputGP(output_dim=n_f, noise_var=[noise_objective] * n_f, exact_feval=[True] * n_f)
        model_c = multi_outputGP(output_dim=n_c, noise_var=[noise_constraints] * n_c, exact_feval=[True] * n_c)

        # --- Aquisition optimizer
        #optimizer for inner acquisition function
        acq_opt = GPyOpt.optimization.AcquisitionOptimizer(optimizer='lbfgs', space=space, model=model_f, model_c=model_c)
        #
        # # --- Initial design
        initial_design = GPyOpt.experiment_design.initial_design('latin', space, 10)

        nz = 1
        acquisition = TS(model=model_f, model_c=model_c, nz=nz, space=space, optimizer=acq_opt)
        evaluator = GPyOpt.core.evaluators.Sequential(acquisition)
        bo = BO(model_f, model_c, space, f, c, acquisition, evaluator, initial_design,
                tag_last_evaluation=True,
                deterministic=True)

        max_iter = 100
        # print("Finished Initialization")
        subfolder = "new_brannin_TS_n_obj_" + str(noise_objective) + "_n_c_" + str(noise_constraints)
        folder = "RESULTS"
        cwd = os.getcwd()
        path = cwd + "/" + folder + "/" + subfolder + '/it_' + str(rep) + '.csv'
        X, Y, C, recommended_val, optimum, Opportunity_cost = bo.run_optimization(max_iter=max_iter, verbosity=False,
                                                                                  path=path, evaluations_file=subfolder,
                                                                                  KG_dynamic_optimisation=False)
        print("Code Ended")

        print("X", X, "Y", Y, "C", C)


function_caller_new_brannin_TS(rep=15)


