import numpy as np
import GPyOpt
from GPyOpt.objective_examples.experiments2d import mistery, test_function_2, new_brannin
import GPy as GPy
from multi_objective import MultiObjective
from multi_outputGP import multi_outputGP
import matplotlib.pyplot as plt
import scipy
from Hybrid_continuous_KG_penalty_adjusted import KG
from bayesian_optimisation_penalised_adjusted import BO
from nEI import nEI
from EI import EI
import pandas as pd
import os
from datetime import datetime

# ALWAYS check cost in
# --- Function to optimize
print("mistery activate")

expdict = {0: [0, 1, 2, 3, 4, 17, 21, 25],
           1: [5, 6, 7, 8, 18, 22, 26],
           2: [9, 10, 11, 12, 19, 23, 27, 29],
           3: [13, 14, 15, 16, 20, 24, 28, 30], }


def function_caller_mistery_penalty_adjusted(it):
    for rep in expdict[it]:
        np.random.seed(rep)
        for noise in [1e-04]:
            for m in [-1000000, -12.65, -37.08, 1.46, None]:
                # func2 = dropwave()
                noise_objective = noise
                noise_constraints = (1e-04) ** 2
                mistery_f = mistery(sd_obj=np.sqrt(noise_objective), sd_c=np.sqrt(noise_constraints), offset=0)

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
                n_f = 1
                n_c = 1
                if m is not None:
                    M_value = np.array([m])
                else:
                    M_value = None
                model_f = multi_outputGP(output_dim=n_f, noise_var=[noise_objective] * n_f, exact_feval=[True] * n_f)
                model_c = multi_outputGP(output_dim=n_c, noise_var=[noise_constraints] * n_c, exact_feval=[True] * n_c)

                # --- Aquisition optimizer
                # optimizer for inner acquisition function
                type_anchor_points_logic = "max_objective"
                acq_opt = GPyOpt.optimization.AcquisitionOptimizer(optimizer="lbfgs", inner_optimizer='lbfgs',
                                                                   space=space,
                                                                   model=model_f, model_c=model_c,
                                                                   anchor_points_logic=type_anchor_points_logic)
                #
                # # --- Initial design
                # initial design
                initial_design = GPyOpt.experiment_design.initial_design('latin', space, 10)

                nz = 60  # (n_c+1)
                acquisition = KG(model=model_f, model_c=model_c, space=space, nz=nz, optimizer=acq_opt)

                evaluator = GPyOpt.core.evaluators.Sequential(acquisition)
                bo = BO(model_f, model_c, space, f, c, acquisition, evaluator, initial_design,
                        deterministic=False, predefined_penalty=M_value)

                stop_date = datetime(2030, 5, 10, 7)  # year month day hour
                max_iter = 20
                # print("Finished Initialization")
                subfolder = "mistery_cKG_penalty_adjusted_n_obj_" + str(noise_objective) + "_n_c_" + str(
                    noise_constraints)
                folder = "RESULTS"
                cwd = os.getcwd()
                if M_value is None:
                    path = cwd + "/" + folder + "/" + subfolder + '/continuous/it_' + str(rep) + '.csv'
                else:
                    path = cwd + "/" + folder + "/" + subfolder + '/' + str(M_value[0]) + '/it_' + str(rep) + '.csv'

                file_name = 'it_' + str(rep) + '.csv'
                try:
                    if not os.path.isdir(path):
                        os.makedirs(path, exist_ok=True)

                    if not os.path.isfile(path + "/" + file_name):
                        bo.run_optimization(max_iter=max_iter,
                                            verbosity=False,
                                            path=path + "/" + file_name,
                                            stop_date=stop_date,
                                            rep=rep,
                                            evaluations_file=subfolder,
                                            KG_dynamic_optimisation=True)
                except:
                    pass
                print("Code Ended")

# function_caller_mistery_penalty_adjusted(it=0)
