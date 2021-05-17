import numpy as np
import GPyOpt
from GPyOpt.objective_examples.experiments2d import mistery, dropwave
from Real_Experiments.FC_Neural_Network.real_functions_caller import FC_NN_test_function
import GPy as GPy
from multi_objective import MultiObjective
from multi_outputGP import multi_outputGP
import matplotlib.pyplot as plt
import scipy
from Hybrid_continuous_KG_v2 import KG
from bayesian_optimisation import BO
import pandas as pd
# from nEI import nEI
from nEI import nEI
# from EI import EI
import os
from datetime import datetime
import time
import tensorflow as tf
#ALWAYS check cost in
# --- Function to optimize
print("NN TS activate")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
def function_caller_NN_nEI(rep_base):
    rep_base = rep_base + 100
    for it in range(1):
        rep = rep_base + 10**(it)
        np.random.seed(rep)

        function_rejected = True
        s = 0
        while function_rejected or s <= 1:
            # for i in range(2):

            try:
                threshold = 1.1e-2  #seconds
                RMITD_f = FC_NN_test_function(max_time=threshold)
                function_rejected = False
                s += 1
            except:
                function_rejected = True
                print("function_rejected check path inside function")
                pass

        # --- Attributes
        #repeat same objective function to solve a 1 objective problem
        f = MultiObjective([RMITD_f.f])
        c = MultiObjective([RMITD_f.c])


        # --- Attributes
        #repeat same objective function to solve a 1 objective problem

        #c2 = MultiObjective([test_c2])
        # --- Space
        #define space of variables
        space = GPyOpt.Design_space(space=[{'name': 'var_1', 'type': 'continuous', 'domain': (0, 1)},  #Learning rate
                                           {'name': 'var_2', 'type': 'continuous', 'domain': (0, 1)},  #Drop-out rate 1
                                           {'name': 'var_3', 'type': 'continuous', 'domain': (0, 1)},  #Drop-out rate 2
                                           {'name': 'var_3', 'type': 'continuous', 'domain': (0, 1)},  # Drop-out rate 2
                                           {'name': 'var_5', 'type': 'continuous', 'domain': (3, 12)},  # units 1
                                           {'name': 'var_7', 'type': 'continuous', 'domain': (3, 12)},  # units 1
                                           {'name': 'var_7', 'type': 'continuous', 'domain': (3, 12)},  # units 1
                                           {'name': 'var_7', 'type': 'continuous', 'domain': (0, 1)},  # beta 1 rate
                                           {'name': 'var_7', 'type': 'continuous', 'domain': (0, 1)}])# beta 2 rate





        n_f = 1
        n_c = 1
        model_f = multi_outputGP(output_dim = n_f,   exact_feval=[False]*n_f)
        model_c = multi_outputGP(output_dim = n_c,  noise_var=[1e-4]*n_c, exact_feval=[True]*n_c)


        # --- Aquisition optimizer
        #optimizer for inner acquisition function
        type_anchor_points_logic = "max_objective"
        acq_opt = GPyOpt.optimization.AcquisitionOptimizer(optimizer="lbfgs",inner_optimizer='lbfgs',space=space, model=model_f, model_c=model_c,anchor_points_logic=type_anchor_points_logic)
        #
        # # --- Initial design
        #initial design
        initial_design = GPyOpt.experiment_design.initial_design('latin', space, 18)

        nz = 1
        acquisition = nEI(model=model_f, model_c=model_c, space=space, nz=nz, optimizer=acq_opt)
        evaluator = GPyOpt.core.evaluators.Sequential(acquisition)
        bo = BO(model_f, model_c, space, f, c, acquisition, evaluator, initial_design,
                tag_last_evaluation=True,
                deterministic=True)


        stop_date = datetime(2022, 5, 16, 7)# year month day hour
        max_iter  = 50
        # print("Finished Initialization")
        subfolder = "NN_nEI_"
        folder = "RESULTS"
        cwd = os.getcwd()
        path =cwd + "/" + folder + "/" + subfolder + '/it_' + str(rep) + '.csv'
        X, Y, C, recommended_val, optimum, Opportunity_cost = bo.run_optimization(max_iter=max_iter, verbosity=False,stop_date= stop_date,
                                                                                  path=path,compute_OC=False,
                                                                                  evaluations_file=subfolder,
                                                                                  KG_dynamic_optimisation=False)

        print("Code Ended")
        print("X",X,"Y",Y, "C", C)
# function_caller_NN_nEI(21)



