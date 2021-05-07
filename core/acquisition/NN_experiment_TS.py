import numpy as np
import GPyOpt
from GPyOpt.objective_examples.experiments2d import mistery, dropwave
from Real_Experiments.FC_Neural_Network.real_functions_caller import FC_NN_test_function
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
print("NN TS activate")
def function_caller_NN_TS(rep):
    rep = rep
    np.random.seed(rep)

    for noise in [1e-06]:

        function_rejected = True
        s = 0
        while function_rejected or s <= 1:
            # for i in range(2):
            try:
                RMITD_f = FC_NN_test_function()
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
        space = GPyOpt.Design_space(space=[{'name': 'var_1', 'type': 'continuous', 'domain': (0.0, 0.99)},
                                           {'name': 'var_2', 'type': 'continuous', 'domain': (0.0, 0.99)},
                                           {'name': 'var_2', 'type': 'continuous', 'domain': (5, 12)},
                                           {'name': 'var_2', 'type': 'continuous', 'domain': (5, 12)}])#GPyOpt.Design_space(space =[{'name': 'var_1', 'type': 'continuous', 'domain': (0,100)}])#
        n_f = 1
        n_c = 1
        model_f = multi_outputGP(output_dim = n_f,   exact_feval=[True]*n_f)
        model_c = multi_outputGP(output_dim = n_c,  noise_var=[1e-4]*n_c, exact_feval=[True]*n_c)


        # --- Aquisition optimizer
        #optimizer for inner acquisition function
        acq_opt = GPyOpt.optimization.AcquisitionOptimizer(optimizer='lbfgs', space=space, model = model_f, model_c=model_c)
        #
        # # --- Initial design
        #initial design
        initial_design = GPyOpt.experiment_design.initial_design('latin', space, 10)


        nz=1
        acquisition = TS(model=model_f, model_c=model_c , nz = nz,space=space, optimizer = acq_opt)
        evaluator = GPyOpt.core.evaluators.Sequential(acquisition)
        bo = BO(model_f, model_c, space, f, c, acquisition, evaluator, initial_design,
                tag_last_evaluation  =True,
                deterministic=True)


        max_iter  = 100
        # print("Finished Initialization")
        subfolder = "mistery_TS_" + str(noise)
        folder = "RESULTS"
        cwd = os.getcwd()
        path =cwd + "/" + folder + "/" + subfolder + '/it_' + str(rep) + '.csv'
        X, Y, C, recommended_val, optimum, Opportunity_cost = bo.run_optimization(max_iter=max_iter, verbosity=False,
                                                                                  path=path, evaluations_file=subfolder,
                                                                                  KG_dynamic_optimisation=False)
        print("Code Ended")

        print("X",X,"Y",Y, "C", C)


function_caller_NN_TS(rep=21)




