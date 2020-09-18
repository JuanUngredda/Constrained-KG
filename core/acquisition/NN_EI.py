import numpy as np
import GPyOpt
from Real_Experiments.FC_Neural_Network.real_functions_caller import FC_NN_test_function

import GPy as GPy
from multi_objective import MultiObjective
from multi_outputGP import multi_outputGP
import matplotlib.pyplot as plt
import scipy
from continuous_KG import KG
from bayesian_optimisation_benchmark import BO
import pandas as pd
import os

#ALWAYS check cost in
# --- Function to optimize

def function_caller_NN_EI(rep):
    
    for rep in range(15):
        rep = rep + 4
        np.random.seed(rep)
        # func2 = dropwave()

        NN_f = FC_NN_test_function()

        # --- Attributes
        #repeat same objective function to solve a 1 objective problem
        f = MultiObjective([NN_f.f])
        c = MultiObjective([NN_f.c])

        # --- Attributes
        #repeat same objective function to solve a 1 objective problem

        #c2 = MultiObjective([test_c2])
        # --- Space
        #define space of variables
        space =  GPyOpt.Design_space(space =[{'name': 'var_1', 'type': 'continuous', 'domain': (0.0,0.99)},
                                             {'name': 'var_2', 'type': 'continuous', 'domain': (0.0,0.99)},
                                             {'name': 'var_2', 'type': 'continuous', 'domain': (5,12)},
                                             {'name': 'var_2', 'type': 'continuous', 'domain': (5,12)}])#GPyOpt.Design_space(space =[{'name': 'var_1', 'type': 'continuous', 'domain': (0,100)}])#
        n_f = 1
        n_c = 1
        noise = 0.02**2
        model_f = multi_outputGP(output_dim = n_f,   noise_var=[noise]*n_f, exact_feval=[False]*n_f, normalizer=True)
        model_c = multi_outputGP(output_dim = n_c,  noise_var=[1e-21]*n_c, exact_feval=[True]*n_c)


        # --- Aquisition optimizer
        #optimizer for inner acquisition function
        acq_opt = GPyOpt.optimization.AcquisitionOptimizer(optimizer='lbfgs', space=space, model=model_f, model_c=model_c)
        #
        # # --- Initial design
        #initial design
        init_num_samples = 5
        initial_design = GPyOpt.experiment_design.initial_design('latin', space, init_num_samples)

        nz = 1
        acquisition = KG(model=model_f, model_c=model_c , space=space, nz=nz, optimizer = acq_opt)
        evaluator = GPyOpt.core.evaluators.Sequential(acquisition)
        bo = BO(model_f, model_c, space, f, c, acquisition, evaluator, initial_design, expensive=True)


        max_iter  = 30
        # print("Finished Initialization")
        X, Y, C, Opportunity_cost = bo.run_optimization(max_iter = max_iter,verbosity=False)
        print("Code Ended")

        C_bool = np.product(np.concatenate(C, axis=1) < 0, axis=1)
        data = {}
        print("C",C)
        print("np.array(Opportunity_cost).reshape(-1)",np.array(Opportunity_cost).reshape(-1))
        print("np.array(Y).reshape(-1)",np.array(Y).reshape(-1))
        print("np.array(C_bool).reshape(-1)",np.array(C_bool).reshape(-1))
        data["Opportunity_cost"] = np.concatenate((np.zeros(init_num_samples), np.array(Opportunity_cost).reshape(-1)))
        data["Y"] = np.array(Y).reshape(-1)
        data["C_bool"] = np.array(C_bool).reshape(-1)

        gen_file = pd.DataFrame.from_dict(data)
        folder = "RESULTS"
        subfolder = "NN_EI"
        cwd = os.getcwd()

        path = cwd + "/" + folder +"/"+ subfolder +'/it_' + str(rep)+ '.csv'

        if os.path.isdir(cwd + "/" + folder +"/"+ subfolder) == False:
            os.makedirs(cwd + "/" + folder +"/"+ subfolder)

        gen_file.to_csv(path_or_buf=path)

        print("X",X,"Y",Y, "C", C)


# for r in range(20):
#     function_caller_NN_EI(rep=r)


