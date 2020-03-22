import numpy as np
import GPyOpt
from GPyOpt.objective_examples.experiments2d import mistery,  test_function_2, new_brannin
import GPy as GPy
from multi_objective import MultiObjective
from multi_outputGP import multi_outputGP
import matplotlib.pyplot as plt
import scipy
from continuous_KG import KG
from bayesian_optimisation import BO
import pandas as pd
import os

#ALWAYS check cost in
# --- Function to optimize

def function_caller_test_func_2(rep):
    np.random.seed(rep)

    # func2 = dropwave()
    test_function_2_f = test_function_2(sd=1e-6)

    # --- Attributes
    #repeat same objective function to solve a 1 objective problem
    f = MultiObjective([test_function_2_f.f])
    c = MultiObjective([test_function_2_f.c1, test_function_2_f.c2, test_function_2_f.c3])

    # --- Attributes
    #repeat same objective function to solve a 1 objective problem

    #c2 = MultiObjective([test_c2])
    # --- Space
    #define space of variables
    space =  GPyOpt.Design_space(space =[{'name': 'var_1', 'type': 'continuous', 'domain': (0,1)},{'name': 'var_2', 'type': 'continuous', 'domain': (0,1)}])#GPyOpt.Design_space(space =[{'name': 'var_1', 'type': 'continuous', 'domain': (0,100)}])#
    n_f = 1
    n_c = 3
    model_f = multi_outputGP(output_dim = n_f,   noise_var=[1e-6]*n_f, exact_feval=[True]*n_f)
    model_c = multi_outputGP(output_dim = n_c,  noise_var=[1e-6]*n_c, exact_feval=[True]*n_c)


    # --- Aquisition optimizer
    #optimizer for inner acquisition function
    acq_opt = GPyOpt.optimization.AcquisitionOptimizer(optimizer='lbfgs', space=space, model=model_f)
    #
    # # --- Initial design
    #initial design
    initial_design = GPyOpt.experiment_design.initial_design('latin', space, 10)


    acquisition = KG(model=model_f, model_c=model_c , space=space, optimizer = acq_opt)
    evaluator = GPyOpt.core.evaluators.Sequential(acquisition)
    bo = BO(model_f, model_c, space, f, c, acquisition, evaluator, initial_design)


    max_iter  = 45
    # print("Finished Initialization")
    X, Y, C, Opportunity_cost = bo.run_optimization(max_iter = max_iter,verbosity=False)
    print("Code Ended")

    data = {}
    data["Opportunity_cost"] = np.array(Opportunity_cost).reshape(-1)

    gen_file = pd.DataFrame.from_dict(data)
    folder = "RESULTS"
    subfolder = "test_function_2"
    cwd = os.getcwd()
    print("cwd", cwd)
    path = cwd + "/" + folder +"/"+ subfolder +'/it_' + str(rep)+ '.csv'
    if os.path.isdir(cwd + "/" + folder +"/"+ subfolder) == False:
        os.makedirs(cwd + "/" + folder +"/"+ subfolder)

    gen_file.to_csv(path_or_buf=path)

    print("X",X,"Y",Y, "C", C)

function_caller(rep=2)


