import numpy as np
import GPyOpt

from Real_Experiments.LunarLander.real_functions_caller import constrained_LunarLanderBenchmark
import GPy as GPy
from multi_objective import MultiObjective
from multi_outputGP import multi_outputGP
import matplotlib.pyplot as plt
import scipy
from EI import EI
from bayesian_optimisation_benchmark import BO
import pandas as pd
import os

#ALWAYS check cost in
# --- Function to optimize

def function_caller_NN_EI(rep):


    np.random.seed(rep)
    m_terrains = 10

    lunarlander_class = constrained_LunarLanderBenchmark(m_terrains=m_terrains, minimum_reward= 200)
    # lunar_lander_constraints = lunarlander_class.c_builder()
    # --- Attributes
    #repeat same objective function to solve a 1 objective problem
    f = MultiObjective([lunarlander_class.f])
    c = MultiObjective([lunarlander_class.c])

    # --- Attributes
    #repeat same objective function to solve a 1 objective problem
    input_size = 12
    # --- Space
    #define space of variables
    space =  GPyOpt.Design_space(space =[{'name': 'var', 'type': 'continuous', 'domain': (0.0,2)}]*input_size)#GPyOpt.Design_space(space =[{'name': 'var_1', 'type': 'continuous', 'domain': (0,100)}])#
    n_f = 1
    n_c = 1
    model_f = multi_outputGP(output_dim = n_f,   noise_var=[1e-06]*n_f, exact_feval=[True]*n_f)
    model_c = multi_outputGP(output_dim = n_c,  noise_var=[1e-06]*n_c, exact_feval=[True]*n_c)


    # --- Aquisition optimizer
    #optimizer for inner acquisition function
    acq_opt = GPyOpt.optimization.AcquisitionOptimizer(optimizer='lbfgs', space=space, model=model_f, model_c=model_c)
    #
    # # --- Initial design
    #initial design
    init_num_samples = 50
    initial_design = GPyOpt.experiment_design.initial_design('latin', space, init_num_samples)

    nz = 1
    acquisition = EI(model=model_f, model_c=model_c , space=space, nz=nz, optimizer = acq_opt)
    evaluator = GPyOpt.core.evaluators.Sequential(acquisition)
    bo = BO(model_f, model_c, space, f, c, acquisition, evaluator, initial_design, expensive=True)


    max_iter  = 1000
    # print("Finished Initialization")

    subfolder = "LunarLander_cEI"
    folder = "RESULTS"
    cwd = os.getcwd()
    path = cwd + "/" + folder + "/" + subfolder + '/it_' + str(rep) + '.csv'

    X, Y, C, Opportunity_cost = bo.run_optimization(max_iter = max_iter,verbosity=False, path=path,
                                                    evaluations_file=subfolder, rep=rep)
    print("Code Ended")


    print("X",X,"Y",Y, "C", C)


# function_caller_NN_EI(1)

