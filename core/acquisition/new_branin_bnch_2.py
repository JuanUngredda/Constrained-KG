import numpy as np
import GPyOpt
from GPyOpt.objective_examples.experiments2d import mistery,  test_function_2, new_brannin
import GPy as GPy
from multi_objective import MultiObjective
from multi_outputGP import multi_outputGP
import matplotlib.pyplot as plt
import scipy
from Hybrid_continuous_KG_penalised import KG
from nEI import nEI
from EI import EI
from bayesian_optimisation import BO
import pandas as pd
import os


#ALWAYS check cost in
# --- Function to optimize
print("new_branin activate")
def function_caller_new_branin_bnch_2(rep):
    rep = rep
    np.random.seed(rep)
    for noise in [ 1e-06]:
        # func2 = dropwave()
        new_brannin_f = new_brannin(sd=np.sqrt(noise))

        # --- Attributes
        # repeat same objective function to solve a 1 objective problem
        f = MultiObjective([new_brannin_f.f])
        c = MultiObjective([new_brannin_f.c])

        # --- Attributes
        # repeat same objective function to solve a 1 objective problem

        # c2 = MultiObjective([test_c2])
        # --- Space
        # define space of variables
        space =  GPyOpt.Design_space(space =[{'name': 'var_1', 'type': 'continuous', 'domain': (-5,10)},{'name': 'var_2', 'type': 'continuous', 'domain': (0,15)}])#GPyOpt.Design_space(space =[{'name': 'var_1', 'type': 'continuous', 'domain': (0,100)}])#
        n_f = 1
        n_c = 1
        model_f = multi_outputGP(output_dim=n_f, noise_var=[noise] * n_f, exact_feval=[True] * n_f)#, normalizer=True)
        model_c = multi_outputGP(output_dim=n_c, noise_var=[1e-10] * n_c, exact_feval=[True] * n_c)

        # --- Aquisition optimizer
        #optimizer for inner acquisition function
        type_anchor_points_logic = "max_objective"
        acq_opt = GPyOpt.optimization.AcquisitionOptimizer(optimizer="lbfgs",inner_optimizer='lbfgs',space=space, model=model_f, model_c=model_c,anchor_points_logic=type_anchor_points_logic)
        #
        # # --- Initial design
        #initial design
        initial_design = GPyOpt.experiment_design.initial_design('latin', space, 10)

        nz = 60 # (n_c+1)
        acquisition = KG(model=model_f, model_c=model_c , space=space, nz=nz, optimizer = acq_opt)

        if noise<1e-3:
            print("EI final step")
            Last_Step_acq = EI(model=model_f, model_c=model_c, space=space, nz=nz, optimizer=acq_opt)
        else:
            print("nEI final step")
            Last_Step_acq = nEI(model=model_f, model_c=model_c , space=space, nz=nz, optimizer = acq_opt)
        last_step_evaluator = GPyOpt.core.evaluators.Sequential(Last_Step_acq)
        evaluator = GPyOpt.core.evaluators.Sequential(acquisition)
        bo = BO(model_f, model_c, space, f, c, acquisition, evaluator, initial_design,
                ls_evaluator=last_step_evaluator,
                ls_acquisition = Last_Step_acq,
                tag_last_evaluation  =True,
                deterministic=True)


        max_iter  = 100
        # print("Finished Initialization")
        subfolder = "new_branin_penalised_hybrid_KG_" + str(noise)
        folder = "RESULTS"
        cwd = os.getcwd()
        path =cwd + "/" + folder + "/" + subfolder + '/it_' + str(rep) + '.csv'
        X, Y, C, recommended_val, optimum, Opportunity_cost = bo.run_optimization(max_iter = max_iter,verbosity=True, path=path,
                                                                                  evaluations_file=subfolder,
                                                                                  KG_dynamic_optimisation=True)

        print("Code Ended")
        print("X",X,"Y",Y, "C", C)
# function_caller_new_branin_bnch_2(rep=28)


