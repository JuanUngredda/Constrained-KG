import numpy as np
import GPyOpt
from GPyOpt.objective_examples.experiments2d import mistery,  test_function_2, new_brannin
import GPy as GPy
from multi_objective import MultiObjective
from multi_outputGP import multi_outputGP
import matplotlib.pyplot as plt
import scipy
from Hybrid_continuous_KG_v2 import KG
from bayesian_optimisation import BO
from nEI import nEI
from EI import EI
import pandas as pd
import os
from datetime import datetime

#ALWAYS check cost in
# --- Function to optimize
print("test_fun_2 activate")
def function_caller_test_func_2_v2(rep):
    rep = rep
    np.random.seed(rep)
    for noise in [1e-06, 1.0]:
        # func2 = dropwave()
        test_function_2_f = test_function_2(sd=np.sqrt(noise))

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
        model_f = multi_outputGP(output_dim = n_f,   noise_var=[noise]*n_f, exact_feval=[True]*n_f)#, normalizer=True)
        model_c = multi_outputGP(output_dim = n_c,  noise_var=[1e-06]*n_c, exact_feval=[True]*n_c)

        # x = np.linspace(0,1,300)
        # y = np.linspace(0,1,300)
        # X,Y = np.meshgrid(x,y)
        # fval = []
        # data = np.zeros((2,1))
        # best_fval = -100
        # for i in x:
        #     for j in y:
        #
        #         data = np.array([[i,j]])
        #         yval, _ = f.evaluate(data)
        #         cval,_ = c.evaluate(data)
        #
        #         val= np.array(yval).reshape(-1)* np.product(np.array(np.array(cval)<0).reshape(-1))
        #         fval.append(val)
        #         # if val > best_fval:
        #         #     best_fval = val
        #         #     best_sol = np.array([i,j])
        #
        # fval = np.ma.masked_where(np.array(fval).reshape(Y.shape) == 0, np.array(fval).reshape(Y.shape))
        # cmap = plt.cm.inferno
        # cmap.set_bad(color='black')
        # fig, ax = plt.subplots()
        # ax.contourf(Y,X, fval, shading='auto',cmap=cmap, zorder=0)
        #
        # plt.scatter([0.2018], [0.833], color="green", marker="x" ,s=100, label="optimum", linewidths=3,zorder=1)
        # plt.scatter([0], [0], label="infeasible area", color='lightgray')
        # ax.tick_params(axis="x", labelsize=15)
        # ax.tick_params(axis="y", labelsize=15)
        # ax.set_facecolor('lightgray')
        # plt.legend(loc="upper right")
        # plt.savefig(
        #     "/home/juan/Documents/repos_data/Constrained-KG/RESULTS/plot_saved_data/plots/test_function.jpg",
        #     bbox_inches="tight")
        # plt.show()
        #
        # raise
        # --- Aquisition optimizer
        #optimizer for inner acquisition function
        type_anchor_points_logic = "max_objective"
        acq_opt = GPyOpt.optimization.AcquisitionOptimizer(optimizer="lbfgs",inner_optimizer='lbfgs',space=space, model=model_f, model_c=model_c,anchor_points_logic=type_anchor_points_logic)
        #
        # # --- Initial design
        #initial design
        initial_design = GPyOpt.experiment_design.initial_design('latin', space, 10)

        nz = 50 # (n_c+1)
        acquisition = KG(model=model_f, model_c=model_c , space=space, nz=nz, optimizer = acq_opt)
        if noise < 1e-3:
            Last_Step_acq = EI(model=model_f, model_c=model_c, space=space, nz=nz, optimizer=acq_opt)
        else:
            Last_Step_acq = nEI(model=model_f, model_c=model_c, space=space, nz=nz, optimizer=acq_opt)
        last_step_evaluator = GPyOpt.core.evaluators.Sequential(Last_Step_acq)
        evaluator = GPyOpt.core.evaluators.Sequential(acquisition)
        bo = BO(model_f, model_c, space, f, c, acquisition, evaluator, initial_design,
                ls_evaluator=last_step_evaluator,
                ls_acquisition = Last_Step_acq,
                tag_last_evaluation  =True,
                deterministic=False)

        stop_date = datetime(2022, 5, 8, 7)  # year month day hour
        max_iter  = 100
        # print("Finished Initialization")
        subfolder = "test_function_2_hybrid_KG_" + str(noise)
        folder = "RESULTS"
        cwd = os.getcwd()
        path =cwd + "/" + folder + "/" + subfolder + '/it_' + str(rep) + '.csv'
        X, Y, C, recommended_val, optimum, Opportunity_cost = bo.run_optimization(max_iter=max_iter, verbosity=False,
                                                                                  stop_date=stop_date,
                                                                                  path=path,
                                                                                  evaluations_file=subfolder,
                                                                                  KG_dynamic_optimisation=True )

        print("Code Ended")
        print("X",X,"Y",Y, "C", C)

# function_caller_test_func_2_v2(rep=4)


