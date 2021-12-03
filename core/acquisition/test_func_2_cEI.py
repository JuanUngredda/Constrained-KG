import numpy as np
import GPyOpt
from GPyOpt.objective_examples.experiments2d import mistery,  test_function_2, new_brannin
import GPy as GPy
from multi_objective import MultiObjective
from multi_outputGP import multi_outputGP
import matplotlib.pyplot as plt
import scipy
from EI import EI
from bayesian_optimisation import BO
import pandas as pd
import os
from datetime import datetime
#ALWAYS check cost in
# --- Function to optimize
print("test funs cEI activate")
def function_caller_test_func_2_cEI(rep):
    rep = rep
    np.random.seed(rep)
    for noise in [1.0]:
        # func2 = dropwave()
        noise_objective = noise
        noise_constraints = 1e-06#(0.1) ** 2
        test_function_2_f = test_function_2(sd_obj=np.sqrt(noise_objective), sd_c=np.sqrt(noise_constraints))

        # --- Attributes
        #repeat same objective function to solve a 1 objective problem
        f = MultiObjective([test_function_2_f.f])
        c = MultiObjective([test_function_2_f.c1, test_function_2_f.c2, test_function_2_f.c3])

        space = GPyOpt.Design_space(space=[{'name': 'var_1', 'type': 'continuous', 'domain': (0, 1)},
                                           {'name': 'var_2', 'type': 'continuous', 'domain': (0, 1)}])
        plot_design = GPyOpt.experiment_design.initial_design('latin', space, 100000)

        step_function_c1 = np.zeros(len(plot_design))
        step_function_c2 = np.zeros(len(plot_design))
        step_function_c3 = np.zeros(len(plot_design))

        c1 = test_function_2_f.c1(plot_design)
        step_function_c1[c1<0] = 1
        c2 = test_function_2_f.c2(plot_design)
        step_function_c2[c2 < 0] = 1
        c3 = test_function_2_f.c3(plot_design)
        step_function_c3[c3 < 0] = 1

        fig, (ax1, ax2) = plt.subplots(1, 2)

        fig.suptitle("constraint 1: $((x1 - 3)^2 + (x2 + 2)^2)*np.exp(-x2^7)-12$")
        ax1.set_title("continuous response")
        ax1.scatter(plot_design[:,0], plot_design[:,1], c=c1)
        ax1.scatter(0.5,0.85, color="red", marker="x")

        ax2.set_title("binary response")
        ax2.scatter(plot_design[:, 0], plot_design[:, 1], c=step_function_c1)
        ax2.scatter(0.5, 0.85, color="red", marker="x")
        plt.show()

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle("constraint 2: $10*x1 + x2 -7$")
        ax1.set_title("continuous response")
        ax1.scatter(plot_design[:, 0], plot_design[:, 1], c=c2)
        ax1.scatter(0.5, 0.85, color="red", marker="x")

        ax2.set_title("binary response")
        ax2.scatter(plot_design[:, 0], plot_design[:, 1], c=step_function_c2)
        ax2.scatter(0.5, 0.85, color="red", marker="x")
        plt.show()

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle("constraint 3: $(x1 - 0.5)^2    + (x2 - 0.5)^2 -0.2$")
        ax1.set_title("continuous response")
        ax1.scatter(plot_design[:, 0], plot_design[:, 1], c=c3)
        ax1.scatter(0.5, 0.85, color="red", marker="x")

        ax2.set_title("binary response")
        ax2.scatter(plot_design[:, 0], plot_design[:, 1], c=step_function_c3)
        ax2.scatter(0.5, 0.85, color="red", marker="x")
        plt.show()


        overall_step_function = step_function_c1 * step_function_c2 * step_function_c3
        plt.title("overall step surface")
        plt.scatter(plot_design[:, 0], plot_design[:, 1], c=overall_step_function)
        plt.scatter(0.5, 0.85, color="red", marker="x")
        plt.show()
        raise
        # --- Attributes
        #repeat same objective function to solve a 1 objective problem

        #c2 = MultiObjective([test_c2])
        # --- Space
        #define space of variables
        space =  GPyOpt.Design_space(space =[{'name': 'var_1', 'type': 'continuous', 'domain': (0,1)},{'name': 'var_2', 'type': 'continuous', 'domain': (0,1)}])#GPyOpt.Design_space(space =[{'name': 'var_1', 'type': 'continuous', 'domain': (0,100)}])#
        n_f = 1
        n_c = 3
        model_f = multi_outputGP(output_dim = n_f,   noise_var=[noise_objective]*n_f, exact_feval=[True]*n_f)
        model_c = multi_outputGP(output_dim = n_c,  noise_var=[noise_constraints]*n_c, exact_feval=[True]*n_c)


        # --- Aquisition optimizer
        #optimizer for inner acquisition function
        acq_opt = GPyOpt.optimization.AcquisitionOptimizer(optimizer='lbfgs', space=space, model=model_f, model_c=model_c)
        #
        # # --- Initial design
        #initial design
        initial_design = GPyOpt.experiment_design.initial_design('latin', space, 10)


        nz=1
        acquisition = EI(model=model_f, model_c=model_c , space=space, nz=nz, optimizer = acq_opt)
        evaluator = GPyOpt.core.evaluators.Sequential(acquisition)
        bo = BO(model_f, model_c, space, f, c, acquisition, evaluator, initial_design,
                deterministic=False)


        max_iter  = 100
        # print("Finished Initialization")
        stop_date = datetime(2022, 5, 9, 7)  # year month day hour
        subfolder = "test_fun_n_obj_" + str(noise_objective) + "_n_c_" + str(noise_constraints)
        folder = "RESULTS"
        cwd = os.getcwd()
        path =cwd + "/" + folder + "/" + subfolder + '/it_' + str(rep) + '.csv'
        X, Y, C, recommended_val, optimum, Opportunity_cost = bo.run_optimization(max_iter = max_iter,verbosity=False,
                                                                                  stop_date = stop_date,
                                                                                  path=path,evaluations_file=subfolder,
                                                                                  KG_dynamic_optimisation=False)

        print("X",X,"Y",Y, "C", C)

function_caller_test_func_2_cEI(rep=21)


