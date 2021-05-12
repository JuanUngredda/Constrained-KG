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
from nEI import nEI
from EI import EI
import os
from datetime import datetime
import time
import tensorflow as tf
#ALWAYS check cost in
# --- Function to optimize
print("NN TS activate")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
def function_caller_NN_cKG(rep):

    for i in range(10):
        rep = rep +i
        np.random.seed(rep)

        function_rejected = True
        s = 0
        while function_rejected or s <= 1:
            # for i in range(2):

            try:
                threshold = 2.6e-2 #seconds
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
        space = GPyOpt.Design_space(space=[{'name': 'var_1', 'type': 'continuous', 'domain': (1e-6, 0.1)},  #Learning rate
                                           {'name': 'var_2', 'type': 'continuous', 'domain': (0.0, 0.99)},  #Drop-out rate 1
                                           {'name': 'var_3', 'type': 'continuous', 'domain': (0.0, 0.99)},  #Drop-out rate 2
                                           {'name': 'var_4', 'type': 'continuous', 'domain': (0.0, 0.99)},# Drop-out rate 3
                                           {'name': 'var_5', 'type': 'continuous', 'domain': (3, 12)},  # units 1
                                           {'name': 'var_6', 'type': 'continuous', 'domain': (3, 12)},# units 2
                                           {'name': 'var_7', 'type': 'continuous', 'domain': (3, 12)}])# units 3

        x = np.array([[1e-3, 0.3, 0.3,0.3, 5,5,5],
                      [1e-3, 0.3, 0.3,0.3, 7,7,7],
                      [1e-3, 0.3, 0.3,0.3, 10,10,10]])

        # cval = RMITD_f.f(x)
        # print("cval",cval, "mean", np.mean(cval), "std", np.std(cval))
        start = time.time()
        cval = RMITD_f.c(x)
        print("cval", cval)
        # raise
        if np.all(cval<0):
            print("restriction is not doing anything")

            raise
        raise
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
        initial_design = GPyOpt.experiment_design.initial_design('latin', space, 14)

        nz = 60 # (n_c+1)
        acquisition = KG(model=model_f, model_c=model_c , space=space, nz=nz, optimizer = acq_opt)

        Last_Step_acq = nEI(model=model_f, model_c=model_c , space=space, nz=nz, optimizer = acq_opt)
        last_step_evaluator = GPyOpt.core.evaluators.Sequential(Last_Step_acq)
        evaluator = GPyOpt.core.evaluators.Sequential(acquisition)

       #  X_init = np.array([[1.07151786e-02, 6.01071429e-01, 3.18214286e-01 ,2.47500000e-01,  9.75000000e+00, 1.03928571e+01, 3.32142857e+00],
       #           [3.21435357e-02, 1.76785714e-01 ,1.76785714e-01 ,8.83928571e-01 , 1.16785714e+01 ,9.10714286e+00 ,1.03928571e+01],
       #           [3.92863214e-02, 7.42500000e-01, 6.01071429e-01, 3.18214286e-01 , 5.89285714e+00 ,3.96428571e+00 ,3.96428571e+00],
       #           [6.78574643e-02, 8.13214286e-01, 7.42500000e-01, 7.42500000e-01 , 7.82142857e+00 ,9.75000000e+00 ,1.16785714e+01],
       #           [4.64291071e-02 ,6.71785714e-01, 1.06071429e-01, 1.76785714e-01,  4.60714286e+00, 4.60714286e+00, 7.17857143e+00],
       #           [7.50002500e-02 ,5.30357143e-01, 2.47500000e-01 ,1.06071429e-01 , 5.25000000e+00 ,5.89285714e+00 ,5.89285714e+00],
       #           [5.35718929e-02 ,1.06071429e-01, 5.30357143e-01, 6.01071429e-01 , 3.32142857e+00 ,8.46428571e+00 ,9.75000000e+00],
       #           [8.92858214e-02 ,3.88928571e-01, 8.83928571e-01, 3.53571429e-02 , 6.53571429e+00 ,1.10357143e+01 ,6.53571429e+00],
       #           [3.57239286e-03 ,4.59642857e-01, 4.59642857e-01, 5.30357143e-01 , 8.46428571e+00 ,3.32142857e+00 ,7.82142857e+00],
       #           [8.21430357e-02 ,8.83928571e-01, 8.13214286e-01, 3.88928571e-01 , 1.10357143e+01 ,7.82142857e+00 ,9.10714286e+00],
       #           [2.50007500e-02 ,9.54642857e-01, 3.53571429e-02, 4.59642857e-01 , 3.96428571e+00 ,6.53571429e+00 ,5.25000000e+00],
       #           [1.78579643e-02 ,3.53571429e-02, 6.71785714e-01, 6.71785714e-01 , 9.10714286e+00 ,1.16785714e+01 ,4.60714286e+00],
       #           [9.64286071e-02 ,2.47500000e-01, 3.88928571e-01, 9.54642857e-01 , 7.17857143e+00 ,5.25000000e+00 ,8.46428571e+00],
       #           [6.07146786e-02 ,3.18214286e-01, 9.54642857e-01, 8.13214286e-01 , 1.03928571e+01 ,7.17857143e+00 ,1.10357143e+01],
       #           [3.05437452e-02 ,5.66454977e-01, 4.70179650e-01, 4.18535096e-01 , 7.22063887e+00 ,3.66974987e+00 ,6.47967381e+00]])
       # # #
       #  Y_init = [np.array([[0.1144    ],
       # [0.1123    ],
       # [0.50620002],
       # [0.1142    ],
       # [0.3075    ],
       # [0.40869999],
       # [0.0976    ],
       # [0.1123    ],
       # [0.83700001],
       # [0.0931    ],
       # [0.1145    ],
       # [0.1125    ],
       # [0.1112    ],
       # [0.1076    ],
       # [0.096     ]])]
       # # #
       #  C_init = [np.array([[ 1.7756976 ],
       # [ 2.7852118 ],
       # [-0.9650333 ],
       # [ 2.37180871],
       # [-0.9986938 ],
       # [-0.94701512],
       # [ 0.53443157],
       # [ 0.91552611],
       # [ 0.23070045],
       # [ 1.97647892],
       # [-1.0990923 ],
       # [ 2.11170844],
       # [-0.23868296],
       # [ 1.72708951],
       # [-0.54592923]])]

        # X_init = X_init,
        # Y_init = Y_init,
        # C_init = C_init,

        bo = BO(model_f, model_c, space, f, c, acquisition, evaluator,  initial_design,
                ls_evaluator=last_step_evaluator,
                ls_acquisition = Last_Step_acq,
                tag_last_evaluation  =True,
                deterministic=False)

        stop_date = datetime(2022, 5, 10, 7) # year month day hour
        max_iter  = 50
        # print("Finished Initialization")
        subfolder = "NN_hybrid_KG"
        folder = "RESULTS"
        cwd = os.getcwd()
        path =cwd + "/" + folder + "/" + subfolder + '/it_' + str(rep) + '.csv'
        X, Y, C, recommended_val, optimum, Opportunity_cost = bo.run_optimization(max_iter=max_iter, verbosity=False,stop_date= stop_date,
                                                                                  path=path,compute_OC=False,
                                                                                  evaluations_file=subfolder,
                                                                                  KG_dynamic_optimisation=True)

        print("Code Ended")
        print("X",X,"Y",Y, "C", C)
function_caller_NN_cKG(rep=0)




