import numpy as np
import GPyOpt
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

def function_caller_NN_TS(rep):

    np.random.seed(rep)
    # func2 = dropwave()
    function_rejected = True
    s = 0
    while function_rejected or s<=1:
    #for i in range(2):
        try:
            RMITD_f = FC_NN_test_function()
            function_rejected = False
            s+=1
        except:
            function_rejected = True
            print("function_rejected check path inside function")
            pass
    import time


    # --- Attributes
    #repeat same objective function to solve a 1 objective problem
    f = MultiObjective([RMITD_f.f])
    c = MultiObjective([RMITD_f.c])

    # --- Attributes
    #repeat same objective function to solve a 1 objective problem

    #c2 = MultiObjective([test_c2])
    # --- Space
    #define space of variables
    space =  GPyOpt.Design_space(space =[{'name': 'var_1', 'type': 'continuous', 'domain': (0.0,0.99)},
                                         {'name': 'var_2', 'type': 'continuous', 'domain': (0.0,0.99)},
                                         {'name': 'var_2', 'type': 'continuous', 'domain': (5,12)},
                                         {'name': 'var_2', 'type': 'continuous', 'domain': (5,12)}])#GPyOpt.Design_space(space =[{'name': 'var_1', 'type': 'continuous', 'domain': (0,100)}])#

    # x = GPyOpt.experiment_design.initial_design('random', space, 5)
    x = np.array([[0.5,0.5, 5, 5]])
    x = np.repeat(x, 100, axis=0)
    # discrete_dims = [2, 3]
    # x[:, discrete_dims] = np.round(x[:, discrete_dims])
    start = time.time()
    fval = RMITD_f.f(x)
    print("fval", fval, "mean", np.mean(fval), "std",np.std(fval),"MSE", np.std(fval)/np.sqrt(len(fval)))
    stop = time.time()
    print("time performance", stop - start)

    start = time.time()
    cval = RMITD_f.c(x)
    print("cval", cval, "mean", np.mean(cval), "MSE", np.std(cval) / np.sqrt(len(cval)))
    stop = time.time()
    print("time time", stop-start)

    # data = {}
    # data["Y"] = np.array(fval).reshape(-1)
    # data["C"] = np.array(cval).reshape(-1)
    #
    # gen_file = pd.DataFrame.from_dict(data)
    #
    # import pathlib
    # checkpoint_dir = pathlib.Path(__file__).parent.absolute()
    # checkpoint_dir = str(checkpoint_dir) + "/NN_stats/"
    #
    # path = checkpoint_dir +'/YC.csv'
    # if os.path.isdir(checkpoint_dir ) == False:
    #     os.makedirs(checkpoint_dir )
    #
    # gen_file.to_csv(path_or_buf=path)
    # path = checkpoint_dir  + '/X.csv'
    # np.savetxt(path, x, delimiter=",")
    raise

    n_f = 1
    n_c = 1
    noise = 0.002**2
    model_f = multi_outputGP(output_dim = n_f,   exact_feval=[False]*n_f)
    model_c = multi_outputGP(output_dim = n_c,  noise_var=[1e-04]*n_c, exact_feval=[True]*n_c)


    # --- Aquisition optimizer
    #optimizer for inner acquisition function
    acq_opt = GPyOpt.optimization.AcquisitionOptimizer(optimizer='lbfgs', space=space, model=model_f, model_c=model_c)
    #
    # # --- Initial design
    #initial design
    init_num_samples = 18
    initial_design = GPyOpt.experiment_design.initial_design('latin', space, init_num_samples)

    nz = 1
    acquisition = TS(model=model_f, model_c=model_c , space=space, nz=nz, optimizer = acq_opt)
    evaluator = GPyOpt.core.evaluators.Sequential(acquisition)
    bo = BO(model_f, model_c, space, f, c, acquisition, evaluator, initial_design, expensive=True, deterministic=False)


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
    subfolder = "NN_TS"
    cwd = os.getcwd()
    print("cwd", cwd)
    path = cwd + "/" + folder +"/"+ subfolder +'/it_' + str(rep)+ '.csv'
    if os.path.isdir(cwd + "/" + folder +"/"+ subfolder) == False:
        os.makedirs(cwd + "/" + folder +"/"+ subfolder)

    gen_file.to_csv(path_or_buf=path)

    print("X",X,"Y",Y, "C", C)


function_caller_NN_TS(rep=21)


