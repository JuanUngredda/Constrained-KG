import numpy as np
import GPyOpt
from GPyOpt.objective_examples.experiments2d import mistery, dropwave
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

def function_caller(rep):
    np.random.seed(rep)

    class GP_test():
        """
    A toy function GP

    ARGS
     min: scalar defining min range of inputs
     max: scalar defining max range of inputs
     seed: int, RNG seed
     x_dim: designs dimension
     a_dim: input dimensions
     xa: n*d matrix, points in space to eval testfun
     NoiseSD: additive gaussaint noise SD

    RETURNS
     output: vector of length nrow(xa)
     """

        def __init__(self, xamin, xamax, seed=11, x_dim=1):
            self.seed = seed
            self.dx = x_dim
            self.da = 0
            self.dxa = x_dim
            self.xmin = np.array([xamin for i in range(self.dxa)])
            self.xmax = np.array([xamax for i in range(self.dxa)])
            vr = 4.
            ls = 10
            self.HP =  [vr,ls]
            self.KERNEL = GPy.kern.RBF(input_dim=self.dxa, variance=vr, lengthscale=([ls] * self.dxa), ARD=True)
            self.generate_function()

        def __call__(self, xa, noise_std=1e-2):
            assert len(xa.shape) == 2, "xa must be an N*d matrix, each row a d point"
            assert xa.shape[1] == self.dxa, "Test_func: wrong dimension inputed"

            xa = self.check_input(xa)

            ks = self.KERNEL.K(xa, self.XF)
            out = np.dot(ks, self.invCZ)

            E = np.random.normal(0, noise_std, xa.shape[0])

            return (out.reshape(-1, 1) + E.reshape(-1, 1))

        def generate_function(self):
            print("Generating test function")
            np.random.seed(self.seed)

            self.XF = np.random.uniform(size=(50, self.dxa)) * (self.xmax - self.xmin) + self.xmin


            mu = np.zeros(self.XF.shape[0])

            C = self.KERNEL.K(self.XF, self.XF)

            Z = np.random.multivariate_normal(mu, C).reshape(-1, 1)
            invC = np.linalg.inv(C + np.eye(C.shape[0]) * 1e-3)

            self.invCZ = np.dot(invC, Z)

        def check_input(self, x):
            if not x.shape[1] == self.dxa or (x > self.xmax).any() or (x < self.xmin).any():
                raise ValueError("x is wrong dim or out of bounds")
            return x


    # func2 = dropwave()
    mistery_f =mistery(sd=1e-6)

    # --- Attributes
    #repeat same objective function to solve a 1 objective problem
    f = MultiObjective([mistery_f.f])
    c = MultiObjective([mistery_f.c])

    # --- Attributes
    #repeat same objective function to solve a 1 objective problem

    #c2 = MultiObjective([test_c2])
    # --- Space
    #define space of variables
    space =  GPyOpt.Design_space(space =[{'name': 'var_1', 'type': 'continuous', 'domain': (0,5)},{'name': 'var_2', 'type': 'continuous', 'domain': (0,5)}])#GPyOpt.Design_space(space =[{'name': 'var_1', 'type': 'continuous', 'domain': (0,100)}])#
    n_f = 1
    n_c = 1
    model_f = multi_outputGP(output_dim = n_f,   noise_var=[1e-6]*n_f, exact_feval=[True]*n_f)
    model_c = multi_outputGP(output_dim = n_c,  noise_var=[1e-6]*n_c, exact_feval=[True]*n_c)


    # --- Aquisition optimizer
    #optimizer for inner acquisition function
    acq_opt = GPyOpt.optimization.AcquisitionOptimizer(optimizer='lbfgs', space=space)
    #
    # # --- Initial design
    #initial design
    initial_design = GPyOpt.experiment_design.initial_design('latin', space, 10)


    acquisition = KG(model=model_f, model_c=model_c , space=space, optimizer = acq_opt)
    evaluator = GPyOpt.core.evaluators.Sequential(acquisition)
    bo = BO(model_f, model_c, space, f, c, acquisition, evaluator, initial_design)


    max_iter  = 45
    # print("Finished Initialization")
    X, Y, C, Opportunity_cost = bo.run_optimization(max_iter = max_iter,verbosity=True)
    print("Code Ended")

    data = {}
    data["Opportunity_cost"] = np.array(Opportunity_cost).reshape(-1)

    gen_file = pd.DataFrame.from_dict(data)
    folder = "RESULTS"
    subfolder = "Mistery_bnch"
    cwd = os.getcwd()
    print("cwd", cwd)
    path = cwd + "/" + folder +"/"+ subfolder +'/it_' + str(rep)+ '.csv'
    if os.path.isdir(cwd + "/" + folder +"/"+ subfolder) == False:
        os.makedirs(cwd + "/" + folder +"/"+ subfolder)

    gen_file.to_csv(path_or_buf=path)

    print("X",X,"Y",Y, "C", C)

#function_caller(rep=2)


