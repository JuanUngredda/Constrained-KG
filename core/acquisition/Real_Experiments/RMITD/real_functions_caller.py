
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from RMITD import RMITD_core


# class RMITD_test_function():
#     '''
#     Six hump camel function
#
#     :param bounds: the box constraints to define the domain in which the function is optimized.
#     :param sd: standard deviation, to generate noisy evaluations of the function.
#     '''
#
#     def __init__(self, bounds=None, sd=None):
#         self.input_dim = 4
#         if bounds is None:
#             self.bounds = [(0, 150), (0, 150), (0, 150), (0, 150)]
#         else:
#             self.bounds = bounds
#         self.min = np.nan
#         self.fmin = np.nan
#         self.sd = sd
#         self.name = 'RMITD'
#         self.simulation_run = 10000.0
#
#
#         self.eng = matlab.engine.start_matlab()
#         path = "/home/rawsys/matjiu/Constrained-KG/core/acquisition/Real_Experiments/RMITD"#"/home/juan/Documents/PhD/GitHub_Reps/Constrained-KG/core/acquisition/Real_Experiments/RMITD"##
#         os.chdir(path)
#         print(os.getcwd())
#
#     def f(self, x, offset=0, true_val=False):
#
#         if len(x.shape) == 1:
#             x = x.reshape(1, -1)
#
#         out_vals = []
#         for i in range(x.shape[0]):
#             x_i = np.array(x[i]).reshape(-1)
#             x_i = list(x_i)
#
#             input_value = matlab.double(x_i)[0]
#             seed = int(time.time()) * 1.0
#
#             if true_val:
#                 print("eval real value")
#                 reps = []
#                 for i in range(50):
#                     seed = int(time.time()) * 1.0
#                     out = self.eng.RMITD(input_value, self.simulation_run, seed, False)
#                     reps.append(out)
#
#                 print("np.mean(reps)", np.mean(reps))
#                 print("np.std(reps)", np.std(reps))
#                 print("MSE", np.std(reps) / np.sqrt(len(reps)))
#                 out_vals.append(np.mean(reps))
#             else:
#
#                 fn = self.eng.RMITD(input_value, self.simulation_run, seed, False)
#                 out_vals.append(fn)
#
#         out_vals = np.array(out_vals).reshape(-1)
#         out_vals = out_vals.reshape(-1, 1)
#
#         return out_vals
#
#     def c(self, x, true_val=False):
#
#         if len(x.shape) == 1:
#             x = x.reshape(1, -1)
#         b = x[:, 0]
#         r = x[:, 1:]
#
#         constraint = np.sum(r, axis=1) - b
#         constraint = np.array(constraint).reshape(-1)
#         return constraint.reshape(-1, 1)
#
#     def func_val(self, x):
#
#         Y = self.f(x, true_val=True)
#         C = self.c(x)
#         out = Y * (C <= 0)
#         out = np.array(out).reshape(-1)
#         return -out

class RMITD_test_function():
    '''
    Six hump camel function

    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''

    def __init__(self, bounds=None, sd=None):
        self.input_dim = 4
        if bounds is None:
            self.bounds = [(0, 150), (0, 150), (0, 150), (0, 150)]
        else:
            self.bounds = bounds
        self.min = np.nan
        self.fmin = np.nan
        self.sd = sd
        self.name = 'RMITD'
        self.simulation_run = 10000

    def f(self, x, offset=0, true_val=False):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        out_vals = []
        for i in range(x.shape[0]):
            input_value= np.array(x[i]).reshape(-1)

            seed = int(time.time()) * 1.0

            if true_val:
                print("eval real value")
                reps = []
                for i in range(5000):
                    seed = int(time.time())
                    out = RMITD_core(input_value, self.simulation_run, seed)
                    reps.append(out)

                print("np.mean(reps)", np.mean(reps))
                print("np.std(reps)", np.std(reps))
                print("MSE", np.std(reps) / np.sqrt(len(reps)))
                out_vals.append(np.mean(reps))
            else:

                fn = RMITD_core(input_value, self.simulation_run, seed)
                out_vals.append(fn)

        out_vals = np.array(out_vals).reshape(-1)
        out_vals = out_vals.reshape(-1, 1)

        return out_vals

    def c(self, x, true_val=False):

        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        b = x[:, 0]
        r = x[:, 1:]

        constraint = np.sum(r, axis=1) - b
        constraint = np.array(constraint).reshape(-1)
        return constraint.reshape(-1, 1)

    def func_val(self, x):

        Y = self.f(x, true_val=True)
        C = self.c(x)
        out = Y * (C <= 0)
        out = np.array(out).reshape(-1)
        return -out

fun = RMITD_test_function()

x = np.array([100,20,30,20])
print(fun.f(x))
print(fun.func_val(x))