import numpy as np
from GPyOpt.objective_examples.experiments2d import mistery_torch,  test_function_2, new_brannin
import matplotlib.pyplot as plt
import pandas as pd
import os
from time import time as time
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.priors.torch_priors import GammaPrior
#ALWAYS check cost in
# --- Function to optimize
from botorch.test_functions import Hartmann
import torch
from botorch.models import FixedNoiseGP, ModelListGP, SingleTaskGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.acquisition.objective import ConstrainedMCObjective
from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.acquisition.analytic import ConstrainedExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.exceptions import BadInitialCandidatesWarning
import time
from botorch.optim import optimize_acqf
from Transformation_Translation import Translate
from Last_Step import Constrained_Mean_Response
from botorch.utils.transforms import (
    concatenate_pending_points,
    match_batch_shape,
    t_batch_mode_transform,
)
import warnings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)
dtype = torch.double
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.test_functions.multi_objective import BraninCurrin, DTLZ1, DTLZ2, ZDT3, BNH
#

bounds = torch.tensor([[0, 0], [1.0, 1.0] ], device=device, dtype=dtype)
input_dim=2
ub = bounds[1, :]
lb = bounds[0, :]
delta = ub - lb
rand_x = torch.rand(10000, input_dim, device=device, dtype=dtype) * delta + lb

y_values = BNH().evaluate_true(rand_x )
magenta_point = torch.tensor([[0.75, 0.25]], device=device, dtype=dtype)
y_value_magenta = BNH().evaluate_true(magenta_point)

pareto_mask = is_non_dominated(-y_values)
pareto_front  = y_values[pareto_mask]

# plt.scatter(rand_x[:,0], rand_x[:,1])
# plt.scatter(magenta_point[:,0],magenta_point[:,1], color="magenta", s=150)
# plt.xlabel("X1")
# plt.ylabel("X2")
# plt.title("Solution Space")
# plt.tight_layout()
# # plt.savefig("/home/juan/Documents/repos_data/BOPL_Cornell-Warwick/PLOTS_PRESENTATIONS/BNH_SS.pdf")
# plt.show()

#plt.scatter(y_values[:,0], y_values[:,1])
plt.scatter(pareto_front [:,0],pareto_front [:,1])
plt.xlabel("min F1")
plt.ylabel("min F2")
plt.title("Objective Space")
plt.tight_layout()
plt.savefig("/home/juan/Documents/repos_data/BOPL_Cornell-Warwick/PLOTS_PRESENTATIONS/BNH_PF_2d.pdf")
plt.show()

from botorch.utils.multi_objective.pareto import is_non_dominated
# from botorch.test_functions.multi_objective import BraninCurrin, DTLZ1, DTLZ2, ZDT3, BNH
#
# bounds = torch.tensor([[0, 0,0,0], [1.0, 1.0,1.0,1.0] ], device=device, dtype=dtype)
# input_dim=4
# ub = bounds[1, :]
# lb = bounds[0, :]
# delta = ub - lb
# rand_x = torch.rand(100000, input_dim, device=device, dtype=dtype) * delta + lb
#
# y_values = DTLZ2(dim =4, num_objectives=3).gen_pareto_front(2000)
#
# # magenta_point = torch.tensor([[0.75, 0.25]], device=device, dtype=dtype)
#
# # pareto_mask = is_non_dominated(y_values)
# # pareto_front  = y_values[pareto_mask]
# # pareto_solutions = rand_x[pareto_mask]
# plt.ion()
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(y_values[:,0], y_values[:,1], y_values[:,2])
# ax.set_xlabel('max F1')
# ax.set_ylabel('max F2')
# ax.set_zlabel('max F3')
# ax.set_title("Objective Space")
# ax.view_init(elev=10., azim=60)
# plt.tight_layout()
# plt.legend()
# plt.savefig("/home/juan/Documents/repos_data/BOPL_Cornell-Warwick/PLOTS_PRESENTATIONS/DTLZ3_PF.pdf")
# plt.show()

