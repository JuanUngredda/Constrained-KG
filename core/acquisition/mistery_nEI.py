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
import warnings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)
dtype = torch.double

def function_caller_mistery_nEI(rep):
    for noise in [ 1e-6,1.0]:

        torch.manual_seed(rep)
        NOISE_SE = noise
        N_BATCH = 50
        initial_points = 10
        MC_SAMPLES = 256


        problem_class = mistery_torch(sd=np.sqrt(0.0))
        bounds = torch.tensor([[0.0, 0.0], [5.0, 5.0] ], device=device, dtype=dtype)
        input_dim = problem_class.input_dim

        def objective_function(X):
            return torch.tensor(problem_class.f(X), device=device, dtype=dtype)

        def outcome_constraint(X):
            return torch.tensor(problem_class.c(X), device=device, dtype=dtype)

        def weighted_obj(X):
            """Feasibility weighted objective; zero if not feasible."""
            return objective_function(X) * (outcome_constraint(X) <= 0).type_as(X)

        #train_yvar = torch.tensor(NOISE_SE ** 2, device=device, dtype=dtype)
        train_cvar = torch.tensor(1e-6, device=device, dtype=dtype)

        def obj_callable(Z):
            return Z[..., 0]

        def constraint_callable(Z):
            return Z[..., 1]

        # define a feasibility-weighted objective for optimization
        constrained_obj = ConstrainedMCObjective(
            objective=obj_callable,
            constraints=[constraint_callable],
        )

        def generate_initial_data(n=10):
            # generate training data
            ub = bounds[1, :]
            lb = bounds[0, :]
            delta = ub - lb
            train_x = torch.rand(n, input_dim, device=device, dtype=dtype) * delta + lb
            exact_obj = objective_function(train_x).unsqueeze(-1)  # add output dimension
            exact_con = outcome_constraint(train_x).unsqueeze(-1)  # add output dimension
            train_obj = exact_obj + NOISE_SE * torch.randn_like(exact_obj)
            train_con = exact_con
            best_observed_value = weighted_obj(train_x).max().item()
            return train_x, train_obj, train_con, best_observed_value

        def initialize_model(train_x, train_obj, train_con, state_dict=None):
            # define models for objective and constraint
            covar_module = ScaleKernel(RBFKernel(
                    ard_num_dims=train_x.shape[-1]
                ),)
            model_obj = SingleTaskGP(train_x, train_obj, outcome_transform=Translate_Object, covar_module=covar_module)#FixedNoiseGP(train_x, train_obj, train_cvar.expand_as(train_obj)).to(train_x) #
            model_con = FixedNoiseGP(train_x, train_con, train_cvar.expand_as(train_con)).to(train_x)
            # combine into a multi-output GP model
            model = ModelListGP(model_obj, model_con)
            mll = SumMarginalLogLikelihood(model.likelihood, model)
            # load state dict if it is passed
            if state_dict is not None:
                model.load_state_dict(state_dict)
            return mll, model

        def optimize_acqf_and_get_observation(acq_func, diagnostics = False):
            """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
            # optimize
            BATCH_SIZE = 1

            candidates, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=bounds,
                q=BATCH_SIZE,
                num_restarts=10,
                raw_samples=512,  # used for intialization heuristic
                options={"batch_limit": 5, "maxiter": 200},
            )
            # observe new values
            if diagnostics:

                ub = bounds[1, :]
                lb = bounds[0, :]
                delta = ub - lb
                rand_x = torch.rand(10000, input_dim, device=device, dtype=dtype) * delta + lb#torch.rand(10000, 1, input_dim, device=device, dtype=dtype) * delta + lb
                print("rand_x ",rand_x .shape)
                acq_func_val = weighted_obj(rand_x)
                # acq_func_val = acq_func.forward(rand_x).unsqueeze(-1)
                # acq_func_val = acq_func_val.squeeze(-1).detach().numpy()
                # print("max val", np.max(acq_func_val),"min", np.min(acq_func_val))
                #plt.scatter(rand_x[:, :, 0].squeeze(-1), rand_x[:,:, 1].squeeze(-1), c = acq_func_val)
                plt.scatter(rand_x[:, 0].squeeze(-1), rand_x[:, 1].squeeze(-1), c=acq_func_val)
                plt.scatter(train_x_nei[:,0],train_x_nei[:,1], color="red")
                plt.scatter(train_x_nei[-1, 0], train_x_nei[-1, 1], color="magenta")
                plt.show()

                plt.plot( best_observed_all_nei)
                plt.show()

            new_x = candidates.detach()
            exact_obj = objective_function(new_x).unsqueeze(-1)  # add output dimension
            exact_con = outcome_constraint(new_x).unsqueeze(-1)  # add output dimension
            new_obj = exact_obj + NOISE_SE * torch.randn_like(exact_obj)
            new_con = exact_con
            return new_x, new_obj, new_con

        warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
        warnings.filterwarnings('ignore', category=RuntimeWarning)

        verbose = True

        best_observed_all_nei, best_observed_all_ei= [], []

        # average over multiple trials
        best_observed_nei, best_observed_ei = [], []

        # call helper functions to generate initial training data and initialize model
        train_x_nei, train_obj_nei, train_con_nei, best_observed_value_nei = generate_initial_data(n=initial_points)
        Translate_Object = Translate(m=1, Y=train_obj_nei)
        train_x_ei, train_obj_ei, train_con_ei, best_observed_value_ei =train_x_nei, train_obj_nei, train_con_nei, best_observed_value_nei
        mll_nei, model_nei = initialize_model(train_x_nei, train_obj_nei, train_con_nei)
        best_observed_nei.append(best_observed_value_nei)

        # run N_BATCH rounds of BayesOpt after the initial random batch

        for iteration in range(1, N_BATCH + 1):
            t0 = time.time()

            # fit the models
            fit_gpytorch_model(mll_nei)

            # define the qEI and qNEI acquisition modules using a QMC sampler
            qmc_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)

            qNEI = qNoisyExpectedImprovement(
                model=model_nei,
                X_baseline=train_x_nei,
                sampler=qmc_sampler,
                objective=constrained_obj,
            )


            # optimize and get new observation
            new_x_nei, new_obj_nei, new_con_nei = optimize_acqf_and_get_observation(qNEI, diagnostics = False)

            # update training points
            train_x_nei = torch.cat([train_x_nei, new_x_nei])
            train_obj_nei = torch.cat([train_obj_nei, new_obj_nei])
            train_con_nei = torch.cat([train_con_nei, new_con_nei])

            # update progress
            best_value_nei = weighted_obj(train_x_nei).max().item()
            best_observed_nei.append(best_value_nei)

            # reinitialize the models so they are ready for fitting on next iteration
            # use the current state dict to speed up fitting

            mll_nei, model_nei = initialize_model(
                train_x_nei,
                train_obj_nei,
                train_con_nei,
                model_nei.state_dict(),
            )

            Last_Step = Constrained_Mean_Response(
                model=model_nei,
                best_f=0.0, #dummy variable really, doesnt do anything since I take max/min of posterior mean
                objective=constrained_obj
                )


            last_x_nei, last_obj_nei, last_con_nei = optimize_acqf_and_get_observation(Last_Step)


            # update progress
            best_value_nei = weighted_obj(train_x_nei).max().item()
            best_value_last_step = weighted_obj(last_x_nei).max().item()

            # print("last_x_nei,",last_x_nei)
            # print("best_value_nei",best_value_nei)
            # print("best_value_last_step ",best_value_last_step )
            # print("[best_value_nei , best_value_last_step ]",[best_value_nei , best_value_last_step ])
            best_value = np.max([best_value_nei , best_value_last_step ])
            # print("best_value",best_value)
            t1 = time.time()

            if verbose:
                print(
                    f"\niteration {iteration:>2}: best_value (qNEI) = "
                    f"({best_value:>4.2f}), "
                    f"time = {t1 - t0:>4.2f}.", end=""
                )
            else:
                print(".", end="")

            best_observed_all_nei.append(best_value)
            data = {}
            print(" best_observed_all_nei",  best_observed_all_nei)
            data["Opportunity_cost"] = np.array(best_observed_all_nei).reshape(-1)

            gen_file = pd.DataFrame.from_dict(data)
            folder = "RESULTS"
            subfolder = "mistery_nEI_" + str(noise)
            cwd = os.getcwd()
            path = cwd + "/" + folder +"/"+ subfolder +'/it_' + str(rep)+ '.csv'
            if os.path.isdir(cwd + "/" + folder +"/"+ subfolder) == False:
                print("directory results: ", cwd + "/" + folder +"/"+ subfolder)
                os.makedirs(cwd + "/" + folder +"/"+ subfolder)

            gen_file.to_csv(path_or_buf=path)




# function_caller_mistery_nEI(rep=2)


