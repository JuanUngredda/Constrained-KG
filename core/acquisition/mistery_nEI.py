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
from scipy.stats import norm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)
dtype = torch.double

def function_caller_mistery_nEI(it):

    repepetitions = [it, it + 20]
    for rep in repepetitions:
        np.random.seed(rep)
        for noise in [ 1.0]:
            torch.manual_seed(rep)
            NOISE_SE = noise
            NOISE_SE_constraint = np.sqrt(0.1)
            N_BATCH = 100
            initial_points = 10
            MC_SAMPLES = 250


            problem_class = mistery_torch()#Noise Included in the loop
            bounds = torch.tensor([[0, 0], [5.0, 5.0] ], device=device, dtype=dtype)
            input_dim = problem_class.input_dim

            def objective_function(X):
                return torch.tensor(problem_class.f(X), device=device, dtype=dtype)

            def outcome_constraint(X):
                return torch.tensor(problem_class.c(X), device=device, dtype=dtype)

            def weighted_obj(X):
                """Feasibility weighted objective; zero if not feasible."""
                return objective_function(X) * (outcome_constraint(X) <= 0).type_as(X)

            #train_yvar = torch.tensor(NOISE_SE ** 2, device=device, dtype=dtype)
            train_cvar = torch.tensor(NOISE_SE_constraint**2, device=device, dtype=dtype)
            train_yvar = torch.tensor(noise, device=device, dtype=dtype)

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
                train_con = exact_con + NOISE_SE_constraint * torch.randn_like(exact_con)
                best_observed_value = weighted_obj(train_x).max().item()
                return train_x, train_obj, train_con, best_observed_value

            def recommended_value(X, model):
                posterior_means  = model.posterior(X).mean
                posterior_var = model.posterior(X).variance

                posterior_means = posterior_means.detach().numpy()
                posterior_var = posterior_var.detach().numpy()

                pf = probability_feasibility_multi_gp(mu=posterior_means[:,1:], var=posterior_var[:,1:])

                objective_mean = posterior_means[:,0]
                predicted_fit = objective_mean.reshape(-1) * pf.reshape(-1)

                return predicted_fit

            def probability_feasibility_multi_gp(mu, var):
                Fz = []
                print("mu.shape[1]", mu.shape[1])
                for m in range(mu.shape[1]):
                    Fz.append(probability_feasibility(mu[:,m], var[:,m]))
                Fz = np.product(Fz, axis=0)
                return Fz

            def probability_feasibility(mean=None, var=None, l=0):

                std = np.sqrt(var).reshape(-1, 1)
                mean = mean.reshape(-1, 1)
                norm_dist = norm(mean, std)
                Fz = norm_dist.cdf(l)
                return Fz



            def initialize_model(train_x, train_obj, train_con, state_dict=None):
                # define models for objective and constraint
                covar_module = ScaleKernel(RBFKernel(
                        ard_num_dims=train_x.shape[-1]
                    ),)

                #
                model_obj = FixedNoiseGP(train_x, train_obj, train_yvar.expand_as(train_obj), covar_module=covar_module).to(train_x)  #SingleTaskGP(train_x, train_obj, outcome_transform=Translate_Object, covar_module=covar_module)##
                model_con = FixedNoiseGP(train_x, train_con, train_cvar.expand_as(train_con)).to(train_x)
                # combine into a multi-output GP model
                model = ModelListGP(model_obj, model_con)
                mll = SumMarginalLogLikelihood(model.likelihood, model)
                # load state dict if it is passed
                if state_dict is not None:
                    model.load_state_dict(state_dict)
                return mll, model

            def optimize_acqf_and_get_observation(acq_func, check_acqu_val_sample=None, diagnostics = False):
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
                new_x = candidates.detach()
                # if check_acqu_val_sample is None:
                #     new_x = candidates.detach()
                # else:
                #     new_x = candidates.detach()
                #     new_acq_func_val = acq_func.forward(new_x).unsqueeze(-1)
                #     new_acq_func_val = new_acq_func_val.squeeze(-1).detach().numpy()
                #     last_acq_func_val = acq_func.forward(check_acqu_val_sample).unsqueeze(-1)
                #     last_acq_func_val = last_acq_func_val.squeeze(-1).detach().numpy()
                #     if last_acq_func_val >= new_acq_func_val:
                #         new_x = check_acqu_val_sample
                #     else:
                #         new_x = new_x

                if diagnostics:

                    ub = bounds[1, :]
                    lb = bounds[0, :]
                    delta = ub - lb
                    rand_x = torch.rand(500, input_dim, device=device, dtype=dtype) * delta + lb#torch.rand(10000, 1, input_dim, device=device, dtype=dtype) * delta + lb
                    print("rand_x ",rand_x .shape)
                    #acq_func_val = weighted_obj(rand_x)
                    # acq_func_val = acq_func.forward(rand_x).unsqueeze(-1)
                    # acq_func_val = acq_func_val.squeeze(-1).detach().numpy()
                    acq_func_val = acq_func.forward(new_x).unsqueeze(-1)
                    acq_func_val = acq_func_val.squeeze(-1).detach().numpy()
                    print("new_x", new_x, "acq_func_val",acq_func_val)

                    acq_func_val = acq_func.forward(check_acqu_val_sample).unsqueeze(-1)
                    acq_func_val = acq_func_val.squeeze(-1).detach().numpy()
                    print("last_x", check_acqu_val_sample, "last_acq_func_val",acq_func_val)

                    # print("max val", torch.max(acq_func_val),"min", torch.min(acq_func_val))
                    # plt.scatter(rand_x[:, :, 0].squeeze(-1), rand_x[:,:, 1].squeeze(-1), c = acq_func_val)
                    # plt.scatter(rand_x[:, 0].squeeze(-1), rand_x[:, 1].squeeze(-1), c=acq_func_val)
                    # plt.scatter(train_x_nei[:,0],train_x_nei[:,1], color="red")
                    # plt.scatter(new_x[:,0], new_x[:,1], color="magenta", s=100)
                    # plt.show()

                    # plt.plot( best_observed_all_nei)
                    # plt.show()

                exact_obj = objective_function(new_x).unsqueeze(-1)  # add output dimension
                exact_con = outcome_constraint(new_x).unsqueeze(-1)  # add output dimension
                new_obj = exact_obj + NOISE_SE * torch.randn_like(exact_obj)
                new_con = exact_con  + NOISE_SE_constraint * torch.randn_like(exact_con)
                return new_x, new_obj, new_con



            warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
            warnings.filterwarnings('ignore', category=RuntimeWarning)

            verbose = True

            best_observed_all_nei_GP, best_observed_all_ei= [], []
            best_observed_all_nei_sampled, best_observed_all_ei_sampled = [], []
            # average over multiple trials
            best_observed_nei, best_observed_ei = [], []
            # print("best value")
            # True_value = True_func_opt(model=weighted_obj,
            #         best_f=0.0,  # dummy variable really, doesnt do anything since I only take max/min of posterior mean
            #         objective=constrained_obj)
            #
            # optimize_acqf_and_get_observation(True_value, diagnostics=True)
            # raise
            # call helper functions to generate initial training data and initialize model
            train_x_nei, train_obj_nei, train_con_nei, best_observed_value_nei = generate_initial_data(n=initial_points)
            Translate_Object = Translate(m=1, Y=train_obj_nei)
            #train_x_ei, train_obj_ei, train_con_ei, best_observed_value_ei =train_x_nei, train_obj_nei, train_con_nei, best_observed_value_nei
            mll_nei, model_nei = initialize_model(train_x_nei, train_obj_nei, train_con_nei)
            best_observed_nei.append(best_observed_value_nei)

            # run N_BATCH rounds of BayesOpt after the initial random batch
            Last_Step = Constrained_Mean_Response(
                model=model_nei,
                best_f=0.0,  # dummy variable really, doesnt do anything since I only take max/min of posterior mean
                objective=constrained_obj
            )

            last_x_nei, last_obj_nei, last_con_nei = optimize_acqf_and_get_observation(Last_Step, diagnostics=False)

            for iteration in range(1, N_BATCH + 1):
                t0 = time.time()

                # fit the models
                fit_gpytorch_model(mll_nei)

                # define the qEI and qNEI acquisition modules using a QMC sampler
                qmc_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)

                # qNEI_baseline_X = torch.cat([train_x_nei, match_batch_shape(last_x_nei, train_x_nei)], dim=-2)
                qNEI = qNoisyExpectedImprovement(
                    model=model_nei,
                    X_baseline=train_x_nei,
                    sampler=qmc_sampler,
                    objective=constrained_obj,
                )

                # optimize and get new observation
                new_x_nei, new_obj_nei, new_con_nei = optimize_acqf_and_get_observation(qNEI)

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
                    best_f=0.0, #dummy variable really, doesnt do anything since I only take max/min of posterior mean
                    objective=constrained_obj
                    )


                last_x_nei, last_obj_nei, last_con_nei = optimize_acqf_and_get_observation(Last_Step, diagnostics = False)

                # update progress
                value_recommended_design_GP = weighted_obj(last_x_nei)
                best_value_GP = np.array(value_recommended_design_GP).reshape(-1)

                GP_vals_sampled = recommended_value(train_x_nei, model_nei)
                value_recommended_design = weighted_obj(train_x_nei[np.argmax(GP_vals_sampled )])

                best_value_sampled = np.array(value_recommended_design).reshape(-1)
                t1 = time.time()

                if verbose:
                    print("best value", best_value_GP)
                    # print(
                    #     f"\niteration {iteration:>2}: best_value (qNEI) = "
                    #     f"({best_value:>4.2f}), "
                    #     f"time = {t1 - t0:>4.2f}.", end=""
                    # )
                else:
                    print(".", end="")

                best_observed_all_nei_sampled.append(best_value_sampled)
                best_observed_all_nei_GP.append(best_value_GP)
                data = {}
                print(" best_observed_all_nei",  best_observed_all_nei_GP)
                data["OC GP mean"] = np.array(best_observed_all_nei_GP).reshape(-1)
                data["OC GP sampled"] = np.array(best_observed_all_nei_sampled).reshape(-1)

                gen_file = pd.DataFrame.from_dict(data)
                folder = "RESULTS"
                subfolder = "mistery_nEI_n_obj_" + str(NOISE_SE) + "_n_c_" + str(NOISE_SE_constraint)
                cwd = os.getcwd()
                path = cwd + "/" + folder +"/"+ subfolder +'/it_' + str(rep)+ '.csv'
                print("directory results: ", cwd + "/" + folder + "/" + subfolder)
                if os.path.isdir(cwd + "/" + folder +"/"+ subfolder) == False:

                    os.makedirs(cwd + "/" + folder +"/"+ subfolder)

                gen_file.to_csv(path_or_buf=path)



# function_caller_mistery_nEI(1)


