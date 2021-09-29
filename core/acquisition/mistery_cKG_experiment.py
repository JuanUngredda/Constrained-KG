import numpy as np
import GPyOpt
from GPyOpt.objective_examples.experiments2d import mistery,  test_function_2, new_brannin
import GPy as GPy
from multi_objective import MultiObjective
from multi_outputGP import multi_outputGP
import matplotlib.pyplot as plt
import scipy
from Hybrid_continuous_KG_v2 import KG
from nEI import nEI
from EI import EI
from bayesian_optimisation import BO
import pandas as pd
import os
from datetime import datetime

#ALWAYS check cost in
# --- Function to optimize
print("mistery activate")
def function_caller_mistery_v2(it):
    repepetitions = [it, it + 20]
    for rep in repepetitions:
        np.random.seed(rep)
        for noise in [1]:
            # func2 = dropwave()
            noise_objective = noise
            noise_constraints = (np.sqrt(0.1)) ** 2
            mistery_f = mistery(sd_obj=np.sqrt(noise_objective), sd_c=np.sqrt(noise_constraints))


            # --- Attributes
            # repeat same objective function to solve a 1 objective problem
            f = MultiObjective([mistery_f.f])
            c = MultiObjective([mistery_f.c])


            # --- Attributes
            # repeat same objective function to solve a 1 objective problem

            # c2 = MultiObjective([test_c2])
            # --- Space
            # define space of variables
            space = GPyOpt.Design_space(space=[{'name': 'var_1', 'type': 'continuous', 'domain': (0, 5)},
                                               {'name': 'var_2', 'type': 'continuous', 'domain': (0,
                                                                                                  5)}])  # GPyOpt.Design_space(space =[{'name': 'var_1', 'type': 'continuous', 'domain': (0,100)}])#

            verbose = False
            if verbose:
                mistery_f_denoised = mistery(sd_obj=0, sd_c=0)
                plot_design = GPyOpt.experiment_design.initial_design('latin', space, 100000)

                fvals = mistery_f_denoised.f(plot_design)  # f.evaluate(plot_design)
                cvals = mistery_f_denoised.c(plot_design)  # c.evaluate(plot_design)
                cvalsbool = np.array(cvals).reshape(-1) < 0
                fvals = np.array(fvals).reshape(-1)
                X_argmax  = np.atleast_2d(plot_design[np.argmax(fvals[cvalsbool])])

                max_fvals = []
                for _ in range(100):
                    fvals_noised = mistery_f.f(X_argmax) #f.evaluate(plot_design)
                    # cvals = mistery_f.c(X_argmax) #c.evaluate(plot_design)
                    # cvalsbool = np.array(cvals).reshape(-1) < 0
                    fvals_noised = np.array(fvals_noised).reshape(-1)
                    max_fvals.append(fvals_noised)

                signal_to_noise_ratio = np.abs(np.mean(max_fvals))/np.std(max_fvals)
                print("signal_to_noise_ratio",signal_to_noise_ratio)
                # print("max constrained value", np.max(fvals[cvalsbool]))
                # print("max", np.max(fvals), "min", np.min(fvals))
                # print("max", np.max(cvals), "min", np.min(cvals))

                plt.scatter(plot_design[:,0][cvalsbool],plot_design[:,1][cvalsbool], c=fvals[cvalsbool])
                plt.show()
                raise


            n_f = 1
            n_c = 1
            model_f = multi_outputGP(output_dim=n_f, noise_var=[noise_objective] * n_f, exact_feval=[True] * n_f)
            model_c = multi_outputGP(output_dim=n_c, noise_var=[noise_constraints] * n_c, exact_feval=[True] * n_c)

            # --- Aquisition optimizer
            #optimizer for inner acquisition function
            type_anchor_points_logic = "max_objective"
            acq_opt = GPyOpt.optimization.AcquisitionOptimizer(optimizer="lbfgs",inner_optimizer='lbfgs',space=space, model=model_f, model_c=model_c,anchor_points_logic=type_anchor_points_logic)
            #
            # # --- Initial design
            #initial design
            initial_design = GPyOpt.experiment_design.initial_design('latin', space, 10)

            nz = 20 # (n_c+1)
            acquisition = KG(model=model_f, model_c=model_c , space=space, nz=nz, optimizer = acq_opt)


            # Last_Step_acq = EI(model=model_f, model_c=model_c, space=space, nz=nz, optimizer=acq_opt)
            # last_step_evaluator = GPyOpt.core.evaluators.Sequential(Last_Step_acq)
           #  X_init = np.array([[4.75,       4.25      ],
           #                   [3.75   ,    0.75      ],
           #                   [0.75   ,    1.75      ],
           #                   [1.75   ,    2.25      ],
           #                   [4.25   ,    3.25      ],
           #                   [1.25   ,    0.25      ],
           #                   [0.25   ,    3.75      ],
           #                   [3.25   ,    1.25      ],
           #                   [2.75   ,    2.75      ],
           #                   [2.25   ,    4.75      ],
           #                   [2.13413426 ,1.3603269 ],
           #                   [2.59097193 ,1.31497054],
           #                   [2.42045612 ,1.39429358],
           #                   [2.25943463 ,1.73224851],
           #                   [2.42300214 ,1.57616307],
           #                   [2.48244551 ,1.60732955],
           #                   [2.55200197 ,1.70676824],
           #                   [3.17943266 ,2.24769729],
           #                   [3.08372938 ,2.5819454 ],
           #                   [3.19293485 ,2.2131103 ],
           #                   [2.97930961 ,2.35106384],
           #                   [2.90149632 ,2.45703171],
           #                   [2.96944669 ,2.44136762],
           #                   [2.85212147 ,2.55512475],
           #                   [2.93050247 ,2.45802541],
           #                   [3.093959   ,2.37270014],
           #                   [2.94790762 ,2.45768595],
           #                   [2.87436072 ,2.50989179],
           #                   [2.79451169 ,2.51542546],
           #                   [2.80157684 ,2.32344828],
           #                   [2.70840447 ,2.20560584],
           #                   [2.63533457 ,2.20455297],
           #                   [2.60875242 ,2.23153824],
           #                   [2.59197707 ,2.2376022 ],
           #                   [2.62726918 ,2.23571093],
           #                   [2.86539591 ,2.18625216],
           #                   [2.82210632 ,2.19699379],
           #                   [2.8064866  ,2.22624543],
           #                   [2.8502085  ,2.1913562 ],
           #                   [2.79280862 ,2.24519927],
           #                   [2.87335645 ,2.21943706],
           #                   [2.91970809 ,2.18261118],
           #                   [2.98511373 ,2.19628047],
           #                   [2.88415265 ,2.23877187],
           #                   [2.86739265 ,2.30045937],
           #                   [2.92265889 ,2.24949148],
           #                   [2.89169006 ,2.2551087 ],
           #                   [2.798006   ,2.31861975],
           #                   [2.76368351 ,2.35142737],
           #                   [2.76489136 ,2.36411183],
           #                   [2.74757136 ,2.38509848],
           #                   [2.76604964 ,2.36661932],
           #                   [2.81578767 ,2.32447362],
           #                   [2.77943158 ,2.34301823],
           #                   [2.75660935 ,2.36242297],
           #                   [2.75686007 ,2.35672411],
           #                   [2.85013435 ,2.28406806],
           #                   [2.83188249 ,2.27816021],
           #                   [2.80056553 ,2.34551295],
           #                   [2.77089415 ,2.34963441],
           #                   [2.81432504 ,2.3180047 ],
           #                   [2.83934886 ,2.26458729],
           #                   [2.7862044  ,2.34582496],
           #                   [2.78995196 ,2.34427367]])
           #  Y_init = [np.array([[-13.51089387],
           # [  0.34889704],
           # [ 16.04219994],
           # [ 15.50607926],
           # [  3.00359198],
           # [ 11.82230836],
           # [  9.99248063],
           # [ 10.32433011],
           # [ 21.15157922],
           # [ -4.83529894],
           # [  9.35741271],
           # [  9.62625464],
           # [ 10.79914693],
           # [ 12.75662111],
           # [ 12.26695101],
           # [ 13.70236401],
           # [ 14.21899279],
           # [ 19.72968618],
           # [ 16.31492504],
           # [ 18.07342906],
           # [ 19.71782449],
           # [ 20.7648935 ],
           # [ 21.77267764],
           # [ 20.13641826],
           # [ 19.49930484],
           # [ 20.05563834],
           # [ 18.94959095],
           # [ 20.4505024 ],
           # [ 21.00658471],
           # [ 21.44868359],
           # [ 21.26404267],
           # [ 20.73260176],
           # [ 22.03951094],
           # [ 20.48654884],
           # [ 18.4951515 ],
           # [ 20.90954063],
           # [ 20.31861917],
           # [ 21.52279115],
           # [ 20.32460192],
           # [ 20.24498546],
           # [ 21.08652083],
           # [ 19.21790448],
           # [ 21.75559386],
           # [ 18.92530512],
           # [ 22.54211288],
           # [ 22.05391648],
           # [ 22.26511428],
           # [ 20.45635333],
           # [ 19.41049656],
           # [ 20.07532757],
           # [ 22.78946173],
           # [ 21.73182341],
           # [ 20.79402206],
           # [ 20.75021213],
           # [ 21.29892497],
           # [ 20.04914191],
           # [ 21.78987973],
           # [ 19.96707446],
           # [ 22.25615446],
           # [ 20.58022055],
           # [ 21.70264345],
           # [ 20.59756956],
           # [ 20.30384453],
           # [ 21.13783074]])]
           #  C_init = [np.array([[ 0.09270081],
           # [-0.57230272],
           # [ 0.69954749],
           # [ 1.03676213],
           # [-0.70148025],
           # [-0.51728374],
           # [-0.69153631],
           # [-0.63144431],
           # [ 0.00640706],
           # [ 0.28154011],
           # [ 0.11844466],
           # [-0.7530985 ],
           # [-0.36602609],
           # [ 0.37667838],
           # [-0.62512039],
           # [-0.14137656],
           # [-0.49759322],
           # [-0.59445912],
           # [ 0.25593333],
           # [-0.72151868],
           # [-0.60979958],
           # [ 0.04011939],
           # [-0.69491812],
           # [ 0.23399983],
           # [ 0.36587043],
           # [-0.87838342],
           # [-0.42185747],
           # [-0.33768377],
           # [ 0.31935719],
           # [-0.04375182],
           # [-0.51586552],
           # [-0.24270635],
           # [-0.22580923],
           # [ 0.10645628],
           # [ 0.85472349],
           # [-0.36494302],
           # [-0.1640804 ],
           # [ 0.1626044 ],
           # [-0.4451563 ],
           # [ 0.30277015],
           # [ 0.13693654],
           # [ 0.22055848],
           # [-0.8217789 ],
           # [-0.19109809],
           # [ 0.23071537],
           # [-0.25448743],
           # [-0.80615021],
           # [-0.31796655],
           # [-0.08583837],
           # [-0.23297213],
           # [ 0.14444403],
           # [ 0.23520624],
           # [-0.23655085],
           # [-0.22686031],
           # [-0.02991218],
           # [ 0.43962796],
           # [-0.14975701],
           # [-0.3078071 ],
           # [-0.18949137],
           # [ 0.19802665],
           # [ 0.19195155],
           # [-0.53766287],
           # [-0.06449952],
           # [ 0.20938725]])]


            evaluator = GPyOpt.core.evaluators.Sequential(acquisition)
            bo = BO(model_f, model_c, space, f, c, acquisition, evaluator, initial_design,
                    ls_evaluator=None,#last_step_evaluator,
                    ls_acquisition=None,#Last_Step_acq,
                    deterministic=False)

            stop_date = datetime(2022, 5, 9, 7)  # year month day hour
            max_iter  = 100
            # print("Finished Initialization")
            subfolder = "mistery_cKG_n_obj_" + str(noise_objective) + "_n_c_" + str(noise_constraints)
            folder = "RESULTS"
            cwd = os.getcwd()
            path =cwd + "/" + folder + "/" + subfolder + '/it_' + str(rep) + '.csv'
            X, Y, C, recommended_val, optimum, Opportunity_cost = bo.run_optimization(max_iter = max_iter,
                                                                                      verbosity=False,
                                                                                      path=path,
                                                                                      stop_date=stop_date,
                                                                                      compute_OC=True,
                                                                                      evaluations_file=subfolder,
                                                                                      KG_dynamic_optimisation=True)

            print("Code Ended")
            print("X",X,"Y",Y, "C", C)
# function_caller_mistery_v2(rep=4)


