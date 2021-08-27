# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .optimizer import OptLbfgs, OptSgd, OptDirect, OptCma, apply_optimizer, choose_optimizer
from .anchor_points_generator import ObjectiveAnchorPointsGenerator, ThompsonSamplingAnchorPointsGenerator
from ..core.task.space import Design_space
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
from scipy.stats import norm
from ..experiment_design import initial_design

max_objective_anchor_points_logic = "max_objective"
thompson_sampling_anchor_points_logic = "thompsom_sampling"
sobol_design_type = "sobol"
random_design_type = "random"


class AcquisitionOptimizer(object):
    """
    General class for acquisition optimizers defined in domains with mix of discrete, continuous, bandit variables

    :param space: design space class from GPyOpt.
    :param optimizer: optimizer to use. Can be selected among:
        - 'lbfgs': L-BFGS.
        - 'DIRECT': Dividing Rectangles.
        - 'CMA': covariance matrix adaptation.
    """

    def __init__(self, space, optimizer='lbfgs', inner_optimizer='lbfgs', **kwargs):

        self.space              = space
        self.optimizer_name     = optimizer
        self.inner_optimizer_name     = inner_optimizer
        self.kwargs             = kwargs
        self.inner_anchor_points = None
        self.outer_anchor_points = None
        ### -- save extra options than can be passed to the optimizer
        if 'model' in self.kwargs:
            self.model = self.kwargs['model']

        if 'model_c' in self.kwargs:

            self.model_c = self.kwargs['model_c']

        # print("self.kwargs:", self.kwargs)
        if 'anchor_points_logic' in self.kwargs:
            self.type_anchor_points_logic = self.kwargs['anchor_points_logic']
        else:
            self.type_anchor_points_logic = max_objective_anchor_points_logic

        self.counter = 0
        ## -- Context handler: takes
        self.context_manager = ContextManager(space)


    def optimize(self, f=None, df=None, f_df=None, duplicate_manager=None, re_use=False ,num_samples=5000,optimizer_type=None, **kwargs):
        """
        Optimizes the input function.

        :param f: function to optimize.
        :param df: gradient of the function to optimize.
        :param f_df: returns both the function to optimize and its gradient.

        """

        self.kwargs = kwargs
        if 'dynamic_parameter_function' in self.kwargs:
            self.dynamic_parameter_function = self.kwargs['dynamic_parameter_function']

        self.f = f
        self.df = df
        self.f_df = f_df

        print("getting anchor points")
        ## --- Update the optimizer, in case context has beee passed.

        if 'dynamic_parameter_function' in self.kwargs:
            print("Optimising Hybrid KG")
            self.optimizer = choose_optimizer("Nelder_Mead", self.context_manager.noncontext_bounds)
        else:
            self.optimizer = choose_optimizer(self.optimizer_name, self.context_manager.noncontext_bounds)

        # if 'dynamic_parameter_function' in self.kwargs:
        #     print("setting fix discretisation for anchor points")
        #     discretisation = initial_design("latin",self.space, 1000)#self.generate_points_pf(N=1000) #
        #     self.dynamic_parameter_function(optimize_discretization=False, optimize_random_Z=True,
        #                                     fixed_discretisation=discretisation)


        ## --- Selecting the anchor points and removing duplicates
        if self.type_anchor_points_logic == max_objective_anchor_points_logic:
            print("max objectives. Num of samples:", num_samples)
            anchor_points_generator = ObjectiveAnchorPointsGenerator(self.space, random_design_type, f, num_samples=num_samples)
        elif self.type_anchor_points_logic == thompson_sampling_anchor_points_logic:
            print("thompson sampling")
            anchor_points_generator = ThompsonSamplingAnchorPointsGenerator(self.space, random_design_type, self.model, self.model_c)

        ## -- Select the anchor points (with context)
        if re_use == True:
            anchor_points = self.old_anchor_points
        else:



            anchor_points = anchor_points_generator.get(num_anchor=3, X_sampled_values=self.model.get_X_values(),
                                                        duplicate_manager=duplicate_manager,
                                                        context_manager=self.context_manager)
            anchor_points_ls = self.optimize_final_evaluation()
            anchor_points = np.concatenate((anchor_points, anchor_points_ls))

            sampled_X_vals = self.model.get_X_values()
            sampled_Y_vals_pred = self.model.predict(sampled_X_vals)[0]
            pf = self.probability_feasibility_multi_gp(sampled_X_vals, self.model_c).reshape(-1)
            best_sampled_X_idx = np.argmax(sampled_Y_vals_pred*pf)
            best_sampled_X = sampled_X_vals[best_sampled_X_idx]


            # print("best_sampled_X",best_sampled_X, "max", np.max(sampled_Y_vals_pred*pf))
            anchor_points = np.concatenate((np.atleast_2d(best_sampled_X) , anchor_points))

            best_sampled_X_idx_unconstrained = np.argmax(sampled_Y_vals_pred)
            best_sampled_X_unconstrained = sampled_X_vals[best_sampled_X_idx_unconstrained]
            anchor_points = np.concatenate((np.atleast_2d(best_sampled_X_unconstrained), anchor_points))
            print("anchor_points ",anchor_points )

                # anchor_points_vals = f(anchor_points)
            # print("anchor_points",anchor_points, "anchor_points_vals",anchor_points_vals)
            if False: #True: #p.sum(anchor_points_vals)==0:
                print("feasible points failed, changed to best posterior mean")
                optimized_points = []
                anchor_points_ls = self.optimize_final_evaluation()
                for a in anchor_points_ls:
                    if 'dynamic_parameter_function' in self.kwargs:

                        self.dynamic_parameter_function(optimize_discretization=True, optimize_random_Z=False,
                                                        fixed_discretisation=None)  # discretisation)


                    optimised_anchor_point = apply_optimizer(self.optimizer, a.flatten(), f=f, df=None,
                                                             f_df=f_df,
                                                             duplicate_manager=duplicate_manager,
                                                             context_manager=self.context_manager,
                                                             space=self.space)
                    optimized_points.append(optimised_anchor_point)
                x_min, fx_min = min(optimized_points, key=lambda t: t[1])
                print(" posterior best sample x_min, fx_min", x_min, fx_min)

                # return x_min, fx_min

        ## --- Applying local optimizers at the anchor points and update bounds of the optimizer (according to the context)

        # if self.outer_anchor_points is not None:
        #     # print("self.inner_anchor_points, anchor_points", self.inner_anchor_points, anchor_points)
        #     anchor_points = np.concatenate((self.outer_anchor_points, anchor_points))

        print("optimising anchor points....")

        if 'dynamic_parameter_function' in self.kwargs:
            optimized_points = []
            for a in anchor_points:
                optimised_anchor_point_x = a*1

                if 'dynamic_parameter_function' in self.kwargs:

                    self.dynamic_parameter_function(optimize_discretization=False, optimize_random_Z=False,
                                                    fixed_discretisation= discretisation)
                # print("optimiser type", self.optimizer_name)

                optimised_anchor_point = apply_optimizer(self.optimizer, optimised_anchor_point_x.flatten(), f=f, df=None, f_df=f_df,
                                                        duplicate_manager=duplicate_manager, context_manager=self.context_manager,
                                                     space = self.space)

                optimized_points.append(optimised_anchor_point)
            print("anchor_points",anchor_points)
            print("optimised_anchor_point",optimized_points)
        else:
            if 'additional_anchor_points' in self.kwargs:
                anchor_points = np.concatenate((anchor_points, kwargs["additional_anchor_points"]))
            optimized_points = []
            for a in anchor_points:
                optimised_anchor_point = apply_optimizer(self.optimizer, a.flatten(), f=f, df=None,
                                                         f_df=f_df,
                                                         duplicate_manager=duplicate_manager,
                                                         context_manager=self.context_manager,
                                                         space=self.space)
                optimized_points.append(optimised_anchor_point)
        stop = time.time()

        print("optimised points", optimized_points)
        x_min, fx_min = min(optimized_points, key=lambda t: t[1])
        self.outer_anchor_points = x_min
        print("acq val: ", x_min, fx_min)
        if np.sum(fx_min) == 0:
            if 'dynamic_parameter_function' in self.kwargs:
                self.dynamic_parameter_function(optimize_discretization=True, optimize_random_Z=False,
                                                fixed_discretisation=None)  # discretisation)
                optimized_points = []
                anchor_points_ls = self.optimize_final_evaluation()
                for a in anchor_points_ls:
                    optimised_anchor_point = apply_optimizer(self.optimizer, a.flatten(), f=f, df=None,
                                                             f_df=f_df,
                                                             duplicate_manager=duplicate_manager,
                                                             context_manager=self.context_manager,
                                                             space=self.space)
                    optimized_points.append(optimised_anchor_point)
                x_min, fx_min = min(optimized_points, key=lambda t: t[1])
                print(" x_min, fx_min", x_min, fx_min)
            return x_min, fx_min


        return x_min, fx_min

    def optimize_inner_func(self, f=None, df=None, f_df=None, duplicate_manager=None, extra_point =None ,reuse=False, num_samples=7000):
        """
        Optimizes the input function.

        :param f: function to optimize.
        :param df: gradient of the function to optimize.
        :param f_df: returns both the function to optimize and its gradient.

        """
        self.f = f
        self.df = df
        self.f_df = f_df

        ## --- Update the optimizer, in case context has beee passed.

        # print("self.inner_optimizer_name", self.inner_optimizer_name)
        self.inner_optimizer = choose_optimizer(self.inner_optimizer_name, self.context_manager.noncontext_bounds)

        ## --- Selecting the anchor points and removing duplicates


        # if self.type_anchor_points_logic == max_objective_anchor_points_logic:
        anchor_points_generator = ObjectiveAnchorPointsGenerator(self.space, random_design_type, f, num_samples= num_samples)
        # elif self.type_anchor_points_logic == thompson_sampling_anchor_points_logic:
        #     anchor_points_generator = ThompsonSamplingAnchorPointsGenerator(self.space, sobol_design_type, self.model)

        ## -- Select the anchor points (with context)

        anchor_points = anchor_points_generator.get(num_anchor=1,duplicate_manager=duplicate_manager, context_manager=self.context_manager)
        if extra_point is not None:
            anchor_points = np.concatenate((anchor_points, extra_point))

        # if self.inner_anchor_points is not None:
        #     # print("self.inner_anchor_points, anchor_points",self.inner_anchor_points, anchor_points)
        #     anchor_points = np.concatenate((self.inner_anchor_points, anchor_points))

        ## --- Applying local optimizers at the anchor points and update bounds of the optimizer (according to the context)

        optimized_points = [apply_optimizer(self.inner_optimizer, a.flatten(), f=f, df=None, f_df=f_df, duplicate_manager=duplicate_manager, context_manager=self.context_manager, space = self.space) for a in anchor_points]
        x_min, fx_min = min(optimized_points, key=lambda t:t[1])
        self.inner_anchor_points = x_min
        # print("optimized_points",optimized_points)
        # import matplotlib
        # discretisation = initial_design("latin",self.space, 1000)
        # fvals = f(discretisation)
        # plt.scatter(discretisation[:,0], discretisation[:,1], c=np.array(fvals).reshape(-1))
        # plt.scatter(x_min[:, 0], x_min[:, 1], color="red")
        # plt.title("OPTIMISED POINTS")
        # plt.show()

        return x_min, fx_min

    def optimize_final_evaluation(self):

        out = self.optimize_inner_func(f=self.expected_improvement, duplicate_manager=None, num_samples=1000)
        EI_suggested_sample =  self.space.zip_inputs(out[0])

        return EI_suggested_sample

    def expected_improvement(self, X, offset=1e-4):
        '''
        Computes the EI at points X based on existing samples X_sample
        and Y_sample using a Gaussian process surrogate model.

        Args:
            X: Points at which EI shall be computed (m x d).
            X_sample: Sample locations (n x d).
            Y_sample: Sample values (n x 1).
            gpr: A GaussianProcessRegressor fitted to samples.
            xi: Exploitation-exploration trade-off parameter.

        Returns:
            Expected improvements at points X.
        '''


        mu = self.model.posterior_mean(X)
        mu = mu.reshape(-1)
        pf = self.probability_feasibility_multi_gp(X,self.model_c).reshape(-1)

        return -(mu *pf).reshape(-1)

    def probability_feasibility_multi_gp(self, x, model, mean=None, cov=None,  l=0):
        # print("model",model.output)
        x = np.atleast_2d(x)
        Fz = []
        for m in range(model.output_dim):
            Fz.append(self.probability_feasibility( x, model.output[m], l))
        Fz = np.product(Fz,axis=0)
        return Fz


    def generate_points_pf(self, N):

        proposed_points = []
        n = 0
        while n<N:
            X = initial_design("latin",self.space, 100)
            pf = self.probability_feasibility_multi_gp(X, self.model_c).reshape(-1)

            n += len(X[pf.reshape(-1) > 0.1])
            proposed_points.append(X[pf.reshape(-1) > 0.1])

        return np.vstack(proposed_points)[:N, :]



    def probability_feasibility(self, x, model, mean=None, cov=None, grad=False, l=0):

        model = model.model
        # kern = model.kern
        # X = model.X

        mean = model.posterior_mean(x)
        var = model.posterior_variance(x, noise=False)

        std = np.sqrt(var).reshape(-1, 1)

        aux_var = np.reciprocal(var)
        mean = mean.reshape(-1, 1)

        norm_dist = norm(mean, std)
        fz = norm_dist.pdf(l)
        Fz = norm_dist.cdf(l)

        if grad == True:
            grad_mean, grad_var = model.predictive_gradients(x)
            grad_std = (1 / 2.0) * grad_var

            # cov = kern.K(X, X) + np.eye(X.shape[0]) * 1e-3
            # L = scipy.linalg.cholesky(cov, lower=True)
            # u = scipy.linalg.solve(L, np.eye(X.shape[0]))
            # Ainv = scipy.linalg.solve(L.T, u)

            dims = range(x.shape[1])
            grad_Fz = []

            for d in dims:
                grd_mean_d = grad_mean[:, d].reshape(-1, 1)
                grd_std_d = grad_std[:, d].reshape(-1, 1)
                grad_Fz.append(fz * aux_var * (mean * grd_std_d - grd_mean_d * std))
            grad_Fz = np.stack(grad_Fz, axis=1)
            return Fz.reshape(-1, 1), grad_Fz[:, :, 0]
        else:
            return Fz.reshape(-1, 1)


class ContextManager(object):
    """
    class to handle the context variable in the optimizer
    :param space: design space class from GPyOpt.
    :param context: dictionary of variables and their contex values
    """

    def __init__ (self, space, context = None):
        self.space              = space
        self.all_index          = list(range(space.model_dimensionality))
        self.all_index_obj      = list(range(len(self.space.config_space_expanded)))
        self.context_index      = []
        self.context_value      = []
        self.context_index_obj  = []
        self.nocontext_index_obj= self.all_index_obj
        self.noncontext_bounds  = self.space.get_bounds()[:]
        self.noncontext_index   = self.all_index[:]

        if context is not None:
            #print('context')

            ## -- Update new context
            for context_variable in context.keys():
                variable = self.space.find_variable(context_variable)
                self.context_index += variable.index_in_model
                self.context_index_obj += variable.index_in_objective
                self.context_value += variable.objective_to_model(context[context_variable])

            ## --- Get bounds and index for non context
            self.noncontext_index = [idx for idx in self.all_index if idx not in self.context_index]
            self.noncontext_bounds = [self.noncontext_bounds[idx] for idx in  self.noncontext_index]

            ## update non context index in objective
            self.nocontext_index_obj = [idx for idx in self.all_index_obj if idx not in self.context_index_obj]



    def _expand_vector(self,x):
        '''
        Takes a value x in the subspace of not fixed dimensions and expands it with the values of the fixed ones.
        :param x: input vector to be expanded by adding the context values
        '''
        x = np.atleast_2d(x)
        x_expanded = np.zeros((x.shape[0],self.space.model_dimensionality))
        x_expanded[:,np.array(self.noncontext_index).astype(int)]  = x
        x_expanded[:,np.array(self.context_index).astype(int)]  = self.context_value
        return x_expanded
