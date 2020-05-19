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

    def __init__(self, space, optimizer='sgd', inner_optimizer='lbfgs', **kwargs):

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

        if 'anchor_points_logic' in self.kwargs:
            self.type_anchor_points_logic = self.kwargs['type_anchor_points_logic']
        else:
            self.type_anchor_points_logic = max_objective_anchor_points_logic

        ## -- Context handler: takes
        self.context_manager = ContextManager(space)


    def optimize(self, f=None, df=None, f_df=None, duplicate_manager=None, re_use=False ,sweet_spot=True, num_samples=50, verbose=True):
        """
        Optimizes the input function.

        :param f: function to optimize.
        :param df: gradient of the function to optimize.
        :param f_df: returns both the function to optimize and its gradient.

        """


        self.f = f
        self.df = df
        self.f_df = f_df

        print("getting anchor points")
        ## --- Update the optimizer, in case context has beee passed.
        self.optimizer = choose_optimizer(self.optimizer_name, self.context_manager.noncontext_bounds)

        ## --- Selecting the anchor points and removing duplicates
        if self.type_anchor_points_logic == max_objective_anchor_points_logic:
            # print("max objectives")
            anchor_points_generator = ObjectiveAnchorPointsGenerator(self.space, random_design_type, f, num_samples=num_samples)
        elif self.type_anchor_points_logic == thompson_sampling_anchor_points_logic:
            # print("thompson sampling")
            anchor_points_generator = ThompsonSamplingAnchorPointsGenerator(self.space, sobol_design_type, self.model)
           
        ## -- Select the anchor points (with context)
        if re_use == True:
            anchor_points = self.old_anchor_points
        else:
            anchor_points = anchor_points_generator.get(num_anchor=1,X_sampled_values=self.model.get_X_values() ,duplicate_manager=duplicate_manager, context_manager=self.context_manager)
            self.old_anchor_points = anchor_points
        print("getting that sweet spot that you like")
        if sweet_spot:
            EI_suggested_sample = self.optimize_final_evaluation()
            EI_suggested_sample = EI_suggested_sample.reshape(-1)
            EI_suggested_sample = EI_suggested_sample.reshape(1,-1)
            print("EI_suggested_samples", EI_suggested_sample, "anchor_points", anchor_points)
            anchor_points = np.concatenate((EI_suggested_sample, anchor_points))

        ## --- Applying local optimizers at the anchor points and update bounds of the optimizer (according to the context)

        # if self.outer_anchor_points is not None:
        #     # print("self.inner_anchor_points, anchor_points", self.inner_anchor_points, anchor_points)
        #     anchor_points = np.concatenate((self.outer_anchor_points, anchor_points))

        print("optimising anchor points....")
        optimized_points = [apply_optimizer(self.optimizer, a.flatten(), f=f, df=None, f_df=f_df, duplicate_manager=duplicate_manager, context_manager=self.context_manager, space = self.space) for a in anchor_points]

        x_min, fx_min = min(optimized_points, key=lambda t: t[1])
        # self.outer_anchor_points = x_min

        print("anchor_points", anchor_points)
        print("optimized_points", optimized_points)
        if False:
            opt_x = np.array([np.array(i[0]).reshape(-1) for i in optimized_points])

            bounds =self.space.get_bounds()
            x_plot = np.random.random((1000,2))*(np.array([bounds[0][1], bounds[1][1]]) - np.array([bounds[0][0], bounds[1][0]])) +  np.array([bounds[0][0], bounds[1][0]])
            f_vals = np.array([f(i) for i in x_plot]).reshape(-1)
            plt.scatter(np.array(x_plot[:,0]).reshape(-1), (x_plot[:,1]).reshape(-1), c=np.array(f_vals).reshape(-1))
            plt.scatter(anchor_points[:,0], anchor_points[:,1], color="magenta")
            plt.scatter(opt_x[:, 0], opt_x[:, 1], color="magenta", marker="x")
            plt.scatter(x_min[:, 0], x_min[:, 1], color="red")
            plt.title("OPTIMISED POINTS")
            plt.show()

        # print("self.inner_anchor_points",self.inner_anchor_points)
        # print("x_m
        # print("x_min, fx_min",x_min, fx_min)
        # print("x_min",x_min,"fx_min",fx_min)
        #x_min, fx_min = min([apply_optimizer(self.optimizer, a, f=f, df=None, f_df=f_df, duplicate_manager=duplicate_manager, context_manager=self.context_manager, space = self.space) for a in anchor_points], key=lambda t:t[1])                   
        return x_min, fx_min
    
    
    def optimize_inner_func(self, f=None, df=None, f_df=None, duplicate_manager=None, num_samples=400):
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
        self.inner_optimizer = choose_optimizer(self.inner_optimizer_name, self.context_manager.noncontext_bounds)

        ## --- Selecting the anchor points and removing duplicates

        if self.type_anchor_points_logic == max_objective_anchor_points_logic:
            anchor_points_generator = ObjectiveAnchorPointsGenerator(self.space, random_design_type, f, num_samples= num_samples)
        elif self.type_anchor_points_logic == thompson_sampling_anchor_points_logic:
            anchor_points_generator = ThompsonSamplingAnchorPointsGenerator(self.space, sobol_design_type, self.model)
           
        ## -- Select the anchor points (with context)

        anchor_points = anchor_points_generator.get(num_anchor=1,duplicate_manager=duplicate_manager, context_manager=self.context_manager)


        if self.inner_anchor_points is not None:

            anchor_points = np.concatenate((self.inner_anchor_points, anchor_points))

        ## --- Applying local optimizers at the anchor points and update bounds of the optimizer (according to the context)
        # print("anchor_points",anchor_points, "shape", anchor_points.shape)

        # print("anchor_points inner func",anchor_points)
        # print("value_anchor points inner func", f(anchor_points))

        # print("anchor_points",anchor_points)


        optimized_points = [apply_optimizer(self.inner_optimizer, a.flatten(), f=f, df=None, f_df=f_df, duplicate_manager=duplicate_manager, context_manager=self.context_manager, space = self.space) for a in anchor_points]
        x_min, fx_min = min(optimized_points, key=lambda t:t[1])
        self.inner_anchor_points = x_min

        # opt_x = np.array([np.array(i[0]).reshape(-1) for i in optimized_points])
        # print("optimized_points", optimized_points)
        #
        # bounds =self.space.get_bounds()
        # x_plot = np.random.random((1000,2))*(np.array([bounds[0][1], bounds[1][1]]) - np.array([bounds[0][0], bounds[1][0]])) +  np.array([bounds[0][0], bounds[1][0]])
        #
        # f_vals = np.array([f(i) for i in x_plot]).reshape(-1)
        # # print("min max", np.min(f_vals ), np.max(f_vals ))
        # plt.scatter(np.array(x_plot[:,0]).reshape(-1), (x_plot[:,1]).reshape(-1), c=np.array(f_vals).reshape(-1))
        # plt.scatter(anchor_points[:,0], anchor_points[:,1], color="magenta")
        # plt.scatter(opt_x[:,0],opt_x[:,1], color="magenta", marker="x")
        # plt.scatter(x_min[:,0], x_min[:,1], color="red")
        # plt.show()
        # print("self.inner_anchor_points",self.inner_anchor_points)
        # print("x_min, fx_min",x_min, fx_min)
        return x_min, fx_min

    def optimize_final_evaluation(self):

        out = self.optimize(f=self.expected_improvement, duplicate_manager=None, re_use=False, num_samples=1000, sweet_spot=False, verbose=False)
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

        self.Y = self.model.get_Y_values()
        self.C = self.model_c.get_Y_values()
        mu = self.model.posterior_mean(X)
        sigma = self.model.posterior_variance(X, noise=False)

        sigma = np.sqrt(sigma).reshape(-1, 1)
        mu = mu.reshape(-1,1)
        # bool_C = np.product(np.concatenate(self.C, axis=1) < 0, axis=1)
        # func_val = self.Y * bool_C.reshape(-1, 1)
        # mu_sample_opt = np.max(func_val) - offset
        # #print("mu_sample_opt", mu_sample_opt)
        # with np.errstate(divide='warn'):
        #     imp = mu - mu_sample_opt
        #     Z = imp / sigma
        #     ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        #     ei[sigma == 0.0] = 0.0
        pf = self.probability_feasibility_multi_gp(X,self.model_c).reshape(-1,1)
        return -(mu *pf )

    def probability_feasibility_multi_gp(self, x, model, mean=None, cov=None,  l=0):
        # print("model",model.output)
        x = np.atleast_2d(x)
        Fz = []
        for m in range(model.output_dim):
            Fz.append(self.probability_feasibility( x, model.output[m], l))
        Fz = np.product(Fz,axis=0)
        return Fz

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
                # K1 = np.diag(np.dot(np.dot(kern.dK_dX(x, X, d), Ainv), kern.dK_dX2(X, x, d)))
                # K2 = np.diag(kern.dK2_dXdX2(x, x, d, d))
                # var_grad = K2 - K1
                # var_grad = var_grad.reshape(-1, 1)
                grd_mean_d = grad_mean[:, d].reshape(-1, 1)
                grd_std_d = grad_std[:, d].reshape(-1, 1)
                # print("grd_mean ",grd_mean_d, "var_grad ",grd_std_d )
                # print("std",std)
                # print("aux_var", aux_var)
                # print("fz",fz)
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
