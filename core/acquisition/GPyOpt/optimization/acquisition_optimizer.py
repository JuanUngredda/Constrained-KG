# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .optimizer import OptLbfgs, OptSgd, OptDirect, OptCma, apply_optimizer, choose_optimizer
from .anchor_points_generator import ObjectiveAnchorPointsGenerator, ThompsonSamplingAnchorPointsGenerator
from ..core.task.space import Design_space
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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

        ### -- save extra options than can be passed to the optimizer
        if 'model' in self.kwargs:
            self.model = self.kwargs['model']

        if 'anchor_points_logic' in self.kwargs:
            self.type_anchor_points_logic = self.kwargs['type_anchor_points_logic']
        else:
            self.type_anchor_points_logic = max_objective_anchor_points_logic

        ## -- Context handler: takes
        self.context_manager = ContextManager(space)


    def optimize(self, f=None, df=None, f_df=None, duplicate_manager=None, re_use=False, num_samples=2):
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
            anchor_points = anchor_points_generator.get(num_anchor=2,X_sampled_values=self.model.get_X_values() ,duplicate_manager=duplicate_manager, context_manager=self.context_manager)
            self.old_anchor_points = anchor_points



        ## --- Applying local optimizers at the anchor points and update bounds of the optimizer (according to the context)
        print("anchor_points",anchor_points)
        print("value_anchor points", f(anchor_points))

        # print("f",f(anchor_points[0]))
        import time
        time_start = time.time()
        optimized_points = [apply_optimizer(self.optimizer, a.flatten(), f=f, df=None, f_df=f_df, duplicate_manager=duplicate_manager, context_manager=self.context_manager, space = self.space) for a in anchor_points]
        time_stop = time.time()
        # print("time optimizer anchor points", time_stop - time_start)
        # print("anchor_points", anchor_points)
        # print("optimized_points", optimized_points)
        x_min, fx_min = min(optimized_points, key=lambda t:t[1])

        # print("x_min",x_min,"fx_min",fx_min)
        #x_min, fx_min = min([apply_optimizer(self.optimizer, a, f=f, df=None, f_df=f_df, duplicate_manager=duplicate_manager, context_manager=self.context_manager, space = self.space) for a in anchor_points], key=lambda t:t[1])                   
        return x_min, fx_min
    
    
    def optimize_inner_func(self, f=None, df=None, f_df=None, duplicate_manager=None):
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
            anchor_points_generator = ObjectiveAnchorPointsGenerator(self.space, random_design_type, f, num_samples= 20)
        elif self.type_anchor_points_logic == thompson_sampling_anchor_points_logic:
            anchor_points_generator = ThompsonSamplingAnchorPointsGenerator(self.space, sobol_design_type, self.model)
           
        ## -- Select the anchor points (with context)

        anchor_points = anchor_points_generator.get(num_anchor=6,duplicate_manager=duplicate_manager, context_manager=self.context_manager)
        
        ## --- Applying local optimizers at the anchor points and update bounds of the optimizer (according to the context)
        # print("anchor_points",anchor_points, "shape", anchor_points.shape)

        # print("anchor_points inner func",anchor_points)
        # print("value_anchor points inner func", f(anchor_points))

        # print("anchor_points",anchor_points)

        optimized_points = [apply_optimizer(self.inner_optimizer, a.flatten(), f=f, df=None, f_df=f_df, duplicate_manager=duplicate_manager, context_manager=self.context_manager, space = self.space) for a in anchor_points]

        # opt_x = np.array([np.array(i[0]).reshape(-1) for i in optimized_points])
        # print("optimized_points", optimized_points)

        #plot_optimized_points = np.array(optimized_points).reshape(-1)
        # plot_optimized_points = plot_optimized_points

        # x_plot = np.random.random((1000,2))*5
        # f_vals = np.array([f(i) for i in x_plot]).reshape(-1)
        # f_anchor = np.array([f(i) for i in anchor_points]).reshape(-1)
        # # f_opt = np.array([f(i) for i in plot_optimized_points]).reshape(-1)
        # print("min max", np.min(f_vals ), np.max(f_vals ))
        # plt.scatter(np.array(x_plot[:,0]).reshape(-1), (x_plot[:,1]).reshape(-1), c=np.array(f_vals).reshape(-1))
        # plt.scatter(anchor_points[:,0], anchor_points[:,1], color="magenta")
        # plt.scatter(opt_x[:,0],opt_x[:,1], color="magenta", marker="x")
        # # plt.scatter(plot_optimized_points.reshape(-1), f_opt, color="red")
        x_min, fx_min = min(optimized_points, key=lambda t:t[1])
        # plt.scatter(x_min[:,0], x_min[:,1], color="red")
        # plt.show()

        # print("x_min, fx_min",x_min, fx_min)
        return x_min, fx_min


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
