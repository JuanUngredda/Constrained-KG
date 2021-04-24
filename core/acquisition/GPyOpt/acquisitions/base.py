# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from ..core.task.cost import constant_cost_withGradients
import matplotlib.pyplot as plt
import numpy as np

class AcquisitionBase(object):
    """
    Base class for acquisition functions in Bayesian Optimization

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer

    """



    def __init__(self, model, space, optimizer,model_c=None, cost_withGradients=None):

        self.analytical_gradient_prediction = False
        self.model = model
        self.space = space
        self.optimizer = optimizer
        self.analytical_gradient_acq = self.analytical_gradient_prediction and self.model.analytical_gradient_prediction # flag from the model to test if gradients are available

        if model_c != None:
            self.model_c = model_c

        if cost_withGradients is  None:
            self.cost_withGradients = constant_cost_withGradients
        else:
            self.cost_withGradients = cost_withGradients

    @staticmethod
    def fromDict(model, space, optimizer, cost_withGradients, config):
        raise NotImplementedError()


    def acquisition_function(self,x):

        """
        Takes an acquisition and weights it so the domain and cost are taken into account.
        """
        f_acqu = self._compute_acq(x)
        cost_x, _ = self.cost_withGradients(x)

        return -(f_acqu*self.space.indicator_constraints(x)).reshape(-1)#/cost_x f_acqu*self.space.indicator_constraints(x) #

    def generate_random_vectors(self, optimize_discretization=True, optimize_random_Z=False):
        raise NotImplementedError()

    def _plots(self, test_samples):
        raise NotImplementedError()

    def acquisition_function_withGradients(self, x):

        """
        Takes an acquisition and it gradient and weights it so the domain and cost are taken into account.
        """

        f_acqu, df_acq_cost = self._compute_acq_withGradients(x)

        return -f_acqu*self.space.indicator_constraints(x), -df_acq_cost*self.space.indicator_constraints(x)

    def optimize(self, duplicate_manager=None, re_use=False, dynamic_optimisation=False):
        """
        Optimizes the acquisition function (uses a flag from the model to use gradients or not).
        """

        if not self.analytical_gradient_acq:
            if dynamic_optimisation:
                out = self.optimizer.optimize(f=self.acquisition_function, duplicate_manager=duplicate_manager, re_use=re_use,
                                              dynamic_parameter_function=self.generate_random_vectors)
            else:
                out = self.optimizer.optimize(f=self.acquisition_function, duplicate_manager=duplicate_manager, re_use=re_use)
        else:
            if dynamic_optimisation:
                out = self.optimizer.optimize(f=self.acquisition_function, f_df=self.acquisition_function_withGradients,
                                              duplicate_manager=duplicate_manager, re_use=re_use,
                                              dynamic_parameter_function=self.generate_random_vectors)
            else:
                out = self.optimizer.optimize(f=self.acquisition_function, f_df=self.acquisition_function_withGradients,
                                              duplicate_manager=duplicate_manager, re_use=re_use)

        print("out", out)
        return out

    def current_compute_acq(self):

        raise NotImplementedError('')

    def _compute_acq(self,x):

        raise NotImplementedError('')

    def _compute_acq_withGradients(self, x):

        raise NotImplementedError('')
