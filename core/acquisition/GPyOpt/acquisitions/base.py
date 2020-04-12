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

        self.analytical_gradient_prediction = True
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

    def gradient_sanity_check_1D(self, f, grad_f, delta=1e-4):
        X = self.test_samples
        numerical_grad = []
        analytical_grad = []
        func_val = []
        for x in X:
            x = x.reshape(1, -1)
            f_delta = np.array(f(x + delta)).reshape(-1)
            f_val = np.array(f(x)).reshape(-1)
            func_val.append(f_val)
            numerical_grad.append((f_delta - f_val) / delta)
            analytical_grad.append(grad_f(x))

        func_val = np.array(func_val).reshape(-1)
        numerical_grad = np.array(numerical_grad).reshape(-1)
        analytical_grad = np.array(analytical_grad).reshape(-1)

        dif = np.abs(numerical_grad - analytical_grad)
        print("dif mean", np.mean(dif), "dif min", np.min(dif), "dif max", np.max(dif))

        # PLOTS
        fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6)

        ax3.scatter(X, func_val, label="actual function")
        ax3.legend()
        ax4.scatter(X, numerical_grad, label="numerical")
        ax4.legend()
        ax5.scatter(X, analytical_grad, label="analytical")
        ax5.legend()

        ax6.scatter(X, dif.reshape(-1), label="errors")
        ax6.legend()
        plt.title(name)
        plt.show()


    def grad_sanity_check_2D(self,  f, grad_f, delta= 1e-4):
        print("inside gradient_sanity_check_2D")
        initial_design = np.random.random((100,2))*100

        fixed_dim =1
        variable_dim = 0
        v1 = np.repeat(np.array(initial_design[0, fixed_dim]), len(initial_design[:, 1])).reshape(-1, 1)
        v2 = initial_design[:, variable_dim].reshape(-1, 1)
        X = np.concatenate((v1, v2), axis=1)

        numerical_grad = []
        analytical_grad = []
        func_val = []
        dim = X.shape[1]
        delta_matrix = np.identity(dim)

        for x in X:

            x = x.reshape(1,-1)
            f_val = np.array(f(x)).reshape(-1)
            f_delta = []
            for i in range(dim):

                one_side = np.array(f(x + delta_matrix[i]*delta)).reshape(-1)
                two_side = np.array(f(x - delta_matrix[i]*delta)).reshape(-1)
                f_delta.append(one_side - two_side)

            func_val.append(f_val)
            f_delta = np.array(f_delta).reshape(-1)
            numerical_grad.append(np.array(f_delta/(2*delta)).reshape(-1))
            analytical_grad.append(grad_f(x).reshape(-1))
            print("analytical", grad_f(x).reshape(-1), "numerical", np.array(f_delta/(2*delta)).reshape(-1))

        func_val = np.array(func_val)
        numerical_grad = np.array(numerical_grad)
        analytical_grad = np.array(analytical_grad)

        dif = np.abs(numerical_grad - analytical_grad)
        print("dif mean", np.mean(dif,axis=0), "dif min", np.min(dif,axis=0), "dif max", np.max(dif,axis=0))

        #PLOTS
        fig, (ax1, ax2, ax3,ax4, ax5, ax6) = plt.subplots(6)


        # ax1.scatter(v2.reshape(-1), np.array(self.mean_plot).reshape(-1),label="mean")
        # ax1.legend()
        # ax2.scatter(v2.reshape(-1), np.array(self.var_plot).reshape(-1), label="var")
        # ax2.legend()
        ax3.scatter(v2.reshape(-1), np.array(func_val).reshape(-1), label="actual function")
        ax3.legend()
        ax4.scatter(v2.reshape(-1),np.array(numerical_grad[:,variable_dim]).reshape(-1), label="numerical")
        ax4.legend()
        ax5.scatter(v2.reshape(-1),np.array(analytical_grad[:,variable_dim]).reshape(-1), label="analytical")
        ax5.legend()

        ax6.scatter(v2.reshape(-1), dif[:,variable_dim].reshape(-1), label="errors")
        ax6.legend()
        plt.show()

    def acquisition_function(self,x):

        """
        Takes an acquisition and weights it so the domain and cost are taken into account.
        """
        f_acqu = self._compute_acq(x)
        cost_x, _ = self.cost_withGradients(x)

        # print("f_acqu ",f_acqu )
        # print("cost_x",cost_x)
        # print("-(f_acqu*self.space.indicator_constraints(x))/cost_x",-(f_acqu*self.space.indicator_constraints(x))/cost_x)
        print("-(f_acqu*self.space.indicator_constraints(x))#/cost_x",-(f_acqu*self.space.indicator_constraints(x)))#/cost_x)
        return f_acqu*self.space.indicator_constraints(x) #-(f_acqu*self.space.indicator_constraints(x))#/cost_x

    def current_acquisition_function(self):

        """
        Takes an acquisition and weights it so the domain and cost are taken into account.
        """
        f_acqu = self.current_compute_acq()
        cost_x, _ = self.cost_withGradients(x)

        return -(f_acqu*self.space.indicator_constraints(x))#/cost_x

    def acquisition_function_withGradients(self, x):

        """
        Takes an acquisition and it gradient and weights it so the domain and cost are taken into account.
        """

        f_acqu, df_acq_cost = self._compute_acq_withGradients(x)
        # print("acquisition_function_withGradients",f_acqu, df_acqu)
        cost_x, cost_grad_x = self.cost_withGradients(x)
        # print("cost_x, cost_grad_x",cost_x, cost_grad_x)
        f_acq_cost = f_acqu/cost_x
        #df_acq_cost = (df_acqu*cost_x - f_acqu*cost_grad_x)/(cost_x**2)
        # print("-f_acq_cost*self.space.indicator_constraints(x), -df_acq_cost*self.space.indicator_constraints(x)",-f_acq_cost*self.space.indicator_constraints(x), -df_acq_cost*self.space.indicator_constraints(x))

        print("-f_acq_cost*self.space.indicator_constraints(x), -df_acq_cost*self.space.indicator_constraints(x)",f_acq_cost*self.space.indicator_constraints(x), df_acq_cost*self.space.indicator_constraints(x))
        return  df_acq_cost*self.space.indicator_constraints(x) # -f_acq_cost*self.space.indicator_constraints(x), -df_acq_cost*self.space.indicator_constraints(x)

    def optimize(self, duplicate_manager=None, re_use=False):
        """
        Optimizes the acquisition function (uses a flag from the model to use gradients or not).
        """

        if not self.analytical_gradient_acq:

            out = self.optimizer.optimize(f=self.acquisition_function, duplicate_manager=duplicate_manager, re_use=re_use)

        else:
            print("------------------------------INSIDE TEST---------------------------------------------------")
            self.gradient_sanity_check_1D(f=self.acquisition_function, grad_f=self.acquisition_function_withGradients)
            print("------------------------------OUTSIDE TEST---------------------------------------------------")
            out = self.optimizer.optimize(f=self.acquisition_function, f_df=self.acquisition_function_withGradients, duplicate_manager=duplicate_manager, re_use=re_use)


        return out

    def current_compute_acq(self):

        raise NotImplementedError('')

    def _compute_acq(self,x):

        raise NotImplementedError('')

    def _compute_acq_withGradients(self, x):

        raise NotImplementedError('')
