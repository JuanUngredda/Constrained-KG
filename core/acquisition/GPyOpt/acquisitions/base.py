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


    def acquisition_function(self,x):

        """
        Takes an acquisition and weights it so the domain and cost are taken into account.
        """
        f_acqu = self._compute_acq(x)
        cost_x, _ = self.cost_withGradients(x)

        # print("f_acqu ",f_acqu )
        # print("cost_x",cost_x)
        # print("-(f_acqu*self.space.indicator_constraints(x))/cost_x",-(f_acqu*self.space.indicator_constraints(x))/cost_x)
        # print("-(f_acqu*self.space.indicator_constraints(x))#/cost_x",-(f_acqu*self.space.indicator_constraints(x)))#/cost_x)
        # print("f_acqu",f_acqu)
        # print("self.space.indicator_constraints(x))",self.space.indicator_constraints(x))
        # print("-(f_acqu*self.space.indicator_constraints(x))",-(f_acqu*self.space.indicator_constraints(x)))
        return -(f_acqu*self.space.indicator_constraints(x))#/cost_x f_acqu*self.space.indicator_constraints(x) #

    def compute_mu_xopt(self,x):
        f_acqu = self._compute_mu(x)
        cost_x, _ = self.cost_withGradients(x)

        # print("f_acqu ",f_acqu )
        # print("cost_x",cost_x)
        # print("-(f_acqu*self.space.indicator_constraints(x))/cost_x",-(f_acqu*self.space.indicator_constraints(x))/cost_x)
        # print("-(f_acqu*self.space.indicator_constraints(x))#/cost_x",-(f_acqu*self.space.indicator_constraints(x)))#/cost_x)
        # print("f_acqu",f_acqu)
        # print("self.space.indicator_constraints(x))",self.space.indicator_constraints(x))
        # print("-(f_acqu*self.space.indicator_constraints(x))",-(f_acqu*self.space.indicator_constraints(x)))
        return -(f_acqu*self.space.indicator_constraints(x))#/cost_x f_acqu*self.space.indicator_constraints(x) #

    def compute_mu_xopt_with_gradients(self,x):
        f_acqu = self._compute_mu_withGradients(x)
        cost_x, _ = self.cost_withGradients(x)

        # print("f_acqu ",f_acqu )
        # print("cost_x",cost_x)
        # print("-(f_acqu*self.space.indicator_constraints(x))/cost_x",-(f_acqu*self.space.indicator_constraints(x))/cost_x)
        # print("-(f_acqu*self.space.indicator_constraints(x))#/cost_x",-(f_acqu*self.space.indicator_constraints(x)))#/cost_x)
        # print("f_acqu",f_acqu)
        # print("self.space.indicator_constraints(x))",self.space.indicator_constraints(x))
        # print("-(f_acqu*self.space.indicator_constraints(x))",-(f_acqu*self.space.indicator_constraints(x)))
        return -(f_acqu*self.space.indicator_constraints(x))#/cost_x f_acqu*self.space.indicator_constraints(x) #
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

        #df_acq_cost = (df_acqu*cost_x - f_acqu*cost_grad_x)/(cost_x**2)
        # print("-f_acq_cost*self.space.indicator_constraints(x), -df_acq_cost*self.space.indicator_constraints(x)",-f_acq_cost*self.space.indicator_constraints(x), -df_acq_cost*self.space.indicator_constraints(x))

        # print("-f_acq_cost*self.space.indicator_constraints(x), -df_acq_cost*self.space.indicator_constraints(x)",f_acq_cost*self.space.indicator_constraints(x), df_acq_cost*self.space.indicator_constraints(x))
        # print("self.space.indicator_constraints(x)",self.space.indicator_constraints(x), "f_acq_cost",f_acq_cost)
        # print("-f_acq_cost*self.space.indicator_constraints(x)",-f_acq_cost*self.space.indicator_constraints(x))
        return -f_acqu*self.space.indicator_constraints(x), -df_acq_cost*self.space.indicator_constraints(x) # df_acq_cost*self.space.indicator_constraints(x) #

    def optimize(self, duplicate_manager=None, re_use=False):
        """
        Optimizes the acquisition function (uses a flag from the model to use gradients or not).
        """

        if not self.analytical_gradient_acq:
            out = self.optimizer.optimize(f=self.acquisition_function, duplicate_manager=duplicate_manager, re_use=re_use)
        else:
            # print("sanity check")
            #self.gradient_sanity_check_1D(f=self.acquisition_function, grad_f=self.acquisition_function_withGradients)
            #self._gradient_sanity_check_2D(f=self.compute_mu_xopt, grad_f=self.compute_mu_xopt_with_gradients)
            #self._gradient_sanity_check_2D(f=self.acquisition_function, grad_f=self.acquisition_function_withGradients)
            # self._gradient_sanity_check_2D_TEST2(f_df=self.acquisition_function_withGradients)
            # print("end sanity check")
            import time
            # start = time.time()
            out = self.optimizer.optimize(f=self.acquisition_function, f_df=self.acquisition_function_withGradients, duplicate_manager=duplicate_manager, re_use=re_use)
            # stop = time.time()
            # print("finished optimisation...diagnostics start", stop-start)
            #self.check_output2D(f = self.acquisition_function, out = out)
#            self.check_output(f = self.acquisition_function, out = out)

        return out

    def _gradient_sanity_check_2D_TEST2(self, f_df, delta=1e-4):
        X = np.random.random((250,2))*(np.array([[10,15]])-np.array([[-5,0]])) + np.array([[-5,0]])

        numerical_grad = []
        analytical_grad = []
        func_val = []

        for x in X:
            for i in range(10):
                x = x.reshape(1, -1)
                acq, grad_acq = f_df(x)
                func_val.append(np.array(acq).reshape(-1))
                analytical_grad.append(np.array(grad_acq).reshape(-1))

        func_val = np.array(func_val)
        analytical_grad = np.array(analytical_grad)
        analytical_grad_magnitude = np.sqrt(np.sum(analytical_grad**2,axis=1))
        # PLOTS
        fig, (ax1, ax2) = plt.subplots(2)

        ax1.scatter(X[:,0], X[:,1], c=np.array(func_val).reshape(-1), label="actual function")
        ax1.legend()
        ax2.scatter(X[:,0], X[:,1], c=np.array(analytical_grad_magnitude).reshape(-1), label="analytical magnitude")
        ax2.legend()
        plt.title("gradients test")
        plt.show()


    def _gradient_sanity_check_2D(self, f, grad_f, delta=1e-10):
        initial_design = np.random.random((80,2))*5 # self.test_samples
        fixed_dim =0
        variable_dim = 1
        v1 = np.repeat(np.array(initial_design[0, fixed_dim]), len(initial_design[:, fixed_dim])).reshape(-1, 1)
        v2 = initial_design[:, variable_dim ].reshape(-1, 1)
        X = np.concatenate((v1, v2), axis=1)

        numerical_grad = []
        analytical_grad = []
        func_val = []
        dim = X.shape[1]
        delta_matrix = np.identity(dim)
        for x in X:
            x = x.reshape(1, -1)
            f_val = np.array(f(x)).reshape(-1)
            f_delta = []
            for i in range(dim):
                one_side = np.array(f(x + delta_matrix[i] * delta)).reshape(-1)
                two_side = np.array(f(x - delta_matrix[i] * delta)).reshape(-1)
                print("one_side", one_side, "two_side", two_side)
                f_delta.append(one_side - two_side)

            func_val.append(f_val)
            f_delta = np.array(f_delta).reshape(-1)
            numerical_grad.append(np.array(f_delta / (2 * delta)).reshape(-1))

            print("FD", np.array(f_delta / (2 * delta)).reshape(-1), "analytical", grad_f(x).reshape(-1))
            analytical_grad.append(grad_f(x).reshape(-1))

        func_val = np.array(func_val)
        numerical_grad = np.array(numerical_grad)
        analytical_grad = np.array(analytical_grad)

        dif = np.abs(numerical_grad - analytical_grad)
        print("dif mean", np.mean(dif, axis=0), "dif min", np.min(dif, axis=0), "dif max", np.max(dif, axis=0))

        # PLOTS
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)

        print("v2", v2)
        print("np.array(func_val).reshape(-1)", np.array(func_val).reshape(-1))
        ax1.scatter(v2.reshape(-1), np.array(func_val).reshape(-1), label="actual function")
        ax1.legend()
        ax2.scatter(v2.reshape(-1), np.array(numerical_grad[:, variable_dim]).reshape(-1), label="numerical")
        ax2.legend()
        ax3.scatter(v2.reshape(-1), np.array(analytical_grad[:, variable_dim]).reshape(-1), label="analytical")
        ax3.legend()
        ax4.scatter(v2.reshape(-1), dif[:, variable_dim].reshape(-1), label="errors")
        ax4.legend()

        plt.show()

    def gradient_sanity_check_1D(self, f, grad_f, delta=1e-4):
        X = np.random.random((80,1))*20 # self.test_samples
        numerical_grad = []
        analytical_grad = []
        func_val = []
        acc = []
        for x in X:
            x = x.reshape(1, -1)
            f_delta = np.array(f(x + delta)).reshape(-1)
            f_val = np.array(f(x-delta)).reshape(-1)
            func_val.append(f_val)
            analytical = grad_f(x)
            # print("f_delta", f_delta, "f_val", f_val)
            numerical = (f_delta - f_val) / (2*delta)
            numerical_grad.append(numerical)
            analytical_grad.append(analytical)

            if np.array(analytical).reshape(-1) >0:
                acc.append(1)
            else:
                acc.append(-1)
            print("analytical_grad",analytical,"numerical_grad",numerical)

        acc = np.array(acc).reshape(-1)
        func_val = np.array(func_val).reshape(-1)
        numerical_grad = np.array(numerical_grad).reshape(-1)
        analytical_grad = np.array(analytical_grad).reshape(-1)

        dif = np.abs(numerical_grad - analytical_grad)
        print("dif mean", np.mean(dif), "dif min", np.min(dif), "dif max", np.max(dif))

        # PLOTS
        fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6)

        ax2.scatter(X, acc, label="accuracy")
        ax3.scatter(X, func_val, label="actual function")
        ax3.legend()
        ax4.scatter(X, numerical_grad, label="numerical")
        ax4.legend()
        ax5.scatter(X, analytical_grad, label="analytical")
        ax5.legend()
        ax6.scatter(X, dif.reshape(-1), label="errors")
        ax6.legend()

        plt.show()

    def check_output(self, f , out):
        X = np.random.random((100,1))*100 # self.test_samples

        func_val = []
        for x in X:
            print("x",x)
            x = x.reshape(1, -1)
            f_val = np.array(f(x)).reshape(-1)
            func_val.append(f_val)
        func_val = np.array(func_val).reshape(-1)

        # PLOTS
        fig, (ax1) = plt.subplots(1)
        ax1.scatter(X, func_val, label="acq func")
        ax1.scatter(np.array(out[0]).reshape(-1), np.array(out[1]).reshape(-1), color="red")
        plt.show()

    def check_output2D(self, f , out):
        X = np.linspace(0, 5, 20)  # self.test_samples
        xx, yy = np.meshgrid(X, X)

        position = np.array([[i, j] for i in X for j in X])

        func_val = np.array(f(position)).reshape(len(X),len(X))
        print("max, min", np.max(func_val), np.min(func_val))
        print("position", position[np.argmax(func_val)],position[np.argmin(func_val)] )
        fig = plt.figure(figsize=(8, 8))
        ax = fig.gca()
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 5)
        ax.contourf(xx, yy, func_val, cmap='coolwarm')
        ax.scatter(np.array(out[0]).reshape(-1)[0], np.array(out[0]).reshape(-1)[0], color="red")
        plt.show()

        # # PLOTS
        # fig, (ax1) = plt.subplots(1)
        # ax1.scatter(X[:,0].reshape(-1),X[:,1].reshape(-1) ,c=func_val, label="acq func")
        # ax1.scatter(np.array(out[0]).reshape(-1)[0], np.array(out[0]).reshape(-1)[0],  color="red")
        # plt.show()

    def current_compute_acq(self):

        raise NotImplementedError('')

    def _compute_acq(self,x):

        raise NotImplementedError('')

    def _compute_acq_withGradients(self, x):

        raise NotImplementedError('')
