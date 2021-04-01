# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np


class BaseOptimizer:
    """Base (Stochastic) gradient descent optimizer
    Parameters
    ----------
    params : list, length = len(coefs_) + len(intercepts_)
        The concatenated list containing coefs_ and intercepts_ in MLP model.
        Used for initializing velocities and updating params
    learning_rate_init : float, default=0.1
        The initial learning rate used. It controls the step-size in updating
        the weights
    Attributes
    ----------
    learning_rate : float
        the current learning rate
    """

    def __init__(self, params, learning_rate_init=0.1):
        self.params = [param for param in params]
        self.learning_rate_init = learning_rate_init
        self.learning_rate = float(learning_rate_init)

    def update_params(self, grads):
        """Update parameters with given gradients
        Parameters
        ----------
        grads : list, length = len(params)
            Containing gradients with respect to coefs_ and intercepts_ in MLP
            model. So length should be aligned with params
        """
        updates = self._get_updates(grads)
        for param, update in zip(self.params, updates):
            param += update

    def iteration_ends(self, time_step):
        """Perform update to learning rate and potentially other states at the
        end of an iteration
        """
        pass

    def trigger_stopping(self, msg, verbose):
        """Decides whether it is time to stop training
        Parameters
        ----------
        msg : str
            Message passed in for verbose output
        verbose : bool
            Print message to stdin if True
        Returns
        -------
        is_stopping : bool
            True if training needs to stop
        """
        if verbose:
            print(msg + " Stopping.")
        return True


class AdamOptimizer(BaseOptimizer):
    """Stochastic gradient descent optimizer with Adam
    Note: All default values are from the original Adam paper
    Parameters
    ----------
    params : list, length = len(coefs_) + len(intercepts_)
        The concatenated list containing coefs_ and intercepts_ in MLP model.
        Used for initializing velocities and updating params
    learning_rate_init : float, default=0.001
        The initial learning rate used. It controls the step-size in updating
        the weights
    beta_1 : float, default=0.9
        Exponential decay rate for estimates of first moment vector, should be
        in [0, 1)
    beta_2 : float, default=0.999
        Exponential decay rate for estimates of second moment vector, should be
        in [0, 1)
    epsilon : float, default=1e-8
        Value for numerical stability
    Attributes
    ----------
    learning_rate : float
        The current learning rate
    t : int
        Timestep
    ms : list, length = len(params)
        First moment vectors
    vs : list, length = len(params)
        Second moment vectors
    References
    ----------
    Kingma, Diederik, and Jimmy Ba.
    "Adam: A method for stochastic optimization."
    arXiv preprint arXiv:1412.6980 (2014).
    """

    def __init__(self, params, learning_rate_init=0.001, beta_1=0.9,
                 beta_2=0.999, epsilon=1e-8):
        super().__init__(params, learning_rate_init)

        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.t = 0
        self.ms = [np.zeros_like(param) for param in params]
        self.vs = [np.zeros_like(param) for param in params]

    def _get_updates(self, grads):
        """Get the values used to update params with given gradients
        Parameters
        ----------
        grads : list, length = len(coefs_) + len(intercepts_)
            Containing gradients with respect to coefs_ and intercepts_ in MLP
            model. So length should be aligned with params
        Returns
        -------
        updates : list, length = len(grads)
            The values to add to params
        """
        self.t += 1
        self.ms = [self.beta_1 * m + (1 - self.beta_1) * grad
                   for m, grad in zip(self.ms, grads)]
        self.vs = [self.beta_2 * v + (1 - self.beta_2) * (grad ** 2)
                   for v, grad in zip(self.vs, grads)]
        self.learning_rate = (self.learning_rate_init *
                              np.sqrt(1 - self.beta_2 ** self.t) /
                              (1 - self.beta_1 ** self.t))
        updates = [-self.learning_rate * m / (np.sqrt(v) + self.epsilon)
                   for m, v in zip(self.ms, self.vs)]

        return updates

class Optimizer(object):
    """
    Class for a general acquisition optimizer.

    :param bounds: list of tuple with bounds of the optimizer
    """

    def __init__(self, bounds):
        self.bounds = bounds

    def optimize(self, x0, f=None, df=None, f_df=None):
        """
        :param x0: initial point for a local optimizer.
        :param f: function to optimize.
        :param df: gradient of the function to optimize.
        :param f_df: returns both the function to optimize and its gradient.
        """
        raise NotImplementedError("The optimize method is not implemented in the parent class.")


class OptSgd(Optimizer):
    '''
    (Stochastic) gradient descent algorithm.
    '''
    def __init__(self, bounds, maxiter=50):
        super(OptSgd, self).__init__(bounds)
        self.maxiter = maxiter

    def optimize(self, x0, f=None, df=None, f_df=None):
        """
        :param x0: initial point for a local optimizer.
        :param f: function to optimize.
        :param df: gradient of the function to optimize.
        :param f_df: returns both the function to optimize and its gradient.
        """

        x = x0
        fx, dfx = f_df(x)
        for t  in range(self.maxiter):

            x = x - 0.3*np.power(t+1,-0.7)*dfx
            #print(x)
            for k in range(x.shape[1]):
                if x[0,k] < self.bounds[k][0]:
                    x[0,k] = self.bounds[k][0]
                elif x[0,k] > self.bounds[k][1]:
                    x[0,k] = self.bounds[k][1]
                    
            f_previous = fx       
            fx, dfx = f_df(x)

            if np.absolute(fx - f_previous) < 1e-5:
                break

        x = np.atleast_2d(x)
        fx = np.atleast_2d(fx)
        return x, fx
    

class OptLbfgs(Optimizer):
    '''
    Wrapper for l-bfgs-b to use the true or the approximate gradients.
    '''
    def __init__(self, bounds, maxiter=50):
        super(OptLbfgs, self).__init__(bounds)
        self.maxiter = maxiter

    def optimize(self, x0, f=None, df=None, f_df=None):
        """
        :param x0: initial point for a local optimizer.
        :param f: function to optimize.
        :param df: gradient of the function to optimize.
        :param f_df: returns both the function to optimize and its gradient.
        """

        import scipy.optimize
        if f_df is None and df is not None:
            f_df = lambda x: float(f(x)), df(x)

        # print("self.maxiter", self.maxiter)
        if f_df is None and df is None:

            res = scipy.optimize.fmin_l_bfgs_b(f, x0=x0, bounds=self.bounds, approx_grad=True, maxiter=self.maxiter) #factr=1e4
        else:

            res = scipy.optimize.fmin_l_bfgs_b(f_df, x0=x0, bounds=self.bounds, maxiter=self.maxiter, factr=1e7)

        ### --- We check here if the the optimizer moved. It it didn't we report x0 and f(x0) as scipy can return NaNs
        if res[2]['task'] == b'ABNORMAL_TERMINATION_IN_LNSRCH':
            print("ABNORMAL_TERMINATION_IN_LNSRCH")
            result_x = np.atleast_2d(res[0])
            result_fx = np.atleast_2d(res[1])
            # result_x  = np.atleast_2d(x0)
            # result_fx =  np.atleast_2d(f(x0))
        else:
            result_x = np.atleast_2d(res[0])
            result_fx = np.atleast_2d(res[1])
            
        #print(res[2])
        #print("Optimize result_x, result_fx",result_x, result_fx)
        return result_x, result_fx

class Nelder_Mead(Optimizer):
    '''
    Wrapper for l-bfgs-b to use the true or the approximate gradients.
    '''

    def __init__(self, bounds, maxiter=20):
        super(Nelder_Mead, self).__init__(bounds)
        self.maxiter = maxiter

    def optimize(self, x0, f=None, df=None, f_df=None):
        """
        :param x0: initial point for a local optimizer.
        :param f: function to optimize.
        :param df: gradient of the function to optimize.
        :param f_df: returns both the function to optimize and its gradient.
        """

        import scipy.optimize
        if f_df is None and df is not None:
            f_df = lambda x: float(f(x)), df(x)

        if f_df is None and df is None:
            print("self.maxiter",self.maxiter)
            res = scipy.optimize.minimize(f, method="nelder-mead",x0=x0, bounds=self.bounds, tol=1e-06, options={ 'maxiter':self.maxiter})  # factr=1e4
        else:

            print("Use an optimiser with gradient information")

        x0 = np.atleast_2d(x0)
        result_x = np.atleast_2d(res["x"])

        for k in range(x0.shape[1]):
            if result_x[0, k] < self.bounds[k][0]:
                result_x[0, k] = self.bounds[k][0]
            elif result_x[0, k] > self.bounds[k][1]:
                result_x[0, k] = self.bounds[k][1]
        ### --- We check here if the the optimizer moved. It it didn't we report x0 and f(x0) as scipy can return NaNs
        if res['message'] == b'ABNORMAL_TERMINATION_IN_LNSRCH':
            # result_x = np.atleast_2d(x0)
            # result_fx = np.atleast_2d(f(x0))
            print("ABNORMAL_TERMINATION_IN_LNSRCH")
            result_x = np.atleast_2d(res["x"])
            result_fx = np.atleast_2d(res["fun"])
        else:
            result_x = np.atleast_2d(res["x"])
            result_fx = np.atleast_2d(res["fun"])

        return result_x, result_fx


from scipy import optimize
from copy import copy, deepcopy

class Adam(Optimizer):
    def __init__(self, bounds, maxiter=200, iters=100, learning_rate=0.2, av=3, beta_1 = 0.5, beta_2=0.5):
        super(Adam, self).__init__(bounds)
        self.maxiter = maxiter
        self.learning_rate = learning_rate
        self.beta1 = beta_1
        self.beta2 = beta_2
        self.av = av
        self.iters = iters

    def optimize(self, x0, f=None, df=None, f_df=None):
        """
        Adam SGD optimizer.
        Parameters
        ----------
        val_grad: function that returns both fun value and gradient
        x0: np array, starting point
        iters: int, number of Adam iterations
        av: int, number of points to average over to predict f
        learning_rate : float, default=0.5
            The initial learning rate used. It controls the step-size in updating
            the weights
        beta1 : float, default=0.5
            Exponential decay rate for estimates of first moment vector, should be
            in [0, 1)
        beta2 : float, default=0.5
            Exponential decay rate for estimates of second moment vector, should be
            in [0, 1)
        Returns
        -------
        top_x: best observed x value
        top_fn: best observed output
        """
        assert self.iters > self.av, "cannot average over fewer points than iterations!"

        optimizer = AdamOptimizer(params=[x0],
                                  learning_rate_init=self.learning_rate,
                                  beta_1=self.beta1,
                                  beta_2=self.beta2)

        f_hist = []
        top_f = -np.Inf
        top_x = 0
        lb = np.zeros(x0.shape[1])
        ub = np.zeros(x0.shape[1])
        for k in range(x0.shape[1]):
            lb[k] = self.bounds[k][0]
            ub[k] = self.bounds[k][1]

        for _ in range(self.iters):

            x = optimizer.params[0]
            f, g = f_df(x)

            f_hist.append(f)

            # update params and (optionally) force x to stay in bounds
            optimizer.update_params([-g])

            optimizer.params[0] = np.clip(
                a=optimizer.params[0], a_min=lb, a_max=ub)


            if self.iters > self.av:
                pred_f = np.mean(f_hist[-self.av:])

            # of course this is stochastic but ohwell!
            if pred_f > top_f:
                top_f = copy(pred_f)
                top_x = copy(x)

        return top_x, top_f


class OptDirect(Optimizer):
    '''
    Wrapper for DIRECT optimization method. It works partitioning iteratively the domain
    of the function. Only requires f and the box constraints to work.

    '''
    def __init__(self, bounds, maxiter=1000):
        super(OptDirect, self).__init__(bounds)
        self.maxiter = maxiter
        assert self.space.has_types['continuous']

    def optimize(self, x0, f=None, df=None, f_df=None):
        """
        :param x0: initial point for a local optimizer.
        :param f: function to optimize.
        :param df: gradient of the function to optimize.
        :param f_df: returns both the function to optimize and its gradient.
        """
        # Based on the documentation of DIRECT, it does not seem we can pass through an initial point x0
        try:
            from DIRECT import solve
            def DIRECT_f_wrapper(f):
                def g(x, user_data):
                    return f(np.array([x])), 0
                return g
            lB = np.asarray(self.bounds)[:,0]
            uB = np.asarray(self.bounds)[:,1]
            x,_,_ = solve(DIRECT_f_wrapper(f),lB,uB, maxT=self.maxiter)
            return np.atleast_2d(x), f(np.atleast_2d(x))
        except ImportError:
            print("Cannot find DIRECT library, please install it to use this option.")


class OptCma(Optimizer):
    '''
    Wrapper the Covariance Matrix Adaptation Evolutionary strategy (CMA-ES) optimization method. It works generating
    an stochastic search based on multivariate Gaussian samples. Only requires f and the box constraints to work.

    '''
    def __init__(self, bounds, maxiter=5):
        super(OptCma, self).__init__(bounds)
        self.maxiter = maxiter

    def optimize(self, x0, f=None, df=None, f_df=None):
        """
        :param x0: initial point for a local optimizer.
        :param f: function to optimize.
        :param df: gradient of the function to optimize.
        :param f_df: returns both the function to optimize and its gradient.
        """
        try:
            import cma
            def CMA_f_wrapper(f):
                def g(x):
                    return f(np.array([x]))[0][0]
                return g
            lB = np.asarray(self.bounds)[:,0]
            uB = np.asarray(self.bounds)[:,1]
            x = cma.fmin(CMA_f_wrapper(f), x0, 0.6, options={"bounds":[lB, uB], "verbose":-1})[0]
            return np.atleast_2d(x), f(np.atleast_2d(x))
        except ImportError:
            print("Cannot find cma library, please install it to use this option.")
        except:
            print("CMA does not work in problems of dimension 1.")


def apply_optimizer(optimizer, x0, f=None, df=None, f_df=None, duplicate_manager=None, context_manager=None, space=None):
    """
    :param x0: initial point for a local optimizer (x0 can be defined with or without the context included).
    :param f: function to optimize.
    :param df: gradient of the function to optimize.
    :param f_df: returns both the function to optimize and its gradient.
    :param duplicate_manager: logic to check for duplicate (always operates in the full space, context included)
    :param context_manager: If provided, x0 (and the optimizer) operates in the space without the context
    :param space: GPyOpt class design space.
    """

   # x0 = np.atleast_2d(x0)
    

    ## --- Compute a new objective that inputs non context variables but that takes into account the values of the context ones.
    ## --- It does nothing if no context is passed
    #problem = OptimizationWithContext(x0=x0, f=f, df=df, f_df=f_df, context_manager=context_manager)

    #if context_manager:
        #print('context manager')
        #add_context = lambda x : context_manager._expand_vector(x)
    #else:
        #add_context = lambda x : x

    #if duplicate_manager and duplicate_manager.is_unzipped_x_duplicate(x0):
        #raise ValueError("The starting point of the optimizer cannot be a duplicate.")

    ## --- Optimize point
    #optimized_x, suggested_fx = optimizer.optimize(problem.x0_nocontext, problem.f_nocontext, problem.df_nocontext, problem.f_df_nocontext)
        
    ## --- Add context and round according to the type of variables of the design space
    #suggested_x_with_context = add_context(optimized_x)
    #suggested_x_with_context_rounded = space.round_optimum(suggested_x_with_context)

    ## --- Run duplicate_manager
    #if duplicate_manager and duplicate_manager.is_unzipped_x_duplicate(suggested_x_with_context_rounded):
        #suggested_x, suggested_fx = x0, np.atleast_2d(f(x0))
    #else:
        #suggested_x, suggested_fx = suggested_x_with_context_rounded, f(suggested_x_with_context_rounded)

    #if f(x0).shape[0]==2:
    # print("x0",x0,"f",f(x0))
    suggested_x, suggested_fx = optimizer.optimize(x0, f, df, f_df)
    #suggested_fx = f(suggested_x)


    return suggested_x, suggested_fx


class OptimizationWithContext(object):

    def __init__(self, x0, f, df=None, f_df=None, context_manager=None):
        '''
        Constructor of an objective function that takes as input a vector x of the non context variables
        and retunrs a value in which the context variables have been fixed.
        '''
        self.x0 = np.atleast_2d(x0)
        self.f = f
        self.df = df
        self.f_df = f_df
        self.context_manager = context_manager

        if not context_manager:

            self.x0_nocontext = x0
            self.f_nocontext  =  self.f
            self.df_nocontext  =  self.df
            self.f_df_nocontext = self.f_df

        else:
            #print('context')
            self.x0_nocontext = self.x0[:,self.context_manager.noncontext_index]
            self.f_nocontext  = self.f_nc
            if self.f_df is None:
                self.df_nocontext = None
                self.f_df_nocontext = None
            else:
                self.df_nocontext = self.df
                self.f_df_nocontext  = self.f_df#self.f_df_nc

    def f_nc(self,x):
        '''
        Wrapper of *f*: takes an input x with size of the noncontext dimensions
        expands it and evaluates the entire function.
        '''
        x = np.atleast_2d(x)
        xx = self.context_manager._expand_vector(x)
        if x.shape[0] == 1:
            return self.f(xx)[0]
        else:
            return self.f(xx)

    def df_nc(self,x):
        '''
        Wrapper of the derivative of *f*: takes an input x with size of the not
        fixed dimensions expands it and evaluates the gradient of the entire function.
        '''
        x = np.atleast_2d(x)
        xx = self.context_manager._expand_vector(x)
        _, df_nocontext_xx = self.f_df(xx)
        df_nocontext_xx = df_nocontext_xx[:,np.array(self.context_manager.noncontext_index)]
        return df_nocontext_xx

    def f_df_nc(self,x):
        '''
        Wrapper of the derivative of *f*: takes an input x with size of the not
        fixed dimensions expands it and evaluates the gradient of the entire function.
        '''
        x = np.atleast_2d(x)
        xx = self.context_manager._expand_vector(x)
        f_nocontext_xx , df_nocontext_xx = self.f_df(xx)
        df_nocontext_xx = df_nocontext_xx[:,np.array(self.context_manager.noncontext_index)]
        return f_nocontext_xx, df_nocontext_xx


def choose_optimizer(optimizer_name, bounds):
        """
        Selects the type of local optimizer
        """          
        if optimizer_name == 'lbfgs':

            optimizer = OptLbfgs(bounds)
        
        elif optimizer_name == 'sgd':

            optimizer = OptSgd(bounds)

        elif optimizer_name == 'DIRECT':

            optimizer = OptDirect(bounds)

        elif optimizer_name == 'CMA':

            optimizer = OptCma(bounds)

        elif optimizer_name == 'Nelder_Mead':

            optimizer = Nelder_Mead(bounds)

        elif optimizer_name == "Adam":

            optimizer = Adam(bounds)
        else:
            raise InvalidVariableNameError('Invalid optimizer selected.')

        return optimizer
