# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

try:
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
except:
    pass
import numpy as np
from ..util.general import reshape
import torch
import math


class function2d:
    '''
    This is a benchmark of bi-dimensional functions interesting to optimize. 

    '''
    
    def plot(self):
        bounds = self.bounds
        x1 = np.linspace(bounds[0][0], bounds[0][1], 100)
        x2 = np.linspace(bounds[1][0], bounds[1][1], 100)
        X1, X2 = np.meshgrid(x1, x2)
        X = np.hstack((X1.reshape(100*100,1),X2.reshape(100*100,1)))
        Y = self.f(X)

        plt.figure()    
        plt.contourf(X1, X2, Y.reshape((100,100)),100)
        if (len(self.min)>1):    
            plt.plot(np.array(self.min)[:,0], np.array(self.min)[:,1], 'w.', markersize=20, label=u'Observations')
        else:
            plt.plot(self.min[0][0], self.min[0][1], 'w.', markersize=20, label=u'Observations')
        plt.colorbar()
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title(self.name)
        plt.show()


class rosenbrock(function2d):
    '''
    Cosines function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,bounds=None,sd=None):
        self.input_dim = 2
        if bounds is  None: self.bounds = [(-0.5,3),(-1.5,2)]
        else: self.bounds = bounds
        self.min = [(0, 0)]
        self.fmin = 0
        if sd==None: self.sd = 0
        else: self.sd=sd
        self.name = 'Rosenbrock'

    def f(self,X):
        X = reshape(X,self.input_dim)
        n = X.shape[0]
        if X.shape[1] != self.input_dim:
            return 'Wrong input dimension'
        else:
            fval = 100*(X[:,1]-X[:,0]**2)**2 + (X[:,0]-1)**2
            if self.sd ==0:
                noise = np.zeros(n).reshape(n,1)
            else:
                noise = np.random.normal(0,self.sd,n).reshape(n,1)
            return -fval.reshape(n,1) + noise


class f1_Binh(function2d):
    '''
    Goldstein function

    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''

    def __init__(self, bounds=None, sd=None):
        self.input_dim = 2
        if bounds is None:
            self.bounds = [(0, 5), (0, 3)]
        else:
            self.bounds = bounds
        self.min = [(0, 0)]
        self.fmin = 0
        if sd == None:
            self.sd = 0
        else:
            self.sd = sd
        self.name = 'f1_Binh'

    def f(self, X):
        X = reshape(X, self.input_dim)
        n = X.shape[0]
        if X.shape[1] != self.input_dim:
            return 'Wrong input dimension'
        else:
            x1 = X[:, 0]
            x2 = X[:, 1]
            fval = 4*x1**2.0 + 4*x2**2.0
            if self.sd == 0:
                noise = np.zeros(n).reshape(n, 1)
            else:
                noise = np.random.normal(0, self.sd, n).reshape(n, 1)
            return -fval.reshape(n, 1) + noise

class f2_Binh(function2d):
    '''
    Goldstein function

    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''

    def __init__(self, bounds=None, sd=None):
        self.input_dim = 2
        if bounds is None:
            self.bounds = [(0, 5), (0, 3)]
        else:
            self.bounds = bounds
        self.min = [(5, 3)]
        self.fmin = 4
        if sd == None:
            self.sd = 0
        else:
            self.sd = sd
        self.name = 'f2_Binh'

    def f(self, X):
        X = reshape(X, self.input_dim)
        n = X.shape[0]
        if X.shape[1] != self.input_dim:
            return 'Wrong input dimension'
        else:
            x1 = X[:, 0]
            x2 = X[:, 1]
            fval = (x1 - 5)**2.0 + (x2 - 5)**2.0
            if self.sd == 0:
                noise = np.zeros(n).reshape(n, 1)
            else:
                noise = np.random.normal(0, self.sd, n).reshape(n, 1)
            return -fval.reshape(n, 1) + noise


class beale(function2d):
    '''
    Cosines function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,bounds=None,sd=None):
        self.input_dim = 2
        if bounds is  None: self.bounds = [(-1,1),(-1,1)]
        else: self.bounds = bounds
        self.min = [(0, 0)]
        self.fmin = 0
        if sd==None: self.sd = 0
        else: self.sd=sd
        self.name = 'Beale'

    def f(self,X):
        X = reshape(X,self.input_dim)
        n = X.shape[0]
        if X.shape[1] != self.input_dim:
            return 'Wrong input dimension'
        else:
            fval = 100*(X[:,1]-X[:,0]**2)**2 + (X[:,0]-1)**2
            if self.sd ==0:
                noise = np.zeros(n).reshape(n,1)
            else:
                noise = np.random.normal(0,self.sd,n).reshape(n,1)
            return -fval.reshape(n,1) + noise


class dropwave(function2d):
    '''
    Cosines function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,bounds=None,sd=None):
        self.input_dim = 2
        if bounds is  None: self.bounds = [(-1,1),(-1,1)]
        else: self.bounds = bounds
        self.min = [(0, 0)]
        self.fmin = 0
        if sd==None: self.sd = 0
        else: self.sd=sd
        self.name = 'dropwave'

    def f(self,X):
        X = reshape(X,self.input_dim)
        n = X.shape[0]
        if X.shape[1] != self.input_dim:
            return 'Wrong input dimension'
        else:
            fval = - (1+np.cos(12*np.sqrt(X[:,0]**2+X[:,1]**2))) / (0.5*(X[:,0]**2+X[:,1]**2)+2) 
            if self.sd ==0:
                noise = np.zeros(n).reshape(n,1)
            else:
                noise = np.random.normal(0,self.sd,n).reshape(n,1)
            return -fval.reshape(n,1) + noise


class cosines(function2d):
    '''
    Cosines function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,bounds=None,sd=None):
        self.input_dim = 2
        if bounds is  None: self.bounds = [(0,1),(0,1)]
        else: self.bounds = bounds
        self.min = [(0.31426205,  0.30249864)]
        self.fmin = -1.59622468
        if sd==None: self.sd = 0
        else: self.sd=sd
        self.name = 'Cosines'

    def f(self,X):
        X = reshape(X,self.input_dim)
        n = X.shape[0]
        if X.shape[1] != self.input_dim:
            return 'Wrong input dimension'
        else:
            u = 1.6*X[:,0]-0.5
            v = 1.6*X[:,1]-0.5
            fval = 1-(u**2 + v**2 - 0.3*np.cos(3*np.pi*u) - 0.3*np.cos(3*np.pi*v) )
            if self.sd ==0:
                noise = np.zeros(n).reshape(n,1)
            else:
                noise = np.random.normal(0,self.sd,n).reshape(n,1)
            return -fval.reshape(n,1) + noise


class branin(function2d):
    '''
    Branin function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,bounds=None,a=None,b=None,c=None,r=None,s=None,t=None,sd=None):
        self.input_dim = 2
        if bounds is  None: self.bounds = [(-5,10),(1,15)]
        else: self.bounds = bounds
        if a==None: self.a = 1
        else: self.a = a           
        if b==None: self.b = 5.1/(4*np.pi**2)
        else: self.b = b
        if c==None: self.c = 5/np.pi
        else: self.c = c
        if r==None: self.r = 6
        else: self.r = r
        if s==None: self.s = 10 
        else: self.s = s
        if t==None: self.t = 1/(8*np.pi)
        else: self.t = t    
        if sd==None: self.sd = 0
        else: self.sd=sd
        self.min = [(-np.pi,12.275),(np.pi,2.275),(9.42478,2.475)] 
        self.fmin = 0.397887
        self.name = 'Branin'
    
    def f(self,X):
        X = reshape(X,self.input_dim)
        n = X.shape[0]
        if X.shape[1] != self.input_dim: 
            return 'Wrong input dimension'  
        else:
            x1 = X[:,0]
            x2 = X[:,1]
            fval = self.a * (x2 - self.b*x1**2 + self.c*x1 - self.r)**2 + self.s*(1-self.t)*np.cos(x1) + self.s 
            if self.sd ==0:
                noise = np.zeros(n).reshape(n,1)
            else:
                noise = np.random.normal(0,self.sd,n).reshape(n,1)
            return -fval.reshape(n,1) - noise


class goldstein(function2d):
    '''
    Goldstein function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,bounds=None,sd=None):
        self.input_dim = 2
        if bounds is  None: self.bounds = [(-2,2),(-2,2)]
        else: self.bounds = bounds
        self.min = [(0,-1)]
        self.fmin = 3
        if sd==None: self.sd = 0
        else: self.sd=sd
        self.name = 'Goldstein'

    def f(self,X):
        X = reshape(X,self.input_dim)
        n = X.shape[0]
        if X.shape[1] != self.input_dim:
            return 'Wrong input dimension'
        else:
            x1 = X[:,0]
            x2 = X[:,1]
            fact1a = (x1 + x2 + 1)**2
            fact1b = 19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2
            fact1 = 1 + fact1a*fact1b
            fact2a = (2*x1 - 3*x2)**2
            fact2b = 18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2
            fact2 = 30 + fact2a*fact2b
            fval = fact1*fact2
            if self.sd ==0:
                noise = np.zeros(n).reshape(n,1)
            else:
                noise = np.random.normal(0,self.sd,n).reshape(n,1)
            return fval.reshape(n,1) + noise

class test_function_2(function2d):
    def __init__(self, bounds=None, sd_obj=None, sd_c = None):
        self.input_dim = 2
        if bounds is None:
            self.bounds = [(0, 1), (0, 1)]
        else:
            self.bounds = bounds
        self.min = [(0.2018, 0.833)]
        self.fmin = 0.748
        self.sd_obj = sd_obj
        self.sd_c = sd_c
        self.name = 'test_function_2'

    def f(self, x, offset=-10, true_val=False):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        n = x.shape[0]
        x1 = x[:, 0]
        x2 = x[:, 1]
        term2 = -(x1 - 1)**2.0
        term3 = -(x2  - 0.5 )** 2.0
        fval = term2 + term3
        if self.sd_obj == 0 or true_val:
            noise = np.zeros(n).reshape(n, 1)
        else:
            noise = np.random.normal(0, self.sd_obj, n).reshape(n, 1)
        # print("fval",-fval.reshape(-1, 1) + noise.reshape(-1, 1))
        return np.array(-(fval.reshape(n,1) + offset)+ noise.reshape(-1, 1)).reshape(-1) #torch.reshape(-(fval.reshape(n,1) + offset)+ noise.reshape(-1, 1), -1)

    def c1(self, x, true_val=False):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        n = x.shape[0]
        x1 = x[:, 0]
        x2 = x[:, 1]
        term1 = (x1 - 3)**2.0
        term2 = (x2 + 2)**2.0
        term3 = -12
        fval = (term1 + term2)*np.exp(-x2**7)+term3
        if self.sd_c == 0 or true_val:
            noise = np.zeros(n).reshape(n, 1)
        else:
            noise = np.random.normal(0, self.sd_c, n).reshape(n, 1)
        return np.array(fval.reshape(n, 1) +  noise.reshape(-1, 1)).reshape(-1)

    def c2(self, x, true_val=False):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        n = x.shape[0]
        x1 = x[:, 0]
        x2 = x[:, 1]
        fval = 10*x1 + x2 -7
        if self.sd_c == 0 or true_val:
            noise = np.zeros(n).reshape(n, 1)
        else:
            noise = np.random.normal(0, self.sd_c, n).reshape(n, 1)
        return np.array(fval.reshape(n, 1) +  noise.reshape(-1, 1)).reshape(-1)

    def c3(self, x, true_val=False):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        n = x.shape[0]
        x1 = x[:, 0]
        x2 = x[:, 1]
        term1 = (x1 - 0.5)**2.0
        term2 = (x2 - 0.5)**2.0
        term3 = -0.2
        fval = term1 + term2 + term3
        if self.sd_c == 0 or true_val:
            noise = np.zeros(n).reshape(n, 1)
        else:
            noise = np.random.normal(0, self.sd_c, n).reshape(n, 1)
        return np.array(fval.reshape(n, 1) +  noise.reshape(-1, 1)).reshape(-1)

    def c(self, x, true_val=False):
        return [self.c1(x, true_val=true_val), self.c2(x, true_val=true_val), self.c3(x, true_val=true_val)]

    def func_val(self, x):
        Y = self.f(x, true_val=True)
        C = self.c(x, true_val=True)
        out = Y.reshape(-1)* np.product(np.concatenate(C, axis=1) < 0, axis=1).reshape(-1)
        out = np.array(out).reshape(-1)
        return -out




class test_function_2_torch(function2d):
    def __init__(self, bounds=None, sd=None):
        self.input_dim = 2
        if bounds is None:
            self.bounds = [(0, 1), (0, 1)]
        else:
            self.bounds = bounds
        self.min = [(0.2018, 0.833)]
        self.fmin = 0.748
        self.sd = sd
        self.name = 'test_function_2'

    def f(self, x, offset=0, true_val=False):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        n = x.shape[0]
        x1 = x[:, 0]
        x2 = x[:, 1]
        term2 = -(x1 - 1)**2.0
        term3 = -(x2  - 0.5 )** 2.0
        fval = term2 + term3
        # if self.sd == 0 or true_val:
        #     noise = np.zeros(n).reshape(n, 1)
        # else:
        #     noise = np.random.normal(0, self.sd, n).reshape(n, 1)
        return torch.reshape(-fval, (-1,))  #torch.reshape(-(fval.reshape(n,1))+ noise.reshape(-1, 1), -1) ##

    def c1(self, x, true_val=False):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        n = x.shape[0]
        x1 = x[:, 0]
        x2 = x[:, 1]
        term1 = (x1 - 3)**2.0
        term2 = (x2 + 2)**2.0
        term3 = -12
        fval = (term1 + term2)*torch.exp(-x2**7)+term3
        # print("fval",-fval.reshape(-1, 1))
        return torch.reshape(fval, (-1,))#torch.reshape(fval, -1)

    def c2(self, x, true_val=False):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        n = x.shape[0]
        x1 = x[:, 0]
        x2 = x[:, 1]
        fval = 10*x1 + x2 -7
        # print("fval",-fval.reshape(-1, 1))
        return torch.reshape(fval, (-1,))#torch.reshape(fval, -1)

    def c3(self, x, true_val=False):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        n = x.shape[0]
        x1 = x[:, 0]
        x2 = x[:, 1]
        term1 = (x1 - 0.5)**2.0
        term2 = (x2 - 0.5)**2.0
        term3 = -0.2
        fval = term1 + term2 + term3
        # print("fval",-fval.reshape(-1, 1))
        return torch.reshape(fval, (-1,))#np.array(fval.reshape(n,1)).reshape(-1)

    def c(self, x, true_val=False):
        return [self.c1(x), self.c2(x), self.c3(x)]

    def func_val(self, x):
        Y = self.f(x, true_val=True)
        C = self.c(x)
        out = Y.reshape(-1)* np.product(np.concatenate(C, axis=1) < 0, axis=1).reshape(-1)
        out = np.array(out).reshape(-1)
        return -out

class mistery(function2d):
    '''
    Six hump camel function

    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''

    def __init__(self, bounds=None, sd_obj=None, sd_c=None):
        self.input_dim = 2
        if bounds is None:
            self.bounds = [(0, 5), (0, 5)]
        else:
            self.bounds = bounds
        self.min = [(2.7450, 2.3523)]
        self.fmin = 1.1743
        self.sd_obj = sd_obj
        self.sd_c = sd_c
        self.name = 'Mistery'

    def f(self, x, offset=0.0, true_val=False):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        n = x.shape[0]
        x1 = x[:, 0]
        x2 = x[:, 1]
        term1 = 2
        term2 = 0.01 * (x2 - x1 ** 2.0) ** 2.0
        term3 = (1 - x1) ** 2
        term4 = 2 * (2 - x2) ** 2
        term5 = 7 * np.sin(0.5 * x1) * np.sin(0.7 * x1 * x2)
        fval = term1 + term2 + term3 + term4 + term5 - 5
        if self.sd_obj == 0 or true_val:
            noise = np.zeros(n).reshape(n, 1)
        else:
            noise = np.random.normal(0, self.sd_obj, n).reshape(n, 1)

        # print("fval",-fval.reshape(-1, 1) + noise.reshape(-1, 1))

        return np.array(-(fval.reshape(n, 1) ) + noise.reshape(-1, 1)).reshape(-1)

    def c(self, x,  true_val=False):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        n = x.shape[0]
        x1 = x[:, 0]
        x2 = x[:, 1]
        fval = -np.sin(x1 - x2 - np.pi / 8.0)

        if self.sd_c == 0 or true_val:
            noise = np.zeros(n).reshape(n, 1)
        else:
            noise = np.random.normal(0, self.sd_c, n).reshape(n, 1)

            # print("signal", fval, "noise", noise)
        return np.array(fval.reshape(n, 1) +  noise.reshape(-1, 1)).reshape(-1)

    def func_val(self, x):
        Y = self.f(x, true_val=True)
        C = self.c(x, true_val=True)
        out = Y * (C < 0)
        out = np.array(out).reshape(-1)
        return -out


class mistery_torch(function2d):
    '''
    Six hump camel function

    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''

    def __init__(self, bounds=None):
        self.input_dim = 2
        if bounds is None:
            self.bounds = [(0, 5), (0, 5)]
        else:
            self.bounds = bounds
        self.min = [(2.7450, 2.3523)]
        self.fmin = 1.1743

        self.name = 'Mistery'

    def f(self, x, offset=0.0, true_val=False):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        n = x.shape[0]
        x1 = x[:, 0]
        x2 = x[:, 1]
        term1 = 2
        term2 = 0.01 * (x2 - x1 ** 2.0) ** 2.0
        term3 = (1 - x1) ** 2
        term4 = 2 * (2 - x2) ** 2
        term5 = 7 * torch.sin(0.5 * x1) * torch.sin(0.7 * x1 * x2)
        fval = term1 + term2 + term3 + term4 + term5 - 5

        return torch.reshape(-fval, (-1,) ) #np.array(-(fval.reshape(n, 1)) + noise.reshape(-1, 1)).reshape(-1) #torch.reshape(-fval, (-1,) )  #

    def c(self, x,  true_val=False):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        n = x.shape[0]
        x1 = x[:, 0]
        x2 = x[:, 1]
        fval = -torch.sin(x1 - x2 - np.pi / 8.0)
        # print("fval",-fval.reshape(-1, 1))
        return torch.reshape(fval, (-1,)) # np.array(fval.reshape(n, 1)).reshape(-1)




class new_brannin(function2d):
    '''
    Six hump camel function

    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''

    def __init__(self, bounds=None, sd_obj=None, sd_c = None):
        self.input_dim = 2
        if bounds is None:
            self.bounds = [(-5, 10), (0, 15)]
        else:
            self.bounds = bounds
        self.min = [(3.26, 0.05)]
        self.fmin = 268.781
        self.sd_obj = sd_obj
        self.sd_c = sd_c
        self.name = 'new_brannin'

    def f(self, x, offset=0,  true_val=False):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        n = x.shape[0]
        x1 = x[:, 0]
        x2 = x[:, 1]
        term1 = -(x1 - 10)**2
        term2 = -(x2 - 15)**2.0
        fval = term1 + term2
        if self.sd_obj == 0 or true_val:
            noise = np.zeros(n).reshape(n, 1)
        else:
            noise = np.random.normal(0, self.sd_obj, n).reshape(n, 1)
        # print("fval",-fval.reshape(-1, 1) + noise.reshape(-1, 1))

        return np.array(-(fval.reshape(n, 1) ) + noise.reshape(-1, 1)).reshape(-1)

    def c(self, x,  true_val=False):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        n = x.shape[0]
        x1 = x[:, 0]
        x2 = x[:, 1]
        term1 = (x2 - (5.1/(4 * np.pi**2.0))*x1**2.0 + (5.0/np.pi)*x1 - 6)**2.0
        term2 = 10 * (1 - (1.0/(8 * np.pi)))*np.cos(x1)
        term3 = 5
        fval = term1 + term2 + term3
        if self.sd_c == 0 or true_val:
            noise = np.zeros(n).reshape(n, 1)
        else:
            noise = np.random.normal(0, self.sd_c, n).reshape(n, 1)

        return np.array(fval.reshape(n, 1) +  noise.reshape(-1, 1)).reshape(-1)

    def func_val(self, x):
        Y = self.f(x, true_val=True)
        C = self.c(x, true_val = True)
        out = Y * (C < 0)
        out = np.array(out).reshape(-1)
        return -out


class new_brannin_torch(function2d):
    '''
    Six hump camel function

    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''

    def __init__(self, bounds=None, sd=None):
        self.input_dim = 2
        if bounds is None:
            self.bounds = [(-5, 10), (0, 15)]
        else:
            self.bounds = bounds
        self.min = [(3.26, 0.05)]
        self.fmin = 268.781
        self.sd = sd
        self.name = 'new_brannin'
        self.pi = torch.acos(torch.zeros(1)).item() * 2

    def f(self, x, offset=0,  true_val=False):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        n = x.shape[0]
        x1 = x[:, 0]
        x2 = x[:, 1]
        term1 = -(x1 - 10)**2
        term2 = -(x2 - 15)**2.0
        fval = term1 + term2

        return torch.reshape(-fval, (-1,)) #np.array(-(fval.reshape(n,1))+ noise.reshape(-1, 1)).reshape(-1) #torch.reshape(-fval, (-1,))  #np.array(-(fval.reshape(n,1) + offset)+ noise.reshape(-1, 1)).reshape(-1)

    def c(self, x,  true_val=False):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        n = x.shape[0]
        x1 = x[:, 0]
        x2 = x[:, 1]
        term1 = (x2 - (5.1/(4 * self.pi**2.0))*x1**2.0 + (5.0/self.pi)*x1 - 6)**2.0
        term2 = 10 * (1 - (1.0/(8 * self.pi)))*torch.cos(x1)
        term3 = 5
        fval = term1 + term2 + term3
        # print("fval",-fval.reshape(-1, 1))
        return torch.reshape(fval, (-1,))  #np.array(fval.reshape(n,1)).reshape(-1)

    def func_val(self, x):
        Y = self.f(x, true_val=True)
        C = self.c(x)
        out = Y * (C < 0)
        out = np.array(out).reshape(-1)
        return -out


class sixhumpcamel(function2d):
    '''
    Six hump camel function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,bounds=None,sd=None):
        self.input_dim = 2
        if bounds is  None: self.bounds = [(-2,2),(-1,1)]
        else: self.bounds = bounds
        self.min = [(0.0898,-0.7126),(-0.0898,0.7126)]
        self.fmin = -1.0316
        if sd==None: self.sd = 0
        else: self.sd=sd
        self.name = 'Six-hump camel'

    def f(self,x):
        x = reshape(x,self.input_dim)
        n = x.shape[0]
        if x.shape[1] != self.input_dim:
            return 'wrong input dimension'
        else:
            x1 = x[:,0]
            x2 = x[:,1]
            term1 = (4-2.1*x1**2+(x1**4)/3) * x1**2
            term2 = x1*x2
            term3 = (-4+4*x2**2) * x2**2
            fval = term1 + term2 + term3
            if self.sd ==0:
                noise = np.zeros(n).reshape(n,1)
            else:
                noise = np.random.normal(0,self.sd,n).reshape(n,1)
            return -fval.reshape(n,1) - noise

class mccormick(function2d):
    '''
    Mccormick function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,bounds=None,sd=None):
        self.input_dim = 2
        if bounds is  None: self.bounds = [(-1.5,4),(-3,4)]
        else: self.bounds = bounds
        self.min = [(-0.54719,-1.54719)]
        self.fmin = -1.9133
        if sd==None: self.sd = 0
        else: self.sd=sd
        self.name = 'Mccormick'

    def f(self,x):
        x = reshape(x,self.input_dim)
        n = x.shape[0]
        if x.shape[1] != self.input_dim:
            return 'wrong input dimension'
        else:
            x1 = x[:,0]
            x2 = x[:,1]
            term1 = np.sin(x1 + x2)
            term2 = (x1 - x2)**2
            term3 = -1.5*x1
            term4 = 2.5*x2
            fval = term1 + term2 + term3 + term4 + 1
            if self.sd ==0:
                noise = np.zeros(n).reshape(n,1)
            else:
                noise = np.random.normal(0,self.sd,n).reshape(n,1)
            return fval.reshape(n,1) + noise

class powers(function2d):
    '''
    Powers function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,bounds=None,sd=None):
        self.input_dim = 2
        if bounds is  None: self.bounds = [(-1,1),(-1,1)]
        else: self.bounds = bounds
        self.min = [(0,0)]
        self.fmin = 0
        if sd==None: self.sd = 0
        else: self.sd=sd
        self.name = 'Sum of Powers'

    def f(self,x):
        x = reshape(x,self.input_dim)
        n = x.shape[0]
        if x.shape[1] != self.input_dim:
            return 'wrong input dimension'
        else:
            x1 = x[:,0]
            x2 = x[:,1]
            fval = abs(x1)**2 + abs(x2)**3
            if self.sd ==0:
                noise = np.zeros(n).reshape(n,1)
            else:
                noise = np.random.normal(0,self.sd,n).reshape(n,1)
            return fval.reshape(n,1) + noise

class eggholder:
    def __init__(self,bounds=None,sd=None):
        self.input_dim = 2
        if bounds is  None: self.bounds = [(-512,512),(-512,512)]
        else: self.bounds = bounds
        self.min = [(512,404.2319)]
        self.fmin = -959.6407
        if sd==None: self.sd = 0
        else: self.sd=sd
        self.name = 'Egg-holder'

    def f(self,X):
        X = reshape(X,self.input_dim)
        n = X.shape[0]
        if X.shape[1] != self.input_dim:
            return 'Wrong input dimension'
        else:
            x1 = X[:,0]
            x2 = X[:,1]
            fval = -(x2+47) * np.sin(np.sqrt(abs(x2+x1/2+47))) + -x1 * np.sin(np.sqrt(abs(x1-(x2+47))))
            if self.sd ==0:
                noise = np.zeros(n).reshape(n,1)
            else:
                noise = np.random.normal(0,self.sd,n).reshape(n,1)
            return fval.reshape(n,1) + noise












