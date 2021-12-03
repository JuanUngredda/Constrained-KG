import numpy as np

class test_function_2():
    def __init__(self, bounds=None):
        self.input_dim = 2
        if bounds is None:
            self.bounds = [(0, 1), (0, 1)]
        else:
            self.bounds = bounds
        self.min = [(0.2018, 0.833)]
        self.fmin = 0.748
        self.name = 'test_function_2'

    def f(self, x):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        n = x.shape[0]
        x1 = x[:, 0]
        x2 = x[:, 1]
        term2 = -(x1 - 1)**2.0
        term3 = -(x2  - 0.5 )** 2.0
        fval = term2 + term3
        return -fval.reshape(n,1)

    def c1(self, x):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        n = x.shape[0]
        x1 = x[:, 0]
        x2 = x[:, 1]
        term1 = (x1 - 3)**2.0
        term2 = (x2 + 2)**2.0
        term3 = -12
        fval = (term1 + term2)*np.exp(-x2**7)+term3
        return fval.reshape(n, 1)

    def c2(self, x):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        n = x.shape[0]
        x1 = x[:, 0]
        x2 = x[:, 1]
        fval = 10*x1 + x2 -7
        return fval.reshape(n, 1)

    def c3(self, x):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        n = x.shape[0]
        x1 = x[:, 0]
        x2 = x[:, 1]
        term1 = (x1 - 0.5)**2.0
        term2 = (x2 - 0.5)**2.0
        term3 = -0.2
        fval = term1 + term2 + term3
        return fval.reshape(n, 1)