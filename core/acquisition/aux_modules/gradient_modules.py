import numpy as np
import matplotlib.pyplot as plt

class gradients(object):
    def __init__(self):


    def compute_gradient_mu_xnew(self,model,x, x_new):
        dmu_dX_inner = model.posterior_mean_gradient(x)
        dcov_dX_inner = self.model.posterior_covariance_gradient_partially_precomputed(x, x_new)

    def compute_gradient_sigma_xnew(self,model,x):

    def compute_gradient_pf_xnew(self,model,x):

    def delta_computation(self,model, x, delta=1e-3):

