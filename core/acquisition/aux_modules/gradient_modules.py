import numpy as np
import matplotlib.pyplot as plt
import scipy
from functools import partial
import warnings

class gradients(object):
    def __init__(self, x_new=None, model=None, xopt =None, Z=None, aux=None, aux2=None, varX=None, dvar_dX=None, test_samples= None, X_inner=None, precompute_grad = False):
        self.xnew = x_new*1
        self.model = model
        self.Z = Z*1
        self.aux = aux*1
        if xopt is not None:
            self.xopt = xopt*1
        if aux2 is not None:
            self.aux2 = aux2*1
        self.test_samples = test_samples
        if varX is not None:
            self.varX = varX*1
        if dvar_dX is not None:
            self.dvar_dX = dvar_dX*1
        if X_inner is not None:
            self.cov = self.model.posterior_covariance_between_points_partially_precomputed(X_inner, self.xnew)[:, :, 0]*1
            if precompute_grad:
                self.dcov = self.model.posterior_covariance_gradient_partially_precomputed(X_inner, self.xnew)*1


    def compute_value_mu_xnew(self, x, m=None):
        if m is None:
            muX_inner = self.model.posterior_mean(x)
            cov =  self.cov*1

            func_val = []
            for j in range(muX_inner.shape[0]):
                a = muX_inner[j]
                b = np.sqrt(self.aux[j] * np.square(cov[j]))
                func_val.append(np.reshape(a + b * self.Z, (len(x), 1)))
            return func_val

        else:
            muX_inner = self.model.output[m].posterior_mean(x)
            cov =  self.cov[m]*1

            #print("compute_value_mu_xnew", cov, self.cov)
            a = muX_inner
            b = np.sqrt(self.aux[m] * np.square(cov))
            func_val = np.reshape(a.reshape(-1) + b.reshape(-1) * self.Z, (len(x), 1))
            return func_val

    def compute_value_mu_xopt(self, xopt, m=None):

        if m is None:
            muX_inner = self.model.posterior_mean(xopt)
            cov = self.model.posterior_covariance_between_points(xopt, self.xnew)[:, 0, 0]

            func_val = []
            for j in range(muX_inner.shape[0]):
                a = muX_inner[j]
                b = np.sqrt(self.aux[j] * np.square(cov[j]))
                func_val.append(np.reshape(a + b * self.Z, (len(xopt), 1)))
            return func_val
        else:
            muX_inner = self.model.output[m].posterior_mean(xopt)
            cov = self.model.posterior_covariance_between_points(xopt, self.xnew)[:, 0, 0][m]*1
            aux = self.aux[m]
            a = muX_inner
            b = np.sqrt(aux * np.square(cov))
            func_val = np.reshape(a.reshape(-1) + b.reshape(-1)* self.Z, (len(xopt), 1))
            return func_val

    def compute_gradient_mu_xnew(self,x ,m=None):

        if m is None:
            dmu_dX_inner = self.model.posterior_mean_gradient(x)
            dcov_dX_inner = self.dcov * 1
            cov = self.cov* 1

            # print("compute_gradient_mu_xnew", cov, self.cov, dcov_dX_inner, self.dcov)
            b = np.sqrt(self.aux * np.square(cov))
            for k in range(x.shape[1]):

                dcov_dX_inner[:, :, k] = np.multiply(cov, dcov_dX_inner[:, :, k])

            db_dX_inner = np.tensordot(self.aux, dcov_dX_inner, axes=1)

            db_dX_inner = np.multiply(np.reciprocal(b), db_dX_inner.T).T

            func_gradient = np.reshape(dmu_dX_inner + db_dX_inner * self.Z, x.shape)

            return func_gradient

        else:
            dmu_dX_inner = self.model.output[m].posterior_mean_gradient(x)
            dcov_dX_inner =  self.dcov[m]*1 # self.model.posterior_covariance_gradient_partially_precomputed(x, self.xnew)
            cov = self.cov[m]*1 # self.model.posterior_covariance_between_points_partially_precomputed(x, self.xnew)[:, :, 0]
            aux = np.array([self.aux[m]*1])

            # print("compute_gradient_mu_xnew", cov, self.cov, dcov_dX_inner, self.dcov)
            b = np.sqrt(aux * np.square(cov))
            for k in range(x.shape[1]):

                dcov_dX_inner[ :, k] = np.multiply(cov, dcov_dX_inner[ :, k])

            dcov_dX_inner = dcov_dX_inner[np.newaxis]
            db_dX_inner = np.tensordot(aux, dcov_dX_inner[:,:,:], axes=1)
            db_dX_inner = np.multiply(np.reciprocal(b), db_dX_inner.T).T
            func_gradient = np.reshape(dmu_dX_inner.reshape(-1) + db_dX_inner.reshape(-1) * self.Z, x.shape)
            return func_gradient

    def compute_grad_mu_xopt(self, xopt,m=None):
        if m is None:
            cov_opt = self.model.posterior_covariance_between_points(xopt, self.xnew)[:, 0, 0]
            dcov_opt_dx = self.model.posterior_covariance_gradient(self.xnew, xopt)[:, 0, :]
            b = np.sqrt(np.dot(self.aux, np.square(cov_opt)))
            term1 = np.multiply(self.varX * cov_opt, dcov_opt_dx.T)
            term2 = np.multiply(np.square(cov_opt), self.dvar_dX.T)
            term3 = (2 * term1 - term2).T
            term4 = np.matmul(self.aux2, term3)
            term5 = 0.5 * self.Z * np.reciprocal(b)
            out = np.array(term5 * term4).reshape(-1)
            return out.reshape(1, -1)
        else:

            cov_opt = self.model.posterior_covariance_between_points(xopt, self.xnew)[:, 0, 0][m]

            dcov_opt_dx = self.model.posterior_covariance_gradient(self.xnew, xopt)[:, 0, :][m]
            b = np.sqrt(np.dot(self.aux[m], np.square(cov_opt)))
            term1 = np.multiply(self.varX[m] * cov_opt, dcov_opt_dx.T)
            term2 = np.multiply(np.square(cov_opt), self.dvar_dX[m].T)
            term3 = (2 * term1 - term2).T

            term4 = np.array(self.aux2[m]) * term3
            term5 = 0.5 * self.Z * np.reciprocal(b)
            out = np.array(term5 * term4).reshape(-1)
            return out.reshape(1, -1)

    def compute_posterior_var_x_new(self, x, m=None):

        if m is None:
            cov = self.cov[m]*1 #self.model.posterior_covariance_between_points_partially_precomputed(x, self.x_new)[:, :, 0]
            kernel_current = self.model.posterior_variance(x, noise=False)
            for j in range(cov.shape[0]):
                b = np.sqrt(self.aux[j] * np.square(cov[j]))
                b = b[np.newaxis]
                kernel_new_x = kernel_current[j] - np.dot(np.transpose(b), b)
                func_val = kernel_new_x
            return np.array(func_val).reshape(-1)
        else:
            cov = self.cov[m]*1 #self.model.posterior_covariance_between_points_partially_precomputed(x, self.x_new)[:, :, 0]
            kernel_current = self.model.output[m].posterior_variance(x, noise=False)
            aux = self.aux[m]

            b = np.sqrt(aux * np.square(cov))

            kernel_new_x = kernel_current.reshape(-1) - (b**2).reshape(-1)
            func_val = kernel_new_x
            return np.array(func_val).reshape(-1)

    def compute_posterior_var_xopt(self, x, m=None):
        if m is None:
            cov = self.model.posterior_covariance_between_points(x, self.xnew)[:, :, 0]
            # print("compute_posterior_var_x_new", cov, self.cov)
            kernel_current = self.model.posterior_variance(x, noise=False)
            func_val = []

            for j in range(cov.shape[0]):
                b = np.sqrt(self.aux[j] * np.square(cov[j]))

                b = b[np.newaxis]
                kernel_new_x = kernel_current - np.dot(np.transpose(b),b)
                func_val.append(kernel_new_x)
            return np.array(func_val).reshape(-1)
        else:
            cov = self.model.posterior_covariance_between_points(x, self.xnew)[:, :, 0][m]
            # print("compute_posterior_var_x_new", cov, self.cov)
            kernel_current = self.model.posterior_variance(x, noise=False)[m]

            b = np.sqrt(self.aux[m] * np.square(cov))

            kernel_new_x = kernel_current - (b**2).reshape(-1)
            func_val = kernel_new_x
            return np.array(func_val).reshape(-1)

    def compute_gradient_posterior_var_x_new(self, x, m=None):
        if m is None:
            K_x_x = self.model.posterior_variance_gradient(x)
            K_x_xnew = self.cov*1 #self.model.posterior_covariance_between_points_partially_precomputed(x, self.x_new)[:, :, 0]
            grad_K_x_xnew = self.dcov*1 # self.model.posterior_covariance_gradient_partially_precomputed(x, self.x_new)
            out = K_x_x  - 2* K_x_xnew * grad_K_x_xnew * self.aux
            return out
        else:
            K_x_x = self.model.output[m].posterior_variance_gradient(x)
            K_x_xnew = self.cov[m]*1 #self.model.posterior_covariance_between_points_partially_precomputed(x, self.x_new)[:, :, 0]
            grad_K_x_xnew = self.dcov[m]*1 # self.model.posterior_covariance_gradient_partially_precomputed(x, self.x_new)
            out = K_x_x  - 2* K_x_xnew * grad_K_x_xnew * self.aux[m]
            return out

    def compute_gradient_posterior_std_x_new(self,x, m=None):

        if m is None:
            posterior_var_xnew = self.compute_posterior_var_x_new(x)
            gradient_posterior_var_x_new = self.compute_gradient_posterior_var_x_new(x)
            gradient_posterior_std_x_new = (1.0/2) * (posterior_var_xnew) **(- 1.0/2) * gradient_posterior_var_x_new
            gradient_posterior_std_x_new = gradient_posterior_std_x_new.reshape(-1)
            return gradient_posterior_std_x_new.reshape(1 , -1)
        else:
            posterior_var_xnew = self.compute_posterior_var_x_new(x, m=m)
            gradient_posterior_var_x_new = self.compute_gradient_posterior_var_x_new(x, m=m)
            gradient_posterior_std_x_new = (1.0/2) * (posterior_var_xnew) **(- 1.0/2) * gradient_posterior_var_x_new
            gradient_posterior_std_x_new = gradient_posterior_std_x_new.reshape(-1)
            return gradient_posterior_std_x_new.reshape(1 , -1)

    def compute_probability_feasibility_multi_gp_xopt(self, xopt, l=0, gradient_flag = False):

        if gradient_flag:
            Fz = []
            grad_Fz = []
            l = 0

            for m in range(self.model.output_dim):

                mean = self.compute_value_mu_xopt(xopt=xopt, m=m)
                cov = self.compute_posterior_var_xopt(x=xopt, m=m)
                Fz_aux = self.compute_probability_feasibility(xopt, self.model.output[m], mean,  cov, l)
                grad_Fz_aux = self.compute_gradient_probability_feasibility_xopt(xopt, self.model.output[m], mean, cov, m=m)

                grad_Fz.append(grad_Fz_aux)
                Fz.append(Fz_aux)


            if len(np.array(Fz).reshape(-1)) == 1:
                return Fz, np.array(grad_Fz).reshape(-1)
            else:
                Fz_aux = np.product(Fz, axis=0)
                Fz = np.array(Fz).reshape(-1)
                grad_Fz = np.vstack(grad_Fz)

                grad_Fz = self.product_gradient_rule(func=np.array(Fz).reshape(-1), grad=np.array(grad_Fz))
                grad_Fz = np.array(grad_Fz).reshape(-1)

                # print("grad_Fz",grad_Fz)
                return Fz_aux, grad_Fz.reshape(1, -1)
        else:
            Fz = []
            for m in range(self.model.output_dim):
                mean = self.compute_value_mu_xopt(xopt=xopt, m=m)
                cov = self.compute_posterior_var_xopt(x=xopt, m=m)

                Fz.append(self.compute_probability_feasibility( xopt, self.model.output[m], mean,  cov, l))
            Fz = np.product(Fz,axis=0)
            Fz = np.array(Fz).reshape(-1)
            return Fz

    def compute_probability_feasibility_multi_gp(self, x, l=0, gradient_flag = False):

        if gradient_flag:
            Fz = []
            grad_Fz = []
            for m in range(self.model.output_dim):
                mean = self.compute_value_mu_xnew(x=x, m=m)
                cov = self.compute_posterior_var_x_new(x=x, m=m)
                Fz.append(self.compute_probability_feasibility( x, self.model.output[m], mean,  cov, l))
                grad_Fz_aux = self.compute_gradient_probability_feasibility(x, self.model.output[m], mean, cov, m=m)
                grad_Fz.append(grad_Fz_aux)

            if len(np.array(Fz).reshape(-1)) == 1:
                Fz = np.array(Fz).reshape(-1)
                return Fz.reshape(-1,1), np.array(grad_Fz).reshape(-1)
            else:
                Fz_aux = np.product(Fz, axis=0)
                Fz = np.array(Fz).reshape(-1)
                grad_Fz = np.vstack(grad_Fz)
                grad_Fz = self.product_gradient_rule(func=np.array(Fz).reshape(-1), grad=np.array(grad_Fz))
                grad_Fz = np.array(grad_Fz).reshape(-1)

                return Fz_aux.reshape(-1,1),  grad_Fz.reshape(1, -1)
        else:
            Fz = []
            for m in range(self.model.output_dim):
                mean = self.compute_value_mu_xnew(x=x, m=m)
                cov = self.compute_posterior_var_x_new(x=x, m=m)

                Fz.append(self.compute_probability_feasibility( x, self.model.output[m], mean,  cov, l))

            Fz = np.product(Fz,axis=0)
            Fz = np.array(Fz).reshape(-1)

            return Fz.reshape(-1,1)

    def compute_grad_probability_feasibility_multi_gp(self, x, l=0):

        Fz = []
        grad_Fz = []
        for m in range(self.model.output_dim):
            mean = self.compute_value_mu_xnew(x=x, m=m)
            cov = self.compute_posterior_var_x_new(x=x, m=m)
            Fz_aux = self.compute_probability_feasibility( x, self.model.output[m], mean, cov, l)
            # print("Fz_aux",Fz_aux)
            grad_Fz_aux = self.compute_gradient_probability_feasibility( x, self.model.output[m], mean, cov,m)
            Fz.append(Fz_aux)
            grad_Fz.append(grad_Fz_aux)

        if len(np.array(Fz).reshape(-1))==1:
            return np.array(grad_Fz).reshape(-1)
        else:
            grad_Fz = self.product_gradient_rule(func = np.array(Fz).reshape(-1), grad = np.array(grad_Fz).reshape(-1))
            grad_Fz = np.array(grad_Fz).reshape(-1)
            # print("grad_Fz",grad_Fz)
            return grad_Fz.reshape(1,-1)

    def compute_gradient_func(self, func, grad):
        return self.product_gradient_rule( func, grad)

    def compute_gradient_probability_feasibility_xopt(self, xopt , model, mean=None, var=None, m=None):

        std = np.sqrt(var).reshape(-1, 1)
        aux_var = np.reciprocal(var)
        mean = mean.reshape(-1, 1)
        fz = scipy.stats.norm.pdf(mean/std)


        if m is None:
            grad_mean = self.compute_grad_mu_xopt(xopt)
            grad_std = self.compute_grad_sigma_xopt(xopt)
        else:
            grad_mean = self.compute_grad_mu_xopt(xopt, m=m)
            grad_std = self.compute_grad_sigma_xopt(xopt ,m=m)

        dims = range(xopt.shape[1])
        grad_Fz = []
        for d in dims:
            grd_mean_d = grad_mean[:, d].reshape(-1, 1)
            grd_std_d = grad_std[:, d].reshape(-1, 1)
            grad_func_val = -fz * aux_var * (grd_mean_d * std - mean * grd_std_d )
            grad_Fz.append(grad_func_val)
        grad_Fz = np.array(grad_Fz).reshape(-1)
        return grad_Fz.reshape(1,-1)

    def compute_gradient_probability_feasibility(self, x, model, mean=None, var=None, m=None):
        model = model.model

        if (mean is None) or (var is None):
            mean = model.posterior_mean(x)
            var = model.posterior_variance(x, noise=False)
        std = np.sqrt(var).reshape(-1, 1)

        aux_var = np.reciprocal(var)
        mean = mean.reshape(-1, 1)

        fz = scipy.stats.norm.pdf(mean/std)
        if m is None:
            grad_mean = self.compute_gradient_mu_xnew(x)
            grad_std = self.compute_gradient_posterior_std_x_new(x)
        else:
            grad_mean = self.compute_gradient_mu_xnew(x,m=m)
            grad_std = self.compute_gradient_posterior_std_x_new(x,m=m)
        dims = range(x.shape[1])
        grad_Fz = []

        for d in dims:
            grd_mean_d = grad_mean[:, d].reshape(-1, 1)
            grd_std_d = grad_std[:, d].reshape(-1, 1)
            grad_func_val = -fz * aux_var * (grd_mean_d * std - mean * grd_std_d )
            grad_Fz.append(grad_func_val)
        grad_Fz = np.array(grad_Fz).reshape(-1)
        return grad_Fz.reshape(1,-1)

    def compute_probability_feasibility(self, x, model, mean=None, cov=None, l=0):

        model = model.model
        # if (mean is None) and (cov is None):
        #
        #     mean = model.posterior_mean(x)
        #     cov = model.posterior_variance(x, noise=False)

        std = np.sqrt(cov).reshape(-1, 1)

        mean = mean.reshape(-1, 1)
        norm = scipy.stats.norm(mean, std)
        Fz = norm.cdf(l)
        return Fz.reshape(-1, 1)

    def product_gradient_rule(self, func, grad):


        func = func + np.random.normal(0,1e-10)
        recip = np.reciprocal(func)
        prod = np.product(func)

        vect1 = prod * recip
        vect1 = vect1.reshape(-1)
        vect1 = vect1.reshape(1,-1)


        vect2 = grad

        grad = np.dot(vect1, vect2)

        return grad

    def posterior_std(self,x):
        cov = []
        x = x.reshape(1, -1)
        cov_val = np.array(self.compute_posterior_var_x_new( x=x )).reshape(-1)
        cov.append(np.sqrt(cov_val))
        out = np.array(cov).reshape(-1)
        return out

    def compute_b(self,xopt, x_new = None, m=None):
        # print("computing b")
        # print("self.xnew ", self.xnew, "self.varX", self.varX, "self.dvar_dX ", self.dvar_dX, "self.aux",
        #       self.aux,
        #       "self.aux2", self.aux2, "xopt", xopt, "cov_opt", self.model.posterior_covariance_between_points(xopt, self.xnew)[:, 0, 0][m])

        if m is None:

            if x_new is None:
                cov_opt = self.model.posterior_covariance_between_points(xopt, self.xnew)[:, 0, 0]


                b = np.matmul(np.sqrt(self.aux),cov_opt)

            else:
                cov_opt = self.model.posterior_covariance_between_points(xopt, x_new)[:, 0, 0]


                b = np.matmul(np.sqrt(self.aux),cov_opt)



            return np.array(b).reshape(-1)
        else:
            if x_new is None:
                cov_opt = self.model.posterior_covariance_between_points(xopt, self.xnew)[:, 0, 0][m]
                cov_opt = np.array(cov_opt).reshape(-1)
                aux = np.array([self.aux[m]])

                b = np.matmul(np.sqrt(aux),cov_opt) #np.sqrt(self.aux[m] *np.square(cov_opt))

            else:
                cov_opt = self.model.posterior_covariance_between_points(xopt, x_new)[:, 0, 0][m]
                cov_opt = np.array(cov_opt).reshape(-1)
                aux = np.array([self.aux[m]])
                b = np.matmul(np.sqrt(aux),cov_opt)

            return np.array(b).reshape(-1)

    def compute_grad_b(self, xopt, m=None):

        if m is None:
            cov_opt = self.model.posterior_covariance_between_points( xopt, self.xnew)[:, 0, 0]
            dcov_opt_dx = self.model.posterior_covariance_gradient(self.xnew, xopt)[:, 0, :]

            # Analitical gradient
            term1 = np.sqrt(self.aux) * dcov_opt_dx
            #term1 = dcov_opt_dx

            self.aux_cov_grad = dcov_opt_dx * 1

            self.aux_var_grad =  -(1.0/2) * self.varX**(-3.0/2.0) * self.dvar_dX *1

            term2 = -(1.0/2) * self.varX**(-3.0/2.0) * self.dvar_dX * cov_opt
            grad_b = term1.reshape(-1) +term2.reshape(-1) #

            ##fuck this shit. Estimated, at least for now
            # delta = 1e-4
            # dim = xopt.shape[1]
            # delta_matrix = np.identity(dim)
            # x = self.xnew*1
            # x = x.reshape(1, -1)
            # f_delta = []
            # for i in range(dim):
            #     one_side = np.array(self.trial_compute_b_xopt(x + delta_matrix[i] * delta)).reshape(-1)
            #     two_side = np.array(self.trial_compute_b_xopt(x - delta_matrix[i] * delta)).reshape(-1)
            #     f_delta.append(one_side - two_side)
            #
            # self.xnew = x* 1
            # f_delta = np.array(f_delta).reshape(-1)
            # grad_b = np.array(f_delta / (2 * delta)).reshape(-1)

            return np.array(grad_b).reshape(-1)
        else:
            cov_opt = self.model.posterior_covariance_between_points(xopt, self.xnew)[:, 0, 0][m]
            dcov_opt_dx = self.model.posterior_covariance_gradient(self.xnew, xopt)[:, 0, :][m]


            #Analitical gradient
            term1 = np.sqrt(self.aux[m]) * dcov_opt_dx
            self.aux_cov_grad = dcov_opt_dx * 1

            self.aux_cov_opt_2 = cov_opt * 1
            term2 = -(1.0 / 2) * (self.varX[m])** (-3.0 / 2.0) * self.dvar_dX[m] * cov_opt
            grad_b = term1.reshape(-1) + term2.reshape(-1)

            # delta = 1e-4
            # dim = xopt.shape[1]
            # delta_matrix = np.identity(dim)
            # x = self.xnew*1
            # x = x.reshape(1, -1)
            # f_delta = []
            # for i in range(dim):
            #     one_side = np.array(self.trial_compute_b_xopt(x + delta_matrix[i] * delta, m=m)).reshape(-1)
            #     two_side = np.array(self.trial_compute_b_xopt(x - delta_matrix[i] * delta, m=m)).reshape(-1)
            #     f_delta.append(one_side - two_side)
            #
            # self.xnew = x* 1
            # f_delta = np.array(f_delta).reshape(-1)
            # grad_b = np.array(f_delta / (2 * delta)).reshape(-1)
            return np.array(grad_b).reshape(-1)

    def compute_grad_sigma_xopt(self, xopt, m=None):
        if m is None:
            b = self.compute_b(xopt)
            grad_b = self.compute_grad_b(xopt)
            grad_posterior_var_xopt = -2 * b * grad_b

            posterior_var_xopt = self.compute_posterior_var_xopt(xopt)
            gradient_posterior_std_xopt = (1.0/2) * np.sqrt(np.reciprocal(posterior_var_xopt))  * np.array(grad_posterior_var_xopt).reshape(-1)
            gradient_posterior_std_xopt = gradient_posterior_std_xopt.reshape(-1)
            return gradient_posterior_std_xopt.reshape(1 , -1) #grad_posterior_var_xopt.reshape(1 , -1) #
        else:
            b = self.compute_b(xopt, m=m )
            grad_b = self.compute_grad_b(xopt, m=m )
            grad_posterior_var_xopt = -2 * b * grad_b

            posterior_var_xopt = self.compute_posterior_var_xopt(xopt, m=m)

            gradient_posterior_std_xopt = (1.0 / 2) * np.sqrt(np.reciprocal(posterior_var_xopt)) * np.array(
                grad_posterior_var_xopt).reshape(-1)
            gradient_posterior_std_xopt = gradient_posterior_std_xopt.reshape(-1)
            return gradient_posterior_std_xopt.reshape(1, -1)  # grad_posterior_var_xopt.reshape(1 , -1) #

    def test_mode(self):
        print("TEST MODE ON")
        f_1 = [self.compute_value_mu_xnew, self.compute_gradient_mu_xnew] #works 1D & 2D
        f_2 = [self.compute_probability_feasibility_multi_gp, self.compute_grad_probability_feasibility_multi_gp] #works 1D
        f_3 = [self.compute_posterior_var_x_new, self.compute_gradient_posterior_var_x_new] #works 1D &2D
        f_5 = [self.posterior_std, self.compute_gradient_posterior_std_x_new] #works 1D

        f_7 = [self.trial_compute_sigma_xopt,  self.trial_compute_grad_sigma_xopt]
        f_8 = [self.trial_compute_b_xopt ,self.trial_compute_grad_b_xopt]
        f_9 = [self.trial_compute_probability_feasibility_multi_gp_xopt, self.trial_compute_grad_probability_feasibility_multi_gp_xopt]
        f_10 = [self.trial_compute_KG_xopt, self.trial_compute_grad_KG_xopt]
        f_11 = [self.trial_compute_mu_xopt, self.trial_compute_grad_mu_xopt]
        f = [ f_8]
        #self.future_mean_covariance()
        for index in range(len(f)):
            #self.gradient_cov_check()
            self.gradient_sanity_check_2D(f[index][0], f[index][1])
            #self.gradient_sanity_check_1D( f[index][0], f[index][1])


    def future_mean_covariance(self):
        X = self.test_samples
        mean = []
        cov = []
        pf = []
        for x in X:
            x = x.reshape(1, -1)
            mean_val = np.array(self.compute_value_mu_xnew(x=x)).reshape(-1)
            pf.append(np.array(self.compute_probability_feasibility_multi_gp(x=x)).reshape(-1))
            cov_val = np.array(self.compute_posterior_var_x_new( x=x )).reshape(-1)
            mean.append(mean_val)
            cov.append(cov_val)

        self.pf = np.array(pf).reshape(-1)
        self.mean_plot= np.array(mean).reshape(-1)
        self.var_plot = np.array(cov).reshape(-1)

        # print("x_new",self.x_new)
        # print("min max", np.min(mean), np.max(mean))
        # print("np.array(X[:,0])",np.array(X[:,0]).shape)
        # print("np.array(X[:,1])", np.array(X[:,1]).shape)

        # plt.scatter(np.array(X[:,0]).reshape(-1),np.array(X[:,1]).reshape(-1), c = self.mean_plot)
        # plt.scatter(X, mean + 1.95*std)
        # plt.scatter(X, mean - 1.95 * std)
        # plt.title("future gp change")
        # plt.show()
        # plt.scatter(np.array(X[:,0]).reshape(-1),np.array(X[:,1]).reshape(-1), c = self.pf)
        # plt.show()

    def gradient_cov_check(self, delta=1e-4):
        initial_design = self.test_samples
        print("initial_design",initial_design)
        fixed_dim = 0
        variable_dim = 1
        v1 = np.repeat(np.array(initial_design[0, fixed_dim]), len(initial_design[:, fixed_dim])).reshape(-1, 1)
        v2 = initial_design[:, variable_dim ].reshape(-1, 1)
        X = np.concatenate((v1, v2), axis=1)
        print("X",X)
        f = self.model.posterior_covariance_between_points
        grad_f = self.model.posterior_covariance_gradient

        numerical_grad = []
        analytical_grad = []
        func_val = []
        dim = X.shape[1]
        delta_matrix = np.identity(dim)
        for x in X:
            x = x.reshape(1, -1)

            f_val = np.array(f(self.xopt, x)[:, 0, 0]).reshape(-1)
            f_delta = []
            for i in range(dim):
                one_side = np.array(f(self.xopt, x + delta_matrix[i] * delta)[:, 0, 0]).reshape(-1)
                two_side = np.array(f(self.xopt, x - delta_matrix[i] * delta)[:, 0, 0]).reshape(-1)
                f_delta.append(one_side - two_side)

            func_val.append(f_val)
            f_delta = np.array(f_delta).reshape(-1)
            numerical_grad.append(np.array(f_delta / (2 * delta)).reshape(-1))

#            print("FD", np.array(f_delta / (2 * delta)).reshape(-1), "analytical", grad_f(x).reshape(-1))
            analytical_grad.append(grad_f(x , self.xopt)[:, 0, :].reshape(-1))


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
    def gradient_sanity_check_2D(self,  f, grad_f, delta= 1e-7):
        initial_design = self.test_samples
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

            print("FD",np.array(f_delta/(2*delta)) , "analytical", grad_f(x))

            analytical_grad.append(grad_f(x).reshape(-1))

        func_val = np.array(func_val)
        numerical_grad = np.array(numerical_grad)
        analytical_grad = np.array(analytical_grad)

        dif = np.abs(numerical_grad - analytical_grad)
        print("dif mean", np.mean(dif,axis=0), "dif min", np.min(dif,axis=0), "dif max", np.max(dif,axis=0))

        #PLOTS
        fig, (ax1, ax2, ax3,ax4) = plt.subplots(4)

        print("v2", v2)
        print("np.array(func_val).reshape(-1)", np.array(func_val).reshape(-1))
        ax1.scatter(v2.reshape(-1), np.array(func_val).reshape(-1), label="actual function")
        ax1.legend()
        ax2.scatter(v2.reshape(-1),np.array(numerical_grad[:,variable_dim]).reshape(-1), label="numerical")
        ax2.legend()
        ax3.scatter(v2.reshape(-1),np.array(analytical_grad[:,variable_dim]).reshape(-1), label="analytical")
        ax3.legend()
        ax4.scatter(v2.reshape(-1), dif[:,variable_dim].reshape(-1), label="errors")
        ax4.legend()

        plt.show()

    def gradient_sanity_check_1D(self, f, grad_f, delta= 1e-4):
        X = self.test_samples
        numerical_grad = []
        analytical_grad = []
        func_val = []

        for x in X:
            x = x.reshape(1, -1)

            # print("self.xnew", self.xnew)
            # print("x+delta",x+delta)
            # print("x-delta",x-delta)
            f_delta = np.array(f(x+delta)).reshape(-1)
            f_val = np.array(f(x-delta)).reshape(-1)


            func_val.append(f_val)
            numerical_grad.append((f_delta-f_val)/(2*delta))
            analytical_grad.append(grad_f(x))
            print("numerical",(f_delta-f_val)/(2*delta), "analytical", grad_f(x))
            # print("current mean", self.current_mean, "current var", self.current_var)

        func_val = np.array(func_val).reshape(-1)
        numerical_grad = np.array(numerical_grad).reshape(-1)
        analytical_grad = np.array(analytical_grad ).reshape(-1)

        dif = np.abs(numerical_grad - analytical_grad)
        print("dif mean", np.mean(dif), "dif min", np.min(dif), "dif max", np.max(dif))

        #PLOTS
        fig, (ax1, ax2, ax3,ax4, ax5, ax6) = plt.subplots(6)


        ax3.scatter(X, func_val, label="actual function")
        ax3.legend()
        ax4.scatter(X,numerical_grad, label="numerical")
        ax4.legend()
        ax5.scatter(X,analytical_grad, label="analytical")
        ax5.legend()

        ax6.scatter(X, dif.reshape(-1), label="errors")
        ax6.legend()
        plt.show()

    def trial_compute_grad_mu_xopt(self, xnew):
        self.xnew = xnew
        self.model.partial_precomputation_for_covariance(xnew)
        self.model.partial_precomputation_for_covariance_gradient(xnew)
        self.varX= self.model.posterior_variance(xnew, noise=False)[:, 0]
        self.dvar_dX = self.model.posterior_variance_gradient(xnew)[:,0,:]
        self.aux = np.reciprocal(self.varX)
        self.aux2 = np.square(np.reciprocal(self.varX))

        grad_mu_xopt = self.compute_grad_mu_xopt(self.xopt,m=0)

        return np.array(grad_mu_xopt).reshape(-1)

    def trial_compute_KG_xopt(self, xnew):
        self.xnew = xnew
        self.model.partial_precomputation_for_covariance(xnew)
        self.model.partial_precomputation_for_covariance_gradient(xnew)
        self.varX= self.model.posterior_variance(xnew)[:, 0]
        self.dvar_dX = self.model.posterior_variance_gradient(xnew)[:,0,:]
        self.aux = np.reciprocal(self.varX)
        self.aux2 = np.square(np.reciprocal(self.varX))

        mu_xopt = self.compute_value_mu_xopt(x=self.xopt)
        Fz_xopt = self.compute_probability_feasibility_multi_gp_xopt(x=self.xopt)

        # print("mu_xopt",mu_xopt,"Fz_xopt",Fz_xopt)
        return np.array(mu_xopt * Fz_xopt).reshape(-1)

    def trial_compute_grad_KG_xopt(self, xnew):
        self.xnew = xnew
        self.model.partial_precomputation_for_covariance(xnew)
        self.model.partial_precomputation_for_covariance_gradient(xnew)
        self.varX= self.model.posterior_variance(xnew)[:, 0]
        self.dvar_dX = self.model.posterior_variance_gradient(xnew)[:,0,:]
        self.aux = np.reciprocal(self.varX)
        self.aux2 = np.square(np.reciprocal(self.varX))

        mu_xopt = self.compute_value_mu_xopt(x=self.xopt)
        grad_mu_xopt = self.compute_grad_mu_xopt(self.xopt)

        Fz_xopt = self.compute_probability_feasibility_multi_gp_xopt(x=self.xopt)
        grad_Fz_xopt = self.compute_grad_probability_feasibility_multi_gp_xopt(xopt=self.xopt)
        grad_func = self.product_gradient_rule(func=np.array([np.array(mu_xopt).reshape(-1), Fz_xopt.reshape(-1)]),
                                     grad=np.array(
                                         [grad_mu_xopt.reshape(-1), grad_Fz_xopt.reshape(-1)]))
        return np.array(grad_func).reshape(-1)

    def trial_compute_mu_xopt(self, xnew):
        self.xnew = xnew
        self.varX= self.model.posterior_variance(xnew, noise=False)[:, 0]
        self.dvar_dX = self.model.posterior_variance_gradient(xnew)[:,0,:]
        self.aux = np.reciprocal(self.varX)
        self.aux2 = np.square(np.reciprocal(self.varX))

        return self.compute_value_mu_xopt(self.xopt, m=0)

    def trial_compute_var_xopt(self, xnew):
        self.xnew = xnew
        self.varX= self.model.posterior_variance(xnew)[:, 0]
        self.dvar_dX = self.model.posterior_variance_gradient(xnew)[:,0,:]
        self.aux = np.reciprocal(self.varX)
        self.aux2 = np.square(np.reciprocal(self.varX))
        return self.compute_posterior_var_x_new(self.xopt)

    def trial_compute_grad_b_xopt(self, xnew):

        self.xnew = xnew*1
        self.varX= self.model.posterior_variance(xnew, noise=True)[:, 0]*1
        self.dvar_dX = self.model.posterior_variance_gradient(xnew)[:,0,:]*1
        self.aux = np.reciprocal(self.varX)*1
        self.aux2 = np.square(np.reciprocal(self.varX))*1

        return self.compute_grad_b(xopt = self.xopt)

    def trial_compute_b_xopt(self, xnew, m=None):

        self.xnew = xnew*1
        self.varX= self.model.posterior_variance(xnew, noise=True)[:, 0]*1
        self.dvar_dX = self.model.posterior_variance_gradient(xnew)[:,0,:]*1
        self.aux = np.reciprocal(self.varX)*1
        self.aux2 = np.square(np.reciprocal(self.varX))*1

        return self.compute_b(xopt = self.xopt, m=m)

    def trial_compute_grad_sigma_xopt(self, xnew):

        self.xnew = xnew*1
        self.varX= self.model.posterior_variance(xnew, noise=True)[:, 0]*1
        self.dvar_dX = self.model.posterior_variance_gradient(xnew)[:,0,:]*1
        self.aux = np.reciprocal(self.varX)*1
        self.aux2 = np.square(np.reciprocal(self.varX))*1
        self.cov = self.model.posterior_covariance_between_points(self.xopt, xnew)[:, :, 0] * 1

        return self.compute_grad_sigma_xopt(xopt = self.xopt,m=0)

    def trial_compute_sigma_xopt(self, xnew):
        self.xnew = xnew*1
        self.varX= self.model.posterior_variance(xnew, noise=True)[:, 0]*1
        self.dvar_dX = self.model.posterior_variance_gradient(xnew)[:,0,:]*1
        self.aux = np.reciprocal(self.varX)
        self.aux2 = np.square(np.reciprocal(self.varX))
        self.cov = self.model.posterior_covariance_between_points(self.xopt, xnew)[:, :, 0] * 1

        return np.sqrt(self.compute_posterior_var_xopt(self.xopt,m=0))

    def trial_compute_grad_probability_feasibility_multi_gp_xopt(self, xnew):
        # print("-------------------------------------trial_compute_grad_probability_feasibility_multi_gp_xopt------------------------")
        self.xnew = xnew
        self.model.partial_precomputation_for_covariance(xnew)
        self.model.partial_precomputation_for_covariance_gradient(xnew)
        self.varX= self.model.posterior_variance(xnew, noise=True)[:, 0]
        self.dvar_dX = self.model.posterior_variance_gradient(xnew)[:,0,:]
        self.aux = np.reciprocal(self.varX)
        self.aux2 = np.square(np.reciprocal(self.varX))
        # print("xnew", self.xnew, "self.varX", self.varX, "dvar_dX", self.dvar_dX, "aux", self.aux, "aux2", self.aux2,self.xopt)
        # print("--------------------------------------------------------------------------------------------------------------------")
        return self.compute_probability_feasibility_multi_gp_xopt_check(self.xopt, gradient_flag = True)

    def compute_probability_feasibility_multi_gp_xopt_check(self, xopt, l=0, gradient_flag = False):

        if gradient_flag:
            Fz = []
            grad_Fz = []
            l = 0

            for m in range(self.model.output_dim):

                mean = self.compute_value_mu_xopt(xopt=xopt, m=m)
                cov = self.compute_posterior_var_xopt(x=xopt, m=m)
                Fz_aux = self.compute_probability_feasibility(xopt, self.model.output[m], mean,  cov, l)
                grad_Fz_aux = self.compute_gradient_probability_feasibility_xopt(xopt, self.model.output[m], mean, cov, m=m)

                grad_Fz.append(grad_Fz_aux)
                Fz.append(Fz_aux)


            if len(np.array(Fz).reshape(-1)) == 1:
                return  np.array(grad_Fz).reshape(-1)
            else:
                Fz_aux = np.product(Fz, axis=0)
                Fz = np.array(Fz).reshape(-1)
                grad_Fz = np.vstack(grad_Fz)

                grad_Fz = self.product_gradient_rule(func=np.array(Fz).reshape(-1), grad=np.array(grad_Fz))
                grad_Fz = np.array(grad_Fz).reshape(-1)

                # print("grad_Fz",grad_Fz)
                return grad_Fz.reshape(1, -1)
        else:
            Fz = []
            for m in range(self.model.output_dim):
                mean = self.compute_value_mu_xopt(xopt=xopt, m=m)
                cov = self.compute_posterior_var_xopt(x=xopt, m=m)

                Fz.append(self.compute_probability_feasibility( xopt, self.model.output[m], mean,  cov, l))
            Fz = np.product(Fz,axis=0)
            Fz = np.array(Fz).reshape(-1)
            return Fz

    def trial_compute_probability_feasibility_multi_gp_xopt(self, xnew):
        self.xnew = xnew
        self.model.partial_precomputation_for_covariance(xnew)
        self.model.partial_precomputation_for_covariance_gradient(xnew)
        self.varX= self.model.posterior_variance(xnew, noise=True)[:, 0]
        self.dvar_dX = self.model.posterior_variance_gradient(xnew)[:,0,:]
        self.aux = np.reciprocal(self.varX)
        self.aux2 = np.square(np.reciprocal(self.varX))

        return self.compute_probability_feasibility_multi_gp_xopt_check(self.xopt, gradient_flag = False)

