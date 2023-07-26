# Copyright (c) 2018, Raul Astudillo Marban

import numpy as np
from GPyOpt.experiment_design import initial_design
from GPyOpt.acquisitions.base import AcquisitionBase
from GPyOpt.core.task.cost import constant_cost_withGradients
from GPyOpt.experiment_design import initial_design
from aux_modules.gradient_modules import gradients
from scipy.stats import norm
import scipy
import time
import matplotlib.pyplot as plt
# from pathos.multiprocessing import ProcessingPool as Pool
# from multiprocessing import Pool
from itertools import permutations, product
import numpy as np
import itertools
import pylab
from scipy.linalg import lapack


class KG(AcquisitionBase):
    """
    Multi-attribute knowledge gradient acquisition function

    :param model: GPyOpt class of model.
    :param space: GPyOpt class of domain.
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer.
    :param utility: utility function. See utility class for details.
    """

    analytical_gradient_prediction = False

    def __init__(self, model, space, model_c=None, nz=5, optimizer=None, cost_withGradients=None, utility=None,
                 true_func=None, true_const=None, underlying_discretisation=None):
        self.optimizer = optimizer
        self.utility = utility
        self.MCMC = False
        self.base_points_cap_size = nz
        self.counter = 0
        self.current_max_value = np.inf
        self.Z_samples_obj = None
        self.Z_samples_const = None
        self.Z_cdKG = None
        self.true_func = true_func
        self.true_const = true_const
        self.saved_Nx = -10
        self.base_discretisation = None
        self.underlying_discretisation = underlying_discretisation
        self.optimise_discretisation = True
        # self.n_base_points = nz
        self.name = "Constrained_KG"
        self.fixed_discretisation = None
        self._test_important_values_for_estimated_optimum = []
        self._test_Fz_values = []

        super(KG, self).__init__(model=model, space=space, optimizer=optimizer, model_c=model_c,
                                 cost_withGradients=cost_withGradients)
        if cost_withGradients == None:
            self.cost_withGradients = constant_cost_withGradients
        else:
            print('LBC acquisition does now make sense with cost. Cost set to constant.')
            self.cost_withGradients = constant_cost_withGradients

        dimy = 1
        if model_c is None:
            self.dim = dimy
        else:
            dimc = self.model_c.output_dim
            self.dimc = dimc
            self.dim = dimy + dimc

    def _compute_acq(self, X):
        """
        Computes the aquisition function

        :param X: set of points at which the acquisition function is evaluated. Should be a 2d array.
        """
        # print("_compute_acq")

        X = np.atleast_2d(X)

        if self.model_c is None:
            marginal_acqX = self._unconstrained_marginal_acq(X)
        else:
            marginal_acqX = self._marginal_acq(X)
        KG = np.reshape(marginal_acqX, (X.shape[0], 1))

        return KG

    def generate_random_vectors(self, base_c_quantiles=None,
                                optimize_discretization=True,
                                optimize_random_Z=False,
                                fixed_discretisation=None):

        self.update_current_best()
        self.base_marg_points = 40
        self.n_marginalisation_points = np.array([-2.64, -0.67, 0, 0.67, 2.64])

        self.n_base_points = len(self.n_marginalisation_points)
        self.optimise_discretisation = optimize_discretization

        if fixed_discretisation is not None:

            self.fixed_discretisation = True

            sampled_X = self.model.get_X_values()
            # self.update_current_best()

            self.base_discretisation = fixed_discretisation
            extended_fixed_discretisation = np.concatenate((fixed_discretisation, sampled_X))

            self.fixed_discretisation_values = extended_fixed_discretisation
            self.X_Discretisation = extended_fixed_discretisation

        else:
            self.base_discretisation = None
            self.fixed_discretisation = False
            if optimize_discretization == True:
                self.X_Discretisation = None

        if base_c_quantiles is not None:

            clist = base_c_quantiles
            res = list(itertools.product(*clist))
            list(res)
            self.Z_cdKG = np.array(list(res))

            alllist = [self.n_marginalisation_points] + base_c_quantiles
            res = list(itertools.product(*alllist))

            if np.array(list(res)).shape[0] > self.base_points_cap_size:
                subset_pick = np.random.choice(range(np.array(list(res)).shape[0]), self.base_points_cap_size,
                                               replace=False)
                self.Z_obj = np.array(list(res))[subset_pick, :1]  # np.atleast_2d(self.n_marginalisation_points).T #
                self.Z_const = np.array(list(res))[subset_pick, 1:]  # constraint_quantiles #
            else:
                self.Z_obj = np.array(list(res))[:, :1]  # np.atleast_2d(self.n_marginalisation_points).T #
                self.Z_const = np.array(list(res))[:, 1:]  # constraint_quantiles #

    def _marginal_acq(self, X):
        self.update_current_best()

        varX_obj = self.model.posterior_variance(X, noise=True)
        varX_c = self.model_c.posterior_variance(X, noise=True)

        acqX = np.zeros((X.shape[0], 1))

        for i in range(0, len(X)):
            x = np.atleast_2d(X[i])

            # For each x new precompute covariance matrices for
            self.model.partial_precomputation_for_covariance(x)
            self.model.partial_precomputation_for_covariance_gradient(x)

            self.model_c.partial_precomputation_for_covariance(x)
            self.model_c.partial_precomputation_for_covariance_gradient(x)

            # Precompute aux_obj and aux_c for computational efficiency.

            aux_obj = np.reciprocal(varX_obj[:, i])
            aux_c = np.reciprocal(varX_c[:, i])

            quantiles = self.constraint_sensitivity_and_constraint_quantile_generation(aux_c=aux_c,
                                                                                       xnew=x)

            if self.base_discretisation is None:
                self.generate_random_vectors(base_c_quantiles=quantiles,
                                             optimize_discretization=self.optimise_discretisation,
                                             optimize_random_Z=True,
                                             fixed_discretisation=None)
            else:
                self.generate_random_vectors(base_c_quantiles=quantiles,
                                             optimize_discretization=self.optimise_discretisation,
                                             optimize_random_Z=True,
                                             fixed_discretisation=self.base_discretisation)

            # Create discretisation for discrete KG.
            if self.fixed_discretisation is False:
                if self.optimise_discretisation:
                    self.generate_random_vectors(base_c_quantiles=quantiles, optimize_discretization=True,
                                                 optimize_random_Z=True,
                                                 fixed_discretisation=None)

                    self.X_Discretisation = self.Discretisation_X(index=i, X=X, aux_obj=aux_obj, aux_c=aux_c)
                    # print("updated discretisation")
                    # print("quantiles", quantiles)
                    self.optimise_discretisation = False
            # print("X discretisation", self.X_Discretisation)

            kg_val = self.discrete_KG(Xd=self.X_Discretisation,
                                      xnew=x,
                                      Zc=self.Z_cdKG,
                                      aux_obj=aux_obj,
                                      aux_c=aux_c)
            acqX[i, :] = kg_val

        return acqX.reshape(-1)

    def _unconstrained_marginal_acq(self, X):

        varX_obj = self.model.posterior_variance(X, noise=True)

        acqX = np.zeros((X.shape[0], 1))

        for i in range(0, len(X)):
            x = np.atleast_2d(X[i])

            # For each x new precompute covariance matrices for
            self.model.partial_precomputation_for_covariance(x)
            self.model.partial_precomputation_for_covariance_gradient(x)

            aux_obj = np.reciprocal(varX_obj[:, i])
            # Create discretisation for discrete KG.
            # if self.fixed_discretisation is False:

            self.X_Discretisation = self._unconstrained_discretisation_X(index=i, X=X, aux_obj=aux_obj)

            # print("X discretisation", self.X_Discretisation)

            kg_val = self.unconstrained_discrete_KG(Xd=self.X_Discretisation,
                                                    xnew=x,
                                                    Zc=self.Z_cdKG,
                                                    aux_obj=aux_obj)
            acqX[i, :] = kg_val

        return acqX.reshape(-1)

    def unconstrained_discrete_KG(self, Xd, xnew, Zc, aux_obj, grad=False, verbose=False):
        xnew = np.atleast_2d(xnew)
        # Xd = np.concatenate((Xd, self.fixed_discretisation_values))
        Xd = np.concatenate((Xd, xnew))
        # Xd = np.concatenate((Xd, self.current_max_xopt))
        self.grad = grad
        out = []

        MM = self.model.predict(Xd)[0].reshape(-1)  # move up
        SS_Xd = np.array(
            self.model.posterior_covariance_between_points_partially_precomputed(Xd, xnew)[:, :, :]).reshape(-1)
        inv_sd = np.asarray(np.sqrt(aux_obj)).reshape(())

        SS = SS_Xd * inv_sd
        MM = MM.reshape(-1)
        SS = SS.reshape(-1)

        marginal_KG = []

        KG = self._unconstrained_parallel_KG(MM=MM, SS=SS, verbose=verbose)

        KG = np.clip(KG, 0, np.inf)
        marginal_KG.append(KG)
        out.append(marginal_KG)

        KG_value = np.mean(out)
        # gradKG_value = np.mean(gradout, axis=?)

        assert ~np.isnan(KG_value);
        "KG cant be nan"
        return KG_value  # , gradKG_value

    def _unconstrained_parallel_KG(self, MM, SS, Xd=None, verbose=False):
        """
        Calculates the linear epigraph, i.e. the boundary of the set of points
        in 2D lying above a collection of straight lines y=a+bx.
        Parameters
        ----------
        a
            Vector of intercepts describing a set of straight lines
        b
            Vector of slopes describing a set of straight lines
        tol
            Minimum slope (in absolute value) different from zero
        Returns
        -------
        KGCB
            average hieght of the epigraph
        grad_a
            dKGCB/db, vector
        grad_b
            dKGCB/db, vector
        """
        a = MM  # np.array(self.c_MM[:, index]).reshape(-1)
        b = SS  # np.array(self.c_SS[:, index]).reshape(-1)

        assert len(b) > 0, "must provide slopes"
        assert len(a) == len(b), f"#intercepts != #slopes, {len(a)}, {len(b)}"
        # ensure 1D
        a = np.array(a).reshape(-1)
        b = np.array(b).reshape(-1)
        assert len(a) > 0, "must provide slopes"
        assert len(a) == len(b), f"#intercepts != #slopes, {len(a)}, {len(b)}"

        # ensure 1D
        a = np.atleast_1d(a.squeeze())
        b = np.atleast_1d(b.squeeze())  # - np.mean(b)
        max_a_index = np.argmax(a)
        maxa = np.max(a)
        n_elems = len(a)

        if np.all(np.abs(b) < 0.000000001):
            return np.array([0])  # , np.zeros(a.shape), np.zeros(b.shape)

        # order by ascending b and descending a
        order = np.lexsort((-a, b))
        a = a[order]
        b = b[order]

        # exclude duplicated b (or super duper similar b)
        threshold = (b[-1] - b[0]) * 0.00001
        diff_b = b[1:] - b[:-1]
        keep = diff_b > threshold
        keep = np.concatenate([[True], keep])
        keep[np.argmax(a)] = True
        order = order[keep]
        a = a[keep]
        b = b[keep]

        # initialize
        idz = [0]
        i_last = 0
        x = [-np.inf]

        n_lines = len(a)
        # main loop TODO describe logic
        # TODO not pruning properly, e.g. a=[0,1,2], b=[-1,0,1]
        # returns x=[-inf, -1, -1, inf], shouldn't affect kgcb
        while i_last < n_lines - 1:
            i_mask = np.arange(i_last + 1, n_lines)
            x_mask = -(a[i_last] - a[i_mask]) / (b[i_last] - b[i_mask])

            best_pos = np.argmin(x_mask)
            idz.append(i_mask[best_pos])
            x.append(x_mask[best_pos])

            i_last = idz[-1]

        x.append(np.inf)

        x = np.array(x)
        idz = np.array(idz)

        # found the epigraph, now compute the expectation
        a = a[idz]
        b = b[idz]
        order = order[idz]

        pdf = norm.pdf(x)
        cdf = norm.cdf(x)

        KG = np.sum(a * (cdf[1:] - cdf[:-1]) + b * (pdf[:-1] - pdf[1:]))
        KG -= np.max(a)

        z_all = np.linspace(x[1], x[-2], 100)
        for j in range(len(a)):

            print(x[1], x[-2])
            if x[j] < -30:
                z = np.linspace(-3, x[j + 1], 3)
            elif x[j + 1] > 30:
                z = np.linspace(x[j], 3, 3)
            else:
                z = np.linspace(x[j], x[j + 1], 3)
            mu_star = a.reshape(-1)[j] + b.reshape(-1)[j] * z
            mu_star_all = a.reshape(-1)[j] + b.reshape(-1)[j] * z_all
            plt.plot(z_all, mu_star_all,
                     color="grey",
                     linewidth=3,
                     alpha=0.3, zorder=0)
            plt.plot(z, mu_star,
                     color="red",
                     linewidth=3,
                     zorder=-1)

        plt.plot(z_all, mu_star_all,
                 color="grey",
                 linewidth=3,
                 alpha=0.3, zorder=0, label=" $\mu_{i}^{n+1}(x)$")
        plt.plot(z, mu_star,
                 color="red",
                 linewidth=3,
                 zorder=-1, label="$\max_{x}$  $\mu_{i}^{n+1}(x)$")
        plt.xlabel("$Z_{y}$", size=14)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.legend(loc="upper left", fontsize=15)
        plt.xlim(-2.4, 2.4)
        plt.savefig("/home/juan/Documents/repos_data/Constrained-KG/RESULTS/plot_saved_data/plots/Zy_epigraph.jpg")
        plt.show()
        raise
        # if verbose:
        #     print("a_sorted ",a_sorted )
        #     print("b_sorted", b_sorted)
        #
        #     print("current max",np.max(a) )
        #     print("idz", idz)
        #     print("a", a)
        #     print("b", b)
        #     plt.scatter(Xd.reshape(-1), np.array(self.c_MM[:, index]).reshape(-1))
        #     plt.plot(np.linspace(0,5,2), np.repeat(np.max(a), 2), color="red")
        #     plt.show()
        #     # raise
        # if KG < -1e-5:
        #     print("KG cant be negative")
        #     print("np.sum(a * (cdf[1:] - cdf[:-1]) + b * (pdf[:-1] - pdf[1:]))",
        #           np.sum(a * (cdf[1:] - cdf[:-1]) + b * (pdf[:-1] - pdf[1:])))
        #     print("self.bases_value[index]", np.max(a))
        #     print("KG", KG)
        #     raise
        #
        # KG = np.clip(KG, 0, np.inf)

        if np.isnan(KG):
            print("KG", KG)
            print("self.bases_value[index]", max_a_index)
            print("a", a)
            print("b", b)
            raise

        return KG

    def _unconstrained_discretisation_X(self, index, X, aux_obj):
        """

             """
        i = index
        x = np.atleast_2d(X[i])

        statistics_precision = []
        self.Z_obj = np.array([-2.64, -1.96, -0.67, 0, 0.67, 1.96, 2.64])
        X_discretisation = np.zeros((len(self.Z_obj), X.shape[1]))

        # efficiency = 0
        self.new_anchors_flag = True
        complete_function = []
        xnew_vals = []
        for z in range(len(self.Z_obj)):
            Z_obj = self.Z_obj[z]

            # inner function of maKG acquisition function.
            # current_xval, current_max = self._compute_current_max(x, Z_const, aux_c)

            def inner_func(X_inner):
                X_inner = np.atleast_2d(X_inner)
                # X_inner = X_inner.astype("int")
                grad_obj = gradients(x_new=x, model=self.model, Z=Z_obj, aux=aux_obj,
                                     X_inner=X_inner)  # , test_samples = self.test_samples)
                mu_xnew = grad_obj.compute_value_mu_xnew(x=X_inner)

                func_val = mu_xnew
                return -func_val.reshape(-1)  # mu_xnew , Fz

            # inner function of maKG acquisition function with its gradient.
            plot_samples = np.linspace(4.0, 7, 60)[:, np.newaxis]
            complete_function.append(inner_func(plot_samples))
            xnew_vals.append(-inner_func(x))
            inner_opt_x, inner_opt_val = self.optimizer.optimize_inner_func(f=inner_func,
                                                                            f_df=None)

            statistics_precision.append(inner_opt_val)
            X_discretisation[z] = inner_opt_x.reshape(-1)

        plot_samples = np.linspace(4.0, 7, 60)[:, np.newaxis]
        mu = self.model.posterior_mean(plot_samples).reshape(-1)
        var = self.model.posterior_variance(plot_samples, noise=False).reshape(-1)

        print(X_discretisation, -np.array(statistics_precision).reshape(-1))
        plt.plot(plot_samples.reshape(-1), mu, color="grey", linestyle='dashed')
        plt.fill_between(plot_samples.reshape(-1),
                         mu - 3.1 * np.sqrt(var),
                         mu + 3.1 * np.sqrt(var), color="lightcoral", alpha=0.3)
        plt.scatter(X_discretisation, -np.array(statistics_precision).reshape(-1),
                    label="$\max_{x}$  $\mu_{i}^{n+1}(x)$",
                    color="red", s=60, edgecolors='black', zorder=1)
        plt.plot(plot_samples.reshape(-1), -np.array(complete_function).T, color="grey", alpha=0.5)
        plt.scatter(np.repeat(x, len(xnew_vals)), np.array(xnew_vals).reshape(-1), s=60, color="white",
                    edgecolors="black", zorder=1)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(loc="upper left", fontsize=15)
        plt.xlim(4, 7)
        plt.savefig("/home/juan/Documents/repos_data/Constrained-KG/RESULTS/plot_saved_data/plots/max_u.jpg")
        plt.show()

        self.new_anchors_flag = False
        return X_discretisation

    def constraint_sensitivity_and_constraint_quantile_generation(self, aux_c, xnew):

        base_quantiles = [np.array([-3, 3])] * self.model_c.output_dim
        base_quantiles = np.array(base_quantiles).T

        x_inner = np.atleast_2d(self.current_max_xopt)
        xnew = np.atleast_2d(xnew)
        grad_c = gradients(x_new=xnew, model=self.model_c, Z=base_quantiles, aux=aux_c, X_inner=x_inner)

        Fz = grad_c.compute_probability_feasibility_each_gp(x=x_inner)
        Fz = np.squeeze(Fz)
        if len(Fz.shape) == 1:
            Fz = np.atleast_2d(Fz).T

        else:
            Fz = Fz.T

        delta = np.abs(Fz[0, :] - Fz[1, :])

        quantiles = []
        self._test_Fz_values.append(delta)
        for d in delta:
            if d < 1e-4:
                quantiles.append(np.array([0]))
                self._test_important_values_for_estimated_optimum.append(0)
            else:
                quantiles.append(np.array([-2.64, 0, 2.64]))
                self._test_important_values_for_estimated_optimum.append(1)
        return quantiles

    def Discretisation_X(self, index, X, aux_obj, aux_c):
        """

             """
        i = index
        x = np.atleast_2d(X[i])

        statistics_precision = []
        X_discretisation = np.zeros((len(self.Z_obj), X.shape[1]))

        # efficiency = 0
        self.new_anchors_flag = True
        for z in range(len(self.Z_obj)):

            Z_obj = self.Z_obj[z]
            Z_const = self.Z_const[z]

            # inner function of maKG acquisition function.
            # current_xval, current_max = self._compute_current_max(x, Z_const, aux_c)

            def inner_func(X_inner):
                X_inner = np.atleast_2d(X_inner)
                # X_inner = X_inner.astype("int")
                grad_obj = gradients(x_new=x, model=self.model, Z=Z_obj, aux=aux_obj,
                                     X_inner=X_inner)  # , test_samples = self.test_samples)
                mu_xnew = grad_obj.compute_value_mu_xnew(x=X_inner)

                Z_const = self.Z_const[z]
                if len(Z_const.shape) == 1:
                    Z_const = np.atleast_2d(Z_const)

                grad_c = gradients(x_new=x, model=self.model_c, Z=Z_const, aux=aux_c,
                                   X_inner=X_inner)  # , test_samples = initial_design('random', self.space, 1000))

                Fz = grad_c.compute_probability_feasibility_multi_gp(x=X_inner, l=0)

                func_val = mu_xnew * Fz  # - current_max

                return -func_val.reshape(-1)  # mu_xnew , Fz

            # inner function of maKG acquisition function with its gradient.
            random_indexes = np.random.choice(range(len(self.underlying_discretisation)), size=500, replace=False)
            fX_values = inner_func(self.underlying_discretisation[random_indexes])
            inner_opt_x = self.underlying_discretisation[np.argmin(fX_values)]

            statistics_precision.append(np.max(fX_values))
            X_discretisation[z] = inner_opt_x.reshape(-1)

        self.new_anchors_flag = False
        return X_discretisation

    def probability_feasibility_multi_gp(self, x, model, l=0):
        # print("model",model.output)
        x = np.atleast_2d(x)
        Fz = []
        for m in range(model.output_dim):
            Fz.append(self.probability_feasibility(x, model.output[m], l))
        Fz = np.product(Fz, axis=0)
        return Fz

    def probability_feasibility(self, x, model, l=0):

        mean = model.posterior_mean(x)
        var = model.posterior_variance(x, noise=False)
        std = np.sqrt(var).reshape(-1, 1)

        mean = mean.reshape(-1, 1)

        norm_dist = norm(mean, std)
        Fz = norm_dist.cdf(l)

        return Fz.reshape(-1, 1)

    def _compute_acq_withGradients(self, X):
        """
        """
        # print("_compute_acq_withGradients")

        raise
        X = np.atleast_2d(X)

        marginal_acqX, marginal_dacq_dX = self._marginal_acq_with_gradient(X)

        acqX = np.reshape(marginal_acqX, (X.shape[0], 1))
        dacq_dX = np.reshape(marginal_dacq_dX, X.shape)
        KG = acqX
        return np.array(KG).reshape(-1), np.array(dacq_dX).reshape(-1)

    def _marginal_acq_with_gradient(self, X):
        # print("_marginal_acq")
        """

        """
        # Initialise marginal acquisition variables

        # Precompute posterior pariance at vector points X. These are the same through every cKG loop.

        varX_obj = self.model.posterior_variance(X, noise=True)
        varX_c = self.model_c.posterior_variance(X, noise=True)
        acqX = np.zeros((X.shape[0], 1))
        dacq_dX = np.zeros((X.shape[0], X.shape[1], 1))

        for i in range(0, len(X)):
            x = np.atleast_2d(X[i])

            # For each x new precompute covariance matrices for

            self.model.partial_precomputation_for_covariance(x)
            self.model.partial_precomputation_for_covariance_gradient(x)

            self.model_c.partial_precomputation_for_covariance(x)
            self.model_c.partial_precomputation_for_covariance_gradient(x)

            # Precompute aux_obj and aux_c for computational efficiency.

            aux_obj = np.reciprocal(varX_obj[:, i])
            aux_c = np.reciprocal(varX_c[:, i])

            # Create discretisation for discrete KG.
            start = time.time()
            # print("self.fixed_discretisation",self.fixed_discretisation)
            # print("self.optimise_discretisation",self.optimise_discretisation)
            # if test_mode:
            #     self.X_Discretisation = self.Discretisation_X(index=i, X=X, aux_obj=aux_obj, aux_c=aux_c)
            if self.fixed_discretisation is False:
                if self.optimise_discretisation:
                    print("optimise_discretisation", self.optimise_discretisation)
                    self.X_Discretisation = self.Discretisation_X(index=i, X=X, aux_obj=aux_obj, aux_c=aux_c)
                    print("updated discretisation")
                    self.optimise_discretisation = False
            # print("x", x)
            kg_val, kg_grad = self.discrete_KG(Xd=self.X_Discretisation, xnew=x, Zc=self.Z_cdKG, aux_obj=aux_obj,
                                               aux_c=aux_c, grad=True)
            # print("kg",  kg_val)
            acqX[i, :] = kg_val
            dacq_dX[i, :, 0] = kg_grad
        # print("acqX",acqX)

        return acqX.reshape(-1), dacq_dX

    def discrete_KG(self, Xd, xnew, Zc, aux_obj, aux_c, grad=False, verbose=False):
        xnew = np.atleast_2d(xnew)
        # Xd = np.concatenate((Xd, self.fixed_discretisation_values))
        Xd = np.concatenate((Xd, xnew))
        Xd = np.concatenate((Xd, self.current_max_xopt))
        self.grad = grad
        out = []
        grad_c = gradients(x_new=xnew, model=self.model_c, Z=Zc, aux=aux_c,
                           X_inner=Xd)  # , test_samples = initial_design('random', self.space, 1000))
        Fz = grad_c.compute_probability_feasibility_multi_gp(x=Xd, l=0)

        MM = self.model.predict(Xd)[0].reshape(-1)  # move up
        SS_Xd = np.array(
            self.model.posterior_covariance_between_points_partially_precomputed(Xd, xnew)[:, :, :]).reshape(-1)
        inv_sd = np.asarray(np.sqrt(aux_obj)).reshape(())

        SS = SS_Xd * inv_sd
        MM = MM.reshape(-1)
        SS = SS.reshape(-1)

        grad_c = gradients(x_new=xnew, model=self.model_c, Z=Zc, aux=aux_c,
                           X_inner=self.current_max_xopt)
        Fz_current = grad_c.compute_probability_feasibility_multi_gp(x=self.current_max_xopt, l=0).reshape(-1)
        MM_current = self.model.predict(self.current_max_xopt)[0].reshape(-1)  # MM[-1]

        # print("Fz_current ",Fz_current.shape, "MM_current", MM_current.shape)
        # print("Fz_current ", Fz_current, "MM_current", MM_current)
        marginal_KG = []

        for zc in range(Zc.shape[0]):
            VoI_future = self.parallel_KG(MM=MM * np.array(Fz[:, zc]).reshape(-1),
                                          SS=SS * np.array(Fz[:, zc]).reshape(-1), verbose=verbose)
            VoI_current = MM_current * np.array(Fz_current[zc]).reshape(-1)
            KG = VoI_future - VoI_current

            try:
                if KG < -1e-5:
                    print("max future", np.max(MM * np.array(Fz[:, zc]).reshape(-1)))
                    print("MM current", MM_current * np.array(Fz_current[zc]).reshape(-1))
                    print("VoI_future", VoI_future)
                    print("VoI_current", VoI_current)
                    print("KG", KG)

            except:
                print("max future", np.max(MM * np.array(Fz[:, zc]).reshape(-1)))
                print("MM current", MM_current * np.array(Fz_current[zc]).reshape(-1))
                print("VoI_future", VoI_future)
                print("VoI_current", VoI_current)
                print("KG", KG)

            KG = np.clip(KG, 0, np.inf)
            marginal_KG.append(KG)

        out.append(marginal_KG)
        # if verbose:
        #     print("Zc", np.array_split(Zc, 1))
        #     print("marginal_KG",out)
        #     print("KG_value",np.mean(out))

        KG_value = np.mean(out)
        # gradKG_value = np.mean(gradout, axis=?)
        assert ~np.isnan(KG_value);
        "KG cant be nan"
        return KG_value  # , gradKG_value

    def parallel_KG(self, MM, SS, Xd=None, verbose=False):
        """
        Calculates the linear epigraph, i.e. the boundary of the set of points
        in 2D lying above a collection of straight lines y=a+bx.
        Parameters
        ----------
        a
            Vector of intercepts describing a set of straight lines
        b
            Vector of slopes describing a set of straight lines
        tol
            Minimum slope (in absolute value) different from zero
        Returns
        -------
        KGCB
            average hieght of the epigraph
        grad_a
            dKGCB/db, vector
        grad_b
            dKGCB/db, vector
        """
        a = MM  # np.array(self.c_MM[:, index]).reshape(-1)
        b = SS  # np.array(self.c_SS[:, index]).reshape(-1)

        assert len(b) > 0, "must provide slopes"
        assert len(a) == len(b), f"#intercepts != #slopes, {len(a)}, {len(b)}"
        # ensure 1D
        a = np.array(a).reshape(-1)
        b = np.array(b).reshape(-1)
        assert len(a) > 0, "must provide slopes"
        assert len(a) == len(b), f"#intercepts != #slopes, {len(a)}, {len(b)}"

        # ensure 1D
        a = np.atleast_1d(a.squeeze())
        b = np.atleast_1d(b.squeeze())  # - np.mean(b)
        max_a_index = np.argmax(a)
        maxa = np.max(a)
        n_elems = len(a)

        if np.all(np.abs(b) < 0.000000001):
            return np.array([0])  # , np.zeros(a.shape), np.zeros(b.shape)

        # order by ascending b and descending a
        order = np.lexsort((-a, b))
        a = a[order]
        b = b[order]

        # exclude duplicated b (or super duper similar b)
        threshold = (b[-1] - b[0]) * 0.00001
        diff_b = b[1:] - b[:-1]
        keep = diff_b > threshold
        keep = np.concatenate([[True], keep])
        keep[np.argmax(a)] = True
        order = order[keep]
        a = a[keep]
        b = b[keep]

        # initialize
        idz = [0]
        i_last = 0
        x = [-np.inf]

        n_lines = len(a)
        # main loop TODO describe logic
        # TODO not pruning properly, e.g. a=[0,1,2], b=[-1,0,1]
        # returns x=[-inf, -1, -1, inf], shouldn't affect kgcb
        while i_last < n_lines - 1:
            i_mask = np.arange(i_last + 1, n_lines)
            x_mask = -(a[i_last] - a[i_mask]) / (b[i_last] - b[i_mask])

            best_pos = np.argmin(x_mask)
            idz.append(i_mask[best_pos])
            x.append(x_mask[best_pos])

            i_last = idz[-1]

        x.append(np.inf)

        x = np.array(x)
        idz = np.array(idz)

        # found the epigraph, now compute the expectation
        a = a[idz]
        b = b[idz]
        order = order[idz]

        pdf = norm.pdf(x)
        cdf = norm.cdf(x)

        KG = np.sum(a * (cdf[1:] - cdf[:-1]) + b * (pdf[:-1] - pdf[1:]))
        # KG -= np.max(a)

        if KG - maxa < - 1e-5:
            print("x", x)
            print("a", a)
            print("b", b)
            print("maxa", maxa)
            print("KG", KG)
            z_all = np.linspace(x[1], x[-2], 100)
            for j in range(len(a)):

                print(x[1], x[-2])
                if x[j] < -30:
                    z = np.linspace(-3, x[j + 1], 3)
                elif x[j + 1] > 30:
                    z = np.linspace(x[j], 3, 3)
                else:
                    z = np.linspace(x[j], x[j + 1], 3)
                mu_star = a.reshape(-1)[j] + b.reshape(-1)[j] * z
                mu_star_all = a.reshape(-1)[j] + b.reshape(-1)[j] * z_all
                plt.plot(z_all, mu_star_all, color="grey")
                plt.plot(z, mu_star)
            plt.show()
            raise
        # if verbose:
        #     print("a_sorted ",a_sorted )
        #     print("b_sorted", b_sorted)
        #
        #     print("current max",np.max(a) )
        #     print("idz", idz)
        #     print("a", a)
        #     print("b", b)
        #     plt.scatter(Xd.reshape(-1), np.array(self.c_MM[:, index]).reshape(-1))
        #     plt.plot(np.linspace(0,5,2), np.repeat(np.max(a), 2), color="red")
        #     plt.show()
        #     # raise
        # if KG < -1e-5:
        #     print("KG cant be negative")
        #     print("np.sum(a * (cdf[1:] - cdf[:-1]) + b * (pdf[:-1] - pdf[1:]))",
        #           np.sum(a * (cdf[1:] - cdf[:-1]) + b * (pdf[:-1] - pdf[1:])))
        #     print("self.bases_value[index]", np.max(a))
        #     print("KG", KG)
        #     raise
        #
        # KG = np.clip(KG, 0, np.inf)

        if np.isnan(KG):
            print("KG", KG)
            print("self.bases_value[index]", max_a_index)
            print("a", a)
            print("b", b)
            raise

        return KG

    def run_inner_func_vals(self, f):
        X = initial_design('random', self.space, 1000)
        f_val = []
        for x in X:
            x = x.reshape(1, -1)
            f_val.append(np.array(f(x)).reshape(-1))
        print("f_val min max", np.min(f_val), np.max(f_val))
        plt.scatter(X[:, 0], X[:, 1], c=np.array(f_val).reshape(-1))
        plt.show()

    def update_current_best(self):
        n_observations = self.model.get_X_values().shape[0]
        if n_observations > self.counter:
            print("updating current best..........")
            self.counter = n_observations
            self.current_max_xopt, self.current_max_value = self._compute_current_max()
        assert self.current_max_value.reshape(-1) is not np.inf;
        "error ocurred updating current best"

    def current_func(self, X_inner):
        X_inner = np.atleast_2d(X_inner)
        mu = self.model.posterior_mean(X_inner)[0]
        mu = np.array(mu).reshape(-1)
        Fz = self.probability_feasibility_multi_gp(x=X_inner, model=self.model_c).reshape(-1)
        return -(mu * Fz).reshape(-1)

    def _compute_current_max(self):
        random_indexes = np.random.choice(range(len(self.underlying_discretisation)), size=np.min(5000, len(self.underlying_discretisation)), replace=False)
        fX_vals = self.current_func(self.underlying_discretisation[random_indexes])
        inner_opt_x = self.underlying_discretisation[np.argmax(-fX_vals)][None, :]
        inner_opt_val = np.min(fX_vals)

        inner_opt_x = np.array(inner_opt_x).reshape(-1)
        inner_opt_x = np.atleast_2d(inner_opt_x)
        return inner_opt_x, -inner_opt_val

    def probability_feasibility_multi_gp(self, x, model, mean=None, cov=None, l=0):

        x = np.atleast_2d(x)

        Fz = []
        for m in range(model.output_dim):
            Fz.append(self.probability_feasibility(x, model.output[m], l))
        Fz = np.product(Fz, axis=0)
        return Fz

    def _plots(self, test_samples, discretisation=None):

        if False:
            def current_func(X_inner, Z_const, aux_c):
                X_inner = np.atleast_2d(X_inner)
                mu = self.model.posterior_mean(X_inner)[0]
                mu = np.array(mu).reshape(-1)
                grad_c = gradients(x_new=x, model=self.model_c, Z=Z_const, aux=aux_c,
                                   X_inner=X_inner)  # , test_samples = initial_design('random', self.space, 1000))

                Fz = grad_c.compute_probability_feasibility_multi_gp(x=X_inner, l=0).reshape(-1)
                return -(mu * Fz).reshape(-1)

            plot_samples = initial_design('random', self.space, 1000)
            test_samples = np.sort(test_samples, axis=0)
            plot_samples = np.sort(plot_samples, axis=0)
            # print("test_samples",test_samples)
            varX_obj = self.model.posterior_variance(test_samples, noise=True)
            varX_c = self.model_c.posterior_variance(test_samples, noise=True)

            colors = ["magenta", "green", "red", "blue", "orange"]
            Z_obj = np.array([-3, -1.64, -2.64, -0.67, 0, 0.67, 1.64, 2.64, 3])
            Z_const = np.array([-3, -1.64, -2.64, -0.67, 0, 0.67, 1.64, 2.64, 3])

            optimums_objs = []
            optimums_const = []
            optimiser_objs = []
            optimiser_const = []
            future_means_const = []
            future_means_objs = []
            for i in range(0, len(test_samples)):
                x = np.atleast_2d(test_samples[i])
                # For each x new precompute covariance matrices for
                self.model.partial_precomputation_for_covariance(x)
                self.model.partial_precomputation_for_covariance_gradient(x)
                self.model_c.partial_precomputation_for_covariance(x)
                self.model_c.partial_precomputation_for_covariance_gradient(x)
                # Precompute aux_obj and aux_c for computational efficiency.
                aux_obj = np.reciprocal(varX_obj[:, i])
                aux_c = np.reciprocal(varX_c[:, i])
                maxgraph = []
                feasibility_graph = []
                posterior_mean_c = []
                posterior_var_c = []
                for zc in range(len(Z_const)):
                    individual_maxgraph = []

                    grad_c = gradients(x_new=x, model=self.model_c, Z=Z_const[zc], aux=aux_c,
                                       X_inner=plot_samples)  # , test_samples = initial_design('random', self.space, 1000))
                    feasibility_graph.append(
                        grad_c.compute_probability_feasibility_multi_gp(x=plot_samples, l=0).reshape(-1))
                    posterior_mean_c.append(grad_c.compute_value_mu_xnew(x=plot_samples).reshape(-1))
                    posterior_var_c.append(grad_c.compute_posterior_var_x_new(x=plot_samples).reshape(-1))
                    for zo in range(len(Z_obj)):
                        def inner_func(X_inner, verbose=False):
                            X_inner = np.atleast_2d(X_inner)
                            # X_inner = X_inner.astype("int")
                            grad_obj = gradients(x_new=x, model=self.model, Z=Z_obj[zo], aux=aux_obj,
                                                 X_inner=X_inner)  # , test_samples = self.test_samples)
                            mu_xnew = grad_obj.compute_value_mu_xnew(x=X_inner)

                            grad_c = gradients(x_new=x, model=self.model_c, Z=Z_const[zc], aux=aux_c,
                                               X_inner=X_inner)  # , test_samples = initial_design('random', self.space, 1000))

                            Fz = grad_c.compute_probability_feasibility_multi_gp(x=X_inner, l=0)

                            func_val = mu_xnew * Fz  # - inner_opt_val

                            return -func_val.reshape(-1)

                        def inner_func_with_gradient(X_inner):
                            # print("inner_func_with_gradient")

                            X_inner = np.atleast_2d(X_inner)
                            # X_inner = X_inner.astype("int")

                            grad_obj = gradients(x_new=x, model=self.model, Z=Z_obj[zo], aux=aux_obj, X_inner=X_inner,
                                                 precompute_grad=True)
                            mu_xnew = grad_obj.compute_value_mu_xnew(x=X_inner)

                            grad_mu_xnew = grad_obj.compute_gradient_mu_xnew(x=X_inner)

                            grad_c = gradients(x_new=x, model=self.model_c, Z=Z_const[zc], aux=aux_c, X_inner=X_inner,
                                               precompute_grad=True)

                            Fz, grad_Fz = grad_c.compute_probability_feasibility_multi_gp(x=X_inner, l=0,
                                                                                          gradient_flag=True)
                            func_val = np.array(mu_xnew * Fz)  # - self.control_variate

                            func_grad_val = np.array(mu_xnew).reshape(-1) * grad_Fz.reshape(-1) + Fz.reshape(
                                -1) * grad_mu_xnew.reshape(
                                -1)  # grad_c.product_gradient_rule(func = np.array([np.array(mu_xnew).reshape(-1), Fz.reshape(-1)]), grad = np.array([grad_mu_xnew.reshape(-1) ,grad_Fz.reshape(-1) ]))

                            assert ~ np.isnan(func_val);
                            "nans found"
                            print("-func_val, -func_grad_val", -func_val, -func_grad_val)
                            return -func_val, func_grad_val

                        def inner_func_predict(X_inner):
                            X_inner = np.atleast_2d(X_inner)
                            # X_inner = X_inner.astype("int")
                            grad_obj = gradients(x_new=x, model=self.model, Z=Z_obj[zo], aux=aux_obj,
                                                 X_inner=X_inner)
                            mu_xnew = self.model.predict(X_inner)[0].reshape(-1)
                            # var_xnew = self.model.predict(X_inner)[1].reshape(-1)
                            var_xnew = grad_obj.compute_b(xopt=X_inner)

                            grad_c = gradients(x_new=x, model=self.model_c, Z=Z_const[zc], aux=aux_c,
                                               X_inner=X_inner)

                            Fz = grad_c.compute_probability_feasibility_multi_gp(x=X_inner, l=0)

                            func_val = mu_xnew.reshape(-1) * Fz.reshape(-1)  # - inner_opt_val

                            return func_val.reshape(-1), var_xnew.reshape(-1) * (np.array(Fz).reshape(-1))

                        if discretisation is not None:
                            current_test_samples_mean = -current_func(X_inner=plot_samples, Z_const=Z_const[zc],
                                                                      aux_c=aux_c)
                            next_test_samples_mean = -inner_func(plot_samples)
                            mean_n1, varn_n1 = inner_func_predict(plot_samples)

                            current_inner_opt_x, current_inner_opt_val = self._compute_current_max(x, Z_const[zc],
                                                                                                   aux_c)

                            discretisation_vals = -inner_func(discretisation)
                            next_inner_opt_x = discretisation[np.argmax(discretisation_vals)]
                            # print("func_val2",inner_func(next_inner_opt_x , verbose=True))
                            next_inner_opt_val = -np.max(discretisation_vals)

                            individual_maxgraph.append(-next_inner_opt_val)
                            maxgraph.append(-next_inner_opt_val)
                            plt.plot(plot_samples, mean_n1, color="lightcoral")
                            plt.plot(plot_samples, mean_n1 + 2.64 * varn_n1, color="lightcoral", alpha=0.3)
                            plt.plot(plot_samples, mean_n1 - 2.64 * varn_n1, color="lightcoral", alpha=0.3)
                            plt.fill_between(plot_samples.reshape(-1), mean_n1 - 1.96 * varn_n1,
                                             mean_n1 + 1.96 * varn_n1, color="lightcoral", alpha=0.2)

                            plt.plot(plot_samples, current_test_samples_mean, color="grey")
                            plt.plot(plot_samples, next_test_samples_mean, color="grey", linestyle='dashed')
                            plt.scatter(current_inner_opt_x, current_inner_opt_val, color="red")
                            plt.scatter(next_inner_opt_x, -next_inner_opt_val, color="green")
                            plt.scatter(np.array(discretisation).reshape(-1),
                                        np.repeat([0], len(np.array(discretisation).reshape(-1))), color="blue")
                            plt.scatter(x, [0], color="magenta")

                        else:

                            next_test_samples_mean = -inner_func(plot_samples)
                            next_inner_opt_x, next_inner_opt_val = self.optimizer.optimize_inner_func(f=inner_func,
                                                                                                      f_df=None)
                            individual_maxgraph.append(-next_inner_opt_val)
                            maxgraph.append(-next_inner_opt_val)

                            if Z_obj[zo] == 0:
                                future_means_const.append(next_test_samples_mean)
                                optimums_const.append(-next_inner_opt_val)
                                optimiser_const.append(next_inner_opt_x)
                                if Z_const[zc] == 0:
                                    current_mean_const = next_test_samples_mean
                                    current_optimums_const = -next_inner_opt_val
                                    current_optimiser_const = next_inner_opt_x

                            if Z_const[zc] == 0:
                                future_means_objs.append(next_test_samples_mean)
                                optimums_objs.append(-next_inner_opt_val)
                                optimiser_objs.append(next_inner_opt_x)
                                if Z_obj[zo] == 0:
                                    current_mean_objs = next_test_samples_mean
                                    current_optimums_objs = -next_inner_opt_val
                                    current_optimiser_objs = next_inner_opt_x

                ####plot Zy
                plt.plot(plot_samples, current_mean_objs, color="black", linestyle="dashed", zorder=0)
                plt.plot(plot_samples, np.array(future_means_objs).T, alpha=0.4, color="grey", zorder=0)
                plt.fill_between(plot_samples.reshape(-1), np.min(future_means_objs, axis=0),
                                 np.max(future_means_objs, axis=0), color="lightcoral",
                                 alpha=0.7, zorder=0)

                plt.scatter(np.array(optimiser_objs).reshape(-1), np.array(optimums_objs).reshape(-1), color="red",
                            label="$\max_{x}$  $\mu_{i}^{n+1}(x) PF_{j}^{n+1}(x)$", edgecolors='black', zorder=1)
                plt.scatter(np.array(current_optimiser_objs).reshape(-1), np.array(current_optimums_objs).reshape(-1),
                            color="green", zorder=2, edgecolors='black')
                plt.xlim(1.2, 3.5)
                plt.ylim(-0.3, 1.5)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.legend(loc="upper left", fontsize=15)
                # plt.savefig(
                #     "/home/juan/Documents/repos_data/Constrained-KG/RESULTS/plot_saved_data/plots/Zy_change.jpg",
                #     bbox_inches="tight")
                plt.show()

                # PLOTS Zc
                plt.plot(plot_samples, current_mean_const, color="black", linestyle="dashed", zorder=0)
                plt.plot(plot_samples, np.array(future_means_const).T, alpha=0.4, color="grey", zorder=0)
                plt.fill_between(plot_samples.reshape(-1), np.min(future_means_const, axis=0),
                                 np.max(future_means_const, axis=0), color="lightcoral",
                                 alpha=0.7, zorder=0)

                plt.scatter(np.array(optimiser_const).reshape(-1), np.array(optimums_const).reshape(-1), color="red",
                            label="$\max_{x}$  $\mu_{i}^{n+1}(x) PF_{j}^{n+1}(x)$", zorder=1, edgecolors='black')
                plt.scatter(np.array(current_optimiser_const).reshape(-1), np.array(current_optimums_const).reshape(-1),
                            color="green", zorder=2, edgecolors='black')
                plt.legend(loc="upper left", fontsize=15)
                plt.xlim(1.2, 3.5)
                plt.ylim(-0.3, 1.2)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                # plt.savefig(
                #     "/home/juan/Documents/repos_data/Constrained-KG/RESULTS/plot_saved_data/plots/Zc_change.jpg",
                #     bbox_inches="tight")
                plt.show()

                Zo, Zc = np.meshgrid(Z_obj, Z_const)
                maxvals = np.array(maxgraph).reshape(Zo.shape)

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_wireframe(Zc, Zo, maxvals, color="red")
                ax.set_ylabel('$Z_{y}$', fontsize=15)
                ax.set_xlabel('$Z_{c}$', fontsize=15)
                ax.set_title('$\max_{x}$  $\mu_{i}^{n+1}(x) PF_{j}^{n+1}(x)$', fontsize=15)

                plt.savefig(
                    "/home/juan/Documents/repos_data/Constrained-KG/RESULTS/plot_saved_data/plots/epigraph.jpg",
                    bbox_inches="tight")
                plt.show()

            # plt.plot(plot_samples, np.array(feasibility_graph).T)
            # plt.title("feasible region")
            # plt.show()

            # plt.plot(plot_samples, np.array(posterior_mean_c).T)
            # plt.plot(plot_samples, np.array(self.model_c.predict(plot_samples)[0]).reshape(-1), color="black")
            # plot_samples = np.linspace(0,10,100)[:,np.newaxis]
            varX_c = self.model_c.posterior_variance(test_samples, noise=True)
            aux_c = np.reciprocal(varX_c[:, 0])
            grad_c = gradients(x_new=test_samples, model=self.model_c, Z=Z_const[0], aux=aux_c,
                               X_inner=plot_samples)

            var_xnew = grad_c.compute_b(xopt=plot_samples)

            plt.title("b constraint")
            plt.scatter(test_samples, 0, color="magenta")
            plt.scatter(plot_samples, var_xnew)
            plt.scatter(np.array(self.model_c.get_X_values()).reshape(-1),
                        np.repeat(0, len(np.array(self.model_c.get_X_values()).reshape(-1))), color="green")
            plt.show()

            varX_c = self.model.posterior_variance(test_samples, noise=True)
            aux_c = np.reciprocal(varX_c[:, 0])
            grad_c = gradients(x_new=test_samples, model=self.model, Z=Z_const[0], aux=aux_c,
                               X_inner=plot_samples)

            var_xnew = grad_c.compute_b(xopt=plot_samples)

            plt.title("b objective")
            plt.scatter(test_samples, 0, color="magenta")
            plt.scatter(plot_samples, var_xnew)
            plt.scatter(np.array(self.model_c.get_X_values()).reshape(-1),
                        np.repeat(0, len(np.array(self.model_c.get_X_values()).reshape(-1))), color="green")
            plt.show()

            # print("corr", self.model_c.posterior_covariance_between_points(test_samples, test_samples))
            # print("varX_c", varX_c)
            # print("b", np.sqrt(varX_c))
            # raise
            plot_samples = initial_design('random', self.space, 1000)
            # plt.plot(plot_samples, np.array(self.model_c.predict(plot_samples)[0]).reshape(-1) + 1.95*np.sqrt(var_xnew), color="black")
            # plt.plot(plot_samples,
            #          np.array(self.model_c.predict(plot_samples)[0]).reshape(-1) - 1.95 * np.sqrt(var_xnew), color="black")
            # plt.title("posterior means constraints")
            # plt.show()
            #
            # plt.title("posterior var constraints")
            # plt.plot(plot_samples, np.array(posterior_var_c).T)
            # plt.show()
        if True:

            def current_penalised_func(X_inner, Z_const, aux_c):
                X_inner = np.atleast_2d(X_inner)
                mu = self.model.posterior_mean(X_inner)
                var = self.model.posterior_variance(X_inner, 1e-4)
                mu_c = self.model_c.posterior_mean(X_inner)
                var_c = self.model_c.posterior_variance(X_inner, 1e-4)
                mu = np.array(mu).reshape(-1)

                Fz = gradients.compute_probability_feasibility(mu_c, var_c.reshape(-1))

                return -(mu * Fz).reshape(-1), -(mu).reshape(-1), var, mu_c.reshape(-1), var_c.reshape(-1), (
                    Fz).reshape(-1)

            from matplotlib.pyplot import figure
            kg_val = []
            plot_kg_samples = np.linspace(4, 7, 90)[:, np.newaxis]  # initial_design('random', self.space, 500)
            plot_kg_samples = np.sort(plot_kg_samples, axis=0)
            varX_obj = self.model.posterior_variance(plot_kg_samples, noise=True)
            varX_c = self.model_c.posterior_variance(plot_kg_samples, noise=True)

            for i in range(len(plot_kg_samples)):
                x = np.atleast_2d(plot_kg_samples[i])
                self.model.partial_precomputation_for_covariance(x)
                self.model.partial_precomputation_for_covariance_gradient(x)
                self.model_c.partial_precomputation_for_covariance(x)
                self.model_c.partial_precomputation_for_covariance_gradient(x)

                aux_obj = np.reciprocal(varX_obj[:, i])
                aux_c = np.reciprocal(varX_c[:, i])
                kg_val.append(
                    self.discrete_KG(Xd=self.fixed_discretisation_values, xnew=x, Zc=self.Z_cdKG, aux_obj=aux_obj,
                                     aux_c=aux_c, grad=False))

            varX_obj = self.model.posterior_variance(test_samples, noise=True)
            varX_c = self.model_c.posterior_variance(test_samples, noise=True)
            for i in range(len(test_samples)):
                x = np.atleast_2d(test_samples[i])
                self.model.partial_precomputation_for_covariance(x)
                self.model.partial_precomputation_for_covariance_gradient(x)
                self.model_c.partial_precomputation_for_covariance(x)
                self.model_c.partial_precomputation_for_covariance_gradient(x)

                aux_obj = np.reciprocal(varX_obj[:, i])
                aux_c = np.reciprocal(varX_c[:, i])
                best_kg_val = self.discrete_KG(Xd=self.fixed_discretisation_values, xnew=x, Zc=self.Z_cdKG,
                                               aux_obj=aux_obj, aux_c=aux_c, grad=False)

            def penalised_mean_GP(X_inner):
                # print("inner_func_with_gradient")

                X_inner = np.atleast_2d(X_inner)

                mu_xnew = self.model.predict(X_inner)[0]
                var_xnew = self.model.posterior_variance(X_inner, noise=False)

                Fz = self.probability_feasibility_multi_gp(x=X_inner, model=self.model_c).reshape(-1)

                mean = np.array(mu_xnew.reshape(-1) * Fz)  # - self.control_variate
                var = np.array(var_xnew.reshape(-1) * Fz ** 2)

                return mean, var

            figure(figsize=(8, 6))
            plot_gp_samples = initial_design('random', self.space, 500)
            plot_gp_samples = np.sort(plot_gp_samples, axis=0)

            mean_current_GP, var_current_GP = penalised_mean_GP(plot_gp_samples)
            plt.plot(plot_gp_samples.reshape(-1), np.array(mean_current_GP).reshape(-1), color="black",
                     linestyle="dashed", alpha=0.5, zorder=0)
            # plt.fill_between(plot_gp_samples.reshape(-1), np.array(mean_current_GP).reshape(-1) - 1.65* np.sqrt(var_current_GP),
            #                  np.array(mean_current_GP).reshape(-1) + 1.65* np.sqrt(var_current_GP), color="lightcoral", alpha=0.3, zorder=0)

            current_max = np.max(mean_current_GP)
            current_optimiser = plot_gp_samples[np.argmax(mean_current_GP)]

            plt.scatter(test_samples.reshape(-1), penalised_mean_GP(test_samples)[0], color="red", edgecolors="black",
                        label="$max_{x} cKG(x)$", s=60, zorder=2)
            plt.scatter(self.model.get_X_values(), np.array(self.model.get_Y_values()).reshape(-1) * np.array(
                np.array(self.model_c.get_Y_values()).reshape(-1) < 0).reshape(-1), color="white", edgecolors="black",
                        s=60, zorder=1)
            plt.scatter(current_optimiser, current_max, color="orange", edgecolors="black",
                        label="$max_{x} \mu^{n}(x) PF^{n}(x)$", s=60, zorder=2)
            plt.xlabel("$X$", fontsize=15)
            plt.ylabel("$\mu^{n}(x) PF^{n}(x)$", fontsize=15)
            # plt.title("n = " + str(len(self.model.get_X_values())), fontsize=14)
            plt.xlim(4, 7)

            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)

            x = plot_gp_samples
            y = self.true_func(x, true_val=True)
            c = self.true_const(x)
            truefval = y.reshape(-1) * np.array(c < 0).reshape(-1)

            current_max = np.max(truefval)
            current_optimiser = x[np.argmax(truefval)]
            plt.plot(x[truefval > 0], truefval[truefval > 0], color="green", label="True function", zorder=-1)
            plt.plot(x[truefval == 0], truefval[truefval == 0], color="green", linestyle="--", label="True function",
                     zorder=-1)
            plt.scatter(current_optimiser, current_max, color="green", edgecolor="black", zorder=2,
                        label="$\max_{x} f$")
            # plt.legend(loc="upper right", prop={'size': 14})

            plt.savefig("/home/juan/Documents/repos_data/PhD_Thesis/pictures_revised/posterior_GP_{n}_v2.pdf".format(
                n=len(self.model.get_X_values())),
                        bbox_inches="tight")
            plt.show()

            plot_samples = initial_design('random', self.space, 1000)
            plot_samples = np.sort(plot_samples, axis=0)
            penalised_f_vals, f_vals, f_var, c_mean_vals, c_variance, PF = current_penalised_func(X_inner=plot_samples,
                                                                                                  Z_const=[0],
                                                                                                  aux_c=np.array([[1]]))

            plt.plot(x, y, color="green", label="True function", zorder=-1)
            plt.plot(plot_samples, -f_vals, color="black", linestyle="--", label="$\mu_{y}^{n}(x)$")
            plt.fill_between(plot_samples.reshape(-1),
                             -np.array(f_vals).reshape(-1) - 1.95 * np.sqrt(f_var).reshape(-1),
                             -np.array(f_vals).reshape(-1) + 1.95 * np.sqrt(f_var).reshape(-1), color="lightcoral",
                             alpha=0.3, zorder=0)

            plt.scatter(self.model.get_X_values(), np.array(self.model.get_Y_values()).reshape(-1), color="white",
                        edgecolors="black",
                        s=60, zorder=1)
            plt.xlabel("$\mathbb{X}$", fontsize=15)
            plt.ylabel("$\mathbb{Y}$", fontsize=20)
            plt.ylim(-3, 3)
            plt.xlim(4, 7)
            plt.legend(prop={'size': 14})
            plt.savefig(
                "/home/juan/Documents/repos_data/PhD_Thesis/pictures_revised/posterior_GP_objective_func_{n}_v2.pdf".format(
                    n=len(self.model.get_X_values())),
                bbox_inches="tight")
            plt.show()

            fig, axs = plt.subplots(2, gridspec_kw={'height_ratios': [4, 1]})

            axs[0].plot(x, c, color="red")
            axs[0].plot(plot_samples, c_mean_vals, color="black", linestyle="--", label="$\mu_{c}^{n}(x)$")
            axs[0].fill_between(plot_samples.reshape(-1),
                                np.array(c_mean_vals).reshape(-1) - 1.95 * np.sqrt(c_variance).reshape(-1),
                                np.array(c_mean_vals).reshape(-1) + 1.95 * np.sqrt(c_variance).reshape(-1),
                                color="lightcoral",
                                alpha=0.3, zorder=0)
            axs[0].hlines(y=0, xmin=4, xmax=7, linewidth=2, color='grey', linestyles="--", label="Threshold")
            axs[0].scatter(self.model_c.get_X_values(), np.array(self.model_c.get_Y_values()).reshape(-1),
                           color="white", edgecolors="black",
                           s=60, zorder=1)
            axs[0].set_xlabel("$\mathbb{X}$", fontsize=20)
            axs[0].set_ylabel("$\mathbb{Y}$", fontsize=20)
            axs[0].set_ylim(-3, 3)
            axs[0].set_xlim(4, 7)

            axs[1].plot(plot_samples, PF, color="black")
            axs[1].set_xlim(4, 7)
            axs[1].set_ylabel("PF", fontsize=20)
            axs[0].legend(prop={'size': 14})
            plt.savefig(
                "/home/juan/Documents/repos_data/PhD_Thesis/pictures_revised/posterior_GP_constraint_func_{n}_v2.pdf".format(
                    n=len(self.model.get_X_values())),
                bbox_inches="tight")
            plt.show()

            plt.plot(x.reshape(-1)[x.reshape(-1) < 4.6], y.reshape(-1)[x.reshape(-1) < 4.6], color="green", linewidth=3)
            plt.plot(x.reshape(-1)[x.reshape(-1) > 5.8], y.reshape(-1)[x.reshape(-1) > 5.8], label="feasible",
                     color="green", linewidth=3)
            plt.plot(x.reshape(-1)[(4.6 < x.reshape(-1)) & (x.reshape(-1) < 5.8)],
                     y.reshape(-1)[(4.6 < x.reshape(-1)) & (x.reshape(-1) < 5.8)] * 0, color="red",
                     linestyle="--", linewidth=3, label="infeasible")
            plt.plot(plot_samples, - penalised_f_vals, color="black", linestyle="--", label="PF * $\mu_{y}^{n}(x)$")
            plt.scatter(self.model.get_X_values(), np.array(self.model.get_Y_values()).reshape(-1) * np.array(
                np.array(self.model_c.get_Y_values()).reshape(-1) < 0).reshape(-1), color="white", edgecolors="black",
                        s=60, zorder=1)
            plt.ylim(-3, 3)
            plt.xlim(4, 7)
            plt.xlabel("$\mathbb{X}$", fontsize=20)
            plt.ylabel("$\mathbb{Y}$", fontsize=20)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.legend(prop={'size': 14})
            plt.savefig(
                "/home/juan/Documents/repos_data/PhD_Thesis/pictures_revised/posterior_GP_penalised_func_{n}_v2.pdf".format(
                    n=len(self.model.get_X_values())),
                bbox_inches="tight")
            plt.show()

            figure(figsize=(8, 6))
            plt.scatter(test_samples.reshape(-1), best_kg_val.reshape(-1), color="red", edgecolors="black",
                        label="$argmax_{x} cKG(x)$", s=60, zorder=1)
            plt.plot(plot_kg_samples.reshape(-1), np.array(kg_val).reshape(-1), color="black", linewidth=3, zorder=0)
            plt.xlabel("$X$", fontsize=15)
            plt.ylabel("$cKG$", fontsize=15)
            plt.xlim(4, 7)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            # plt.legend(loc="upper right", prop={'size': 14})
            plt.savefig(
                "/home/juan/Documents/repos_data/Constrained-KG/RESULTS/plot_saved_data/plots/cKG_{n}_v2.pdf".format(
                    n=len(self.model.get_X_values())),
                bbox_inches="tight")
            plt.show()

    def probability_feasibility(self, x, model, l=0):

        model = model.model

        mean = model.posterior_mean(x)
        var = model.posterior_variance(x, noise=False)

        std = np.sqrt(var).reshape(-1, 1)
        mean = mean.reshape(-1, 1)
        norm_dist = norm(mean, std)
        Fz = norm_dist.cdf(l)

        return Fz.reshape(-1, 1)
