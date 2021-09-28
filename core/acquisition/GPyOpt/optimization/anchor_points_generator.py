# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from ..experiment_design import initial_design
from scipy.stats import norm
from ..core.errors import FullyExploredOptimizationDomainError
from ..core.task.space import Design_space

class AnchorPointsGenerator(object):

    def __init__(self, space, design_type, num_samples):

        self.space = space
        self.design_type = design_type
        self.num_samples = num_samples

    def get_anchor_point_scores(self, X):
        raise NotImplementedError("get_anchor_point_scores is not implemented in the parent class.")

    def get(self, num_anchor=3, X_sampled_values=None,duplicate_manager=None, unique=False, context_manager=None, verbose=False):
        ## --- We use the context handler to remove duplicates only over the non-context variables
        #if context_manager and not self.space._has_bandit():
            #space_configuration_without_context = [self.space.config_space_expanded[idx] for idx in context_manager.nocontext_index_obj]
            #space = Design_space(space_configuration_without_context)
            #add_context = lambda x : context_manager._expand_vector(x)
        #else:
            #space = self.space
            #add_context = lambda x: x

        ## --- Generate initial design
        X = initial_design(self.design_type, self.space, self.num_samples)

        #if unique:
            #sorted_design = sorted(list({tuple(x) for x in X}))
            #X = space.unzip_inputs(np.vstack(sorted_design))
        #else:
            #X = space.unzip_inputs(X)

        ## --- Add context variables
        #X = add_context(X)

        #if duplicate_manager:
            #is_duplicate = duplicate_manager.is_unzipped_x_duplicate
        #else:
            # In absence of duplicate manager, we never detect duplicates
            #is_duplicate = lambda _ : False

        #non_duplicate_anchor_point_indexes = [index for index, x in enumerate(X) if not is_duplicate(x)]

        #if not non_duplicate_anchor_point_indexes:
            #raise FullyExploredOptimizationDomainError("No anchor points could be generated ({} used samples, {} requested anchor points).".format(self.num_samples,num_anchor))

        #if len(non_duplicate_anchor_point_indexes) < num_anchor:
            # Since logging has not been setup yet, I do not know how to express warnings...I am using standard print for now.
            #print("Warning: expecting {} anchor points, only {} available.".format(num_anchor, len(non_duplicate_anchor_point_indexes)))

        #X = X[non_duplicate_anchor_point_indexes,:]


        if X_sampled_values is not None:

            X = np.concatenate((X[X_sampled_values.shape[0]:],X_sampled_values))

        # print("X", X.shape)
        scores = self.get_anchor_point_scores(X)
        # print("scores", scores.shape)
        # print("num_anchor", num_anchor)
        anchor_points = X[np.argsort(scores)[:min(len(scores),num_anchor)], :]
        # print("anchor_points",anchor_points.shape)
        # raise
        # print("np.argsort(scores)[:min(len(scores),num_anchor)]",np.sort(scores)[:min(len(scores),num_anchor)])
        if verbose:
            import matplotlib.pyplot as plt
            plt.plot(X[np.argsort(scores)], np.sort(scores))
            plt.show()
        return anchor_points


class ThompsonSamplingAnchorPointsGenerator(AnchorPointsGenerator):

    def __init__(self, space, design_type, model, cmodel = None, num_samples=25000):
        '''
        From and initial design, it selects the location using (marginal) Thompson sampling
        using the predictive distribution of a model

        model: NOTE THAT THE MODEL HERE IS is a GPyOpt model: returns mean and standard deviation
        '''
        super(ThompsonSamplingAnchorPointsGenerator, self).__init__(space, design_type, num_samples)
        self.model = model
        self.constraint = False
        if cmodel is not None:
            self.constraint = True
            self.model_c = cmodel

    def get_anchor_point_scores(self, X):

        posterior_means, posterior_stds = self.model.predict(X)
        Y = np.array([np.random.normal(m, s) for m, s in zip(posterior_means, posterior_stds)]).flatten()

        if self.constraint:
            constraints_posterior_means, constraints_posterior_stds = self.model_c.predict(X)
            # print("constraints_posterior_means",constraints_posterior_means.shape)
            # print("constraints_posterior_stds ",constraints_posterior_stds.shape)

            Yc = np.zeros(constraints_posterior_means.shape)
            for c in range(constraints_posterior_means.shape[0]):
                Yc[c,:] = np.array([np.random.normal(m, s) for m, s in zip(constraints_posterior_means[c,:], constraints_posterior_stds[c,:])]).flatten()

            Yc[Yc >= 0] = 0
            Yc[Yc<0] = 1
            constraint_vals = np.product(Yc,axis=0)

            Y = np.multiply(Y, constraint_vals)

        # import matplotlib.pyplot as plt
        # plt.scatter(X[:,0], X[:,1], c=Y)
        # plt.show()
        return -Y


class ObjectiveAnchorPointsGenerator(AnchorPointsGenerator):

    def __init__(self, space, design_type, objective, num_samples=50):
        '''
        From an initial design, it selects the locations with the minimum value according to some objective.
        :param model_space: set to true when the samples need to be obtained for the input domain of the model

        '''
        super(ObjectiveAnchorPointsGenerator, self).__init__(space, design_type, num_samples)
        self.objective = objective

    def get_anchor_point_scores(self, X):
        # print("X", X.shape)
        out = np.array(self.objective(X)).flatten()
        # print("out", out.shape)
        return out


class RandomAnchorPointsGenerator(AnchorPointsGenerator):

    def __init__(self, space, design_type, num_samples=15):
        '''
        From an initial design, it selects the locations randomly, according to the specified design_type generation scheme.
        :param model_space: set to true when the samples need to be obtained for the input domain of the model

        '''
        super(RandomAnchorPointsGenerator, self).__init__(space, design_type, num_samples)

    def get_anchor_point_scores(self, X):
        return range(X.shape[0])
