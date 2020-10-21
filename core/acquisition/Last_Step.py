from botorch.acquisition.monte_carlo import MCAcquisitionFunction

import math
from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.objective import IdentityMCObjective, MCAcquisitionObjective
from botorch.acquisition.utils import prune_inferior_points
from botorch.exceptions.errors import UnsupportedError
from botorch.models.model import Model
from botorch.sampling.samplers import MCSampler, SobolQMCNormalSampler
from botorch.utils.transforms import (
    concatenate_pending_points,
    match_batch_shape,
    t_batch_mode_transform,
)
from torch import Tensor


class Constrained_Mean_Response(MCAcquisitionFunction):
    r"""MC-based batch Expected Improvement.
    This computes qEI by
    (1) sampling the joint posterior over q points
    (2) evaluating the improvement over the current best for each sample
    (3) maximizing over q
    (4) averaging over the samples
    `qEI(X) = E(max(max Y - best_f, 0)), Y ~ f(X), where X = (x_1,...,x_q)`
    Example:
        # >>> model = SingleTaskGP(train_X, train_Y)
        # >>> best_f = train_Y.max()[0]
        # >>> sampler = SobolQMCNormalSampler(1000)
        # >>> qEI = qExpectedImprovement(model, best_f, sampler)
        # >>> qei = qEI(test_X)
    """

    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        X_pending: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> None:
        r"""q-Expected Improvement.
        Args:
            model: A fitted model.
            best_f: The best objective value observed so far (assumed noiseless). Can be
                a `batch_shape`-shaped tensor, which in case of a batched model
                specifies potentially different values for each element of the batch.
            sampler: The sampler used to draw base samples. Defaults to
                `SobolQMCNormalSampler(num_samples=500, collapse_batch_dims=True)`
            objective: The MCAcquisitionObjective under which the samples are evalauted.
                Defaults to `IdentityMCObjective()`.
            X_pending:  A `m x d`-dim Tensor of `m` design points that have been
                submitted for function evaluation but have not yet been evaluated.
                Concatenated into X upon forward call. Copied and set to have no
                gradient.
        """
        super().__init__(
            model=model, sampler=sampler, objective=objective, X_pending=X_pending
        )
        if not torch.is_tensor(best_f):
            best_f = torch.tensor(float(best_f))
        self.register_buffer("best_f", best_f)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qExpectedImprovement on the candidate set `X`.
        Args:
            X: A `batch_shape x q x d`-dim Tensor of t-batches with `q` `d`-dim design
                points each.
        Returns:
            A `batch_shape'`-dim Tensor of Expected Improvement values at the given
            design points `X`, where `batch_shape'` is the broadcasted batch shape of
            model and input `X`.
        """
        posterior = self.model.posterior(X)
        samples = self.sampler(posterior)
        obj = self.objective(samples)
        q_ei = obj.max(dim=-1)[0].mean(dim=0)
        return q_ei

# class Constrained_Mean_Response(AnalyticAcquisitionFunction):
#     r"""Constrained Expected Improvement (feasibility-weighted).
#     Computes the analytic expected improvement for a Normal posterior
#     distribution, weighted by a probability of feasibility. The objective and
#     constraints are assumed to be independent and have Gaussian posterior
#     distributions. Only supports the case `q=1`. The model should be
#     multi-outcome, with the index of the objective and constraints passed to
#     the constructor.
#     `Constrained_EI(x) = EI(x) * Product_i P(y_i \in [lower_i, upper_i])`,
#     where `y_i ~ constraint_i(x)` and `lower_i`, `upper_i` are the lower and
#     upper bounds for the i-th constraint, respectively.
#     Example:
#     """
#
#     def __init__(
#         self,
#         model: Model,
#         best_f: Union[float, Tensor],
#         objective_index: int,
#         constraints: Dict[int, Tuple[Optional[float], Optional[float]]],
#         maximize: bool = True,
#     ) -> None:
#         r"""Analytic Constrained Expected Improvement.
#         Args:
#             model: A fitted single-outcome model.
#             best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
#                 the best function value observed so far (assumed noiseless).
#             objective_index: The index of the objective.
#             constraints: A dictionary of the form `{i: [lower, upper]}`, where
#                 `i` is the output index, and `lower` and `upper` are lower and upper
#                 bounds on that output (resp. interpreted as -Inf / Inf if None)
#             maximize: If True, consider the problem a maximization problem.
#         """
#         # use AcquisitionFunction constructor to avoid check for objective
#         super(AnalyticAcquisitionFunction, self).__init__(model=model)
#         self.objective = None
#         self.maximize = maximize
#         self.objective_index = objective_index
#         self.constraints = constraints
#         self.register_buffer("best_f", torch.as_tensor(best_f))
#         self._preprocess_constraint_bounds(constraints=constraints)
#         self.register_forward_pre_hook(convert_to_target_pre_hook)
#
#     @t_batch_mode_transform(expected_q=1)
#     def forward(self, X: Tensor) -> Tensor:
#         r"""Evaluate Constrained Expected Improvement on the candidate set X.
#         Args:
#             X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
#                 points each.
#         Returns:
#             A `(b)`-dim Tensor of Expected Improvement values at the given
#             design points `X`.
#         """
#         self.best_f = self.best_f.to(X)
#         posterior = self._get_posterior(X=X)
#         means = posterior.mean.squeeze(dim=-2)  # (b) x m
#         sigmas = posterior.variance.squeeze(dim=-2).sqrt().clamp_min(1e-9)  # (b) x m
#
#         # (b) x 1
#         oi = self.objective_index
#         mean_obj = means[..., oi : oi + 1]
#         sigma_obj = sigmas[..., oi : oi + 1]
#         u = mean_obj #(mean_obj - self.best_f.expand_as(mean_obj)) / sigma_obj
#         if not self.maximize:
#             u = -u
#         # normal = Normal(
#         #     torch.zeros(1, device=u.device, dtype=u.dtype),
#         #     torch.ones(1, device=u.device, dtype=u.dtype),
#         # )
#         # ei_pdf = torch.exp(normal.log_prob(u))  # (b) x 1
#         # ei_cdf = normal.cdf(u)
#         # ei = sigma_obj * (ei_pdf + u * ei_cdf)
#         prob_feas = self._compute_prob_feas(X=X, means=means, sigmas=sigmas)
#         ei = u.mul(prob_feas)
#         return ei.squeeze(dim=-1)
#
#     def _preprocess_constraint_bounds(
#         self, constraints: Dict[int, Tuple[Optional[float], Optional[float]]]
#     ) -> None:
#         r"""Set up constraint bounds.
#         Args:
#             constraints: A dictionary of the form `{i: [lower, upper]}`, where
#                 `i` is the output index, and `lower` and `upper` are lower and upper
#                 bounds on that output (resp. interpreted as -Inf / Inf if None)
#         """
#         con_lower, con_lower_inds = [], []
#         con_upper, con_upper_inds = [], []
#         con_both, con_both_inds = [], []
#         con_indices = list(constraints.keys())
#         if len(con_indices) == 0:
#             raise ValueError("There must be at least one constraint.")
#         if self.objective_index in con_indices:
#             raise ValueError(
#                 "Output corresponding to objective should not be a constraint."
#             )
#         for k in con_indices:
#             if constraints[k][0] is not None and constraints[k][1] is not None:
#                 if constraints[k][1] <= constraints[k][0]:
#                     raise ValueError("Upper bound is less than the lower bound.")
#                 con_both_inds.append(k)
#                 con_both.append([constraints[k][0], constraints[k][1]])
#             elif constraints[k][0] is not None:
#                 con_lower_inds.append(k)
#                 con_lower.append(constraints[k][0])
#             elif constraints[k][1] is not None:
#                 con_upper_inds.append(k)
#                 con_upper.append(constraints[k][1])
#         # tensor-based indexing is much faster than list-based advanced indexing
#         self.register_buffer("con_lower_inds", torch.tensor(con_lower_inds))
#         self.register_buffer("con_upper_inds", torch.tensor(con_upper_inds))
#         self.register_buffer("con_both_inds", torch.tensor(con_both_inds))
#         # tensor indexing
#         self.register_buffer("con_both", torch.tensor(con_both, dtype=torch.float))
#         self.register_buffer("con_lower", torch.tensor(con_lower, dtype=torch.float))
#         self.register_buffer("con_upper", torch.tensor(con_upper, dtype=torch.float))
#
#     def _compute_prob_feas(self, X: Tensor, means: Tensor, sigmas: Tensor) -> Tensor:
#         r"""Compute feasibility probability for each batch of X.
#         Args:
#             X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
#                 points each.
#             means: A `(b) x m`-dim Tensor of means.
#             sigmas: A `(b) x m`-dim Tensor of standard deviations.
#         Returns:
#             A `(b) x 1`-dim tensor of feasibility probabilities
#         Note: This function does case-work for upper bound, lower bound, and both-sided
#         bounds. Another way to do it would be to use 'inf' and -'inf' for the
#         one-sided bounds and use the logic for the both-sided case. But this
#         causes an issue with autograd since we get 0 * inf.
#         TODO: Investigate further.
#         """
#         output_shape = X.shape[:-2] + torch.Size([1])
#         prob_feas = torch.ones(output_shape, device=X.device, dtype=X.dtype)
#
#         if len(self.con_lower_inds) > 0:
#             self.con_lower_inds = self.con_lower_inds.to(device=X.device)
#             normal_lower = _construct_dist(means, sigmas, self.con_lower_inds)
#             prob_l = 1 - normal_lower.cdf(self.con_lower)
#             prob_feas = prob_feas.mul(torch.prod(prob_l, dim=-1, keepdim=True))
#         if len(self.con_upper_inds) > 0:
#             self.con_upper_inds = self.con_upper_inds.to(device=X.device)
#             normal_upper = _construct_dist(means, sigmas, self.con_upper_inds)
#             prob_u = normal_upper.cdf(self.con_upper)
#             prob_feas = prob_feas.mul(torch.prod(prob_u, dim=-1, keepdim=True))
#         if len(self.con_both_inds) > 0:
#             self.con_both_inds = self.con_both_inds.to(device=X.device)
#             normal_both = _construct_dist(means, sigmas, self.con_both_inds)
#             prob_u = normal_both.cdf(self.con_both[:, 1])
#             prob_l = normal_both.cdf(self.con_both[:, 0])
#             prob_feas = prob_feas.mul(torch.prod(prob_u - prob_l, dim=-1, keepdim=True))
#         return prob_feas

# def _construct_dist(means: Tensor, sigmas: Tensor, inds: Tensor) -> Normal:
#     mean = means.index_select(dim=-1, index=inds)
#     sigma = sigmas.index_select(dim=-1, index=inds)
#     return Normal(loc=mean, scale=sigma)