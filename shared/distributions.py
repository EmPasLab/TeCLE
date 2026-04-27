import torch
from torch.distributions import Categorical, Normal

import base


def categorical_distribution(logits: torch.Tensor) -> torch.distributions.Categorical:
    return Categorical(logits=logits)


def normal_distribution(mu: torch.Tensor, sigma: torch.Tensor) -> torch.distributions.Normal:
    return Normal(mu, sigma)


def categorical_importance_sampling_ratios(
    pi_logits_t: torch.Tensor, mu_logits_t: torch.Tensor, a_t: torch.Tensor
) -> torch.Tensor:
    base.assert_rank_and_dtype(pi_logits_t, (2, 3), torch.float32)
    base.assert_rank_and_dtype(mu_logits_t, (2, 3), torch.float32)
    base.assert_rank_and_dtype(a_t, (1, 2), torch.long)

    pi_m = Categorical(logits=pi_logits_t)
    mu_m = Categorical(logits=mu_logits_t)

    pi_logprob_a_t = pi_m.log_prob(a_t)
    mu_logprob_a_t = mu_m.log_prob(a_t)

    return torch.exp(pi_logprob_a_t - mu_logprob_a_t)
