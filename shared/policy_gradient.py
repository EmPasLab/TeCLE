from typing import NamedTuple, Optional
import torch
from torch.distributions import Categorical
import base


class EntropyExtra(NamedTuple):
    entropy: Optional[torch.Tensor]


def value_loss(target: torch.Tensor, predict: torch.Tensor) -> base.LossOutput:
    base.assert_rank_and_dtype(target, (1, 2), torch.float32)
    base.assert_rank_and_dtype(predict, (1, 2), torch.float32)

    assert target.shape == predict.shape

    loss = 0.5 * torch.square(target - predict)

    if len(loss.shape) == 2:

        loss = torch.mean(loss, dim=0)

    return base.LossOutput(loss, extra=None)


def entropy_loss(logits_t: torch.Tensor) -> base.LossOutput:
    base.assert_rank_and_dtype(logits_t, (2, 3), torch.float32)

    m = Categorical(logits=logits_t)
    entropy = m.entropy()

    if len(entropy.shape) == 2:

        entropy = torch.mean(entropy, dim=0)

    return base.LossOutput(entropy, None)


def policy_gradient_loss(
    logits_t: torch.Tensor,
    a_t: torch.Tensor,
    adv_t: torch.Tensor,
) -> base.LossOutput:
    base.assert_rank_and_dtype(logits_t, (2, 3), torch.float32)
    base.assert_rank_and_dtype(a_t, (1, 2), torch.long)
    base.assert_rank_and_dtype(adv_t, (1, 2), torch.float32)

    base.assert_batch_dimension(a_t, logits_t.shape[0])
    base.assert_batch_dimension(adv_t, logits_t.shape[0])

    if len(logits_t.shape) == 3:
        base.assert_batch_dimension(a_t, logits_t.shape[1], 1)
        base.assert_batch_dimension(adv_t, logits_t.shape[1], 1)

    m = Categorical(logits=logits_t)
    logprob_a_t = m.log_prob(a_t).view_as(adv_t)
    loss = logprob_a_t * adv_t.detach()

    if len(loss.shape) == 2:

        loss = torch.mean(loss, dim=0)

    return base.LossOutput(loss, extra=None)


def clipped_surrogate_gradient_loss(
    prob_ratios_t: torch.Tensor,
    adv_t: torch.Tensor,
    epsilon: float,
) -> base.LossOutput:
    base.assert_rank_and_dtype(prob_ratios_t, 1, torch.float32)
    base.assert_rank_and_dtype(adv_t, 1, torch.float32)

    clipped_ratios_t = torch.clamp(prob_ratios_t, 1.0 - epsilon, 1.0 + epsilon)
    clipped_objective = torch.min(prob_ratios_t * adv_t.detach(), clipped_ratios_t * adv_t.detach())

    return base.LossOutput(clipped_objective, extra=None)
