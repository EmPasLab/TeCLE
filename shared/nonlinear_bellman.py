from typing import NamedTuple, Callable, Any
import functools

import torch

import base
import value_learning
import multistep
import transforms


class TxPair(NamedTuple):
    apply: Callable[[Any], Any]
    apply_inv: Callable[[Any], Any]


IDENTITY_PAIR = TxPair(transforms.identity, transforms.identity)
SIGNED_LOGP1_PAIR = TxPair(transforms.signed_logp1, transforms.signed_expm1)
SIGNED_HYPERBOLIC_PAIR = TxPair(transforms.signed_hyperbolic, transforms.signed_parabolic)
HYPERBOLIC_SIN_PAIR = TxPair(transforms.hyperbolic_arcsin, transforms.hyperbolic_sin)


def transform_values(build_targets, *value_argnums):
    @functools.wraps(build_targets)
    def wrapped_build_targets(tx_pair, *args, **kwargs):
        tx_args = list(args)
        for index in value_argnums:
            tx_args[index] = tx_pair.apply_inv(tx_args[index])

        targets = build_targets(*tx_args, **kwargs)
        return tx_pair.apply(targets)

    return wrapped_build_targets


transformed_general_off_policy_returns_from_action_values = transform_values(
    multistep.general_off_policy_returns_from_action_values, 0
)


def transformed_retrace(
    q_tm1: torch.Tensor,
    q_t: torch.Tensor,
    a_tm1: torch.Tensor,
    a_t: torch.Tensor,
    r_t: torch.Tensor,
    discount_t: torch.Tensor,
    pi_t: torch.Tensor,
    mu_t: torch.Tensor,
    lambda_: float,
    eps: float = 1e-8,
    tx_pair: TxPair = IDENTITY_PAIR,
) -> base.LossOutput:
    base.assert_rank_and_dtype(q_tm1, 3, torch.float32)
    base.assert_rank_and_dtype(q_t, 3, torch.float32)
    base.assert_rank_and_dtype(a_tm1, 2, torch.long)
    base.assert_rank_and_dtype(a_t, 2, torch.long)
    base.assert_rank_and_dtype(r_t, 2, torch.float32)
    base.assert_rank_and_dtype(discount_t, 2, torch.float32)
    base.assert_rank_and_dtype(pi_t, 3, torch.float32)
    base.assert_rank_and_dtype(mu_t, 2, torch.float32)

    pi_a_t = base.batched_index(pi_t, a_t)
    c_t = torch.minimum(torch.tensor(1.0), pi_a_t / (mu_t + eps)) * lambda_

    with torch.no_grad():
        target_tm1 = transformed_general_off_policy_returns_from_action_values(tx_pair, q_t, a_t, r_t, discount_t, c_t, pi_t)
    q_a_tm1 = base.batched_index(q_tm1, a_tm1)
    td_error = target_tm1 - q_a_tm1
    loss = 0.5 * td_error**2

    return base.LossOutput(loss, value_learning.QExtra(target=target_tm1, td_error=td_error))
