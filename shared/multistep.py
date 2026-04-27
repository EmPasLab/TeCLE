import torch

import base


def n_step_bellman_target(
    r_t: torch.Tensor,
    done: torch.Tensor,
    q_t: torch.Tensor,
    gamma: float,
    n_steps: int,
) -> torch.Tensor:
    base.assert_rank_and_dtype(r_t, 2, torch.float32)
    base.assert_rank_and_dtype(done, 2, torch.bool)
    base.assert_rank_and_dtype(q_t, 2, torch.float32)

    base.assert_batch_dimension(done, q_t.shape[0])
    base.assert_batch_dimension(r_t, q_t.shape[0])
    base.assert_batch_dimension(done, q_t.shape[1], 1)
    base.assert_batch_dimension(r_t, q_t.shape[1], 1)

    bellman_target = torch.concat(
        [torch.zeros_like(q_t[0:1]), q_t] + [q_t[-1:] / gamma**k for k in range(1, n_steps)], dim=0
    )

    done = torch.concat([done] + [torch.zeros_like(done[0:1])] * n_steps, dim=0)
    rewards = torch.concat([r_t] + [torch.zeros_like(r_t[0:1])] * n_steps, dim=0)

    for _ in range(n_steps):
        rewards = rewards[:-1]
        done = done[:-1]
        bellman_target = rewards + gamma * (1.0 - done.float()) * bellman_target[1:]

    return bellman_target


def truncated_generalized_advantage_estimation(
    r_t: torch.Tensor,
    value_t: torch.Tensor,
    value_tp1: torch.Tensor,
    discount_tp1: torch.Tensor,
    lambda_: float,
) -> torch.Tensor:
    base.assert_rank_and_dtype(r_t, 1, torch.float32)
    base.assert_rank_and_dtype(value_t, 1, torch.float32)
    base.assert_rank_and_dtype(value_tp1, 1, torch.float32)
    base.assert_rank_and_dtype(discount_tp1, 1, torch.float32)

    lambda_ = torch.ones_like(discount_tp1) * lambda_

    delta_t = r_t + discount_tp1 * value_tp1 - value_t

    advantage_t = torch.zeros_like(delta_t, dtype=torch.float32)

    gae_t = 0
    for i in reversed(range(len(delta_t))):
        gae_t = delta_t[i] + discount_tp1[i] * lambda_[i] * gae_t
        advantage_t[i] = gae_t

    return advantage_t


def general_off_policy_returns_from_action_values(
    q_t: torch.Tensor,
    a_t: torch.Tensor,
    r_t: torch.Tensor,
    discount_t: torch.Tensor,
    c_t: torch.Tensor,
    pi_t: torch.Tensor,
) -> torch.Tensor:
    base.assert_rank_and_dtype(q_t, 3, torch.float32)
    base.assert_rank_and_dtype(a_t, 2, torch.long)
    base.assert_rank_and_dtype(r_t, 2, torch.float32)
    base.assert_rank_and_dtype(discount_t, 2, torch.float32)
    base.assert_rank_and_dtype(c_t, 2, torch.float32)
    base.assert_rank_and_dtype(pi_t, 3, torch.float32)

    for i in (0, 1):
        base.assert_batch_dimension(a_t, q_t.shape[i], i)
        base.assert_batch_dimension(r_t, q_t.shape[i], i)
        base.assert_batch_dimension(discount_t, q_t.shape[i], i)
        base.assert_batch_dimension(c_t, q_t.shape[i], i)
        base.assert_batch_dimension(pi_t, q_t.shape[i], i)

    exp_q_t = (pi_t * q_t).sum(axis=-1)

    q_a_t = base.batched_index(q_t, a_t)[:-1, ...]
    c_t = c_t[:-1, ...]

    return general_off_policy_returns_from_q_and_v(q_a_t, exp_q_t, r_t, discount_t, c_t)


def general_off_policy_returns_from_q_and_v(
    q_t: torch.Tensor,
    v_t: torch.Tensor,
    r_t: torch.Tensor,
    discount_t: torch.Tensor,
    c_t: torch.Tensor,
) -> torch.Tensor:
    base.assert_rank_and_dtype(q_t, 2, torch.float32)
    base.assert_rank_and_dtype(v_t, 2, torch.float32)
    base.assert_rank_and_dtype(r_t, 2, torch.float32)
    base.assert_rank_and_dtype(discount_t, 2, torch.float32)
    base.assert_rank_and_dtype(c_t, 2, torch.float32)

    for i in (0, 1):
        base.assert_batch_dimension(v_t, r_t.shape[i], i)
        base.assert_batch_dimension(discount_t, r_t.shape[i], i)

    g = r_t[-1] + discount_t[-1] * v_t[-1]
    returns = [g]
    for i in reversed(range(q_t.shape[0])):
        g = r_t[i] + discount_t[i] * (v_t[i] - c_t[i] * q_t[i] + c_t[i] * g)
        returns.insert(0, g)

    return torch.stack(returns, dim=0).detach()
