from typing import NamedTuple, Optional
import torch
import torch.nn.functional as F

import base
import multistep


class QExtra(NamedTuple):
    target: Optional[torch.Tensor]
    td_error: Optional[torch.Tensor]


class DoubleQExtra(NamedTuple):
    target: torch.Tensor
    td_error: torch.Tensor
    best_action: torch.Tensor


class Extra(NamedTuple):
    target: Optional[torch.Tensor]


def qlearning(
    q_tm1: torch.Tensor,
    a_tm1: torch.Tensor,
    r_t: torch.Tensor,
    discount_t: torch.Tensor,
    q_t: torch.Tensor,
) -> base.LossOutput:
    base.assert_rank_and_dtype(q_tm1, 2, torch.float32)
    base.assert_rank_and_dtype(a_tm1, 1, torch.long)
    base.assert_rank_and_dtype(r_t, 1, torch.float32)
    base.assert_rank_and_dtype(discount_t, 1, torch.float32)
    base.assert_rank_and_dtype(q_t, 2, torch.float32)

    base.assert_batch_dimension(a_tm1, q_tm1.shape[0])
    base.assert_batch_dimension(r_t, q_tm1.shape[0])
    base.assert_batch_dimension(discount_t, q_tm1.shape[0])
    base.assert_batch_dimension(q_t, q_tm1.shape[0])

    with torch.no_grad():
        target_tm1 = r_t + discount_t * torch.max(q_t, dim=1)[0]
    qa_tm1 = base.batched_index(q_tm1, a_tm1)

    td_error = target_tm1 - qa_tm1
    loss = 0.5 * td_error**2

    return base.LossOutput(loss, QExtra(target_tm1, td_error))


def double_qlearning(
    q_tm1: torch.Tensor,
    a_tm1: torch.Tensor,
    r_t: torch.Tensor,
    discount_t: torch.Tensor,
    q_t_value: torch.Tensor,
    q_t_selector: torch.Tensor,
) -> base.LossOutput:
    base.assert_rank_and_dtype(q_tm1, 2, torch.float32)
    base.assert_rank_and_dtype(a_tm1, 1, torch.long)
    base.assert_rank_and_dtype(r_t, 1, torch.float32)
    base.assert_rank_and_dtype(discount_t, 1, torch.float32)
    base.assert_rank_and_dtype(q_t_value, 2, torch.float32)
    base.assert_rank_and_dtype(q_t_selector, 2, torch.float32)

    base.assert_batch_dimension(a_tm1, q_tm1.shape[0])
    base.assert_batch_dimension(r_t, q_tm1.shape[0])
    base.assert_batch_dimension(discount_t, q_tm1.shape[0])
    base.assert_batch_dimension(q_t_value, q_tm1.shape[0])
    base.assert_batch_dimension(q_t_selector, q_tm1.shape[0])

    best_action = torch.argmax(q_t_selector, dim=1)

    double_q_bootstrapped = base.batched_index(q_t_value, best_action)

    with torch.no_grad():
        target_tm1 = r_t + discount_t * double_q_bootstrapped

    qa_tm1 = base.batched_index(q_tm1, a_tm1)

    td_error = target_tm1 - qa_tm1
    loss = 0.5 * td_error**2

    return base.LossOutput(loss, DoubleQExtra(target_tm1, td_error, best_action))


def _slice_with_actions(embeddings: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    batch_size, action_dim = embeddings.shape[:2]

    act_idx = actions[:, None]

    values = torch.reshape(torch.ones(actions.shape, dtype=torch.int8, device=actions.device), [-1])

    act_range = torch.arange(0, batch_size, dtype=torch.int64)[:, None].to(device=actions.device)

    indices = torch.concat([act_range, act_idx], 1)

    actions_mask = torch.sparse_coo_tensor(indices.t(), values, [batch_size, action_dim])
    with torch.no_grad():
        actions_mask = actions_mask.to_dense().bool()

    sliced_emb = torch.masked_select(embeddings, actions_mask[..., None])

    sliced_emb = sliced_emb.reshape(embeddings.shape[0], -1)
    return sliced_emb


def l2_project(z_p: torch.Tensor, p: torch.Tensor, z_q: torch.Tensor) -> torch.Tensor:
    vmin, vmax = z_q[0], z_q[-1]
    d_pos = torch.concat([z_q, vmin[None]], 0)[1:]
    d_neg = torch.concat([vmax[None], z_q], 0)[:-1]

    z_p = torch.clamp(z_p, min=vmin, max=vmax)[:, None, :]

    d_pos = (d_pos - z_q)[None, :, None]
    d_neg = (z_q - d_neg)[None, :, None]
    z_q = z_q[None, :, None]

    d_neg = torch.where(d_neg > 0, 1.0 / d_neg, torch.zeros_like(d_neg))
    d_pos = torch.where(d_pos > 0, 1.0 / d_pos, torch.zeros_like(d_pos))

    delta_qp = z_p - z_q
    d_sign = (delta_qp >= 0.0).to(dtype=p.dtype)

    delta_hat = (d_sign * delta_qp * d_pos) - ((1.0 - d_sign) * delta_qp * d_neg)
    p = p[:, None, :]
    return torch.sum(torch.clamp(1.0 - delta_hat, min=0.0, max=1.0) * p, 2)


def categorical_dist_qlearning(
    atoms_tm1: torch.Tensor,
    logits_q_tm1: torch.Tensor,
    a_tm1: torch.Tensor,
    r_t: torch.Tensor,
    discount_t: torch.Tensor,
    atoms_t: torch.Tensor,
    logits_q_t: torch.Tensor,
) -> base.LossOutput:
    base.assert_rank_and_dtype(logits_q_tm1, 3, torch.float32)
    base.assert_rank_and_dtype(a_tm1, 1, torch.long)
    base.assert_rank_and_dtype(r_t, 1, torch.float32)
    base.assert_rank_and_dtype(discount_t, 1, torch.float32)
    base.assert_rank_and_dtype(logits_q_t, 3, torch.float32)
    base.assert_rank_and_dtype(atoms_tm1, 1, torch.float32)
    base.assert_rank_and_dtype(atoms_t, 1, torch.float32)

    base.assert_batch_dimension(a_tm1, logits_q_tm1.shape[0])
    base.assert_batch_dimension(r_t, logits_q_tm1.shape[0])
    base.assert_batch_dimension(discount_t, logits_q_tm1.shape[0])
    base.assert_batch_dimension(logits_q_t, logits_q_tm1.shape[0])
    base.assert_batch_dimension(atoms_tm1, logits_q_tm1.shape[-1])
    base.assert_batch_dimension(atoms_t, logits_q_tm1.shape[-1])

    target_z = r_t[:, None] + discount_t[:, None] * atoms_t[None, :]

    q_t_probs = F.softmax(logits_q_t, dim=-1)
    q_t_mean = torch.sum(q_t_probs * atoms_t, 2)
    pi_t = torch.argmax(q_t_mean, 1)

    p_target_z = _slice_with_actions(q_t_probs, pi_t)

    with torch.no_grad():
        target_tm1 = l2_project(target_z, p_target_z, atoms_tm1)

    logit_qa_tm1 = _slice_with_actions(logits_q_tm1, a_tm1)

    loss = F.cross_entropy(input=logit_qa_tm1, target=target_tm1, reduction='none')

    return base.LossOutput(loss, Extra(target_tm1))


def categorical_dist_double_qlearning(
    atoms_tm1: torch.Tensor,
    logits_q_tm1: torch.Tensor,
    a_tm1: torch.Tensor,
    r_t: torch.Tensor,
    discount_t: torch.Tensor,
    atoms_t: torch.Tensor,
    logits_q_t: torch.Tensor,
    q_t_selector: torch.Tensor,
) -> base.LossOutput:
    base.assert_rank_and_dtype(logits_q_tm1, 3, torch.float32)
    base.assert_rank_and_dtype(a_tm1, 1, torch.long)
    base.assert_rank_and_dtype(r_t, 1, torch.float32)
    base.assert_rank_and_dtype(discount_t, 1, torch.float32)
    base.assert_rank_and_dtype(logits_q_t, 3, torch.float32)
    base.assert_rank_and_dtype(q_t_selector, 2, torch.float32)
    base.assert_rank_and_dtype(atoms_tm1, 1, torch.float32)
    base.assert_rank_and_dtype(atoms_t, 1, torch.float32)

    base.assert_batch_dimension(a_tm1, logits_q_tm1.shape[0])
    base.assert_batch_dimension(r_t, logits_q_tm1.shape[0])
    base.assert_batch_dimension(discount_t, logits_q_tm1.shape[0])
    base.assert_batch_dimension(logits_q_t, logits_q_tm1.shape[0])
    base.assert_batch_dimension(q_t_selector, logits_q_tm1.shape[0])
    base.assert_batch_dimension(atoms_tm1, logits_q_tm1.shape[-1])
    base.assert_batch_dimension(atoms_t, logits_q_tm1.shape[-1])

    target_z = r_t[:, None] + discount_t[:, None] * atoms_t[None, :]

    q_t_probs = F.softmax(logits_q_t, dim=-1)
    pi_t = torch.argmax(q_t_selector, dim=1)

    p_target_z = _slice_with_actions(q_t_probs, pi_t)

    with torch.no_grad():
        target_tm1 = l2_project(target_z, p_target_z, atoms_tm1)

    logit_qa_tm1 = _slice_with_actions(logits_q_tm1, a_tm1)

    loss = F.cross_entropy(input=logit_qa_tm1, target=target_tm1, reduction='none')

    return base.LossOutput(loss, Extra(target_tm1))


def huber_loss(x: torch.Tensor, k: float = 1.0) -> torch.Tensor:
    return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))


def _quantile_regression_loss(
    dist_src: torch.Tensor,
    tau_src: torch.Tensor,
    dist_target: torch.Tensor,
    huber_param: float = 0.0,
) -> torch.Tensor:
    base.assert_rank_and_dtype(dist_src, 2, torch.float32)
    base.assert_rank_and_dtype(tau_src, 2, torch.float32)
    base.assert_rank_and_dtype(dist_target, 2, torch.float32)

    base.assert_batch_dimension(tau_src, dist_src.shape[0])
    base.assert_batch_dimension(dist_target, dist_src.shape[0])

    delta = dist_target.unsqueeze(1) - dist_src.unsqueeze(-1)

    delta_neg = (delta < 0.0).float().detach()
    weight = torch.abs(tau_src.unsqueeze(-1) - delta_neg)

    if huber_param > 0.0:
        loss = huber_loss(delta, huber_param)
    else:
        loss = torch.abs(delta)
    loss *= weight

    return torch.sum(torch.mean(loss, dim=-1), dim=1)


def quantile_q_learning(
    dist_q_tm1: torch.Tensor,
    tau_q_tm1: torch.Tensor,
    a_tm1: torch.Tensor,
    r_t: torch.Tensor,
    discount_t: torch.Tensor,
    dist_q_t: torch.Tensor,
    huber_param: float = 0.0,
) -> base.LossOutput:
    base.assert_rank_and_dtype(dist_q_tm1, 3, torch.float32)
    base.assert_rank_and_dtype(tau_q_tm1, 2, torch.float32)
    base.assert_rank_and_dtype(a_tm1, 1, torch.long)
    base.assert_rank_and_dtype(r_t, 1, torch.float32)
    base.assert_rank_and_dtype(discount_t, 1, torch.float32)
    base.assert_rank_and_dtype(dist_q_t, 3, torch.float32)

    base.assert_batch_dimension(a_tm1, dist_q_tm1.shape[0])
    base.assert_batch_dimension(r_t, dist_q_tm1.shape[0])
    base.assert_batch_dimension(discount_t, dist_q_tm1.shape[0])
    base.assert_batch_dimension(dist_q_t, dist_q_tm1.shape[0])

    dist_qa_tm1 = base.batched_index(dist_q_tm1, a_tm1, 2)

    q_t_selector = torch.mean(dist_q_t, dim=1)
    a_t = torch.argmax(q_t_selector, dim=1)

    dist_qa_t = base.batched_index(dist_q_t, a_t, 2)

    with torch.no_grad():
        dist_target_tm1 = r_t[:, None] + discount_t[:, None] * dist_qa_t

    loss = _quantile_regression_loss(dist_qa_tm1, tau_q_tm1, dist_target_tm1, huber_param)
    return base.LossOutput(loss, Extra(dist_target_tm1))


def quantile_double_q_learning(
    dist_q_tm1: torch.Tensor,
    tau_q_tm1: torch.Tensor,
    a_tm1: torch.Tensor,
    r_t: torch.Tensor,
    discount_t: torch.Tensor,
    dist_q_t: torch.Tensor,
    q_t_selector: torch.Tensor,
    huber_param: float = 0.0,
) -> base.LossOutput:
    base.assert_rank_and_dtype(dist_q_tm1, 3, torch.float32)
    base.assert_rank_and_dtype(tau_q_tm1, 2, torch.float32)
    base.assert_rank_and_dtype(a_tm1, 1, torch.long)
    base.assert_rank_and_dtype(r_t, 1, torch.float32)
    base.assert_rank_and_dtype(discount_t, 1, torch.float32)
    base.assert_rank_and_dtype(dist_q_t, 3, torch.float32)
    base.assert_rank_and_dtype(q_t_selector, 3, torch.float32)

    base.assert_batch_dimension(a_tm1, dist_q_tm1.shape[0])
    base.assert_batch_dimension(r_t, dist_q_tm1.shape[0])
    base.assert_batch_dimension(discount_t, dist_q_tm1.shape[0])
    base.assert_batch_dimension(dist_q_t, dist_q_tm1.shape[0])
    base.assert_batch_dimension(q_t_selector, dist_q_tm1.shape[0])

    dist_qa_tm1 = base.batched_index(dist_q_tm1, a_tm1, 2)

    q_t_selector = torch.mean(q_t_selector, dim=1)
    a_t = torch.argmax(q_t_selector, dim=1)

    dist_qa_t = base.batched_index(dist_q_t, a_t, 2)

    with torch.no_grad():
        dist_target_tm1 = r_t[:, None] + discount_t[:, None] * dist_qa_t

    loss = _quantile_regression_loss(dist_qa_tm1, tau_q_tm1, dist_target_tm1, huber_param)
    return base.LossOutput(loss, Extra(dist_target_tm1))


def retrace(
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
        target_tm1 = multistep.general_off_policy_returns_from_action_values(q_t, a_t, r_t, discount_t, c_t, pi_t)

    qa_tm1 = base.batched_index(q_tm1, a_tm1)

    td_error = target_tm1 - qa_tm1
    loss = 0.5 * td_error**2

    return base.LossOutput(loss, QExtra(target=target_tm1, td_error=td_error))
