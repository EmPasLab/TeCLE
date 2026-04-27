# Copyright 2022 The Deep RL Zoo Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PPO-Disagree agent class.

Exploration by ensemble disagreement: an ensemble of forward dynamics models is trained
to predict next-state features; the variance of their predictions is used as an intrinsic
reward signal to encourage exploration of novel states.

From the paper "Self-Supervised Exploration via Disagreement"
https://arxiv.org/abs/1906.04161

From the paper "Proximal Policy Optimization Algorithms"
https://arxiv.org/abs/1707.06347.
"""
from typing import NamedTuple, Mapping, Tuple, Optional, Iterable, Text
import multiprocessing
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# pylint: disable=import-error
import type as types_lib
import policy_gradient as rl
from schedule import LinearSchedule
import utils
import distributions
import multistep
import base
import normalizer


class DisagreeModuleOutput(NamedTuple):
    """Disagreement module output (scalars, detached from compute graph)."""

    disagree_loss: Optional[torch.Tensor]
    intrinsic_reward_mean: Optional[float]


class Transition(NamedTuple):
    s_t: Optional[np.ndarray]
    a_t: Optional[int]
    logprob_a_t: Optional[float]
    returns_t: Optional[float]
    advantage_t: Optional[float]
    s_tp1: Optional[np.ndarray]


class Actor(types_lib.Agent):
    """PPO-Disagree actor for discrete action spaces."""

    def __init__(
        self,
        rank: int,
        data_queue: multiprocessing.Queue,
        policy_network: torch.nn.Module,
        unroll_length: int,
        device: torch.device,
        shared_params: dict,
    ) -> None:
        if not 1 <= unroll_length:
            raise ValueError(f'Expect unroll_length to be integer greater than or equal to 1, got {unroll_length}')

        self.rank = rank
        self.agent_name = f'PPO-Disagree-actor{rank}'
        self._queue = data_queue
        self._policy_network = policy_network.to(device=device)
        for p in self._policy_network.parameters():
            p.requires_grad = False
        self._device = device

        self._shared_params = shared_params

        self._unroll_length = unroll_length
        self._unroll_sequence = []

        self._step_t = -1

        self._s_tm1 = None
        self._a_tm1 = None
        self._logprob_a_tm1 = None

    def step(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        """Given current timestep, return action a_t, and push transition into global queue."""
        self._step_t += 1

        a_t, logprob_a_t = self.act(timestep)

        if self._a_tm1 is not None:
            self._unroll_sequence.append(
                (
                    self._s_tm1,
                    self._a_tm1,
                    self._logprob_a_tm1,
                    timestep.reward,
                    timestep.observation,
                    timestep.done,
                )
            )

            if len(self._unroll_sequence) == self._unroll_length:
                self._queue.put(self._unroll_sequence)
                self._unroll_sequence = []
                self._update_actor_network()

        self._s_tm1 = timestep.observation
        self._a_tm1 = a_t
        self._logprob_a_tm1 = logprob_a_t

        return a_t

    def reset(self) -> None:
        """This method should be called at the beginning of every episode."""
        self._s_tm1 = None
        self._a_tm1 = None
        self._logprob_a_tm1 = None

    def act(self, timestep: types_lib.TimeStep) -> Tuple[types_lib.Action]:
        'Given timestep, return an action.'
        return self._choose_action(timestep)

    def _update_actor_network(self):
        state_dict = self._shared_params['policy_network']
        if state_dict is not None:
            if self._device != 'cpu':
                state_dict = {k: v.to(device=self._device) for k, v in state_dict.items()}
            self._policy_network.load_state_dict(state_dict)

    @torch.no_grad()
    def _choose_action(self, timestep: types_lib.TimeStep) -> Tuple[types_lib.Action]:
        """Given timestep, choose action a_t."""
        s_t = torch.from_numpy(timestep.observation[None, ...]).to(device=self._device, dtype=torch.float32)
        pi_logits_t = self._policy_network(s_t).pi_logits
        pi_dist_t = distributions.categorical_distribution(pi_logits_t)
        a_t = pi_dist_t.sample()
        logprob_a_t = pi_dist_t.log_prob(a_t)
        return a_t.cpu().item(), logprob_a_t.cpu().item()

    @property
    def statistics(self) -> Mapping[Text, float]:
        """Returns current agent statistics as a dictionary."""
        return {}


class Learner(types_lib.Learner):
    """PPO-Disagree learner for discrete action spaces (Atari)."""

    def __init__(
        self,
        policy_network: nn.Module,
        policy_optimizer: torch.optim.Optimizer,
        disagree_network: nn.Module,
        disagree_optimizer: torch.optim.Optimizer,
        clip_epsilon: LinearSchedule,
        discount: float,
        gae_lambda: float,
        total_unroll_length: int,
        update_k: int,
        ext_coeff: float,
        int_coeff: float,
        entropy_coef: float,
        value_coef: float,
        clip_grad: bool,
        max_grad_norm: float,
        device: torch.device,
        shared_params: dict,
    ) -> None:
        if not 1 <= total_unroll_length:
            raise ValueError(f'Expect total_unroll_length >= 1, got {total_unroll_length}')
        if not 1 <= update_k:
            raise ValueError(f'Expect update_k >= 1, got {update_k}')
        if not 0.0 <= entropy_coef <= 1.0:
            raise ValueError(f'Expect entropy_coef in [0, 1], got {entropy_coef}')
        if not 0.0 <= value_coef <= 1.0:
            raise ValueError(f'Expect value_coef in (0, 1], got {value_coef}')

        self.agent_name = 'PPO-Disagree-learner'
        self._policy_network = policy_network.to(device=device)
        self._policy_network.train()
        self._policy_optimizer = policy_optimizer

        self._disagree_network = disagree_network.to(device=device)
        self._disagree_network.train()
        self._disagree_optimizer = disagree_optimizer

        self._device = device
        self._shared_params = shared_params

        self._int_reward_normalizer = normalizer.TorchRunningMeanStd(shape=(1,), device=self._device)

        self._ext_coeff = ext_coeff
        self._int_coeff = int_coeff

        self._storage = []
        self._total_unroll_length = total_unroll_length
        self._batch_size = int(np.ceil(total_unroll_length / 4).item())
        self._update_k = update_k

        self._entropy_coef = entropy_coef
        self._value_coef = value_coef
        self._clip_epsilon = clip_epsilon

        self._clip_grad = clip_grad
        self._max_grad_norm = max_grad_norm
        self._discount = discount
        self._gae_lambda = gae_lambda

        self._step_t = -1
        self._update_t = 0
        self._policy_loss_t = np.nan
        self._value_loss_t = np.nan
        self._entropy_loss_t = np.nan
        self._disagree_loss_t = np.nan
        self._intrinsic_reward_t = np.nan

    def step(self) -> Iterable[Mapping[Text, float]]:
        self._step_t += 1
        if len(self._storage) < self._total_unroll_length:
            return
        return self._learn()

    def reset(self) -> None:
        self._storage = []

    def received_item_from_queue(self, unroll_sequences: Iterable[Tuple]) -> None:
        s_t, a_t, logprob_a_t, r_t, s_tp1, done_tp1 = map(list, zip(*unroll_sequences))

        int_r_t = self._compute_intrinsic_rewards(s_t, a_t, s_tp1)
        r_combined = [self._ext_coeff * re + self._int_coeff * ri for re, ri in zip(r_t, int_r_t)]
        returns_t, advantage_t = self._compute_returns_and_advantages(s_t, r_combined, s_tp1, done_tp1)

        zipped_sequence = zip(s_t, a_t, logprob_a_t, returns_t, advantage_t, s_tp1)
        self._storage += zipped_sequence

    def get_policy_state_dict(self):
        return {k: v.cpu() for k, v in self._policy_network.state_dict().items()}

    def _learn(self) -> Iterable[Mapping[Text, float]]:
        num_samples = len(self._storage)

        for _ in range(self._update_k):
            binned_indices = utils.split_indices_into_bins(self._batch_size, num_samples, shuffle=True)
            for indices in binned_indices:
                transitions = [self._storage[i] for i in indices]

                s_t, a_t, logprob_a_t, returns_t, advantage_t, s_tp1 = map(list, zip(*transitions))
                stacked = Transition(
                    s_t=np.stack(s_t, axis=0),
                    a_t=np.stack(a_t, axis=0),
                    logprob_a_t=np.stack(logprob_a_t, axis=0),
                    returns_t=np.stack(returns_t, axis=0),
                    advantage_t=np.stack(advantage_t, axis=0),
                    s_tp1=np.stack(s_tp1, axis=0),
                )

                self._update_disagree_network(stacked)
                self._update_policy_network(stacked)
                self._update_t += 1
                yield self.statistics

        self._shared_params['policy_network'] = self.get_policy_state_dict()
        del self._storage[:]

    def _update_disagree_network(self, transitions: Transition) -> None:
        self._disagree_optimizer.zero_grad()
        loss, disagree_output = self._calc_disagree_loss(transitions)
        loss.backward()

        if self._clip_grad:
            torch.nn.utils.clip_grad_norm_(
                self._disagree_network.parameters(),
                max_norm=self._max_grad_norm,
                error_if_nonfinite=True,
            )

        self._disagree_optimizer.step()
        self._disagree_loss_t = disagree_output.disagree_loss.item()
        self._intrinsic_reward_t = disagree_output.intrinsic_reward_mean

    def _update_policy_network(self, transitions: Transition) -> None:
        self._policy_optimizer.zero_grad()
        loss = self._calc_policy_loss(transitions)
        loss.backward()

        if self._clip_grad:
            torch.nn.utils.clip_grad_norm_(
                self._policy_network.parameters(),
                max_norm=self._max_grad_norm,
                error_if_nonfinite=True,
            )

        self._policy_optimizer.step()

    def _calc_disagree_loss(self, transitions: Transition) -> Tuple[torch.Tensor, DisagreeModuleOutput]:
        s_t = torch.from_numpy(transitions.s_t).to(device=self._device, dtype=torch.float32)
        a_t = torch.from_numpy(transitions.a_t).to(device=self._device, dtype=torch.int64)
        s_tp1 = torch.from_numpy(transitions.s_tp1).to(device=self._device, dtype=torch.float32)

        base.assert_rank_and_dtype(s_t, (2, 4), torch.float32)
        base.assert_rank_and_dtype(s_tp1, (2, 4), torch.float32)
        base.assert_rank_and_dtype(a_t, 1, torch.long)

        output = self._disagree_network(s_t, a_t, s_tp1)

        member_losses = [F.mse_loss(pred, output.features_tp1) for pred in output.predictions]
        disagree_loss = torch.stack(member_losses).mean()

        with torch.no_grad():
            preds_stacked = torch.stack(output.predictions, dim=0)
            int_reward_mean = preds_stacked.var(dim=0).mean().item()

        return disagree_loss, DisagreeModuleOutput(
            disagree_loss=disagree_loss.detach(),
            intrinsic_reward_mean=int_reward_mean,
        )

    def _calc_policy_loss(self, transitions: Transition) -> torch.Tensor:
        s_t = torch.from_numpy(transitions.s_t).to(device=self._device, dtype=torch.float32)
        a_t = torch.from_numpy(transitions.a_t).to(device=self._device, dtype=torch.int64)
        behavior_logprob_a_t = torch.from_numpy(transitions.logprob_a_t).to(device=self._device, dtype=torch.float32)
        returns_t = torch.from_numpy(transitions.returns_t).to(device=self._device, dtype=torch.float32)
        advantage_t = torch.from_numpy(transitions.advantage_t).to(device=self._device, dtype=torch.float32)

        base.assert_rank_and_dtype(s_t, (2, 4), torch.float32)
        base.assert_rank_and_dtype(a_t, 1, torch.long)
        base.assert_rank_and_dtype(returns_t, 1, torch.float32)
        base.assert_rank_and_dtype(advantage_t, 1, torch.float32)
        base.assert_rank_and_dtype(behavior_logprob_a_t, 1, torch.float32)

        policy_output = self._policy_network(s_t)
        pi_logits_t = policy_output.pi_logits
        v_t = policy_output.value.squeeze(-1)

        pi_dist_t = distributions.categorical_distribution(pi_logits_t)
        entropy_loss = pi_dist_t.entropy()

        pi_logprob_a_t = pi_dist_t.log_prob(a_t)
        ratio = torch.exp(pi_logprob_a_t - behavior_logprob_a_t)

        if ratio.shape != advantage_t.shape:
            raise RuntimeError(f'Expect ratio and advantage_t have same shape, got {ratio.shape} and {advantage_t.shape}')

        policy_loss = rl.clipped_surrogate_gradient_loss(ratio, advantage_t, self.clip_epsilon).loss
        value_loss = rl.value_loss(returns_t, v_t).loss

        policy_loss = torch.mean(policy_loss)
        entropy_loss = torch.mean(entropy_loss)
        value_loss = torch.mean(value_loss)

        loss = -(policy_loss + self._entropy_coef * entropy_loss) + self._value_coef * value_loss

        self._policy_loss_t = policy_loss.detach().cpu().item()
        self._value_loss_t = value_loss.detach().cpu().item()
        self._entropy_loss_t = entropy_loss.detach().cpu().item()

        return loss

    @torch.no_grad()
    def _compute_intrinsic_rewards(
        self,
        s_t_list: Iterable[np.ndarray],
        a_t_list: Iterable[int],
        s_tp1_list: Iterable[np.ndarray],
    ) -> list:
        s_t = torch.from_numpy(np.stack(s_t_list, axis=0)).to(device=self._device, dtype=torch.float32)
        a_t = torch.from_numpy(np.stack(a_t_list, axis=0)).to(device=self._device, dtype=torch.int64)
        s_tp1 = torch.from_numpy(np.stack(s_tp1_list, axis=0)).to(device=self._device, dtype=torch.float32)

        output = self._disagree_network(s_t, a_t, s_tp1)

        preds_stacked = torch.stack(output.predictions, dim=0)
        int_reward = preds_stacked.var(dim=0).mean(dim=-1)

        self._int_reward_normalizer.update(int_reward.unsqueeze(-1))
        int_reward_normed = self._int_reward_normalizer.normalize(int_reward.unsqueeze(-1)).squeeze(-1)
        int_reward_normed = torch.clamp(int_reward_normed, -10.0, 10.0)

        return int_reward_normed.cpu().numpy().tolist()

    @torch.no_grad()
    def _compute_returns_and_advantages(
        self,
        s_t: Iterable[np.ndarray],
        r_t: Iterable[float],
        s_tp1: Iterable[np.ndarray],
        done_tp1: Iterable[bool],
    ):
        stacked_s_t = torch.from_numpy(np.stack(s_t, axis=0)).to(device=self._device, dtype=torch.float32)
        stacked_r_t = torch.tensor(r_t, dtype=torch.float32, device=self._device)
        stacked_s_tp1 = torch.from_numpy(np.stack(s_tp1, axis=0)).to(device=self._device, dtype=torch.float32)
        stacked_done_tp1 = torch.from_numpy(np.stack(done_tp1, axis=0)).to(device=self._device, dtype=torch.bool)

        discount_tp1 = (~stacked_done_tp1).float() * self._discount

        output_t = self._policy_network(stacked_s_t)
        v_t = output_t.value.squeeze(-1)
        v_tp1 = self._policy_network(stacked_s_tp1).value.squeeze(-1)

        advantage_t = multistep.truncated_generalized_advantage_estimation(
            stacked_r_t, v_t, v_tp1, discount_tp1, self._gae_lambda
        )
        return_t = advantage_t + v_t

        advantage_t = (advantage_t - advantage_t.mean()) / (advantage_t.std() + 1e-8)

        return return_t.cpu().numpy(), advantage_t.cpu().numpy()

    @property
    def clip_epsilon(self):
        return self._clip_epsilon(self._step_t)

    @property
    def statistics(self) -> Mapping[Text, float]:
        return {
            'policy_loss': self._policy_loss_t,
            'value_loss': self._value_loss_t,
            'entropy_loss': self._entropy_loss_t,
            'disagree_loss': self._disagree_loss_t,
            'intrinsic_reward': self._intrinsic_reward_t,
            'updates': self._update_t,
            'clip_epsilon': self.clip_epsilon,
        }


class GaussianActor(Actor):
    """PPO-Disagree actor for continuous action spaces."""

    def __init__(
        self,
        rank: int,
        data_queue: multiprocessing.Queue,
        policy_network: torch.nn.Module,
        unroll_length: int,
        device: torch.device,
        shared_params: dict,
    ) -> None:
        super().__init__(rank, data_queue, policy_network, unroll_length, device, shared_params)
        self.agent_name = f'PPO-Disagree-Gaussian-actor{rank}'

    @torch.no_grad()
    def _choose_action(self, timestep: types_lib.TimeStep) -> Tuple[np.ndarray]:
        """Given timestep, choose continuous action a_t from a Gaussian distribution."""
        s_t = torch.from_numpy(timestep.observation[None, ...]).to(device=self._device, dtype=torch.float32)
        pi_mu, pi_sigma = self._policy_network(s_t)
        pi_dist_t = distributions.normal_distribution(pi_mu, pi_sigma)
        a_t = pi_dist_t.sample()
        logprob_a_t = pi_dist_t.log_prob(a_t).sum(axis=-1)
        return a_t.squeeze(0).cpu().numpy(), logprob_a_t.squeeze(0).cpu().numpy()


class GaussianLearner(types_lib.Learner):
    """PPO-Disagree learner for continuous action spaces (MuJoCo)."""

    def __init__(
        self,
        policy_network: nn.Module,
        policy_optimizer: torch.optim.Optimizer,
        critic_network: nn.Module,
        critic_optimizer: torch.optim.Optimizer,
        disagree_network: nn.Module,
        disagree_optimizer: torch.optim.Optimizer,
        clip_epsilon: LinearSchedule,
        discount: float,
        gae_lambda: float,
        total_unroll_length: int,
        update_k: int,
        ext_coeff: float,
        int_coeff: float,
        entropy_coef: float,
        clip_grad: bool,
        max_grad_norm: float,
        device: torch.device,
        shared_params: dict,
    ) -> None:
        if not 1 <= total_unroll_length:
            raise ValueError(f'Expect total_unroll_length >= 1, got {total_unroll_length}')
        if not 1 <= update_k:
            raise ValueError(f'Expect update_k >= 1, got {update_k}')
        if not 0.0 <= entropy_coef <= 1.0:
            raise ValueError(f'Expect entropy_coef in [0, 1], got {entropy_coef}')

        self.agent_name = 'PPO-Disagree-GaussianLearner'
        self._policy_network = policy_network.to(device=device)
        self._policy_network.train()
        self._policy_optimizer = policy_optimizer

        self._critic_network = critic_network.to(device=device)
        self._critic_network.train()
        self._critic_optimizer = critic_optimizer

        self._disagree_network = disagree_network.to(device=device)
        self._disagree_network.train()
        self._disagree_optimizer = disagree_optimizer

        self._device = device
        self._shared_params = shared_params

        self._int_reward_normalizer = normalizer.TorchRunningMeanStd(shape=(1,), device=self._device)

        self._ext_coeff = ext_coeff
        self._int_coeff = int_coeff

        self._storage = []
        self._total_unroll_length = total_unroll_length
        self._batch_size = min(512, int(np.ceil(total_unroll_length / 4).item()))
        self._update_k = update_k

        self._entropy_coef = entropy_coef
        self._clip_epsilon = clip_epsilon

        self._clip_grad = clip_grad
        self._max_grad_norm = max_grad_norm
        self._discount = discount
        self._gae_lambda = gae_lambda

        self._step_t = -1
        self._update_t = 0
        self._policy_loss_t = np.nan
        self._value_loss_t = np.nan
        self._entropy_loss_t = np.nan
        self._disagree_loss_t = np.nan
        self._intrinsic_reward_t = np.nan

    def step(self) -> Iterable[Mapping[Text, float]]:
        self._step_t += 1
        if len(self._storage) < self._total_unroll_length:
            return
        return self._learn()

    def reset(self) -> None:
        self._storage = []

    def received_item_from_queue(self, unroll_sequences: Iterable[Tuple]) -> None:
        s_t, a_t, logprob_a_t, r_t, s_tp1, done_tp1 = map(list, zip(*unroll_sequences))

        int_r_t = self._compute_intrinsic_rewards(s_t, a_t, s_tp1)
        r_combined = [self._ext_coeff * re + self._int_coeff * ri for re, ri in zip(r_t, int_r_t)]
        returns_t, advantage_t = self._compute_returns_and_advantages(s_t, r_combined, s_tp1, done_tp1)

        zipped_sequence = zip(s_t, a_t, logprob_a_t, returns_t, advantage_t, s_tp1)
        self._storage += zipped_sequence

    def get_policy_state_dict(self):
        return {k: v.cpu() for k, v in self._policy_network.state_dict().items()}

    def _learn(self) -> Iterable[Mapping[Text, float]]:
        num_samples = len(self._storage)

        for _ in range(self._update_k):
            binned_indices = utils.split_indices_into_bins(self._batch_size, num_samples, shuffle=True)
            for indices in binned_indices:
                transitions = [self._storage[i] for i in indices]
                self._update_disagree_network(transitions)
                self._update_policy_network(transitions)
                self._update_value_network(transitions)
                self._update_t += 1
                yield self.statistics

        self._shared_params['policy_network'] = self.get_policy_state_dict()
        del self._storage[:]

    def _update_disagree_network(self, transitions: Iterable[Tuple]) -> None:
        s_t, a_t, _, _, _, s_tp1 = map(list, zip(*transitions))

        s_t = torch.from_numpy(np.stack(s_t, axis=0)).to(device=self._device, dtype=torch.float32)
        a_t = torch.from_numpy(np.stack(a_t, axis=0)).to(device=self._device, dtype=torch.float32)
        s_tp1 = torch.from_numpy(np.stack(s_tp1, axis=0)).to(device=self._device, dtype=torch.float32)

        self._disagree_optimizer.zero_grad()
        output = self._disagree_network(s_t, a_t, s_tp1)

        member_losses = [F.mse_loss(pred, output.features_tp1) for pred in output.predictions]
        loss = torch.stack(member_losses).mean()
        loss.backward()

        if self._clip_grad:
            torch.nn.utils.clip_grad_norm_(
                self._disagree_network.parameters(),
                max_norm=self._max_grad_norm,
                error_if_nonfinite=True,
            )

        self._disagree_optimizer.step()
        self._disagree_loss_t = loss.detach().cpu().item()

        with torch.no_grad():
            preds_stacked = torch.stack(output.predictions, dim=0)
            self._intrinsic_reward_t = preds_stacked.var(dim=0).mean().item()

    def _update_policy_network(self, transitions: Iterable[Tuple]) -> None:
        self._policy_optimizer.zero_grad()
        loss = self._calc_policy_loss(transitions)
        loss.backward()

        if self._clip_grad:
            torch.nn.utils.clip_grad_norm_(
                self._policy_network.parameters(),
                max_norm=self._max_grad_norm,
                error_if_nonfinite=True,
            )

        self._policy_optimizer.step()

    def _update_value_network(self, transitions: Iterable[Tuple]) -> None:
        self._critic_optimizer.zero_grad()
        loss = self._calc_value_loss(transitions)
        loss.backward()

        if self._clip_grad:
            torch.nn.utils.clip_grad_norm_(
                self._critic_network.parameters(),
                max_norm=self._max_grad_norm,
                error_if_nonfinite=True,
            )

        self._critic_optimizer.step()

    def _calc_policy_loss(self, transitions: Iterable[Tuple]) -> torch.Tensor:
        s_t, a_t, logprob_a_t, _, advantage_t, _ = map(list, zip(*transitions))

        s_t = torch.from_numpy(np.stack(s_t, axis=0)).to(device=self._device, dtype=torch.float32)
        a_t = torch.from_numpy(np.stack(a_t, axis=0)).to(device=self._device, dtype=torch.float32)
        behavior_logprob_a_t = torch.from_numpy(np.stack(logprob_a_t, axis=0)).to(device=self._device, dtype=torch.float32)
        advantage_t = torch.from_numpy(np.stack(advantage_t, axis=0)).to(device=self._device, dtype=torch.float32)

        base.assert_rank_and_dtype(s_t, (2, 4), torch.float32)
        base.assert_rank_and_dtype(a_t, 2, torch.float32)
        base.assert_rank_and_dtype(behavior_logprob_a_t, 1, torch.float32)

        pi_mu, pi_sigma = self._policy_network(s_t)
        pi_dist_t = distributions.normal_distribution(pi_mu, pi_sigma)
        entropy_loss = pi_dist_t.entropy()

        pi_logprob_a_t = pi_dist_t.log_prob(a_t).sum(axis=-1)
        ratio = torch.exp(pi_logprob_a_t - behavior_logprob_a_t)

        if ratio.shape != advantage_t.shape:
            raise RuntimeError(f'Expect ratio and advantage_t have same shape, got {ratio.shape} and {advantage_t.shape}')

        policy_loss = rl.clipped_surrogate_gradient_loss(ratio, advantage_t, self.clip_epsilon).loss
        policy_loss = torch.mean(policy_loss)
        entropy_loss = torch.mean(entropy_loss)

        loss = -(policy_loss + self._entropy_coef * entropy_loss)

        self._policy_loss_t = policy_loss.detach().cpu().item()
        self._entropy_loss_t = entropy_loss.detach().cpu().item()

        return loss

    def _calc_value_loss(self, transitions: Iterable[Tuple]) -> torch.Tensor:
        s_t, _, _, returns_t, _, _ = map(list, zip(*transitions))

        s_t = torch.from_numpy(np.stack(s_t, axis=0)).to(device=self._device, dtype=torch.float32)
        returns_t = torch.from_numpy(np.stack(returns_t, axis=0)).to(device=self._device, dtype=torch.float32)

        base.assert_rank_and_dtype(s_t, (2, 4), torch.float32)
        base.assert_rank_and_dtype(returns_t, 1, torch.float32)

        v_t = self._critic_network(s_t).squeeze(-1)
        value_loss = rl.value_loss(returns_t, v_t).loss
        value_loss = torch.mean(value_loss)

        self._value_loss_t = value_loss.detach().cpu().item()
        return value_loss

    @torch.no_grad()
    def _compute_intrinsic_rewards(
        self,
        s_t_list: Iterable[np.ndarray],
        a_t_list: Iterable[np.ndarray],
        s_tp1_list: Iterable[np.ndarray],
    ) -> list:
        s_t = torch.from_numpy(np.stack(s_t_list, axis=0)).to(device=self._device, dtype=torch.float32)
        a_t = torch.from_numpy(np.stack(a_t_list, axis=0)).to(device=self._device, dtype=torch.float32)
        s_tp1 = torch.from_numpy(np.stack(s_tp1_list, axis=0)).to(device=self._device, dtype=torch.float32)

        output = self._disagree_network(s_t, a_t, s_tp1)

        preds_stacked = torch.stack(output.predictions, dim=0)
        int_reward = preds_stacked.var(dim=0).mean(dim=-1)

        self._int_reward_normalizer.update(int_reward.unsqueeze(-1))
        int_reward_normed = self._int_reward_normalizer.normalize(int_reward.unsqueeze(-1)).squeeze(-1)
        int_reward_normed = torch.clamp(int_reward_normed, -10.0, 10.0)

        return int_reward_normed.cpu().numpy().tolist()

    @torch.no_grad()
    def _compute_returns_and_advantages(
        self,
        s_t: Iterable[np.ndarray],
        r_t: Iterable[float],
        s_tp1: Iterable[np.ndarray],
        done_tp1: Iterable[bool],
    ):
        stacked_s_t = torch.from_numpy(np.stack(s_t, axis=0)).to(device=self._device, dtype=torch.float32)
        stacked_r_t = torch.tensor(r_t, dtype=torch.float32, device=self._device)
        stacked_s_tp1 = torch.from_numpy(np.stack(s_tp1, axis=0)).to(device=self._device, dtype=torch.float32)
        stacked_done_tp1 = torch.from_numpy(np.stack(done_tp1, axis=0)).to(device=self._device, dtype=torch.bool)

        discount_tp1 = (~stacked_done_tp1).float() * self._discount

        v_t = self._critic_network(stacked_s_t).squeeze(-1)
        v_tp1 = self._critic_network(stacked_s_tp1).squeeze(-1)

        advantage_t = multistep.truncated_generalized_advantage_estimation(
            stacked_r_t, v_t, v_tp1, discount_tp1, self._gae_lambda
        )
        return_t = advantage_t + v_t

        advantage_t = (advantage_t - advantage_t.mean()) / (advantage_t.std() + 1e-8)

        return return_t.cpu().numpy(), advantage_t.cpu().numpy()

    @property
    def clip_epsilon(self):
        return self._clip_epsilon(self._step_t)

    @property
    def statistics(self) -> Mapping[Text, float]:
        return {
            'policy_loss': self._policy_loss_t,
            'value_loss': self._value_loss_t,
            'entropy_loss': self._entropy_loss_t,
            'disagree_loss': self._disagree_loss_t,
            'intrinsic_reward': self._intrinsic_reward_t,
            'updates': self._update_t,
            'clip_epsilon': self.clip_epsilon,
        }
