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
"""PPO-Noisy agent class.

Learning Progress Monitoring (LPM): a dual-network approach with a forward dynamics
model f_theta and an error prediction model g_phi.  The intrinsic reward is the
improvement signal  r_int = g_phi(s_t, a_t) - log_MSE(f_theta(s_t, a_t), phi(s_{t+1})).
This rewards genuine learning progress and gives ~0 reward for unlearnable noise
(noisy-TV robustness).

From the paper "Beyond Noisy-TVs: Noise-Robust Exploration via Learning Progress Monitoring"
https://arxiv.org/abs/2509.25438

From the paper "Proximal Policy Optimization Algorithms"
https://arxiv.org/abs/1707.06347.
"""
import collections
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


class LpmModuleOutput(NamedTuple):
    """LPM module output (scalars, detached from compute graph)."""

    dynamics_loss: Optional[torch.Tensor]
    error_loss: Optional[torch.Tensor]
    intrinsic_reward_mean: Optional[float]


class Transition(NamedTuple):
    s_t: Optional[np.ndarray]
    a_t: Optional[int]
    logprob_a_t: Optional[float]
    returns_t: Optional[float]    # computed from combined (ext + int) reward
    advantage_t: Optional[float]  # computed from combined (ext + int) reward
    s_tp1: Optional[np.ndarray]   # needed to train the LPM dynamics head


class Actor(types_lib.Agent):
    """PPO-Noisy actor for discrete action spaces."""

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
        self.agent_name = f'PPO-Noisy-actor{rank}'
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
                    self._s_tm1,               # s_t
                    self._a_tm1,               # a_t
                    self._logprob_a_tm1,        # logprob_a_t
                    timestep.reward,            # r_t (extrinsic)
                    timestep.observation,       # s_tp1
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
    """PPO-Noisy learner for discrete action spaces (Atari).

    The LPM network trains a dynamics model f_theta and an error model g_phi.
    Intrinsic reward = g_phi(s_t, a_t) - log_MSE(f_theta, phi(s_{t+1})).
    """

    def __init__(
        self,
        policy_network: nn.Module,
        policy_optimizer: torch.optim.Optimizer,
        lpm_network: nn.Module,
        lpm_optimizer: torch.optim.Optimizer,
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
        error_buffer_size: int,
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

        self.agent_name = 'PPO-Noisy-learner'
        self._policy_network = policy_network.to(device=device)
        self._policy_network.train()
        self._policy_optimizer = policy_optimizer

        self._lpm_network = lpm_network.to(device=device)
        self._lpm_network.train()
        self._lpm_optimizer = lpm_optimizer

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

        self._error_buffer = collections.deque(maxlen=error_buffer_size)

        self._step_t = -1
        self._update_t = 0
        self._policy_loss_t = np.nan
        self._value_loss_t = np.nan
        self._entropy_loss_t = np.nan
        self._dynamics_loss_t = np.nan
        self._error_loss_t = np.nan
        self._intrinsic_reward_t = np.nan

    def step(self) -> Iterable[Mapping[Text, float]]:
        """Increment learner step, and potentially do a update when called."""
        self._step_t += 1

        if len(self._storage) < self._total_unroll_length:
            return

        return self._learn()

    def reset(self) -> None:
        """Should be called at the beginning of every iteration."""
        self._storage = []

    def received_item_from_queue(self, unroll_sequences: Iterable[Tuple]) -> None:
        """Received item sent by actors through multiprocessing queue."""
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

                self._update_lpm_network(stacked)
                self._update_policy_network(stacked)
                self._update_t += 1
                yield self.statistics

        self._shared_params['policy_network'] = self.get_policy_state_dict()
        del self._storage[:]

    def _update_lpm_network(self, transitions: Transition) -> None:
        """Train dynamics head on current rollout; train error head on error buffer samples."""
        s_t = torch.from_numpy(transitions.s_t).to(device=self._device, dtype=torch.float32)
        a_t = torch.from_numpy(transitions.a_t).to(device=self._device, dtype=torch.int64)
        s_tp1 = torch.from_numpy(transitions.s_tp1).to(device=self._device, dtype=torch.float32)

        self._lpm_optimizer.zero_grad()

        features_t = self._lpm_network.encode(s_t)
        pred_features_tp1 = self._lpm_network.predict_next_features(features_t, a_t)
        features_tp1 = self._lpm_network.encode(s_tp1).detach()
        dynamics_loss = F.mse_loss(pred_features_tp1, features_tp1)

        error_loss = torch.tensor(0.0, device=self._device)
        if len(self._error_buffer) >= self._batch_size:
            buf_indices = np.random.randint(0, len(self._error_buffer), size=self._batch_size)
            buf_data = [self._error_buffer[i] for i in buf_indices]
            buf_s_t, buf_a_t, buf_log_mse = zip(*buf_data)

            buf_s_t = torch.from_numpy(np.stack(buf_s_t, axis=0)).to(device=self._device, dtype=torch.float32)
            buf_a_t = torch.from_numpy(np.array(buf_a_t, dtype=np.int64)).to(device=self._device)
            buf_log_mse = torch.tensor(buf_log_mse, dtype=torch.float32, device=self._device)

            buf_features_t = self._lpm_network.encode(buf_s_t)
            predicted_log_error = self._lpm_network.predict_log_error(buf_features_t, buf_a_t).squeeze(-1)
            error_loss = F.mse_loss(predicted_log_error, buf_log_mse)

        total_loss = dynamics_loss + error_loss
        total_loss.backward()

        if self._clip_grad:
            torch.nn.utils.clip_grad_norm_(
                self._lpm_network.parameters(),
                max_norm=self._max_grad_norm,
                error_if_nonfinite=True,
            )

        self._lpm_optimizer.step()
        self._dynamics_loss_t = dynamics_loss.detach().cpu().item()
        self._error_loss_t = error_loss.detach().cpu().item()

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
    def _compute_intrinsic_rewards(self, s_t_list, a_t_list, s_tp1_list) -> list:
        """Compute per-step LPM intrinsic rewards.

        r_int = g_phi(s_t, a_t) - log_MSE(f_theta(s_t, a_t), phi(s_{t+1}))
        """
        s_t = torch.from_numpy(np.stack(s_t_list, axis=0)).to(device=self._device, dtype=torch.float32)
        a_t = torch.from_numpy(np.array(a_t_list, dtype=np.int64)).to(device=self._device)
        s_tp1 = torch.from_numpy(np.stack(s_tp1_list, axis=0)).to(device=self._device, dtype=torch.float32)

        features_t = self._lpm_network.encode(s_t)
        pred_features_tp1 = self._lpm_network.predict_next_features(features_t, a_t)
        features_tp1 = self._lpm_network.encode(s_tp1)

        mse_per_sample = F.mse_loss(pred_features_tp1, features_tp1, reduction='none').mean(dim=-1)
        log_mse = torch.log(mse_per_sample + 1e-8)

        predicted_log_error = self._lpm_network.predict_log_error(features_t, a_t).squeeze(-1)
        r_int = predicted_log_error - log_mse

        self._int_reward_normalizer.update(r_int.unsqueeze(-1))
        r_int_normed = self._int_reward_normalizer.normalize(r_int.unsqueeze(-1)).squeeze(-1)
        r_int_normed = torch.clamp(r_int_normed, -10.0, 10.0)

        self._intrinsic_reward_t = r_int.mean().item()

        s_t_np = np.stack(s_t_list, axis=0)
        a_t_np = np.array(a_t_list, dtype=np.int64)
        log_mse_np = log_mse.cpu().numpy()
        for i in range(len(s_t_list)):
            self._error_buffer.append((s_t_np[i], a_t_np[i], float(log_mse_np[i])))

        return r_int_normed.cpu().numpy().tolist()

    @torch.no_grad()
    def _compute_returns_and_advantages(self, s_t, r_t, s_tp1, done_tp1):
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
            'dynamics_loss': self._dynamics_loss_t,
            'error_loss': self._error_loss_t,
            'intrinsic_reward': self._intrinsic_reward_t,
            'updates': self._update_t,
            'clip_epsilon': self.clip_epsilon,
        }


class GaussianActor(Actor):
    """PPO-Noisy actor for continuous action spaces."""

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
        self.agent_name = f'PPO-Noisy-Gaussian-actor{rank}'

    @torch.no_grad()
    def _choose_action(self, timestep: types_lib.TimeStep) -> Tuple[np.ndarray]:
        s_t = torch.from_numpy(timestep.observation[None, ...]).to(device=self._device, dtype=torch.float32)
        pi_mu, pi_sigma = self._policy_network(s_t)
        pi_dist_t = distributions.normal_distribution(pi_mu, pi_sigma)
        a_t = pi_dist_t.sample()
        logprob_a_t = pi_dist_t.log_prob(a_t).sum(axis=-1)
        return a_t.squeeze(0).cpu().numpy(), logprob_a_t.squeeze(0).cpu().numpy()


class GaussianLearner(types_lib.Learner):
    """PPO-Noisy learner for continuous action spaces (MuJoCo).

    Uses separate policy and critic networks.
    """

    def __init__(
        self,
        policy_network: nn.Module,
        policy_optimizer: torch.optim.Optimizer,
        critic_network: nn.Module,
        critic_optimizer: torch.optim.Optimizer,
        lpm_network: nn.Module,
        lpm_optimizer: torch.optim.Optimizer,
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
        error_buffer_size: int,
        device: torch.device,
        shared_params: dict,
    ) -> None:
        if not 1 <= total_unroll_length:
            raise ValueError(f'Expect total_unroll_length >= 1, got {total_unroll_length}')
        if not 1 <= update_k:
            raise ValueError(f'Expect update_k >= 1, got {update_k}')
        if not 0.0 <= entropy_coef <= 1.0:
            raise ValueError(f'Expect entropy_coef in [0, 1], got {entropy_coef}')

        self.agent_name = 'PPO-Noisy-GaussianLearner'
        self._policy_network = policy_network.to(device=device)
        self._policy_network.train()
        self._policy_optimizer = policy_optimizer

        self._critic_network = critic_network.to(device=device)
        self._critic_network.train()
        self._critic_optimizer = critic_optimizer

        self._lpm_network = lpm_network.to(device=device)
        self._lpm_network.train()
        self._lpm_optimizer = lpm_optimizer

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

        self._error_buffer = collections.deque(maxlen=error_buffer_size)

        self._step_t = -1
        self._update_t = 0
        self._policy_loss_t = np.nan
        self._value_loss_t = np.nan
        self._entropy_loss_t = np.nan
        self._dynamics_loss_t = np.nan
        self._error_loss_t = np.nan
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
                self._update_lpm_network(transitions)
                self._update_policy_network(transitions)
                self._update_value_network(transitions)
                self._update_t += 1
                yield self.statistics

        self._shared_params['policy_network'] = self.get_policy_state_dict()
        del self._storage[:]

    def _update_lpm_network(self, transitions: Iterable[Tuple]) -> None:
        s_t, a_t, _, _, _, s_tp1 = map(list, zip(*transitions))

        s_t_t = torch.from_numpy(np.stack(s_t, axis=0)).to(device=self._device, dtype=torch.float32)
        a_t_t = torch.from_numpy(np.stack(a_t, axis=0)).to(device=self._device, dtype=torch.float32)
        s_tp1_t = torch.from_numpy(np.stack(s_tp1, axis=0)).to(device=self._device, dtype=torch.float32)

        self._lpm_optimizer.zero_grad()

        features_t = self._lpm_network.encode(s_t_t)
        pred_features_tp1 = self._lpm_network.predict_next_features(features_t, a_t_t)
        features_tp1 = self._lpm_network.encode(s_tp1_t).detach()
        dynamics_loss = F.mse_loss(pred_features_tp1, features_tp1)

        error_loss = torch.tensor(0.0, device=self._device)
        batch_size = len(transitions)
        if len(self._error_buffer) >= batch_size:
            buf_indices = np.random.randint(0, len(self._error_buffer), size=batch_size)
            buf_data = [self._error_buffer[i] for i in buf_indices]
            buf_s_t, buf_a_t, buf_log_mse = zip(*buf_data)

            buf_s_t_t = torch.from_numpy(np.stack(buf_s_t, axis=0)).to(device=self._device, dtype=torch.float32)
            buf_a_t_t = torch.from_numpy(np.stack(buf_a_t, axis=0)).to(device=self._device, dtype=torch.float32)
            buf_log_mse_t = torch.tensor(buf_log_mse, dtype=torch.float32, device=self._device)

            buf_features_t = self._lpm_network.encode(buf_s_t_t)
            predicted_log_error = self._lpm_network.predict_log_error(buf_features_t, buf_a_t_t).squeeze(-1)
            error_loss = F.mse_loss(predicted_log_error, buf_log_mse_t)

        total_loss = dynamics_loss + error_loss
        total_loss.backward()

        if self._clip_grad:
            torch.nn.utils.clip_grad_norm_(
                self._lpm_network.parameters(),
                max_norm=self._max_grad_norm,
                error_if_nonfinite=True,
            )

        self._lpm_optimizer.step()
        self._dynamics_loss_t = dynamics_loss.detach().cpu().item()
        self._error_loss_t = error_loss.detach().cpu().item()

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
        behavior_logprob_a_t = torch.from_numpy(np.stack(logprob_a_t, axis=0)).to(
            device=self._device, dtype=torch.float32
        )
        advantage_t = torch.from_numpy(np.stack(advantage_t, axis=0)).to(device=self._device, dtype=torch.float32)

        base.assert_rank_and_dtype(s_t, (2, 4), torch.float32)
        base.assert_rank_and_dtype(a_t, 2, torch.float32)
        base.assert_rank_and_dtype(behavior_logprob_a_t, 1, torch.float32)

        pi_mu, pi_sigma = self._policy_network(s_t)
        pi_dist_t = distributions.normal_distribution(pi_mu, pi_sigma)
        entropy_loss = pi_dist_t.entropy()

        pi_logprob_a_t = pi_dist_t.log_prob(a_t).sum(axis=-1)
        ratio = torch.exp(pi_logprob_a_t - behavior_logprob_a_t)

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
    def _compute_intrinsic_rewards(self, s_t_list, a_t_list, s_tp1_list) -> list:
        s_t = torch.from_numpy(np.stack(s_t_list, axis=0)).to(device=self._device, dtype=torch.float32)
        a_t = torch.from_numpy(np.stack(a_t_list, axis=0)).to(device=self._device, dtype=torch.float32)
        s_tp1 = torch.from_numpy(np.stack(s_tp1_list, axis=0)).to(device=self._device, dtype=torch.float32)

        features_t = self._lpm_network.encode(s_t)
        pred_features_tp1 = self._lpm_network.predict_next_features(features_t, a_t)
        features_tp1 = self._lpm_network.encode(s_tp1)

        mse_per_sample = F.mse_loss(pred_features_tp1, features_tp1, reduction='none').mean(dim=-1)
        log_mse = torch.log(mse_per_sample + 1e-8)

        predicted_log_error = self._lpm_network.predict_log_error(features_t, a_t).squeeze(-1)
        r_int = predicted_log_error - log_mse

        self._int_reward_normalizer.update(r_int.unsqueeze(-1))
        r_int_normed = self._int_reward_normalizer.normalize(r_int.unsqueeze(-1)).squeeze(-1)
        r_int_normed = torch.clamp(r_int_normed, -10.0, 10.0)

        self._intrinsic_reward_t = r_int.mean().item()

        s_t_np = np.stack(s_t_list, axis=0)
        a_t_np = np.stack(a_t_list, axis=0)
        log_mse_np = log_mse.cpu().numpy()
        for i in range(len(s_t_list)):
            self._error_buffer.append((s_t_np[i], a_t_np[i], float(log_mse_np[i])))

        return r_int_normed.cpu().numpy().tolist()

    @torch.no_grad()
    def _compute_returns_and_advantages(self, s_t, r_t, s_tp1, done_tp1):
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
            'dynamics_loss': self._dynamics_loss_t,
            'error_loss': self._error_loss_t,
            'intrinsic_reward': self._intrinsic_reward_t,
            'updates': self._update_t,
            'clip_epsilon': self.clip_epsilon,
        }
