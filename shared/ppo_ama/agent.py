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
"""PPO-AMA agent class.

AMA (Aleatoric Modelling for Agents) extends ICM by replacing the deterministic
forward model with a probabilistic one that predicts both mean and log-variance
of next-state features. The aleatoric (irreducible) uncertainty is subtracted
from the prediction error to compute the intrinsic reward, which avoids giving
high rewards for stochastic but uncontrollable observations (e.g. noisy TVs).

Intrinsic reward = max(0, mse - variance)
                 = max(0, ||phi(s_tp1) - mu||^2 - exp(log_var))

From the paper "How to Stay Curious while Avoiding Noisy TVs"

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

torch.autograd.set_detect_anomaly(True)


class AmaModuleOutput(NamedTuple):
    """AMA module output"""

    inverse_loss: Optional[torch.Tensor]
    forward_nll_loss: Optional[torch.Tensor]
    intrinsic_reward: Optional[torch.Tensor]


class Transition(NamedTuple):
    s_t: Optional[np.ndarray]
    a_t: Optional[int]
    logprob_a_t: Optional[float]
    returns_t: Optional[float]
    advantage_t: Optional[float]
    s_tp1: Optional[np.ndarray]


class Actor(types_lib.Agent):
    """PPO-AMA actor"""

    def __init__(
        self,
        rank: int,
        data_queue: multiprocessing.Queue,
        policy_network: torch.nn.Module,
        unroll_length: int,
        device: torch.device,
        shared_params: dict,
    ) -> None:
        """
        Args:
            rank: the rank for the actor.
            data_queue: a multiprocessing.Queue to send collected transitions to learner process.
            policy_network: the policy network for worker to make action choice.
            unroll_length: rollout length.
            device: PyTorch runtime device.
            shared_params: a shared dict, so we can later update the parameters for actors.
        """
        if not 1 <= unroll_length:
            raise ValueError(f'Expect unroll_length to be integer greater than or equal to 1, got {unroll_length}')

        self.rank = rank
        self.agent_name = f'PPO-AMA-actor{rank}'
        self._queue = data_queue
        self._policy_network = policy_network.to(device=device)
        # Disable autograd for actor networks.
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
        """Given current timestep, return action a_t, and push transition into global queue"""
        self._step_t += 1

        a_t, logprob_a_t = self.act(timestep)

        if self._a_tm1 is not None:
            self._unroll_sequence.append(
                (
                    self._s_tm1,        # s_t
                    self._a_tm1,        # a_t
                    self._logprob_a_tm1,  # logprob_a_t
                    timestep.reward,    # r_t
                    timestep.observation,  # s_tp1
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
        """Given timestep, choose action a_t"""
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
    """PPO-AMA learner"""

    def __init__(
        self,
        policy_network: nn.Module,
        policy_optimizer: torch.optim.Optimizer,
        ama_network: nn.Module,
        ama_optimizer: torch.optim.Optimizer,
        clip_epsilon: LinearSchedule,
        discount: float,
        gae_lambda: float,
        total_unroll_length: int,
        update_k: int,
        intrinsic_lambda: float,
        ama_beta: float,
        policy_loss_coef: float,
        entropy_coef: float,
        value_coef: float,
        clip_grad: bool,
        max_grad_norm: float,
        device: torch.device,
        shared_params: dict,
    ) -> None:
        """
        Args:
            policy_network: the policy network we want to train.
            policy_optimizer: the optimizer for policy network.
            ama_network: the AMA module network (aleatoric forward model + inverse model).
            ama_optimizer: the optimizer for AMA module network.
            clip_epsilon: external scheduler to decay clip epsilon.
            discount: the gamma discount for future rewards.
            gae_lambda: lambda for the GAE general advantage estimator.
            total_unroll_length: wait until collects this samples before update networks.
            update_k: update k times when it's time to do learning.
            intrinsic_lambda: scaling factor for intrinsic reward.
            ama_beta: weights inverse model loss against the forward NLL loss.
            policy_loss_coef: weights policy loss against the AMA module loss.
            entropy_coef: the coefficient of entropy loss.
            value_coef: the coefficient of state-value loss.
            clip_grad: if True, clip gradients norm.
            max_grad_norm: the maximum gradient norm for clip grad.
            device: PyTorch runtime device.
            shared_params: a shared dict, so we can later update the parameters for actors.
        """
        if not 1 <= total_unroll_length:
            raise ValueError(f'Expect total_unroll_length to be greater than 1, got {total_unroll_length}')
        if not 0.0 <= discount <= 1.0:
            raise ValueError(f'Expect discount to in the range [0.0, 1.0], got {discount}')
        if not 1 <= update_k:
            raise ValueError(f'Expect update_k to be integer greater than or equal to 1, got {update_k}')
        if not 0.0 <= intrinsic_lambda:
            raise ValueError(f'Expect intrinsic_lambda to be greater than or equal to 0.0, got {intrinsic_lambda}')
        if not 0.0 <= ama_beta <= 1.0:
            raise ValueError(f'Expect ama_beta to in the range [0.0, 1.0], got {ama_beta}')
        if not 0.0 <= policy_loss_coef <= 1.0:
            raise ValueError(f'Expect policy_loss_coef to in the range [0.0, 1.0], got {policy_loss_coef}')
        if not 0.0 <= entropy_coef <= 1.0:
            raise ValueError(f'Expect entropy_coef to [0.0, 1.0], got {entropy_coef}')
        if not 0.0 <= value_coef <= 1.0:
            raise ValueError(f'Expect value_coef to [0.0, 1.0], got {value_coef}')

        self.agent_name = 'PPO-AMA-learner'
        self._policy_network = policy_network.to(device=device)
        self._policy_optimizer = policy_optimizer
        self._ama_network = ama_network.to(device=device)
        self._ama_optimizer = ama_optimizer
        self._device = device

        self._shared_params = shared_params

        # Running normalizer for intrinsic rewards
        self._int_reward_normalizer = normalizer.TorchRunningMeanStd(shape=(1,), device=self._device)

        self._intrinsic_lambda = intrinsic_lambda
        self._ama_beta = ama_beta
        self._policy_loss_coef = policy_loss_coef

        self._storage = []
        self._total_unroll_length = total_unroll_length
        self._batch_size = min(512, int(np.ceil(total_unroll_length / 4).item()))
        self._update_k = update_k

        self._entropy_coef = entropy_coef
        self._value_coef = value_coef
        self._clip_epsilon = clip_epsilon

        self._clip_grad = clip_grad
        self._max_grad_norm = max_grad_norm
        self._discount = discount
        self._gae_lambda = gae_lambda

        # Counters and logging
        self._step_t = -1
        self._update_t = 0
        self._policy_loss_t = np.nan
        self._value_loss_t = np.nan
        self._entropy_loss_t = np.nan
        self._ama_inverse_loss_t = np.nan
        self._ama_forward_nll_loss_t = np.nan
        self.intrinsic_reward_t = torch.zeros(1)

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
        """Received item send by actors through multiprocessing queue."""
        s_t, a_t, logprob_a_t, r_t, s_tp1, done_tp1 = map(list, zip(*unroll_sequences))

        returns_t, advantage_t = self._compute_returns_and_advantages(s_t, r_t, s_tp1, done_tp1)

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
                stacked_transitions = Transition(
                    s_t=np.stack(s_t, axis=0),
                    a_t=np.stack(a_t, axis=0),
                    logprob_a_t=np.stack(logprob_a_t, axis=0),
                    returns_t=np.stack(returns_t, axis=0),
                    advantage_t=np.stack(advantage_t, axis=0),
                    s_tp1=np.stack(s_tp1, axis=0),
                )

                ama_output = self._update_ama_network(stacked_transitions)
                self._update_policy_network(stacked_transitions, ama_output)
                self._update_t += 1
                yield self.statistics

        self._shared_params['policy_network'] = self.get_policy_state_dict()
        del self._storage[:]

    def _update_ama_network(self, transitions: Transition) -> AmaModuleOutput:
        self._ama_optimizer.zero_grad()
        loss, ama_output = self._calc_ama_loss(transitions=transitions)
        loss.backward()

        if self._clip_grad:
            torch.nn.utils.clip_grad_norm_(
                self._ama_network.parameters(),
                max_norm=self._max_grad_norm,
                error_if_nonfinite=True,
            )

        self._ama_optimizer.step()
        return ama_output

    def _update_policy_network(self, transitions: Transition, ama_output: AmaModuleOutput) -> None:
        self._policy_optimizer.zero_grad()
        loss = self._calc_policy_loss(transitions, ama_output)
        loss.backward()

        if self._clip_grad:
            torch.nn.utils.clip_grad_norm_(
                self._policy_network.parameters(),
                max_norm=self._max_grad_norm,
                error_if_nonfinite=True,
            )

        self._policy_optimizer.step()

    def _calc_ama_loss(self, transitions: Transition) -> Tuple[torch.Tensor, AmaModuleOutput]:
        s_t = torch.from_numpy(transitions.s_t).to(device=self._device, dtype=torch.float32)
        a_t = torch.from_numpy(transitions.a_t).to(device=self._device, dtype=torch.int64)
        s_tp1 = torch.from_numpy(transitions.s_tp1).to(device=self._device, dtype=torch.float32)

        base.assert_rank_and_dtype(s_t, (2, 4), torch.float32)
        base.assert_rank_and_dtype(s_tp1, (2, 4), torch.float32)
        base.assert_rank_and_dtype(a_t, 1, torch.long)

        ama_output = self._ama_network(s_t, a_t, s_tp1)
        pred_mu = ama_output.pred_features_mu          # [batch, feature_dim]
        pred_log_var = ama_output.pred_features_log_var  # [batch, feature_dim]
        features_tp1 = ama_output.features              # [batch, feature_dim]
        pred_pi_logits_a_t = ama_output.pi_logits       # [batch, action_dim]

        # Inverse model loss: predict a_t from (s_t, s_tp1) features
        inverse_losses = F.cross_entropy(pred_pi_logits_a_t, a_t, reduction='none')  # [batch]

        # Forward model NLL loss: -log N(features_tp1; pred_mu, exp(pred_log_var))
        # = 0.5 * exp(-log_var) * (mu - target)^2 + 0.5 * log_var
        mse_per_dim = torch.square(pred_mu - features_tp1.detach())  # [batch, feature_dim]
        forward_nll_per_dim = 0.5 * torch.exp(-pred_log_var) * mse_per_dim + 0.5 * pred_log_var
        forward_nll_losses = forward_nll_per_dim.mean(dim=1)  # [batch]

        # AMA intrinsic reward: prediction error minus aleatoric uncertainty
        mse_per_sample = mse_per_dim.mean(dim=1).clone().detach()          # [batch]
        variance_per_sample = torch.exp(pred_log_var).mean(dim=1).detach()  # [batch]
        intrinsic_reward = self._intrinsic_lambda * torch.clamp(
            mse_per_sample - variance_per_sample, min=0.0
        )  # [batch]

        # Normalize intrinsic reward
        self._int_reward_normalizer.update(intrinsic_reward)
        intrinsic_reward = self._int_reward_normalizer.normalize(intrinsic_reward)
        intrinsic_reward = torch.clamp(intrinsic_reward, -10, 10)

        self.intrinsic_reward_t = intrinsic_reward.detach()

        inverse_loss = inverse_losses.mean()
        forward_nll_loss = forward_nll_losses.mean()
        ama_loss = inverse_loss + forward_nll_loss

        self._ama_inverse_loss_t = inverse_loss.detach().cpu().item()
        self._ama_forward_nll_loss_t = forward_nll_loss.detach().cpu().item()

        return ama_loss, AmaModuleOutput(
            inverse_loss=inverse_loss.detach(),
            forward_nll_loss=forward_nll_loss.detach(),
            intrinsic_reward=intrinsic_reward,
        )

    def _calc_policy_loss(self, transitions: Transition, ama_output: AmaModuleOutput) -> torch.Tensor:
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

        ama_inverse_loss = ama_output.inverse_loss
        ama_forward_nll_loss = ama_output.forward_nll_loss
        ama_intrinsic_reward = ama_output.intrinsic_reward

        if ama_inverse_loss.requires_grad or ama_forward_nll_loss.requires_grad or ama_intrinsic_reward.requires_grad:
            raise RuntimeError('Expect tensors from AMA module do not require gradients')

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

        policy_loss = torch.mean(policy_loss, dim=0)
        entropy_loss = torch.mean(entropy_loss, dim=0)
        value_loss = torch.mean(value_loss, dim=0)

        loss = -(policy_loss + self._entropy_coef * entropy_loss) + self._value_coef * value_loss

        # Re-weight policy loss, add AMA inverse and forward NLL losses
        loss = self._policy_loss_coef * loss + (1.0 - self._ama_beta) * ama_inverse_loss + self._ama_beta * ama_forward_nll_loss

        self._policy_loss_t = policy_loss.detach().cpu().item()
        self._value_loss_t = value_loss.detach().cpu().item()
        self._entropy_loss_t = entropy_loss.detach().cpu().item()

        return loss

    @torch.no_grad()
    def _compute_returns_and_advantages(
        self,
        s_t: Iterable[np.ndarray],
        r_t: Iterable[float],
        s_tp1: Iterable[np.ndarray],
        done_tp1: Iterable[bool],
    ):
        """Compute returns, GAE estimated advantages"""
        stacked_s_t = torch.from_numpy(np.stack(s_t, axis=0)).to(device=self._device, dtype=torch.float32)
        stacked_r_t = torch.from_numpy(np.stack(r_t, axis=0)).to(device=self._device, dtype=torch.float32)
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
        """Call external clip epsilon scheduler"""
        return self._clip_epsilon(self._step_t)

    @property
    def statistics(self) -> Mapping[Text, float]:
        """Returns current agent statistics as a dictionary."""
        return {
            'policy_loss': self._policy_loss_t,
            'value_loss': self._value_loss_t,
            'entropy_loss': self._entropy_loss_t,
            'ama_inverse_loss': self._ama_inverse_loss_t,
            'ama_forward_nll_loss': self._ama_forward_nll_loss_t,
            'updates': self._update_t,
            'clip_epsilon': self.clip_epsilon,
            'intrinsic_reward': self.intrinsic_reward_t,
        }


class GaussianTransition(NamedTuple):
    s_t: Optional[np.ndarray]
    a_t: Optional[np.ndarray]
    logprob_a_t: Optional[float]
    returns_t: Optional[float]
    advantage_t: Optional[float]
    s_tp1: Optional[np.ndarray]


class GaussianActor(Actor):
    """PPO-AMA Gaussian actor for continuous action space."""

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
        self.agent_name = f'PPO-AMA-Gaussian-actor{rank}'

    @torch.no_grad()
    def _choose_action(self, timestep: types_lib.TimeStep) -> Tuple[np.ndarray]:
        """Given timestep, choose continuous action a_t using Gaussian policy."""
        s_t = torch.from_numpy(timestep.observation[None, ...]).to(device=self._device, dtype=torch.float32)
        pi_mu, pi_sigma = self._policy_network(s_t)
        pi_dist_t = distributions.normal_distribution(pi_mu, pi_sigma)
        a_t = pi_dist_t.sample()
        logprob_a_t = pi_dist_t.log_prob(a_t).sum(axis=-1)
        return a_t.squeeze(0).cpu().numpy(), logprob_a_t.squeeze(0).cpu().numpy()


class GaussianLearner(types_lib.Learner):
    """PPO-AMA Gaussian learner for continuous action space."""

    def __init__(
        self,
        policy_network: nn.Module,
        policy_optimizer: torch.optim.Optimizer,
        critic_network: nn.Module,
        critic_optimizer: torch.optim.Optimizer,
        ama_network: nn.Module,
        ama_optimizer: torch.optim.Optimizer,
        clip_epsilon: LinearSchedule,
        discount: float,
        gae_lambda: float,
        total_unroll_length: int,
        update_k: int,
        intrinsic_lambda: float,
        ama_beta: float,
        policy_loss_coef: float,
        entropy_coef: float,
        clip_grad: bool,
        max_grad_norm: float,
        device: torch.device,
        shared_params: dict,
    ) -> None:
        if not 1 <= total_unroll_length:
            raise ValueError(f'Expect total_unroll_length to be greater than 1, got {total_unroll_length}')
        if not 0.0 <= discount <= 1.0:
            raise ValueError(f'Expect discount in [0.0, 1.0], got {discount}')
        if not 1 <= update_k:
            raise ValueError(f'Expect update_k >= 1, got {update_k}')
        if not 0.0 <= intrinsic_lambda:
            raise ValueError(f'Expect intrinsic_lambda >= 0.0, got {intrinsic_lambda}')
        if not 0.0 <= ama_beta <= 1.0:
            raise ValueError(f'Expect ama_beta in [0.0, 1.0], got {ama_beta}')
        if not 0.0 <= policy_loss_coef <= 1.0:
            raise ValueError(f'Expect policy_loss_coef in [0.0, 1.0], got {policy_loss_coef}')
        if not 0.0 <= entropy_coef <= 1.0:
            raise ValueError(f'Expect entropy_coef in [0.0, 1.0], got {entropy_coef}')

        self.agent_name = 'PPO-AMA-GaussianLearner'
        self._policy_network = policy_network.to(device=device)
        self._policy_network.train()
        self._policy_optimizer = policy_optimizer

        self._critic_network = critic_network.to(device=device)
        self._critic_optimizer = critic_optimizer

        self._ama_network = ama_network.to(device=device)
        self._ama_optimizer = ama_optimizer
        self._device = device

        self._shared_params = shared_params

        self._int_reward_normalizer = normalizer.TorchRunningMeanStd(shape=(1,), device=self._device)

        self._intrinsic_lambda = intrinsic_lambda
        self._ama_beta = ama_beta
        self._policy_loss_coef = policy_loss_coef

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
        self._ama_inverse_loss_t = np.nan
        self._ama_forward_nll_loss_t = np.nan
        self.intrinsic_reward_t = torch.zeros(1)

    def step(self) -> Iterable[Mapping[Text, float]]:
        self._step_t += 1
        if len(self._storage) < self._total_unroll_length:
            return
        return self._learn()

    def reset(self) -> None:
        self._storage = []

    def received_item_from_queue(self, unroll_sequences: Iterable[Tuple]) -> None:
        s_t, a_t, logprob_a_t, r_t, s_tp1, done_tp1 = map(list, zip(*unroll_sequences))
        returns_t, advantage_t = self._compute_returns_and_advantages(s_t, r_t, s_tp1, done_tp1)
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
                stacked_transitions = GaussianTransition(
                    s_t=np.stack(s_t, axis=0),
                    a_t=np.stack(a_t, axis=0),
                    logprob_a_t=np.stack(logprob_a_t, axis=0),
                    returns_t=np.stack(returns_t, axis=0),
                    advantage_t=np.stack(advantage_t, axis=0),
                    s_tp1=np.stack(s_tp1, axis=0),
                )

                ama_output = self._update_ama_network(stacked_transitions)
                self._update_policy_network(stacked_transitions, ama_output)
                self._update_value_network(stacked_transitions)
                self._update_t += 1
                yield self.statistics

        self._shared_params['policy_network'] = self.get_policy_state_dict()
        del self._storage[:]

    def _update_ama_network(self, transitions: GaussianTransition) -> AmaModuleOutput:
        self._ama_optimizer.zero_grad()
        loss, ama_output = self._calc_ama_loss(transitions=transitions)
        loss.backward()

        if self._clip_grad:
            torch.nn.utils.clip_grad_norm_(
                self._ama_network.parameters(),
                max_norm=self._max_grad_norm,
                error_if_nonfinite=True,
            )

        self._ama_optimizer.step()
        return ama_output

    def _update_policy_network(self, transitions: GaussianTransition, ama_output: AmaModuleOutput) -> None:
        self._policy_optimizer.zero_grad()
        loss = self._calc_policy_loss(transitions, ama_output)
        loss.backward()

        if self._clip_grad:
            torch.nn.utils.clip_grad_norm_(
                self._policy_network.parameters(),
                max_norm=self._max_grad_norm,
                error_if_nonfinite=True,
            )

        self._policy_optimizer.step()

    def _update_value_network(self, transitions: GaussianTransition) -> None:
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

    def _calc_ama_loss(self, transitions: GaussianTransition) -> Tuple[torch.Tensor, AmaModuleOutput]:
        s_t = torch.from_numpy(transitions.s_t).to(device=self._device, dtype=torch.float32)
        a_t = torch.from_numpy(transitions.a_t).to(device=self._device, dtype=torch.float32)
        s_tp1 = torch.from_numpy(transitions.s_tp1).to(device=self._device, dtype=torch.float32)

        base.assert_rank_and_dtype(s_t, 2, torch.float32)
        base.assert_rank_and_dtype(s_tp1, 2, torch.float32)
        base.assert_rank_and_dtype(a_t, 2, torch.float32)

        ama_output = self._ama_network(s_t, a_t, s_tp1)
        pred_mu = ama_output.pred_features_mu
        pred_log_var = ama_output.pred_features_log_var
        features_tp1 = ama_output.features
        pred_action = ama_output.pred_action

        inverse_losses = F.mse_loss(pred_action, a_t, reduction='none').mean(dim=-1)  # [batch]

        mse_per_dim = torch.square(pred_mu - features_tp1.detach())
        forward_nll_per_dim = 0.5 * torch.exp(-pred_log_var) * mse_per_dim + 0.5 * pred_log_var
        forward_nll_losses = forward_nll_per_dim.mean(dim=1)

        mse_per_sample = mse_per_dim.mean(dim=1).clone().detach()
        variance_per_sample = torch.exp(pred_log_var).mean(dim=1).detach()
        intrinsic_reward = self._intrinsic_lambda * torch.clamp(
            mse_per_sample - variance_per_sample, min=0.0
        )

        self._int_reward_normalizer.update(intrinsic_reward)
        intrinsic_reward = self._int_reward_normalizer.normalize(intrinsic_reward)
        intrinsic_reward = torch.clamp(intrinsic_reward, -10, 10)

        self.intrinsic_reward_t = intrinsic_reward.detach()

        inverse_loss = inverse_losses.mean()
        forward_nll_loss = forward_nll_losses.mean()
        ama_loss = inverse_loss + forward_nll_loss

        self._ama_inverse_loss_t = inverse_loss.detach().cpu().item()
        self._ama_forward_nll_loss_t = forward_nll_loss.detach().cpu().item()

        return ama_loss, AmaModuleOutput(
            inverse_loss=inverse_loss.detach(),
            forward_nll_loss=forward_nll_loss.detach(),
            intrinsic_reward=intrinsic_reward,
        )

    def _calc_policy_loss(self, transitions: GaussianTransition, ama_output: AmaModuleOutput) -> torch.Tensor:
        s_t = torch.from_numpy(transitions.s_t).to(device=self._device, dtype=torch.float32)
        a_t = torch.from_numpy(transitions.a_t).to(device=self._device, dtype=torch.float32)
        behavior_logprob_a_t = torch.from_numpy(transitions.logprob_a_t).to(device=self._device, dtype=torch.float32)
        advantage_t = torch.from_numpy(transitions.advantage_t).to(device=self._device, dtype=torch.float32)

        base.assert_rank_and_dtype(s_t, 2, torch.float32)
        base.assert_rank_and_dtype(a_t, 2, torch.float32)
        base.assert_rank_and_dtype(behavior_logprob_a_t, 1, torch.float32)
        base.assert_rank_and_dtype(advantage_t, 1, torch.float32)

        ama_inverse_loss = ama_output.inverse_loss
        ama_forward_nll_loss = ama_output.forward_nll_loss
        ama_intrinsic_reward = ama_output.intrinsic_reward

        if ama_inverse_loss.requires_grad or ama_forward_nll_loss.requires_grad or ama_intrinsic_reward.requires_grad:
            raise RuntimeError('Expect tensors from AMA module do not require gradients')

        pi_mu, pi_sigma = self._policy_network(s_t)
        pi_dist_t = distributions.normal_distribution(pi_mu, pi_sigma)

        entropy_loss = pi_dist_t.entropy()
        pi_logprob_a_t = pi_dist_t.log_prob(a_t).sum(axis=-1)
        ratio = torch.exp(pi_logprob_a_t - behavior_logprob_a_t)

        if ratio.shape != advantage_t.shape:
            raise RuntimeError(f'Expect ratio and advantage_t have same shape, got {ratio.shape} and {advantage_t.shape}')

        policy_loss = rl.clipped_surrogate_gradient_loss(ratio, advantage_t, self.clip_epsilon).loss
        policy_loss = torch.mean(policy_loss, dim=0)
        entropy_loss = torch.mean(entropy_loss)

        loss = -(policy_loss + self._entropy_coef * entropy_loss)
        loss = self._policy_loss_coef * loss + (1.0 - self._ama_beta) * ama_inverse_loss + self._ama_beta * ama_forward_nll_loss

        self._policy_loss_t = policy_loss.detach().cpu().item()
        self._entropy_loss_t = entropy_loss.detach().cpu().item()

        return loss

    def _calc_value_loss(self, transitions: GaussianTransition) -> torch.Tensor:
        s_t = torch.from_numpy(transitions.s_t).to(device=self._device, dtype=torch.float32)
        returns_t = torch.from_numpy(transitions.returns_t).to(device=self._device, dtype=torch.float32)

        base.assert_rank_and_dtype(s_t, 2, torch.float32)
        base.assert_rank_and_dtype(returns_t, 1, torch.float32)

        v_t = self._critic_network(s_t).squeeze(-1)
        value_loss = rl.value_loss(returns_t, v_t).loss
        value_loss = torch.mean(value_loss)

        self._value_loss_t = value_loss.detach().cpu().item()
        return value_loss

    @torch.no_grad()
    def _compute_returns_and_advantages(
        self,
        s_t: Iterable[np.ndarray],
        r_t: Iterable[float],
        s_tp1: Iterable[np.ndarray],
        done_tp1: Iterable[bool],
    ):
        stacked_s_t = torch.from_numpy(np.stack(s_t, axis=0)).to(device=self._device, dtype=torch.float32)
        stacked_r_t = torch.from_numpy(np.stack(r_t, axis=0)).to(device=self._device, dtype=torch.float32)
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
            'ama_inverse_loss': self._ama_inverse_loss_t,
            'ama_forward_nll_loss': self._ama_forward_nll_loss_t,
            'updates': self._update_t,
            'clip_epsilon': self.clip_epsilon,
        }
