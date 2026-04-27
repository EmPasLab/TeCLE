from typing import Mapping, Tuple, Iterable, Text
import multiprocessing
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import type as types_lib
from schedule import LinearSchedule
import policy_gradient as rl
import distributions
import multistep
import utils
import base
import normalizer
from networks.curiosity import (
    TecleInverseConvNet as InverseNet,
    TecleCVAEConvNet as cVAE,
    TecleInverseMlpNet as GaussianInverseNet,
    TecleCVAEMlpNet as GaussianCVAE,
    tecle_cvae_loss as cvae_loss,
    gaussian_tecle_cvae_loss as gaussian_cvae_loss,
    powerlaw_psd_gaussian,
)

torch.autograd.set_detect_anomaly(True)


class Actor(types_lib.Agent):
    def __init__(
        self,
        rank: int,
        data_queue: multiprocessing.Queue,
        policy_network: torch.nn.Module,
        unroll_length: int,
        device: torch.device,
        shared_params: dict,
        algo: str,
    ) -> None:
        if not 1 <= unroll_length:
            raise ValueError(f'Expect unroll_length to be integer greater than or equal to 1, got {unroll_length}')

        self.rank = rank
        self.agent_name = f'{algo}-actor{rank}'
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
        self._a_tm2 = None
        self._a_tm1 = None
        self._logprob_a_tm1 = None

    def step(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        self._step_t += 1

        a_t, logprob_a_t, ext_value, int_value = self.act(timestep)

        if self._a_tm2 is not None:
            self._unroll_sequence.append(
                (
                    self._s_tm1,  
                    self._s_tm2,  
                    self._a_tm1,  
                    self._a_tm2, 
                    self._logprob_a_tm1, 
                    ext_value,
                    int_value,
                    timestep.reward, 
                    timestep.done
                )
            )

            if len(self._unroll_sequence) == self._unroll_length:
                self._queue.put(self._unroll_sequence)
                self._unroll_sequence = []
                self._update_actor_network()
        
        if self._s_tm1 is not None:
            self._s_tm2 = self._s_tm1
        
        self._s_tm1 = timestep.observation

        if self._a_tm1 is not None:
            self._a_tm2 = self._a_tm1
        self._a_tm1 = a_t 
        self._logprob_a_tm1 = logprob_a_t

        return a_t

    def reset(self) -> None:
        self._s_tm1 = None
        self._a_tm1 = None
        self._a_tm2 = None
        self._logprob_a_tm1 = None

    def act(self, timestep: types_lib.TimeStep) -> Tuple[types_lib.Action]:
        return self._choose_action(timestep)

    def _update_actor_network(self):
        state_dict = self._shared_params['policy_network']
        if state_dict is not None:
            if self._device != 'cpu':
                state_dict = {k: v.to(device=self._device) for k, v in state_dict.items()}
            self._policy_network.load_state_dict(state_dict)

    @torch.no_grad()
    def _choose_action(self, timestep: types_lib.TimeStep) -> Tuple[types_lib.Action, float, float, float]: 
        s_t = torch.from_numpy(timestep.observation[None, ...]).to(device=self._device, dtype=torch.float32) 
        output = self._policy_network(s_t)
        pi_logits_t = output.pi_logits

        pi_dist_t = distributions.categorical_distribution(pi_logits_t)
        a_t = pi_dist_t.sample()
        logprob_a_t = pi_dist_t.log_prob(a_t)

        ext_value, int_value = output.ext_baseline, output.int_baseline
        return (
            a_t.cpu().item(),
            logprob_a_t.cpu().item(),
            ext_value.squeeze(0).cpu().item(),
            int_value.squeeze(0).cpu().item(),
        )

    @property
    def statistics(self) -> Mapping[Text, float]:
        return {}


class Learner(types_lib.Learner):
    def __init__(
        self,
        policy_network: nn.Module,
        policy_optimizer: torch.optim.Optimizer,
        TeCLE_network: nn.Module,
        inv_network: nn.Module,
        TeCLE_optimizer: torch.optim.Optimizer,
        inv_optimizer: torch.optim.Optimizer,
        obs_clip: float,
        clip_epsilon: LinearSchedule,
        ext_discount: float,
        int_discount: float,
        gae_lambda: float,
        total_unroll_length: int,
        update_k: int,
        entropy_coef: float,
        value_coef: float,
        clip_grad: bool,
        max_grad_norm: float,
        device: torch.device,
        shared_params: dict,
        algo: str,
    ) -> None:
        if not 1 <= total_unroll_length:
            raise ValueError(f'Expect total_unroll_length to be greater than 1, got {total_unroll_length}')
        if not 1 <= update_k:
            raise ValueError(f'Expect update_k to be integer greater than or equal to 1, got {update_k}')
        if not 0.0 <= entropy_coef <= 1.0:
            raise ValueError(f'Expect entropy_coef to [0.0, 1.0], got {entropy_coef}')
        if not 0.0 <= value_coef <= 1.0:
            raise ValueError(f'Expect value_coef to [0.0, 1.0], got {value_coef}')

        self.agent_name = f'{algo}-learner'
        self._policy_network = policy_network.to(device=device)
        self._policy_optimizer = policy_optimizer

        self._TeCLE_network = TeCLE_network.to(device=device)
        self._inv_network = inv_network.to(device=device)
        self._TeCLE_optimizer = TeCLE_optimizer
        self._inv_optimizer = inv_optimizer

        self._device = device

        self._shared_params = shared_params

        self._obs_clip = obs_clip

        self._int_reward_normalizer = normalizer.RunningMeanStd(shape=(1,))
        self._obs_normalizer = normalizer.TorchRunningMeanStd(shape=(1, 84, 84), device=self._device)

        self._storage = []
        self._total_unroll_length = total_unroll_length

        self._batch_size = min(512, int(np.ceil(total_unroll_length / 4).item()))

        self._update_k = update_k

        self._entropy_coef = entropy_coef
        self._value_coef = value_coef
        self._clip_epsilon = clip_epsilon

        self._clip_grad = clip_grad
        self._max_grad_norm = max_grad_norm
        self._ext_discount = ext_discount
        self._int_discount = int_discount
        self._gae_lambda = gae_lambda

        self._step_t = -1
        self._update_t = 0
        self._policy_loss_t = np.nan
        self._value_loss_t = np.nan
        self._entropy_loss_t = np.nan
        self._TeCLE_loss_t = np.nan
        self._inv_loss_t = np.nan
        self.normed_int_r_t_t = 0

    def step(self) -> Iterable[Mapping[Text, float]]:
        self._step_t += 1

        if len(self._storage) < self._total_unroll_length:
            return

        return self._learn()

    def reset(self) -> None:
        self._storage = []

    def received_item_from_queue(self, unroll_sequences: Iterable[Tuple]) -> None:
        observations, observaions_tm1, actions, a_tm1, logprob_actions, ext_values, int_values, rewards, dones = map(list, zip(*unroll_sequences))
    
        s_t = observations[:-1] 
        s_tm1 = observaions_tm1[:-1]

        a_t = actions[:-1]

        a_tm1 = a_tm1[:-1]

        logprob_a_t = logprob_actions[:-1]

        ext_v_t = ext_values[:-1]

        ext_r_t = rewards[1:]

        ext_v_tp1 = ext_values[1:]

        done_tp1 = dones[1:]

        int_v_t = int_values[:-1]
        
        int_v_tp1 = int_values[1:]

        (ext_return_t, ext_advantage_t) = self._compute_returns_and_advantages(
            ext_v_t, ext_r_t, ext_v_tp1, done_tp1, self._ext_discount
        )

        TeCLE_s_t = [s[-1:, ...] for s in s_t]
        TeCLE_s_tm1 = [s[-1:, ...] for s in s_tm1]
        int_r_t = self._compute_int_reward(TeCLE_s_t, TeCLE_s_tm1, a_tm1)
        

        (int_return_t, int_advantage_t) = self._compute_returns_and_advantages(
            int_v_t,
            int_r_t,
            int_v_tp1,
            np.zeros_like(done_tp1),
            self._int_discount,
        )

        zipped_sequence = list(
            zip(s_t, s_tm1, a_t, a_tm1, logprob_a_t, ext_return_t, ext_advantage_t, TeCLE_s_t, TeCLE_s_tm1, int_return_t, int_advantage_t)
        )

        self._storage += zipped_sequence

    def get_policy_state_dict(self):
        return {k: v.cpu() for k, v in self._policy_network.state_dict().items()}

    def init_obs_stats(self, obs_list):
        self._normalize_obs(obs_list, True)

    def _learn(self) -> Iterable[Mapping[Text, float]]:
        num_samples = len(self._storage)

        for i in range(self._update_k):

            binned_indices = utils.split_indices_into_bins(self._batch_size, num_samples, shuffle=True)
            for indices in binned_indices:
                mini_batch = [self._storage[i] for i in indices] 

                self._update_policy_network(mini_batch)
                
                self._update_TeCLE_network(mini_batch)
                self._update_t += 1
                yield self.statistics

        self._shared_params['policy_network'] = self.get_policy_state_dict()

        del self._storage[:]

    def _update_TeCLE_network(self, samples):
        self._TeCLE_optimizer.zero_grad()
        self._inv_optimizer.zero_grad()

        _, _, _, a_tm1, _, _, _, TeCLE_s_t, TeCLE_s_tm1, _, _ = map(list, zip(*samples))
        normed_s_tm1 = self._normalize_obs(TeCLE_s_tm1, True)
        normed_s_tm1 = normed_s_tm1.to(device=self._device, dtype=torch.float32)
        
        normed_s_t = self._normalize_obs(TeCLE_s_t, True) 
        normed_s_t = normed_s_t.to(device=self._device, dtype=torch.float32)

        a_tm1 = torch.tensor(a_tm1).to(device=self._device, dtype=torch.long)
        pi_logits, features_t = self._inv_network(normed_s_t, normed_s_tm1)
        features_t_detach = features_t.clone().detach()
        s_hat_t, mean_t, logvar_t = self._TeCLE_network(features_t_detach, a_tm1)
        TeCLE_loss = cvae_loss(features_t_detach, s_hat_t, mean_t, logvar_t)
        inv_loss = F.cross_entropy(pi_logits, a_tm1, reduction='none')
            

        TeCLE_loss = torch.mean(TeCLE_loss)
        inv_loss = torch.mean(inv_loss)

        TeCLE_loss.backward()
        inv_loss.backward()

        if self._clip_grad:
            torch.nn.utils.clip_grad_norm_(
                self._TeCLE_network.parameters(),
                max_norm=self._max_grad_norm,
                error_if_nonfinite=True,
            )

        self._TeCLE_optimizer.step()
        self._inv_optimizer.step()

        self._TeCLE_loss_t = TeCLE_loss.detach().cpu().item()
        self._inv_loss_t = inv_loss.detach().cpu().item()

    def _update_policy_network(self, mini_batch):
        self._policy_optimizer.zero_grad()

        (s_t, _, a_t, _, logprob_a_t, ext_return_t, ext_advantage_t, _, _, int_return_t, int_advantage_t) = map(list, zip(*mini_batch))

        s_t = torch.from_numpy(np.stack(s_t, axis=0)).to(device=self._device, dtype=torch.float32)
        a_t = torch.from_numpy(np.stack(a_t, axis=0)).to(device=self._device, dtype=torch.int64)
        behavior_logprob_a_t = torch.from_numpy(np.stack(logprob_a_t, axis=0)).to(device=self._device, dtype=torch.float32)
        ext_return_t = torch.from_numpy(np.stack(ext_return_t, axis=0)).to(device=self._device, dtype=torch.float32)
        ext_advantage_t = torch.from_numpy(np.stack(ext_advantage_t, axis=0)).to(device=self._device, dtype=torch.float32)
        int_return_t = torch.from_numpy(np.stack(int_return_t, axis=0)).to(device=self._device, dtype=torch.float32)
        int_advantage_t = torch.from_numpy(np.stack(int_advantage_t, axis=0)).to(device=self._device, dtype=torch.float32)

        base.assert_rank_and_dtype(s_t, (2, 4), torch.float32)
        base.assert_rank_and_dtype(a_t, 1, torch.long)
        base.assert_rank_and_dtype(ext_return_t, 1, torch.float32)
        base.assert_rank_and_dtype(ext_advantage_t, 1, torch.float32)
        base.assert_rank_and_dtype(int_return_t, 1, torch.float32)
        base.assert_rank_and_dtype(int_advantage_t, 1, torch.float32)
        base.assert_rank_and_dtype(behavior_logprob_a_t, 1, torch.float32)

        pi_logits_t, ext_v_t, int_v_t = self._policy_network(s_t)

        pi_dist_t = distributions.categorical_distribution(pi_logits_t)
        pi_logprob_a_t = pi_dist_t.log_prob(a_t)
        entropy_loss = pi_dist_t.entropy()

        advantage_t = 2.0 * ext_advantage_t + 1.0 * int_advantage_t

        ratio = torch.exp(pi_logprob_a_t - behavior_logprob_a_t)

        if ratio.shape != advantage_t.shape:
            raise RuntimeError(f'Expect ratio and advantages have same shape, got {ratio.shape} and {advantage_t.shape}')
        policy_loss = rl.clipped_surrogate_gradient_loss(ratio, advantage_t, self.clip_epsilon).loss

        ext_v_loss = rl.value_loss(ext_return_t, ext_v_t.squeeze(-1)).loss
        int_v_loss = rl.value_loss(int_return_t, int_v_t.squeeze(-1)).loss

        value_loss = ext_v_loss + int_v_loss

        policy_loss = torch.mean(policy_loss)
        entropy_loss = torch.mean(entropy_loss)
        value_loss = torch.mean(value_loss)

        loss = -(policy_loss + self._entropy_coef * entropy_loss) + self._value_coef * value_loss

        loss.backward()

        if self._clip_grad:
            torch.nn.utils.clip_grad_norm_(
                self._policy_network.parameters(),
                max_norm=self._max_grad_norm,
                error_if_nonfinite=True,
            )

        self._policy_optimizer.step()

        self._policy_loss_t = policy_loss.detach().cpu().item()
        self._value_loss_t = value_loss.detach().cpu().item()
        self._entropy_loss_t = entropy_loss.detach().cpu().item()

    @torch.no_grad()     
    def _compute_int_reward(self, TeCLE_s_t, TeCLE_s_tm1, a_tm1, rnd = True):
        if rnd == True:    
            normed_s_tm1 = self._normalize_obs(TeCLE_s_tm1) 
            normed_s_tm1 = normed_s_tm1.to(device=self._device, dtype=torch.float32)

            normed_s_t = self._normalize_obs(TeCLE_s_t) 
            normed_s_t = normed_s_t.to(device=self._device, dtype=torch.float32)

        a_tm1 = torch.tensor(a_tm1).to(device=self._device, dtype=torch.long)

        _, features_t = self._inv_network(normed_s_t, normed_s_tm1)
        features_t_detach = features_t.clone().detach()

        s_hat, _, _ = self._TeCLE_network(features_t_detach, a_tm1)
        
        s_hat = s_hat.view(features_t_detach.shape[0], -1)
        features_t_detach = features_t_detach.view(features_t_detach.shape[0], -1)
        int_r_t = torch.square(s_hat - features_t_detach).mean(dim=1).detach().cpu().numpy()

        normed_int_r_t = self._normalize_int_rewards(int_r_t)
        self.normed_int_r_t_t = (torch.mean(torch.tensor(normed_int_r_t))).cpu().item()
        
        return normed_int_r_t

    @torch.no_grad()
    def _normalize_obs(self, obs_list, update_stats=False):
        tacked_obs = torch.from_numpy(np.stack(obs_list, axis=0)).to(device=self._device, dtype=torch.float32)

        normed_obs = self._obs_normalizer.normalize(tacked_obs) 

        normed_obs = normed_obs.clamp(-self._obs_clip, self._obs_clip) 
        if update_stats:
            self._obs_normalizer.update(tacked_obs) 

        return normed_obs

    @torch.no_grad()
    def _compute_returns_and_advantages(
        self,
        v_t: Iterable[np.ndarray],
        r_t: Iterable[float],
        v_tp1: Iterable[np.ndarray],
        done_tp1: Iterable[bool],
        discount: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        v_t = torch.from_numpy(np.stack(v_t, axis=0)).to(device=self._device, dtype=torch.float32)
        r_t = torch.from_numpy(np.stack(r_t, axis=0)).to(device=self._device, dtype=torch.float32)
        v_tp1 = torch.from_numpy(np.stack(v_tp1, axis=0)).to(device=self._device, dtype=torch.float32)
        done_tp1 = torch.from_numpy(np.stack(done_tp1, axis=0)).to(device=self._device, dtype=torch.bool)

        discount_tp1 = (~done_tp1).float() * discount

        advantage_t = multistep.truncated_generalized_advantage_estimation(r_t, v_t, v_tp1, discount_tp1, self._gae_lambda)

        return_t = advantage_t + v_t

        advantage_t = advantage_t.cpu().numpy()
        return_t = return_t.cpu().numpy()

        return (return_t, advantage_t)

    def _normalize_int_rewards(self, int_rewards):
        intrinsic_returns = []
        rewems = 0
        for t in reversed(range(len(int_rewards))):
            rewems = rewems * self._int_discount + int_rewards[t] 
            intrinsic_returns.insert(0, rewems)
        self._int_reward_normalizer.update(np.ravel(intrinsic_returns).reshape(-1, 1))

        normed_int_rewards = int_rewards / np.sqrt(self._int_reward_normalizer.var + 1e-8) 

        return normed_int_rewards.tolist()

    @property
    def clip_epsilon(self):
        return self._clip_epsilon(self._step_t)

    @property
    def statistics(self) -> Mapping[Text, float]:
        return {
            'policy_loss': self._policy_loss_t,
            'value_loss': self._value_loss_t,
            'entropy_loss': self._entropy_loss_t,
            'TeCLE_loss': self._TeCLE_loss_t,
            'inv_loss': self._inv_loss_t,
            'intrinsic_reward': self.normed_int_r_t_t,
            'updates': self._update_t,
            'clip_epsilon': self.clip_epsilon,
        }


class GaussianActor(Actor):
    @torch.no_grad()
    def _choose_action(self, timestep: types_lib.TimeStep) -> Tuple[types_lib.Action, float, float, float]:
        s_t = torch.from_numpy(timestep.observation[None, ...]).to(device=self._device, dtype=torch.float32)
        output = self._policy_network(s_t)
        pi_mu, pi_sigma = output.pi_mu, output.pi_sigma
        pi_dist_t = distributions.normal_distribution(pi_mu, pi_sigma)
        a_t = pi_dist_t.sample()
        logprob_a_t = pi_dist_t.log_prob(a_t).sum(dim=-1)

        ext_value = output.ext_baseline
        int_value = output.int_baseline
        return (
            a_t.squeeze(0).cpu().numpy(),
            logprob_a_t.cpu().item(),
            ext_value.squeeze(0).cpu().item(),
            int_value.squeeze(0).cpu().item(),
        )


class GaussianLearner(Learner):
    def __init__(
        self,
        policy_network: nn.Module,
        policy_optimizer: torch.optim.Optimizer,
        tecle_network: nn.Module,
        inv_network: nn.Module,
        tecle_optimizer: torch.optim.Optimizer,
        inv_optimizer: torch.optim.Optimizer,
        obs_clip: float,
        clip_epsilon: LinearSchedule,
        ext_discount: float,
        int_discount: float,
        gae_lambda: float,
        total_unroll_length: int,
        update_k: int,
        entropy_coef: float,
        value_coef: float,
        clip_grad: bool,
        max_grad_norm: float,
        state_dim: int,
        device: torch.device,
        shared_params: dict,
    ) -> None:
        if not 1 <= total_unroll_length:
            raise ValueError(f'Expect total_unroll_length >= 1, got {total_unroll_length}')
        if not 1 <= update_k:
            raise ValueError(f'Expect update_k >= 1, got {update_k}')

        self.agent_name = 'PPO-TeCLE-Gaussian-learner'
        self._policy_network = policy_network.to(device=device)
        self._policy_optimizer = policy_optimizer

        self._TeCLE_network = tecle_network.to(device=device)
        self._inv_network = inv_network.to(device=device)
        self._TeCLE_optimizer = tecle_optimizer
        self._inv_optimizer = inv_optimizer

        self._device = device
        self._shared_params = shared_params
        self._obs_clip = obs_clip

        self._int_reward_normalizer = normalizer.RunningMeanStd(shape=(1,))
        self._obs_normalizer = normalizer.TorchRunningMeanStd(shape=(state_dim,), device=device)

        self._storage = []
        self._total_unroll_length = total_unroll_length
        self._batch_size = min(512, int(np.ceil(total_unroll_length / 4).item()))
        self._update_k = update_k

        self._entropy_coef = entropy_coef
        self._value_coef = value_coef
        self._clip_epsilon = clip_epsilon
        self._clip_grad = clip_grad
        self._max_grad_norm = max_grad_norm
        self._ext_discount = ext_discount
        self._int_discount = int_discount
        self._gae_lambda = gae_lambda

        self._step_t = -1
        self._update_t = 0
        self._policy_loss_t = np.nan
        self._value_loss_t = np.nan
        self._entropy_loss_t = np.nan
        self._TeCLE_loss_t = np.nan
        self._inv_loss_t = np.nan
        self.normed_int_r_t_t = 0

    def init_obs_stats(self, obs_list):
        stacked = torch.from_numpy(np.stack(obs_list, axis=0)).to(device=self._device, dtype=torch.float32)
        self._obs_normalizer.update(stacked)

    def received_item_from_queue(self, unroll_sequences: Iterable[Tuple]) -> None:
        observations, obs_tm1, actions, a_tm1, logprob_actions, ext_values, int_values, rewards, dones = map(
            list, zip(*unroll_sequences)
        )

        s_t = observations[:-1]
        s_tm1 = obs_tm1[:-1]
        a_t = actions[:-1]
        a_tm1 = a_tm1[:-1]
        logprob_a_t = logprob_actions[:-1]
        ext_v_t = ext_values[:-1]
        ext_r_t = rewards[1:]
        ext_v_tp1 = ext_values[1:]
        done_tp1 = dones[1:]
        int_v_t = int_values[:-1]
        int_v_tp1 = int_values[1:]

        (ext_return_t, ext_advantage_t) = self._compute_returns_and_advantages(
            ext_v_t, ext_r_t, ext_v_tp1, done_tp1, self._ext_discount
        )

        K = len(s_t)
        noise_beta = self._TeCLE_network.noise_beta
        if noise_beta > 0.0:
            raw = powerlaw_psd_gaussian(noise_beta, (self._TeCLE_network.nhid, K))
            noise_seq = torch.tensor(raw, dtype=torch.float32).T
        else:
            noise_seq = None

        int_r_t = self._compute_int_reward(s_t, s_tm1, a_tm1, noise_seq=noise_seq)

        (int_return_t, int_advantage_t) = self._compute_returns_and_advantages(
            int_v_t, int_r_t, int_v_tp1, np.zeros_like(done_tp1), self._int_discount
        )

        noise_list = [noise_seq[i] if noise_seq is not None else None for i in range(K)]
        zipped = list(zip(s_t, s_tm1, a_t, a_tm1, logprob_a_t, ext_return_t, ext_advantage_t, int_return_t, int_advantage_t, noise_list))
        self._storage += zipped

    def _update_TeCLE_network(self, samples):
        self._TeCLE_optimizer.zero_grad()
        self._inv_optimizer.zero_grad()

        _, s_tm1, _, a_tm1, _, _, _, _, _, noise_list = map(list, zip(*samples))
        s_t_batch, _, _, _, _, _, _, _, _, _ = map(list, zip(*samples))

        normed_s_tm1 = self._normalize_obs(s_tm1)
        normed_s_t = self._normalize_obs(s_t_batch)

        a_tm1 = torch.from_numpy(np.stack(a_tm1, axis=0)).to(device=self._device, dtype=torch.float32)

        pred_action, features_t = self._inv_network(normed_s_t, normed_s_tm1)
        features_t_detach = features_t.clone().detach()

        f_min = features_t_detach.min(dim=-1, keepdim=True).values
        f_max = features_t_detach.max(dim=-1, keepdim=True).values
        features_norm = (features_t_detach - f_min) / (f_max - f_min + 1e-8)

        eps = torch.stack(noise_list).to(device=self._device) if noise_list[0] is not None else None
        recon, mean, logvar = self._TeCLE_network(features_norm, a_tm1, eps=eps)
        tecle_loss = gaussian_cvae_loss(features_norm, recon, mean, logvar)
        inv_loss = F.mse_loss(pred_action, a_tm1)

        tecle_loss = torch.mean(tecle_loss) if tecle_loss.dim() > 0 else tecle_loss
        inv_loss = torch.mean(inv_loss) if inv_loss.dim() > 0 else inv_loss

        tecle_loss.backward()
        inv_loss.backward()

        if self._clip_grad:
            torch.nn.utils.clip_grad_norm_(self._TeCLE_network.parameters(), max_norm=self._max_grad_norm)

        self._TeCLE_optimizer.step()
        self._inv_optimizer.step()

        self._TeCLE_loss_t = tecle_loss.detach().cpu().item()
        self._inv_loss_t = inv_loss.detach().cpu().item()

    def _update_policy_network(self, mini_batch):
        self._policy_optimizer.zero_grad()

        s_t, _, a_t, _, logprob_a_t, ext_return_t, ext_advantage_t, int_return_t, int_advantage_t, _ = map(
            list, zip(*mini_batch)
        )

        s_t = torch.from_numpy(np.stack(s_t, axis=0)).to(device=self._device, dtype=torch.float32)
        a_t = torch.from_numpy(np.stack(a_t, axis=0)).to(device=self._device, dtype=torch.float32)
        behavior_logprob_a_t = torch.tensor(logprob_a_t, device=self._device, dtype=torch.float32)
        ext_return_t = torch.from_numpy(np.stack(ext_return_t, axis=0)).to(device=self._device, dtype=torch.float32)
        ext_advantage_t = torch.from_numpy(np.stack(ext_advantage_t, axis=0)).to(device=self._device, dtype=torch.float32)
        int_return_t = torch.from_numpy(np.stack(int_return_t, axis=0)).to(device=self._device, dtype=torch.float32)
        int_advantage_t = torch.from_numpy(np.stack(int_advantage_t, axis=0)).to(device=self._device, dtype=torch.float32)

        output = self._policy_network(s_t)
        pi_dist_t = distributions.normal_distribution(output.pi_mu, output.pi_sigma)
        pi_logprob_a_t = pi_dist_t.log_prob(a_t).sum(dim=-1)
        entropy_loss = pi_dist_t.entropy().sum(dim=-1)

        advantage_t = 2.0 * ext_advantage_t + 1.0 * int_advantage_t

        ratio = torch.exp(pi_logprob_a_t - behavior_logprob_a_t)
        policy_loss = rl.clipped_surrogate_gradient_loss(ratio, advantage_t, self.clip_epsilon).loss

        ext_v_loss = rl.value_loss(ext_return_t, output.ext_baseline.squeeze(-1)).loss
        int_v_loss = rl.value_loss(int_return_t, output.int_baseline.squeeze(-1)).loss

        policy_loss = torch.mean(policy_loss)
        entropy_loss = torch.mean(entropy_loss)
        value_loss = torch.mean(ext_v_loss + int_v_loss)

        loss = -(policy_loss + self._entropy_coef * entropy_loss) + self._value_coef * value_loss
        loss.backward()

        if self._clip_grad:
            torch.nn.utils.clip_grad_norm_(self._policy_network.parameters(), max_norm=self._max_grad_norm)

        self._policy_optimizer.step()

        self._policy_loss_t = policy_loss.detach().cpu().item()
        self._value_loss_t = value_loss.detach().cpu().item()
        self._entropy_loss_t = entropy_loss.detach().cpu().item()

    @torch.no_grad()
    def _compute_int_reward(self, s_t_list, s_tm1_list, a_tm1_list, noise_seq=None):
        normed_s_tm1 = self._normalize_obs(s_tm1_list)
        normed_s_t = self._normalize_obs(s_t_list)

        a_tm1 = torch.from_numpy(np.stack(a_tm1_list, axis=0)).to(device=self._device, dtype=torch.float32)

        _, features_t = self._inv_network(normed_s_t, normed_s_tm1)
        features_t_detach = features_t.clone().detach()
        f_min = features_t_detach.min(dim=-1, keepdim=True).values
        f_max = features_t_detach.max(dim=-1, keepdim=True).values
        features_norm = (features_t_detach - f_min) / (f_max - f_min + 1e-8)

        eps = noise_seq.to(device=self._device) if noise_seq is not None else None
        recon, _, _ = self._TeCLE_network(features_norm, a_tm1, eps=eps)
        int_r_t = torch.square(recon - features_norm).mean(dim=-1).detach().cpu().numpy()

        normed = self._normalize_int_rewards(int_r_t)
        self.normed_int_r_t_t = float(np.mean(normed))
        return normed

    @torch.no_grad()
    def _normalize_obs(self, obs_list, update_stats=False):
        stacked = torch.from_numpy(np.stack(obs_list, axis=0)).to(device=self._device, dtype=torch.float32)
        normed = self._obs_normalizer.normalize(stacked)
        normed = normed.clamp(-self._obs_clip, self._obs_clip)
        if update_stats:
            self._obs_normalizer.update(stacked)
        return normed
