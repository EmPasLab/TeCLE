import numpy
import torch
import torch.nn as nn
from torch_ac.torch_ac.utils import DictList
from torch_ac.torch_ac.algos.base import BaseAlgo
from torch_ac.torch_ac.algos import common, normalizer
import numpy as np
from torch_ac.torch_ac.utils.noisy_tv_wrapper import NoisyTVWrapper

class RndConvNet(nn.Module):

    def __init__(self, state_dim: int, is_target: bool = False, latent_dim: int = 256) -> None:
        super().__init__()

        h, w, c = state_dim
        
        h, w = common.calc_conv2d_output((h, w), 3, 1)
        h, w = common.calc_conv2d_output((h, w), 3, 1)
        
        conv2d_out_size = 64 * h * w
        

        self.body = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Flatten(),
        )
        
        if is_target:
            self.head = nn.Linear(conv2d_out_size, latent_dim)
        else:
            self.head = nn.Sequential(
                nn.Linear(conv2d_out_size, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, latent_dim),
            )


        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, numpy.sqrt(2))
                module.bias.data.zero_()

        if is_target:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.body(x)
        return self.head(x)


class PPO_rnd_rev_Algo(BaseAlgo):

    def __init__(self, envs, acmodel, noisy_tv, reward_weighting, normalise_rewards, clip_grad, device=None, num_frames_per_proc=None, 
                 discount=0.99, lr=0.001, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-8, clip_eps=0.2, epochs=4, batch_size=256, preprocess_obss=None,
                 reshape_reward=None):
        num_frames_per_proc = num_frames_per_proc or 128

        super().__init__(envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward)
        
        shape = (self.num_frames_per_proc, self.num_procs)
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.noisy_tv = noisy_tv
        assert self.batch_size % self.recurrence == 0

        self.optimizer = torch.optim.Adam(self.acmodel.parameters(), lr, eps=adam_eps)
        self.batch_num = 0
        
        state_dim = self.obs[0]['image'].shape
        
        rnd_state_dim = state_dim
        
        self.novel_states_visited = torch.zeros(*shape, device=self.device)
        self.dones = [None] * (shape[0])
        self._int_discount =0.99
        self.intrinsic_rewards = torch.zeros(*shape, device=self.device)
        self.intrinsic_reward_buffer = []
        
        self.ext_values = torch.zeros(*shape, device=self.device)
        self.int_values = torch.zeros(*shape, device=self.device)
        self.visitation_counts = np.zeros(
            (self.env.envs[0].unwrapped.width, self.env.envs[0].unwrapped.height)
        )
        
        self.rnd_states = [None] * (shape[0])
        self.clip_grad = clip_grad
        
        self.ext_advantages = torch.zeros(*shape, device=self.device)
        
        self._target_network = RndConvNet(state_dim = state_dim, is_target = True).to(device=device)
        self._predict_network = RndConvNet(state_dim = state_dim, is_target = False).to(device=device)
        for p in self._target_network.parameters():
            p.requires_grad = False
        self.rnd_optimizer = torch.optim.Adam(
        self._predict_network.parameters(),
        lr=lr,
        )
        int_discount = 0.99
        self._int_discount = int_discount
        self._int_reward_normalizer = normalizer.RunningMeanStd(shape=(1,))
        
        self._rnd_obs_normalizer = normalizer.TorchRunningMeanStd(shape=(3, 7, 7), device=self.device)
        rnd_experience_proportion = min(
                                        1.0, 32 / self.num_procs
                                        )
        self._rnd_experience_proportion = rnd_experience_proportion
        self._max_grad_norm = max_grad_norm
        rnd_obs_clip=5
        self.rnd_obs_clip = rnd_obs_clip
        self.env = NoisyTVWrapper(self.env, self.noisy_tv)
        
        
    def compute_intrinsic_rewards(self, rnd_s_t):
        
        normed_s_t = self._normalize_rnd_obs(rnd_s_t, True)
        normed_s_t = normed_s_t.to(device=self.device, dtype=torch.float32)

        pred = self._predict_network(normed_s_t)
        with torch.no_grad():
            target = self._target_network(normed_s_t)

        int_r_t = torch.square(pred - target).mean(dim=1).detach().cpu().numpy()
        int_r_t = int_r_t.reshape(128,-1)
        
        normed_int_r_t = self._normalize_int_rewards(int_r_t)

        return normed_int_r_t
    
    def _normalize_int_rewards(self, int_rewards):


        intrinsic_returns = []
        rewems = 0
        for t in reversed(range(len(int_rewards))):
            rewems = rewems * self._int_discount + int_rewards[t]
            intrinsic_returns.insert(0, rewems)
        self._int_reward_normalizer.update(numpy.ravel(intrinsic_returns).reshape(-1, 1))

        normed_int_rewards = int_rewards / numpy.sqrt(self._int_reward_normalizer.var + 1e-8)

        return normed_int_rewards.tolist()

    @torch.no_grad()
    def _normalize_rnd_obs(self, rnd_obs_list, update_stats=False):

        tacked_obs = torch.from_numpy(numpy.stack(rnd_obs_list.cpu(), axis=0)).to(device=self.device, dtype=torch.float32)

        normed_obs = self._rnd_obs_normalizer.normalize(tacked_obs)
        normed_obs = normed_obs.clamp(-self.rnd_obs_clip, self.rnd_obs_clip)

        if update_stats:
            self._rnd_obs_normalizer.update(tacked_obs)

        return normed_obs
    
    def _update_rnd_predictor_network(self, rnd_s_t):
        normed_s_t = self._normalize_rnd_obs(rnd_s_t, True)
        normed_s_t = normed_s_t.to(device=self.device, dtype=torch.float32)

        pred_t = self._predict_network(normed_s_t)
        with torch.no_grad():
            target_t = self._target_network(normed_s_t)

        rnd_loss = torch.square(pred_t - target_t).mean(dim=1)


        if self._rnd_experience_proportion < 1:
            mask = torch.rand(rnd_loss.size())
            mask = torch.where(mask < self._rnd_experience_proportion, 1.0, 0.0).to(device=self.device, dtype=torch.float32)
            rnd_loss = rnd_loss * mask


        rnd_loss = torch.mean(rnd_loss)
        
        return rnd_loss
        
    def _compute_returns_and_advantages(
        self,
        v_t,
        r_t,
        v_tp1,
        done_tp1,
        discount,
    ):

        v_t = torch.from_numpy(np.stack(v_t, axis=0)).to(device=self.device, dtype=torch.float32)
        r_t = torch.from_numpy(np.stack(r_t, axis=0)).to(device=self.device, dtype=torch.float32)
        v_tp1 = torch.from_numpy(np.stack(v_tp1, axis=0)).to(device=self.device, dtype=torch.float32)
        done_tp1 = torch.from_numpy(np.stack(done_tp1, axis=0)).to(device=self.device, dtype=torch.bool)

        discount_tp1 = (~done_tp1).float() * discount
        
        advantage_t = common.truncated_generalized_advantage_estimation(r_t, 
                                                                        v_t, 
                                                                        v_tp1, 
                                                                        discount_tp1, 
                                                                        self.gae_lambda)

        return_t = advantage_t + v_t

        
        advantage_t = advantage_t.cpu().numpy()
        return_t = return_t.cpu().numpy()

        return (return_t, advantage_t)

    
    def update_visitation_counts(self, envs):
        for i, env in enumerate(envs):
            
            if self.visitation_counts[env.unwrapped.agent_pos[0]][env.unwrapped.agent_pos[1]] == 0:
                pass
                
            self.visitation_counts[env.unwrapped.agent_pos[0]][env.unwrapped.agent_pos[1]] += 1

    
    def collect_experiences(self):
        
        for i in range(self.num_frames_per_proc):


            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
            with torch.no_grad():
                if self.acmodel.recurrent:
                    dist, ext_value, int_value, memory = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
                else:
                    dist, ext_value, int_value, = self.acmodel(preprocessed_obs)
            action = dist.sample() 
            
            obs, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())
            done = tuple(a | b for a, b in zip(terminated, truncated))
            self.dones[i] = done

            self.update_visitation_counts(self.env.envs)
            self.obss[i] = self.obs
            self.obs = obs
            rnd_state = preprocessed_obs.image.permute(0, 3, 1, 2)
            
            self.rnd_state = rnd_state
            self.rnd_states[i] = self.rnd_state
            
            
            if self.acmodel.recurrent:
                self.memories[i] = self.memory
                self.memory = memory
            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.actions[i] = action
            self.ext_values[i] = ext_value
            self.int_values[i] = int_value
            
            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
            else:
                self.rewards[i] = torch.tensor(reward, device=self.device)
            self.log_probs[i] = dist.log_prob(action)
            self.novel_states_visited[i] = np.count_nonzero(self.visitation_counts)


            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            for i, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i].item())
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())

            self.log_episode_return *= self.mask
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask

        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        with torch.no_grad():
            if self.acmodel.recurrent:
                _, next_ext_value, next_int_value, _ = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
            else:
                _, next_ext_value, next_int_value = self.acmodel(preprocessed_obs)

        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
            next_ext_value = self.ext_values[i+1] if i < self.num_frames_per_proc - 1 else next_ext_value
            next_int_value = self.int_values[i+1] if i < self.num_frames_per_proc - 1 else next_int_value
            next_ext_advantage = self.ext_advantages[i+1] if i < self.num_frames_per_proc - 1 else 0

            ext_delta = self.rewards[i] + self.discount * next_ext_value * next_mask - self.ext_values[i]
            self.ext_advantages[i] = ext_delta + self.discount * self.gae_lambda * next_ext_advantage * next_mask


        exps = DictList()
        exps.obs = [self.obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]
        exps.rnd_state = [self.rnd_states[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]
        exps.rnd_state = torch.stack(exps.rnd_state, dim=0)
        if self.acmodel.recurrent:

            exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])

            exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)

        exps.action = self.actions.transpose(0, 1).reshape(-1)
        exps.ext_value = self.ext_values.transpose(0, 1).reshape(-1)
        exps.int_value = self.int_values.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.ext_advantage = self.ext_advantages.transpose(0, 1).reshape(-1)
        exps.ext_returnn = exps.ext_value + exps.ext_advantage
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)
        novel_states_visited = self.novel_states_visited.transpose(0, 1).reshape(-1)


        exps.obs = self.preprocess_obss(exps.obs, device=self.device)
        exps.int_value_tp1 = self.int_values[1:]
        next_int_value = next_int_value.reshape(1, -1)
        int_value_tp1 = torch.cat([exps.int_value_tp1, next_int_value],dim=0)
        intrinsic_reward = self.compute_intrinsic_rewards(exps.rnd_state)
        intrinsic_reward = torch.tensor(intrinsic_reward)
        
        (int_return_t, int_advantage_t) = self._compute_returns_and_advantages(self.int_values.cpu(),
                                                                               intrinsic_reward.cpu(),
                                                                               int_value_tp1.cpu(),
                                                                               np.zeros_like(self.dones), 
                                                                               self._int_discount,
                                                                                )
        exps.intrinsic_rewards = intrinsic_reward.transpose(0, 1).reshape(-1)
        exps.int_advantages = int_advantage_t.transpose(0, 1).reshape(-1)
        exps.int_returnn = int_return_t.transpose(0, 1).reshape(-1)
        exps.int_value_tp1  = int_value_tp1.transpose(0, 1).reshape(-1)
        keep = max(self.log_done_counter, self.num_procs)
        logs = {
            "intrinsic_rewards": intrinsic_reward,
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames,
            "novel_states_visited": novel_states_visited,
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return exps, logs, self.visitation_counts

    def update_parameters(self, exps):


        for _ in range(self.epochs):


            log_entropies = []
            log_ext_values = []
            log_int_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []
            log_rnd_losses = []

            for inds in self._get_batches_starting_indexes():


                batch_entropy = 0
                batch_ext_value = 0
                batch_int_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_loss = 0
                batch_rnd_loss = 0


                if self.acmodel.recurrent:
                    memory = exps.memory[inds]

                for i in range(self.recurrence):


                    sb = exps[inds + i]


                    if self.acmodel.recurrent:
                        dist, ext_value, int_value, memory = self.acmodel(sb.obs, memory * sb.mask)
                    else:
                        dist, ext_value, int_value, _ = self.acmodel(sb.obs)

                    entropy = dist.entropy().mean()

                    advantage_t = 2.0* sb.ext_advantage + 1.0 * torch.tensor(sb.int_advantages).to(self.device)
                    
                    ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                    surr1 = ratio * advantage_t
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantage_t
                    policy_loss = -torch.min(surr1, surr2).mean()

                    ext_value_clipped = sb.ext_value + torch.clamp(ext_value - sb.ext_value, -self.clip_eps, self.clip_eps)
                    surr1 = (ext_value - sb.ext_returnn).pow(2)
                    surr2 = (ext_value_clipped - sb.ext_returnn).pow(2)
                    ext_value_loss = torch.max(surr1, surr2).mean()

                    int_value_clipped = sb.int_value + torch.clamp(int_value - sb.int_value, -self.clip_eps, self.clip_eps)
                    surr1 = (int_value - torch.tensor(sb.int_returnn).to(self.device)).pow(2)
                    surr2 = (int_value_clipped - torch.tensor(sb.int_returnn).to(self.device)).pow(2)
                    int_value_loss = torch.max(surr1, surr2).mean()

                    value_loss = ext_value_loss + int_value_loss
                    
                    loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

                    rnd_loss = self._update_rnd_predictor_network(sb.rnd_state)


                    batch_entropy += entropy.item()
                    batch_ext_value += ext_value.mean().item()
                    batch_int_value += int_value.mean().item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_loss += loss
                    batch_rnd_loss += rnd_loss


                    if self.acmodel.recurrent and i < self.recurrence - 1:
                        exps.memory[inds + i + 1] = memory.detach()


                batch_entropy /= self.recurrence
                batch_ext_value /= self.recurrence
                batch_int_value /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_loss /= self.recurrence
                batch_loss /= self.recurrence
                batch_rnd_loss /= self.recurrence


                self.optimizer.zero_grad()
                self.rnd_optimizer.zero_grad()
                batch_loss.backward()
                batch_rnd_loss.backward()
                grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.acmodel.parameters()) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
                self.optimizer.step()

                self.rnd_optimizer.step()

                if self.clip_grad:
                    torch.nn.utils.clip_grad_norm_(
                        self._predict_network.parameters(),
                        max_norm=self._max_grad_norm,
                        error_if_nonfinite=True,
                    )


                log_entropies.append(batch_entropy)
                log_ext_values.append(batch_ext_value)
                log_int_values.append(batch_int_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm)
                log_rnd_losses.append(batch_rnd_loss.detach().cpu())


        logs = {
            "entropy": numpy.mean(log_entropies),
            "ext_value": numpy.mean(log_ext_values),
            "int_value": numpy.mean(log_int_values),
            "policy_loss": numpy.mean(log_policy_losses),
            "value_loss": numpy.mean(log_value_losses),
            "grad_norm": numpy.mean(log_grad_norms),
            "rnd_loss": numpy.mean(log_rnd_losses)
        }

        return logs

    def _get_batches_starting_indexes(self):

        indexes = numpy.arange(0, self.num_frames, self.recurrence)
        indexes = numpy.random.permutation(indexes)


        if self.batch_num % 2 == 1:
            indexes = indexes[(indexes + self.recurrence) % self.num_frames_per_proc != 0]
            indexes += self.recurrence // 2
        self.batch_num += 1

        num_indexes = self.batch_size // self.recurrence
        batches_starting_indexes = [indexes[i:i+num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes
