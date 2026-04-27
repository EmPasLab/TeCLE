import numpy
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_ac.torch_ac.algos.base import BaseAlgo
from torch_ac.torch_ac.algos import common, normalizer
import numpy as np
from torch_ac.torch_ac.utils import DictList
from torch_ac.torch_ac.utils.noisy_tv_wrapper import NoisyTVWrapper
class IcmNatureConvNet(nn.Module):

    def __init__(self, state_dim: int, action_dim: int, device) -> None:
        super().__init__()

        self.action_dim = action_dim
        self.device = device

        h, w, c = state_dim
        h, w = common.calc_conv2d_output((h, w), 3, 1, 1)
        h, w = common.calc_conv2d_output((h, w), 3, 1, 1)
        h, w = common.calc_conv2d_output((h, w), 3, 1, 1)
        conv2d_out_size = 32 * h * w  

        self.body = self.net = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )


        self.forward_net = nn.Sequential(
            nn.Linear(conv2d_out_size + self.action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, conv2d_out_size),
            nn.ReLU(),
        )


        self.inverse_net = nn.Sequential(
            nn.Linear(conv2d_out_size * 2, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

        common.initialize_weights(self)
        

    def forward(self, s_tm1: torch.Tensor, a_tm1: torch.Tensor, s_t: torch.Tensor):
        a_tm1_onehot = F.one_hot(a_tm1, self.action_dim).float()
        features_tm1 = self.body(s_tm1).to(self.device)
        features_t = self.body(s_t).to(self.device)


        forward_input = torch.cat([features_tm1, a_tm1_onehot], dim=-1)
        pred_features_t = self.forward_net(forward_input)


        inverse_input = torch.cat([features_tm1, features_t], dim=-1)
        pi_logits_a_tm1 = self.inverse_net(inverse_input)

        return pi_logits_a_tm1, pred_features_t, features_t
    
    
class PPO_icm_Algo(BaseAlgo):

    def __init__(self, envs, acmodel, noisy_tv, clip_grad, intrinsic_lambda, device=None, num_frames_per_proc=None, discount=0.99, lr=0.001, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-8, clip_eps=0.2, epochs=4, batch_size=256, preprocess_obss=None,
                 reshape_reward=None):
        num_frames_per_proc = num_frames_per_proc or 128

        super().__init__(envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward)

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.noisy_tv = noisy_tv
        assert self.batch_size % self.recurrence == 0

        self.optimizer = torch.optim.Adam(self.acmodel.parameters(), lr, eps=adam_eps)
        self.batch_num = 0
        self.visitation_counts = np.zeros(
            (self.env.envs[0].unwrapped.width, self.env.envs[0].unwrapped.height)
        )
        self._int_reward_normalizer = normalizer.TorchRunningMeanStd(shape=(1,), device=self.device)
        state_dim = self.obs[0]['image'].shape
        self.icmnet = IcmNatureConvNet(state_dim=state_dim,
                                       action_dim=envs[0].action_space.n,
                                       device=self.device).to(self.device)
        
        self.icm_optimizer = torch.optim.Adam(
        self.icmnet.parameters(),
        lr=lr,
        )
        shape = (self.num_frames_per_proc, self.num_procs)
        self.intrinsic_rewards = torch.zeros(*shape, device=self.device)
        self.intrinsic_reward_buffer = []
        self.extrinsic_rewards = torch.zeros(*shape, device=self.device)
        self.extrinsic_reward_buffer = []
        self.novel_states_visited = torch.zeros(*shape, device=self.device)
        self.ext_values = torch.zeros(*shape, device=self.device)
        self.int_values = torch.zeros(*shape, device=self.device)
        
        
        self.rnd_states = [None] * (shape[0])
        self.clip_grad = clip_grad
        
        self.ext_advantages = torch.zeros(*shape, device=self.device)
        self.int_advantages = torch.zeros(*shape, device=self.device)
        
        self._intrinsic_lambda = intrinsic_lambda
        self._max_grad_norm = max_grad_norm
        
        self.obss_p1 = [None] * (shape[0])
        self.obss_p0 = [None] * (shape[0])
        
        self._policy_loss_coef = 1.0
        self._icm_beta = 0.2
        
        self.log_episode_ext_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_reshaped_ext_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_int_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_reshaped_int_return = torch.zeros(self.num_procs, device=self.device)
        self.log_int_return = [0] * self.num_procs
        self.log_reshaped_int_return = [0] * self.num_procs
        self.log_ext_return = [0] * self.num_procs
        self.log_reshaped_ext_return = [0] * self.num_procs
        self.env = NoisyTVWrapper(self.env, self.noisy_tv)
        
    
    def _update_icm_network(self, s_t, s_tp1, action):
        self.icm_optimizer.zero_grad()
        icm_loss, inverse_loss, forward_loss, intrinsic_reward = self._calc_icm_loss(s_t,
                                                                                     s_tp1, 
                                                                                     action)
        icm_loss.backward()

        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(
                self.icmnet.parameters(),
                max_norm=self._max_grad_norm,
                error_if_nonfinite=True,
            )

        self.icm_optimizer.step()

        return inverse_loss, forward_loss, intrinsic_reward
    
    def _calc_icm_loss(self, s_t, s_tp1, action):
        
        pi_logits_a_tm1, pred_features_t, features_t = self.icmnet(s_t, action, s_tp1)
        
        inverse_losses = F.cross_entropy(pi_logits_a_tm1, action, reduction='none')  
        forward_losses = torch.mean(0.5 * torch.square(pred_features_t - features_t), dim=1) 
        
        intrinsic_reward = self._intrinsic_lambda * forward_losses.clone().detach()
        
        self._int_reward_normalizer.update(intrinsic_reward)

        intrinsic_reward = self._int_reward_normalizer.normalize(intrinsic_reward)

        intrinsic_reward = torch.clamp(intrinsic_reward, -10, 10)

        inverse_loss = inverse_losses.mean()
        forward_loss = forward_losses.mean()
        

        icm_loss = inverse_loss + forward_loss
        
        return icm_loss, inverse_loss.detach(), forward_loss.detach(), intrinsic_reward
    
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
                    dist, ext_value, memory = self.acmodel(preprocessed_obs, 
                                                           self.memory * self.mask.unsqueeze(1))
                else:
                    dist, ext_value, = self.acmodel(preprocessed_obs)
            action = dist.sample() 
            obs, ext_reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())
            done = tuple(a | b for a, b in zip(terminated, truncated))
            
            
            a_tm1 = action.to(torch.int64)
            s_tm1 = preprocessed_obs.image.permute(0, 3, 1, 2) 
            s_t = self.preprocess_obss(obs, device=self.device).image.permute(0, 3, 1, 2)
            
            self.update_visitation_counts(self.env.envs)
            self.obss[i] = self.obs  
            self.obs = obs          
            self.obss_p1[i] = s_t 
            self.obss_p0[i] = s_tm1 
            
            if self.acmodel.recurrent:
                self.memories[i] = self.memory
                self.memory = memory
            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.actions[i] = action
            self.ext_values[i] = ext_value
            
            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, ext_reward, done)
                ], device=self.device)
                
            else:
                self.rewards[i] = torch.tensor(ext_reward, device=self.device)
                
            
            self.log_probs[i] = dist.log_prob(action)
            self.novel_states_visited[i] = np.count_nonzero(self.visitation_counts)


            self.log_episode_return += torch.tensor(ext_reward, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)
                
            self.log_episode_ext_return += torch.tensor(ext_reward, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_ext_return += self.extrinsic_rewards[i]
            
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
                _, next_ext_value, _ = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
            else:
                _, next_ext_value = self.acmodel(preprocessed_obs)

        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
            next_ext_value = self.ext_values[i+1] if i < self.num_frames_per_proc - 1 else next_ext_value
            next_ext_advantage = self.ext_advantages[i+1] if i < self.num_frames_per_proc - 1 else 0

            ext_delta = self.rewards[i] + self.discount * next_ext_value * next_mask - self.ext_values[i]
            self.ext_advantages[i] = ext_delta + self.discount * self.gae_lambda * next_ext_advantage * next_mask
            

        exps = DictList()
        exps.obs = [self.obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]
        
        exps.obs_icm = [self.obss_p0[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]
        exps.obs_p1 = [self.obss_p1[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]
        exps.obs_icm = torch.stack(exps.obs_icm, dim=0)
        exps.obs_p1 = torch.stack(exps.obs_p1, dim=0)
        
        if self.acmodel.recurrent:

            exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])

            exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)

        exps.action = self.actions.transpose(0, 1).reshape(-1)
        exps.ext_value = self.ext_values.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.ext_advantage = self.ext_advantages.transpose(0, 1).reshape(-1)
        exps.ext_returnn = exps.ext_value + exps.ext_advantage
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)
        novel_states_visited = self.novel_states_visited.transpose(0, 1).reshape(-1)
        

        exps.obs = self.preprocess_obss(exps.obs, device=self.device)
        

        keep = max(self.log_done_counter, self.num_procs)

        logs = {
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
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []
            log_inverse_loss = []
            log_forward_loss = []
            log_intrinsic_reward = []

            for inds in self._get_batches_starting_indexes():


                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_loss = 0
                batch_inverse_loss = 0
                batch_forward_loss = 0
                batch_intrinsic_reward = 0


                if self.acmodel.recurrent:
                    memory = exps.memory[inds]

                for i in range(self.recurrence):


                    sb = exps[inds + i]


                    if self.acmodel.recurrent:
                        dist, value, memory = self.acmodel(sb.obs, memory * sb.mask)
                    else:
                        dist, value = self.acmodel(sb.obs)

                    entropy = dist.entropy().mean()
                    
                    a_tm1 = sb.action.to(torch.int64)

                    inverse_loss, forward_loss, intrinsic_reward = self._update_icm_network(s_t=sb.obs_icm,
                                                                                            s_tp1=sb.obs_p1,
                                                                                            action=a_tm1)
                    
                    
                    ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                    surr1 = ratio * sb.ext_advantage
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.ext_advantage
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_clipped = sb.ext_value + torch.clamp(value - sb.ext_value, -self.clip_eps, self.clip_eps)
                    surr1 = (value - sb.ext_returnn).pow(2)
                    surr2 = (value_clipped - sb.ext_returnn).pow(2)
                    value_loss = torch.max(surr1, surr2).mean()

                    loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

                    loss = self._policy_loss_coef * loss + (1.0 - self._icm_beta) * inverse_loss + self._icm_beta * forward_loss


                    batch_entropy += entropy.item()
                    batch_value += value.mean().item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_loss += loss
                    batch_inverse_loss += inverse_loss
                    batch_forward_loss += forward_loss
                    batch_intrinsic_reward += intrinsic_reward.mean()


                    if self.acmodel.recurrent and i < self.recurrence - 1:
                        exps.memory[inds + i + 1] = memory.detach()


                batch_entropy /= self.recurrence
                batch_value /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_loss /= self.recurrence
                batch_loss /= self.recurrence
                batch_inverse_loss /= self.recurrence
                batch_forward_loss /= self.recurrence
                batch_intrinsic_reward /= self.recurrence


                self.optimizer.zero_grad()
                batch_loss.backward()
                grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.acmodel.parameters()) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
                self.optimizer.step()


                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm)
                log_inverse_loss.append(batch_inverse_loss.cpu())
                log_forward_loss.append(batch_forward_loss.cpu())
                log_intrinsic_reward.append(batch_intrinsic_reward.cpu())


        logs = {
            "entropy": numpy.mean(log_entropies),
            "value": numpy.mean(log_values),
            "policy_loss": numpy.mean(log_policy_losses),
            "value_loss": numpy.mean(log_value_losses),
            "grad_norm": numpy.mean(log_grad_norms),
            "inverse_loss": numpy.mean(log_inverse_loss),
            "forward_loss": numpy.mean(log_forward_loss),
            "intrinsic_reward": numpy.mean(log_intrinsic_reward)
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
