from typing import Mapping, Tuple, Text
import numpy as np
import torch


import type as types_lib
import distributions
from networks.policy import ImpalaActorCriticNetworkInputs
from networks.value import RnnDqnNetworkInputs, NguNetworkInputs, Agent57NetworkInputs
from curiosity import EpisodicBonusModule, RndLifeLongBonusModule
from transforms import compute_transformed_q


HiddenState = Tuple[torch.Tensor, torch.Tensor]


def apply_egreedy_policy(
    q_values: torch.Tensor,
    epsilon: float,
    random_state: np.random.RandomState,
) -> types_lib.Action:
    action_dim = q_values.shape[-1]
    if random_state.rand() <= epsilon:
        a_t = random_state.randint(0, action_dim)
    else:
        a_t = q_values.argmax(-1).cpu().item()
    return a_t


class EpsilonGreedyActor(types_lib.Agent):
    def __init__(
        self,
        network: torch.nn.Module,
        exploration_epsilon: float,
        random_state: np.random.RandomState,
        device: torch.device,
        name: str = 'DQN-greedy',
    ):
        self.agent_name = name
        self._device = device
        self._network = network.to(device=device)
        self._exploration_epsilon = exploration_epsilon
        self._random_state = random_state

    def step(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        return self._select_action(timestep)

    def reset(self) -> None:
        pass

    @torch.no_grad()
    def _select_action(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        s_t = torch.tensor(timestep.observation[None, ...]).to(device=self._device, dtype=torch.float32)
        q_t = self._network(s_t).q_values
        return apply_egreedy_policy(q_t, self._exploration_epsilon, self._random_state)

    @property
    def statistics(self) -> Mapping[Text, float]:
        return {
            'exploration_epsilon': self._exploration_epsilon,
        }


class IqnEpsilonGreedyActor(EpsilonGreedyActor):
    def __init__(
        self,
        network: torch.nn.Module,
        exploration_epsilon: float,
        random_state: np.random.RandomState,
        device: torch.device,
        tau_samples: int,
    ):
        super().__init__(
            network,
            exploration_epsilon,
            random_state,
            device,
            'IQN-greedy',
        )
        self._tau_samples = tau_samples

    @torch.no_grad()
    def _select_action(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        s_t = torch.tensor(timestep.observation[None, ...]).to(device=self._device, dtype=torch.float32)
        q_t = self._network(s_t, self._tau_samples).q_values
        return apply_egreedy_policy(q_t, self._exploration_epsilon, self._random_state)


class DrqnEpsilonGreedyActor(EpsilonGreedyActor):
    def __init__(
        self,
        network: torch.nn.Module,
        exploration_epsilon: float,
        random_state: np.random.RandomState,
        device: torch.device,
    ):
        super().__init__(
            network,
            exploration_epsilon,
            random_state,
            device,
            'DRQN-greedy',
        )
        self._lstm_state = None

    @torch.no_grad()
    def _select_action(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        s_t = torch.tensor(timestep.observation[None, None, ...]).to(device=self._device, dtype=torch.float32)
        hidden_s = tuple(s.to(device=self._device) for s in self._lstm_state)
        network_output = self._network(s_t, hidden_s)
        q_t = network_output.q_values
        self._lstm_state = network_output.hidden_s
        return apply_egreedy_policy(q_t, self._exploration_epsilon, self._random_state)

    def reset(self) -> None:
        self._lstm_state = self._network.get_initial_hidden_state(batch_size=1)


class R2d2EpsilonGreedyActor(EpsilonGreedyActor):
    def __init__(
        self,
        network: torch.nn.Module,
        exploration_epsilon: float,
        random_state: np.random.RandomState,
        device: torch.device,
    ):
        super().__init__(
            network,
            exploration_epsilon,
            random_state,
            device,
            'R2D2-greedy',
        )
        self._last_action = None
        self._lstm_state = None

    @torch.no_grad()
    def _select_action(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        s_t = torch.tensor(timestep.observation[None, ...]).to(device=self._device, dtype=torch.float32)
        a_tm1 = torch.tensor(self._last_action).to(device=self._device, dtype=torch.int64)
        r_t = torch.tensor(timestep.reward).to(device=self._device, dtype=torch.float32)
        hidden_s = tuple(s.to(device=self._device) for s in self._lstm_state)

        network_output = self._network(
            RnnDqnNetworkInputs(
                s_t=s_t[None, ...],
                a_tm1=a_tm1[None, ...],
                r_t=r_t[None, ...],
                hidden_s=hidden_s,
            )
        )
        q_t = network_output.q_values
        self._lstm_state = network_output.hidden_s

        a_t = apply_egreedy_policy(q_t, self._exploration_epsilon, self._random_state)
        self._last_action = a_t
        return a_t

    def reset(self) -> None:
        self._last_action = 0
        self._lstm_state = self._network.get_initial_hidden_state(batch_size=1)


class NguEpsilonGreedyActor(EpsilonGreedyActor):
    def __init__(
        self,
        network: torch.nn.Module,
        embedding_network: torch.nn.Module,
        rnd_target_network: torch.nn.Module,
        rnd_predictor_network: torch.nn.Module,
        episodic_memory_capacity: int,
        num_neighbors: int,
        cluster_distance: float,
        kernel_epsilon: float,
        max_similarity: float,
        exploration_epsilon: float,
        random_state: np.random.RandomState,
        device: torch.device,
    ):
        super().__init__(
            network,
            exploration_epsilon,
            random_state,
            device,
            'NGU-greedy',
        )

        self._policy_index = 0
        self._policy_beta = 0

        self._episodic_module = EpisodicBonusModule(
            embedding_network=embedding_network,
            device=device,
            capacity=episodic_memory_capacity,
            num_neighbors=num_neighbors,
            kernel_epsilon=kernel_epsilon,
            cluster_distance=cluster_distance,
            max_similarity=max_similarity,
        )

        self._lifelong_module = RndLifeLongBonusModule(
            target_network=rnd_target_network, predictor_network=rnd_predictor_network, device=device, discount=0.99
        )

        self._last_action = None
        self._episodic_bonus_t = None
        self._lifelong_bonus_t = None
        self._lstm_state = self._network.get_initial_hidden_state(batch_size=1)

    @torch.no_grad()
    def step(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        a_t = self._select_action(timestep)
        s_t = torch.from_numpy(timestep.observation[None, ...]).to(device=self._device, dtype=torch.float32)

        self._lifelong_bonus_t = self._lifelong_module.compute_bonus(s_t)

        self._episodic_bonus_t = self._episodic_module.compute_bonus(s_t)
        return a_t

    @torch.no_grad()
    def _select_action(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        s_t = torch.tensor(timestep.observation[None, ...]).to(device=self._device, dtype=torch.float32)
        a_tm1 = torch.tensor(self._last_action).to(device=self._device, dtype=torch.int64)
        ext_r_t = torch.tensor(timestep.reward).to(device=self._device, dtype=torch.float32)
        int_r_t = torch.tensor(self.intrinsic_reward).to(device=self._device, dtype=torch.float32)
        policy_index = torch.tensor(self._policy_index).to(device=self._device, dtype=torch.int64)
        hidden_s = tuple(s.to(device=self._device) for s in self._lstm_state)

        pi_output = self._network(
            NguNetworkInputs(
                s_t=s_t[None, ...],
                a_tm1=a_tm1[None, ...],
                ext_r_t=ext_r_t[None, ...],
                int_r_t=int_r_t[None, ...],
                policy_index=policy_index[None, ...],
                hidden_s=hidden_s,
            )
        )

        q_t = pi_output.q_values
        self._lstm_state = pi_output.hidden_s

        a_t = apply_egreedy_policy(q_t, self._exploration_epsilon, self._random_state)
        self._last_action = a_t
        return a_t

    def reset(self) -> None:
        self._last_action = 0
        self._episodic_bonus_t = 0.0
        self._lifelong_bonus_t = 0.0
        self._lstm_state = self._network.get_initial_hidden_state(batch_size=1)
        self._episodic_module.reset()

    @property
    def intrinsic_reward(self) -> float:
        return self._episodic_bonus_t * min(max(self._lifelong_bonus_t, 1.0), 5.0)


class Agent57EpsilonGreedyActor(types_lib.Agent):
    def __init__(
        self,
        network: torch.nn.Module,
        embedding_network: torch.nn.Module,
        rnd_target_network: torch.nn.Module,
        rnd_predictor_network: torch.nn.Module,
        episodic_memory_capacity: int,
        num_neighbors: int,
        cluster_distance: float,
        kernel_epsilon: float,
        max_similarity: float,
        exploration_epsilon: float,
        random_state: np.random.RandomState,
        device: torch.device,
    ):
        self.agent_name = 'Agent57-greedy'
        self._network = network.to(device=device)
        self._device = device

        self._random_state = random_state
        self._exploration_epsilon = exploration_epsilon

        self._policy_index = 0
        self._policy_beta = 0

        self._episodic_module = EpisodicBonusModule(
            embedding_network=embedding_network,
            device=device,
            capacity=episodic_memory_capacity,
            num_neighbors=num_neighbors,
            kernel_epsilon=kernel_epsilon,
            cluster_distance=cluster_distance,
            max_similarity=max_similarity,
        )

        self._lifelong_module = RndLifeLongBonusModule(
            target_network=rnd_target_network,
            predictor_network=rnd_predictor_network,
            device=device,
            discount=0.99,
        )

        self._last_action = None
        self._episodic_bonus_t = None
        self._lifelong_bonus_t = None
        self._ext_lstm_state = None
        self._int_lstm_state = None

    @torch.no_grad()
    def step(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        a_t = self._select_action(timestep)

        s_t = torch.from_numpy(timestep.observation[None, ...]).to(device=self._device, dtype=torch.float32)

        self._lifelong_bonus_t = self._lifelong_module.compute_bonus(s_t)

        self._episodic_bonus_t = self._episodic_module.compute_bonus(s_t)

        return a_t

    def reset(self) -> None:
        self._last_action = 0
        self._episodic_bonus_t = 0.0
        self._lifelong_bonus_t = 0.0
        self._ext_lstm_state, self._int_lstm_state = self._network.get_initial_hidden_state(batch_size=1)

        self._episodic_module.reset()

    @torch.no_grad()
    def _select_action(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        input_ = self._prepare_network_input(timestep)

        output = self._network(input_)
        ext_q_t = output.ext_q_values.squeeze()
        int_q_t = output.int_q_values.squeeze()

        q_t = compute_transformed_q(ext_q_t, int_q_t, self._policy_beta)

        self._ext_lstm_state = output.ext_hidden_s
        self._int_lstm_state = output.int_hidden_s

        a_t = apply_egreedy_policy(q_t, self._exploration_epsilon, self._random_state)
        self._last_action = a_t
        return a_t

    def _prepare_network_input(self, timestep: types_lib.TimeStep) -> Agent57NetworkInputs:
        s_t = torch.tensor(timestep.observation[None, ...]).to(device=self._device, dtype=torch.float32)
        a_tm1 = torch.tensor(self._last_action).to(device=self._device, dtype=torch.int64)
        ext_r_t = torch.tensor(timestep.reward).to(device=self._device, dtype=torch.float32)
        int_r_t = torch.tensor(self.intrinsic_reward).to(device=self._device, dtype=torch.float32)
        policy_index = torch.tensor(self._policy_index).to(device=self._device, dtype=torch.int64)
        ext_hidden_s = tuple(s.to(device=self._device) for s in self._ext_lstm_state)
        int_hidden_s = tuple(s.to(device=self._device) for s in self._int_lstm_state)
        return Agent57NetworkInputs(
            s_t=s_t[None, ...],
            a_tm1=a_tm1[None, ...],
            ext_r_t=ext_r_t[None, ...],
            int_r_t=int_r_t[None, ...],
            policy_index=policy_index[None, ...],
            ext_hidden_s=ext_hidden_s,
            int_hidden_s=int_hidden_s,
        )

    @property
    def intrinsic_reward(self) -> float:
        return self._episodic_bonus_t * min(max(self._lifelong_bonus_t, 1.0), 5.0)

    @property
    def statistics(self) -> Mapping[Text, float]:
        return {
            'exploration_epsilon': self._exploration_epsilon,
        }


class PolicyGreedyActor(types_lib.Agent):
    def __init__(
        self,
        network: torch.nn.Module,
        device: torch.device,
        name: str = '',
    ):
        self.agent_name = name
        self._device = device
        self._network = network.to(device=device)

    def step(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        return self.act(timestep)

    def act(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        return self._select_action(timestep)

    def reset(self) -> None:
        pass

    @torch.no_grad()
    def _select_action(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        s_t = torch.tensor(timestep.observation[None, ...]).to(device=self._device, dtype=torch.float32)
        pi_logits_t = self._network(s_t).pi_logits

        a_t = distributions.categorical_distribution(pi_logits_t).sample()

        return a_t.cpu().item()

    @property
    def statistics(self) -> Mapping[Text, float]:
        return {}


class ImpalaGreedyActor(PolicyGreedyActor):
    def __init__(
        self,
        network: torch.nn.Module,
        device: torch.device,
    ) -> None:
        super().__init__(
            network,
            device,
            'IMPALA',
        )

        self._last_action = None
        self._hidden_s = self._network.get_initial_hidden_state(batch_size=1)

    def step(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        a_t = self.act(timestep)

        self._last_action = a_t

        return a_t

    def act(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        a_t = self._choose_action(timestep)
        return a_t

    def reset(self) -> None:
        self._last_action = 0
        self._hidden_s = self._network.get_initial_hidden_state(batch_size=1)

    @torch.no_grad()
    def _choose_action(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        s_t = torch.tensor(timestep.observation[None, ...]).to(device=self._device, dtype=torch.float32)
        a_tm1 = torch.tensor(self._last_action).to(device=self._device, dtype=torch.int64)
        r_t = torch.tensor(timestep.reward).to(device=self._device, dtype=torch.float32)
        done = torch.tensor(timestep.done).to(device=self._device, dtype=torch.bool)

        hidden_s = tuple(s.to(device=self._device) for s in self._hidden_s)

        network_output = self._network(
            ImpalaActorCriticNetworkInputs(
                s_t=s_t[None, ...],
                a_tm1=a_tm1[None, ...],
                r_t=r_t[None, ...],
                done=done[None, ...],
                hidden_s=hidden_s,
            )
        )
        pi_logits_t = network_output.pi_logits.squeeze(0)

        a_t = distributions.categorical_distribution(pi_logits_t).sample()

        self._hidden_s = network_output.hidden_s
        return a_t.cpu().item()

    @property
    def statistics(self) -> Mapping[Text, float]:
        return {}


class GaussianPolicyGreedyActor(PolicyGreedyActor):
    @torch.no_grad()
    def _select_action(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        s_t = torch.tensor(timestep.observation[None, ...]).to(device=self._device, dtype=torch.float32)
        pi_mu, pi_sigma = self._network(s_t)

        a_t = distributions.normal_distribution(pi_mu, pi_sigma).sample()
        return a_t.squeeze(0).cpu().numpy()


class GaussianRndPolicyGreedyActor(PolicyGreedyActor):
    @torch.no_grad()
    def _select_action(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        s_t = torch.tensor(timestep.observation[None, ...]).to(device=self._device, dtype=torch.float32)
        output = self._network(s_t)
        pi_mu, pi_sigma = output.pi_mu, output.pi_sigma
        a_t = distributions.normal_distribution(pi_mu, pi_sigma).sample()
        return a_t.squeeze(0).cpu().numpy()
