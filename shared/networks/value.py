from itertools import chain
from typing import NamedTuple, Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F


from networks import common


class DqnNetworkOutputs(NamedTuple):
    q_values: torch.Tensor


class C51NetworkOutputs(NamedTuple):
    q_values: torch.Tensor
    q_logits: torch.Tensor


class QRDqnNetworkOutputs(NamedTuple):
    q_values: torch.Tensor
    q_dist: torch.Tensor


class IqnNetworkOutputs(NamedTuple):
    q_values: torch.Tensor
    q_dist: torch.Tensor
    taus: torch.Tensor


class RnnDqnNetworkInputs(NamedTuple):
    s_t: torch.Tensor
    a_tm1: torch.Tensor
    r_t: torch.Tensor
    hidden_s: Optional[Tuple[torch.Tensor, torch.Tensor]]


class RnnDqnNetworkOutputs(NamedTuple):
    q_values: torch.Tensor
    hidden_s: Optional[Tuple[torch.Tensor, torch.Tensor]]


class NguNetworkInputs(NamedTuple):
    s_t: torch.Tensor
    a_tm1: torch.Tensor
    ext_r_t: torch.Tensor
    int_r_t: torch.Tensor
    policy_index: torch.Tensor
    hidden_s: Optional[Tuple[torch.Tensor, torch.Tensor]]


class Agent57NetworkInputs(NamedTuple):
    s_t: torch.Tensor
    a_tm1: torch.Tensor
    ext_r_t: torch.Tensor
    int_r_t: torch.Tensor
    policy_index: torch.Tensor
    ext_hidden_s: Optional[Tuple[torch.Tensor, torch.Tensor]]
    int_hidden_s: Optional[Tuple[torch.Tensor, torch.Tensor]]


class Agent57NetworkOutputs(NamedTuple):
    ext_q_values: torch.Tensor
    int_q_values: torch.Tensor
    ext_hidden_s: Optional[Tuple[torch.Tensor, torch.Tensor]]
    int_hidden_s: Optional[Tuple[torch.Tensor, torch.Tensor]]


class DqnMlpNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        if action_dim < 1:
            raise ValueError(f'Expect action_dim to be a positive integer, got {action_dim}')
        if state_dim < 1:
            raise ValueError(f'Expect state_dim to be a positive integer, got {state_dim}')

        super().__init__()

        self.body = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> DqnNetworkOutputs:
        q_values = self.body(x)
        return DqnNetworkOutputs(q_values=q_values)


class DuelingDqnMlpNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        if action_dim < 1:
            raise ValueError(f'Expect action_dim to be a positive integer, got {action_dim}')
        if state_dim < 1:
            raise ValueError(f'Expect state_dim to be a positive integer, got {state_dim}')

        super().__init__()

        self.body = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )

        self.advantage_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

        self.value_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> DqnNetworkOutputs:
        features = self.body(x)

        advantages = self.advantage_head(features)
        values = self.value_head(features)

        q_values = values + (advantages - torch.mean(advantages, dim=1, keepdim=True))

        return DqnNetworkOutputs(q_values=q_values)


class C51DqnMlpNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, atoms: torch.Tensor):
        if action_dim < 1:
            raise ValueError(f'Expect action_dim to be a positive integer, got {action_dim}')
        if state_dim < 1:
            raise ValueError(f'Expect state_dim to be a positive integer, got {state_dim}')
        if len(atoms.shape) != 1:
            raise ValueError(f'Expect atoms to be a 1D tensor, got {atoms.shape}')

        super().__init__()
        self.action_dim = action_dim
        self.atoms = atoms
        self.num_atoms = atoms.size(0)

        self.body = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim * self.num_atoms),
        )

    def forward(self, x: torch.Tensor) -> C51NetworkOutputs:
        x = self.body(x)

        q_logits = x.view(-1, self.action_dim, self.num_atoms)
        q_dist = F.softmax(q_logits, dim=-1)
        atoms = self.atoms[None, None, :].to(device=x.device)
        q_values = torch.sum(q_dist * atoms, dim=-1)

        return C51NetworkOutputs(q_logits=q_logits, q_values=q_values)


class RainbowDqnMlpNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, atoms: torch.Tensor):
        if action_dim < 1:
            raise ValueError(f'Expect action_dim to be a positive integer, got {action_dim}')
        if state_dim < 1:
            raise ValueError(f'Expect state_dim to be a positive integer, got {state_dim}')
        if len(atoms.shape) != 1:
            raise ValueError(f'Expect atoms to be a 1D tensor, got {atoms.shape}')

        super().__init__()

        self.action_dim = action_dim
        self.atoms = atoms
        self.num_atoms = atoms.size(0)

        self.body = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )

        self.advantage_head = nn.Sequential(
            common.NoisyLinear(128, 128),
            nn.ReLU(),
            common.NoisyLinear(128, action_dim * self.num_atoms),
        )
        self.value_head = nn.Sequential(
            common.NoisyLinear(128, 128),
            nn.ReLU(),
            common.NoisyLinear(128, 1 * self.num_atoms),
        )

    def forward(self, x: torch.Tensor) -> C51NetworkOutputs:
        x = self.body(x)
        advantages = self.advantage_head(x)
        values = self.value_head(x)

        advantages = advantages.view(-1, self.action_dim, self.num_atoms)
        values = values.view(-1, 1, self.num_atoms)

        q_logits = values + (advantages - torch.mean(advantages, dim=1, keepdim=True))

        q_logits = q_logits.view(-1, self.action_dim, self.num_atoms)

        q_dist = F.softmax(q_logits, dim=-1)
        atoms = self.atoms[None, None, :].to(device=x.device)
        q_values = torch.sum(q_dist * atoms, dim=-1)

        return C51NetworkOutputs(q_logits=q_logits, q_values=q_values)

    def reset_noise(self) -> None:
        for module in list(chain(*zip(self.advantage_head.modules(), self.value_head.modules()))):
            if isinstance(module, common.NoisyLinear):
                module.reset_noise()


class QRDqnMlpNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, quantiles: torch.Tensor):
        if action_dim < 1:
            raise ValueError(f'Expect action_dim to be a positive integer, got {action_dim}')
        if state_dim < 1:
            raise ValueError(f'Expect state_dim to be a positive integer, got {state_dim}')
        if len(quantiles.shape) != 1:
            raise ValueError(f'Expect quantiles to be a 1D tensor, got {quantiles.shape}')

        super().__init__()
        self.taus = quantiles
        self.num_taus = quantiles.size(0)
        self.action_dim = action_dim

        self.body = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim * self.num_taus),
        )

    def forward(self, x: torch.Tensor) -> QRDqnNetworkOutputs:
        q_dist = self.body(x).view(-1, self.num_taus, self.action_dim)
        q_values = torch.mean(q_dist, dim=1)

        return QRDqnNetworkOutputs(q_values=q_values, q_dist=q_dist)


class IqnMlpNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, latent_dim: int):
        if action_dim < 1:
            raise ValueError(f'Expect action_dim to be a positive integer, got {action_dim}')
        if state_dim < 1:
            raise ValueError(f'Expect state_dim to be a positive integer, got {state_dim}')
        if latent_dim < 1:
            raise ValueError(f'Expect latent_dim to be a positive integer, got {latent_dim}')

        super().__init__()
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        self.pis = torch.arange(1, self.latent_dim + 1).float() * 3.141592653589793

        self.body = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )

        self.embedding_layer = nn.Linear(latent_dim, 128)

        self.value_head = nn.Linear(128, action_dim)

    def sample_taus(self, batch_size: int, num_taus: int) -> torch.Tensor:
        taus = torch.rand((batch_size, num_taus)).to(dtype=torch.float32)
        assert taus.shape == (batch_size, num_taus)
        return taus

    def forward(self, x: torch.Tensor, num_taus: int = 32) -> IqnNetworkOutputs:
        batch_size = x.shape[0]

        features = self.body(x)

        taus = self.sample_taus(batch_size, num_taus).to(device=x.device)

        pis = self.pis[None, None, :].to(device=x.device)
        tau_embedding = torch.cos(pis * taus[:, :, None])

        tau_embedding = tau_embedding.view(batch_size * num_taus, -1)
        tau_embedding = F.relu(self.embedding_layer(tau_embedding))

        tau_embedding = tau_embedding.view(batch_size, num_taus, -1)
        head_input = tau_embedding * features[:, None, :]

        head_input = head_input.view(-1, self.embedding_layer.out_features)

        q_dist = self.value_head(head_input)
        q_dist = q_dist.view(batch_size, -1, self.action_dim)
        q_values = torch.mean(q_dist, dim=1)
        return IqnNetworkOutputs(q_values=q_values, q_dist=q_dist, taus=taus)


class DrqnMlpNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        if action_dim < 1:
            raise ValueError(f'Expect action_dim to be a positive integer, got {action_dim}')
        if state_dim < 1:
            raise ValueError(f'Expect state_dim to be a positive integer, got {state_dim}')

        super().__init__()
        self.action_dim = action_dim

        self.body = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True)

        self.value_head = nn.Linear(self.lstm.hidden_size, action_dim)

    def forward(self, x: torch.Tensor, hidden_s: None) -> RnnDqnNetworkOutputs:
        assert len(x.shape) == 3
        B = x.shape[0]
        T = x.shape[1]

        x = torch.flatten(x, 0, 1)

        x = self.body(x)
        x = x.view(B, T, -1)

        x, hidden_s = self.lstm(x, hidden_s)

        x = torch.flatten(x, 0, 1)
        q_values = self.value_head(x)
        q_values = q_values.view(B, T, -1)
        return RnnDqnNetworkOutputs(q_values=q_values, hidden_s=hidden_s)

    def get_initial_hidden_state(self, batch_size: int) -> Tuple[torch.Tensor]:
        return tuple(torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size) for _ in range(2))


class R2d2DqnMlpNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        if action_dim < 1:
            raise ValueError(f'Expect action_dim to be a positive integer, got {action_dim}')
        if state_dim < 1:
            raise ValueError(f'Expect state_dim to be a positive integer, got {state_dim}')

        super().__init__()
        self.action_dim = action_dim

        self.body = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )

        core_output_size = 128 + self.action_dim + 1

        self.lstm = nn.LSTM(input_size=core_output_size, hidden_size=128, num_layers=1)

        self.advantage_head = nn.Sequential(
            nn.Linear(self.lstm.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )
        self.value_head = nn.Sequential(
            nn.Linear(self.lstm.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, input_: RnnDqnNetworkInputs) -> RnnDqnNetworkOutputs:
        s_t = input_.s_t
        a_tm1 = input_.a_tm1
        r_t = input_.r_t
        hidden_s = input_.hidden_s

        T, B, *_ = s_t.shape
        x = torch.flatten(s_t, 0, 1)

        x = self.body(x)
        x = x.view(T * B, -1)

        one_hot_a_tm1 = F.one_hot(a_tm1.view(T * B), self.action_dim).float().to(device=x.device)

        reward = r_t.view(T * B, 1)
        core_input = torch.cat([x, reward, one_hot_a_tm1], dim=-1)
        core_input = core_input.view(T, B, -1)

        if hidden_s is None:
            hidden_s = self.get_initial_hidden_state(batch_size=B)
            hidden_s = tuple(s.to(device=x.device) for s in hidden_s)

        x, hidden_s = self.lstm(core_input, hidden_s)

        x = torch.flatten(x, 0, 1)
        advantages = self.advantage_head(x)
        values = self.value_head(x)

        q_values = values + (advantages - torch.mean(advantages, dim=1, keepdim=True))
        q_values = q_values.view(T, B, -1)
        return RnnDqnNetworkOutputs(q_values=q_values, hidden_s=hidden_s)

    def get_initial_hidden_state(self, batch_size: int) -> Tuple[torch.Tensor]:
        return tuple(torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size) for _ in range(2))


class NguDqnMlpNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, num_policies: int):
        super().__init__()
        self.action_dim = action_dim
        self.num_policies = num_policies

        self.body = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )

        core_output_size = 128 + self.num_policies + self.action_dim + 1 + 1

        self.lstm = nn.LSTM(input_size=core_output_size, hidden_size=128, num_layers=1)

        self.advantage_head = nn.Sequential(
            nn.Linear(self.lstm.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )
        self.value_head = nn.Sequential(
            nn.Linear(self.lstm.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, input_: NguNetworkInputs) -> RnnDqnNetworkOutputs:
        s_t = input_.s_t
        a_tm1 = input_.a_tm1
        ext_r_t = input_.ext_r_t
        int_r_t = input_.int_r_t
        policy_index = input_.policy_index
        hidden_s = input_.hidden_s

        T, B, *_ = s_t.shape
        x = torch.flatten(s_t, 0, 1)
        x = self.body(x)
        x = x.view(T * B, -1)

        one_hot_beta = F.one_hot(policy_index.view(T * B), self.num_policies).float().to(device=x.device)
        one_hot_a_tm1 = F.one_hot(a_tm1.view(T * B), self.action_dim).float().to(device=x.device)
        int_reward = int_r_t.view(T * B, 1)
        ext_reward = ext_r_t.view(T * B, 1)

        core_input = torch.cat([x, ext_reward, one_hot_a_tm1, int_reward, one_hot_beta], dim=-1)
        core_input = core_input.view(T, B, -1)

        if hidden_s is None:
            hidden_s = self.get_initial_hidden_state(batch_size=B)
            hidden_s = tuple(s.to(device=x.device) for s in hidden_s)

        x, hidden_s = self.lstm(core_input, hidden_s)

        x = torch.flatten(x, 0, 1)
        advantages = self.advantage_head(x)
        values = self.value_head(x)

        q_values = values + (advantages - torch.mean(advantages, dim=1, keepdim=True))
        q_values = q_values.view(T, B, -1)
        return RnnDqnNetworkOutputs(q_values=q_values, hidden_s=hidden_s)

    def get_initial_hidden_state(self, batch_size: int) -> Tuple[torch.Tensor]:
        return tuple(torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size) for _ in range(2))


class DqnConvNet(nn.Module):
    def __init__(self, state_dim: tuple, action_dim: int):
        if action_dim < 1:
            raise ValueError(f'Expect action_dim to be a positive integer, got {action_dim}')
        if len(state_dim) != 3:
            raise ValueError(f'Expect state_dim to be a tuple with [C, H, W], got {state_dim}')

        super().__init__()
        self.action_dim = action_dim
        self.body = common.NatureCnnBackboneNet(state_dim)

        self.value_head = nn.Sequential(
            nn.Linear(self.body.out_features, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )

        common.initialize_weights(self)

    def forward(self, x: torch.Tensor) -> DqnNetworkOutputs:
        x = x.float() / 255.0
        x = self.body(x)
        q_values = self.value_head(x)
        return DqnNetworkOutputs(q_values=q_values)


class DuelingDqnConvNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        if action_dim < 1:
            raise ValueError(f'Expect action_dim to be a positive integer, got {action_dim}')
        if state_dim < 1:
            raise ValueError(f'Expect state_dim to be a positive integer, got {state_dim}')

        super().__init__()

        self.action_dim = action_dim
        self.body = common.NatureCnnBackboneNet(state_dim)

        self.advantage_head = nn.Sequential(
            nn.Linear(self.body.out_features, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )

        self.value_head = nn.Sequential(
            nn.Linear(self.body.out_features, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor) -> DqnNetworkOutputs:
        features = self.body(x)

        advantages = self.advantage_head(features)
        values = self.value_head(features)

        q_values = values + (advantages - torch.mean(advantages, dim=1, keepdim=True))

        return DqnNetworkOutputs(q_values=q_values)


class C51DqnConvNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, atoms: torch.Tensor):
        if action_dim < 1:
            raise ValueError(f'Expect action_dim to be a positive integer, got {action_dim}')
        if len(state_dim) != 3:
            raise ValueError(f'Expect state_dim to be a tuple with [C, H, W], got {state_dim}')
        if len(atoms.shape) != 1:
            raise ValueError(f'Expect atoms to be a 1D tensor, got {atoms.shape}')

        super().__init__()
        self.action_dim = action_dim
        self.atoms = atoms
        self.num_atoms = atoms.size(0)
        self.body = common.NatureCnnBackboneNet(state_dim)

        self.value_head = nn.Sequential(
            nn.Linear(self.body.out_features, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim * self.num_atoms),
        )

        common.initialize_weights(self)

    def forward(self, x: torch.Tensor) -> C51NetworkOutputs:
        x = x.float() / 255.0
        x = self.body(x)
        x = self.value_head(x)

        q_logits = x.view(-1, self.action_dim, self.num_atoms)
        q_dist = F.softmax(q_logits, dim=-1)
        atoms = self.atoms[None, None, :].to(device=x.device)
        q_values = torch.sum(q_dist * atoms, dim=-1)

        return C51NetworkOutputs(q_logits=q_logits, q_values=q_values)


class RainbowDqnConvNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, atoms: torch.Tensor):
        if len(atoms.shape) != 1:
            raise ValueError(f'Expect atoms to be a 1D tensor, got {atoms.shape}')
        if action_dim < 1:
            raise ValueError(f'Expect action_dim to be a positive integer, got {action_dim}')
        if len(state_dim) != 3:
            raise ValueError(f'Expect state_dim to be a tuple with [C, H, W], got {state_dim}')

        super().__init__()
        self.action_dim = action_dim
        self.atoms = atoms
        self.num_atoms = atoms.size(0)

        self.body = common.NatureCnnBackboneNet(state_dim)

        self.advantage_head = nn.Sequential(
            common.NoisyLinear(self.body.out_features, 512),
            nn.ReLU(),
            common.NoisyLinear(512, action_dim * self.num_atoms),
        )

        self.value_head = nn.Sequential(
            common.NoisyLinear(self.body.out_features, 512),
            nn.ReLU(),
            common.NoisyLinear(512, 1 * self.num_atoms),
        )

        common.initialize_weights(self)

    def forward(self, x: torch.Tensor) -> C51NetworkOutputs:
        x = x.float() / 255.0
        x = self.body(x)
        advantages = self.advantage_head(x)
        values = self.value_head(x)

        advantages = advantages.view(-1, self.action_dim, self.num_atoms)
        values = values.view(-1, 1, self.num_atoms)

        q_logits = values + (advantages - torch.mean(advantages, dim=1, keepdim=True))

        q_logits = q_logits.view(-1, self.action_dim, self.num_atoms)

        q_dist = F.softmax(q_logits, dim=-1)
        atoms = self.atoms[None, None, :].to(device=x.device)
        q_values = torch.sum(q_dist * atoms, dim=-1)

        return C51NetworkOutputs(q_logits=q_logits, q_values=q_values)

    def reset_noise(self) -> None:
        for module in list(chain(*zip(self.advantage_head.modules(), self.value_head.modules()))):
            if isinstance(module, common.NoisyLinear):
                module.reset_noise()


class QRDqnConvNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, quantiles: torch.Tensor):
        if action_dim < 1:
            raise ValueError(f'Expect action_dim to be a positive integer, got {action_dim}')
        if len(state_dim) != 3:
            raise ValueError(f'Expect state_dim to be a tuple with [C, H, W], got {state_dim}')
        if len(quantiles.shape) != 1:
            raise ValueError(f'Expect quantiles to be a 1D tensor, got {quantiles.shape}')

        super().__init__()

        self.action_dim = action_dim
        self.taus = quantiles
        self.num_taus = quantiles.size(0)

        self.body = common.NatureCnnBackboneNet(state_dim)

        self.value_head = nn.Sequential(
            nn.Linear(self.body.out_features, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim * self.num_taus),
        )

        common.initialize_weights(self)

    def forward(self, x: torch.Tensor) -> QRDqnNetworkOutputs:
        x = x.float() / 255.0
        x = self.body(x)
        q_dist = self.value_head(x)

        q_dist = q_dist.view(-1, self.num_taus, self.action_dim)
        q_values = torch.mean(q_dist, dim=1)

        return QRDqnNetworkOutputs(q_values=q_values, q_dist=q_dist)


class IqnConvNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, latent_dim: int):
        if action_dim < 1:
            raise ValueError(f'Expect action_dim to be a positive integer, got {action_dim}')
        if len(state_dim) != 3:
            raise ValueError(f'Expect state_dim to be a tuple with [C, H, W], got {state_dim}')
        if latent_dim < 1:
            raise ValueError(f'Expect latent_dim to be a positive integer, got {latent_dim}')

        super().__init__()

        self.action_dim = action_dim
        self.latent_dim = latent_dim

        self.pis = torch.arange(1, self.latent_dim + 1).float() * 3.141592653589793

        self.body = common.NatureCnnBackboneNet(state_dim)
        self.embedding_layer = nn.Linear(latent_dim, self.body.out_features)

        self.value_head = nn.Sequential(
            nn.Linear(self.body.out_features, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )

        common.initialize_weights(self)

    def sample_taus(self, batch_size: int, num_taus: int) -> torch.Tensor:
        taus = torch.rand((batch_size, num_taus)).to(dtype=torch.float32)
        assert taus.shape == (batch_size, num_taus)
        return taus

    def forward(self, x: torch.Tensor, num_taus: int = 64) -> IqnNetworkOutputs:
        batch_size = x.shape[0]

        x = x.float() / 255.0
        features = self.body(x)

        taus = self.sample_taus(batch_size, num_taus).to(device=x.device)

        pis = self.pis[None, None, :].to(device=x.device)
        tau_embedding = torch.cos(pis * taus[:, :, None])

        tau_embedding = tau_embedding.view(batch_size * num_taus, -1)
        tau_embedding = F.relu(self.embedding_layer(tau_embedding))

        tau_embedding = tau_embedding.view(batch_size, num_taus, -1)
        head_input = tau_embedding * features[:, None, :]

        head_input = head_input.view(-1, self.embedding_layer.out_features)

        q_dist = self.value_head(head_input)
        q_dist = q_dist.view(batch_size, -1, self.action_dim)
        q_values = torch.mean(q_dist, dim=1)
        return IqnNetworkOutputs(q_values=q_values, q_dist=q_dist, taus=taus)


class DrqnConvNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        if action_dim < 1:
            raise ValueError(f'Expect action_dim to be a positive integer, got {action_dim}')
        if len(state_dim) != 3:
            raise ValueError(f'Expect state_dim to be a tuple with [C, H, W], got {state_dim}')

        super().__init__()
        self.action_dim = action_dim
        self.body = common.NatureCnnBackboneNet(state_dim)

        self.lstm = nn.LSTM(input_size=self.body.out_features, hidden_size=256, num_layers=1, batch_first=True)

        self.value_head = nn.Sequential(
            nn.Linear(self.lstm.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )

        common.initialize_weights(self)

    def forward(self, x: torch.Tensor, hidden_s: None) -> RnnDqnNetworkOutputs:
        assert len(x.shape) == 5
        B = x.shape[0]
        T = x.shape[1]

        x = torch.flatten(x, 0, 1)
        x = x.float() / 255.0
        x = self.body(x)
        x = x.view(B, T, -1)

        x, hidden_s = self.lstm(x, hidden_s)

        x = torch.flatten(x, 0, 1)
        q_values = self.value_head(x)
        q_values = q_values.view(B, T, -1)
        return RnnDqnNetworkOutputs(q_values=q_values, hidden_s=hidden_s)

    def get_initial_hidden_state(self, batch_size: int) -> Tuple[torch.Tensor]:
        return tuple(torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size) for _ in range(2))


class R2d2DqnConvNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        if action_dim < 1:
            raise ValueError(f'Expect action_dim to be a positive integer, got {action_dim}')
        if len(state_dim) != 3:
            raise ValueError(f'Expect state_dim to be a tuple with [C, H, W], got {state_dim}')

        super().__init__()
        self.action_dim = action_dim

        self.body = common.NatureCnnBackboneNet(state_dim)

        core_output_size = self.body.out_features + self.action_dim + 1

        self.lstm = nn.LSTM(input_size=core_output_size, hidden_size=512, num_layers=1)

        self.advantage_head = nn.Sequential(
            nn.Linear(self.lstm.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )

        self.value_head = nn.Sequential(
            nn.Linear(self.lstm.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        common.initialize_weights(self)

    def forward(self, input_: RnnDqnNetworkInputs) -> RnnDqnNetworkOutputs:
        s_t = input_.s_t
        a_tm1 = input_.a_tm1
        r_t = input_.r_t
        hidden_s = input_.hidden_s

        T, B, *_ = s_t.shape
        x = torch.flatten(s_t, 0, 1)
        x = x.float() / 255.0
        x = self.body(x)
        x = x.view(T * B, -1)

        one_hot_a_tm1 = F.one_hot(a_tm1.view(T * B), self.action_dim).float().to(device=x.device)

        reward = r_t.view(T * B, 1)
        core_input = torch.cat([x, reward, one_hot_a_tm1], dim=-1)
        core_input = core_input.view(T, B, -1)

        if hidden_s is None:
            hidden_s = self.get_initial_hidden_state(batch_size=B)
            hidden_s = tuple(s.to(device=x.device) for s in hidden_s)

        x, hidden_s = self.lstm(core_input, hidden_s)

        x = torch.flatten(x, 0, 1)
        advantages = self.advantage_head(x)
        values = self.value_head(x)

        q_values = values + (advantages - torch.mean(advantages, dim=1, keepdim=True))
        q_values = q_values.view(T, B, -1)
        return RnnDqnNetworkOutputs(q_values=q_values, hidden_s=hidden_s)

    def get_initial_hidden_state(self, batch_size: int) -> Tuple[torch.Tensor]:
        return tuple(torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size) for _ in range(2))


class NguDqnConvNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, num_policies: int):
        super().__init__()
        self.action_dim = action_dim
        self.num_policies = num_policies

        self.body = common.NatureCnnBackboneNet(state_dim)

        core_output_size = self.body.out_features + self.num_policies + self.action_dim + 1 + 1

        self.lstm = nn.LSTM(input_size=core_output_size, hidden_size=512, num_layers=1)

        self.advantage_head = nn.Sequential(
            nn.Linear(self.lstm.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )

        self.value_head = nn.Sequential(
            nn.Linear(self.lstm.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        common.initialize_weights(self)

    def forward(self, input_: NguNetworkInputs) -> RnnDqnNetworkOutputs:
        s_t = input_.s_t
        a_tm1 = input_.a_tm1
        ext_r_t = input_.ext_r_t
        int_r_t = input_.int_r_t
        policy_index = input_.policy_index
        hidden_s = input_.hidden_s

        T, B, *_ = s_t.shape
        x = torch.flatten(s_t, 0, 1)
        x = x.float() / 255.0
        x = self.body(x)
        x = x.view(T * B, -1)

        one_hot_beta = F.one_hot(policy_index.view(T * B), self.num_policies).float().to(device=x.device)
        one_hot_a_tm1 = F.one_hot(a_tm1.view(T * B), self.action_dim).float().to(device=x.device)

        int_reward = int_r_t.view(T * B, 1)
        ext_reward = ext_r_t.view(T * B, 1)

        core_input = torch.cat([x, ext_reward, one_hot_a_tm1, int_reward, one_hot_beta], dim=-1)
        core_input = core_input.view(T, B, -1)

        if hidden_s is None:
            hidden_s = self.get_initial_hidden_state(batch_size=B)
            hidden_s = tuple(s.to(device=x.device) for s in hidden_s)

        x, hidden_s = self.lstm(core_input, hidden_s)

        x = torch.flatten(x, 0, 1)
        advantages = self.advantage_head(x)
        values = self.value_head(x)

        q_values = values + (advantages - torch.mean(advantages, dim=1, keepdim=True))
        q_values = q_values.view(T, B, -1)
        return RnnDqnNetworkOutputs(q_values=q_values, hidden_s=hidden_s)

    def get_initial_hidden_state(self, batch_size: int) -> Tuple[torch.Tensor]:
        return tuple(torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size) for _ in range(2))


class Agent57Conv2dNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, num_policies: int):
        super().__init__()

        self.ext_q = NguDqnConvNet(state_dim=state_dim, action_dim=action_dim, num_policies=num_policies)
        self.int_q = NguDqnConvNet(state_dim=state_dim, action_dim=action_dim, num_policies=num_policies)

    def forward(self, input_: Agent57NetworkInputs) -> Agent57NetworkOutputs:
        ext_input = NguNetworkInputs(
            s_t=torch.clone(input_.s_t),
            a_tm1=torch.clone(input_.a_tm1),
            ext_r_t=torch.clone(input_.ext_r_t),
            int_r_t=torch.clone(input_.int_r_t),
            policy_index=torch.clone(input_.policy_index),
            hidden_s=input_.ext_hidden_s,
        )

        int_input = NguNetworkInputs(
            s_t=input_.s_t,
            a_tm1=input_.a_tm1,
            ext_r_t=input_.ext_r_t,
            int_r_t=input_.int_r_t,
            policy_index=input_.policy_index,
            hidden_s=input_.int_hidden_s,
        )

        ext_output = self.ext_q(ext_input)
        int_output = self.int_q(int_input)

        return Agent57NetworkOutputs(
            ext_q_values=ext_output.q_values,
            int_q_values=int_output.q_values,
            ext_hidden_s=ext_output.hidden_s,
            int_hidden_s=int_output.hidden_s,
        )

    def get_initial_hidden_state(self, batch_size: int) -> Tuple[torch.Tensor]:
        ext_state = self.ext_q.get_initial_hidden_state(batch_size)
        int_state = self.int_q.get_initial_hidden_state(batch_size)

        return (ext_state, int_state)
