import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import NamedTuple, Optional, Tuple


from networks import common


class ActorNetworkOutputs(NamedTuple):
    pi_logits: torch.Tensor


class CriticNetworkOutputs(NamedTuple):
    value: torch.Tensor


class ActorCriticNetworkOutputs(NamedTuple):
    pi_logits: torch.Tensor
    value: torch.Tensor


class ImpalaActorCriticNetworkOutputs(NamedTuple):
    pi_logits: torch.Tensor
    value: torch.Tensor
    hidden_s: torch.Tensor


class ImpalaActorCriticNetworkInputs(NamedTuple):
    s_t: torch.Tensor
    a_tm1: torch.Tensor
    r_t: torch.Tensor
    done: torch.Tensor
    hidden_s: Optional[Tuple[torch.Tensor]]


class RndActorCriticNetworkOutputs(NamedTuple):
    pi_logits: torch.Tensor
    int_baseline: torch.Tensor
    ext_baseline: torch.Tensor


class ActorMlpNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x: torch.Tensor) -> ActorNetworkOutputs:
        pi_logits = self.net(x)

        return ActorNetworkOutputs(pi_logits=pi_logits)


class CriticMlpNet(nn.Module):
    def __init__(self, state_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> CriticNetworkOutputs:
        value = self.net(x)
        return CriticNetworkOutputs(value=value)


class ActorCriticMlpNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.policy_head = nn.Sequential(

            nn.Linear(64, action_dim),
        )
        self.baseline_head = nn.Sequential(

            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> ActorCriticNetworkOutputs:
        features = self.body(x)

        pi_logits = self.policy_head(features)

        value = self.baseline_head(features)

        return ActorCriticNetworkOutputs(pi_logits=pi_logits, value=value)


class GaussianActorMlpNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

        self.mu_head = nn.Sequential(

            nn.Linear(hidden_size, action_dim),
        )

        self.logstd = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        features = self.body(x)

        pi_mu = self.mu_head(features)

        logstd = self.logstd.expand_as(pi_mu)
        pi_sigma = torch.exp(logstd)

        return pi_mu, pi_sigma


class GaussianCriticMlpNet(nn.Module):
    def __init__(self, state_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.net(x)

        return value


class ImpalaActorCriticMlpNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, use_lstm: bool = False) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_lstm = use_lstm

        self.body = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        core_output_size = 64 + self.action_dim + 1

        if self.use_lstm:
            self.lstm = nn.LSTM(input_size=core_output_size, hidden_size=64, num_layers=1)
            core_output_size = 64

        self.policy_head = nn.Sequential(
            nn.Linear(core_output_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

        self.baseline_head = nn.Sequential(
            nn.Linear(core_output_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def get_initial_hidden_state(self, batch_size: int) -> Tuple[torch.Tensor]:
        if self.use_lstm:

            return tuple(torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size) for _ in range(2))
        else:
            return tuple()

    def forward(self, input_: ImpalaActorCriticNetworkInputs) -> ImpalaActorCriticNetworkOutputs:
        s_t = input_.s_t
        a_tm1 = input_.a_tm1
        r_t = input_.r_t
        done = input_.done
        hidden_s = input_.hidden_s

        T, B, *_ = s_t.shape
        x = torch.flatten(s_t, 0, 1)

        x = self.body(x)

        one_hot_a_tm1 = F.one_hot(a_tm1.view(T * B), self.action_dim).float().to(device=x.device)
        rewards = torch.clamp(r_t, -1, 1).view(T * B, 1)
        core_input = torch.cat([x, rewards, one_hot_a_tm1], dim=-1)

        if self.use_lstm:
            assert done.dtype == torch.bool

            core_input = core_input.view(T, B, -1)
            lstm_output_list = []
            notdone = (~done).float()

            if hidden_s is None:
                hidden_s = self.get_initial_hidden_state(B)
                hidden_s = tuple(s.to(device=x.device) for s in hidden_s)

            for inpt, n_d in zip(core_input.unbind(), notdone.unbind()):

                n_d = n_d.view(1, -1, 1)
                hidden_s = tuple(n_d * s for s in hidden_s)
                output, hidden_s = self.lstm(inpt.unsqueeze(0), hidden_s)
                lstm_output_list.append(output)

            core_output = torch.flatten(torch.cat(lstm_output_list), 0, 1)
        else:
            core_output = core_input
            hidden_s = tuple()

        pi_logits = self.policy_head(core_output)

        value = self.baseline_head(core_output)

        pi_logits = pi_logits.view(T, B, self.action_dim)
        value = value.view(T, B)
        return ImpalaActorCriticNetworkOutputs(pi_logits=pi_logits, value=value, hidden_s=hidden_s)


class RndActorCriticMlpNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()

        self.body = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.policy_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

        self.ext_baseline_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.int_baseline_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> RndActorCriticNetworkOutputs:
        features = self.body(x)

        pi_logits = self.policy_head(features)

        ext_baseline = self.ext_baseline_head(features)
        int_baseline = self.int_baseline_head(features)

        return RndActorCriticNetworkOutputs(pi_logits=pi_logits, ext_baseline=ext_baseline, int_baseline=int_baseline)


class ActorConvNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()

        self.body = common.NatureCnnBackboneNet(state_dim)

        self.policy_head = nn.Sequential(
            nn.Linear(self.body.out_features, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )

        common.initialize_weights(self)

    def forward(self, x: torch.Tensor) -> ActorNetworkOutputs:
        x = x.float() / 255.0
        features = self.body(x)

        pi_logits = self.policy_head(features)
        return ActorNetworkOutputs(pi_logits=pi_logits)


class CriticConvNet(nn.Module):
    def __init__(self, state_dim: int) -> None:
        super().__init__()

        self.body = common.NatureCnnBackboneNet(state_dim)

        self.baseline_head = nn.Sequential(
            nn.Linear(self.body.out_features, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        common.initialize_weights(self)

    def forward(self, x: torch.Tensor) -> CriticNetworkOutputs:
        x = x.float() / 255.0
        features = self.body(x)

        value = self.baseline_head(features)
        return CriticNetworkOutputs(value=value)


class ActorCriticConvNet(nn.Module):
    def __init__(self, state_dim: tuple, action_dim: int) -> None:
        super().__init__()

        self.body = common.NatureCnnBackboneNet(state_dim)

        self.policy_head = nn.Sequential(
            nn.Linear(self.body.out_features, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )

        self.baseline_head = nn.Sequential(
            nn.Linear(self.body.out_features, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        common.initialize_weights(self)

    def forward(self, x: torch.Tensor) -> ActorCriticNetworkOutputs:
        x = x.float() / 255.0
        features = self.body(x)

        pi_logits = self.policy_head(features)

        value = self.baseline_head(features)

        return ActorCriticNetworkOutputs(pi_logits=pi_logits, value=value)


class ImpalaActorCriticConvNet(nn.Module):
    def __init__(self, state_dim: tuple, action_dim: int, use_lstm: bool = False) -> None:
        super().__init__()

        self.action_dim = action_dim
        self.use_lstm = use_lstm

        assert state_dim[1] == state_dim[2] == 84

        self.feat_convs = []
        self.resnet1 = []
        self.resnet2 = []

        self.convs = []

        input_channels = state_dim[0]
        for num_ch in [16, 32, 32]:
            feats_convs = []
            feats_convs.append(nn.Conv2d(in_channels=input_channels, out_channels=num_ch, kernel_size=3, stride=1, padding=1))
            feats_convs.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.feat_convs.append(nn.Sequential(*feats_convs))

            input_channels = num_ch

            for i in range(2):
                resnet_block = []
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                if i == 0:
                    self.resnet1.append(nn.Sequential(*resnet_block))
                else:
                    self.resnet2.append(nn.Sequential(*resnet_block))

        self.feat_convs = nn.ModuleList(self.feat_convs)
        self.resnet1 = nn.ModuleList(self.resnet1)
        self.resnet2 = nn.ModuleList(self.resnet2)

        self.fc = nn.Linear(3872, 256)

        core_output_size = self.fc.out_features + self.action_dim + 1

        if self.use_lstm:
            self.lstm = nn.LSTM(input_size=core_output_size, hidden_size=256, num_layers=1)
            core_output_size = 256

        self.policy_head = nn.Sequential(
            nn.Linear(core_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )

        self.baseline_head = nn.Sequential(
            nn.Linear(core_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        common.initialize_weights(self)

    def get_initial_hidden_state(self, batch_size: int) -> Tuple[torch.Tensor]:
        if self.use_lstm:

            return tuple(torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size) for _ in range(2))
        else:
            return tuple()

    def forward(self, input_: ImpalaActorCriticNetworkInputs) -> ImpalaActorCriticNetworkOutputs:
        s_t = input_.s_t
        a_tm1 = input_.a_tm1
        r_t = input_.r_t
        done = input_.done
        hidden_s = input_.hidden_s

        T, B, *_ = s_t.shape
        x = torch.flatten(s_t, 0, 1)
        x = x.float() / 255.0

        res_input = None
        for i, fconv in enumerate(self.feat_convs):
            x = fconv(x)
            res_input = x
            x = self.resnet1[i](x)
            x += res_input
            res_input = x
            x = self.resnet2[i](x)
            x += res_input

        x = F.relu(x)
        x = x.view(T * B, -1)
        x = F.relu(self.fc(x))

        one_hot_a_tm1 = F.one_hot(a_tm1.view(T * B), self.action_dim).float().to(device=x.device)
        rewards = torch.clamp(r_t, -1, 1).view(T * B, 1)
        core_input = torch.cat([x, rewards, one_hot_a_tm1], dim=-1)

        if self.use_lstm:
            assert done.dtype == torch.bool

            core_input = core_input.view(T, B, -1)
            lstm_output_list = []
            notdone = (~done).float()

            if hidden_s is None:
                hidden_s = self.get_initial_hidden_state(B)
                hidden_s = tuple(s.to(device=x.device) for s in hidden_s)

            for inpt, nd in zip(core_input.unbind(), notdone.unbind()):

                nd = nd.view(1, -1, 1)
                hidden_s = tuple(nd * s for s in hidden_s)
                output, hidden_s = self.lstm(inpt.unsqueeze(0), hidden_s)
                lstm_output_list.append(output)
            core_output = torch.flatten(torch.cat(lstm_output_list), 0, 1)
        else:
            core_output = core_input
            hidden_s = tuple()

        pi_logits = self.policy_head(core_output)

        value = self.baseline_head(core_output)

        pi_logits = pi_logits.view(T, B, self.action_dim)
        value = value.view(T, B)
        return ImpalaActorCriticNetworkOutputs(pi_logits=pi_logits, value=value, hidden_s=hidden_s)


class RndActorCriticConvNet(nn.Module):
    def __init__(self, state_dim: tuple, action_dim: int) -> None:
        super().__init__()
        c, h, w = state_dim
        h, w = common.calc_conv2d_output((h, w), 8, 4)
        h, w = common.calc_conv2d_output((h, w), 4, 2)
        h, w = common.calc_conv2d_output((h, w), 3, 1)
        conv2d_out_size = 64 * h * w

        self.body = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(conv2d_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 448),
            nn.ReLU(),
        )

        self.extra_policy_fc = nn.Linear(448, 448)
        self.extra_value_fc = nn.Linear(448, 448)

        self.policy_head = nn.Linear(448, action_dim)
        self.ext_value_head = nn.Linear(448, 1)
        self.int_value_head = nn.Linear(448, 1)

        for layer in self.body.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                layer.bias.data.zero_()

        for layer in [self.extra_policy_fc, self.extra_value_fc]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(0.1))
            layer.bias.data.zero_()

        for layer in [self.policy_head, self.ext_value_head, self.int_value_head]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(0.01))
            layer.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> RndActorCriticNetworkOutputs:
        x = x.float() / 255.0
        features = self.body(x)

        pi_features = features + F.relu(self.extra_policy_fc(features))
        pi_logits = self.policy_head(pi_features)

        value_features = features + F.relu(self.extra_value_fc(features))
        ext_baseline = self.ext_value_head(value_features)
        int_baseline = self.int_value_head(value_features)

        return RndActorCriticNetworkOutputs(pi_logits=pi_logits, ext_baseline=ext_baseline, int_baseline=int_baseline)


class GaussianRndActorCriticNetworkOutputs(NamedTuple):
    pi_mu: torch.Tensor
    pi_sigma: torch.Tensor
    ext_baseline: torch.Tensor
    int_baseline: torch.Tensor


class GaussianRndActorCriticMlpNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256) -> None:
        super().__init__()

        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

        self.policy_mu_head = nn.Linear(hidden_size, action_dim)
        self.logstd = nn.Parameter(torch.zeros(1, action_dim))

        self.ext_baseline_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        self.int_baseline_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> GaussianRndActorCriticNetworkOutputs:
        features = self.body(x)

        pi_mu = self.policy_mu_head(features)
        logstd = self.logstd.expand_as(pi_mu)
        pi_sigma = torch.exp(logstd)

        ext_baseline = self.ext_baseline_head(features)
        int_baseline = self.int_baseline_head(features)

        return GaussianRndActorCriticNetworkOutputs(
            pi_mu=pi_mu, pi_sigma=pi_sigma, ext_baseline=ext_baseline, int_baseline=int_baseline
        )
