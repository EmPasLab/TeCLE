import numpy as np
from numpy.fft import irfft, rfftfreq
from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
from typing import NamedTuple


import base
from networks import common

class IcmNetworkOutput(NamedTuple):
    pi_logits: torch.Tensor
    features: torch.Tensor
    pred_features: torch.Tensor


class IcmMlpNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()

        self.action_dim = action_dim

        feature_vector_size = 128

        self.body = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, feature_vector_size),
            nn.ReLU(),
        )

        self.forward_net = nn.Sequential(
            nn.Linear(feature_vector_size + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_vector_size),
            nn.ReLU(),
        )

        self.inverse_net = nn.Sequential(
            nn.Linear(feature_vector_size * 2, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, s_tm1: torch.Tensor, a_tm1: torch.Tensor, s_t: torch.Tensor) -> IcmNetworkOutput:
        base.assert_rank(s_tm1, 2)
        base.assert_rank(s_t, 2)
        base.assert_rank(a_tm1, 1)

        a_tm1_onehot = F.one_hot(a_tm1, self.action_dim).float()

        features_tm1 = self.body(s_tm1)
        features_t = self.body(s_t)

        forward_input = torch.cat([features_tm1, a_tm1_onehot], dim=-1)
        pred_features_t = self.forward_net(forward_input)

        inverse_input = torch.cat([features_tm1, features_t], dim=-1)
        pi_logits_a_tm1 = self.inverse_net(inverse_input)

        return IcmNetworkOutput(pi_logits=pi_logits_a_tm1, pred_features=pred_features_t, features=features_t)


class IcmNatureConvNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()

        self.action_dim = action_dim

        c, h, w = state_dim
        h, w = common.calc_conv2d_output((h, w), 3, 2, 1)
        h, w = common.calc_conv2d_output((h, w), 3, 2, 1)
        h, w = common.calc_conv2d_output((h, w), 3, 2, 1)
        h, w = common.calc_conv2d_output((h, w), 3, 2, 1)
        conv2d_out_size = 32 * h * w

        self.body = self.net = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
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

    def forward(self, s_tm1: torch.Tensor, a_tm1: torch.Tensor, s_t: torch.Tensor) -> IcmNetworkOutput:
        base.assert_rank(s_tm1, (2, 4))
        base.assert_rank(s_t, (2, 4))
        base.assert_rank(a_tm1, 1)

        a_tm1_onehot = F.one_hot(a_tm1, self.action_dim).float()

        s_tm1 = s_tm1.float() / 255.0
        s_t = s_t.float() / 255.0
        features_tm1 = self.body(s_tm1)
        features_t = self.body(s_t)

        forward_input = torch.cat([features_tm1, a_tm1_onehot], dim=-1)
        pred_features_t = self.forward_net(forward_input) 

        inverse_input = torch.cat([features_tm1, features_t], dim=-1)
        pi_logits_a_tm1 = self.inverse_net(inverse_input)

        return IcmNetworkOutput(pi_logits=pi_logits_a_tm1, pred_features=pred_features_t, features=features_t)


class RndConvNet(nn.Module):
    def __init__(self, state_dim: int, is_target: bool = False, latent_dim: int = 256) -> None:
        super().__init__()

        c, h, w = state_dim
        h, w = common.calc_conv2d_output((h, w), 8, 4)
        h, w = common.calc_conv2d_output((h, w), 4, 2)
        h, w = common.calc_conv2d_output((h, w), 3, 1)
        conv2d_out_size = 64 * h * w

        self.body = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
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
                nn.init.orthogonal_(module.weight, np.sqrt(2))
                module.bias.data.zero_()

        if is_target:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.body(x)
        return self.head(x)

class ESAConvNet(nn.Module):
    def __init__(self, state_dim, action_dim: int = 18):
        super().__init__()

        c, h, w = state_dim

        self.embedd_s_t = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.pred_a_t = nn.Sequential(
            nn.Linear(21, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

        common.initialize_weights(self)

    def forward(self, x):
       
        s_t = self.embedd_s_t(x)
        a_t = self.pred_a_t(s_t)
        return s_t, a_t


class NguEmbeddingConvNet(nn.Module):
    def __init__(self, state_dim: tuple, action_dim: int):
        super().__init__()

        self.embed_size = 32

        self.net = common.NatureCnnBackboneNet(state_dim)

        self.fc = nn.Linear(self.net.out_features, 32)

        self.inverse_head = nn.Sequential(
            nn.Linear(32 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

        common.initialize_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0
        x = self.net(x)

        return F.relu(self.fc(x))

    def inverse_prediction(self, x: torch.Tensor) -> torch.Tensor:
        pi_logits = self.inverse_head(x)
        return pi_logits


class NGURndConvNet(nn.Module):
    def __init__(self, state_dim: int, latent_dim: int = 128, is_target: bool = False) -> None:
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
        )

        self.head = nn.Linear(conv2d_out_size, latent_dim)

        common.initialize_weights(self)

        if is_target:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.body(x)
        return self.head(x)


class GaussianIcmNetworkOutput(NamedTuple):
    pred_action: torch.Tensor
    features: torch.Tensor
    pred_features: torch.Tensor


class GaussianIcmMlpNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()

        self.action_dim = action_dim
        feature_vector_size = 128

        self.body = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, feature_vector_size),
            nn.ReLU(),
        )

        self.forward_net = nn.Sequential(
            nn.Linear(feature_vector_size + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_vector_size),
            nn.ReLU(),
        )

        self.inverse_net = nn.Sequential(
            nn.Linear(feature_vector_size * 2, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, s_tm1: torch.Tensor, a_tm1: torch.Tensor, s_t: torch.Tensor) -> GaussianIcmNetworkOutput:
        base.assert_rank(s_tm1, 2)
        base.assert_rank(s_t, 2)
        base.assert_rank(a_tm1, 2)

        features_tm1 = self.body(s_tm1)
        features_t = self.body(s_t)

        forward_input = torch.cat([features_tm1, a_tm1], dim=-1)
        pred_features_t = self.forward_net(forward_input)

        inverse_input = torch.cat([features_tm1, features_t], dim=-1)
        pred_action = self.inverse_net(inverse_input)

        return GaussianIcmNetworkOutput(pred_action=pred_action, pred_features=pred_features_t, features=features_t)


class RndMlpNet(nn.Module):
    def __init__(self, state_dim: int, is_target: bool = False, latent_dim: int = 128) -> None:
        super().__init__()

        if is_target:
            self.net = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.ReLU(),
                nn.Linear(256, latent_dim),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, latent_dim),
            )

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, np.sqrt(2))
                module.bias.data.zero_()

        if is_target:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DisagreeNetworkOutput(NamedTuple):
    features_tp1: torch.Tensor
    predictions: list


class DisagreeMlpNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, num_ensemble: int = 5) -> None:
        super().__init__()

        feature_dim = 128
        self._feature_dim = feature_dim
        self.num_ensemble = num_ensemble

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
            nn.ReLU(),
        )

        self.ensemble = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim + action_dim, 128),
                nn.ReLU(),
                nn.Linear(128, feature_dim),
            )
            for _ in range(num_ensemble)
        ])

    def forward(self, s_t: torch.Tensor, a_t: torch.Tensor, s_tp1: torch.Tensor) -> DisagreeNetworkOutput:
        base.assert_rank(s_t, 2)
        base.assert_rank(s_tp1, 2)
        base.assert_rank(a_t, 2)

        features_t = self.encoder(s_t)
        features_tp1 = self.encoder(s_tp1).detach()

        forward_input = torch.cat([features_t, a_t], dim=-1)
        predictions = [model(forward_input) for model in self.ensemble]

        return DisagreeNetworkOutput(features_tp1=features_tp1, predictions=predictions)


class AmaNetworkOutput(NamedTuple):
    pi_logits: torch.Tensor
    features: torch.Tensor
    pred_features_mu: torch.Tensor
    pred_features_log_var: torch.Tensor


class AmaNatureConvNet(nn.Module):
    def __init__(self, state_dim: tuple, action_dim: int) -> None:
        super().__init__()

        self.action_dim = action_dim

        c, h, w = state_dim
        h, w = common.calc_conv2d_output((h, w), 3, 2, 1)
        h, w = common.calc_conv2d_output((h, w), 3, 2, 1)
        h, w = common.calc_conv2d_output((h, w), 3, 2, 1)
        h, w = common.calc_conv2d_output((h, w), 3, 2, 1)
        conv2d_out_size = 32 * h * w

        self.body = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.forward_mu_net = nn.Sequential(
            nn.Linear(conv2d_out_size + self.action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, conv2d_out_size),
        )

        self.forward_log_var_net = nn.Sequential(
            nn.Linear(conv2d_out_size + self.action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, conv2d_out_size),
        )

        self.inverse_net = nn.Sequential(
            nn.Linear(conv2d_out_size * 2, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

        common.initialize_weights(self)

    def forward(self, s_tm1: torch.Tensor, a_tm1: torch.Tensor, s_t: torch.Tensor) -> AmaNetworkOutput:
        base.assert_rank(s_tm1, (2, 4))
        base.assert_rank(s_t, (2, 4))
        base.assert_rank(a_tm1, 1)

        a_tm1_onehot = F.one_hot(a_tm1, self.action_dim).float()

        s_tm1 = s_tm1.float() / 255.0
        s_t = s_t.float() / 255.0
        features_tm1 = self.body(s_tm1)
        features_t = self.body(s_t)

        forward_input = torch.cat([features_tm1, a_tm1_onehot], dim=-1)
        pred_mu = self.forward_mu_net(forward_input)
        pred_log_var = self.forward_log_var_net(forward_input)

        inverse_input = torch.cat([features_tm1, features_t], dim=-1)
        pi_logits = self.inverse_net(inverse_input)

        return AmaNetworkOutput(
            pi_logits=pi_logits,
            features=features_t,
            pred_features_mu=pred_mu,
            pred_features_log_var=pred_log_var,
        )


class GaussianAmaNetworkOutput(NamedTuple):
    pred_action: torch.Tensor
    features: torch.Tensor
    pred_features_mu: torch.Tensor
    pred_features_log_var: torch.Tensor


class GaussianAmaMlpNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()

        self.action_dim = action_dim
        feature_dim = 128

        self.body = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
            nn.ReLU(),
        )

        self.forward_mu_net = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
        )

        self.forward_log_var_net = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
        )

        self.inverse_net = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, s_tm1: torch.Tensor, a_tm1: torch.Tensor, s_t: torch.Tensor) -> GaussianAmaNetworkOutput:
        base.assert_rank(s_tm1, 2)
        base.assert_rank(s_t, 2)
        base.assert_rank(a_tm1, 2)

        features_tm1 = self.body(s_tm1)
        features_t = self.body(s_t)

        forward_input = torch.cat([features_tm1, a_tm1], dim=-1)
        pred_mu = self.forward_mu_net(forward_input)
        pred_log_var = self.forward_log_var_net(forward_input)

        inverse_input = torch.cat([features_tm1, features_t], dim=-1)
        pred_action = self.inverse_net(inverse_input)

        return GaussianAmaNetworkOutput(
            pred_action=pred_action,
            features=features_t,
            pred_features_mu=pred_mu,
            pred_features_log_var=pred_log_var,
        )


class LpmConvNet(nn.Module):
    def __init__(self, state_dim: tuple, action_dim: int, feature_dim: int = 512) -> None:
        super().__init__()

        c, h, w = state_dim
        h, w = common.calc_conv2d_output((h, w), 8, 4)
        h, w = common.calc_conv2d_output((h, w), 4, 2)
        h, w = common.calc_conv2d_output((h, w), 3, 1)
        conv_out_size = 64 * h * w

        self.action_dim = action_dim
        self.feature_dim = feature_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(conv_out_size, feature_dim),
            nn.LeakyReLU(),
        )

        self.dynamics_head = nn.Sequential(
            nn.Linear(feature_dim + action_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )

        self.error_head = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        common.initialize_weights(self)

    def encode(self, s: torch.Tensor) -> torch.Tensor:
        return self.encoder(s.float() / 255.0)

    def predict_next_features(self, features_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        a_onehot = F.one_hot(a_t, self.action_dim).float()
        x = torch.cat([features_t, a_onehot], dim=-1)
        return self.dynamics_head(x)

    def predict_log_error(self, features_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        a_onehot = F.one_hot(a_t, self.action_dim).float()
        x = torch.cat([features_t, a_onehot], dim=-1)
        return self.error_head(x)


class LpmMlpNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, feature_dim: int = 128) -> None:
        super().__init__()

        self.action_dim = action_dim
        self.feature_dim = feature_dim

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
            nn.ReLU(),
        )

        self.dynamics_head = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
        )

        self.error_head = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def encode(self, s: torch.Tensor) -> torch.Tensor:
        return self.encoder(s)

    def predict_next_features(self, features_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        x = torch.cat([features_t, a_t], dim=-1)
        return self.dynamics_head(x)

    def predict_log_error(self, features_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        x = torch.cat([features_t, a_t], dim=-1)
        return self.error_head(x)


def powerlaw_psd_gaussian(exponent, size, fmin=0, rng=None):
    try:
        size = list(size)
    except TypeError:
        size = [size]

    samples = size[-1]
    f = rfftfreq(samples)

    if 0 <= fmin <= 0.5:
        fmin = max(fmin, 1./samples)
    else:
        raise ValueError("fmin must be chosen between 0 and 0.5.")

    s_scale = f
    ix = np.sum(s_scale < fmin)
    if ix and ix < len(s_scale):
        s_scale[:ix] = s_scale[ix]
    s_scale = s_scale ** (-exponent / 2.)

    w = s_scale[1:].copy()
    w[-1] *= (1 + (samples % 2)) / 2.
    sigma = 2 * np.sqrt(np.sum(w ** 2)) / samples

    size[-1] = len(f)

    dims_to_add = len(size) - 1
    s_scale = s_scale[(None,) * dims_to_add + (Ellipsis,)]

    if rng is None:
        rng = np.random.default_rng()
    sr = rng.normal(scale=s_scale, size=size)
    si = rng.normal(scale=s_scale, size=size)

    if not (samples % 2):
        si[..., -1] = 0
        sr[..., -1] *= np.sqrt(2)

    si[..., 0] = 0
    sr[..., 0] *= np.sqrt(2)

    s = sr + 1J * si
    return irfft(s, n=samples, axis=-1) / sigma


class _TecleFlatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class _TecleMLP(nn.Module):
    def __init__(self, hidden_size, last_activation=True):
        super().__init__()
        q = []
        for i in range(len(hidden_size) - 1):
            in_dim = hidden_size[i]
            out_dim = hidden_size[i + 1]
            q.append(("Linear_%d" % i, nn.Linear(in_dim, out_dim)))
            if (i < len(hidden_size) - 2) or ((i == len(hidden_size) - 2) and last_activation):
                q.append(("BatchNorm_%d" % i, nn.BatchNorm1d(out_dim)))
                q.append(("ReLU_%d" % i, nn.ReLU(inplace=True)))
        self.mlp = nn.Sequential(OrderedDict(q))

    def forward(self, x):
        return self.mlp(x)


class _TecleEncoder(nn.Module):
    def __init__(self, shape, nhid=16, ncond=0):
        super().__init__()
        c, h, w = shape

        h, w = common.calc_conv2d_output((h, w), 1, 1, 0)
        conv2d_out_size = 64 * h * w

        self.encode = nn.Sequential(
            nn.Conv2d(c, 32, 1, padding=0), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 1, padding=0), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 1, padding=0), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            _TecleFlatten(),
            _TecleMLP([conv2d_out_size, 256, 128]),
        )
        self.calc_mean = _TecleMLP([128 + ncond, 64, nhid], last_activation=False)
        self.calc_logvar = _TecleMLP([128 + ncond, 64, nhid], last_activation=False)

    def forward(self, x, y=None):
        x = self.encode(x)
        if y is None:
            return self.calc_mean(x), self.calc_logvar(x)
        return self.calc_mean(torch.cat((x, y), dim=1)), self.calc_logvar(torch.cat((x, y), dim=1))


class _TecleDecoder(nn.Module):
    def __init__(self, shape, nhid=16, ncond=0):
        super().__init__()
        c, w, h = shape
        self.shape = shape
        self.decode = nn.Sequential(
            _TecleMLP([nhid + ncond, 64, 128, 256, c * w * h], last_activation=False),
            nn.Sigmoid(),
        )

    def forward(self, z, y=None):
        c, w, h = self.shape
        if y is None:
            return self.decode(z).view(-1, c, w, h)
        return self.decode(torch.cat((z, y), dim=1)).view(-1, c, w, h)


class TecleInverseConvNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        c, h, w = state_dim
        h, w = common.calc_conv2d_output((h, w), 3, 2, 1)
        h, w = common.calc_conv2d_output((h, w), 3, 2, 1)
        h, w = common.calc_conv2d_output((h, w), 3, 2, 1)
        h, w = common.calc_conv2d_output((h, w), 3, 2, 1)

        conv2d_out_size = 32 * h * w
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.flat = nn.Flatten()

        self.inverse_net = nn.Sequential(
            nn.Linear(conv2d_out_size * 2, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

        common.initialize_weights(self)

    def forward(self, s_t, s_tm1):
        features_tm1 = self.body(s_tm1)
        flat_features_tm1 = self.flat(features_tm1)
        features_t = self.body(s_t)
        flat_features_t = self.flat(features_t)

        inverse_input = torch.cat([flat_features_tm1, flat_features_t], dim=-1)
        pi_logits = self.inverse_net(inverse_input)

        return pi_logits, features_t


class TecleCVAEConvNet(nn.Module):
    def __init__(self, state_dim, nclass, nhid=16, ncond=16, noise_beta=0.5):
        super().__init__()
        self.dim = nhid

        c, h, w = state_dim

        h, w = common.calc_conv2d_output((h, w), 3, 2, 1)
        h, w = common.calc_conv2d_output((h, w), 3, 2, 1)
        h, w = common.calc_conv2d_output((h, w), 3, 2, 1)
        h, w = common.calc_conv2d_output((h, w), 3, 2, 1)

        body_out_size = (32, h, w)

        self.encoder = _TecleEncoder(body_out_size, nhid, ncond=ncond)
        self.decoder = _TecleDecoder(body_out_size, nhid, ncond=ncond)

        common.initialize_weights(self)

        self.label_embedding = nn.Embedding(nclass, ncond)
        self.noise_beta = noise_beta

    def sampling(self, mean, logvar):
        noise = powerlaw_psd_gaussian(self.noise_beta, mean.shape)
        eps = torch.tensor(noise, dtype=torch.float32, device=mean.device)
        sigma = torch.exp(0.5 * logvar)
        return mean + eps * sigma

    def forward(self, features_t, a_tm1):
        y = self.label_embedding(a_tm1)
        mean, logvar = self.encoder(features_t, y)
        z = self.sampling(mean, logvar)
        return self.decoder(z, y), mean, logvar


def tecle_cvae_loss(X, X_hat, mean, logvar):
    X = F.sigmoid(X)
    reconstruction_loss = F.binary_cross_entropy(X_hat, X)
    KL_divergence = -0.5 * torch.sum(1 + logvar - torch.exp(logvar) - mean ** 2)
    return reconstruction_loss + KL_divergence


class TecleInverseMlpNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.inverse_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )
        common.initialize_weights(self)

    def forward(self, s_t, s_tm1):
        feat_tm1 = self.body(s_tm1)
        feat_t = self.body(s_t)
        pred_action = self.inverse_net(torch.cat([feat_tm1, feat_t], dim=-1))
        return pred_action, feat_t


class TecleCVAEMlpNet(nn.Module):
    def __init__(self, feat_dim: int, action_dim: int, nhid: int = 16, noise_beta: float = 0.0) -> None:
        super().__init__()
        self.nhid = nhid
        self.noise_beta = noise_beta
        ncond = 16

        self.action_proj = nn.Sequential(nn.Linear(action_dim, ncond), nn.ReLU())

        self.encoder_body = nn.Sequential(
            nn.Linear(feat_dim + ncond, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.calc_mean = nn.Linear(64, nhid)
        self.calc_logvar = nn.Linear(64, nhid)

        self.decoder = nn.Sequential(
            nn.Linear(nhid + ncond, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, feat_dim),
            nn.Sigmoid(),
        )

        common.initialize_weights(self)

    def sampling(self, mean, logvar, eps=None):
        if eps is None:
            eps = torch.randn_like(mean)
        sigma = torch.exp(0.5 * logvar)
        return mean + eps * sigma

    def forward(self, features_t, a_tm1, eps=None):
        y = self.action_proj(a_tm1)
        h = self.encoder_body(torch.cat([features_t, y], dim=-1))
        mean = self.calc_mean(h)
        logvar = self.calc_logvar(h)
        z = self.sampling(mean, logvar, eps=eps)
        recon = self.decoder(torch.cat([z, y], dim=-1))
        return recon, mean, logvar


def gaussian_tecle_cvae_loss(features, recon, mean, logvar):
    reconstruction_loss = F.mse_loss(recon, features, reduction='sum')
    KL_divergence = -0.5 * torch.sum(1 + logvar - torch.exp(logvar) - mean ** 2)
    return reconstruction_loss + KL_divergence
