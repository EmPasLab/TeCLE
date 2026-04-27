import torch
import torch.nn.functional as F

import base


def identity(x: torch.Tensor) -> torch.Tensor:
    base.assert_dtype(x, (torch.float16, torch.float32, torch.float64))
    return x


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    base.assert_dtype(x, (torch.float16, torch.float32, torch.float64))
    return torch.sigmoid(x)


def logit(x: torch.Tensor) -> torch.Tensor:
    base.assert_dtype(x, (torch.float16, torch.float32, torch.float64))
    return -torch.log(1.0 / x - 1.0)


def signed_logp1(x: torch.Tensor) -> torch.Tensor:
    base.assert_dtype(x, (torch.float16, torch.float32, torch.float64))
    return torch.sign(x) * torch.log1p(torch.abs(x))


def signed_expm1(x: torch.Tensor) -> torch.Tensor:
    base.assert_dtype(x, (torch.float16, torch.float32, torch.float64))
    return torch.sign(x) * torch.expm1(torch.abs(x))


def signed_hyperbolic(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    base.assert_dtype(x, (torch.float16, torch.float32, torch.float64))
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + eps * x


def hyperbolic_sin(x: torch.Tensor) -> torch.Tensor:
    base.assert_dtype(x, (torch.float16, torch.float32, torch.float64))
    return torch.sinh(x)


def hyperbolic_arcsin(x: torch.Tensor) -> torch.Tensor:
    base.assert_dtype(x, (torch.float16, torch.float32, torch.float64))
    return torch.arcsinh(x)


def signed_parabolic(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    base.assert_dtype(x, (torch.float16, torch.float32, torch.float64))
    z = torch.sqrt(1 + 4 * eps * (eps + 1 + torch.abs(x))) / 2 / eps - 1 / 2 / eps
    return torch.sign(x) * (torch.square(z) - 1)


def power(x: torch.Tensor, p: float) -> torch.Tensor:
    base.assert_dtype(x, (torch.float16, torch.float32, torch.float64))
    q = torch.sqrt(torch.tensor(p))
    return torch.sign(x) * (torch.pow(torch.abs(x) / q + 1.0, p) - 1) / q


def transform_to_2hot(scalar: torch.Tensor, min_value: float, max_value: float, num_bins: int) -> torch.Tensor:
    scalar = torch.clamp(scalar, min_value, max_value)
    scalar_bin = (scalar - min_value) / (max_value - min_value) * (num_bins - 1)
    lower, upper = torch.floor(scalar_bin), torch.ceil(scalar_bin)
    lower_value = (lower / (num_bins - 1.0)) * (max_value - min_value) + min_value
    upper_value = (upper / (num_bins - 1.0)) * (max_value - min_value) + min_value
    p_lower = (upper_value - scalar) / (upper_value - lower_value + 1e-5)
    p_upper = 1 - p_lower
    lower_one_hot = F.one_hot(lower.long(), num_bins) * torch.unsqueeze(p_lower, -1)
    upper_one_hot = F.one_hot(upper.long(), num_bins) * torch.unsqueeze(p_upper, -1)
    return lower_one_hot + upper_one_hot


def transform_from_2hot(probs: torch.Tensor, min_value: float, max_value: float, num_bins: int) -> torch.Tensor:
    support_space = torch.linspace(min_value, max_value, num_bins)
    scalar = torch.sum(probs * torch.unsqueeze(support_space, 0), -1)
    return scalar


def compute_transformed_q(ext_q: torch.Tensor, int_q: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    if not isinstance(beta, torch.Tensor):
        beta = torch.tensor(beta).expand_as(int_q).to(device=ext_q.device)

    if len(beta.shape) < len(int_q.shape):
        beta = beta[..., None].expand_as(int_q)

    return signed_hyperbolic(signed_parabolic(ext_q) + beta * signed_parabolic(int_q))
