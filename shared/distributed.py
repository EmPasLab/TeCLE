from typing import List, Tuple
import numpy as np
import torch


def get_actor_exploration_epsilon(n: int) -> List[float]:
    assert 1 <= n
    return np.power(0.4, np.linspace(1.0, 8.0, num=n)).flatten().tolist()


def calculate_dist_priorities_from_td_error(td_errors: torch.Tensor, eta: float) -> np.ndarray:
    td_errors = torch.clone(td_errors).detach()
    abs_td_errors = torch.abs(td_errors)

    priorities = eta * torch.max(abs_td_errors, dim=0)[0] + (1 - eta) * torch.mean(abs_td_errors, dim=0)
    priorities = torch.clamp(priorities, min=0.0001, max=1000)
    priorities = priorities.cpu().numpy()

    return priorities


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_ngu_policy_betas(
    n: int,
    beta: float = 0.3,
):
    results = []
    for i in range(n):
        if i == 0:
            results.append(0.0)
        elif i == n - 1:
            results.append(beta)
        else:
            _beta_i = beta * sigmoid(10 * ((2 * i - (n - 2)) / (n - 2)))
            results.append(_beta_i)

    return results


def get_ngu_discount_gammas(n: int, gamma_max: float, gamma_min: float) -> float:
    results = []
    for i in range(n):
        _numerator = (n - 1 - i) * np.log(1 - gamma_max) + i * np.log(1 - gamma_min)
        _gamma_i = 1 - np.exp(_numerator / (n - 1))
        results.append(_gamma_i)
    return results


def get_ngu_policy_betas_and_discounts(
    num_policies: int,
    beta: float = 0.3,
    gamma_min: float = 0.99,
    gamma_max: float = 0.997,
) -> Tuple[List[float], List[float]]:
    beta_list = get_ngu_policy_betas(num_policies, beta)
    gamma_list = get_ngu_discount_gammas(num_policies, gamma_max, gamma_min)
    return (beta_list, gamma_list)
