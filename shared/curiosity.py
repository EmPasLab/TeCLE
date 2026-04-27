from typing import NamedTuple, Dict
import numpy as np
import torch
import base
import normalizer


class KNNQueryResult(NamedTuple):
    neighbors: torch.Tensor
    neighbor_indices: torch.Tensor
    neighbor_distances: torch.Tensor


def knn_query(current: torch.Tensor, memory: torch.Tensor, num_neighbors: int) -> KNNQueryResult:
    base.assert_rank_and_dtype(current, 1, torch.float32)
    base.assert_rank_and_dtype(memory, 2, torch.float32)
    base.assert_batch_dimension(current, memory.shape[-1], -1)

    assert memory.shape[0] >= num_neighbors

    distances = torch.cdist(current.unsqueeze(0), memory).squeeze(0).pow(2)

    distances, indices = distances.topk(num_neighbors, largest=False)
    neighbors = torch.stack([memory[i] for i in indices], dim=0)
    return KNNQueryResult(neighbors=neighbors, neighbor_indices=indices, neighbor_distances=distances)


class EpisodicBonusModule:
    def __init__(
        self,
        embedding_network: torch.nn.Module,
        device: torch.device,
        capacity: int,
        num_neighbors: int,
        kernel_epsilon: float = 0.0001,
        cluster_distance: float = 0.008,
        max_similarity: float = 8.0,
        c_constant: float = 0.001,
    ) -> None:
        self._embedding_network = embedding_network.to(device=device)
        self._device = device

        self._memory = torch.zeros(
            capacity, self._embedding_network.embed_size, device=self._device
        )
        self._mask = torch.zeros(capacity, dtype=torch.bool, device=self._device)

        self._capacity = capacity
        self._counter = 0

        self._cdist_normalizer = normalizer.TorchRunningMeanStd(shape=(1,), device=self._device)

        self._num_neighbors = num_neighbors
        self._kernel_epsilon = kernel_epsilon
        self._cluster_distance = cluster_distance
        self._max_similarity = max_similarity
        self._c_constant = c_constant

    def _add_to_memory(self, embedding: torch.Tensor) -> None:
        idx = self._counter % self._capacity
        self._memory[idx] = embedding
        self._mask[idx] = True
        self._counter += 1

    @torch.no_grad()
    def compute_bonus(self, s_t: torch.Tensor) -> float:
        base.assert_rank_and_dtype(s_t, (2, 4), torch.float32)

        embedding = self._embedding_network(s_t).squeeze(0)

        prev_mask = self._mask.clone()

        self._add_to_memory(embedding)

        if self._counter <= self._num_neighbors:
            return 0.0

        knn_query_result = knn_query(embedding, self._memory[prev_mask], self._num_neighbors)

        nn_distances_sq = knn_query_result.neighbor_distances

        self._cdist_normalizer.update_single(nn_distances_sq)

        distance_rate = nn_distances_sq / (self._cdist_normalizer.mean + 1e-8)

        distance_rate = torch.max((distance_rate - self._cluster_distance), torch.tensor(0.0))

        kernel_output = self._kernel_epsilon / (distance_rate + self._kernel_epsilon)

        similarity = torch.sqrt(torch.sum(kernel_output)) + self._c_constant

        if torch.isnan(similarity):
            return 0.0

        if similarity > self._max_similarity:
            return 0.0

        return (1 / similarity).cpu().item()

    def reset(self):
        self._mask = torch.zeros(self._capacity, dtype=torch.bool, device=self._device)
        self._counter = 0

    def update_embedding_network(self, state_dict: Dict) -> None:
        self._embedding_network.load_state_dict(state_dict)


class RndLifeLongBonusModule:
    def __init__(
        self, target_network: torch.nn.Module, predictor_network: torch.nn.Module, device: torch.device, discount: float
    ) -> None:
        self._target_network = target_network.to(device=device)
        self._predictor_network = predictor_network.to(device=device)
        self._device = device
        self._discount = discount

        self._int_reward_normalizer = normalizer.RunningMeanStd(shape=(1,))
        self._rnd_obs_normalizer = normalizer.TorchRunningMeanStd(shape=(1, 84, 84), device=self._device)

    @torch.no_grad()
    def _normalize_rnd_obs(self, rnd_obs):
        rnd_obs = rnd_obs.to(device=self._device, dtype=torch.float32)

        normed_obs = self._rnd_obs_normalizer.normalize(rnd_obs)

        normed_obs = normed_obs.clamp(-5, 5)

        self._rnd_obs_normalizer.update_single(rnd_obs)

        return normed_obs

    def _normalize_int_rewards(self, int_rewards):
        self._int_reward_normalizer.update_single(int_rewards)

        normed_int_rewards = int_rewards / np.sqrt(self._int_reward_normalizer.var + 1e-8)

        return normed_int_rewards.item()

    @torch.no_grad()
    def compute_bonus(self, s_t: torch.Tensor) -> float:
        base.assert_rank_and_dtype(s_t, (2, 4), torch.float32)

        normed_s_t = self._normalize_rnd_obs(s_t)

        pred = self._predictor_network(normed_s_t)
        target = self._target_network(normed_s_t)

        int_r_t = torch.square(pred - target).mean(dim=1).detach().cpu().numpy()

        normed_int_r_t = self._normalize_int_rewards(int_r_t)

        return normed_int_r_t

    def update_predictor_network(self, state_dict: Dict) -> None:
        self._predictor_network.load_state_dict(state_dict)
