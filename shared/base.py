from typing import NamedTuple, Optional, Tuple, Union
import torch
import torch.nn.functional as F


class LossOutput(NamedTuple):
    loss: torch.Tensor
    extra: Optional[NamedTuple]


def assert_rank_and_dtype(tensor: torch.Tensor, rank: Union[int, Tuple[int]], dtype: Union[torch.dtype, Tuple[torch.dtype]]):
    assert_rank(tensor, rank)
    assert_dtype(tensor, dtype)


def assert_rank(tensor: torch.Tensor, rank: Union[int, Tuple[int]]) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise ValueError('Error in rank and/or compatibility check. The input tensor should be a valid torch.Tensor.')
    supported_rank = []
    if isinstance(rank, tuple):
        supported_rank = rank
    else:
        supported_rank.append(rank)
    if len(tensor.shape) not in supported_rank:
        raise ValueError(
            f'Error in rank and/or compatibility check. The input tensor should be rank {rank} torch.Tensor, got {tensor.shape}.'
        )


def assert_dtype(tensor: torch.Tensor, dtype: Union[torch.dtype, Tuple[torch.dtype]]) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise ValueError('Error in rank and/or compatibility check. The input tensor should be a valid torch.Tensor.')
    supported_dtype = []
    if isinstance(dtype, tuple):
        supported_dtype = dtype
    else:
        supported_dtype.append(dtype)
    if tensor.dtype not in supported_dtype:
        raise ValueError(f'Error in rank and/or compatibility check. The input tensor should be {dtype}, got {tensor.dtype}.')


def assert_batch_dimension(tensor: torch.Tensor, batch_dize: int, dim: int = 0) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise ValueError('Error in rank and/or compatibility check. The input tensor should be a valid torch.Tensor.')

    if tensor.shape[dim] != batch_dize:
        raise ValueError(
            f'Error in rank and/or compatibility check. The input tensor should have {batch_dize} entry on batch dimension {dim}, got {tensor.shape}.'
        )


def batched_index(values: torch.Tensor, indices: torch.Tensor, dim: int = -1, keepdims: bool = False) -> torch.Tensor:
    assert_rank(values, (2, 3))
    assert_rank_and_dtype(indices, (1, 2), torch.long)

    assert_batch_dimension(indices, values.shape[0], 0)

    if len(indices.shape) == 2:
        assert_batch_dimension(indices, values.shape[1], 1)

    one_hot_indices = F.one_hot(indices, values.shape[dim]).to(dtype=values.dtype)

    if len(values.shape) == 3 and len(one_hot_indices.shape) == 2:
        one_hot_indices = one_hot_indices.unsqueeze(1)
    return torch.sum(values * one_hot_indices, dim=dim, keepdims=keepdims)
