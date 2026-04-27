from typing import Mapping, Tuple, Text, Any
import os
from pathlib import Path
import torch


class PyTorchCheckpoint:
    def __init__(
        self,
        environment_name: str,
        agent_name: str = 'RLAgent',
        save_dir: str = None,
        iteration: int = 0,
        file_ext: str = 'ckpt',
        restore_only: bool = False,
    ) -> None:
        self.save_dir = save_dir
        self.file_ext = file_ext
        self.base_path = None

        if not restore_only and self.save_dir is not None and self.save_dir != '':
            self.base_path = Path(self.save_dir)
            if not self.base_path.exists():
                self.base_path.mkdir(parents=True, exist_ok=True)

        self.state = AttributeDict()
        self.state.iteration = iteration
        self.state.environment_name = environment_name
        self.state.agent_name = agent_name

    def register_pair(self, pair: Tuple[Text, Any]) -> None:
        assert isinstance(pair, Tuple)

        key, item = pair
        self.state[key] = item

    def save(self) -> str:
        if self.base_path is None:
            return

        file_name = f'{self.state.agent_name}_{self.state.environment_name}_{self.state.iteration}.{self.file_ext}'
        save_path = self.base_path / file_name

        states = self._get_states_dict()
        torch.save(states, save_path)
        return save_path

    def restore(self, file_to_restore: str) -> None:
        if not file_to_restore or not os.path.isfile(file_to_restore) or not os.path.exists(file_to_restore):
            raise ValueError(f'"{file_to_restore}" is not a valid checkpoint file.')

        loaded_state = torch.load(file_to_restore, map_location=torch.device('cpu'))

        if loaded_state['environment_name'] != self.state.environment_name:
            err_msg = f'environment_name "{loaded_state["environment_name"]}" and "{self.state.environment_name}" mismatch.'
            raise RuntimeError(err_msg)
        if 'agent_name' in loaded_state and loaded_state['agent_name'] != self.state.agent_name:
            err_msg = f'agent_name "{loaded_state["agent_name"]}" and "{self.state.agent_name}" mismatch.'
            raise RuntimeError(err_msg)

        loaded_keys = [k for k in loaded_state.keys()]

        for key, item in self.state.items():

            if key not in loaded_keys:
                continue

            if self._is_torch_model(item):
                self.state[key].load_state_dict(loaded_state[key])
            else:
                self.state[key] = loaded_state[key]

    def set_iteration(self, iteration) -> None:
        self.state.iteration = iteration

    def get_iteration(self) -> int:
        return self.state.iteration

    def _get_states_dict(self) -> Mapping[Text, Any]:
        states_dict = {}

        for key, item in self.state.items():
            if self._is_torch_model(item):
                states_dict[key] = item.state_dict()
            else:
                states_dict[key] = item

        return states_dict

    def _is_torch_model(self, obj) -> bool:
        return isinstance(obj, (torch.nn.Module, torch.optim.Optimizer))


class AttributeDict(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        del self[key]
