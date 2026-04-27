import abc

from typing import NamedTuple, Text, Mapping, Iterable, Optional, Any
import numpy as np

Action = int


class TimeStep(NamedTuple):
    observation: Optional[np.ndarray]
    reward: Optional[float]
    done: Optional[bool]
    first: Optional[bool]
    info: Optional[
        Mapping[Text, Any]
    ]


class Agent(abc.ABC):
    agent_name: str
    step_t: int

    @abc.abstractmethod
    def step(self, timestep: TimeStep) -> Action:
        pass

    @abc.abstractmethod
    def reset(self) -> None:
        pass

    @property
    @abc.abstractmethod
    def statistics(self) -> Mapping[Text, float]:
        pass


class Learner(abc.ABC):
    agent_name: str
    step_t: int

    @abc.abstractmethod
    def step(self) -> Iterable[Mapping[Text, float]]:
        pass

    @abc.abstractmethod
    def reset(self) -> None:
        pass

    @abc.abstractmethod
    def received_item_from_queue(self, item: Any) -> None:
        pass

    @property
    @abc.abstractmethod
    def statistics(self) -> Mapping[Text, float]:
        pass
