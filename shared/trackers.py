import timeit
from pathlib import Path
import shutil
import collections
from typing import Any, Text, Iterable, Mapping, Optional, Tuple, Union
import numpy as np
from torch.utils.tensorboard import SummaryWriter


import replay as replay_lib


class EpisodeTracker:
    def __init__(self):
        self._num_steps_since_reset = None
        self._episode_returns = None
        self._episode_steps = None
        self._episode_visited_rooms = None
        self._current_episode_rewards = None
        self._current_episode_step = None

    def step(self, env, timestep_t, agent, a_t) -> None:
        del (env, agent, a_t)

        if timestep_t.first:
            if self._current_episode_rewards:
                raise ValueError('Current episode reward list should be empty.')
            if self._current_episode_step != 0:
                raise ValueError('Current episode step should be zero.')
        else:
            reward = timestep_t.reward

            if isinstance(timestep_t.info, dict) and 'raw_reward' in timestep_t.info:
                reward = timestep_t.info['raw_reward']

            self._current_episode_rewards.append(reward)

        self._num_steps_since_reset += 1
        self._current_episode_step += 1

        if timestep_t.done:
            self._episode_returns.append(sum(self._current_episode_rewards))
            self._episode_steps.append(self._current_episode_step)
            self._current_episode_rewards = []
            self._current_episode_step = 0

            if isinstance(timestep_t.info, dict) and 'episode_visited_rooms' in timestep_t.info:
                self._episode_visited_rooms.append(timestep_t.info['episode_visited_rooms'])

    def reset(self) -> None:
        self._num_steps_since_reset = 0
        self._episode_returns = []
        self._episode_steps = []
        self._episode_visited_rooms = []
        self._current_episode_step = 0
        self._current_episode_rewards = []

    def get(self) -> Mapping[str, Union[int, float, None]]:
        mean_episode_visited_rooms = 0

        if len(self._episode_returns) > 0:
            mean_episode_return = np.array(self._episode_returns).mean()

            if len(self._episode_visited_rooms) > 0:
                mean_episode_visited_rooms = np.array(self._episode_visited_rooms).mean()
        else:
            mean_episode_return = sum(self._current_episode_rewards)

        return {
            'mean_episode_return': mean_episode_return,
            'mean_episode_visited_rooms': mean_episode_visited_rooms,
            'num_episodes': len(self._episode_returns),
            'current_episode_step': self._current_episode_step,
            'num_steps_since_reset': self._num_steps_since_reset,
        }


class StepRateTracker:
    def __init__(self):
        self._num_steps_since_reset = None
        self._start = None

    def step(self, env, timestep_t, agent, a_t) -> None:
        del (env, timestep_t, agent, a_t)

        self._num_steps_since_reset += 1

    def reset(self) -> None:
        self._num_steps_since_reset = 0
        self._start = timeit.default_timer()

    def get(self) -> Mapping[Text, float]:
        duration = timeit.default_timer() - self._start
        if self._num_steps_since_reset > 0:
            step_rate = self._num_steps_since_reset / duration
        else:
            step_rate = np.nan
        return {
            'step_rate': step_rate,
            'num_steps_since_reset': self._num_steps_since_reset,
            'duration': duration,
        }


class TensorboardEpisodeTracker(EpisodeTracker):
    def __init__(self, writer: SummaryWriter):
        super().__init__()
        self._total_steps = 0
        self._total_episodes = 0
        self._writer = writer

    def step(self, env, timestep_t, agent, a_t) -> None:
        super().step(env, timestep_t, agent, a_t)

        self._total_steps += 1

        if timestep_t.done:
            self._total_episodes += 1
            tb_steps = self._total_steps

            episode_return = self._episode_returns[-1]
            episode_step = self._episode_steps[-1]

            self._writer.add_scalar('performance(env_steps)/num_episodes', self._total_episodes, tb_steps)
            self._writer.add_scalar('performance(env_steps)/episode_return', episode_return, tb_steps)
            self._writer.add_scalar('performance(env_steps)/episode_steps', episode_step, tb_steps)

            if isinstance(timestep_t.info, dict) and 'episode_visited_rooms' in timestep_t.info:
                episode_visited_rooms = self._episode_visited_rooms[-1]
                self._writer.add_scalar('performance(env_steps)/episode_visited_rooms', episode_visited_rooms, tb_steps)


class TensorboardStepRateTracker(StepRateTracker):
    def __init__(self, writer: SummaryWriter):
        super().__init__()

        self._total_steps = 0
        self._writer = writer

    def step(self, env, timestep_t, agent, a_t) -> None:
        super().step(env, timestep_t, agent, a_t)

        self._total_steps += 1

        if timestep_t.done:
            time_stats = self.get()
            self._writer.add_scalar('performance(env_steps)/step_rate', time_stats['step_rate'], self._total_steps)


class TensorboardAgentStatisticsTracker:
    def __init__(self, writer: SummaryWriter):
        self._total_steps = 0
        self._writer = writer

    def step(self, env, timestep_t, agent, a_t) -> None:
        del (env, a_t)
        self._total_steps += 1

        if timestep_t.done:
            try:
                stats = agent.statistics
                if stats:
                    for k, v in stats.items():
                        if isinstance(v, (int, float)):
                            self._writer.add_scalar(f'agent_statistics(env_steps)/{k}', v, self._total_steps)
            except Exception:
                pass

    def reset(self) -> None:
        pass

    def get(self) -> Mapping[Text, float]:
        return {}


class TensorboardScreenshotTracker:
    def __init__(self, writer: SummaryWriter, log_interval: int = 100):
        self._total_steps = 0
        self._total_episodes = 0
        self._log_interval = log_interval
        self._writer = writer

    def step(self, env, timestep_t, agent, a_t) -> None:
        del (agent, a_t)

        self._total_steps += 1

        if timestep_t.done:
            self._total_episodes += 1

            if self._total_episodes % self._log_interval == 0:
                try:
                    img = env.render(mode='rgb_array')
                    self._writer.add_image(
                        f'debug(episode)/episode_{self._total_episodes}',
                        img,
                        self._total_steps,
                        dataformats='HWC',
                    )
                except Exception:
                    pass

    def reset(self) -> None:
        pass

    def get(self) -> Mapping[Text, float]:
        return {}


class TensorboardLearnerStatisticsTracker:
    def __init__(self, writer: SummaryWriter):
        self._total_steps = 0
        self._num_steps_since_reset = 0
        self._start = timeit.default_timer()
        self._writer = writer

    def step(self, stats) -> None:
        self._total_steps += 1
        self._num_steps_since_reset += 1

        if self._total_steps % 100 == 0:
            time_stats = self.get()
            self._writer.add_scalar('learner_statistics(learner_steps)/step_rate', time_stats['step_rate'], self._total_steps)

            try:
                if stats:
                    for k, v in stats.items():
                        if isinstance(v, (int, float)):
                            self._writer.add_scalar(f'learner_statistics(learner_steps)/{k}', v, self._total_steps)
            except Exception:
                pass

    def reset(self) -> None:
        self._num_steps_since_reset = 0
        self._start = timeit.default_timer()

    def get(self) -> Mapping[Text, float]:
        duration = timeit.default_timer() - self._start
        if self._num_steps_since_reset > 0:
            step_rate = self._num_steps_since_reset / duration
        else:
            step_rate = np.nan
        return {
            'step_rate': step_rate,
            'num_steps_since_reset': self._num_steps_since_reset,
            'duration': duration,
        }


def make_default_trackers(log_dir=None, debug_screenshots_interval=0):
    if log_dir:
        log_dir = Path(f'runs/{log_dir}')

        if log_dir.exists() and log_dir.is_dir():
            shutil.rmtree(log_dir)

        writer = SummaryWriter(log_dir)

        trackers = [
            TensorboardEpisodeTracker(writer),
            TensorboardStepRateTracker(writer),
            TensorboardAgentStatisticsTracker(writer),
        ]

        if debug_screenshots_interval > 0:
            trackers.append(TensorboardScreenshotTracker(writer, debug_screenshots_interval))

        return trackers

    else:
        return [EpisodeTracker(), StepRateTracker()]


def make_learner_trackers(run_log_dir=None):
    if run_log_dir:
        tb_log_dir = Path(f'runs/{run_log_dir}')

        if tb_log_dir.exists() and tb_log_dir.is_dir():
            shutil.rmtree(tb_log_dir)

        writer = SummaryWriter(tb_log_dir)

        return [TensorboardLearnerStatisticsTracker(writer)]

    else:
        return []


def generate_statistics(
    trackers: Iterable[Any],
    timestep_action_sequence: Iterable[Tuple[Optional[replay_lib.Transition]]],
) -> Mapping[str, Any]:
    for tracker in trackers:
        tracker.reset()

    for env, timestep_t, agent, a_t in timestep_action_sequence:
        for tracker in trackers:
            tracker.step(env, timestep_t, agent, a_t)

    statistics_dicts = (tracker.get() for tracker in trackers)
    return dict(collections.ChainMap(*statistics_dicts))
