import os
import datetime
import numpy as np
import cv2
import logging
import gymnasium as gym
from gymnasium.spaces import Box
from collections import deque
from pathlib import Path


class PicklableClipAction(gym.ActionWrapper):
    def action(self, action):
        return np.clip(action, self.action_space.low, self.action_space.high)


if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_


import type as types_lib

CLASSIC_ENV_NAMES = ['CartPole-v1', 'LunarLander-v2', 'MountainCar-v0', 'Acrobot-v1']


def unwrap(env):
    if hasattr(env, 'unwrapped'):
        return env.unwrapped
    elif hasattr(env, 'env'):
        return unwrap(env.env)
    elif hasattr(env, 'leg_env'):
        return unwrap(env.leg_env)
    else:
        return env


class NoopReset(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        return self.env.step(action)


class FireOnReset(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, action):
        return self.env.step(action)


class StickyAction(gym.Wrapper):
    def __init__(self, env, eps=0.25):
        gym.Wrapper.__init__(self, env)
        self.eps = eps
        self.last_action = 0

    def step(self, action):
        if np.random.uniform() < self.eps:
            action = self.last_action

        self.last_action = action
        return self.env.step(action)

    def reset(self, **kwargs):
        self.last_action = 0
        return self.env.reset(**kwargs)


class LifeLoss(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_terminated = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_terminated = done

        lives = self.env.unwrapped.ale.lives()

        if lives < self.lives and lives > 0:

            info['loss_life'] = True
        else:
            info['loss_life'] = False
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        if self.was_real_terminated:
            obs = self.env.reset(**kwargs)
        else:

            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxAndSkip(gym.Wrapper):
    def __init__(self, env, skip=4):
        gym.Wrapper.__init__(self, env)

        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break

        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ResizeAndGrayscaleFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True):
        super().__init__(env)

        assert self.observation_space.dtype == np.uint8 and len(self.observation_space.shape) == 3

        self.frame_width = width
        self.frame_height = height
        self.grayscale = grayscale
        num_channels = 1 if self.grayscale else 3

        self.observation_space = Box(
            low=0,
            high=255,
            shape=(self.frame_height, self.frame_width, num_channels),
            dtype=np.uint8,
        )

    def observation(self, obs):
        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (self.frame_width, self.frame_height), interpolation=cv2.INTER_AREA)

        if self.grayscale:
            obs = np.expand_dims(obs, -1)

        return obs


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shape = env.observation_space.shape
        self.observation_space = Box(low=0, high=255, shape=(shape[:-1] + (shape[-1] * k,)), dtype=env.observation_space.dtype)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class LazyFrames(object):
    def __init__(self, frames):
        self.dtype = frames[0].dtype
        self.shape = (frames[0].shape[0], frames[0].shape[1], len(frames))
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]


class ScaleFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


class VisitedRoomInfo(gym.Wrapper):
    def __init__(self, env, room_address):
        gym.Wrapper.__init__(self, env)
        self.room_address = room_address
        self.visited_rooms = set()

    def get_current_room(self):
        ram = unwrap(self.env).ale.getRAM()
        assert len(ram) == 128
        return int(ram[self.room_address])

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.visited_rooms.add(self.get_current_room())
        if done:
            info['episode_visited_rooms'] = len(self.visited_rooms)
            self.visited_rooms.clear()
        return obs, rew, done, info


class ObscureObservation(gym.ObservationWrapper):
    def __init__(self, env, epsilon: float = 0.0):
        super().__init__(env)
        if not 0.0 <= epsilon < 1.0:
            raise ValueError(f'Expect obscure epsilon should be between [0.0, 1), got {epsilon}')
        self._eps = epsilon

    def observation(self, obs):
        if self.env.unwrapped.np_random.random() <= self._eps:
            obs = np.zeros_like(obs, dtype=self.observation_space.dtype)
        return obs


class GymV26Compat:
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.spec = getattr(env, 'spec', None)
        self.metadata = getattr(env, 'metadata', {})
        self.reward_range = getattr(env, 'reward_range', (-float('inf'), float('inf')))

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            return result[0]
        return result

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            obs, rew, terminated, truncated, info = result
            return obs, rew, bool(terminated or truncated), info
        return result


    def close(self):
        return self.env.close()

    def render(self, mode='human'):
        return self.env.render(mode)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __getattr__(self, name):
        try:
            env = object.__getattribute__(self, 'env')
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return getattr(env, name)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)


class ClipRewardWithBound(gym.RewardWrapper):
    def __init__(self, env, bound):
        super().__init__(env)
        self.bound = bound

    def reward(self, reward):
        return None if reward is None else max(min(reward, self.bound), -self.bound)


class ObservationChannelFirst(gym.ObservationWrapper):
    def __init__(self, env, scale_obs):
        super().__init__(env)
        old_shape = env.observation_space.shape
        new_shape = (old_shape[-1], old_shape[0], old_shape[1])
        _low, _high = (0.0, 255) if not scale_obs else (0.0, 1.0)
        new_dtype = env.observation_space.dtype if not scale_obs else np.float32
        self.observation_space = Box(low=_low, high=_high, shape=new_shape, dtype=new_dtype)

    def observation(self, obs):
        obs = np.asarray(obs, dtype=self.observation_space.dtype).transpose(2, 0, 1)

        return np.ascontiguousarray(obs, dtype=self.observation_space.dtype)


class ObservationToNumpy(gym.ObservationWrapper):
    def observation(self, obs):
        return np.asarray(obs, dtype=self.observation_space.dtype)


class ClipObservationWithBound(gym.ObservationWrapper):
    def __init__(self, env, max_abs_value):
        super().__init__(env)
        self._max_abs_value = max_abs_value

    def observation(self, obs):
        return np.clip(obs, -self._max_abs_value, self._max_abs_value)


class RecordRawReward(gym.Wrapper):
    def step(self, action):
        result = self.env.step(action)

        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            info['raw_reward'] = reward
            return obs, reward, terminated, truncated, info
        obs, reward, done, info = result
        info['raw_reward'] = reward
        return obs, reward, done, info

    
def create_atari_environment(
    env_name: str,
    seed: int = 1,
    frame_skip: int = 4,
    frame_stack: int = 4,
    frame_height: int = 84,
    frame_width: int = 84,
    noop_max: int = 30,
    max_episode_steps: int = 108000,
    obscure_epsilon: float = 0.0,
    terminal_on_life_loss: bool = False,
    clip_reward: bool = True,
    sticky_action: bool = True,
    scale_obs: bool = False,
    channel_first: bool = True,
) -> gym.Env:
    if 'NoFrameskip' in env_name:
        raise ValueError(f'Environment name should not include NoFrameskip, got {env_name}')

    try:
        import ale_py
        gym.register_envs(ale_py)
    except (ImportError, AttributeError):
        pass

    env = gym.make(f'{env_name}NoFrameskip-v4')
    try:
        env.seed(seed)
    except AttributeError:
        env.unwrapped.reset(seed=seed)

    env = gym.wrappers.TimeLimit(env.env, max_episode_steps=None if max_episode_steps <= 0 else max_episode_steps)

    if noop_max > 0:
        env = NoopReset(env, noop_max=noop_max)
    if sticky_action:
        env = StickyAction(env)
    if frame_skip > 0:
        env = MaxAndSkip(env, skip=frame_skip)

    if obscure_epsilon > 0.0:
        env = ObscureObservation(env, obscure_epsilon)
    if terminal_on_life_loss:
        env = LifeLoss(env)

    env = ResizeAndGrayscaleFrame(env, width=frame_width, height=frame_height)

    if scale_obs:
        env = ScaleFrame(env)

    if clip_reward:
        env = RecordRawReward(env)
        env = ClipRewardWithBound(env, 1.0)

    if frame_stack > 1:
        env = FrameStack(env, frame_stack)
    if channel_first:
        env = ObservationChannelFirst(env, scale_obs)
    else:

        env = ObservationToNumpy(env)

    if 'Montezuma' in env_name or 'Pitfall' in env_name:
        env = VisitedRoomInfo(env, room_address=3 if 'Montezuma' in env_name else 1)

    return env


def create_classic_environment(
    env_name: str,
    seed: int = 1,
    max_abs_reward: int = None,
    obscure_epsilon: float = 0.0,
) -> gym.Env:
    env = gym.make(env_name)

    if max_abs_reward is not None:
        env = RecordRawReward(env)
        env = ClipRewardWithBound(env, abs(max_abs_reward))

    if obscure_epsilon > 0.0:
        env = ObscureObservation(env, obscure_epsilon)

    return env


def create_continuous_environment(
    env_name: str,
    seed: int = 1,
    max_abs_obs: int = 10,
    max_abs_reward: int = 10,
) -> gym.Env:
    env = gym.make(env_name, disable_env_checker=True)

    env = PicklableClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = RecordRawReward(env)
    env = gym.wrappers.NormalizeReward(env)

    env = GymV26Compat(env)

    try:
        env.seed(seed)
    except AttributeError:
        env.unwrapped.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


_ROBOTICS_ENVS = {'FetchReach', 'FetchPush', 'FetchSlide', 'FetchPickAndPlace'}


class DictObsToFlat(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_dim = env.observation_space['observation'].shape[0]
        goal_dim = env.observation_space['desired_goal'].shape[0]
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim + goal_dim,),
            dtype=np.float32,
        )

    def observation(self, obs):
        return np.concatenate([obs['observation'], obs['desired_goal']], axis=-1).astype(np.float32)


def create_robotics_environment(
    env_name: str,
    seed: int = 1,
    max_abs_obs: float = 200.0,
    max_abs_reward: float = 1.0,
) -> gym.Env:
    try:
        import gymnasium_robotics
        gym.register_envs(gymnasium_robotics)
    except ImportError:
        raise ImportError('gymnasium-robotics is not installed. Run: pip install gymnasium-robotics')

    env = gym.make(env_name, reward_type='sparse', disable_env_checker=True)
    env = DictObsToFlat(env)
    env = PicklableClipAction(env)
    env = RecordRawReward(env)
    env = ClipObservationWithBound(env, max_abs_obs)
    env = ClipRewardWithBound(env, max_abs_reward)
    env = GymV26Compat(env)

    try:
        env.unwrapped.reset(seed=seed)
    except Exception:
        pass
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def create_mujoco_environment(
    env_name: str,
    seed: int = 1,
    max_abs_obs: float = 10.0,
    max_abs_reward: float = 10.0,
) -> gym.Env:
    base_name = env_name.split('-')[0]
    if base_name in _ROBOTICS_ENVS:
        return create_robotics_environment(
            env_name=env_name,
            seed=seed,
            max_abs_obs=max_abs_obs,
            max_abs_reward=max_abs_reward,
        )
    return create_continuous_environment(
        env_name=env_name,
        seed=seed,
        max_abs_obs=max_abs_obs,
        max_abs_reward=max_abs_reward,
    )


def play_and_record_video(
    agent: types_lib.Agent,
    env: gym.Env,
    save_dir: str = './recordings',
) -> None:
    if not isinstance(agent, types_lib.Agent):
        raise RuntimeError('Expect agent to have a callable step() method.')

    if save_dir is not None and save_dir != '' and not os.path.exists(save_dir):
        _dir = Path(save_dir)
        _dir.mkdir(parents=True, exist_ok=False)

    assert os.path.exists(save_dir) and os.path.isdir(save_dir)

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    full_save_dir = os.path.join(save_dir, f'{agent.agent_name}_{env.spec.id}_{ts}')
    logging.info(f'Recording self-play video at "{full_save_dir}"')

    env = gym.wrappers.RecordVideo(env, full_save_dir)

    observation = env.reset()
    agent.reset()

    reward = 0.0
    done = False
    first_step = True

    t = 0

    while True:
        timestep_t = types_lib.TimeStep(
            observation=observation,
            reward=reward,
            done=done,
            first=first_step,
            info=None,
        )
        a_t = agent.step(timestep_t)
        observation, reward, done, _ = env.step(a_t)
        t += 1

        first_step = False
        if done:
            break

    env.close()
