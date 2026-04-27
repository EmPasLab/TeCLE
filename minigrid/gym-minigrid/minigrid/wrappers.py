from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import logger, spaces
from gymnasium.core import ActionWrapper, ObservationWrapper, ObsType, Wrapper

from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX, STATE_TO_IDX
from minigrid.core.world_object import Goal


class ReseedWrapper(Wrapper):

    def __init__(self, env, seeds=(0,), seed_idx=0):
        self.seeds = list(seeds)
        self.seed_idx = seed_idx
        super().__init__(env)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        if seed is not None:
            logger.warn(
                "A seed has been passed to `ReseedWrapper.reset` which is ignored."
            )
        seed = self.seeds[self.seed_idx]
        self.seed_idx = (self.seed_idx + 1) % len(self.seeds)
        return self.env.reset(seed=seed, options=options)


class ActionBonus(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.counts = {}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        env = self.unwrapped
        tup = (tuple(env.agent_pos), env.agent_dir, action)


        pre_count = 0
        if tup in self.counts:
            pre_count = self.counts[tup]


        new_count = pre_count + 1
        self.counts[tup] = new_count

        bonus = 1 / math.sqrt(new_count)
        reward += bonus

        return obs, reward, terminated, truncated, info


class PositionBonus(Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.counts = {}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)


        env = self.unwrapped
        tup = tuple(env.agent_pos)


        pre_count = 0
        if tup in self.counts:
            pre_count = self.counts[tup]


        new_count = pre_count + 1
        self.counts[tup] = new_count

        bonus = 1 / math.sqrt(new_count)
        reward += bonus

        return obs, reward, terminated, truncated, info


class ImgObsWrapper(ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space.spaces["image"]

    def observation(self, obs):
        return obs["image"]


class OneHotPartialObsWrapper(ObservationWrapper):

    def __init__(self, env, tile_size=8):
        super().__init__(env)

        self.tile_size = tile_size

        obs_shape = env.observation_space["image"].shape


        num_bits = len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + len(STATE_TO_IDX)

        new_image_space = spaces.Box(
            low=0, high=255, shape=(obs_shape[0], obs_shape[1], num_bits), dtype="uint8"
        )
        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, obs):
        img = obs["image"]
        out = np.zeros(self.observation_space.spaces["image"].shape, dtype="uint8")

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                type = img[i, j, 0]
                color = img[i, j, 1]
                state = img[i, j, 2]

                out[i, j, type] = 1
                out[i, j, len(OBJECT_TO_IDX) + color] = 1
                out[i, j, len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + state] = 1

        return {**obs, "image": out}


class RGBImgObsWrapper(ObservationWrapper):

    def __init__(self, env, tile_size=8):
        super().__init__(env)

        self.tile_size = tile_size

        new_image_space = spaces.Box(
            low=0,
            high=255,
            shape=(
                self.unwrapped.width * tile_size,
                self.unwrapped.height * tile_size,
                3,
            ),
            dtype="uint8",
        )

        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, obs):
        rgb_img = self.get_frame(
            highlight=self.unwrapped.highlight, tile_size=self.tile_size
        )

        return {**obs, "image": rgb_img}


class RGBImgPartialObsWrapper(ObservationWrapper):

    def __init__(self, env, tile_size=8):
        super().__init__(env)


        self.tile_size = tile_size

        obs_shape = env.observation_space.spaces["image"].shape
        new_image_space = spaces.Box(
            low=0,
            high=255,
            shape=(obs_shape[0] * tile_size, obs_shape[1] * tile_size, 3),
            dtype="uint8",
        )

        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, obs):
        rgb_img_partial = self.get_frame(tile_size=self.tile_size, agent_pov=True)

        return {**obs, "image": rgb_img_partial}


class FullyObsWrapper(ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)

        new_image_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width, self.env.height, 3),
            dtype="uint8",
        )

        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, obs):
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array(
            [OBJECT_TO_IDX["agent"], COLOR_TO_IDX["red"], env.agent_dir]
        )

        return {**obs, "image": full_grid}


class DictObservationSpaceWrapper(ObservationWrapper):

    def __init__(self, env, max_words_in_mission=50, word_dict=None):
        super().__init__(env)

        if word_dict is None:
            word_dict = self.get_minigrid_words()

        self.max_words_in_mission = max_words_in_mission
        self.word_dict = word_dict

        self.observation_space = spaces.Dict(
            {
                "image": env.observation_space["image"],
                "direction": spaces.Discrete(4),
                "mission": spaces.MultiDiscrete(
                    [len(self.word_dict.keys())] * max_words_in_mission
                ),
            }
        )

    @staticmethod
    def get_minigrid_words():
        colors = ["red", "green", "blue", "yellow", "purple", "grey"]
        objects = [
            "unseen",
            "empty",
            "wall",
            "floor",
            "box",
            "key",
            "ball",
            "door",
            "goal",
            "agent",
            "lava",
        ]

        verbs = [
            "pick",
            "avoid",
            "get",
            "find",
            "put",
            "use",
            "open",
            "go",
            "fetch",
            "reach",
            "unlock",
            "traverse",
        ]

        extra_words = [
            "up",
            "the",
            "a",
            "at",
            ",",
            "square",
            "and",
            "then",
            "to",
            "of",
            "rooms",
            "near",
            "opening",
            "must",
            "you",
            "matching",
            "end",
            "hallway",
            "object",
            "from",
            "room",
            "maze",
        ]

        all_words = colors + objects + verbs + extra_words
        assert len(all_words) == len(set(all_words))
        return {word: i for i, word in enumerate(all_words)}

    def string_to_indices(self, string, offset=1):
        indices = []

        string = string.replace(",", " , ")
        for word in string.split():
            if word in self.word_dict.keys():
                indices.append(self.word_dict[word] + offset)
            else:
                raise ValueError(f"Unknown word: {word}")
        return indices

    def observation(self, obs):
        obs["mission"] = self.string_to_indices(obs["mission"])
        assert len(obs["mission"]) < self.max_words_in_mission
        obs["mission"] += [0] * (self.max_words_in_mission - len(obs["mission"]))

        return obs


class FlatObsWrapper(ObservationWrapper):

    def __init__(self, env, maxStrLen: int = 96):
        super().__init__(env)

        self.maxStrLen = maxStrLen
        self.numCharCodes = 28

        img_size = np.prod(env.observation_space["image"].shape)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(img_size + self.numCharCodes * self.maxStrLen,),
            dtype="uint8",
        )

        self.cachedStr: str = None

    def observation(self, obs):
        image = obs["image"]
        mission = obs["mission"]


        if mission != self.cachedStr:
            assert (
                len(mission) <= self.maxStrLen
            ), f"mission string too long ({len(mission)} chars)"
            mission = mission.lower()

            str_array = np.zeros(shape=(self.maxStrLen, self.numCharCodes), dtype="uint8")


            for idx, ch in enumerate(mission):
                if "a" <= ch <= "z":
                    chNo = ord(ch) - ord("a")
                elif ch == " ":
                    chNo = ord("z") - ord("a") + 1
                elif ch == ",":
                    chNo = ord("z") - ord("a") + 2
                else:
                    raise ValueError(
                        f"Character {ch} is not available in mission string."
                    )
                assert chNo < self.numCharCodes, f"{ch} : {chNo:d}"
                str_array[idx, chNo] = 1

            self.cachedStr = mission
            self.cachedArray = str_array

        obs = np.concatenate((image.flatten(), self.cachedArray.flatten()))

        return obs


class ViewSizeWrapper(ObservationWrapper):

    def __init__(self, env, agent_view_size=7):
        super().__init__(env)

        assert agent_view_size % 2 == 1
        assert agent_view_size >= 3

        self.agent_view_size = agent_view_size


        new_image_space = gym.spaces.Box(
            low=0, high=255, shape=(agent_view_size, agent_view_size, 3), dtype="uint8"
        )


        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, obs):
        env = self.unwrapped

        grid, vis_mask = env.gen_obs_grid(self.agent_view_size)


        image = grid.encode(vis_mask)

        return {**obs, "image": image}


class DirectionObsWrapper(ObservationWrapper):

    def __init__(self, env, type="slope"):
        super().__init__(env)
        self.goal_position: tuple = None
        self.type = type

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        obs, info = self.env.reset()

        if not self.goal_position:
            self.goal_position = [
                x for x, y in enumerate(self.grid.grid) if isinstance(y, Goal)
            ]

            if len(self.goal_position) >= 1:
                self.goal_position = (
                    int(self.goal_position[0] / self.height),
                    self.goal_position[0] % self.width,
                )

        return self.observation(obs), info

    def observation(self, obs):
        slope = np.divide(
            self.goal_position[1] - self.agent_pos[1],
            self.goal_position[0] - self.agent_pos[0],
        )

        if self.type == "angle":
            obs["goal_direction"] = np.arctan(slope)
        else:
            obs["goal_direction"] = slope

        return obs


class SymbolicObsWrapper(ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)

        new_image_space = spaces.Box(
            low=0,
            high=max(OBJECT_TO_IDX.values()),
            shape=(self.env.width, self.env.height, 3),
            dtype="uint8",
        )
        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, obs):
        objects = np.array(
            [OBJECT_TO_IDX[o.type] if o is not None else -1 for o in self.grid.grid]
        )
        agent_pos = self.env.agent_pos
        ncol, nrow = self.width, self.height
        grid = np.mgrid[:ncol, :nrow]
        _objects = np.transpose(objects.reshape(1, nrow, ncol), (0, 2, 1))

        grid = np.concatenate([grid, _objects])
        grid = np.transpose(grid, (1, 2, 0))
        grid[agent_pos[0], agent_pos[1], 2] = OBJECT_TO_IDX["agent"]
        obs["image"] = grid

        return obs


class StochasticActionWrapper(ActionWrapper):

    def __init__(self, env=None, prob=0.9, random_action=None):
        super().__init__(env)
        self.prob = prob
        self.random_action = random_action

    def action(self, action):
        if np.random.uniform() < self.prob:
            return action
        else:
            if self.random_action is None:
                return self.np_random.integers(0, high=6)
            else:
                return self.random_action


class NoDeath(Wrapper):

    def __init__(self, env, no_death_types: tuple[str, ...], death_cost: float = -1.0):
        assert "goal" not in no_death_types, "goal cannot be a death cell"

        super().__init__(env)
        self.death_cost = death_cost
        self.no_death_types = no_death_types

    def step(self, action):


        front_cell = self.grid.get(*self.front_pos)
        going_to_death = (
            action == self.actions.forward
            and front_cell is not None
            and front_cell.type in self.no_death_types
        )

        obs, reward, terminated, truncated, info = self.env.step(action)


        current_cell = self.grid.get(*self.agent_pos)
        in_death = current_cell is not None and current_cell.type in self.no_death_types

        if terminated and (going_to_death or in_death):
            terminated = False
            reward += self.death_cost

        return obs, reward, terminated, truncated, info
