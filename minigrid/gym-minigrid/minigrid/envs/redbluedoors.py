from __future__ import annotations

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door
from minigrid.minigrid_env import MiniGridEnv


class RedBlueDoorEnv(MiniGridEnv):


    def __init__(self, size=8, max_steps: int | None = None, **kwargs):
        self.size = size
        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 20 * size**2

        super().__init__(
            mission_space=mission_space,
            width=2 * size,
            height=size,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "open the red door then the blue door"

    def _gen_grid(self, width, height):

        self.grid = Grid(width, height)


        self.grid.wall_rect(0, 0, 2 * self.size, self.size)
        self.grid.wall_rect(self.size // 2, 0, self.size, self.size)


        self.place_agent(top=(self.size // 2, 0), size=(self.size, self.size))


        pos = self._rand_int(1, self.size - 1)
        self.red_door = Door("red")
        self.grid.set(self.size // 2, pos, self.red_door)


        pos = self._rand_int(1, self.size - 1)
        self.blue_door = Door("blue")
        self.grid.set(self.size // 2 + self.size - 1, pos, self.blue_door)


        self.mission = "open the red door then the blue door"

    def step(self, action):
        red_door_opened_before = self.red_door.is_open
        blue_door_opened_before = self.blue_door.is_open

        obs, reward, terminated, truncated, info = super().step(action)

        red_door_opened_after = self.red_door.is_open
        blue_door_opened_after = self.blue_door.is_open

        if blue_door_opened_after:
            if red_door_opened_before:
                reward = self._reward()
                terminated = True
            else:
                reward = 0
                terminated = True

        elif red_door_opened_after:
            if blue_door_opened_before:
                reward = 0
                terminated = True

        return obs, reward, terminated, truncated, info
