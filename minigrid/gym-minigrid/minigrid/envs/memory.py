from __future__ import annotations

import numpy as np

from minigrid.core.actions import Actions
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Key, Wall
from minigrid.minigrid_env import MiniGridEnv


class MemoryEnv(MiniGridEnv):


    def __init__(
        self, size=8, random_length=False, max_steps: int | None = None, **kwargs
    ):
        self.size = size
        self.random_length = random_length

        if max_steps is None:
            max_steps = 5 * size**2

        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space,
            width=size,
            height=size,

            see_through_walls=False,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "go to the matching object at the end of the hallway"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)


        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        assert height % 2 == 1
        upper_room_wall = height // 2 - 2
        lower_room_wall = height // 2 + 2
        if self.random_length:
            hallway_end = self._rand_int(4, width - 2)
        else:
            hallway_end = width - 3


        for i in range(1, 5):
            self.grid.set(i, upper_room_wall, Wall())
            self.grid.set(i, lower_room_wall, Wall())
        self.grid.set(4, upper_room_wall + 1, Wall())
        self.grid.set(4, lower_room_wall - 1, Wall())


        for i in range(5, hallway_end):
            self.grid.set(i, upper_room_wall + 1, Wall())
            self.grid.set(i, lower_room_wall - 1, Wall())


        for j in range(0, height):
            if j != height // 2:
                self.grid.set(hallway_end, j, Wall())
            self.grid.set(hallway_end + 2, j, Wall())


        self.agent_pos = np.array((self._rand_int(1, hallway_end + 1), height // 2))
        self.agent_dir = 0


        start_room_obj = self._rand_elem([Key, Ball])
        self.grid.set(1, height // 2 - 1, start_room_obj("green"))

        other_objs = self._rand_elem([[Ball, Key], [Key, Ball]])
        pos0 = (hallway_end + 1, height // 2 - 2)
        pos1 = (hallway_end + 1, height // 2 + 2)
        self.grid.set(*pos0, other_objs[0]("green"))
        self.grid.set(*pos1, other_objs[1]("green"))


        if start_room_obj == other_objs[0]:
            self.success_pos = (pos0[0], pos0[1] + 1)
            self.failure_pos = (pos1[0], pos1[1] - 1)
        else:
            self.success_pos = (pos1[0], pos1[1] - 1)
            self.failure_pos = (pos0[0], pos0[1] + 1)

        self.mission = "go to the matching object at the end of the hallway"

    def step(self, action):
        if action == Actions.pickup:
            action = Actions.toggle
        obs, reward, terminated, truncated, info = super().step(action)

        if tuple(self.agent_pos) == self.success_pos:
            reward = self._reward()
            terminated = True
        if tuple(self.agent_pos) == self.failure_pos:
            reward = 0
            terminated = True

        return obs, reward, terminated, truncated, info
