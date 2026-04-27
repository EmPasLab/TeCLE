from __future__ import annotations

from operator import add

from gymnasium.spaces import Discrete

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Goal
from minigrid.minigrid_env import MiniGridEnv


class DynamicObstaclesEnv(MiniGridEnv):

    def __init__(
        self,
        size=8,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        n_obstacles=4,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir


        if n_obstacles <= size / 2 + 1:
            self.n_obstacles = int(n_obstacles)
        else:
            self.n_obstacles = int(size / 2)

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,

            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

        self.action_space = Discrete(self.actions.forward + 1)
        self.reward_range = (-1, 1)

    @staticmethod
    def _gen_mission():
        return "get to the green goal square"

    def _gen_grid(self, width, height):

        self.grid = Grid(width, height)


        self.grid.wall_rect(0, 0, width, height)


        self.grid.set(width - 2, height - 2, Goal())


        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()


        self.obstacles = []
        for i_obst in range(self.n_obstacles):
            self.obstacles.append(Ball())
            self.place_obj(self.obstacles[i_obst], max_tries=100)

        self.mission = "get to the green goal square"

    def step(self, action):

        if action >= self.action_space.n:
            action = 0


        front_cell = self.grid.get(*self.front_pos)
        not_clear = front_cell and front_cell.type != "goal"


        for i_obst in range(len(self.obstacles)):
            old_pos = self.obstacles[i_obst].cur_pos
            top = tuple(map(add, old_pos, (-1, -1)))

            try:
                self.place_obj(
                    self.obstacles[i_obst], top=top, size=(3, 3), max_tries=100
                )
                self.grid.set(old_pos[0], old_pos[1], None)
            except Exception:
                pass


        obs, reward, terminated, truncated, info = super().step(action)


        if action == self.actions.forward and not_clear:
            reward = -1
            terminated = True
            return obs, reward, terminated, truncated, info

        return obs, reward, terminated, truncated, info
