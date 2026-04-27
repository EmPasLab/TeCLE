from __future__ import annotations

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Lava
from minigrid.minigrid_env import MiniGridEnv


class DistShiftEnv(MiniGridEnv):


    def __init__(
        self,
        width=9,
        height=7,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        strip2_row=2,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.goal_pos = (width - 2, 1)
        self.strip2_row = strip2_row

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * width * height

        super().__init__(
            mission_space=mission_space,
            width=width,
            height=height,

            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "get to the green goal square"

    def _gen_grid(self, width, height):

        self.grid = Grid(width, height)


        self.grid.wall_rect(0, 0, width, height)


        self.put_obj(Goal(), *self.goal_pos)


        for i in range(self.width - 6):
            self.grid.set(3 + i, 1, Lava())
            self.grid.set(3 + i, self.strip2_row, Lava())


        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"
