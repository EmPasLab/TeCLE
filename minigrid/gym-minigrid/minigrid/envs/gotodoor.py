from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door
from minigrid.minigrid_env import MiniGridEnv


class GoToDoorEnv(MiniGridEnv):

    def __init__(self, size=5, max_steps: int | None = None, **kwargs):
        assert size >= 5
        self.size = size
        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[COLOR_NAMES],
        )

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            width=size,
            height=size,

            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission(color: str):
        return f"go to the {color} door"

    def _gen_grid(self, width, height):

        self.grid = Grid(width, height)


        width = self._rand_int(5, width + 1)
        height = self._rand_int(5, height + 1)


        self.grid.wall_rect(0, 0, width, height)


        doorPos = []
        doorPos.append((self._rand_int(2, width - 2), 0))
        doorPos.append((self._rand_int(2, width - 2), height - 1))
        doorPos.append((0, self._rand_int(2, height - 2)))
        doorPos.append((width - 1, self._rand_int(2, height - 2)))


        doorColors = []
        while len(doorColors) < len(doorPos):
            color = self._rand_elem(COLOR_NAMES)
            if color in doorColors:
                continue
            doorColors.append(color)


        for idx, pos in enumerate(doorPos):
            color = doorColors[idx]
            self.grid.set(*pos, Door(color))


        self.place_agent(size=(width, height))


        doorIdx = self._rand_int(0, len(doorPos))
        self.target_pos = doorPos[doorIdx]
        self.target_color = doorColors[doorIdx]


        self.mission = "go to the %s door" % self.target_color

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        ax, ay = self.agent_pos
        tx, ty = self.target_pos


        if action == self.actions.toggle:
            terminated = True


        if action == self.actions.done:
            if (ax == tx and abs(ay - ty) == 1) or (ay == ty and abs(ax - tx) == 1):
                reward = self._reward()
            terminated = True

        return obs, reward, terminated, truncated, info
