from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Box, Key
from minigrid.minigrid_env import MiniGridEnv


class GoToObjectEnv(MiniGridEnv):

    def __init__(self, size=6, numObjs=2, max_steps: int | None = None, **kwargs):
        self.numObjs = numObjs
        self.size = size

        self.obj_types = ["key", "ball", "box"]

        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[COLOR_NAMES, self.obj_types],
        )

        if max_steps is None:
            max_steps = 5 * size**2

        super().__init__(
            mission_space=mission_space,
            width=size,
            height=size,

            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission(color: str, obj_type: str):
        return f"go to the {color} {obj_type}"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)


        self.grid.wall_rect(0, 0, width, height)


        types = ["key", "ball", "box"]

        objs = []
        objPos = []


        while len(objs) < self.numObjs:
            objType = self._rand_elem(types)
            objColor = self._rand_elem(COLOR_NAMES)


            if (objType, objColor) in objs:
                continue

            if objType == "key":
                obj = Key(objColor)
            elif objType == "ball":
                obj = Ball(objColor)
            elif objType == "box":
                obj = Box(objColor)
            else:
                raise ValueError(
                    "{} object type given. Object type can only be of values key, ball and box.".format(
                        objType
                    )
                )

            pos = self.place_obj(obj)
            objs.append((objType, objColor))
            objPos.append(pos)


        self.place_agent()


        objIdx = self._rand_int(0, len(objs))
        self.targetType, self.target_color = objs[objIdx]
        self.target_pos = objPos[objIdx]

        descStr = f"{self.target_color} {self.targetType}"
        self.mission = "go to the %s" % descStr


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
