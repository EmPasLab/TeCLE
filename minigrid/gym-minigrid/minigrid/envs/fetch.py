from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Key
from minigrid.minigrid_env import MiniGridEnv


class FetchEnv(MiniGridEnv):


    def __init__(self, size=8, numObjs=3, max_steps: int | None = None, **kwargs):
        self.numObjs = numObjs
        self.obj_types = ["key", "ball"]

        MISSION_SYNTAX = [
            "get a",
            "go get a",
            "fetch a",
            "go fetch a",
            "you must fetch a",
        ]
        self.size = size
        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[MISSION_SYNTAX, COLOR_NAMES, self.obj_types],
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
    def _gen_mission(syntax: str, color: str, obj_type: str):
        return f"{syntax} {color} {obj_type}"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)


        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        objs = []


        while len(objs) < self.numObjs:
            objType = self._rand_elem(self.obj_types)
            objColor = self._rand_elem(COLOR_NAMES)

            if objType == "key":
                obj = Key(objColor)
            elif objType == "ball":
                obj = Ball(objColor)
            else:
                raise ValueError(
                    "{} object type given. Object type can only be of values key and ball.".format(
                        objType
                    )
                )

            self.place_obj(obj)
            objs.append(obj)


        self.place_agent()


        target = objs[self._rand_int(0, len(objs))]
        self.targetType = target.type
        self.targetColor = target.color

        descStr = f"{self.targetColor} {self.targetType}"


        idx = self._rand_int(0, 5)
        if idx == 0:
            self.mission = "get a %s" % descStr
        elif idx == 1:
            self.mission = "go get a %s" % descStr
        elif idx == 2:
            self.mission = "fetch a %s" % descStr
        elif idx == 3:
            self.mission = "go fetch a %s" % descStr
        elif idx == 4:
            self.mission = "you must fetch a %s" % descStr
        assert hasattr(self, "mission")

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if self.carrying:
            if (
                self.carrying.color == self.targetColor
                and self.carrying.type == self.targetType
            ):
                reward = self._reward()
                terminated = True
            else:
                reward = 0
                terminated = True

        return obs, reward, terminated, truncated, info
