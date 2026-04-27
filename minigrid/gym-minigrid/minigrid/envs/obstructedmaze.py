from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES, DIR_TO_VEC
from minigrid.core.mission import MissionSpace
from minigrid.core.roomgrid import RoomGrid
from minigrid.core.world_object import Ball, Box, Key


class ObstructedMazeEnv(RoomGrid):


    def __init__(
        self,
        num_rows,
        num_cols,
        num_rooms_visited,
        max_steps: int | None = None,
        **kwargs,
    ):
        room_size = 6

        if max_steps is None:
            max_steps = 4 * num_rooms_visited * room_size**2

        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[[COLOR_NAMES[0]]],
        )
        super().__init__(
            mission_space=mission_space,
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            max_steps=max_steps,
            **kwargs,
        )
        self.obj = Ball()

    @staticmethod
    def _gen_mission(color: str):
        return f"pick up the {color} ball"

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)


        self.door_colors = self._rand_subset(COLOR_NAMES, len(COLOR_NAMES))

        self.ball_to_find_color = COLOR_NAMES[0]

        self.blocking_ball_color = COLOR_NAMES[1]

        self.box_color = COLOR_NAMES[2]

        self.mission = "pick up the %s ball" % self.ball_to_find_color

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if action == self.actions.pickup:
            if self.carrying and self.carrying == self.obj:
                reward = self._reward()
                terminated = True

        return obs, reward, terminated, truncated, info

    def add_door(
        self,
        i,
        j,
        door_idx=0,
        color=None,
        locked=False,
        key_in_box=False,
        blocked=False,
    ):

        door, door_pos = super().add_door(i, j, door_idx, color, locked=locked)

        if blocked:
            vec = DIR_TO_VEC[door_idx]
            blocking_ball = Ball(self.blocking_ball_color) if blocked else None
            self.grid.set(door_pos[0] - vec[0], door_pos[1] - vec[1], blocking_ball)

        if locked:
            obj = Key(door.color)
            if key_in_box:
                box = Box(self.box_color)
                box.contains = obj
                obj = box
            self.place_in_room(i, j, obj)

        return door, door_pos


class ObstructedMaze_1Dlhb(ObstructedMazeEnv):

    def __init__(self, key_in_box=True, blocked=True, **kwargs):
        self.key_in_box = key_in_box
        self.blocked = blocked

        super().__init__(num_rows=1, num_cols=2, num_rooms_visited=2, **kwargs)

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        self.add_door(
            0,
            0,
            door_idx=0,
            color=self.door_colors[0],
            locked=True,
            key_in_box=self.key_in_box,
            blocked=self.blocked,
        )

        self.obj, _ = self.add_object(1, 0, "ball", color=self.ball_to_find_color)
        self.place_agent(0, 0)


class ObstructedMaze_Full(ObstructedMazeEnv):

    def __init__(
        self,
        agent_room=(1, 1),
        key_in_box=True,
        blocked=True,
        num_quarters=4,
        num_rooms_visited=25,
        **kwargs,
    ):
        self.agent_room = agent_room
        self.key_in_box = key_in_box
        self.blocked = blocked
        self.num_quarters = num_quarters

        super().__init__(
            num_rows=3, num_cols=3, num_rooms_visited=num_rooms_visited, **kwargs
        )

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        middle_room = (1, 1)


        side_rooms = [(2, 1), (1, 2), (0, 1), (1, 0)][: self.num_quarters]
        for i in range(len(side_rooms)):
            side_room = side_rooms[i]


            self.add_door(
                *middle_room, door_idx=i, color=self.door_colors[i], locked=False
            )

            for k in [-1, 1]:

                self.add_door(
                    *side_room,
                    locked=True,
                    door_idx=(i + k) % 4,
                    color=self.door_colors[(i + k) % len(self.door_colors)],
                    key_in_box=self.key_in_box,
                    blocked=self.blocked,
                )

        corners = [(2, 0), (2, 2), (0, 2), (0, 0)][: self.num_quarters]
        ball_room = self._rand_elem(corners)

        self.obj, _ = self.add_object(
            ball_room[0], ball_room[1], "ball", color=self.ball_to_find_color
        )
        self.place_agent(*self.agent_room)


class ObstructedMaze_2Dl(ObstructedMaze_Full):
    def __init__(self, **kwargs):
        super().__init__((2, 1), False, False, 1, 4, **kwargs)


class ObstructedMaze_2Dlh(ObstructedMaze_Full):
    def __init__(self, **kwargs):
        super().__init__((2, 1), True, False, 1, 4, **kwargs)


class ObstructedMaze_2Dlhb(ObstructedMaze_Full):
    def __init__(self, **kwargs):
        super().__init__((2, 1), True, True, 1, 4, **kwargs)
