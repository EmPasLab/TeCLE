from __future__ import annotations

from minigrid.core.mission import MissionSpace
from minigrid.core.roomgrid import RoomGrid


class UnlockEnv(RoomGrid):


    def __init__(self, max_steps: int | None = None, **kwargs):
        room_size = 6
        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 8 * room_size**2

        super().__init__(
            mission_space=mission_space,
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "open the door"

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)


        door, _ = self.add_door(0, 0, 0, locked=True)

        self.add_object(0, 0, "key", door.color)

        self.place_agent(0, 0)

        self.door = door
        self.mission = "open the door"

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if action == self.actions.toggle:
            if self.door.is_open:
                reward = self._reward()
                terminated = True

        return obs, reward, terminated, truncated, info
