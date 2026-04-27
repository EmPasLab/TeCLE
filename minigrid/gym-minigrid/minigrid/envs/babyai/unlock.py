from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.world_object import Ball, Box, Key
from minigrid.envs.babyai.core.roomgrid_level import RoomGridLevel
from minigrid.envs.babyai.core.verifier import ObjDesc, OpenInstr, PickupInstr


class Unlock(RoomGridLevel):

    def gen_mission(self):

        id = self._rand_int(0, self.num_cols)
        jd = self._rand_int(0, self.num_rows)
        door, pos = self.add_door(id, jd, locked=True)
        locked_room = self.get_room(id, jd)


        while True:
            ik = self._rand_int(0, self.num_cols)
            jk = self._rand_int(0, self.num_rows)
            if ik is id and jk is jd:
                continue
            self.add_object(ik, jk, "key", door.color)
            break


        if self._rand_bool():
            colors = list(filter(lambda c: c is not door.color, COLOR_NAMES))
            self.connect_all(door_colors=colors)
        else:
            self.connect_all()


        for i in range(self.num_cols):
            for j in range(self.num_rows):
                if i is not id or j is not jd:
                    self.add_distractors(i, j, num_distractors=3, all_unique=False)


        while True:
            self.place_agent()
            start_room = self.room_from_pos(*self.agent_pos)

            if start_room is locked_room:
                continue
            break

        self.check_objs_reachable()

        self.instrs = OpenInstr(ObjDesc(door.type, door.color))


class UnlockLocal(RoomGridLevel):

    def __init__(self, distractors=False, **kwargs):
        self.distractors = distractors
        super().__init__(**kwargs)

    def gen_mission(self):
        door, _ = self.add_door(1, 1, locked=True)
        self.add_object(1, 1, "key", door.color)
        if self.distractors:
            self.add_distractors(1, 1, num_distractors=3)
        self.place_agent(1, 1)

        self.instrs = OpenInstr(ObjDesc(door.type))


class KeyInBox(RoomGridLevel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def gen_mission(self):
        door, _ = self.add_door(1, 1, locked=True)


        key = Key(door.color)
        box = Box(self._rand_color(), key)
        self.place_in_room(1, 1, box)

        self.place_agent(1, 1)

        self.instrs = OpenInstr(ObjDesc(door.type))


class UnlockPickup(RoomGridLevel):

    def __init__(self, distractors=False, max_steps: int | None = None, **kwargs):
        self.distractors = distractors
        room_size = 6
        if max is None:
            max_steps = 8 * room_size**2

        super().__init__(
            num_rows=1, num_cols=2, room_size=6, max_steps=max_steps, **kwargs
        )

    def gen_mission(self):

        obj, _ = self.add_object(1, 0, kind="box")

        door, _ = self.add_door(0, 0, 0, locked=True)

        self.add_object(0, 0, "key", door.color)
        if self.distractors:
            self.add_distractors(num_distractors=4)

        self.place_agent(0, 0)

        self.instrs = PickupInstr(ObjDesc(obj.type, obj.color))


class BlockedUnlockPickup(RoomGridLevel):

    def __init__(self, max_steps: int | None = None, **kwargs):
        room_size = 6
        if max_steps is None:
            max_steps = 16 * room_size**2

        super().__init__(
            num_rows=1, num_cols=2, room_size=room_size, max_steps=max_steps, **kwargs
        )

    def gen_mission(self):

        obj, _ = self.add_object(1, 0, kind="box")

        door, pos = self.add_door(0, 0, 0, locked=True)

        color = self._rand_color()
        self.grid.set(pos[0] - 1, pos[1], Ball(color))

        self.add_object(0, 0, "key", door.color)

        self.place_agent(0, 0)

        self.instrs = PickupInstr(ObjDesc(obj.type))


class UnlockToUnlock(RoomGridLevel):

    def __init__(self, max_steps: int | None = None, **kwargs):
        room_size = 6
        if max_steps is None:
            max_steps = 30 * room_size**2

        super().__init__(
            num_rows=1, num_cols=3, room_size=room_size, max_steps=max_steps, **kwargs
        )

    def gen_mission(self):
        colors = self._rand_subset(COLOR_NAMES, 2)


        self.add_door(0, 0, door_idx=0, color=colors[0], locked=True)


        self.add_object(2, 0, kind="key", color=colors[0])


        self.add_door(1, 0, door_idx=0, color=colors[1], locked=True)


        self.add_object(1, 0, kind="key", color=colors[1])

        obj, _ = self.add_object(0, 0, kind="ball")

        self.place_agent(1, 0)

        self.instrs = PickupInstr(ObjDesc(obj.type))
