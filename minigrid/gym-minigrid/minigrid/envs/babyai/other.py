from __future__ import annotations

from minigrid.envs.babyai.core.roomgrid_level import RoomGridLevel
from minigrid.envs.babyai.core.verifier import (
    BeforeInstr,
    GoToInstr,
    ObjDesc,
    OpenInstr,
    PickupInstr,
    PutNextInstr,
)


class ActionObjDoor(RoomGridLevel):

    def __init__(self, **kwargs):
        super().__init__(room_size=7, **kwargs)

    def gen_mission(self):
        objs = self.add_distractors(1, 1, num_distractors=5)
        for _ in range(4):
            door, _ = self.add_door(1, 1, locked=False)
            objs.append(door)

        self.place_agent(1, 1)

        obj = self._rand_elem(objs)
        desc = ObjDesc(obj.type, obj.color)

        if obj.type == "door":
            if self._rand_bool():
                self.instrs = GoToInstr(desc)
            else:
                self.instrs = OpenInstr(desc)
        else:
            if self._rand_bool():
                self.instrs = GoToInstr(desc)
            else:
                self.instrs = PickupInstr(desc)


class FindObjS5(RoomGridLevel):

    def __init__(self, room_size=5, max_steps: int | None = None, **kwargs):
        if max_steps is None:
            max_steps = 20 * room_size**2

        super().__init__(room_size=room_size, max_steps=max_steps, **kwargs)

    def gen_mission(self):

        i = self._rand_int(0, self.num_rows)
        j = self._rand_int(0, self.num_cols)
        obj, _ = self.add_object(i, j)
        self.place_agent(1, 1)
        self.connect_all()

        self.instrs = PickupInstr(ObjDesc(obj.type))


class KeyCorridor(RoomGridLevel):

    def __init__(
        self,
        num_rows=3,
        obj_type="ball",
        room_size=6,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.obj_type = obj_type

        if max_steps is None:
            max_steps = 30 * room_size**2

        super().__init__(
            room_size=room_size, num_rows=num_rows, max_steps=max_steps, **kwargs
        )

    def gen_mission(self):

        for j in range(1, self.num_rows):
            self.remove_wall(1, j, 3)


        room_idx = self._rand_int(0, self.num_rows)
        door, _ = self.add_door(2, room_idx, 2, locked=True)
        obj, _ = self.add_object(2, room_idx, kind=self.obj_type)


        self.add_object(0, self._rand_int(0, self.num_rows), "key", door.color)


        self.place_agent(1, self.num_rows // 2)


        self.connect_all()

        self.instrs = PickupInstr(ObjDesc(obj.type))


class OneRoomS8(RoomGridLevel):

    def __init__(self, room_size=8, **kwargs):
        super().__init__(room_size=room_size, num_rows=1, num_cols=1, **kwargs)

    def gen_mission(self):
        obj, _ = self.add_object(0, 0, kind="ball")
        self.place_agent()
        self.instrs = PickupInstr(ObjDesc(obj.type))


class MoveTwoAcross(RoomGridLevel):

    def __init__(
        self, room_size, objs_per_room, max_steps: int | None = None, **kwargs
    ):
        assert objs_per_room <= 9
        self.objs_per_room = objs_per_room

        if max_steps is None:
            max_steps = 16 * room_size**2

        super().__init__(
            num_rows=1, num_cols=2, room_size=room_size, max_steps=max_steps, **kwargs
        )

    def gen_mission(self):
        self.place_agent(0, 0)


        objs_l = self.add_distractors(0, 0, self.objs_per_room)
        objs_r = self.add_distractors(1, 0, self.objs_per_room)


        self.remove_wall(0, 0, 0)


        objs_l = self._rand_subset(objs_l, 2)
        objs_r = self._rand_subset(objs_r, 2)
        a = objs_l[0]
        b = objs_r[0]
        c = objs_r[1]
        d = objs_l[1]

        self.instrs = BeforeInstr(
            PutNextInstr(ObjDesc(a.type, a.color), ObjDesc(b.type, b.color)),
            PutNextInstr(ObjDesc(c.type, c.color), ObjDesc(d.type, d.color)),
        )
