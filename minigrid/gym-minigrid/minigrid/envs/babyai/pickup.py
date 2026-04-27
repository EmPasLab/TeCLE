from __future__ import annotations

from minigrid.envs.babyai.core.levelgen import LevelGen
from minigrid.envs.babyai.core.roomgrid_level import RejectSampling, RoomGridLevel
from minigrid.envs.babyai.core.verifier import ObjDesc, PickupInstr


class Pickup(RoomGridLevel):

    def gen_mission(self):
        self.place_agent()
        self.connect_all()
        objs = self.add_distractors(num_distractors=18, all_unique=False)
        self.check_objs_reachable()
        obj = self._rand_elem(objs)
        self.instrs = PickupInstr(ObjDesc(obj.type, obj.color))


class UnblockPickup(RoomGridLevel):

    def gen_mission(self):
        self.place_agent()
        self.connect_all()
        objs = self.add_distractors(num_distractors=20, all_unique=False)


        if self.check_objs_reachable(raise_exc=False):
            raise RejectSampling("all objects reachable")

        obj = self._rand_elem(objs)
        self.instrs = PickupInstr(ObjDesc(obj.type, obj.color))


class PickupLoc(LevelGen):

    def __init__(self, **kwargs):


        super().__init__(
            action_kinds=["pickup"],
            instr_kinds=["action"],
            num_rows=1,
            num_cols=1,
            num_dists=8,
            locked_room_prob=0,
            locations=True,
            unblocking=False,
            **kwargs,
        )


class PickupDist(RoomGridLevel):

    def __init__(self, debug=False, **kwargs):
        self.debug = debug
        super().__init__(num_rows=1, num_cols=1, room_size=7, **kwargs)

    def gen_mission(self):

        objs = self.add_distractors(num_distractors=5)
        self.place_agent(0, 0)
        obj = self._rand_elem(objs)
        type = obj.type
        color = obj.color

        select_by = self._rand_elem(["type", "color", "both"])
        if select_by == "color":
            type = None
        elif select_by == "type":
            color = None

        self.instrs = PickupInstr(ObjDesc(type, color), strict=self.debug)


class PickupAbove(RoomGridLevel):

    def __init__(self, max_steps: int | None = None, **kwargs):
        room_size = 6
        if max_steps is None:
            max_steps = 8 * room_size**2

        super().__init__(room_size=room_size, max_steps=max_steps, **kwargs)

    def gen_mission(self):

        obj, pos = self.add_object(1, 0)

        self.add_door(1, 1, 3, locked=False)
        self.place_agent(1, 1)
        self.connect_all()

        self.instrs = PickupInstr(ObjDesc(obj.type, obj.color))
