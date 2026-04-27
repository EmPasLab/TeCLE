from __future__ import annotations

from minigrid.envs.babyai.core.roomgrid_level import RoomGridLevel
from minigrid.envs.babyai.core.verifier import ObjDesc, PutNextInstr


class PutNextLocal(RoomGridLevel):

    def __init__(self, room_size=8, num_objs=8, **kwargs):
        self.num_objs = num_objs
        super().__init__(num_rows=1, num_cols=1, room_size=room_size, **kwargs)

    def gen_mission(self):
        self.place_agent()
        objs = self.add_distractors(num_distractors=self.num_objs, all_unique=True)
        self.check_objs_reachable()
        o1, o2 = self._rand_subset(objs, 2)

        self.instrs = PutNextInstr(
            ObjDesc(o1.type, o1.color), ObjDesc(o2.type, o2.color)
        )


class PutNext(RoomGridLevel):

    def __init__(
        self,
        room_size,
        objs_per_room,
        start_carrying=False,
        max_steps: int | None = None,
        **kwargs,
    ):
        assert room_size >= 4
        assert objs_per_room <= 9
        self.objs_per_room = objs_per_room
        self.start_carrying = start_carrying

        if max_steps is None:
            max_steps = 8 * room_size**2

        super().__init__(
            num_rows=1, num_cols=2, room_size=room_size, max_steps=max_steps, **kwargs
        )

    def gen_mission(self):
        self.place_agent(0, 0)


        objs_l = self.add_distractors(0, 0, self.objs_per_room)
        objs_r = self.add_distractors(1, 0, self.objs_per_room)


        self.remove_wall(0, 0, 0)


        a = self._rand_elem(objs_l)
        b = self._rand_elem(objs_r)


        if self._rand_bool():
            t = a
            a = b
            b = t

        self.obj_a = a

        self.instrs = PutNextInstr(ObjDesc(a.type, a.color), ObjDesc(b.type, b.color))

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)


        if self.start_carrying:
            assert self.obj_a.init_pos is not None
            self.grid.set(*self.obj_a.init_pos, None)
            self.carrying = self.obj_a

        return obs
