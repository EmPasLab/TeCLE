from __future__ import annotations

from minigrid.envs.babyai.core.levelgen import LevelGen
from minigrid.envs.babyai.core.roomgrid_level import RejectSampling, RoomGridLevel
from minigrid.envs.babyai.core.verifier import GoToInstr, ObjDesc


class GoToRedBallGrey(RoomGridLevel):

    def __init__(self, room_size=8, num_dists=7, **kwargs):
        self.num_dists = num_dists
        super().__init__(num_rows=1, num_cols=1, room_size=room_size, **kwargs)

    def gen_mission(self):
        self.place_agent()
        obj, _ = self.add_object(0, 0, "ball", "red")
        dists = self.add_distractors(num_distractors=self.num_dists, all_unique=False)

        for dist in dists:
            dist.color = "grey"


        self.check_objs_reachable()

        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


class GoToRedBall(RoomGridLevel):

    def __init__(self, room_size=8, num_dists=7, **kwargs):
        self.num_dists = num_dists
        super().__init__(num_rows=1, num_cols=1, room_size=room_size, **kwargs)

    def gen_mission(self):
        self.place_agent()
        obj, _ = self.add_object(0, 0, "ball", "red")
        self.add_distractors(num_distractors=self.num_dists, all_unique=False)


        self.check_objs_reachable()

        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


class GoToRedBallNoDists(GoToRedBall):

    def __init__(self, **kwargs):
        super().__init__(room_size=8, num_dists=0, **kwargs)


class GoToObj(RoomGridLevel):

    def __init__(self, room_size=8, **kwargs):
        super().__init__(num_rows=1, num_cols=1, room_size=room_size, **kwargs)

    def gen_mission(self):
        self.place_agent()
        objs = self.add_distractors(num_distractors=1)
        obj = objs[0]
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


class GoToLocal(RoomGridLevel):

    def __init__(self, room_size=8, num_dists=8, **kwargs):
        self.num_dists = num_dists
        super().__init__(num_rows=1, num_cols=1, room_size=room_size, **kwargs)

    def gen_mission(self):
        self.place_agent()
        objs = self.add_distractors(num_distractors=self.num_dists, all_unique=False)
        self.check_objs_reachable()
        obj = self._rand_elem(objs)
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


class GoTo(RoomGridLevel):

    def __init__(
        self,
        room_size=8,
        num_rows=3,
        num_cols=3,
        num_dists=18,
        doors_open=False,
        **kwargs,
    ):
        self.num_dists = num_dists
        self.doors_open = doors_open
        super().__init__(
            num_rows=num_rows, num_cols=num_cols, room_size=room_size, **kwargs
        )

    def gen_mission(self):
        self.place_agent()
        self.connect_all()
        objs = self.add_distractors(num_distractors=self.num_dists, all_unique=False)
        self.check_objs_reachable()
        obj = self._rand_elem(objs)
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


        if self.doors_open:
            self.open_all_doors()


class GoToImpUnlock(RoomGridLevel):

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

        self.connect_all()


        for i in range(self.num_cols):
            for j in range(self.num_rows):
                if i is not id or j is not jd:
                    self.add_distractors(i, j, num_distractors=2, all_unique=False)


        while True:
            self.place_agent()
            start_room = self.room_from_pos(*self.agent_pos)

            if start_room is locked_room:
                continue
            break

        self.check_objs_reachable()


        (obj,) = self.add_distractors(id, jd, num_distractors=1, all_unique=False)
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


class GoToSeq(LevelGen):

    def __init__(self, room_size=8, num_rows=3, num_cols=3, num_dists=18, **kwargs):
        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            num_dists=num_dists,
            action_kinds=["goto"],
            locked_room_prob=0,
            locations=False,
            unblocking=False,
            **kwargs,
        )


class GoToRedBlueBall(RoomGridLevel):

    def __init__(self, room_size=8, num_dists=7, **kwargs):
        self.num_dists = num_dists
        super().__init__(num_rows=1, num_cols=1, room_size=room_size, **kwargs)

    def gen_mission(self):
        self.place_agent()

        dists = self.add_distractors(num_distractors=self.num_dists, all_unique=False)


        for dist in dists:
            if dist.type == "ball" and (dist.color == "blue" or dist.color == "red"):
                raise RejectSampling("can only have one blue or red ball")

        color = self._rand_elem(["red", "blue"])
        obj, _ = self.add_object(0, 0, "ball", color)


        self.check_objs_reachable()

        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


class GoToDoor(RoomGridLevel):

    def __init__(self, **kwargs):
        super().__init__(room_size=7, **kwargs)

    def gen_mission(self):
        objs = []
        for _ in range(4):
            door, _ = self.add_door(1, 1)
            objs.append(door)
        self.place_agent(1, 1)

        obj = self._rand_elem(objs)
        self.instrs = GoToInstr(ObjDesc("door", obj.color))


class GoToObjDoor(RoomGridLevel):

    def __init__(self, **kwargs):
        super().__init__(room_size=8, **kwargs)

    def gen_mission(self):
        self.place_agent(1, 1)
        objs = self.add_distractors(1, 1, num_distractors=8, all_unique=False)

        for _ in range(4):
            door, _ = self.add_door(1, 1)
            objs.append(door)

        self.check_objs_reachable()

        obj = self._rand_elem(objs)
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))
