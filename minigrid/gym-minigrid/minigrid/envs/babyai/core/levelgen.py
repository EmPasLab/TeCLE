from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.roomgrid import Room
from minigrid.envs.babyai.core.roomgrid_level import RoomGridLevel
from minigrid.envs.babyai.core.verifier import (
    LOC_NAMES,
    OBJ_TYPES,
    OBJ_TYPES_NOT_DOOR,
    AfterInstr,
    AndInstr,
    BeforeInstr,
    GoToInstr,
    ObjDesc,
    OpenInstr,
    PickupInstr,
    PutNextInstr,
)


class LevelGen(RoomGridLevel):

    def __init__(
        self,
        room_size=8,
        num_rows=3,
        num_cols=3,
        num_dists=18,
        locked_room_prob=0.5,
        locations=True,
        unblocking=True,
        implicit_unlock=True,
        action_kinds=["goto", "pickup", "open", "putnext"],
        instr_kinds=["action", "and", "seq"],
        **kwargs,
    ):
        self.num_dists = num_dists
        self.locked_room_prob = locked_room_prob
        self.locations = locations
        self.unblocking = unblocking
        self.implicit_unlock = implicit_unlock
        self.action_kinds = action_kinds
        self.instr_kinds = instr_kinds

        self.locked_room = None

        super().__init__(
            room_size=room_size, num_rows=num_rows, num_cols=num_cols, **kwargs
        )

    def gen_mission(self):
        if self._rand_float(0, 1) < self.locked_room_prob:
            self.add_locked_room()

        self.connect_all()

        self.add_distractors(num_distractors=self.num_dists, all_unique=False)


        while True:
            self.place_agent()
            start_room = self.room_from_pos(*self.agent_pos)

            if start_room is self.locked_room:
                continue
            break


        if not self.unblocking:
            self.check_objs_reachable()


        self.instrs = self.rand_instr(
            action_kinds=self.action_kinds, instr_kinds=self.instr_kinds
        )

    def add_locked_room(self):

        while True:
            i = self._rand_int(0, self.num_cols)
            j = self._rand_int(0, self.num_rows)
            door_idx = self._rand_int(0, 4)
            self.locked_room = self.get_room(i, j)


            if self.locked_room.neighbors[door_idx] is None:
                continue

            door, _ = self.add_door(i, j, door_idx, locked=True)


            break


        while True:
            i = self._rand_int(0, self.num_cols)
            j = self._rand_int(0, self.num_rows)
            key_room = self.get_room(i, j)

            if key_room is self.locked_room:
                continue

            self.add_object(i, j, "key", door.color)
            break

    def rand_obj(self, types=OBJ_TYPES, colors=COLOR_NAMES, max_tries=100):

        num_tries = 0


        while True:
            if num_tries > max_tries:
                raise RecursionError("failed to find suitable object")
            num_tries += 1

            color = self._rand_elem([None, *colors])
            type = self._rand_elem(types)

            loc = None
            if self.locations and self._rand_bool():
                loc = self._rand_elem(LOC_NAMES)

            desc = ObjDesc(type, color, loc)


            objs, poss = desc.find_matching_objs(self)


            if len(objs) == 0:
                continue


            if not self.implicit_unlock and isinstance(self.locked_room, Room):
                locked_room = self.locked_room

                pos_not_locked = list(
                    filter(lambda p: not locked_room.pos_inside(*p), poss)
                )

                if len(pos_not_locked) == 0:
                    continue


            return desc

    def rand_instr(self, action_kinds, instr_kinds, depth=0):

        kind = self._rand_elem(instr_kinds)

        if kind == "action":
            action = self._rand_elem(action_kinds)

            if action == "goto":
                return GoToInstr(self.rand_obj())
            elif action == "pickup":
                return PickupInstr(self.rand_obj(types=OBJ_TYPES_NOT_DOOR))
            elif action == "open":
                return OpenInstr(self.rand_obj(types=["door"]))
            elif action == "putnext":
                return PutNextInstr(
                    self.rand_obj(types=OBJ_TYPES_NOT_DOOR), self.rand_obj()
                )

            assert False

        elif kind == "and":
            instr_a = self.rand_instr(
                action_kinds=action_kinds, instr_kinds=["action"], depth=depth + 1
            )
            instr_b = self.rand_instr(
                action_kinds=action_kinds, instr_kinds=["action"], depth=depth + 1
            )
            return AndInstr(instr_a, instr_b)

        elif kind == "seq":
            instr_a = self.rand_instr(
                action_kinds=action_kinds,
                instr_kinds=["action", "and"],
                depth=depth + 1,
            )
            instr_b = self.rand_instr(
                action_kinds=action_kinds,
                instr_kinds=["action", "and"],
                depth=depth + 1,
            )

            kind = self._rand_elem(["before", "after"])

            if kind == "before":
                return BeforeInstr(instr_a, instr_b)
            elif kind == "after":
                return AfterInstr(instr_a, instr_b)

            assert False

        assert False
