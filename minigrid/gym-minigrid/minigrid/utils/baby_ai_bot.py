from __future__ import annotations

import numpy as np

from minigrid.core.world_object import WorldObj
from minigrid.envs.babyai.core.verifier import (
    AfterInstr,
    AndInstr,
    BeforeInstr,
    GoToInstr,
    ObjDesc,
    OpenInstr,
    PickupInstr,
    PutNextInstr,
)


class DisappearedBoxError(Exception):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def manhattan_distance(pos, target):
    return np.abs(target[0] - pos[0]) + np.abs(target[1] - pos[1])


class Subgoal:

    def __init__(self, bot: BabyAIBot, datum=None, reason=None):
        self.bot = bot
        self.datum = datum
        self.reason = reason

        self.update_agent_attributes()

        self.actions = self.bot.mission.unwrapped.actions

    def __repr__(self):
        representation = "("
        representation += type(self).__name__
        if self.datum is not None:
            representation += f": {self.datum}"
        if self.reason is not None:
            representation += f", reason: {self.reason}"
        representation += ")"
        return representation

    def update_agent_attributes(self):
        self.pos = self.bot.mission.unwrapped.agent_pos
        self.dir_vec = self.bot.mission.unwrapped.dir_vec
        self.right_vec = self.bot.mission.unwrapped.right_vec
        self.fwd_pos = self.pos + self.dir_vec
        self.fwd_cell = self.bot.mission.unwrapped.grid.get(*self.fwd_pos)
        self.carrying = self.bot.mission.unwrapped.carrying

    def replan_before_action(self):
        raise NotImplementedError()

    def replan_after_action(self, action_taken):
        pass

    def is_exploratory(self):
        return False

    def _plan_undo_action(self, action_taken):
        if action_taken == self.actions.forward:

            if not np.array_equal(self.bot.prev_agent_pos, self.pos):
                self.bot.stack.append(GoNextToSubgoal(self.bot, self.pos))
        elif action_taken == self.actions.left:
            old_fwd_pos = self.pos + self.right_vec
            self.bot.stack.append(GoNextToSubgoal(self.bot, old_fwd_pos))
        elif action_taken == self.actions.right:
            old_fwd_pos = self.pos - self.right_vec
            self.bot.stack.append(GoNextToSubgoal(self.bot, old_fwd_pos))
        elif (
            action_taken == self.actions.drop
            and self.bot.prev_carrying != self.carrying
        ):

            assert self.fwd_cell.type in ("key", "box", "ball")
            self.bot.stack.append(PickupSubgoal(self.bot))
        elif (
            action_taken == self.actions.pickup
            and self.bot.prev_carrying != self.carrying
        ):

            fwd_cell = self.bot.mission.unwrapped.grid.get(*self.fwd_pos)
            self.bot.stack.append(DropSubgoal(self.bot))
        elif action_taken == self.actions.toggle:

            fwd_cell = self.bot.mission.unwrapped.grid.get(*self.fwd_pos)
            if (
                fwd_cell
                and fwd_cell.type == "door"
                and self.bot.fwd_door_was_open != fwd_cell.is_open
            ):
                self.bot.stack.append(
                    CloseSubgoal(self.bot)
                    if fwd_cell.is_open
                    else OpenSubgoal(self.bot)
                )


class CloseSubgoal(Subgoal):
    def replan_before_action(self):
        assert self.fwd_cell is not None, "Forward cell is empty"
        assert self.fwd_cell.type == "door", "Forward cell has to be a door"
        assert self.fwd_cell.is_open, "Forward door must be open"
        return self.actions.toggle

    def replan_after_action(self, action_taken):
        if action_taken is None or action_taken == self.actions.toggle:
            self.bot.stack.pop()
        elif action_taken in [
            self.actions.forward,
            self.actions.left,
            self.actions.right,
        ]:
            self._plan_undo_action(action_taken)


class OpenSubgoal(Subgoal):

    def replan_before_action(self):
        assert self.fwd_cell is not None, "Forward cell is empty"
        assert self.fwd_cell.type == "door", "Forward cell has to be a door"


        got_the_key = (
            self.carrying
            and self.carrying.type == "key"
            and self.carrying.color == self.fwd_cell.color
        )
        if self.fwd_cell.is_locked and not got_the_key:

            key_desc = ObjDesc("key", self.fwd_cell.color)
            key_desc.find_matching_objs(self.bot.mission)


            if self.carrying:
                self.bot.stack.pop()


                drop_pos_cur = self.bot._find_drop_pos()


                self.bot.stack.append(PickupSubgoal(self.bot))
                self.bot.stack.append(GoNextToSubgoal(self.bot, drop_pos_cur))


                self.bot.stack.append(OpenSubgoal(self.bot))
                self.bot.stack.append(GoNextToSubgoal(self.bot, tuple(self.fwd_pos)))


                self.bot.stack.append(PickupSubgoal(self.bot))
                self.bot.stack.append(GoNextToSubgoal(self.bot, key_desc))


                self.bot.stack.append(DropSubgoal(self.bot))
                self.bot.stack.append(GoNextToSubgoal(self.bot, drop_pos_cur))
            else:


                self.bot.stack.pop()


                self.bot.stack.append(OpenSubgoal(self.bot))
                self.bot.stack.append(GoNextToSubgoal(self.bot, tuple(self.fwd_pos)))


                self.bot.stack.append(PickupSubgoal(self.bot))
                self.bot.stack.append(GoNextToSubgoal(self.bot, key_desc))
            return

        if self.fwd_cell.is_open:
            self.bot.stack.append(CloseSubgoal(self.bot))
            return

        if self.fwd_cell.is_locked and self.reason is None:
            self.bot.stack.pop()
            self.bot.stack.append(OpenSubgoal(self.bot, reason="Unlock"))
            return

        return self.actions.toggle

    def replan_after_action(self, action_taken):
        if action_taken is None or action_taken == self.actions.toggle:
            self.bot.stack.pop()
            if self.reason == "Unlock":


                drop_key_pos = self.bot._find_drop_pos()
                self.bot.stack.append(DropSubgoal(self.bot))
                self.bot.stack.append(GoNextToSubgoal(self.bot, drop_key_pos))
        else:
            self._plan_undo_action(action_taken)


class DropSubgoal(Subgoal):
    def replan_before_action(self):
        assert self.bot.mission.unwrapped.carrying
        assert not self.fwd_cell
        return self.actions.drop

    def replan_after_action(self, action_taken):
        if action_taken is None or action_taken == self.actions.drop:
            self.bot.stack.pop()
        elif action_taken in [
            self.actions.forward,
            self.actions.left,
            self.actions.right,
        ]:
            self._plan_undo_action(action_taken)


class PickupSubgoal(Subgoal):
    def replan_before_action(self):
        assert not self.bot.mission.unwrapped.carrying
        return self.actions.pickup

    def replan_after_action(self, action_taken):
        if action_taken is None or action_taken == self.actions.pickup:
            self.bot.stack.pop()
        elif action_taken in [self.actions.left, self.actions.right]:
            self._plan_undo_action(action_taken)


class GoNextToSubgoal(Subgoal):

    def replan_before_action(self):
        target_obj = None
        if isinstance(self.datum, ObjDesc):
            target_obj, target_pos = self.bot._find_obj_pos(
                self.datum, self.reason == "PutNext"
            )
            if not target_pos:

                self.bot.stack.append(ExploreSubgoal(self.bot))
                return
        elif isinstance(self.datum, WorldObj):
            target_obj = self.datum
            target_pos = target_obj.cur_pos
        else:
            target_pos = tuple(self.datum)


        if (
            self.reason == "Open"
            and target_obj
            and target_obj.type == "door"
            and target_obj.is_locked
        ):
            key_desc = ObjDesc("key", target_obj.color)
            key_desc.find_matching_objs(self.bot.mission)
            if not self.carrying:

                self.bot.stack.pop()
                self.bot.stack.append(
                    GoNextToSubgoal(self.bot, target_obj, reason="Open")
                )
                self.bot.stack.append(PickupSubgoal(self.bot))
                self.bot.stack.append(GoNextToSubgoal(self.bot, key_desc))
                return


        if manhattan_distance(target_pos, self.pos) == (
            1 if self.reason == "PutNext" else 0
        ):

            def steppable(cell):
                return cell is None or (cell.type == "door" and cell.is_open)

            if steppable(self.fwd_cell):
                return self.actions.forward
            if steppable(
                self.bot.mission.unwrapped.grid.get(*(self.pos + self.right_vec))
            ):
                return self.actions.right
            if steppable(
                self.bot.mission.unwrapped.grid.get(*(self.pos - self.right_vec))
            ):
                return self.actions.left

            return self.actions.left


        if self.reason == "PutNext":
            if manhattan_distance(target_pos, self.fwd_pos) == 1:
                if self.fwd_cell is None:
                    self.bot.stack.pop()
                    return
                if self.fwd_cell.type == "door" and self.fwd_cell.is_open:


                    self.bot.stack.append(
                        GoNextToSubgoal(self.bot, self.fwd_pos + 2 * self.dir_vec)
                    )
                    return
        else:
            if np.array_equal(target_pos, self.fwd_pos):
                self.bot.stack.pop()
                return


        path, _, _ = self.bot._shortest_path(
            lambda pos, cell: pos == target_pos,
        )


        if not path:
            path, _, _ = self.bot._shortest_path(
                lambda pos, cell: pos == target_pos, try_with_blockers=True
            )


        if not path:
            self.bot.stack.append(ExploreSubgoal(self.bot))
            return


        next_cell = np.asarray(path[0])


        if np.array_equal(next_cell, self.fwd_pos):
            if self.fwd_cell:
                if self.fwd_cell.type == "door":
                    assert not self.fwd_cell.is_locked
                    if not self.fwd_cell.is_open:
                        self.bot.stack.append(OpenSubgoal(self.bot))
                        return
                    else:
                        return self.actions.forward
                if self.carrying:
                    drop_pos_cur = self.bot._find_drop_pos()
                    drop_pos_block = self.bot._find_drop_pos(drop_pos_cur)

                    self.bot.stack.append(PickupSubgoal(self.bot))
                    self.bot.stack.append(GoNextToSubgoal(self.bot, drop_pos_cur))


                    self.bot.stack.append(DropSubgoal(self.bot))
                    self.bot.stack.append(GoNextToSubgoal(self.bot, drop_pos_block))
                    self.bot.stack.append(PickupSubgoal(self.bot))
                    self.bot.stack.append(GoNextToSubgoal(self.bot, self.fwd_pos))


                    self.bot.stack.append(DropSubgoal(self.bot))
                    self.bot.stack.append(GoNextToSubgoal(self.bot, drop_pos_cur))
                    return
                else:
                    drop_pos = self.bot._find_drop_pos()
                    self.bot.stack.append(DropSubgoal(self.bot))
                    self.bot.stack.append(GoNextToSubgoal(self.bot, drop_pos))
                    self.bot.stack.append(PickupSubgoal(self.bot))
                    return
            else:
                return self.actions.forward


        if np.array_equal(next_cell - self.pos, self.right_vec):
            return self.actions.right
        elif np.array_equal(next_cell - self.pos, -self.right_vec):
            return self.actions.left


        distance_right = self.bot._closest_wall_or_door_given_dir(
            self.pos, self.right_vec
        )
        distance_left = self.bot._closest_wall_or_door_given_dir(
            self.pos, -self.right_vec
        )
        if distance_left > distance_right:
            return self.actions.left
        return self.actions.right

    def replan_after_action(self, action_taken):
        if action_taken in [
            self.actions.pickup,
            self.actions.drop,
            self.actions.toggle,
        ]:
            self._plan_undo_action(action_taken)

    def is_exploratory(self):
        return self.reason == "Explore"


class ExploreSubgoal(Subgoal):
    def replan_before_action(self):

        _, unseen_pos, with_blockers = self.bot._shortest_path(
            lambda pos, cell: not self.bot.vis_mask[pos], try_with_blockers=True
        )

        if unseen_pos:
            self.bot.stack.append(
                GoNextToSubgoal(self.bot, unseen_pos, reason="Explore")
            )
            return None


        def unopened_unlocked_door(pos, cell):
            return (
                cell and cell.type == "door" and not cell.is_locked and not cell.is_open
            )


        def unopened_door(pos, cell):
            return cell and cell.type == "door" and not cell.is_open


        _, door_pos, _ = self.bot._shortest_path(
            unopened_unlocked_door, try_with_blockers=True
        )
        if not door_pos:

            _, door_pos, _ = self.bot._shortest_path(
                unopened_door, try_with_blockers=True
            )


        if door_pos:
            door_obj = self.bot.mission.unwrapped.grid.get(*door_pos)


            got_the_key = (
                self.carrying
                and self.carrying.type == "key"
                and self.carrying.color == door_obj.color
            )
            open_reason = "KeepKey" if door_obj.is_locked and got_the_key else None
            self.bot.stack.pop()
            self.bot.stack.append(OpenSubgoal(self.bot, reason=open_reason))
            self.bot.stack.append(GoNextToSubgoal(self.bot, door_obj, reason="Open"))
            return

        assert False, "0nothing left to explore"

    def is_exploratory(self):
        return True


class BabyAIBot:

    def __init__(self, mission):

        self.mission = mission


        self.vis_mask = np.zeros(
            shape=(mission.unwrapped.width, mission.unwrapped.height), dtype=bool
        )


        self.stack = []


        self._process_instr(mission.unwrapped.instrs)


        self.bfs_counter = 0


        self.bfs_step_counter = 0

    def replan(self, action_taken=None):
        self._process_obs()


        self._check_erroneous_box_opening(action_taken)


        for subgoal in self.stack:
            subgoal.update_agent_attributes()

        if self.stack:
            self.stack[-1].replan_after_action(action_taken)


        while self.stack and self.stack[-1].is_exploratory():
            self.stack.pop()

        suggested_action = None
        while self.stack:
            subgoal = self.stack[-1]
            suggested_action = subgoal.replan_before_action()


            if suggested_action is not None:
                break
        if not self.stack:
            suggested_action = self.mission.unwrapped.actions.done

        self._remember_current_state()

        return suggested_action

    def _find_obj_pos(self, obj_desc, adjacent=False):

        assert len(obj_desc.obj_set) > 0

        best_distance_to_obj = 999
        best_pos = None
        best_obj = None

        for i in range(len(obj_desc.obj_set)):
            if obj_desc.obj_set[i].type == "wall":
                continue
            try:
                if obj_desc.obj_set[i] == self.mission.unwrapped.carrying:
                    continue
                obj_pos = obj_desc.obj_poss[i]

                if self.vis_mask[obj_pos]:
                    shortest_path_to_obj, _, with_blockers = self._shortest_path(
                        lambda pos, cell: pos == obj_pos, try_with_blockers=True
                    )
                    assert shortest_path_to_obj is not None
                    distance_to_obj = len(shortest_path_to_obj)

                    if with_blockers:


                        distance_to_obj = len(shortest_path_to_obj) + (
                            7 if self.mission.unwrapped.carrying else 4
                        )


                    if distance_to_obj == 0:
                        distance_to_obj = 3 if adjacent else 2


                    if adjacent and distance_to_obj == 1:
                        distance_to_obj = 3

                    if distance_to_obj < best_distance_to_obj:
                        best_distance_to_obj = distance_to_obj
                        best_pos = obj_pos
                        best_obj = obj_desc.obj_set[i]
            except IndexError:


                pass

        return best_obj, best_pos

    def _process_obs(self):

        grid, vis_mask = self.mission.unwrapped.gen_obs_grid()

        view_size = self.mission.unwrapped.agent_view_size
        pos = self.mission.unwrapped.agent_pos
        f_vec = self.mission.unwrapped.dir_vec
        r_vec = self.mission.unwrapped.right_vec


        top_left = pos + f_vec * (view_size - 1) - r_vec * (view_size // 2)


        for vis_j in range(0, view_size):
            for vis_i in range(0, view_size):
                if not vis_mask[vis_i, vis_j]:
                    continue


                abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                if abs_i < 0 or abs_i >= self.vis_mask.shape[0]:
                    continue
                if abs_j < 0 or abs_j >= self.vis_mask.shape[1]:
                    continue

                self.vis_mask[abs_i, abs_j] = True

    def _remember_current_state(self):
        self.prev_agent_pos = self.mission.unwrapped.agent_pos
        self.prev_carrying = self.mission.unwrapped.carrying
        fwd_cell = self.mission.unwrapped.grid.get(
            *self.mission.unwrapped.agent_pos + self.mission.unwrapped.dir_vec
        )
        if fwd_cell and fwd_cell.type == "door":
            self.fwd_door_was_open = fwd_cell.is_open
        self.prev_fwd_cell = fwd_cell

    def _closest_wall_or_door_given_dir(self, position, direction):
        distance = 1
        while True:
            position_to_try = position + distance * direction


            if not self.mission.unwrapped.in_view(*position_to_try):
                return distance - 1
            cell = self.mission.unwrapped.grid.get(*position_to_try)
            if cell and (cell.type.endswith("door") or cell.type == "wall"):
                return distance
            distance += 1

    def _breadth_first_search(self, initial_states, accept_fn, ignore_blockers):
        self.bfs_counter += 1

        queue = [(state, None) for state in initial_states]
        grid = self.mission.unwrapped.grid
        previous_pos = dict()

        while len(queue) > 0:
            state, prev_pos = queue[0]
            queue = queue[1:]
            i, j, di, dj = state

            if (i, j) in previous_pos:
                continue

            self.bfs_step_counter += 1

            cell = grid.get(i, j)
            previous_pos[(i, j)] = prev_pos


            if accept_fn((i, j), cell):
                path = []
                pos = (i, j)
                while pos:
                    path.append(pos)
                    pos = previous_pos[pos]
                return path, (i, j), previous_pos


            if not self.vis_mask[i, j]:
                continue

            if cell:
                if cell.type == "wall":
                    continue

                elif cell.type == "door":

                    if not cell.is_open:
                        continue
                elif not ignore_blockers:
                    continue


            for k, l in [(di, dj), (dj, di), (-dj, -di), (-di, -dj)]:
                next_pos = (i + k, j + l)
                next_dir_vec = (k, l)
                next_state = (*next_pos, *next_dir_vec)
                queue.append((next_state, (i, j)))


        return None, None, previous_pos

    def _shortest_path(self, accept_fn, try_with_blockers=False):


        initial_states = [
            (*self.mission.unwrapped.agent_pos, *self.mission.unwrapped.dir_vec)
        ]

        path = finish = None
        with_blockers = False
        path, finish, previous_pos = self._breadth_first_search(
            initial_states, accept_fn, ignore_blockers=False
        )
        if not path and try_with_blockers:
            with_blockers = True
            path, finish, _ = self._breadth_first_search(
                [(i, j, 1, 0) for i, j in previous_pos], accept_fn, ignore_blockers=True
            )
            if path:


                pos = path[-1]
                extra_path = []
                while pos:
                    extra_path.append(pos)
                    pos = previous_pos[pos]
                path = path + extra_path[1:]

        if path:

            path = path[::-1]
            path = path[1:]


        return path, finish, with_blockers

    def _find_drop_pos(self, except_pos=None):

        grid = self.mission.unwrapped.grid

        def match_unblock(pos, cell):


            i, j = pos
            agent_pos = tuple(self.mission.unwrapped.agent_pos)

            if np.array_equal(pos, agent_pos):
                return False

            if except_pos and np.array_equal(pos, except_pos):
                return False

            if not self.vis_mask[i, j] or grid.get(i, j):
                return False


            cell_class = []
            for k, l in [
                (-1, -1),
                (0, -1),
                (1, -1),
                (1, 0),
                (1, 1),
                (0, 1),
                (-1, 1),
                (-1, 0),
            ]:
                nb_pos = (i + k, j + l)
                cell = grid.get(*nb_pos)

                if self.vis_mask[nb_pos] and cell and cell.type == "wall":
                    cell_class.append(1)

                elif (
                    self.vis_mask[nb_pos]
                    and (
                        not cell
                        or (cell.type == "door" and cell.is_open)
                        or nb_pos == agent_pos
                    )
                    and nb_pos != except_pos
                ):
                    cell_class.append(0)

                else:
                    cell_class.append(2)


            changes = 0
            for i in range(8):
                if bool(cell_class[(i + 1) % 8]) != bool(cell_class[i]):
                    changes += 1


            for i in range(8):
                next_i = (i + 1) % 8
                prev_i = (i + 7) % 8
                if (
                    cell_class[i] == 2
                    and cell_class[prev_i] != 0
                    and cell_class[next_i] != 0
                ):
                    return False

            return changes <= 2

        def match_empty(pos, cell):
            i, j = pos

            if np.array_equal(pos, self.mission.unwrapped.agent_pos):
                return False

            if except_pos and np.array_equal(pos, except_pos):
                return False

            if not self.vis_mask[pos] or grid.get(*pos):
                return False

            return True

        _, drop_pos, _ = self._shortest_path(match_unblock)

        if not drop_pos:
            _, drop_pos, _ = self._shortest_path(match_empty)

        if not drop_pos:
            _, drop_pos, _ = self._shortest_path(match_unblock, try_with_blockers=True)

        if not drop_pos:
            _, drop_pos, _ = self._shortest_path(match_empty, try_with_blockers=True)

        return drop_pos

    def _process_instr(self, instr):

        if isinstance(instr, GoToInstr):
            self.stack.append(GoNextToSubgoal(self, instr.desc))
            return

        if isinstance(instr, OpenInstr):
            self.stack.append(OpenSubgoal(self))
            self.stack.append(GoNextToSubgoal(self, instr.desc, reason="Open"))
            return

        if isinstance(instr, PickupInstr):


            self.stack.append(DropSubgoal(self))
            self.stack.append(PickupSubgoal(self))
            self.stack.append(GoNextToSubgoal(self, instr.desc))
            return

        if isinstance(instr, PutNextInstr):
            self.stack.append(DropSubgoal(self))
            self.stack.append(GoNextToSubgoal(self, instr.desc_fixed, reason="PutNext"))
            self.stack.append(PickupSubgoal(self))
            self.stack.append(GoNextToSubgoal(self, instr.desc_move))
            return

        if isinstance(instr, BeforeInstr) or isinstance(instr, AndInstr):
            self._process_instr(instr.instr_b)
            self._process_instr(instr.instr_a)
            return

        if isinstance(instr, AfterInstr):
            self._process_instr(instr.instr_a)
            self._process_instr(instr.instr_b)
            return

        assert False, "unknown instruction type"

    def _check_erroneous_box_opening(self, action):
        if (
            action == self.mission.unwrapped.actions.toggle
            and self.prev_fwd_cell is not None
            and self.prev_fwd_cell.type == "box"
        ):
            raise DisappearedBoxError("A box was opened. I am not sure I can help now.")
