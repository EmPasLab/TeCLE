from __future__ import annotations

from typing import Any, Callable

from gymnasium import spaces
from gymnasium.utils import seeding


def check_if_no_duplicate(duplicate_list: list) -> bool:
    return len(set(duplicate_list)) == len(duplicate_list)


class MissionSpace(spaces.Space[str]):

    def __init__(
        self,
        mission_func: Callable[..., str],
        ordered_placeholders: list[list[str]] | None = None,
        seed: int | seeding.RandomNumberGenerator | None = None,
    ):

        if ordered_placeholders is not None:
            assert (
                len(ordered_placeholders) == mission_func.__code__.co_argcount
            ), f"The number of placeholders {len(ordered_placeholders)} is different from the number of parameters in the mission function {mission_func.__code__.co_argcount}."
            for placeholder_list in ordered_placeholders:
                assert check_if_no_duplicate(
                    placeholder_list
                ), "Make sure that the placeholders don't have any duplicate values."
        else:
            assert (
                mission_func.__code__.co_argcount == 0
            ), f"If the ordered placeholders are {ordered_placeholders}, the mission function shouldn't have any parameters."

        self.ordered_placeholders = ordered_placeholders
        self.mission_func = mission_func

        super().__init__(dtype=str, seed=seed)


        sampled_mission = self.sample()
        assert isinstance(
            sampled_mission, str
        ), f"mission_func must return type str not {type(sampled_mission)}"

    def sample(self) -> str:
        if self.ordered_placeholders is not None:
            placeholders = []
            for rand_var_list in self.ordered_placeholders:
                idx = self.np_random.integers(0, len(rand_var_list))

                placeholders.append(rand_var_list[idx])

            return self.mission_func(*placeholders)
        else:
            return self.mission_func()

    def contains(self, x: Any) -> bool:

        if self.ordered_placeholders is not None:
            check_placeholder_list = []
            for placeholder_list in self.ordered_placeholders:
                for placeholder in placeholder_list:
                    if placeholder in x:
                        check_placeholder_list.append(placeholder)


            check_placeholder_list = list(set(check_placeholder_list))

            start_id_placeholder = []
            end_id_placeholder = []

            new_check_placeholder_list = []
            for placeholder in check_placeholder_list:
                new_start_id_placeholder = [
                    i for i in range(len(x)) if x.startswith(placeholder, i)
                ]
                new_check_placeholder_list += [placeholder] * len(
                    new_start_id_placeholder
                )
                end_id_placeholder += [
                    start_id + len(placeholder) - 1
                    for start_id in new_start_id_placeholder
                ]
                start_id_placeholder += new_start_id_placeholder


            ordered_placeholder_list = sorted(
                zip(
                    start_id_placeholder, end_id_placeholder, new_check_placeholder_list
                )
            )


            remove_placeholder_id = []
            for i, placeholder_1 in enumerate(ordered_placeholder_list):
                starting_id = i + 1
                for j, placeholder_2 in enumerate(
                    ordered_placeholder_list[starting_id:]
                ):

                    if max(placeholder_1[0], placeholder_2[0]) < min(
                        placeholder_1[1], placeholder_2[1]
                    ):
                        remove_placeholder = min(
                            placeholder_1[2], placeholder_2[2], key=len
                        )
                        if remove_placeholder == placeholder_1[2]:
                            remove_placeholder_id.append(i)
                        else:
                            remove_placeholder_id.append(i + j + 1)
            for id in remove_placeholder_id:
                del ordered_placeholder_list[id]

            final_placeholders = [
                placeholder[2] for placeholder in ordered_placeholder_list
            ]


            for orered_placeholder, final_placeholder in zip(
                self.ordered_placeholders, final_placeholders
            ):
                if final_placeholder in orered_placeholder:
                    continue
                else:
                    return False
            try:
                mission_string_with_placeholders = self.mission_func(
                    *final_placeholders
                )
            except Exception as e:
                print(
                    f"{x} is not contained in MissionSpace due to the following exception: {e}"
                )
                return False

            return bool(mission_string_with_placeholders == x)

        else:
            return bool(self.mission_func() == x)

    def __repr__(self) -> str:
        return f"MissionSpace({self.mission_func}, {self.ordered_placeholders})"

    def __eq__(self, other) -> bool:
        if isinstance(other, MissionSpace):

            if self.ordered_placeholders is not None:

                if (
                    len(self.ordered_placeholders) == len(other.ordered_placeholders)
                ) and (
                    all(
                        set(i) == set(j)
                        for i, j in zip(
                            self.ordered_placeholders, other.ordered_placeholders
                        )
                    )
                ):

                    test_placeholders = [""] * len(self.ordered_placeholders)
                    mission = self.mission_func(*test_placeholders)
                    other_mission = other.mission_func(*test_placeholders)
                    return mission == other_mission
            else:

                if other.ordered_placeholders is None:

                    mission = self.mission_func()
                    other_mission = other.mission_func()
                    return mission == other_mission


        return False
