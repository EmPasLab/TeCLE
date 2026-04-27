from __future__ import annotations

import itertools
import logging
import math
from typing import Any, Callable, Collection, Iterable, Iterator, Mapping, TypeVar


import numpy
import numpy as np
from numpy.typing import NBitBase, NDArray

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=NBitBase)


class Contradiction(Exception):

    pass


class TimedOut(Exception):

    pass


class StopEarly(Exception):

    pass


class Solver:

    def __init__(
        self,
        *,
        wave: NDArray[np.bool_],
        adj: Mapping[tuple[int, int], NDArray[numpy.bool_]],
        periodic: bool = False,
        backtracking: bool = False,
        on_backtrack: Callable[[], None] | None = None,
        on_choice: Callable[[int, int, int], None] | None = None,
        on_observe: Callable[[NDArray[numpy.bool_]], None] | None = None,
        on_propagate: Callable[[NDArray[numpy.bool_]], None] | None = None,
        check_feasible: Callable[[NDArray[numpy.bool_]], bool] | None = None,
    ) -> None:
        self.wave = wave
        self.adj = adj
        self.periodic = periodic
        self.backtracking = backtracking
        self.history: list[NDArray[np.bool_]] = []
        self.on_backtrack = on_backtrack
        self.on_choice = on_choice
        self.on_observe = on_observe
        self.on_propagate = on_propagate
        self.check_feasible = check_feasible

    @property
    def is_solved(self) -> bool:
        return (
            self.wave.sum() == self.wave.shape[1] * self.wave.shape[2]
            and (self.wave.sum(axis=0) == 1).all()
        )

    def solve_next(
        self,
        location_heuristic: Callable[[NDArray[numpy.bool_]], tuple[int, int]],
        pattern_heuristic: Callable[[NDArray[np.bool_], NDArray[np.bool_]], int],
    ) -> bool:
        if self.is_solved:
            return True
        if self.check_feasible and not self.check_feasible(self.wave):
            raise Contradiction("Not feasible.")
        if self.backtracking:
            self.history.append(self.wave.copy())
        propagate(
            self.wave, self.adj, periodic=self.periodic, onPropagate=self.on_propagate
        )
        pattern, i, j = None, None, None
        try:
            pattern, i, j = observe(self.wave, location_heuristic, pattern_heuristic)
            if self.on_choice:
                self.on_choice(pattern, i, j)
            self.wave[:, i, j] = False
            self.wave[pattern, i, j] = True
            if self.on_observe:
                self.on_observe(self.wave)
            propagate(
                self.wave,
                self.adj,
                periodic=self.periodic,
                onPropagate=self.on_propagate,
            )
            return False
        except Contradiction:
            if not self.backtracking:
                raise
            if not self.history:
                raise Contradiction("Every permutation has been attempted.")
            if self.on_backtrack:
                self.on_backtrack()
            self.wave = self.history.pop()
            self.wave[pattern, i, j] = False
            return False

    def solve(
        self,
        location_heuristic: Callable[[NDArray[numpy.bool_]], tuple[int, int]],
        pattern_heuristic: Callable[[NDArray[np.bool_], NDArray[np.bool_]], int],
    ) -> NDArray[np.int64]:
        while not self.solve_next(
            location_heuristic=location_heuristic, pattern_heuristic=pattern_heuristic
        ):
            pass
        return numpy.argmax(self.wave, axis=0)


def makeWave(
    n: int, w: int, h: int, ground: Iterable[int] | None = None
) -> NDArray[numpy.bool_]:
    wave: NDArray[numpy.bool_] = numpy.ones((n, w, h), dtype=numpy.bool_)
    if ground is not None:
        wave[:, :, h - 1] = False
        for g in ground:
            wave[
                g,
                :,
            ] = False
            wave[g, :, h - 1] = True


    return wave


def makeAdj(
    adjLists: Mapping[tuple[int, int], Collection[Iterable[int]]]
) -> dict[tuple[int, int], NDArray[numpy.bool_]]:
    adjMatrices = {}

    num_patterns = len(list(adjLists.values())[0])
    for d in adjLists:
        m = numpy.zeros((num_patterns, num_patterns), dtype=bool)
        for i, js in enumerate(adjLists[d]):

            for j in js:
                m[i, j] = 1


        adjMatrices[d] = m
    return adjMatrices


def makeRandomLocationHeuristic(
    preferences: NDArray[np.floating[Any]],
) -> Callable[[NDArray[np.bool_]], tuple[int, int]]:
    def randomLocationHeuristic(wave: NDArray[np.bool_]) -> tuple[int, int]:
        unresolved_cell_mask = numpy.count_nonzero(wave, axis=0) > 1
        cell_weights = numpy.where(unresolved_cell_mask, preferences, numpy.inf)
        row, col = numpy.unravel_index(numpy.argmin(cell_weights), cell_weights.shape)
        return row.item(), col.item()

    return randomLocationHeuristic


def makeEntropyLocationHeuristic(
    preferences: NDArray[np.floating[Any]],
) -> Callable[[NDArray[np.bool_]], tuple[int, int]]:
    def entropyLocationHeuristic(wave: NDArray[np.bool_]) -> tuple[int, int]:
        unresolved_cell_mask = numpy.count_nonzero(wave, axis=0) > 1
        cell_weights = numpy.where(
            unresolved_cell_mask,
            preferences + numpy.count_nonzero(wave, axis=0),
            numpy.inf,
        )
        row, col = numpy.unravel_index(numpy.argmin(cell_weights), cell_weights.shape)
        return row.item(), col.item()

    return entropyLocationHeuristic


def makeAntiEntropyLocationHeuristic(
    preferences: NDArray[np.floating[Any]],
) -> Callable[[NDArray[np.bool_]], tuple[int, int]]:
    def antiEntropyLocationHeuristic(wave: NDArray[np.bool_]) -> tuple[int, int]:
        unresolved_cell_mask = numpy.count_nonzero(wave, axis=0) > 1
        cell_weights = numpy.where(
            unresolved_cell_mask,
            preferences + numpy.count_nonzero(wave, axis=0),
            -numpy.inf,
        )
        row, col = numpy.unravel_index(numpy.argmax(cell_weights), cell_weights.shape)
        return row.item(), col.item()

    return antiEntropyLocationHeuristic


def spiral_transforms() -> Iterator[tuple[int, int]]:
    for N in itertools.count(start=1):
        if N % 2 == 0:
            yield (0, 1)
            for _ in range(N):
                yield (1, 0)
            for _ in range(N):
                yield (0, -1)
        else:
            yield (0, -1)
            for _ in range(N):
                yield (-1, 0)
            for _ in range(N):
                yield (0, 1)


def spiral_coords(x: int, y: int) -> Iterator[tuple[int, int]]:
    yield x, y
    for transform in spiral_transforms():
        x += transform[0]
        y += transform[1]
        yield x, y


def fill_with_curve(
    arr: NDArray[np.floating[T]], curve_gen: Iterable[Iterable[int]]
) -> NDArray[np.floating[T]]:
    arr_len = numpy.prod(arr.shape)
    fill = 0
    for coord in curve_gen:

        if fill < arr_len:
            try:
                arr[tuple(coord)] = fill / arr_len
                fill += 1
            except IndexError:
                pass
        else:
            break

    return arr


def makeSpiralLocationHeuristic(
    preferences: NDArray[np.floating[Any]],
) -> Callable[[NDArray[np.bool_]], tuple[int, int]]:


    spiral_gen = (
        sc for sc in spiral_coords(preferences.shape[0] // 2, preferences.shape[1] // 2)
    )

    cell_order = fill_with_curve(preferences, spiral_gen)

    def spiralLocationHeuristic(wave: NDArray[np.bool_]) -> tuple[int, int]:
        unresolved_cell_mask = numpy.count_nonzero(wave, axis=0) > 1
        cell_weights = numpy.where(unresolved_cell_mask, cell_order, numpy.inf)
        row, col = numpy.unravel_index(numpy.argmin(cell_weights), cell_weights.shape)
        return row.item(), col.item()

    return spiralLocationHeuristic


def makeHilbertLocationHeuristic(
    preferences: NDArray[np.floating[Any]],
) -> Callable[[NDArray[np.bool_]], tuple[int, int]]:
    from hilbertcurve.hilbertcurve import HilbertCurve

    curve_size = math.ceil(math.sqrt(max(preferences.shape[0], preferences.shape[1])))
    logger.debug(curve_size)
    curve_size = 4
    h_curve = HilbertCurve(curve_size, 2)
    h_coords = (h_curve.point_from_distance(i) for i in itertools.count())
    cell_order = fill_with_curve(preferences, h_coords)


    def hilbertLocationHeuristic(wave: NDArray[np.bool_]) -> tuple[int, int]:
        unresolved_cell_mask = numpy.count_nonzero(wave, axis=0) > 1
        cell_weights = numpy.where(unresolved_cell_mask, cell_order, numpy.inf)
        row, col = numpy.unravel_index(numpy.argmin(cell_weights), cell_weights.shape)
        return row.item(), col.item()

    return hilbertLocationHeuristic


def simpleLocationHeuristic(wave: NDArray[np.bool_]) -> tuple[int, int]:
    unresolved_cell_mask = numpy.count_nonzero(wave, axis=0) > 1
    cell_weights = numpy.where(
        unresolved_cell_mask, numpy.count_nonzero(wave, axis=0), numpy.inf
    )
    row, col = numpy.unravel_index(numpy.argmin(cell_weights), cell_weights.shape)
    return row.item(), col.item()


def lexicalLocationHeuristic(wave: NDArray[np.bool_]) -> tuple[int, int]:
    unresolved_cell_mask = numpy.count_nonzero(wave, axis=0) > 1
    cell_weights = numpy.where(unresolved_cell_mask, 1.0, numpy.inf)
    row, col = numpy.unravel_index(numpy.argmin(cell_weights), cell_weights.shape)
    return row.item(), col.item()


def lexicalPatternHeuristic(weights: NDArray[np.bool_], wave: NDArray[np.bool_]) -> int:
    return numpy.nonzero(weights)[0][0].item()


def makeWeightedPatternHeuristic(
    weights: NDArray[np.floating[Any]],
    np_random: numpy.random.Generator | None = None,
):
    num_of_patterns = len(weights)
    np_random: numpy.random.Generator = (
        numpy.random.default_rng() if np_random is None else np_random
    )

    def weightedPatternHeuristic(wave: NDArray[np.bool_], _: NDArray[np.bool_]) -> int:

        weighted_wave: NDArray[np.floating[Any]] = weights * wave
        weighted_wave /= weighted_wave.sum()
        result = np_random.choice(num_of_patterns, p=weighted_wave)
        return result

    return weightedPatternHeuristic


def makeRarestPatternHeuristic(
    weights: NDArray[np.floating[Any]],
    np_random: numpy.random.Generator | None = None,
) -> Callable[[NDArray[np.bool_], NDArray[np.bool_]], int]:
    np_random: numpy.random.Generator = (
        numpy.random.default_rng() if np_random is None else np_random
    )

    def weightedPatternHeuristic(
        wave: NDArray[np.bool_], total_wave: NDArray[np.bool_]
    ) -> int:
        logger.debug(total_wave.shape)

        wave_sums = numpy.sum(total_wave, (1, 2))

        selected_pattern = np_random.choice(
            numpy.where(wave_sums == wave_sums.max())[0]
        )
        return selected_pattern

    return weightedPatternHeuristic


def makeMostCommonPatternHeuristic(
    weights: NDArray[np.floating[Any]],
    np_random: numpy.random.Generator | None = None,
) -> Callable[[NDArray[np.bool_], NDArray[np.bool_]], int]:
    np_random: numpy.random.Generator = (
        numpy.random.default_rng() if np_random is None else np_random
    )

    def weightedPatternHeuristic(
        wave: NDArray[np.bool_], total_wave: NDArray[np.bool_]
    ) -> int:
        logger.debug(total_wave.shape)

        wave_sums = numpy.sum(total_wave, (1, 2))
        selected_pattern = np_random.choice(
            numpy.where(wave_sums == wave_sums.min())[0]
        )
        return selected_pattern

    return weightedPatternHeuristic


def makeRandomPatternHeuristic(
    weights: NDArray[np.floating[Any]],
    np_random: numpy.random.Generator | None = None,
) -> Callable[[NDArray[np.bool_], NDArray[np.bool_]], int]:
    num_of_patterns = len(weights)
    np_random: numpy.random.Generator = (
        numpy.random.default_rng() if np_random is None else np_random
    )

    def randomPatternHeuristic(wave: NDArray[np.bool_], _: NDArray[np.bool_]) -> int:

        weighted_wave = 1.0 * wave
        weighted_wave /= weighted_wave.sum()
        result = np_random.choice(num_of_patterns, p=weighted_wave)
        return result

    return randomPatternHeuristic


def make_global_use_all_patterns() -> Callable[[NDArray[np.bool_]], bool]:
    def global_use_all_patterns(wave: NDArray[np.bool_]) -> bool:
        return numpy.all(numpy.any(wave, axis=(1, 2))).item()

    return global_use_all_patterns


def propagate(
    wave: NDArray[np.bool_],
    adj: Mapping[tuple[int, int], NDArray[numpy.bool_]],
    periodic: bool = False,
    onPropagate: Callable[[NDArray[numpy.bool_]], None] | None = None,
) -> None:
    last_count = wave.sum()

    while True:
        supports = {}
        if periodic:
            padded = numpy.pad(wave, ((0, 0), (1, 1), (1, 1)), mode="wrap")
        else:
            padded = numpy.pad(
                wave, ((0, 0), (1, 1), (1, 1)), mode="constant", constant_values=True
            )


        for d in adj:
            dx, dy = d


            shifted = padded[
                :, 1 + dx : 1 + wave.shape[1] + dx, 1 + dy : 1 + wave.shape[2] + dy
            ]


            supports[d] = (adj[d] @ shifted.reshape(shifted.shape[0], -1)).reshape(
                shifted.shape
            ) > 0


        for d in adj:
            wave *= supports[d]

        if wave.sum() == last_count:
            break
        last_count = wave.sum()

    if onPropagate:
        onPropagate(wave)

    if (wave.sum(axis=0) == 0).any():
        raise Contradiction("Wave is in a contradictory state and can not be solved.")


def observe(
    wave: NDArray[np.bool_],
    locationHeuristic: Callable[[NDArray[np.bool_]], tuple[int, int]],
    patternHeuristic: Callable[[NDArray[np.bool_], NDArray[np.bool_]], int],
) -> tuple[int, int, int]:
    i, j = locationHeuristic(wave)
    pattern = patternHeuristic(wave[:, i, j], wave)
    return pattern, i, j


def run(
    wave: NDArray[np.bool_],
    adj: Mapping[tuple[int, int], NDArray[numpy.bool_]],
    locationHeuristic: Callable[[NDArray[numpy.bool_]], tuple[int, int]],
    patternHeuristic: Callable[[NDArray[np.bool_], NDArray[np.bool_]], int],
    periodic: bool = False,
    backtracking: bool = False,
    onBacktrack: Callable[[], None] | None = None,
    onChoice: Callable[[int, int, int], None] | None = None,
    onObserve: Callable[[NDArray[numpy.bool_]], None] | None = None,
    onPropagate: Callable[[NDArray[numpy.bool_]], None] | None = None,
    checkFeasible: Callable[[NDArray[numpy.bool_]], bool] | None = None,
    onFinal: Callable[[NDArray[numpy.bool_]], None] | None = None,
    depth: int = 0,
    depth_limit: int | None = None,
) -> NDArray[numpy.int64]:
    solver = Solver(
        wave=wave,
        adj=adj,
        periodic=periodic,
        backtracking=backtracking,
        on_backtrack=onBacktrack,
        on_choice=onChoice,
        on_observe=onObserve,
        on_propagate=onPropagate,
        check_feasible=checkFeasible,
    )
    while not solver.solve_next(
        location_heuristic=locationHeuristic, pattern_heuristic=patternHeuristic
    ):
        pass
    if onFinal:
        onFinal(solver.wave)
    return numpy.argmax(solver.wave, axis=0)
