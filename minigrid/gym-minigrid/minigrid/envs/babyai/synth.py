from __future__ import annotations

from minigrid.envs.babyai.core.levelgen import LevelGen


class Synth(LevelGen):

    def __init__(self, room_size=8, num_rows=3, num_cols=3, num_dists=18, **kwargs):


        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            num_dists=num_dists,
            instr_kinds=["action"],
            locations=False,
            unblocking=True,
            implicit_unlock=False,
            **kwargs,
        )


class SynthLoc(LevelGen):

    def __init__(self, **kwargs):


        super().__init__(
            instr_kinds=["action"],
            locations=True,
            unblocking=True,
            implicit_unlock=False,
            **kwargs,
        )


class SynthSeq(LevelGen):

    def __init__(self, **kwargs):


        super().__init__(
            locations=True, unblocking=True, implicit_unlock=False, **kwargs
        )


class MiniBossLevel(LevelGen):

    def __init__(self, **kwargs):
        super().__init__(
            num_cols=2,
            num_rows=2,
            room_size=5,
            num_dists=7,
            locked_room_prob=0.25,
            **kwargs,
        )


class BossLevel(LevelGen):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class BossLevelNoUnlock(LevelGen):

    def __init__(self, **kwargs):
        super().__init__(locked_room_prob=0, implicit_unlock=False, **kwargs)
