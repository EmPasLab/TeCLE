from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from typing_extensions import Literal

PATTERN_PATH = Path(__file__).parent / "patterns"


@dataclass
class WFCConfig:

    pattern_path: Path
    tile_size: int = 1
    pattern_width: int = 2
    rotations: int = 8
    output_periodic: bool = False
    input_periodic: bool = False
    loc_heuristic: Literal[
        "lexical", "spiral", "entropy", "anti-entropy", "simple", "random"
    ] = "entropy"
    choice_heuristic: Literal["lexical", "rarest", "weighted", "random"] = "weighted"
    backtracking: bool = False

    @property
    def wfc_kwargs(self):
        try:
            from imageio.v2 import imread
        except ImportError as e:
            from gymnasium.error import DependencyNotInstalled

            raise DependencyNotInstalled(
                'imageio is missing, please run `pip install "minigrid[wfc]"`'
            ) from e
        kwargs = asdict(self)
        kwargs["image"] = imread(kwargs.pop("pattern_path"))[:, :, :3]
        return kwargs


WFC_PRESETS = {
    "MazeSimple": WFCConfig(
        pattern_path=PATTERN_PATH / "SimpleMaze.png",
        tile_size=1,
        pattern_width=2,
        output_periodic=False,
        input_periodic=False,
    ),
    "DungeonMazeScaled": WFCConfig(
        pattern_path=PATTERN_PATH / "ScaledMaze.png",
        tile_size=1,
        pattern_width=2,
        output_periodic=True,
        input_periodic=True,
    ),
    "RoomsFabric": WFCConfig(
        pattern_path=PATTERN_PATH / "Fabric.png",
        tile_size=1,
        pattern_width=3,
        output_periodic=False,
        input_periodic=False,
    ),
    "ObstaclesBlackdots": WFCConfig(
        pattern_path=PATTERN_PATH / "Blackdots.png",
        tile_size=1,
        pattern_width=2,
        output_periodic=False,
        input_periodic=False,
    ),
    "ObstaclesAngular": WFCConfig(
        pattern_path=PATTERN_PATH / "Angular.png",
        tile_size=1,
        pattern_width=3,
        output_periodic=True,
        input_periodic=True,
    ),
    "ObstaclesHogs3": WFCConfig(
        pattern_path=PATTERN_PATH / "Hogs.png",
        tile_size=1,
        pattern_width=3,
        output_periodic=True,
        input_periodic=True,
    ),
}


WFC_PRESETS_INCONSISTENT = {
    "MazeKnot": WFCConfig(
        pattern_path=PATTERN_PATH / "Knot.png",
        tile_size=1,
        pattern_width=3,
        output_periodic=True,
        input_periodic=True,
    ),
    "MazeWall": WFCConfig(
        pattern_path=PATTERN_PATH / "SimpleWall.png",
        tile_size=1,
        pattern_width=2,
        output_periodic=True,
        input_periodic=True,
    ),
    "RoomsOffice": WFCConfig(
        pattern_path=PATTERN_PATH / "Office.png",
        tile_size=1,
        pattern_width=3,
        output_periodic=True,
        input_periodic=True,
    ),
    "ObstaclesHogs2": WFCConfig(
        pattern_path=PATTERN_PATH / "Hogs.png",
        tile_size=1,
        pattern_width=2,
        output_periodic=True,
        input_periodic=True,
    ),
    "Skew2": WFCConfig(
        pattern_path=PATTERN_PATH / "Skew2.png",
        tile_size=1,
        pattern_width=3,
        output_periodic=True,
        input_periodic=True,
    ),
}


WFC_PRESETS_SLOW = {
    "Maze": WFCConfig(
        pattern_path=PATTERN_PATH / "Maze.png",
        tile_size=1,
        pattern_width=3,
        output_periodic=True,
        input_periodic=True,
    ),
    "MazeSpirals": WFCConfig(
        pattern_path=PATTERN_PATH / "Spirals.png",
        tile_size=1,
        pattern_width=3,
        output_periodic=True,
        input_periodic=True,
    ),
    "MazePaths": WFCConfig(
        pattern_path=PATTERN_PATH / "Paths.png",
        tile_size=1,
        pattern_width=3,
        output_periodic=True,
        input_periodic=True,
    ),
    "Mazelike": WFCConfig(
        pattern_path=PATTERN_PATH / "Mazelike.png",
        tile_size=1,
        pattern_width=3,
        output_periodic=True,
        input_periodic=True,
    ),
    "Dungeon": WFCConfig(
        pattern_path=PATTERN_PATH / "DungeonExtr.png",
        tile_size=1,
        pattern_width=3,
        output_periodic=True,
        input_periodic=True,
    ),
    "DungeonRooms": WFCConfig(
        pattern_path=PATTERN_PATH / "Rooms.png",
        tile_size=1,
        pattern_width=3,
        output_periodic=True,
        input_periodic=True,
    ),
    "DungeonLessRooms": WFCConfig(
        pattern_path=PATTERN_PATH / "LessRooms.png",
        tile_size=1,
        pattern_width=3,
        output_periodic=True,
        input_periodic=True,
    ),
    "DungeonSpirals": WFCConfig(
        pattern_path=PATTERN_PATH / "SpiralsNeg.png",
        tile_size=1,
        pattern_width=3,
        output_periodic=True,
        input_periodic=True,
    ),
    "RoomsMagicOffice": WFCConfig(
        pattern_path=PATTERN_PATH / "MagicOffice.png",
        tile_size=1,
        pattern_width=3,
        output_periodic=True,
        input_periodic=True,
    ),
    "SkewCave": WFCConfig(
        pattern_path=PATTERN_PATH / "Cave.png",
        tile_size=1,
        pattern_width=3,
        output_periodic=False,
        input_periodic=False,
    ),
    "SkewLake": WFCConfig(
        pattern_path=PATTERN_PATH / "Lake.png",
        tile_size=1,
        pattern_width=3,
        output_periodic=True,
        input_periodic=True,
    ),
}
