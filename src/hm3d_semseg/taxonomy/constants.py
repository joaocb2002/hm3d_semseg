"""Single authoritative model-label definition."""

from __future__ import annotations

from typing import Dict, List

UNKNOWN_ID = 0
IGNORE_INDEX = 255
NUM_CLASSES = 41

# Index in this tuple is the authoritative Matterport mpcat40index (1..40).
MPCAT40_NAMES: List[str] = [
    "wall",
    "floor",
    "chair",
    "door",
    "table",
    "picture",
    "cabinet",
    "cushion",
    "window",
    "sofa",
    "bed",
    "curtain",
    "chest_of_drawers",
    "plant",
    "sink",
    "stairs",
    "ceiling",
    "toilet",
    "stool",
    "towel",
    "mirror",
    "tv_monitor",
    "shower",
    "column",
    "bathtub",
    "counter",
    "fireplace",
    "lighting",
    "beam",
    "railing",
    "shelving",
    "blinds",
    "gym_equipment",
    "seating",
    "board_panel",
    "furniture",
    "appliances",
    "clothes",
    "objects",
    "misc",
]

ID2LABEL: Dict[int, str] = {0: "unknown"}
ID2LABEL.update({index: name for index, name in enumerate(MPCAT40_NAMES, start=1)})
LABEL2ID: Dict[str, int] = {name: index for index, name in ID2LABEL.items()}

OBJECTNAV_SIX: Dict[str, int] = {
    "chair": LABEL2ID["chair"],
    "couch": LABEL2ID["sofa"],
    "potted plant": LABEL2ID["plant"],
    "bed": LABEL2ID["bed"],
    "toilet": LABEL2ID["toilet"],
    "tv": LABEL2ID["tv_monitor"],
}

assert len(ID2LABEL) == NUM_CLASSES
assert set(ID2LABEL) == set(range(NUM_CLASSES))
