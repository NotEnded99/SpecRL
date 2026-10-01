"""TaskGraph: the semantic parse of a natural-language manipulation task.

Clean reimplementation (independent of safe_stl). A TaskGraph captures *intent*:
which template, which objects in which roles, and the ordered verb steps that
achieve the goal. It is consumed by the formula builder to emit an stlcgpp STL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Template(str, Enum):
    PICK_PLACE = "pick-place"
    ARTICULATED = "articulated-manipulation"   # drawer / cabinet door / microwave
    KNOB_TWIST = "knob-twist"                  # stove knob
    PUSH_NO_LIFT = "push-no-lift"
    WINE_RACK_INSERT = "wine-rack-insert"
    NAVIGATION = "navigation"


class ObjectRole(str, Enum):
    TARGET = "target"           # object being manipulated/transported
    CONTAINER = "container"     # receptacle things go into/onto
    LANDMARK = "landmark"       # spatial reference ("between the plate and ramekin")
    FURNITURE = "furniture"
    ARTICULATED = "articulated"
    ROTARY = "rotary"
    SCENE = "scene"


_OBJ_ID = [0]


@dataclass
class ObjectInfo:
    """A mentioned object. Each instance gets a unique id at creation so that pronoun
    references ('it' -> the previous target) share the id (same logical object), while
    two distinct named objects ('white mug' vs 'yellow mug') get different ids."""
    name: str                   # canonical id, e.g. "black_bowl"
    raw_text: str               # original NL span
    role: ObjectRole
    category: str               # ontology category, e.g. "bowl"
    instance_id: int = -1

    def __post_init__(self):
        if self.instance_id < 0:
            _OBJ_ID[0] += 1
            self.instance_id = _OBJ_ID[0]


@dataclass
class TaskStep:
    """One primitive action. verb in {pick,place,stack,insert,open,close,
    turn_on,turn_off,twist,push}. object = acted-on body; target = destination/
    reference; level = top/middle/bottom for articulated joints. obj_instance is the
    logical-object id of `object` (pronoun references share the prior object's id)."""
    verb: str
    object: Optional[str] = None
    target: Optional[str] = None
    relation: Optional[str] = None
    level: Optional[str] = None
    obj_instance: Optional[int] = None


@dataclass
class TaskGraph:
    template: Template
    raw_text: str
    objects: List[ObjectInfo] = field(default_factory=list)
    steps: List[TaskStep] = field(default_factory=list)
    target_object: Optional[str] = None
    placement_target: Optional[str] = None
    reference_object: Optional[str] = None
    spatial_relation: Optional[str] = None

    def object_by_name(self, name: str) -> Optional[ObjectInfo]:
        for o in self.objects:
            if o.name == name:
                return o
        return None

    def objects_with_role(self, role: ObjectRole) -> List[ObjectInfo]:
        return [o for o in self.objects if o.role == role]
