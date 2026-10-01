"""LIBERO object ontology (compact, self-contained).

Maps object mentions in task descriptions to a canonical category + default role
+ physical flags (is_container / is_articulated / is_rotary). `ObjectOntology.scan`
returns non-overlapping, longest-first hits with character spans so the parser can
assign target / placement / landmark roles from the surrounding prepositions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .task_graph import ObjectRole


@dataclass(frozen=True)
class Entry:
    category: str
    default_role: ObjectRole = ObjectRole.TARGET
    is_container: bool = False
    is_articulated: bool = False
    is_rotary: bool = False
    aliases: tuple = ()


def _E(category, role=ObjectRole.TARGET, *, container=False, articulated=False,
       rotary=False, aliases=()):
    return Entry(category, role, container, articulated, rotary, tuple(aliases))


# Multi-word aliases must list the long form first (scan is longest-first).
_ENTRIES: List[Entry] = [
    # tableware
    _E("bowl", container=True, aliases=("black bowl", "white bowl", "middle black bowl",
        "chinese porcelain bowl", "akita black bowl", "bowl")),
    _E("plate", ObjectRole.CONTAINER, container=True, aliases=("left plate", "right plate",
        "plate", "dish")),
    _E("ramekin", ObjectRole.LANDMARK, container=True, aliases=("ramekin", "glazed rim porcelain ramekin")),
    _E("mug", aliases=("yellow and white mug", "white mug", "red mug", "mug")),
    # containers / receptacles
    _E("basket", ObjectRole.CONTAINER, container=True, aliases=("basket",)),
    _E("tray", ObjectRole.CONTAINER, container=True, aliases=("tray",)),
    _E("caddy", ObjectRole.CONTAINER, container=True, aliases=("caddy", "book caddy")),
    _E("wine_rack", ObjectRole.CONTAINER, container=True, aliases=("wine rack", "rack")),
    _E("cookie_box", ObjectRole.CONTAINER, container=True, aliases=("cookie box",)),
    _E("cookies", aliases=("cookies",)),
    # furniture / articulated
    _E("cabinet", ObjectRole.ARTICULATED, articulated=True, aliases=("wooden cabinet", "cabinet")),
    _E("drawer", ObjectRole.ARTICULATED, articulated=True, container=True,
        aliases=("top drawer", "bottom drawer", "middle drawer", "drawer")),
    _E("microwave", ObjectRole.ARTICULATED, articulated=True, container=True, aliases=("microwave",)),
    _E("stove", ObjectRole.FURNITURE, aliases=("stove", "flat stove")),
    _E("table", ObjectRole.FURNITURE, aliases=("table center", "table", "counter")),
    # manipulable food jars / cartons
    _E("alphabet_soup", aliases=("alphabet soup",)),
    _E("tomato_sauce", aliases=("tomato sauce",)),
    _E("bbq_sauce", aliases=("bbq sauce", "barbecue sauce")),
    _E("ketchup", aliases=("ketchup",)),
    _E("salad_dressing", aliases=("salad dressing",)),
    _E("cream_cheese", aliases=("cream cheese box", "cream cheese")),
    _E("butter", aliases=("butter",)),
    _E("milk", aliases=("milk",)),
    _E("chocolate_pudding", aliases=("chocolate pudding",)),
    _E("orange_juice", aliases=("orange juice",)),
    # other manipulables
    _E("wine_bottle", aliases=("wine bottle", "bottle")),
    _E("moka_pot", aliases=("moka pot", "coffee pot")),
    _E("frying_pan", aliases=("frying pan", "pan")),
    _E("book", aliases=("book",)),
]


@dataclass(frozen=True)
class ScanHit:
    start: int
    end: int
    text: str
    entry: Entry


class ObjectOntology:
    def __init__(self, entries: Optional[List[Entry]] = None) -> None:
        self._entries = list(entries if entries is not None else _ENTRIES)
        idx = {}
        for e in self._entries:
            for a in e.aliases:
                idx[a] = e
        self._idx = idx
        self._sorted = sorted(idx.keys(), key=len, reverse=True)

    def scan(self, text: str) -> List[ScanHit]:
        """Non-overlapping hits, longest alias first, returned in text order."""
        consumed = [False] * len(text)
        hits: List[ScanHit] = []
        for a in self._sorted:
            start = 0
            while True:
                i = text.find(a, start)
                if i < 0:
                    break
                if not any(consumed[i:i + len(a)]):
                    for k in range(i, i + len(a)):
                        consumed[k] = True
                    hits.append(ScanHit(i, i + len(a), a, self._idx[a]))
                start = i + 1
        hits.sort(key=lambda h: h.start)
        return hits
