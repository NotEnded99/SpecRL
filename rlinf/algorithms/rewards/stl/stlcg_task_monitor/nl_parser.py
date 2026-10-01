"""Rule-based NL -> TaskGraph parser (self-contained, no safe_stl dependency).

Handles the LIBERO task-description families:
  * pick-place with spatial relations ("pick up the black bowl between the plate
    and the ramekin and place it on the plate")
  * place-into-container ("pick up the alphabet soup and place it in the basket")
  * multi-target ("put both the alphabet soup and the tomato sauce in the basket")
  * articulated ("open the middle drawer of the cabinet", "close the top drawer")
  * rotary ("turn on the stove")

It is intentionally pragmatic: object-role assignment comes from ontology defaults
refined by preposition position.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from .ontology import ObjectOntology, ScanHit
from .task_graph import (ObjectInfo, ObjectRole, TaskGraph, TaskStep, Template)


# verb keyword -> canonical verb. Order matters (longer phrases first).
_VERBS: List[Tuple[str, str]] = [
    ("pick up", "pick"), ("pick", "pick"),
    ("put", "place"), ("place", "place"), ("set", "place"),
    ("stack", "stack"), ("insert", "insert"),
    ("open", "open"), ("close", "close"),
    ("turn on", "turn_on"), ("turn off", "turn_off"),
    ("push", "push"),
]
_PLACEMENT_PREP = ("on top of", "onto", "on", "into", "inside", "in")
_SPATIAL_PREP = ("between", "next to", "beside", "under", "below", "above",
                 "left of", "right of", "in front of", "behind", "of")
_LEVELS = ("top", "middle", "bottom")

# Positional adjectives for object instance disambiguation
# "the left bowl", "the middle black bowl", "the back black bowl"
_POSITIONAL_ADJ = ("left", "right", "middle", "front", "back")

_PRONOUNS = ("it", "them", "their", "they")


def _split_clauses(text: str) -> List[str]:
    """Split a compound task into clauses on ' and <verb>' / ' , ' boundaries."""
    # split on " and " only when followed by a verb keyword
    pat = re.compile(r"\s+and\s+(?=pick|put|place|set|stack|insert|open|close|turn|push)", re.I)
    parts = pat.split(text)
    return [p.strip() for p in parts if p.strip()]


def _find_verb(clause: str) -> Tuple[Optional[str], int, int]:
    """Return (canonical_verb, start, end) of the first verb keyword, else (None,...)."""
    low = clause.lower()
    for kw, canon in _VERBS:
        m = re.search(r"\b" + re.escape(kw) + r"\b", low)
        if m:
            return canon, m.start(), m.end()
    return None, -1, -1


def _level_in(clause: str) -> Optional[str]:
    low = clause.lower()
    for lv in _LEVELS:
        if re.search(r"\b" + lv + r"\b", low):
            return lv
    return None


def _objects_after(word: str, clause: str, hits: List[ScanHit]) -> List[ScanHit]:
    """Hits that start at/after the given keyword's position."""
    m = re.search(r"\b" + re.escape(word) + r"\b", clause.lower())
    if not m:
        return []
    pos = m.end()
    return [h for h in hits if h.start >= pos]


def _parse_clause(clause: str, onto: ObjectOntology,
                  prev_target: Optional[ObjectInfo],
                  prev_container: Optional[str] = None,
                  prev_level: Optional[str] = None) -> Tuple[List[TaskStep], List[ObjectInfo], Optional[ObjectInfo]]:
    """Parse one clause -> (steps, objects, target_object_info)."""
    verb, vs, ve = _find_verb(clause)
    hits = onto.scan(clause)
    # don't treat the verb span as an object (e.g. "pick" vs nothing); hits are object aliases only

    objs: List[ObjectInfo] = []
    for h in hits:
        role = h.entry.default_role
        objs.append(ObjectInfo(name=h.entry.category, raw_text=h.text.strip(),
                               role=role, category=h.entry.category))

    if verb is None:
        return [], objs, prev_target

    def _hit(name_cat: str) -> Optional[ScanHit]:
        for h in hits:
            if h.entry.category == name_cat:
                return h
        return None

    target_info: Optional[ObjectInfo] = None
    step_target: Optional[str] = None
    step_placement: Optional[str] = None
    step_relation: Optional[str] = None
    step_level: Optional[str] = None

    # ---- articulated / rotary verbs ---------------------------------------
    if verb in ("open", "close"):
        art = next((o for o in objs if o.role == ObjectRole.ARTICULATED), None)
        has_it = bool(re.search(r"\b(it|them)\b", clause.lower()))
        if art is not None:
            step_target = art.name
            target_info = art
            step_level = _level_in(clause)
        elif has_it and prev_container:
            # "close it" / "open it" -> the articulated container from a prior clause
            step_target = prev_container
            step_level = _level_in(clause) or prev_level
        steps = [TaskStep(verb=verb, object=step_target, level=step_level)]
        return steps, objs, target_info

    if verb in ("turn_on", "turn_off"):
        stove = next((o for o in objs if o.category == "stove"), None) or \
                next((o for o in objs if o.role == ObjectRole.FURNITURE), None)
        if stove is not None:
            step_target = stove.name
            target_info = stove
        steps = [TaskStep(verb=verb, object=step_target)]
        return steps, objs, target_info

    # ---- pick / place / stack / insert / push -----------------------------
    # placement: first object (any role) after a placement preposition
    placement_hit: Optional[ScanHit] = None
    prep_found = None
    for prep in _PLACEMENT_PREP:
        if not re.search(r"\b" + re.escape(prep) + r"\b", clause.lower()):
            continue
        aft = _objects_after(prep, clause, hits)
        prep_found = prep
        if aft:
            placement_hit = aft[0]
            step_placement = aft[0].entry.category
            step_relation = "into" if prep in ("in", "into", "inside") else "on_top_of"
        break
    # placement prep with no object after it ("put the bowl inside") -> inherit prior container
    if step_placement is None and prep_found in ("in", "into", "inside") and prev_container:
        step_placement = prev_container
        step_relation = "into"

    # the placement object is a container/destination, NOT a target to manipulate
    if placement_hit is not None:
        for o in objs:
            if o.category == placement_hit.entry.category and o.raw_text == placement_hit.text.strip():
                o.role = ObjectRole.CONTAINER

    # targets: ALL TARGET-role objects (handles "both X and Y"); pronoun carry-over
    tg_list = [o for o in objs if o.role == ObjectRole.TARGET]
    pronoun = any(re.search(r"\b" + p + r"\b", clause.lower()) for p in _PRONOUNS)
    if not tg_list and pronoun and prev_target is not None:
        tg_list = [prev_target]
    target_info = tg_list[0] if tg_list else None

    # pronoun placement ("put the moka pot on it" -> it = previous stove/target)
    if step_placement is None and re.search(r"\bon (it|them)\b", clause.lower()) \
            and prev_target is not None:
        step_placement = prev_target.name
        step_relation = "on_top_of"

    # spatial reference as placement fallback ("put X to the right of the plate" /
    # "next to the ramekin") -> the referenced object IS the placement target
    if step_placement is None:
        for prep in ("right of", "left of", "next to", "beside", "in front of",
                     "front of", "behind"):
            aft = _objects_after(prep, clause, hits)
            if aft:
                step_placement = aft[0].entry.category
                step_relation = prep.replace(" ", "_")
                break
    # also record a pure spatial relation (between/under) for bookkeeping
    for prep in ("between", "under"):
        if re.search(r"\b" + re.escape(prep) + r"\b", clause.lower()):
            step_relation = step_relation or prep.replace(" ", "_")
            break

    steps: List[TaskStep] = []
    if not tg_list:
        # no manipulable target (e.g. pure "turn on" handled above); nothing here
        return steps, objs, target_info
    if verb == "pick":
        for tg in tg_list:
            steps.append(TaskStep(verb="pick", object=tg.name, obj_instance=tg.instance_id))
    else:  # place/stack/insert/push — one goal step per target, shared placement
        for tg in tg_list:
            steps.append(TaskStep(verb=verb, object=tg.name, target=step_placement,
                                  relation=step_relation, obj_instance=tg.instance_id))
    return steps, objs, target_info


def parse_task(text: str, onto: Optional[ObjectOntology] = None) -> TaskGraph:
    """Parse a natural-language task description into a TaskGraph."""
    onto = onto or ObjectOntology()
    text = text.strip()
    clauses = _split_clauses(text)

    all_steps: List[TaskStep] = []
    all_objs: List[ObjectInfo] = []
    seen_cats: set = set()
    prev_target: Optional[ObjectInfo] = None
    prev_container: Optional[str] = None
    prev_level: Optional[str] = None
    placement_target: Optional[str] = None
    reference_object: Optional[str] = None
    spatial_relation: Optional[str] = None

    for cl in clauses:
        steps, objs, tgt = _parse_clause(cl, onto, prev_target, prev_container, prev_level)
        for o in objs:
            if o.category not in seen_cats:
                all_objs.append(o)
                seen_cats.add(o.category)
        for s in steps:
            all_steps.append(s)
            if s.verb in ("place", "stack", "insert", "push") and s.target:
                placement_target = s.target
                prev_container = s.target          # carry for "close it" / "inside"
                if s.relation:
                    spatial_relation = s.relation
            if s.verb in ("open", "close") and s.object:
                prev_container = s.object           # articulated container (drawer/microwave)
                if s.level:
                    prev_level = s.level
        if tgt is not None:
            prev_target = tgt

    target_object = prev_target.name if prev_target is not None else None

    # template inference
    verbs = {s.verb for s in all_steps}
    has_pick = "pick" in verbs
    has_art = any(s.verb in ("open", "close") for s in all_steps)
    has_knob = any(s.verb in ("turn_on", "turn_off") for s in all_steps)
    if has_knob:
        template = Template.KNOB_TWIST
    elif has_art:
        template = Template.ARTICULATED
    elif has_pick:
        template = Template.PICK_PLACE
    else:
        template = Template.PICK_PLACE

    return TaskGraph(
        template=template,
        raw_text=text,
        objects=all_objs,
        steps=all_steps,
        target_object=target_object,
        placement_target=placement_target,
        reference_object=reference_object,
        spatial_relation=spatial_relation,
    )
