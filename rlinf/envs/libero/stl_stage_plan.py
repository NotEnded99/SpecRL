"""Audited temporal-stage plans for the 40 LIBERO tasks.

Every atom listed here comes from the atoms exported by the existing LIBERO
STL environment.  An arrow in the audit table becomes a separate Stage.

Multi-object tasks contain two alternative paths.  The runtime evaluates both
specifications and accepts either legal global object order.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence


Atom = tuple[str, Optional[str], Optional[str]]

_UNARY_PREDS = {"open", "close", "turnon", "turnoff"}


def canonical_atom(atom: Sequence[object]) -> Atom:
    """Canonicalize unary predicates to (pred, None, target)."""
    if len(atom) != 3:
        raise ValueError(f"Expected a 3-tuple atom, got {atom!r}.")

    pred = str(atom[0]).lower()
    pred = {"turn_on": "turnon", "turn_off": "turnoff"}.get(pred, pred)
    obj = None if atom[1] is None else str(atom[1])
    target = None if atom[2] is None else str(atom[2])

    if pred in _UNARY_PREDS and target is None and obj is not None:
        obj, target = None, obj

    return pred, obj, target


@dataclass(frozen=True)
class Stage:
    """One gate. All atoms in a stage must be simultaneously satisfied."""

    name: str
    atoms: tuple[Atom, ...]
    virtual: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Stage.name must not be empty.")
        if not self.atoms:
            raise ValueError(f"Stage {self.name!r} has no atoms.")
        object.__setattr__(
            self,
            "atoms",
            tuple(canonical_atom(atom) for atom in self.atoms),
        )


@dataclass(frozen=True)
class StagePath:
    """One legal ordered specification for a task."""

    name: str
    stages: tuple[Stage, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("StagePath.name must not be empty.")
        if not self.stages:
            raise ValueError(f"StagePath {self.name!r} has no stages.")

    @property
    def atoms(self) -> tuple[Atom, ...]:
        return _unique_atoms(
            atom
            for stage in self.stages
            for atom in stage.atoms
        )


@dataclass(frozen=True, init=False)
class StagePlan:
    """One or more legal ordered paths for a task.

    ``stages=...`` remains accepted for compatibility with the earlier
    single-path tests. New code should use ``paths=...``.
    """

    paths: tuple[StagePath, ...]
    gated: bool
    reason: str
    audited: bool

    def __init__(
        self,
        *,
        paths: Optional[Sequence[StagePath]] = None,
        stages: Optional[Sequence[Stage]] = None,
        gated: bool = True,
        reason: str = "",
        audited: bool = True,
    ) -> None:
        if (paths is None) == (stages is None):
            raise ValueError("Provide exactly one of paths= or stages=.")

        if stages is not None:
            resolved_paths = (
                StagePath("default", tuple(stages)),
            )
        else:
            resolved_paths = tuple(paths or ())

        if not resolved_paths:
            raise ValueError("StagePlan must contain at least one path.")

        object.__setattr__(self, "paths", resolved_paths)
        object.__setattr__(self, "gated", bool(gated))
        object.__setattr__(self, "reason", str(reason))
        object.__setattr__(self, "audited", bool(audited))

    @property
    def stages(self) -> tuple[Stage, ...]:
        """Compatibility view: stages of the first candidate path."""
        return self.paths[0].stages

    @property
    def atoms(self) -> tuple[Atom, ...]:
        """Union of atoms required by all candidate paths."""
        return _unique_atoms(
            atom
            for path in self.paths
            for atom in path.atoms
        )

    @property
    def has_alternatives(self) -> bool:
        return len(self.paths) > 1


def _unique_atoms(atoms: Iterable[Atom]) -> tuple[Atom, ...]:
    seen: set[Atom] = set()
    ordered: list[Atom] = []

    for atom in atoms:
        atom = canonical_atom(atom)
        if atom not in seen:
            seen.add(atom)
            ordered.append(atom)

    return tuple(ordered)


def _atom_name(atom: Atom) -> str:
    pred, obj, target = atom
    args = [value for value in (obj, target) if value is not None]
    return f"{pred}({','.join(args)})"


def _path(name: str, *atoms: Atom) -> StagePath:
    """Create a path in which every supplied atom is its own gate."""
    return StagePath(
        name=name,
        stages=tuple(
            Stage(
                name=f"{index + 1:02d}_{_atom_name(atom)}",
                atoms=(canonical_atom(atom),),
            )
            for index, atom in enumerate(atoms)
        ),
    )


def _plan(
    task_name: str,
    *paths: StagePath,
) -> StagePlan:
    return StagePlan(
        paths=paths,
        gated=True,
        audited=True,
        reason=f"audited exported-atom order: {task_name}",
    )


PICK = lambda obj: ("pick", obj, None)
ON = lambda obj, target: ("on", obj, target)
IN = lambda obj, target: ("in", obj, target)
OPEN = lambda target: ("open", None, target)
CLOSE = lambda target: ("close", None, target)
TURNON = lambda target: ("turnon", None, target)


def _build_audited_plans() -> dict[str, StagePlan]:
    plans: dict[str, StagePlan] = {}

    def add(description: str, *paths: StagePath) -> None:
        key = _normalize_description(description)
        if key in plans:
            raise ValueError(f"Duplicate task description: {description}")
        plans[key] = _plan(description, *paths)

    # LIBERO-Goal
    add(
        "open the middle drawer of the cabinet",
        _path("only", OPEN("wooden_cabinet_1_middle_region")),
    )
    add(
        "put the bowl on the stove",
        _path(
            "only",
            PICK("akita_black_bowl_1"),
            ON("akita_black_bowl_1", "flat_stove_1_cook_region"),
        ),
    )
    add(
        "put the wine bottle on top of the cabinet",
        _path(
            "only",
            PICK("wine_bottle_1"),
            ON("wine_bottle_1", "wooden_cabinet_1_top_side"),
        ),
    )
    add(
        "open the top drawer and put the bowl inside",
        _path(
            "drawer_initially_open",
            PICK("akita_black_bowl_1"),
            IN(
                "akita_black_bowl_1",
                "wooden_cabinet_1_top_region",
            ),
        ),
    )
    add(
        "put the bowl on top of the cabinet",
        _path(
            "only",
            PICK("akita_black_bowl_1"),
            ON(
                "akita_black_bowl_1",
                "wooden_cabinet_1_top_side",
            ),
        ),
    )
    add(
        "push the plate to the front of the stove",
        _path(
            "only",
            PICK("plate_1"),
            ON("plate_1", "main_table_stove_front_region"),
        ),
    )
    add(
        "put the cream cheese in the bowl",
        _path(
            "only",
            PICK("cream_cheese_1"),
            ON("cream_cheese_1", "akita_black_bowl_1"),
        ),
    )
    add(
        "turn on the stove",
        _path("only", TURNON("flat_stove_1")),
    )
    add(
        "put the bowl on the plate",
        _path(
            "only",
            PICK("akita_black_bowl_1"),
            ON("akita_black_bowl_1", "plate_1"),
        ),
    )
    add(
        "put the wine bottle on the rack",
        _path(
            "only",
            PICK("wine_bottle_1"),
            ON("wine_bottle_1", "wine_rack_1_top_region"),
        ),
    )

    # LIBERO-Object
    object_tasks = (
        ("alphabet soup", "alphabet_soup_1"),
        ("cream cheese", "cream_cheese_1"),
        ("salad dressing", "salad_dressing_1"),
        ("bbq sauce", "bbq_sauce_1"),
        ("ketchup", "ketchup_1"),
        ("tomato sauce", "tomato_sauce_1"),
        ("butter", "butter_1"),
        ("milk", "milk_1"),
        ("chocolate pudding", "chocolate_pudding_1"),
        ("orange juice", "orange_juice_1"),
    )
    for natural_name, object_id in object_tasks:
        add(
            f"pick up the {natural_name} and place it in the basket",
            _path(
                "only",
                PICK(object_id),
                IN(object_id, "basket_1_contain_region"),
            ),
        )

    # LIBERO-Spatial: all ten tasks export the same two atoms.
    spatial_descriptions = (
        "pick up the black bowl between the plate and the ramekin and place it on the plate",
        "pick up the black bowl next to the ramekin and place it on the plate",
        "pick up the black bowl from table center and place it on the plate",
        "pick up the black bowl on the cookie box and place it on the plate",
        "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
        "pick up the black bowl on the ramekin and place it on the plate",
        "pick up the black bowl next to the cookie box and place it on the plate",
        "pick up the black bowl on the stove and place it on the plate",
        "pick up the black bowl next to the plate and place it on the plate",
        "pick up the black bowl on the wooden cabinet and place it on the plate",
    )
    for description in spatial_descriptions:
        add(
            description,
            _path(
                "only",
                PICK("akita_black_bowl_1"),
                ON("akita_black_bowl_1", "plate_1"),
            ),
        )

    # LIBERO-10
    def add_two_object_task(
        description: str,
        first_pick: Atom,
        first_goal: Atom,
        second_pick: Atom,
        second_goal: Atom,
        *,
        prefix: tuple[Atom, ...] = (),
    ) -> None:
        add(
            description,
            _path(
                "object_1_then_object_2",
                *prefix,
                first_pick,
                first_goal,
                second_pick,
                second_goal,
            ),
            _path(
                "object_2_then_object_1",
                *prefix,
                second_pick,
                second_goal,
                first_pick,
                first_goal,
            ),
        )

    add_two_object_task(
        "put both the alphabet soup and the tomato sauce in the basket",
        PICK("alphabet_soup_1"),
        IN("alphabet_soup_1", "basket_1_contain_region"),
        PICK("tomato_sauce_1"),
        IN("tomato_sauce_1", "basket_1_contain_region"),
    )
    add_two_object_task(
        "put both the cream cheese box and the butter in the basket",
        PICK("cream_cheese_1"),
        IN("cream_cheese_1", "basket_1_contain_region"),
        PICK("butter_1"),
        IN("butter_1", "basket_1_contain_region"),
    )
    add(
        "turn on the stove and put the moka pot on it",
        _path(
            "only",
            TURNON("flat_stove_1"),
            PICK("moka_pot_1"),
            ON("moka_pot_1", "flat_stove_1_cook_region"),
        ),
    )
    add(
        "put the black bowl in the bottom drawer of the cabinet and close it",
        _path(
            "only",
            PICK("akita_black_bowl_1"),
            IN(
                "akita_black_bowl_1",
                "white_cabinet_1_bottom_region",
            ),
            CLOSE("white_cabinet_1_bottom_region"),
        ),
    )
    add_two_object_task(
        "put the white mug on the left plate and put the yellow and white mug on the right plate",
        PICK("porcelain_mug_1"),
        ON("porcelain_mug_1", "plate_1"),
        PICK("white_yellow_mug_1"),
        ON("white_yellow_mug_1", "plate_2"),
    )
    add(
        "pick up the book and place it in the back compartment of the caddy",
        _path(
            "only",
            PICK("black_book_1"),
            IN(
                "black_book_1",
                "desk_caddy_1_back_contain_region",
            ),
        ),
    )
    add_two_object_task(
        "put the white mug on the plate and put the chocolate pudding to the right of the plate",
        PICK("porcelain_mug_1"),
        ON("porcelain_mug_1", "plate_1"),
        PICK("chocolate_pudding_1"),
        ON(
            "chocolate_pudding_1",
            "living_room_table_plate_right_region",
        ),
    )
    add_two_object_task(
        "put both the alphabet soup and the cream cheese box in the basket",
        PICK("alphabet_soup_1"),
        IN("alphabet_soup_1", "basket_1_contain_region"),
        PICK("cream_cheese_1"),
        IN("cream_cheese_1", "basket_1_contain_region"),
    )
    add_two_object_task(
        "put both moka pots on the stove",
        PICK("moka_pot_1"),
        ON("moka_pot_1", "flat_stove_1_cook_region"),
        PICK("moka_pot_2"),
        ON("moka_pot_2", "flat_stove_1_cook_region"),
        prefix=(TURNON("flat_stove_1"),),
    )
    add(
        "put the yellow and white mug in the microwave and close it",
        _path(
            "only",
            PICK("white_yellow_mug_1"),
            IN(
                "white_yellow_mug_1",
                "microwave_1_heating_region",
            ),
            CLOSE("microwave_1"),
        ),
    )

    if len(plans) != 40:
        raise AssertionError(
            f"Expected 40 audited tasks, constructed {len(plans)}."
        )

    return plans


def _normalize_description(description: str) -> str:
    return " ".join(str(description).strip().lower().split())


def _ground_role_paths(
    role_paths: Sequence[StagePath],
    audited_plan: Optional[StagePlan],
) -> tuple[Optional[tuple[StagePath, ...]], str]:
    """Bind a role-level plan's roles to concrete names via an audited plan.

    Matching is per path (paired by index) and in-order per predicate: every
    LLM atom must find an unconsumed audited atom with the same predicate, and
    every audited atom must be consumed.  Roles bind through the matched
    atoms' slots; a role bound to two different concrete names fails.

    Returns ``(grounded_paths, reason)``; ``grounded_paths`` is ``None`` when
    the plan cannot be fully grounded.
    """
    if audited_plan is None:
        return None, "no audited plan for this description"

    if len(role_paths) != len(audited_plan.paths):
        return None, (
            f"path count differs (llm={len(role_paths)}, "
            f"audited={len(audited_plan.paths)})"
        )

    # Binding keys are (pred, role): one NL role may legitimately bind to
    # different physical entities per predicate, e.g. turnon(stove) binds the
    # fixture body while on(pot, stove) binds its cooking site.
    binding: dict[tuple[str, str], str] = {}
    grounded_paths: list[StagePath] = []

    for role_path, audited_path in zip(role_paths, audited_plan.paths):
        role_atoms = [
            atom for stage in role_path.stages for atom in stage.atoms
        ]
        audited_atoms = [
            atom for stage in audited_path.stages for atom in stage.atoms
        ]
        used = [False] * len(audited_atoms)
        concrete_atoms: list[Atom] = []

        for atom in role_atoms:
            pred = atom[0]
            match_idx = next(
                (
                    i
                    for i, (candidate, consumed) in enumerate(
                        zip(audited_atoms, used)
                    )
                    if not consumed and candidate[0] == pred
                ),
                None,
            )
            if match_idx is None:
                return None, (
                    f"predicate {pred!r} in llm path {role_path.name!r} "
                    "has no audited counterpart"
                )
            used[match_idx] = True
            matched = audited_atoms[match_idx]

            for role_slot, concrete_slot in (
                (atom[1], matched[1]),
                (atom[2], matched[2]),
            ):
                if role_slot is None or concrete_slot is None:
                    continue
                key = (pred, role_slot)
                previous = binding.get(key)
                if previous is not None and previous != concrete_slot:
                    return None, (
                        f"role {role_slot!r} under {pred!r} bound to both "
                        f"{previous!r} and {concrete_slot!r}"
                    )
                binding[key] = concrete_slot

            concrete_atoms.append(matched)

        if not all(used):
            missing = [
                audited_atoms[i]
                for i, consumed in enumerate(used)
                if not consumed
            ]
            return None, (
                f"audited atoms missing from llm path "
                f"{role_path.name!r}: {missing}"
            )

        grounded_paths.append(
            StagePath(
                name=role_path.name,
                stages=tuple(
                    Stage(
                        name=f"{index + 1:02d}_{_atom_name(atom)}",
                        atoms=(atom,),
                    )
                    for index, atom in enumerate(concrete_atoms)
                ),
            )
        )

    return tuple(grounded_paths), f"bound {len(binding)} roles"


def _parity_records_to_entries(records: list) -> dict[str, dict]:
    """Convert parity-results records into the plan-table entry schema.

    Each record's ``llm_formula`` (``{"paths": [...]}`` with dict atoms
    ``{"pred", "obj", "target"}``) becomes an ungrounded entry keyed by the
    record's ``description``; atoms are rewritten to the loader's
    ``(pred, obj, target)`` tuple form.  Records without ``llm_formula``
    (parse failures) are skipped.
    """
    entries: dict[str, dict] = {}
    for record in records:
        formula = record.get("llm_formula")
        if not formula:
            continue
        entries[_normalize_description(record["description"])] = {
            "paths": [
                {
                    "name": path_entry.get("name", "only"),
                    "stages": [
                        {
                            "atoms": [
                                [
                                    atom["pred"],
                                    atom.get("obj"),
                                    atom.get("target"),
                                ]
                                for atom in stage["atoms"]
                            ],
                        }
                        for stage in path_entry["stages"]
                    ],
                }
                for path_entry in formula["paths"]
            ],
            "grounded": False,
        }
    return entries


def _load_plans_from_json(path: str) -> dict[str, StagePlan]:
    """Load the plan table from a static JSON export.

    The schema matches the output of
    ``examples/embodiment/agm_stage_runtime/dump_stage_plans.py``.  Keys are raw
    task descriptions and are normalized here, so lookups behave identically to
    the in-code table.

    Entries marked ``"grounded": false`` (e.g. the role-level LLM export
    ``llm_symbolic_parity_results_revised.json``) are grounded at load time by
    aligning them against the in-code audited table (see
    :func:`_ground_role_paths`).  Entries whose grounding fails fall back to
    the audited concrete plan, so a misparse can never corrupt the staged
    reward.

    The file may also be in the parity-results schema -- a LIST of per-task
    records (as written by ``llm_symbolic_parity.py`` and its revised
    variant) -- in which case each record's ``llm_formula`` is used as the
    (ungrounded) paths for its ``description``.
    """
    with open(path, "r", encoding="utf-8") as handle:
        raw: Any = json.load(handle)

    if isinstance(raw, list):
        raw = _parity_records_to_entries(raw)

    entries: dict[str, dict] = {}
    needs_grounding = False
    for description, entry in raw.items():
        key = _normalize_description(description)
        if key in entries:
            raise ValueError(
                f"Duplicate task description in {path!r}: {description!r}"
            )
        entries[key] = entry
        if not bool(entry.get("grounded", True)):
            needs_grounding = True

    audited_plans = _build_audited_plans() if needs_grounding else None

    plans: dict[str, StagePlan] = {}
    report: dict[str, str] = {}
    counts = {"grounded_from_llm": 0, "fallback_audited": 0, "loaded_as_is": 0}

    for key, entry in entries.items():
        paths = tuple(
            StagePath(
                name=str(path_entry["name"]),
                stages=tuple(
                    Stage(
                        # Stage names are optional in the schema: role-level
                        # (ungrounded) exports omit them; synthesize the same
                        # NN_pred(...) style _path uses so Stage's non-empty
                        # name check holds.
                        name=str(
                            stage.get("name")
                            or f"{stage_index + 1:02d}_"
                            + "_then_".join(
                                _atom_name(canonical_atom(atom))
                                for atom in stage["atoms"]
                            )
                        ),
                        atoms=tuple(
                            canonical_atom(atom)
                            for atom in stage["atoms"]
                        ),
                    )
                    for stage_index, stage in enumerate(path_entry["stages"])
                ),
            )
            for path_entry in entry["paths"]
        )

        if bool(entry.get("grounded", True)):
            plans[key] = StagePlan(
                paths=paths,
                gated=bool(entry.get("gated", True)),
                reason=str(entry.get("reason", "")),
                audited=bool(entry.get("audited", True)),
            )
            report[key] = "loaded_as_is"
            counts["loaded_as_is"] += 1
            continue

        grounded, ground_reason = _ground_role_paths(
            paths,
            audited_plans.get(key) if audited_plans else None,
        )

        if grounded is not None:
            plans[key] = StagePlan(
                paths=grounded,
                gated=True,
                reason=(
                    "llm-generated; grounded against audited table "
                    f"({ground_reason})"
                ),
                audited=False,
            )
            report[key] = "grounded_from_llm"
            counts["grounded_from_llm"] += 1
            continue

        audited = audited_plans.get(key) if audited_plans else None
        if audited is not None:
            plans[key] = audited
            report[key] = f"fallback_audited: {ground_reason}"
            counts["fallback_audited"] += 1
        else:
            # No audited binding exists (new task); omit the entry so the
            # runtime joint fallback applies, exactly like an unknown task.
            report[key] = f"skipped: {ground_reason}"

    if not plans:
        raise ValueError(f"No stage plans found in {path!r}.")

    global _PLAN_JSON_GROUNDING_REPORT
    _PLAN_JSON_GROUNDING_REPORT = report
    print(
        f"[AGM_PLAN_JSON] loaded {len(plans)} plans from {path}: "
        f"grounded_from_llm={counts['grounded_from_llm']}, "
        f"fallback_audited={counts['fallback_audited']}, "
        f"loaded_as_is={counts['loaded_as_is']}",
        flush=True,
    )

    return plans


# RLINF_AGM_STAGE_PLAN_JSON selects a static JSON export of the audited table
# (see run_SpecRL_P.sh).  Unset keeps the in-code table.
_PLAN_JSON_PATH = os.environ.get("RLINF_AGM_STAGE_PLAN_JSON", "").strip()

# Populated by _load_plans_from_json: per-description grounding status, e.g.
# {"put the bowl on the stove": "grounded_from_llm", ...}.  Debug/audit only.
_PLAN_JSON_GROUNDING_REPORT: dict[str, str] = {}
if _PLAN_JSON_PATH:
    AUDITED_PLANS_BY_DESCRIPTION = _load_plans_from_json(_PLAN_JSON_PATH)
else:
    AUDITED_PLANS_BY_DESCRIPTION = _build_audited_plans()


_STANDARD_SUITE_TASKS: dict[str, tuple[str, ...]] = {
    "libero_goal": tuple(
        description
        for description in AUDITED_PLANS_BY_DESCRIPTION
        if description in {
            "open the middle drawer of the cabinet",
            "put the bowl on the stove",
            "put the wine bottle on top of the cabinet",
            "open the top drawer and put the bowl inside",
            "put the bowl on top of the cabinet",
            "push the plate to the front of the stove",
            "put the cream cheese in the bowl",
            "turn on the stove",
            "put the bowl on the plate",
            "put the wine bottle on the rack",
        }
    ),
}

# Keep standard-suite task-id lookup explicit and independent of dictionary
# filtering order.
_STANDARD_SUITE_TASKS["libero_goal"] = (
    "open the middle drawer of the cabinet",
    "put the bowl on the stove",
    "put the wine bottle on top of the cabinet",
    "open the top drawer and put the bowl inside",
    "put the bowl on top of the cabinet",
    "push the plate to the front of the stove",
    "put the cream cheese in the bowl",
    "turn on the stove",
    "put the bowl on the plate",
    "put the wine bottle on the rack",
)
_STANDARD_SUITE_TASKS["libero_object"] = tuple(
    f"pick up the {name} and place it in the basket"
    for name in (
        "alphabet soup",
        "cream cheese",
        "salad dressing",
        "bbq sauce",
        "ketchup",
        "tomato sauce",
        "butter",
        "milk",
        "chocolate pudding",
        "orange juice",
    )
)
_STANDARD_SUITE_TASKS["libero_spatial"] = (
    "pick up the black bowl between the plate and the ramekin and place it on the plate",
    "pick up the black bowl next to the ramekin and place it on the plate",
    "pick up the black bowl from table center and place it on the plate",
    "pick up the black bowl on the cookie box and place it on the plate",
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
    "pick up the black bowl on the ramekin and place it on the plate",
    "pick up the black bowl next to the cookie box and place it on the plate",
    "pick up the black bowl on the stove and place it on the plate",
    "pick up the black bowl next to the plate and place it on the plate",
    "pick up the black bowl on the wooden cabinet and place it on the plate",
)
_STANDARD_SUITE_TASKS["libero_10"] = (
    "put both the alphabet soup and the tomato sauce in the basket",
    "put both the cream cheese box and the butter in the basket",
    "turn on the stove and put the moka pot on it",
    "put the black bowl in the bottom drawer of the cabinet and close it",
    "put the white mug on the left plate and put the yellow and white mug on the right plate",
    "pick up the book and place it in the back compartment of the caddy",
    "put the white mug on the plate and put the chocolate pudding to the right of the plate",
    "put both the alphabet soup and the cream cheese box in the basket",
    "put both moka pots on the stove",
    "put the yellow and white mug in the microwave and close it",
)


def _joint_fallback(
    goal_atoms: Sequence[Atom],
    reason: str,
) -> StagePlan:
    atoms = _unique_atoms(canonical_atom(atom) for atom in goal_atoms)
    if not atoms:
        raise ValueError("Cannot build a fallback plan without goal atoms.")

    return StagePlan(
        stages=(Stage("joint_goal", atoms),),
        gated=False,
        audited=False,
        reason=reason,
    )


def build_stage_plan(
    task_suite_name: str,
    task_id: int,
    goal_atoms: Sequence[Atom],
    task_description: Optional[str] = None,
) -> StagePlan:
    """Return an audited plan, using task text for LIBERO-40 robustness.

    LIBERO-40's global task-id ordering is deliberately not assumed. The
    environment should pass ``task_description``; standard 10-task suites can
    also fall back to their local task id.
    """
    suite = str(task_suite_name).strip().lower()
    description = (
        _normalize_description(task_description)
        if task_description
        else ""
    )

    if not description and suite in _STANDARD_SUITE_TASKS:
        descriptions = _STANDARD_SUITE_TASKS[suite]
        if 0 <= int(task_id) < len(descriptions):
            description = _normalize_description(
                descriptions[int(task_id)]
            )

    plan = AUDITED_PLANS_BY_DESCRIPTION.get(description)
    canonical_goals = _unique_atoms(
        canonical_atom(atom) for atom in goal_atoms
    )

    if plan is None:
        return _joint_fallback(
            canonical_goals,
            reason=(
                "no audited task-text match; preserving current goal atoms "
                "as one joint stage"
            ),
        )

    planned_non_pick = {
        atom for atom in plan.atoms if atom[0] != "pick"
    }
    actual_non_pick = {
        atom for atom in canonical_goals if atom[0] != "pick"
    }

    if planned_non_pick != actual_non_pick:
        return _joint_fallback(
            canonical_goals,
            reason=(
                "audited task matched by text but its non-pick atoms did "
                f"not match runtime goal atoms: planned={planned_non_pick}, "
                f"runtime={actual_non_pick}"
            ),
        )

    return plan
