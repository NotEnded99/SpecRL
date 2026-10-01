"""Layer 1 (shared by both branches): NL -> observation-agnostic symbolic STL.

The symbolic STL formula has the SAME temporal structure (ordered stages /
alternative legal paths) as the audited LIBERO plans, but every concrete
instance name is replaced by a semantic ROLE token, e.g.::

    "put the black bowl in the bottom drawer of the cabinet and close it"
        pick(bowl) -> in(bowl, cabinet_bottom_drawer_site) -> close(cabinet_bottom_drawer_site)

Roles (``bowl``, ``cabinet_bottom_drawer_site``) are NOT environment instance
names; they are abstract semantic symbols.  Both the privileged branch and the
vision branch consume this symbolic formula and differ only in how they bind
roles to concrete evidence (Layer 2 grounding).

The authoritative temporal structure comes from the existing audited plan table
(:data:`stl_stage_plan.AUDITED_PLANS_BY_DESCRIPTION`), so the symbolic plan is
guaranteed to carry the same stages / atoms / alternative paths the current
AGM-stage env uses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from rlinf.envs.libero.stl_stage_plan import (
    AUDITED_PLANS_BY_DESCRIPTION,
    Atom,
    Stage,
    StagePath,
    StagePlan,
    _normalize_description,
    build_stage_plan,
    canonical_atom,
)

# Readable role labels: strip the same decorative prefixes the NL grounder uses
# (see nl_grounder.build_vocab) and mark LIBERO *site/region* targets, which are
# functional surfaces (cook_region, top_side, contain_region, ...) rather than
# manipulable object bodies.
_PREFIXES = (
    "akita_",
    "wooden_",
    "flat_",
    "glazed_rim_porcelain_",
    "chefmate_8_",
    "porcelain_",
    "red_",
    "yellow_",
    "white_",
    "new_",
)
# Longest first so composite suffixes win over their tails.
_REGION_SUFFIXES = (
    "_back_contain_region",
    "_heating_region",
    "_contain_region",
    "_cook_region",
    "_middle_region",
    "_bottom_region",
    "_top_region",
    "_right_region",
    "_front_region",
    "_top_side",
    "_region",
)
_SITE_MARK = "_site"


def _semantic_role(name: Optional[str]) -> Optional[str]:
    """Concrete instance/site name -> a readable semantic role token."""
    if name is None:
        return None
    role = name.lower()
    for prefix in _PREFIXES:
        if role.startswith(prefix):
            role = role[len(prefix):]
            break
    role = re.sub(r"_\d+$", "", role)  # trailing instance index (_1, _2)
    is_site = False
    for suffix in _REGION_SUFFIXES:
        if role.endswith(suffix):
            role = role[: -len(suffix)]
            is_site = True
            break
    role = role.strip("_")
    if not role:
        role = name.lower()
    return role + (_SITE_MARK if is_site else "")


def _concrete_to_role(plan: StagePlan) -> Dict[str, str]:
    """Map every distinct concrete name in the plan to a unique role token.

    Duplicate categories (e.g. ``moka_pot_1`` and ``moka_pot_2`` both -> 'moka_pot')
    are disambiguated with a suffix so the mapping is invertible.
    """
    ordered_names = []
    seen = set()
    for atom in plan.atoms:
        for slot in (atom[1], atom[2]):
            if slot and slot not in seen:
                seen.add(slot)
                ordered_names.append(slot)

    concrete_to_role: Dict[str, str] = {}
    used_roles: set = set()
    for name in ordered_names:
        base = _semantic_role(name) or name.lower()
        role = base
        if role in used_roles:
            for tag in ("_b", "_c", "_d", "_e", "_f"):
                if base + tag not in used_roles:
                    role = base + tag
                    break
        used_roles.add(role)
        concrete_to_role[name] = role
    return concrete_to_role


def _role_atom(atom: Atom, concrete_to_role: Dict[str, str]) -> Atom:
    pred, obj, target = canonical_atom(atom)
    return (
        pred,
        concrete_to_role.get(obj) if obj else None,
        concrete_to_role.get(target) if target else None,
    )


@dataclass(frozen=True)
class AtomSpatial:
    """Optional spatial-disambiguation annotation on one role-atom.

    Carries the positional phrase that disambiguates an object among several
    same-category instances, e.g. "the bowl *between the plate and the ramekin*"
    -> ``AtomSpatial(relation="between", refs=("plate", "ramekin"))`` on the
    ``pick(bowl)`` atom. ``refs`` are themselves role tokens.

    This is metadata for the vision branch; the privileged grounder ignores it
    (the audited binding already names the correct instance). It never alters
    predicate evaluation -- the closed-vocab Atom tuple is unchanged.
    """

    relation: str  # "between" | "next_to" | "on_the" | "in_the" | ...
    refs: Tuple[str, ...]  # role tokens of the landmark references


@dataclass(frozen=True)
class SymbolicStage:
    """One gate: a conjunction of role-atoms that must hold simultaneously."""

    name: str
    atoms: Tuple[Atom, ...]  # atoms whose obj/target are ROLE tokens


@dataclass(frozen=True)
class SymbolicPath:
    """One legal ordered specification, expressed with role tokens."""

    name: str
    stages: Tuple[SymbolicStage, ...]


@dataclass(frozen=True)
class SymbolicStagePlan:
    """Observation-agnostic STL formula for one task.

    ``paths`` (role atoms) is the observation-agnostic view consumed by both
    branches.  ``concrete_plan`` / ``role_to_concrete`` are OPTIONAL: the
    audited path fills them (it knows the real LIBERO instance names, and the
    shared runtime needs a concrete plan); an LLM-generated or otherwise
    ungrounded symbolic plan leaves them unset until a grounder binds the roles
    to concrete evidence.
    """

    description: str
    gated: bool
    audited: bool
    reason: str
    has_alternatives: bool
    paths: Tuple[SymbolicPath, ...]
    concrete_plan: Optional[StagePlan] = None
    role_to_concrete: Tuple[Tuple[str, str], ...] = ()
    # Optional per-atom spatial disambiguation (role-atom -> annotation). Empty
    # by default; populated only when a spatial phrase disambiguates an object
    # (e.g. "between the plate and the ramekin" on pick(bowl)). Ignored by the
    # privileged grounder; carried for the vision branch.
    atom_spatial: Tuple[Tuple[Atom, "AtomSpatial"], ...] = ()

    @property
    def atoms(self) -> Tuple[Atom, ...]:
        """Union of role-atoms across all paths (deduplicated, order-preserving)."""
        seen: set = set()
        ordered = []
        for path in self.paths:
            for stage in path.stages:
                for atom in stage.atoms:
                    if atom not in seen:
                        seen.add(atom)
                        ordered.append(atom)
        return tuple(ordered)


def build_symbolic_plan(
    description: str,
    task_suite_name: str = "libero_40",
    task_id: int = 0,
) -> SymbolicStagePlan:
    """Parse a natural-language instruction into a symbolic (role-based) STL plan.

    The temporal structure is the audited one, so this is identical in shape to
    what the current AGM-stage env produces via ``build_stage_plan``; only the
    instance names are abstracted to semantic roles.
    """
    norm = _normalize_description(description)
    plan: Optional[StagePlan] = AUDITED_PLANS_BY_DESCRIPTION.get(norm)
    if plan is None:
        # No audited match: delegate to build_stage_plan, which returns a joint
        # fallback stage over whatever goal atoms are available.
        plan = build_stage_plan(
            task_suite_name=task_suite_name,
            task_id=task_id,
            goal_atoms=[],
            task_description=description,
        )

    concrete_to_role = _concrete_to_role(plan)
    role_to_concrete = tuple(sorted(
        {role: concrete for concrete, role in concrete_to_role.items()}.items()
    ))

    symbolic_paths = tuple(
        SymbolicPath(
            name=path.name,
            stages=tuple(
                SymbolicStage(
                    name=stage.name,
                    atoms=tuple(
                        _role_atom(atom, concrete_to_role) for atom in stage.atoms
                    ),
                )
                for stage in path.stages
            ),
        )
        for path in plan.paths
    )

    return SymbolicStagePlan(
        description=description,
        gated=plan.gated,
        audited=plan.audited,
        reason=plan.reason,
        has_alternatives=plan.has_alternatives,
        concrete_plan=plan,
        role_to_concrete=role_to_concrete,
        paths=symbolic_paths,
    )
