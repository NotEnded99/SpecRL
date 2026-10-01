"""Layer 2: grounding -- bind symbolic roles to concrete evidence.

This is the ONLY layer where the privileged and vision branches differ.

* :class:`PrivilegedGrounder` binds roles to concrete LIBERO object / site names
  using the privileged scene state.  Its binding reproduces the audited plan
  instance names exactly, so the resulting STL formula is identical to the
  current AGM-stage env (the parity guarantee).

* :class:`VisionGrounder` is an **interface only** -- it reserves the hook a
  future vision module will implement (bind roles to image instances / target
  regions).  It is intentionally not implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, Tuple, runtime_checkable

from rlinf.envs.libero.stl_stage_plan import Atom, canonical_atom

from rlinf.envs.libero.nlstl.symbolic_plan import SymbolicStagePlan


@dataclass(frozen=True)
class Grounding:
    """Result of grounding a symbolic plan: concrete atoms + the role binding."""

    atoms: Tuple[Atom, ...]
    binding: Tuple[Tuple[str, str], ...]  # role -> concrete name
    source: str  # 'privileged' | 'vision'


@runtime_checkable
class Grounder(Protocol):
    """Bind the roles of a symbolic plan to concrete evidence."""

    def ground(
        self,
        symbolic_plan: SymbolicStagePlan,
        evidence: Optional[Any] = None,
    ) -> Grounding:  # pragma: no cover - protocol
        ...


def _role_to_concrete_atom(atom: Atom, role_to_concrete) -> Atom:
    pred, obj_role, target_role = atom
    obj = role_to_concrete.get(obj_role) if obj_role else None
    target = role_to_concrete.get(target_role) if target_role else None
    return canonical_atom((pred, obj, target))


class PrivilegedGrounder:
    """Bind symbolic roles to concrete LIBERO instance names.

    The authoritative binding is the audited one (``symbolic_plan.role_to_concrete``),
    which is exactly what the current AGM-stage env uses, so the grounded atoms
    match the existing STL formula bit-for-bit.  When a live privileged
    ``evidence`` (a :class:`~rlinf.envs.libero.nlstl.evidence.UnifiedEvidence`
    built from the MuJoCo state) is supplied, the grounder additionally
    re-resolves each category from the scene via ``resolve_object_key`` and
    asserts it agrees with the audited binding -- i.e. the privileged scene is
    genuinely used, not bypassed.
    """

    def __init__(self, verify_against_scene: bool = True) -> None:
        self.verify_against_scene = verify_against_scene

    def ground(
        self,
        symbolic_plan: SymbolicStagePlan,
        evidence: Optional[Any] = None,
    ) -> Grounding:
        role_to_concrete = dict(symbolic_plan.role_to_concrete)

        if evidence is not None and self.verify_against_scene:
            self._verify_against_scene(symbolic_plan, evidence, role_to_concrete)

        concrete_atoms = tuple(
            _role_to_concrete_atom(atom, role_to_concrete)
            for atom in symbolic_plan.atoms
        )
        return Grounding(
            atoms=concrete_atoms,
            binding=symbolic_plan.role_to_concrete,
            source="privileged",
        )

    @staticmethod
    def _verify_against_scene(
        symbolic_plan: SymbolicStagePlan,
        evidence: Any,
        role_to_concrete,
    ) -> None:
        """Re-resolve each role's category from the live scene and check parity.

        Lazy import keeps the symbolic/grounding layers importable without the
        MuJoCo / libero stack.
        """
        from rlinf.algorithms.rewards.stl.stlcg_task_monitor.loader import (
            resolve_object_key,
        )

        ep = evidence.get_episode()
        for role, concrete in role_to_concrete.items():
            if role.endswith("_site"):
                continue  # site/region targets are resolved by name, not category
            resolved = resolve_object_key(concrete, ep)
            if resolved is not None and resolved != concrete:
                raise RuntimeError(
                    "Privileged grounding mismatch for role "
                    f"{role!r}: scene resolved to {resolved!r}, "
                    f"audited binding is {concrete!r}."
                )


class VisionGrounder:
    """Reserved interface for the vision branch.

    A future implementation will bind each symbolic role to evidence extracted
    from images (detections / segmentation masks / target regions) and emit a
    :class:`Grounding` whose concrete atoms are image-space identifiers plus a
    :class:`~rlinf.envs.libero.nlstl.evidence.UnifiedEvidence` filled from
    vision.  Predicate evaluation and STL aggregation stay shared.

    Concretely, to feed the *shared* predicate code a vision grounder must, per
    role, populate the unified-evidence fields the predicates read (object
    centres, target surface points/sites, articulated joint progress, ...).  See
    ``evidence.evidence_from_vision`` for the contract.  Not implemented here.
    """

    def ground(
        self,
        symbolic_plan: SymbolicStagePlan,
        evidence: Optional[Any] = None,
    ) -> Grounding:  # pragma: no cover - interface only
        raise NotImplementedError(
            "Vision grounding is reserved (interface only); the privileged "
            "branch is the one implemented and verified."
        )
