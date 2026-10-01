"""Layer 1 (LLM variant): NL -> observation-agnostic symbolic STL via a GPT.

This replaces the audited-table lookup in :mod:`symbolic_plan` with an LLM call,
so symbolic formulas can be generated for *new* tasks without hand-authoring.
It produces the SAME :class:`SymbolicStagePlan` type, so grounding / predicate /
runtime layers are unchanged.

Design points
-------------
* The predicate vocabulary is CLOSED (``pick/on/in/open/close/turn_on/
  turn_off``). External names are mapped to the backend's canonical names. The
  prompt constrains the model to this set and the validator rejects anything
  else.
* Roles are semantic snake_case categories (``bowl``, ``bottom_drawer``), never
  simulator instance names or coordinates -- the formula stays observation-
  agnostic; binding to concrete evidence is the grounder's job (Layer 2).
* Each atom is its own sequential stage; independent multi-object subgoals yield
  two alternative paths (the two legal object orderings).
* Output is strict JSON; on parse/validation failure the error is fed back and
  the call retried a bounded number of times.  There is NO silent fallback to
  the audited table -- failures raise (by design).
"""

from __future__ import annotations

import json
import re
from typing import Any, List, Mapping, Optional, Tuple

from rlinf.envs.libero.nlstl.llm_client import LLMClient, get_default_client
from rlinf.envs.libero.nlstl.symbolic_plan import (
    AtomSpatial,
    SymbolicStage,
    SymbolicStagePlan,
    SymbolicPath,
)
from rlinf.envs.libero.stl_stage_plan import Atom, canonical_atom

# Closed predicate vocabulary + arity. The shared backend (goal_atom_margin /
# pred_pick_agm) only implements these.
PREDICATES: Mapping[str, Mapping[str, bool]] = {
    "pick": {"obj": True, "target": False},
    "on": {"obj": True, "target": True},
    "in": {"obj": True, "target": True},
    "open": {"obj": False, "target": True},
    "close": {"obj": False, "target": True},
    "turn_on": {"obj": False, "target": True},
    "turn_off": {"obj": False, "target": True},
}

_RUNTIME_PREDICATE_ALIASES = {
    "turn_on": "turnon",
    "turn_off": "turnoff",
}
_ROLE_NAME_RE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")

_SYSTEM_PROMPT = """You convert a natural-language robot-manipulation instruction into
an observation-agnostic symbolic Signal Temporal Logic (STL)
specification.

OUTPUT FORMAT

Return STRICT JSON only. Do not include explanations, comments, or
Markdown code fences. The output must follow this structure:

{
  "paths": [
    {
      "name": "<short_name>",
      "stages": [
        {
          "atoms": [
            {
              "pred": "...",
              "obj": "...",
              "target": "..."
            }
          ]
        }
      ]
    }
  ]
}

ATOMIC PREDICATES

The predicate vocabulary is CLOSED. Use only the following predicates:

1. pick(obj)

Meaning: Grasp object obj.

Arguments: obj is required; target must be omitted.

2. on(obj, target)

Meaning: Place obj on the surface of target.

Arguments: Both obj and target are required.

3. in(obj, target)

Meaning: Place obj inside the container target.

Arguments: Both obj and target are required.

4. open(target)

Meaning: Open an articulated fixture, such as a drawer or door.

Arguments: target is required; obj must be omitted.

5. close(target)

Meaning: Close an articulated fixture.

Arguments: target is required; obj must be omitted.

6. turn_on(target)

Meaning: Turn on a fixture, such as a stove.

Arguments: target is required; obj must be omitted.

7. turn_off(target)

Meaning: Turn off a fixture.

Arguments: target is required; obj must be omitted.

GENERATION RULES

1. Semantic role names

The values of obj and target must be semantic role names in singular
snake_case form.

Preserve every qualifying adjective in the instruction, especially
color and type descriptors, so that each name faithfully identifies
the referenced entity.

Examples:
- "the black bowl" -> "black_bowl"
- "the yellow mug" -> "yellow_mug"
- "the white mug" -> "white_mug"
- "alphabet soup" -> "alphabet_soup"

Use the bare category only when the instruction provides no
qualifier:
- "the bowl" -> "bowl"
- "the stove" -> "stove"

Never use simulator instance names, object IDs, coordinates, or
observation-dependent identifiers. The same entity must retain the
same semantic role name everywhere in the output.

2. Sequential stages

Each atomic predicate must form its own sequential stage. Therefore,
every stages[i].atoms array contains exactly one atom.

List all stages in their required execution order. An object must
always be picked before applying on and in.

3. Preconditions

Place required preconditions before the operations that depend on
them.

Examples:
- Turn on the stove before placing a pot on it.
- Pick and place an object before closing the fixture containing it.
- Open a closed container before placing an object inside it.

4. Independent subgoals

If the instruction contains two independent object-manipulation
subgoals, emit two valid paths representing both execution orders:
- one path executing X before Y;
- one path executing Y before X.

Otherwise, emit exactly one path with the name "only".

5. Inapplicable fields

Omit fields that do not apply to a predicate:
- pick has no target;
- open, close, turn_on, and turn_off have no obj.

Do not include inapplicable fields with empty strings.

6. Spatial disambiguation

An atom may contain an optional spatial object when the instruction
uses a positional phrase to disambiguate a manipulated object.

Examples of such phrases include:
- "the black bowl between the plate and the ramekin";
- "the black bowl next to the cookie box";
- "the black bowl on the stove";
- "the black bowl in the top drawer".

Attach spatial only to the atom whose entity it disambiguates,
usually the corresponding pick atom:

{
  "pred": "pick",
  "obj": "black_bowl",
  "spatial": {
    "relation": "between",
    "refs": ["plate", "ramekin"]
  }
}

The value of relation must be a snake_case spatial relation, such as:
- "between"
- "next_to"
- "on_the"
- "in_the"
- "left_of"
- "right_of"
- "middle"

The entries in refs must be semantic role names for the reference
landmarks. Do not treat the reference landmarks as manipulation
targets unless the instruction explicitly requires manipulating them.

Include spatial only when an explicit positional phrase is needed for
entity disambiguation. Otherwise, omit it entirely.

FINAL VALIDATION

Before returning the JSON, verify that:
1. Every predicate belongs to the closed vocabulary.
2. Every stage contains exactly one atom.
3. All required arguments are present.
4. All inapplicable arguments are omitted.
5. Semantic role names are singular and use snake_case.
6. Entity names preserve all qualifiers from the instruction.
7. The stages follow a valid execution order.
8. The output contains no text outside the JSON object.
"""


def build_messages(description: str) -> List[Mapping[str, str]]:
    """Assemble exactly the supplied system prompt and target instruction."""
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Instruction: {description.strip()}"},
    ]


def parse_formula(text: str) -> dict:
    """Extract and json-parse the formula object from the model output."""
    cleaned = text.strip()
    # strip ```json ... ``` fences if present
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, re.S)
    if fenced:
        cleaned = fenced.group(1)
    else:
        # fall back to the outermost {...} block
        match = re.search(r"\{.*\}", cleaned, re.S)
        if match:
            cleaned = match.group(0)
    try:
        obj = json.loads(cleaned)
    except Exception as err:  # noqa: BLE001
        raise ValueError(f"LLM output is not valid JSON: {err}; raw={text[:200]!r}")
    if not isinstance(obj, dict):
        raise ValueError(f"Expected a JSON object, got {type(obj).__name__}.")
    return obj


def validate_formula(obj: dict) -> None:
    """Validate the closed-vocabulary predicate structure. Raises on any issue."""
    paths = obj.get("paths")
    if not isinstance(paths, list) or not paths:
        raise ValueError("Missing non-empty 'paths' list.")
    for pi, path in enumerate(paths):
        if not isinstance(path, dict):
            raise ValueError(f"path[{pi}] is not an object.")
        stages = path.get("stages")
        if not isinstance(stages, list) or not stages:
            raise ValueError(f"path[{pi}] missing non-empty 'stages'.")
        for si, stage in enumerate(stages):
            atoms = stage.get("atoms") if isinstance(stage, dict) else None
            if not isinstance(atoms, list) or len(atoms) != 1:
                raise ValueError(
                    f"path[{pi}].stages[{si}] must contain exactly one atom."
                )
            for ai, atom in enumerate(atoms):
                _validate_atom(atom, where=f"path[{pi}].stages[{si}].atoms[{ai}]")


def _validate_atom(atom: Any, where: str) -> None:
    if not isinstance(atom, dict):
        raise ValueError(f"{where} is not an object.")
    pred = atom.get("pred")
    if pred not in PREDICATES:
        raise ValueError(
            f"{where}: unknown predicate {pred!r}; must be one of {sorted(PREDICATES)}."
        )
    arity = PREDICATES[pred]
    obj = atom.get("obj")
    target = atom.get("target")
    if arity["obj"] and not (isinstance(obj, str) and obj.strip()):
        raise ValueError(f"{where}: predicate {pred!r} requires a non-empty 'obj'.")
    if not arity["obj"] and "obj" in atom:
        raise ValueError(f"{where}: predicate {pred!r} must omit 'obj'.")
    if arity["target"] and not (isinstance(target, str) and target.strip()):
        raise ValueError(f"{where}: predicate {pred!r} requires a non-empty 'target'.")
    if not arity["target"] and "target" in atom:
        raise ValueError(f"{where}: predicate {pred!r} must omit 'target'.")
    for field_name, value in (("obj", obj), ("target", target)):
        if value is not None and not _ROLE_NAME_RE.fullmatch(value):
            raise ValueError(
                f"{where}: {field_name!r} must be singular snake_case, got {value!r}."
            )
    spatial = atom.get("spatial")
    if spatial is not None:
        _validate_spatial(spatial, where=f"{where}.spatial")


def _validate_spatial(spatial: Any, where: str) -> None:
    """Validate an optional atom-level spatial annotation."""
    if not isinstance(spatial, dict):
        raise ValueError(f"{where} must be an object.")
    relation = spatial.get("relation")
    if not (isinstance(relation, str) and _ROLE_NAME_RE.fullmatch(relation)):
        raise ValueError(f"{where}: 'relation' must be snake_case.")
    refs = spatial.get("refs")
    if (not isinstance(refs, list)) or not refs:
        raise ValueError(f"{where}: 'refs' must be a non-empty list.")
    for ri, ref in enumerate(refs):
        if not (isinstance(ref, str) and _ROLE_NAME_RE.fullmatch(ref)):
            raise ValueError(f"{where}: refs[{ri}] must be snake_case.")


def formula_to_symbolic_plan(obj: dict, description: str) -> SymbolicStagePlan:
    """Convert a validated formula dict into a role-based SymbolicStagePlan."""
    paths = []
    atom_spatial: List[Tuple[Atom, AtomSpatial]] = []
    seen_spatial: set = set()
    for path in obj["paths"]:
        stages = []
        for stage in path["stages"]:
            atoms: Tuple[Atom, ...] = tuple(
                canonical_atom((
                    _RUNTIME_PREDICATE_ALIASES.get(a["pred"], a["pred"]),
                    a.get("obj"),
                    a.get("target"),
                ))
                for a in stage["atoms"]
            )
            stages.append(SymbolicStage(name=stage.get("name", ""), atoms=atoms))
            # Collect optional atom-level spatial annotations (dedup, order-preserving).
            for raw_atom, role_atom in zip(stage["atoms"], atoms):
                sp = raw_atom.get("spatial")
                if sp is None or role_atom in seen_spatial:
                    continue
                seen_spatial.add(role_atom)
                atom_spatial.append(
                    (role_atom, AtomSpatial(
                        relation=sp["relation"], refs=tuple(sp["refs"]),
                    ))
                )
        paths.append(
            SymbolicPath(name=path.get("name", "only"), stages=tuple(stages))
        )
    has_alt = len(paths) > 1
    return SymbolicStagePlan(
        description=description,
        gated=True,
        audited=False,  # LLM-generated, not from the audited table
        reason="llm-generated symbolic STL",
        has_alternatives=has_alt,
        paths=tuple(paths),
        concrete_plan=None,  # ungrounded; binding happens in Layer 2
        role_to_concrete=(),
        atom_spatial=tuple(atom_spatial),
    )


def build_symbolic_plan_llm(
    description: str,
    client: Optional[LLMClient] = None,
    *,
    temperature: float = 0.0,
    max_retries: int = 2,
    max_tokens: Optional[int] = None,
    return_raw: bool = False,
):
    """Generate a symbolic STL plan for ``description`` with a GPT.

    No silent fallback: on invalid output after ``max_retries`` corrections, or
    if the API is unreachable / unconfigured, this raises.

    With ``return_raw=True`` the return value is
    ``(SymbolicStagePlan, raw_formula_dict)`` where ``raw_formula_dict`` is the
    validated LLM output VERBATIM (``{"paths": [...]}`` with structured
    pred/obj/target atoms, path names, stage boundaries and spatial
    annotations).  Callers that archive the LLM output for later human
    revision should store this dict, not a flattened string summary: the raw
    structure round-trips losslessly while a flattened rendering does not.
    """
    client = client or get_default_client()
    messages = build_messages(description)
    last_err: Optional[BaseException] = None

    for attempt in range(max_retries + 1):
        try:
            text = client.complete(
                messages, temperature=temperature, max_tokens=max_tokens
            )
        except Exception as err:  # noqa: BLE001 - transport failure is terminal here
            raise RuntimeError(
                f"LLM call failed for {description!r}: {err!r}"
            ) from err

        try:
            obj = parse_formula(text)
            validate_formula(obj)
            plan = formula_to_symbolic_plan(obj, description)
            return (plan, obj) if return_raw else plan
        except ValueError as err:
            last_err = err
            # Feed the failure back to the model and retry once more.
            messages = messages + [
                {"role": "assistant", "content": text},
                {
                    "role": "user",
                    "content": (
                        f"That was invalid: {err}. Output ONLY valid JSON "
                        "matching the schema and the closed predicate vocabulary."
                    ),
                },
            ]

    raise RuntimeError(
        f"build_symbolic_plan_llm failed for {description!r} after "
        f"{max_retries + 1} attempts: {last_err!r}"
    )
