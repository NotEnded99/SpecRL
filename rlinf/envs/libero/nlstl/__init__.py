"""Shared natural-language -> STL reward framework for LIBERO.

Pipeline (see the package docstrings for each layer)::

    NL instruction
        |  build_symbolic_plan            (Layer 1, shared)
        v
    symbolic STL formula (semantic roles)
        |  Grounder.ground                (Layer 2, privileged | vision)
        v
    grounded atoms  +  UnifiedEvidence    (Layer 3, shared state format)
        |  compute_atom_margins           (Layer 4, shared predicates)
        v
    per-atom robustness margins
        |  AGMStageRuntime.update         (Layer 5, shared STL/AGM aggregation)
        v
    task robustness rho + shaping         (Layer 6, -> env reward)

Only Layer 2 (grounding) and the state-extraction half of Layer 3 differ between
the privileged and vision branches; everything else is shared and reused from the
existing LIBERO STL code.
"""

from rlinf.envs.libero.nlstl.evidence import (
    UnifiedEvidence,
    compute_atom_margins,
    evidence_from_priv,
    evidence_from_vision,
    make_pred_config,
)
from rlinf.envs.libero.nlstl.grounding import (
    Grounder,
    Grounding,
    PrivilegedGrounder,
    VisionGrounder,
)
from rlinf.envs.libero.nlstl.llm_client import (
    LLMClient,
    OfflineLLMClient,
    OpenAIChatClient,
    get_default_client,
)
from rlinf.envs.libero.nlstl.llm_symbolic import build_symbolic_plan_llm
from rlinf.envs.libero.nlstl.pipeline import NLSTLPipeline
from rlinf.envs.libero.nlstl.symbolic_plan import (
    SymbolicPath,
    SymbolicStage,
    SymbolicStagePlan,
    build_symbolic_plan,
)

__all__ = [
    # Layer 1: symbolic plan (audited table OR LLM)
    "build_symbolic_plan",
    "build_symbolic_plan_llm",
    "SymbolicStagePlan",
    "SymbolicPath",
    "SymbolicStage",
    # Layer 1 LLM client
    "LLMClient",
    "OpenAIChatClient",
    "OfflineLLMClient",
    "get_default_client",
    # Layer 2: grounding (privileged implemented, vision reserved)
    "Grounder",
    "Grounding",
    "PrivilegedGrounder",
    "VisionGrounder",
    # Layer 3: unified evidence (shared format + state extraction)
    "UnifiedEvidence",
    "evidence_from_priv",
    "evidence_from_vision",
    "make_pred_config",
    "compute_atom_margins",
    # Layers 4-6: orchestrator (shared)
    "NLSTLPipeline",
]
