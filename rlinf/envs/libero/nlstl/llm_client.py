"""LLM client abstraction for the NL -> symbolic STL step.

The online client speaks the OpenAI-compatible Chat Completions API over HTTP
(``requests`` -- the ``openai`` package is not required), configured entirely
from environment variables so an API key / endpoint / model can be supplied
without code changes:

    OPENAI_API_KEY      required for the online client
    OPENAI_BASE_URL     default ``https://api.openline.com/v1``  (any OpenAI-
                        compatible gateway: OpenAI, Azure OpenAI, vLLM, etc.)
    RLINF_LLM_MODEL     default ``gpt-4o-mini``
    RLINF_LLM_TIMEOUT   default 60 (seconds)

An explicit **offline** client is provided for testing the parsing / validation
/ downstream pipeline WITHOUT a network call: it returns the audited role
formula for the 40 known tasks and raises for anything else.  Opt in with
``RLINF_LLM_CLIENT=offline`` or by passing ``OfflineLLMClient()``.  It is NOT an
automatic fallback -- the online client raises if ``OPENAI_API_KEY`` is missing,
so LLM failures always surface (per the design choice: no silent fallback).
"""

from __future__ import annotations

import json
import os
import re
import time
from abc import ABC, abstractmethod
from typing import Any, List, Mapping, Optional

# Default gateway (OpenAI-compatible). !!! SECRET: _DEFAULT_API_KEY is a private
# key -- do NOT commit or share it. Move it to the OPENAI_API_KEY env var (and
# gitignore this file) before sharing the repo. Env vars always override these.
_DEFAULT_BASE_URL = "https://yibuapi.com/v1"
_DEFAULT_API_KEY = "sk-0o7rabPTImw3BXx3OnUMIosm1eSOZydP3q80LzwKA8FpLnJ9"
_DEFAULT_MODEL = "gpt-4o-mini"


class LLMClient(ABC):
    """Minimal chat-completion interface used by the symbolic-formula generator."""

    @abstractmethod
    def complete(
        self,
        messages: List[Mapping[str, str]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.0,
        json_mode: bool = True,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> str:
        """Return the assistant message content for the given chat messages."""
        raise NotImplementedError


class OpenAIChatClient(LLMClient):
    """OpenAI-compatible Chat Completions client (HTTP, JSON mode, retry)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.api_key = (
            api_key or os.environ.get("OPENAI_API_KEY") or _DEFAULT_API_KEY
        )
        self.base_url = (
            base_url
            or os.environ.get("OPENAI_BASE_URL")
            or _DEFAULT_BASE_URL
        ).rstrip("/")
        self.model = (
            model or os.environ.get("RLINF_LLM_MODEL") or _DEFAULT_MODEL
        )
        self.timeout = float(timeout or os.environ.get("RLINF_LLM_TIMEOUT") or 60)
        if not self.api_key:
            raise RuntimeError(
                "OpenAIChatClient needs OPENAI_API_KEY (or pass api_key=). Set it, "
                "or use OfflineLLMClient / RLINF_LLM_CLIENT=offline for testing."
            )

    def complete(
        self,
        messages,
        *,
        model=None,
        temperature=0.0,
        json_mode=True,
        max_tokens=None,
        timeout=None,
        retries: int = 3,
    ) -> str:
        import requests  # local import: keep module import light

        url = f"{self.base_url}/chat/completions"
        payload: dict = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if max_tokens:
            payload["max_tokens"] = max_tokens
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_err: Optional[BaseException] = None
        for attempt in range(max(1, retries)):
            try:
                resp = requests.post(
                    url, headers=headers, json=payload, timeout=timeout or self.timeout
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except Exception as err:  # noqa: BLE001 - retry any transient failure
                last_err = err
                if attempt + 1 < retries:
                    time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(
            f"OpenAIChatClient failed after {retries} attempts: {last_err!r}"
        )


class OfflineLLMClient(LLMClient):
    """Test client: returns the audited role formula for known tasks.

    This lets the whole NL->symbolic pipeline (prompt building, JSON parsing,
    validation, conversion, downstream plumbing) be exercised without a network
    call or API key.  It is deliberately NOT a production fallback: for any
    instruction without an audited plan it raises, so tests never pass on
    fabricated output.
    """

    def complete(
        self,
        messages,
        *,
        model=None,
        temperature=0.0,
        json_mode=True,
        max_tokens=None,
        timeout=None,
    ) -> str:
        from rlinf.envs.libero.nlstl.symbolic_plan import (
            AUDITED_PLANS_BY_DESCRIPTION,
            _concrete_to_role,
            _normalize_description,
        )
        from rlinf.envs.libero.stl_stage_plan import canonical_atom

        desc = _extract_instruction(messages)
        norm = _normalize_description(desc)
        plan = AUDITED_PLANS_BY_DESCRIPTION.get(norm)
        if plan is None:
            raise RuntimeError(
                f"OfflineLLMClient has no audited formula for {desc!r}; "
                "use the online client for unseen instructions."
            )
        c2r = _concrete_to_role(plan)
        paths = []
        for path in plan.paths:
            stages = []
            for stage in path.stages:
                atoms = []
                for atom in stage.atoms:
                    pred, obj, target = canonical_atom(atom)
                    external_pred = {
                        "turnon": "turn_on",
                        "turnoff": "turn_off",
                    }.get(pred, pred)
                    payload = {"pred": external_pred}
                    if obj is not None:
                        payload["obj"] = c2r.get(obj, obj)
                    if target is not None:
                        payload["target"] = c2r.get(target, target)
                    atoms.append(payload)
                stages.append({"atoms": atoms})
            paths.append({"name": path.name, "stages": stages})
        return json.dumps({"paths": paths})


def get_default_client() -> LLMClient:
    """Pick the client from env: ``RLINF_LLM_CLIENT=offline`` -> offline, else online."""
    if os.environ.get("RLINF_LLM_CLIENT", "").lower() == "offline":
        return OfflineLLMClient()
    return OpenAIChatClient()


def _extract_instruction(messages) -> str:
    """Pull the target instruction out of the chat messages.

    The prompt places the instruction after an ``Instruction:`` marker on the
    final user turn; fall back to the whole last user message if absent.
    """
    last_user = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user = msg.get("content", "")
            break
    match = re.search(r"Instruction:\s*(.+)$", last_user, re.S)
    return (match.group(1).strip() if match else last_user.strip())
