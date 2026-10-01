"""Lazy opt-in registry override for the additive visual AGM environment."""

from __future__ import annotations

import builtins
import os
import sys
from types import ModuleType


_ORIGINAL_IMPORT = builtins.__import__


def _install_override(env_registry: ModuleType) -> None:
    if getattr(env_registry, "_visual_agm_override_installed", False):
        return
    original_get_env_cls = getattr(env_registry, "get_env_cls", None)
    if original_get_env_cls is None:
        return

    def get_env_cls(env_type, env_cfg=None):
        env_type_value = getattr(env_type, "value", env_type)
        if str(env_type_value) == "libero":
            from rlinf.envs.libero.libero_env_visual_agm_stage import (
                LiberoEnv,
            )

            return LiberoEnv
        return original_get_env_cls(env_type, env_cfg)

    env_registry.get_env_cls = get_env_cls
    env_registry._visual_agm_override_installed = True


def _lazy_import(name, globals=None, locals=None, fromlist=(), level=0):
    module = _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)
    if name == "rlinf.envs" or name.startswith("rlinf.envs."):
        env_registry = sys.modules.get("rlinf.envs")
        if env_registry is not None:
            _install_override(env_registry)
    return module


def _enable_lazy_override() -> None:
    if os.environ.get("RLINF_USE_VISUAL_AGM_STAGE_ENV") != "1":
        return
    if getattr(builtins.__import__, "_visual_agm_import_hook", False):
        return
    _lazy_import._visual_agm_import_hook = True
    builtins.__import__ = _lazy_import


_enable_lazy_override()
