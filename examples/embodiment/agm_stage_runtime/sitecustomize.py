"""Lazy opt-in registry override for the experimental AGM-stage LIBERO env.

This module is automatically imported in every Python interpreter when its
directory is on PYTHONPATH.  It must stay lightweight: Ray prestarts many
Python workers, so importing rlinf.envs here would make all workers perform
heavy simulator imports during startup.

Instead, this module waits until rlinf.envs is imported normally, and only
then replaces get_env_cls for the "libero" environment type.
"""

from __future__ import annotations

import builtins
import os
import sys
from types import ModuleType


_ORIGINAL_IMPORT = builtins.__import__


def _install_override(env_registry: ModuleType) -> None:
    """Patch the already-imported rlinf.envs registry once."""

    if getattr(env_registry, "_agm_stage_override_installed", False):
        return

    original_get_env_cls = getattr(env_registry, "get_env_cls", None)
    if original_get_env_cls is None:
        # rlinf.envs may still be partway through its own import.
        # A later import call will retry installation.
        return

    def get_env_cls(env_type, env_cfg=None):
        env_type_value = getattr(env_type, "value", env_type)

        if str(env_type_value) == "libero":
            # Delayed until a LIBERO environment is actually requested.
            # SpecRL_P selects the privileged-state AGM-stage environment.
            from rlinf.envs.libero.libero_env_agm_stage import LiberoEnv

            return LiberoEnv

        return original_get_env_cls(env_type, env_cfg)

    env_registry.get_env_cls = get_env_cls
    env_registry._agm_stage_override_installed = True


def _lazy_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Keep Python startup light; patch only after rlinf.envs is available."""

    module = _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)

    if name == "rlinf.envs" or name.startswith("rlinf.envs."):
        env_registry = sys.modules.get("rlinf.envs")
        if env_registry is not None:
            _install_override(env_registry)

    return module


def _enable_lazy_override() -> None:
    agm_enabled = os.environ.get("RLINF_USE_AGM_STAGE_ENV") == "1"
    if not agm_enabled:
        return

    if getattr(builtins.__import__, "_agm_stage_import_hook", False):
        return

    _lazy_import._agm_stage_import_hook = True
    builtins.__import__ = _lazy_import


_enable_lazy_override()
