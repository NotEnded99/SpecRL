# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Reward registry.

LAZY: reward families (math / vqa / code / searchr1 / rstar2) are NOT imported
when this package is imported. Each family pulls in heavy optional deps
(latex2sympy2, tree-sitter, tensorflow, ...); importing them eagerly here meant
that merely importing a sibling subpackage (e.g. ``stl.stlcg_task_monitor``,
used by the LIBERO STL reward env) triggered those imports as a side effect and
broke downstream model inference. Families are now imported on first lookup in
:meth:`get_rule_based_reward_class`. Manual ``register_reward(name, cls)`` calls
still work and take precedence.
"""

from rlinf.utils.logging import get_logger

_logger = get_logger()

# name -> (import_path, attr). Imported lazily on first get_rule_based_reward_class.
_REWARD_SOURCES = {
    "math": ("rlinf.algorithms.rewards.math", "MathReward"),
    "vqa": ("rlinf.algorithms.rewards.vqa", "VQAReward"),
    "code_offline": ("rlinf.algorithms.rewards.code", "CodeRewardOffline"),
    "searchr1": ("rlinf.algorithms.rewards.searchr1", "SearchR1Reward"),
    "rstar2": ("rlinf.algorithms.rewards.rstar2", "Rstar2Reward"),
}

reward_registry: dict = {}


def register_reward(name: str, reward_class: type):
    assert name not in reward_registry, f"Reward {name} already registered"
    reward_registry[name] = reward_class


def get_rule_based_reward_class(name: str):
    if name not in reward_registry and name in _REWARD_SOURCES:
        import_path, attr = _REWARD_SOURCES[name]
        try:
            module = __import__(import_path, fromlist=[attr])
            register_reward(name, getattr(module, attr))
        except ImportError as e:
            _logger.debug(
                "Reward family '%s' not registered (optional import %s failed: %s)",
                name, import_path, e,
            )
    assert name in reward_registry, f"Reward {name} not found"
    return reward_registry[name]
