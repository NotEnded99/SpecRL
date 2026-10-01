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

from enum import Enum


class SupportedEnvType(Enum):
    MANISKILL = "maniskill"
    LIBERO = "libero"
    LIBERO_STL = "libero_stl"
    ROBOTWIN = "robotwin"
    ISAACLAB = "isaaclab"
    METAWORLD = "metaworld"
    BEHAVIOR = "behavior"
    CALVIN = "calvin"
    ROBOCASA = "robocasa"
    REALWORLD = "realworld"
    FRANKASIM = "frankasim"
    HABITAT = "habitat"
    OPENSORAWM = "opensora_wm"
    WANWM = "wan_wm"
    GENESIS = "genesis"
    EMBODICHAIN = "embodichain"
    ROBOVERSE = "roboverse"
    D4RL = "d4rl"
    POLARIS = "polaris"


def get_env_cls(env_type: str, env_cfg=None):
    """Return the LIBERO environment class included in the SpecRL bundle.

    Upstream RLinf supports many environments. This source bundle intentionally
    retains only the LIBERO runtime required by SpecRL_P and SpecRL_V.
    """

    env_type = SupportedEnvType(env_type)
    if env_type != SupportedEnvType.LIBERO:
        raise NotImplementedError(
            f"Environment type {env_type.value!r} is not included in the "
            "SpecRL source bundle; only 'libero' is supported"
        )

    from rlinf.envs.libero.libero_env import LiberoEnv

    return LiberoEnv
