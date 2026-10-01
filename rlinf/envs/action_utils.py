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

"""Action conversion retained for the LIBERO-only SpecRL bundle."""

import numpy as np
import torch

from rlinf.config import SupportedModel
from rlinf.envs import SupportedEnvType


def prepare_actions_for_libero(
    raw_chunk_actions,
    model_type,
) -> np.ndarray:
    """Convert policy actions to LIBERO's gripper convention when needed."""

    chunk_actions = raw_chunk_actions
    if SupportedModel(model_type) in [
        SupportedModel.OPENVLA,
        SupportedModel.OPENVLA_OFT,
        SupportedModel.GR00T_N1D6,
        SupportedModel.GR00T_N1D7,
    ]:
        chunk_actions[..., -1] = 2 * chunk_actions[..., -1] - 1
        chunk_actions[..., -1] = np.sign(chunk_actions[..., -1]) * -1.0
    return chunk_actions


def prepare_actions(
    raw_chunk_actions,
    env_type: str,
    model_type: str,
    num_action_chunks,
    action_dim,
    action_scale: float = 1.0,
    policy: str = "widowx_bridge",
    wm_env_type=None,
) -> torch.Tensor | np.ndarray:
    """Prepare actions for the LIBERO environment used by SpecRL_P/V.

    Extra arguments remain in the signature because workers call this shared
    interface. They are intentionally unused by the LIBERO conversion.
    """

    del num_action_chunks, action_dim, action_scale, policy, wm_env_type

    if isinstance(raw_chunk_actions, torch.Tensor):
        raw_chunk_actions = raw_chunk_actions.detach().cpu().contiguous()
        if raw_chunk_actions.dtype == torch.bfloat16:
            raw_chunk_actions = raw_chunk_actions.float()
        raw_chunk_actions = raw_chunk_actions.numpy()

    resolved_env_type = SupportedEnvType(env_type)
    if resolved_env_type != SupportedEnvType.LIBERO:
        raise NotImplementedError(
            f"Environment type {resolved_env_type.value!r} is not included in "
            "the SpecRL source bundle; only 'libero' is supported"
        )

    return prepare_actions_for_libero(
        raw_chunk_actions=raw_chunk_actions,
        model_type=model_type,
    )
