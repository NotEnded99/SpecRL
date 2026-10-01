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
"""STL-robustness based reward for embodied RL.

This package currently hosts the ``stlcg_task_monitor`` sub-package, a
self-contained NL -> STL online robustness monitor over LIBERO trajectories
built on top of the vendored ``stlcg++`` engine
(``third_party/stlcg-plus-plus``). It is the foundation for an upcoming
``STLReward`` class that uses STL robustness degree as the RL reward signal.

The reward class is intentionally NOT registered yet (see ROADMAP below); the
monitor modules are kept import-light here so importing ``rlinf.algorithms``
does not pull in torch / h5py.
"""
