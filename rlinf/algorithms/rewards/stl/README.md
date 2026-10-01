# STL-robustness reward (`rlinf/algorithms/rewards/stl/`)

Goal: **use STL robustness degree as the RL reward** for embodied (LIBERO) tasks.

## Layout

- `stlcg_task_monitor/` — self-contained NL → STL online robustness monitor
  (ported from `openvla/experiments/robot/libero/stlcg_task_monitor`).
  - `nl_parser.py` / `ontology.py` / `bddl_goals.py` — task spec → predicate graph
  - `task_graph.py` — manipulated-objects / grasp-then-goal decomposition
  - `predicates.py` — 8 atomic LIBERO predicates + per-step margin signals
  - `formula_builder.py` — TaskGraph → `stlcgpp` STL formula + stacked signal tensor
  - `loader.py` — episode (h5) loader / signal extraction
  - `monitor.py` / `fast_online.py` — online + suffix robustness traces
  - `calibration.py`, `diagnose_errors.py`, `verify_predicates.py`, `plot.py`,
    `sample_plots.py`, `__main__.py` — offline diagnostic / calibration tooling
- STL engine: vendored at `third_party/stlcg-plus-plus/` (`stlcgpp`). The engine
  is an external dependency (Karen Leung, PyTorch port of STLCG, on PyPI as
  `stlcgpp`); it is vendored rather than pip-installed so the import in
  `formula_builder.py` works offline. You may alternatively `pip install -e
  third_party/stlcg-plus-plus`.

## Import wiring

`formula_builder.py` resolves `stlcgpp` by inserting the repo-relative path
`third_party/stlcg-plus-plus` onto `sys.path`. If you relocate either the
monitor or the engine, update that one `_STLCGPP` path constant.

## ROADMAP — turning this into an RL reward

The monitor currently consumes *recorded* episodes (`loader.load_episode`). To
serve as a live reward it must be adapted to the RLinf reward / env interface:

1. **Reward class.** Add `STLReward` (e.g. `stl/__init__.py` or a new
   `stl/stl_reward.py`) following the interface used by the existing rewards
   (`math`, `vqa`, ...). Register it via
   `register_reward("stl", STLReward)` in `rlinf/algorithms/rewards/__init__.py`.
2. **Live signal source.** Replace/branch the h5 loader so the signal tensor is
   built from in-flight env state (gripper pose, object poses, joint states)
   rather than a recorded trajectory. For LIBERO this hooks into
   `rlinf/envs/libero/`.
3. **Reward shape.** Decide the robustness→reward mapping: per-step margin,
   suffix-trace delta (`monitor.py`), or terminal robustness degree; align with
   `algorithm.adv_type` / reward normalization.
4. **Config.** Add YAML keys (formula / predicate thresholds / robustness
   transform) and any validation in `rlinf/config.py::validate_cfg`.
5. **Tests.** Unit tests on the formula builder + a small e2e config under
   `tests/e2e_tests/embodied/` once wired into a LIBERO run.

## ONLINE STL REWARD (implemented)

Step 2 + step 3 above are implemented as a **LIBERO env subclass** rather than a
reward-registry class, because the privileged MuJoCo state needed for the margins
only lives inside the env worker subprocesses:

- `rlinf/envs/libero/libero_env_stl_reward.py` — `LiberoSTLRewardEnv(LiberoEnv)`.
  Overrides only `reset` (rebuilds the per-env task formula + resets the tracker
  on every episode boundary, since auto-resets flow through `reset`) and
  `_calc_step_reward` (adds an STL potential-shaping term on top of the inherited
  sparse success bonus):
  ```
  r_t = reward_coef * terminations + stl_reward_scale * (stl_gamma * rho_t - rho_{t-1})
  ```
  where `rho_t = rho(phi, s[0:t+1])` is the **prefix STL robustness** of the
  task formula `phi = AND_k EVENTUALLY(grasp_k AND EVENTUALLY_[0,tau] goal_k)`,
  updated incrementally (`OnlineRobustnessTracker`, O(tau*K)/step, bit-exact vs
  `fast_online.online_prefix_true` — verified in
  `tests/test_stl_tracker_vs_fastonline.py`). Per-step predicate margins reuse
  `predicates.py` (pick/on/stack/in/open/close/turn_on/turn_off) on a length-1
  `Episode` built from the live privileged state.

- `rlinf/envs/libero/venv.py` — two new pipe commands, `get_privileged_state`
  (body/site/joint/contact/gripper arrays, read straight from the worker's
  `env.sim`, `obj_body_tree` memoized per task) and `get_goal_atoms` (the task's
  BDDL `goal_state`), dispatched per-env mirroring `set_init_state`.

- Registration: env type `libero_stl` in `rlinf/envs/__init__.py`
  (`SupportedEnvType.LIBERO_STL`).

- Example config: `examples/embodiment/config/libero_spatial_ppo_openvlaoft_stl.yaml`
  (env knobs `stl_reward`, `stl_tau`, `stl_reward_scale`, `stl_gamma`,
  `stl_clip`, `stl_pred_config`). Run with
  `bash examples/embodiment/run_embodiment.sh libero_spatial_ppo_openvlaoft_stl`.

**v1 performance note.** Each step fetches privileged state from every env worker
sequentially (one pipe RPC per env per sub-step). Correctness-first; batched /
in-worker reward computation is a future optimization if this becomes a
bottleneck. All privileged reads are best-effort — on any failure the affected
env falls back to the sparse-only reward for that step, so a faulty read never
breaks a rollout.

**Known issue in the vendored monitor.** `stlcg_task_monitor/fast_online.py`'s
`online_prefix_true` wrapper passes the string `"true"` as the `methods`
argument, which makes `online_prefix` iterate over characters. Use
`online_prefix(... )["true"]` (default tuple) instead. The online reward here
uses its own incremental tracker (validated bit-exact), so it is unaffected.

