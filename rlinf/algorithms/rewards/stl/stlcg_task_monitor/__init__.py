"""
stlcg_task_monitor — self-contained NL -> STL(stlcg++) online robustness monitor
over recorded LIBERO trajectories (all 4 suites, 8 atomic predicates).

No dependency on the repo's safe_stl package. Only external STL engine is the
vendored stlcg-plus-plus (../stlcg-plus-plus). NL parsing, ontology, signal
extraction, formula wiring, and the online monitor all live in this package.

Quick start:
    python -m experiments.robot.libero.stlcg_task_monitor \
        --traj trajectories_gf/libero_spatial/task00_ep000.h5
    python -m experiments.robot.libero.stlcg_task_monitor \
        --traj_dir trajectories_gf/libero_goal
"""
