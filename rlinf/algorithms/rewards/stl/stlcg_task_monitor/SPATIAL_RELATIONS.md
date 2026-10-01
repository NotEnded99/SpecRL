# LIBERO Spatial Relation Resolution

## Overview

When multiple instances of the same object type exist (e.g., 3 black bowls), natural language descriptions often include spatial constraints to disambiguate which instance to use. This document describes the supported spatial relations and how they are resolved.

## Supported Relations (from LIBERO benchmarks)

### libero_spatial (10 tasks)

| Relation | Example Task | Description |
|----------|--------------|-------------|
| `between` | "pick up the black bowl **between the plate and the ramekin**" | Object is spatially between two landmarks |
| `next_to` | "pick up the black bowl **next to the cookie box**" | Object is near (beside) a landmark |
| `on_the` | "pick up the black bowl **on the stove**" | Object is on top of a surface/container |
| `in_the` | "pick up the black bowl **in the top drawer**" | Object is inside a container |

### libero_90 (90 tasks)

| Relation | Example Task | Description |
|----------|--------------|-------------|
| `left` | "the **left** bowl", "to the **left** of the plate" | Leftmost object or object to the left of a landmark |
| `right` | "the **right** bowl", "to the **right** of the plate" | Rightmost object or object to the right of a landmark |
| `front` | "at the **front** on the plate" | Frontmost object (+x direction) |
| `back` | "at the **back** on the plate" | Backmost object (-x direction) |
| `middle` | "the **middle** black bowl" | Object with median position among instances |
| `under` | "the book **under** the cabinet shelf" | Object below a landmark |

## Resolution Algorithm

### 1. Position Extraction
From the initial privileged state (step 0):
```python
object_positions = {obj_key: (x, y, z)}  # manipulable objects
body_positions = {body_name: (x, y, z)}  # fixtures (stove, cabinet, etc.)
```

### 2. Constraint Scoring
Each candidate object is scored based on how well it satisfies the spatial constraint:

- **between(A, B)**: `score = -perpendicular_distance - 0.5 * distance_to_midpoint`
  - Object should be on the line between A and B
  - Object should be close to the midpoint

- **left_of(ref)**: `score = candidate.y - ref.y`
  - Positive if candidate is to the left (+y direction in robot frame)

- **right_of(ref)**: `score = ref.y - candidate.y`
  - Positive if candidate is to the right (-y direction)

- **front**: `score = 1.0 if max_x else 0.0`
  - Object with maximum x-coordinate

- **back**: `score = 1.0 if min_x else 0.0`
  - Object with minimum x-coordinate

- **middle**: `score = -abs(candidate.y - median_y)`
  - Object closest to median y-coordinate

- **under(ref)**: `score = ref.z - candidate.z - xy_distance`
  - Must be below ref in z
  - Should be close in xy-plane

- **on_the(ref)**: `score = -xy_distance + z_score`
  - Should be close in xy
  - Should be slightly above in z (5cm optimal)

- **in_the(ref)**: `score = -xy_distance - abs(z_diff)`
  - Should be very close in xy (inside container)
  - Should be at similar height (inside volume)

### 3. Selection
The candidate with the highest score is selected as the resolved instance.

## Coordinate Frame

LIBERO uses a robot-centric coordinate frame:
- **+x**: front (away from robot)
- **-x**: back (toward robot)
- **+y**: left
- **-y**: right
- **+z**: up
- **-z**: down

## Usage Example

```python
from rlinf.algorithms.rewards.stl.stlcg_task_monitor.spatial_resolver import (
    SpatialConstraint,
    resolve_object_key_with_spatial,
)

# In libero_env._stl_setup or formula_builder.resolve_steps:

# 1. Parse NL task description
graph = parse_task("pick up the black bowl between the plate and the ramekin")
constraint = extract_spatial_constraint_from_taskgraph(graph)
# -> SpatialConstraint(relation="between", landmarks=["plate", "ramekin"])

# 2. Resolve using initial state
ep = _build_step_episode(priv)  # privileged state at step 0
resolved_bowl = resolve_object_key_with_spatial("bowl", ep, constraint)
# -> "akita_black_bowl_1" (the one between plate and ramekin)

# 3. Use resolved instance for STL margin computation
gm, _name = goal_atom_margin(ep, "on", resolved_bowl, "plate", cfg)
```

## Statistics from LIBERO Benchmarks

| Benchmark | Total Tasks | Spatial Tasks |
|-----------|-------------|---------------|
| libero_spatial | 10 | 10 (100%) |
| libero_90 | 90 | ~40 (44%) |
| libero_goal | 10 | 0 (0%) |
| libero_object | 10 | 0 (0%) |

Spatial relation frequency in libero_90:
- `top`: 22 occurrences (drawer levels, on top of)
- `right`: 15 occurrences
- `left`: 10 occurrences
- `front`: 7 occurrences
- `middle`: 5 occurrences
- `bottom`: 5 occurrences
- `back`: 4 occurrences
- `next_to`: 3 occurrences
- `under`: 2 occurrences
- `between`: 1 occurrence
