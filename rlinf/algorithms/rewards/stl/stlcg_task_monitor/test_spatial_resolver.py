"""
Comprehensive example demonstrating spatial relation resolution for LIBERO tasks.

This shows how to handle all spatial relations found in LIBERO benchmarks:
  - libero_spatial: between, next_to, on_the, in_the
  - libero_90: left, right, front, back, middle, under
"""

from __future__ import annotations

import numpy as np

# These would be imported in actual use:
# from rlinf.algorithms.rewards.stl.stlcg_task_monitor.spatial_resolver import (
#     SpatialConstraint,
#     resolve_object_by_spatial_constraint,
# )


def demo_all_spatial_relations():
    """
    Demonstrate resolution for all LIBERO spatial relation types.
    """
    print("=" * 80)
    print("LIBERO Spatial Relation Resolution Examples")
    print("=" * 80)

    # ========================================================================
    # 1. BETWEEN (libero_spatial)
    # ========================================================================
    print("\n1. BETWEEN: 'the black bowl between the plate and the ramekin'")
    print("-" * 60)

    object_positions = {
        "akita_black_bowl_1": np.array([0.30, 0.00, 0.45]),  # CENTER (between)
        "akita_black_bowl_2": np.array([0.50, 0.20, 0.45]),  # far right
        "akita_black_bowl_3": np.array([0.10, -0.15, 0.45]),  # far left
        "plate_1": np.array([0.20, 0.00, 0.42]),  # left landmark
        "ramekin_1": np.array([0.40, 0.00, 0.42]),  # right landmark
    }

    constraint = SpatialConstraint(relation="between", landmarks=["plate", "ramekin"])
    result = resolve_object_by_spatial_constraint("bowl", constraint, object_positions)
    print(f"  Resolved: {result}")
    print(f"  Position: {object_positions[result]}")
    print(f"  Expected: akita_black_bowl_1 (at [0.30, 0.00, 0.45])")

    # ========================================================================
    # 2. NEXT TO (libero_spatial)
    # ========================================================================
    print("\n2. NEXT_TO: 'the black bowl next to the cookie box'")
    print("-" * 60)

    object_positions = {
        "akita_black_bowl_1": np.array([0.25, 0.10, 0.45]),  # near cookie box
        "akita_black_bowl_2": np.array([0.50, 0.30, 0.45]),  # far
        "akita_black_bowl_3": np.array([0.10, -0.20, 0.45]),  # far
        "cookie_box_1": np.array([0.30, 0.05, 0.42]),
    }

    constraint = SpatialConstraint(relation="next_to", landmarks=["cookie_box"])
    result = resolve_object_by_spatial_constraint("bowl", constraint, object_positions)
    print(f"  Resolved: {result}")
    print(f"  Position: {object_positions[result]}")
    print(f"  Expected: akita_black_bowl_1 (closest to cookie box)")

    # ========================================================================
    # 3. ON THE (libero_spatial)
    # ========================================================================
    print("\n3. ON_THE: 'the black bowl on the stove'")
    print("-" * 60)

    object_positions = {
        "akita_black_bowl_1": np.array([0.30, 0.00, 0.50]),  # ON stove (slightly above)
        "akita_black_bowl_2": np.array([0.50, 0.30, 0.45]),  # on table
        "akita_black_bowl_3": np.array([0.10, -0.20, 0.45]),  # on table
    }
    body_positions = {
        "flat_stove_1": np.array([0.30, 0.00, 0.42]),
    }

    constraint = SpatialConstraint(relation="on_the", landmarks=["stove"])
    result = resolve_object_by_spatial_constraint("bowl", constraint, object_positions, body_positions)
    print(f"  Resolved: {result}")
    print(f"  Position: {object_positions[result]}")
    print(f"  Expected: akita_black_bowl_1 (on the stove)")

    # ========================================================================
    # 4. IN THE (libero_spatial)
    # ========================================================================
    print("\n4. IN_THE: 'the black bowl in the top drawer'")
    print("-" * 60)

    object_positions = {
        "akita_black_bowl_1": np.array([0.30, 0.00, 0.35]),  # IN drawer
        "akita_black_bowl_2": np.array([0.50, 0.30, 0.45]),  # on table
        "akita_black_bowl_3": np.array([0.10, -0.20, 0.45]),  # on table
    }
    body_positions = {
        "wooden_cabinet_1": np.array([0.30, 0.00, 0.30]),
    }

    constraint = SpatialConstraint(relation="in_the", landmarks=["cabinet"])
    result = resolve_object_by_spatial_constraint("bowl", constraint, object_positions, body_positions)
    print(f"  Resolved: {result}")
    print(f"  Position: {object_positions[result]}")
    print(f"  Expected: akita_black_bowl_1 (in the drawer)")

    # ========================================================================
    # 5. LEFT (libero_90)
    # ========================================================================
    print("\n5. LEFT: 'the left bowl' or 'to the left of the plate'")
    print("-" * 60)

    object_positions = {
        "akita_black_bowl_1": np.array([0.30, 0.20, 0.45]),  # RIGHT (+y)
        "akita_black_bowl_2": np.array([0.30, 0.00, 0.45]),  # MIDDLE
        "akita_black_bowl_3": np.array([0.30, -0.20, 0.45]),  # LEFT (-y)
    }

    # Case 1: "the left bowl" (no landmark)
    constraint = SpatialConstraint(relation="left", landmarks=[])
    result = resolve_object_by_spatial_constraint("bowl", constraint, object_positions)
    print(f"  'the left bowl' -> {result}")
    print(f"  Position: {object_positions[result]}")
    print(f"  Expected: akita_black_bowl_3 (leftmost = min y)")

    # Case 2: "to the left of the plate"
    body_positions = {"plate_1": np.array([0.30, 0.00, 0.42])}
    constraint = SpatialConstraint(relation="left_of", landmarks=["plate"])
    result = resolve_object_by_spatial_constraint("bowl", constraint, object_positions, body_positions)
    print(f"  'to the left of the plate' -> {result}")
    print(f"  Expected: akita_black_bowl_1 (y > plate.y)")

    # ========================================================================
    # 6. RIGHT (libero_90)
    # ========================================================================
    print("\n6. RIGHT: 'the right bowl' or 'to the right of the plate'")
    print("-" * 60)

    # Case 1: "the right bowl" (no landmark)
    constraint = SpatialConstraint(relation="right", landmarks=[])
    result = resolve_object_by_spatial_constraint("bowl", constraint, object_positions)
    print(f"  'the right bowl' -> {result}")
    print(f"  Position: {object_positions[result]}")
    print(f"  Expected: akita_black_bowl_1 (rightmost = max y)")

    # Case 2: "to the right of the plate"
    constraint = SpatialConstraint(relation="right_of", landmarks=["plate"])
    result = resolve_object_by_spatial_constraint("bowl", constraint, object_positions, body_positions)
    print(f"  'to the right of the plate' -> {result}")
    print(f"  Expected: akita_black_bowl_3 (y < plate.y)")

    # ========================================================================
    # 7. FRONT (libero_90)
    # ========================================================================
    print("\n7. FRONT: 'the front bowl' or 'at the front on the plate'")
    print("-" * 60)

    object_positions = {
        "akita_black_bowl_1": np.array([0.40, 0.00, 0.45]),  # FRONT (+x)
        "akita_black_bowl_2": np.array([0.30, 0.00, 0.45]),  # MIDDLE
        "akita_black_bowl_3": np.array([0.20, 0.00, 0.45]),  # BACK (-x)
    }

    constraint = SpatialConstraint(relation="front", landmarks=[])
    result = resolve_object_by_spatial_constraint("bowl", constraint, object_positions)
    print(f"  'the front bowl' -> {result}")
    print(f"  Position: {object_positions[result]}")
    print(f"  Expected: akita_black_bowl_1 (frontmost = max x)")

    # ========================================================================
    # 8. BACK (libero_90)
    # ========================================================================
    print("\n8. BACK: 'the back bowl' or 'at the back on the plate'")
    print("-" * 60)

    constraint = SpatialConstraint(relation="back", landmarks=[])
    result = resolve_object_by_spatial_constraint("bowl", constraint, object_positions)
    print(f"  'the back bowl' -> {result}")
    print(f"  Position: {object_positions[result]}")
    print(f"  Expected: akita_black_bowl_3 (backmost = min x)")

    # ========================================================================
    # 9. MIDDLE (libero_90)
    # ========================================================================
    print("\n9. MIDDLE: 'the middle black bowl'")
    print("-" * 60)

    object_positions = {
        "akita_black_bowl_1": np.array([0.30, 0.20, 0.45]),  # LEFT
        "akita_black_bowl_2": np.array([0.30, 0.00, 0.45]),  # MIDDLE
        "akita_black_bowl_3": np.array([0.30, -0.20, 0.45]),  # RIGHT
    }

    constraint = SpatialConstraint(relation="middle", landmarks=[])
    result = resolve_object_by_spatial_constraint("bowl", constraint, object_positions)
    print(f"  'the middle black bowl' -> {result}")
    print(f"  Position: {object_positions[result]}")
    print(f"  Expected: akita_black_bowl_2 (median y)")

    # ========================================================================
    # 10. UNDER (libero_90)
    # ========================================================================
    print("\n10. UNDER: 'the book under the cabinet shelf'")
    print("-" * 60)

    object_positions = {
        "book_1": np.array([0.30, 0.00, 0.30]),  # UNDER shelf
        "book_2": np.array([0.50, 0.00, 0.50]),  # ABOVE shelf
        "book_3": np.array([0.30, 0.50, 0.45]),  # far, not under
    }
    body_positions = {
        "cabinet_shelf_1": np.array([0.30, 0.00, 0.45]),
    }

    constraint = SpatialConstraint(relation="under", landmarks=["cabinet_shelf"])
    result = resolve_object_by_spatial_constraint("book", constraint, object_positions, body_positions)
    print(f"  Resolved: {result}")
    print(f"  Position: {object_positions[result]}")
    print(f"  Expected: book_1 (under the shelf, z < shelf.z)")

    print("\n" + "=" * 80)
    print("All spatial relations demonstrated!")
    print("=" * 80)


# Placeholder for actual SpatialConstraint and resolve function
class SpatialConstraint:
    def __init__(self, relation, landmarks):
        self.relation = relation
        self.landmarks = landmarks


def resolve_object_by_spatial_constraint(canonical, constraint, object_positions, body_positions=None):
    return None  # placeholder


if __name__ == "__main__":
    demo_all_spatial_relations()
