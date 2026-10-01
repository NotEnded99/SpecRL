"""Monitor-only visual STL approach robustness."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from robosuite.utils import camera_utils


@dataclass(frozen=True)
class VisualApproachEstimate:
    valid: bool
    distance_m: float = float("nan")
    robustness: float = float("nan")
    surface_p01_distance_m: float = float("nan")
    surface_p05_distance_m: float = float("nan")
    surface_p10_distance_m: float = float("nan")
    surface_p01_robustness: float = float("nan")
    surface_p05_robustness: float = float("nan")
    surface_p10_robustness: float = float("nan")
    center_world: tuple[float, float, float] | None = None
    z_p05_m: float = float("nan")
    z_p50_m: float = float("nan")
    z_p95_m: float = float("nan")
    mask_pixels: int = 0
    depth_pixels: int = 0
    reason: str = ""
    # In-memory only. The shadow recorder consumes this RGB-D point cloud for
    # relation predicates; it is never serialized in the main STL CSV.
    points_world: np.ndarray | None = None


def _robust_center(points: np.ndarray) -> np.ndarray:
    """Median centre after removing the outer 10% radial outliers."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    points = points[np.all(np.isfinite(points), axis=1)]

    if len(points) == 0:
        raise ValueError("no finite points")

    center = np.median(points, axis=0)

    if len(points) < 20:
        return center

    radius = np.linalg.norm(points - center, axis=1)
    cutoff = np.quantile(radius, 0.90)
    retained = points[radius <= cutoff]

    if len(retained) == 0:
        return center

    return np.median(retained, axis=0)


def estimate_visual_approach(
    *,
    sim,
    camera_name: str,
    mask: np.ndarray,
    raw_depth: np.ndarray,
    eef_world: np.ndarray,
    intrinsic: np.ndarray | None = None,
    camera_to_world: np.ndarray | None = None,
    near_m: float | None = None,
    far_m: float | None = None,
    grasp_radius_m: float = 0.10,
    surface_radius_m: float = 0.05,
    min_pixels: int = 20,
    depth_quantiles: tuple[float, float] = (0.05, 0.95),
    metric_depth: np.ndarray | None = None,
) -> VisualApproachEstimate:
    """Compute rho_visual = grasp_radius - ||eef - visual centre||.

    ``mask`` and ``raw_depth`` must use exactly the same image orientation.
    If RGB is rotated with ``[::-1, ::-1]`` before YOLOE, depth must receive
    the identical rotation.
    """
    mask = np.asarray(mask, dtype=bool)
    raw_depth = np.asarray(raw_depth, dtype=np.float64).squeeze()

    if mask.shape != raw_depth.shape:
        return VisualApproachEstimate(
            valid=False,
            mask_pixels=int(mask.sum()),
            reason=f"shape_mismatch:{mask.shape}!={raw_depth.shape}",
        )

    mask_pixels = int(mask.sum())
    if mask_pixels < min_pixels:
        return VisualApproachEstimate(
            valid=False,
            mask_pixels=mask_pixels,
            reason="mask_too_small",
        )

    try:
        if sim is not None:
            metric_depth = camera_utils.get_real_depth_map(
                sim,
                raw_depth,
            )
            intrinsic = (
                camera_utils.get_camera_intrinsic_matrix(
                    sim,
                    camera_name,
                    raw_depth.shape[0],
                    raw_depth.shape[1],
                )
            )
            camera_to_world = (
                camera_utils.get_camera_extrinsic_matrix(
                    sim,
                    camera_name,
                )
            )
        elif metric_depth is None:
            if (
                intrinsic is None
                or camera_to_world is None
                or near_m is None
                or far_m is None
            ):
                raise ValueError(
                    "remote calibration is incomplete"
                )

            near_value = float(near_m)
            far_value = float(far_m)

            if (
                near_value <= 0.0
                or far_value <= near_value
            ):
                raise ValueError(
                    "invalid remote depth clipping planes"
                )

            metric_depth = near_value / (
                1.0
                - raw_depth
                * (1.0 - near_value / far_value)
            )

        else:
            # The conversion from MuJoCo's normalized depth to metric depth
            # depends only on the camera frame and clipping planes, not on an
            # instance mask.  Callers that evaluate several YOLOE masks from
            # the same frame may therefore compute it once and reuse it.  The
            # exact expression above remains the authoritative computation;
            # this branch only avoids repeating it for every detection.
            metric_depth = np.asarray(metric_depth, dtype=np.float64).squeeze()
            if metric_depth.shape != raw_depth.shape:
                raise ValueError(
                    "precomputed metric depth shape mismatch: "
                    f"{metric_depth.shape}!={raw_depth.shape}"
                )

        intrinsic = np.asarray(
            intrinsic,
            dtype=np.float64,
        )
        camera_to_world = np.asarray(
            camera_to_world,
            dtype=np.float64,
        )

        valid = (
            mask
            & np.isfinite(metric_depth)
            & (metric_depth > 0.02)
        )

        depth_values = metric_depth[valid]
        if len(depth_values) < min_pixels:
            return VisualApproachEstimate(
                valid=False,
                mask_pixels=mask_pixels,
                depth_pixels=int(len(depth_values)),
                reason="insufficient_depth",
            )

        low_q, high_q = depth_quantiles
        low = np.quantile(depth_values, low_q)
        high = np.quantile(depth_values, high_q)
        valid &= (
            (metric_depth >= low)
            & (metric_depth <= high)
        )

        rows, cols = np.nonzero(valid)
        z = metric_depth[rows, cols]

        fx = float(intrinsic[0, 0])
        fy = float(intrinsic[1, 1])
        cx = float(intrinsic[0, 2])
        cy = float(intrinsic[1, 2])

        x = (cols.astype(np.float64) - cx) * z / fx
        # MuJoCo / OpenGL image rows grow downwards, whereas the camera-frame
        # y axis used by robosuite's camera-to-world extrinsic grows upwards.
        # Neglecting this sign made objects above the image centre project
        # below the table (and vice versa), producing a systematic x/z bias.
        y = (cy - rows.astype(np.float64)) * z / fy

        camera_points = np.column_stack(
            (x, y, z, np.ones_like(z))
        )
        world_h = (
            camera_to_world @ camera_points.T
        ).T
        world_points = (
            world_h[:, :3] / world_h[:, 3:4]
        )

        center = _robust_center(world_points)
        eef = np.asarray(
            eef_world,
            dtype=np.float64,
        ).reshape(3)

        distance = float(np.linalg.norm(eef - center))
        robustness = float(grasp_radius_m - distance)

        # Robust distances from the gripper centre to the
        # nearest visible target surface. Percentiles are used
        # instead of the absolute minimum to suppress isolated
        # depth or mask-boundary outliers.
        point_distances = np.linalg.norm(
            world_points - eef[None, :],
            axis=1,
        )
        (
            surface_p01_distance,
            surface_p05_distance,
            surface_p10_distance,
        ) = np.quantile(
            point_distances,
            (0.01, 0.05, 0.10),
        )
        z_p05, z_p50, z_p95 = np.quantile(
            world_points[:, 2],
            (0.05, 0.50, 0.95),
        )

        return VisualApproachEstimate(
            valid=True,
            distance_m=distance,
            robustness=robustness,
            surface_p01_distance_m=float(
                surface_p01_distance
            ),
            surface_p05_distance_m=float(
                surface_p05_distance
            ),
            surface_p10_distance_m=float(
                surface_p10_distance
            ),
            surface_p01_robustness=float(
                surface_radius_m
                - surface_p01_distance
            ),
            surface_p05_robustness=float(
                surface_radius_m
                - surface_p05_distance
            ),
            surface_p10_robustness=float(
                surface_radius_m
                - surface_p10_distance
            ),
            center_world=tuple(float(x) for x in center),
            z_p05_m=float(z_p05),
            z_p50_m=float(z_p50),
            z_p95_m=float(z_p95),
            mask_pixels=mask_pixels,
            depth_pixels=int(len(world_points)),
            points_world=world_points,
        )

    except Exception as error:
        return VisualApproachEstimate(
            valid=False,
            mask_pixels=mask_pixels,
            reason=(
                f"{type(error).__name__}:"
                f"{str(error)[:120]}"
            ),
        )


def visual_pick_margin(
    approach_robustness: float,
    two_finger_grasp: bool,
) -> float:
    """Legacy oracle-contact comparison; not a visual-only score."""
    if two_finger_grasp:
        return 0.01

    return min(
        float(approach_robustness),
        -1.0e-6,
    )
