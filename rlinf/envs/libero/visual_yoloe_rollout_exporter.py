"""Opt-in simulator-label export for offline YOLOE repair datasets.

This module is diagnostics-only. It must never provide values to the visual
monitor, AGM runtime, reward, termination, or success computation.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import cv2
import numpy as np


CLASS_NAMES = [
    "black bowl",
    "plate",
    "ramekin",
    "cookie box",
    "alphabet soup",
    "cream cheese",
    "salad dressing",
    "bbq sauce",
    "ketchup",
    "tomato sauce",
    "butter",
    "milk carton",
    "chocolate pudding",
    "orange juice",
    "wine bottle",
    "moka pot",
    "white mug",
    "yellow and white mug",
    "black book",
    "basket",
    "stove",
    "wooden cabinet",
    "white cabinet",
    "wine rack",
    "desk caddy",
    "microwave",
]
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
INSTANCE_CLASS_RULES = [
    (("akita_black_bowl",), "black bowl"),
    (("glazed_rim_porcelain_ramekin", "ramekin"), "ramekin"),
    (("cookies", "cookie_box"), "cookie box"),
    (("alphabet_soup",), "alphabet soup"),
    (("cream_cheese",), "cream cheese"),
    (("salad_dressing",), "salad dressing"),
    (("bbq_sauce",), "bbq sauce"),
    (("tomato_sauce",), "tomato sauce"),
    (("ketchup",), "ketchup"),
    (("butter",), "butter"),
    (("chocolate_pudding",), "chocolate pudding"),
    (("orange_juice",), "orange juice"),
    (("milk",), "milk carton"),
    (("wine_bottle",), "wine bottle"),
    (("moka_pot",), "moka pot"),
    (("white_yellow_mug", "yellow_white_mug"), "yellow and white mug"),
    (("porcelain_mug", "white_mug"), "white mug"),
    (("black_book",), "black book"),
    (("plate",), "plate"),
    (("basket",), "basket"),
    (("flat_stove", "stove"), "stove"),
    (("wooden_cabinet",), "wooden cabinet"),
    (("white_cabinet",), "white cabinet"),
    (("wine_rack",), "wine rack"),
    (("desk_caddy", "caddy"), "desk caddy"),
    (("microwave",), "microwave"),
]


def class_name_for_instance(name: str) -> str | None:
    lowered = re.sub(
        r"_+",
        "_",
        str(name).strip().lower().replace("-", "_"),
    )
    for markers, class_name in INSTANCE_CLASS_RULES:
        if any(marker in lowered for marker in markers):
            return class_name
    return None


def mask_to_polygon(
    mask: np.ndarray,
    width: int,
    height: int,
    min_contour_area: float,
) -> list[float] | None:
    contours = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )[-2]
    contours = [
        contour
        for contour in contours
        if cv2.contourArea(contour) >= min_contour_area
    ]
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    epsilon = max(1.0, 0.002 * cv2.arcLength(contour, True))
    points = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    if len(points) < 3:
        return None
    values = []
    for x, y in points:
        values.extend([
            float(np.clip(x / width, 0.0, 1.0)),
            float(np.clip(y / height, 0.0, 1.0)),
        ])
    return values


class VisualYoloeRolloutExporter:
    """Write sparse rollout RGB and simulator instance masks as YOLO labels."""

    def __init__(
        self,
        output_root: str | Path,
        interval: int = 5,
        cameras=("agentview", "robot0_eye_in_hand"),
        task_ids=(35,),
        required_class_names=("black book", "desk caddy"),
        val_every: int = 5,
        min_mask_pixels: int = 20,
        min_contour_area: float = 20.0,
    ):
        self.output_root = Path(output_root).expanduser().resolve()
        self.interval = int(interval)
        self.cameras = tuple(str(value) for value in cameras)
        self.task_ids = {int(value) for value in task_ids}
        required_class_names = tuple(str(value) for value in required_class_names)
        unknown_required = sorted(
            set(required_class_names).difference(CLASS_TO_ID)
        )
        if unknown_required:
            raise ValueError(
                "unknown required YOLOE classes: "
                + ", ".join(unknown_required)
            )
        self.required_class_ids = {
            CLASS_TO_ID[value] for value in required_class_names
        }
        self.val_every = int(val_every)
        self.min_mask_pixels = int(min_mask_pixels)
        self.min_contour_area = float(min_contour_area)
        if self.interval < 1 or self.val_every < 2:
            raise ValueError("interval must be positive and val_every >= 2")
        if not self.cameras or not self.task_ids or not self.required_class_ids:
            raise ValueError(
                "cameras, task_ids, and required_class_names must be nonempty"
            )

        for split in ("train", "val"):
            (self.output_root / "images" / split).mkdir(
                parents=True,
                exist_ok=True,
            )
            (self.output_root / "labels" / split).mkdir(
                parents=True,
                exist_ok=True,
            )
        self.worker_dir = self.output_root / "workers" / f"worker_pid_{os.getpid()}"
        self.worker_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.worker_dir / "index.jsonl"
        self._episodes = {}
        self._seen = set()
        self._write_yaml_once()

    def _write_yaml_once(self) -> None:
        yaml_path = self.output_root / "libero40.yaml"
        lines = [
            f"path: {self.output_root}",
            "train: images/train",
            "val: images/val",
            "names:",
        ]
        lines.extend(
            f"  {index}: {json.dumps(name)}"
            for index, name in enumerate(CLASS_NAMES)
        )
        try:
            with yaml_path.open("x", encoding="utf-8") as stream:
                stream.write("\n".join(lines) + "\n")
        except FileExistsError:
            pass

    def start_episode(
        self,
        env_id: int,
        task_id: int,
        trial_id: int,
        instance_id_to_name: dict,
        raw_obs: dict | None,
    ) -> None:
        env_id = int(env_id)
        task_id = int(task_id)
        trial_id = int(trial_id)
        self._episodes[env_id] = {
            "task_id": task_id,
            "trial_id": trial_id,
            "instance_id_to_name": {
                int(key): str(value)
                for key, value in (instance_id_to_name or {}).items()
            },
        }
        if task_id in self.task_ids and raw_obs is not None:
            self.record_step(env_id, 0, raw_obs, force=True)

    def record_step(
        self,
        env_id: int,
        step: int,
        raw_obs: dict,
        force: bool = False,
    ) -> int:
        env_id = int(env_id)
        step = int(step)
        episode = self._episodes.get(env_id)
        if episode is None or episode["task_id"] not in self.task_ids:
            return 0
        if not force and step % self.interval != 0:
            return 0

        split = "val" if episode["trial_id"] % self.val_every == 0 else "train"
        saved = 0
        for camera in self.cameras:
            key = (env_id, episode["task_id"], episode["trial_id"], step, camera)
            if key in self._seen:
                continue
            if self._save_camera_sample(
                env_id=env_id,
                task_id=episode["task_id"],
                trial_id=episode["trial_id"],
                step=step,
                camera=camera,
                split=split,
                raw_obs=raw_obs,
                instance_id_to_name=episode["instance_id_to_name"],
            ):
                self._seen.add(key)
                saved += 1
        return saved

    def _save_camera_sample(
        self,
        *,
        env_id: int,
        task_id: int,
        trial_id: int,
        step: int,
        camera: str,
        split: str,
        raw_obs: dict,
        instance_id_to_name: dict[int, str],
    ) -> bool:
        image_key = f"{camera}_image"
        segmentation_key = f"{camera}_segmentation_instance"
        if image_key not in raw_obs or segmentation_key not in raw_obs:
            raise KeyError(
                f"rollout export requires {image_key!r} and {segmentation_key!r}; "
                "add +env.eval.init_params.camera_segmentations=instance"
            )
        image = np.ascontiguousarray(np.asarray(raw_obs[image_key])[::-1, ::-1])
        segmentation = np.ascontiguousarray(
            np.asarray(raw_obs[segmentation_key]).squeeze()[::-1, ::-1]
        )
        height, width = image.shape[:2]
        labels = []
        visible_class_ids = set()
        class_pixels = {}
        for instance_id, instance_name in instance_id_to_name.items():
            mask = segmentation == int(instance_id)
            pixels = int(mask.sum())
            if pixels < self.min_mask_pixels:
                continue
            class_name = class_name_for_instance(instance_name)
            if class_name is None:
                continue
            class_id = CLASS_TO_ID[class_name]
            polygon = mask_to_polygon(
                mask,
                width,
                height,
                self.min_contour_area,
            )
            if polygon is None:
                continue
            labels.append((class_id, polygon))
            visible_class_ids.add(class_id)
            class_pixels[class_name] = class_pixels.get(class_name, 0) + pixels

        if not (visible_class_ids & self.required_class_ids):
            return False

        stem = (
            f"pid{os.getpid()}_env{env_id:02d}_task{task_id:02d}_"
            f"trial{trial_id:03d}_step{step:04d}_{camera}"
        )
        image_path = self.output_root / "images" / split / f"{stem}.png"
        label_path = self.output_root / "labels" / split / f"{stem}.txt"
        if image_path.exists() or label_path.exists():
            raise FileExistsError(f"rollout export collision: {stem}")
        if not cv2.imwrite(
            str(image_path),
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        ):
            raise IOError(f"failed to write {image_path}")
        lines = [
            f"{class_id} " + " ".join(f"{value:.6f}" for value in polygon)
            for class_id, polygon in labels
        ]
        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        record = {
            "image": str(image_path.relative_to(self.output_root)),
            "label": str(label_path.relative_to(self.output_root)),
            "split": split,
            "env_id": env_id,
            "task_id": task_id,
            "trial_id": trial_id,
            "step": step,
            "camera": camera,
            "visible_classes": [CLASS_NAMES[value] for value in sorted(visible_class_ids)],
            "class_pixels": class_pixels,
        }
        with self.index_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        return True
