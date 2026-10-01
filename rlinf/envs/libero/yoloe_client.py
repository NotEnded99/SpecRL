import io
import struct
import urllib.request

import cv2
import numpy as np


class YoloeClient:
    def __init__(
        self,
        base_url="http://127.0.0.1:8010",
        timeout=60,
        raw_transport=False,
    ):
        base_url = base_url.rstrip("/")
        self.infer_url = base_url + "/infer"
        self.infer_batch_url = base_url + "/infer_batch"
        self.infer_raw_url = base_url + "/infer_raw"
        self.infer_batch_raw_url = base_url + "/infer_batch_raw"
        self.timeout = timeout
        self.raw_transport = bool(raw_transport)
        self._batch_fallback_reported = False

    @staticmethod
    def _encode_image(image_rgb):
        image_rgb = np.asarray(image_rgb, dtype=np.uint8)

        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(
                f"Expected RGB HWC image, got {image_rgb.shape}"
            )

        image_bgr = cv2.cvtColor(
            image_rgb,
            cv2.COLOR_RGB2BGR,
        )
        success, encoded = cv2.imencode(".png", image_bgr)

        if not success:
            raise RuntimeError("Failed to encode YOLOE input image")

        return image_rgb, encoded.tobytes()

    @staticmethod
    def _raw_image_payload(image_rgb):
        image_rgb = np.asarray(image_rgb, dtype=np.uint8)
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(
                f"Expected RGB HWC image, got {image_rgb.shape}"
            )
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        height, width = image_rgb.shape[:2]
        payload = struct.pack("!III", height, width, 3) + image_bgr.tobytes()
        return image_rgb, payload

    @staticmethod
    def _detections_from_arrays(
        class_ids,
        class_names,
        confidences,
        boxes,
        masks,
    ):
        if len(class_names) != len(class_ids):
            raise RuntimeError(
                "YOLOE class-name and detection counts do not match"
            )
        if len(confidences) != len(class_ids):
            raise RuntimeError(
                "YOLOE confidence and detection counts do not match"
            )
        if len(boxes) != len(class_ids):
            raise RuntimeError(
                "YOLOE box and detection counts do not match"
            )
        if masks.shape[0] != len(class_ids):
            raise RuntimeError(
                "YOLOE detection and mask counts do not match"
            )

        return [
            {
                "class_id": int(class_ids[index]),
                "class_name": str(class_names[index]),
                "confidence": float(confidences[index]),
                "box": boxes[index],
                "mask": masks[index],
            }
            for index in range(len(class_ids))
        ]

    def infer(self, image_rgb):
        if self.raw_transport:
            image_rgb, encoded = self._raw_image_payload(image_rgb)
            infer_url = self.infer_raw_url
            content_type = "application/x-yoloe-bgr8"
        else:
            image_rgb, encoded = self._encode_image(image_rgb)
            infer_url = self.infer_url
            content_type = "image/png"

        request = urllib.request.Request(
            infer_url,
            data=encoded,
            headers={"Content-Type": content_type},
            method="POST",
        )

        with urllib.request.urlopen(
            request,
            timeout=self.timeout,
        ) as response:
            payload = response.read()

        with np.load(
            io.BytesIO(payload),
            allow_pickle=False,
        ) as archive:
            class_ids = archive["class_ids"].copy()
            class_names = archive["class_names"].copy()
            confidences = archive["confidences"].copy()
            boxes = archive["boxes"].copy()
            masks = archive["masks"].copy().astype(bool)
            image_shape = archive["image_shape"].copy()

        expected_shape = tuple(image_rgb.shape[:2])

        if tuple(image_shape) != expected_shape:
            raise RuntimeError(
                f"YOLOE image shape mismatch: "
                f"{tuple(image_shape)} != {expected_shape}"
            )

        return self._detections_from_arrays(
            class_ids,
            class_names,
            confidences,
            boxes,
            masks,
        )

    @staticmethod
    def _batch_payload(encoded_images):
        count = len(encoded_images)
        header = struct.pack("!I", count)
        lengths = b"".join(
            struct.pack("!I", len(encoded))
            for encoded in encoded_images
        )
        return header + lengths + b"".join(encoded_images)

    def _infer_batch_once(self, images_rgb):
        encoded_images = []
        expected_shapes = []
        if self.raw_transport:
            normalized_images = []
            bgr_images = []
            for image_rgb in images_rgb:
                normalized = np.asarray(image_rgb, dtype=np.uint8)
                if normalized.ndim != 3 or normalized.shape[2] != 3:
                    raise ValueError(
                        f"Expected RGB HWC image, got {normalized.shape}"
                    )
                normalized_images.append(normalized)
                bgr_images.append(cv2.cvtColor(
                    normalized, cv2.COLOR_RGB2BGR
                ))
                expected_shapes.append(tuple(normalized.shape[:2]))
            if len(set(expected_shapes)) != 1:
                raise ValueError(
                    "all images in a raw batch must have the same shape"
                )
            height, width = expected_shapes[0]
            batch_payload = (
                struct.pack("!III", len(bgr_images), height, width)
                + b"".join(image.tobytes() for image in bgr_images)
            )
            infer_batch_url = self.infer_batch_raw_url
            content_type = "application/x-yoloe-bgr8-batch"
        else:
            for image_rgb in images_rgb:
                normalized, encoded = self._encode_image(image_rgb)
                encoded_images.append(encoded)
                expected_shapes.append(tuple(normalized.shape[:2]))
            batch_payload = self._batch_payload(encoded_images)
            infer_batch_url = self.infer_batch_url
            content_type = "application/x-yoloe-png-batch"

        request = urllib.request.Request(
            infer_batch_url,
            data=batch_payload,
            headers={"Content-Type": content_type},
            method="POST",
        )

        with urllib.request.urlopen(
            request,
            timeout=self.timeout,
        ) as response:
            payload = response.read()

        with np.load(
            io.BytesIO(payload),
            allow_pickle=False,
        ) as archive:
            offsets = archive["detection_offsets"].copy()
            class_ids = archive["class_ids"].copy()
            class_names = archive["class_names"].copy()
            confidences = archive["confidences"].copy()
            boxes = archive["boxes"].copy()
            masks = archive["masks"].copy().astype(bool)
            image_shapes = archive["image_shapes"].copy()

        count = len(images_rgb)
        if offsets.shape != (count + 1,):
            raise RuntimeError(
                "YOLOE batch offsets have invalid shape: "
                f"{offsets.shape} != {(count + 1,)}"
            )
        if image_shapes.shape != (count, 2):
            raise RuntimeError(
                "YOLOE batch image-shape table has invalid shape: "
                f"{image_shapes.shape} != {(count, 2)}"
            )
        if int(offsets[0]) != 0 or int(offsets[-1]) != len(class_ids):
            raise RuntimeError("YOLOE batch offsets do not cover detections")
        if np.any(np.diff(offsets) < 0):
            raise RuntimeError("YOLOE batch offsets are not monotonic")

        outputs = []
        for index, expected_shape in enumerate(expected_shapes):
            actual_shape = tuple(int(x) for x in image_shapes[index])
            if actual_shape != expected_shape:
                raise RuntimeError(
                    "YOLOE batch image shape mismatch at index "
                    f"{index}: {actual_shape} != {expected_shape}"
                )
            start = int(offsets[index])
            end = int(offsets[index + 1])
            outputs.append(self._detections_from_arrays(
                class_ids[start:end],
                class_names[start:end],
                confidences[start:end],
                boxes[start:end],
                masks[start:end],
            ))

        return outputs

    def infer_batch(
        self,
        images_rgb,
        *,
        max_batch_size=32,
        fallback_to_single=True,
    ):
        """Infer a sequence of RGB images with order-preserving results."""
        images_rgb = list(images_rgb)
        if not images_rgb:
            return []
        max_batch_size = int(max_batch_size)
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")

        outputs = []
        for start in range(0, len(images_rgb), max_batch_size):
            chunk = images_rgb[start:start + max_batch_size]
            try:
                outputs.extend(self._infer_batch_once(chunk))
            except Exception as error:
                if not fallback_to_single:
                    raise
                if not self._batch_fallback_reported:
                    self._batch_fallback_reported = True
                    print(
                        "[YOLOE_BATCH_FALLBACK] "
                        f"error={type(error).__name__}: {error}",
                        flush=True,
                    )
                outputs.extend(self.infer(image) for image in chunk)

        return outputs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8010",
    )
    args = parser.parse_args()

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise FileNotFoundError(args.image)

    client = YoloeClient(args.url)
    detections = client.infer(
        cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    )

    print(f"detections: {len(detections)}")
    for detection in detections:
        print(
            f"{detection['class_name']}: "
            f"confidence={detection['confidence']:.4f}, "
            f"pixels={int(detection['mask'].sum())}"
        )
