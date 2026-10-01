#!/usr/bin/env python3
"""Standalone YOLOE segmentation HTTP service."""

import argparse
import io
import struct
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
from ultralytics import YOLOE


CLASSES = [
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


class InferenceHandler(BaseHTTPRequestHandler):
    model = None
    device = 0
    image_size = 640
    confidence = 0.05
    max_batch_size = 32
    access_log = False
    inference_lock = threading.Lock()

    def _send(self, status, payload, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path != "/health":
            self._send(404, b"not found", "text/plain")
            return
        self._send(200, b"ok batch raw", "text/plain")

    @staticmethod
    def _decode_image(encoded):
        image_bgr = cv2.imdecode(
            np.frombuffer(encoded, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if image_bgr is None:
            raise ValueError("invalid encoded image")
        return image_bgr

    @staticmethod
    def _decode_raw_image(payload):
        if len(payload) < 12:
            raise ValueError("raw payload is missing image shape")
        height, width, channels = struct.unpack("!III", payload[:12])
        if height < 1 or width < 1 or channels != 3:
            raise ValueError(
                f"invalid raw image shape: {(height, width, channels)}"
            )
        expected = int(height) * int(width) * int(channels)
        if len(payload) - 12 != expected:
            raise ValueError(
                f"raw image byte count mismatch: {len(payload) - 12} != {expected}"
            )
        return np.frombuffer(payload, dtype=np.uint8, offset=12).reshape(
            int(height), int(width), 3
        ).copy()

    @staticmethod
    def _decode_raw_batch(payload):
        if len(payload) < 12:
            raise ValueError("raw batch payload is missing its header")
        count, height, width = struct.unpack("!III", payload[:12])
        if count < 1 or height < 1 or width < 1:
            raise ValueError(
                f"invalid raw batch shape: {(count, height, width, 3)}"
            )
        expected = int(count) * int(height) * int(width) * 3
        if len(payload) - 12 != expected:
            raise ValueError(
                f"raw batch byte count mismatch: {len(payload) - 12} != {expected}"
            )
        array = np.frombuffer(payload, dtype=np.uint8, offset=12).reshape(
            int(count), int(height), int(width), 3
        ).copy()
        return [array[index] for index in range(int(count))]

    @classmethod
    def _decode_batch(cls, payload):
        if len(payload) < 4:
            raise ValueError("batch payload is missing image count")
        count = struct.unpack("!I", payload[:4])[0]
        if count < 1:
            raise ValueError("batch must contain at least one image")
        if count > cls.max_batch_size:
            raise ValueError(
                f"batch size {count} exceeds maximum {cls.max_batch_size}"
            )
        header_size = 4 + 4 * count
        if len(payload) < header_size:
            raise ValueError("batch payload is missing image lengths")
        lengths = struct.unpack(
            f"!{count}I",
            payload[4:header_size],
        )
        if any(length < 1 for length in lengths):
            raise ValueError("batch contains an empty encoded image")
        if sum(lengths) != len(payload) - header_size:
            raise ValueError("batch image lengths do not match payload size")

        images = []
        offset = header_size
        for length in lengths:
            images.append(cls._decode_image(payload[offset:offset + length]))
            offset += length
        return images

    @staticmethod
    def _result_arrays(result, height, width):
        if result.boxes is None:
            class_ids = np.empty((0,), dtype=np.int64)
            confidences = np.empty((0,), dtype=np.float32)
            boxes = np.empty((0, 4), dtype=np.float32)
        else:
            class_ids = (
                result.boxes.cls.cpu().numpy().astype(np.int64)
            )
            confidences = (
                result.boxes.conf.cpu().numpy().astype(np.float32)
            )
            boxes = (
                result.boxes.xyxy.cpu().numpy().astype(np.float32)
            )

        class_names = np.asarray(
            [result.names[int(x)] for x in class_ids],
            dtype="<U64",
        )

        if result.masks is None:
            masks = np.zeros(
                (len(class_ids), height, width),
                dtype=np.uint8,
            )
        else:
            raw_masks = result.masks.data.cpu().numpy()
            masks = np.stack(
                [
                    cv2.resize(
                        mask.astype(np.float32),
                        (width, height),
                        interpolation=cv2.INTER_NEAREST,
                    ) > 0.5
                    for mask in raw_masks
                ],
                axis=0,
            ).astype(np.uint8)

        if len(masks) != len(class_ids):
            raise RuntimeError(
                "mask and detection counts differ: "
                f"{len(masks)} != {len(class_ids)}"
            )

        return class_ids, class_names, confidences, boxes, masks

    @staticmethod
    def _npz_payload(**arrays):
        output = io.BytesIO()
        np.savez_compressed(output, **arrays)
        return output.getvalue()

    def _infer_single(self, payload, *, raw=False):
        image_bgr = (
            self._decode_raw_image(payload)
            if raw else self._decode_image(payload)
        )
        height, width = image_bgr.shape[:2]
        with self.inference_lock:
            result = self.model.predict(
                source=image_bgr,
                imgsz=self.image_size,
                conf=self.confidence,
                device=self.device,
                verbose=False,
            )[0]
        class_ids, class_names, confidences, boxes, masks = (
            self._result_arrays(result, height, width)
        )
        return self._npz_payload(
            class_ids=class_ids,
            class_names=class_names,
            confidences=confidences,
            boxes=boxes,
            masks=masks,
            image_shape=np.asarray([height, width], dtype=np.int32),
        )

    def _infer_batch(self, payload, *, raw=False):
        images_bgr = (
            self._decode_raw_batch(payload)
            if raw else self._decode_batch(payload)
        )
        if len(images_bgr) > self.max_batch_size:
            raise ValueError(
                f"batch size {len(images_bgr)} exceeds maximum "
                f"{self.max_batch_size}"
            )
        shapes = [tuple(image.shape[:2]) for image in images_bgr]
        if len(set(shapes)) != 1:
            raise ValueError(
                "all images in a batch must have the same height and width"
            )
        with self.inference_lock:
            results = list(self.model.predict(
                source=images_bgr,
                batch=len(images_bgr),
                imgsz=self.image_size,
                conf=self.confidence,
                device=self.device,
                verbose=False,
            ))
        if len(results) != len(images_bgr):
            raise RuntimeError(
                "YOLOE returned an unexpected result count: "
                f"{len(results)} != {len(images_bgr)}"
            )

        per_image = [
            self._result_arrays(result, *shape)
            for result, shape in zip(results, shapes)
        ]
        counts = [len(arrays[0]) for arrays in per_image]
        offsets = np.zeros((len(counts) + 1,), dtype=np.int32)
        offsets[1:] = np.cumsum(counts, dtype=np.int32)
        total = int(offsets[-1])
        height, width = shapes[0]

        if total:
            class_ids = np.concatenate([arrays[0] for arrays in per_image])
            class_names = np.concatenate([arrays[1] for arrays in per_image])
            confidences = np.concatenate([arrays[2] for arrays in per_image])
            boxes = np.concatenate([arrays[3] for arrays in per_image])
            masks = np.concatenate([arrays[4] for arrays in per_image])
        else:
            class_ids = np.empty((0,), dtype=np.int64)
            class_names = np.empty((0,), dtype="<U64")
            confidences = np.empty((0,), dtype=np.float32)
            boxes = np.empty((0, 4), dtype=np.float32)
            masks = np.empty((0, height, width), dtype=np.uint8)

        return self._npz_payload(
            detection_offsets=offsets,
            class_ids=class_ids,
            class_names=class_names,
            confidences=confidences,
            boxes=boxes,
            masks=masks,
            image_shapes=np.asarray(shapes, dtype=np.int32),
        )

    def do_POST(self):
        if self.path not in {
            "/infer", "/infer_batch", "/infer_raw", "/infer_batch_raw"
        }:
            self._send(404, b"not found", "text/plain")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = self.rfile.read(length)
            is_batch = self.path in {"/infer_batch", "/infer_batch_raw"}
            is_raw = self.path in {"/infer_raw", "/infer_batch_raw"}
            output = (
                self._infer_batch(payload, raw=is_raw)
                if is_batch
                else self._infer_single(payload, raw=is_raw)
            )
            self._send(
                200,
                output,
                "application/octet-stream",
            )

        except Exception as error:
            message = (
                f"{type(error).__name__}: {error}"
            ).encode("utf-8", errors="replace")
            self._send(500, message, "text/plain")

    def log_message(self, format_string, *args):
        if not self.access_log:
            return
        print(
            "[YOLOE]",
            self.address_string(),
            format_string % args,
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--access-log", action="store_true")
    args = parser.parse_args()

    print("Loading:", args.weights, flush=True)
    model = YOLOE(args.weights)
    print("Setting classes...", flush=True)
    model.set_classes(CLASSES)

    InferenceHandler.model = model
    InferenceHandler.device = args.device
    InferenceHandler.image_size = args.imgsz
    InferenceHandler.confidence = args.conf
    InferenceHandler.max_batch_size = args.max_batch_size
    InferenceHandler.access_log = bool(args.access_log)

    server = ThreadingHTTPServer(
        (args.host, args.port),
        InferenceHandler,
    )

    print(
        f"YOLOE ready: http://{args.host}:{args.port}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
