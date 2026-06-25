#!/usr/bin/env python3
"""Local webcam sidecar for approximate aggregate forward-facing attention.

This is not eye tracking. It estimates a room-level signal from face detections
and coarse facial landmark geometry, then normalizes against a forward-facing
calibration baseline.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_MODEL = "./models/face_detection_yunet_2023mar.onnx"
LOW_ATTENTION_FLOOR = 0.15
MIN_BASELINE = 0.05
CLEAR_FORWARD_THRESHOLD = 0.80


@dataclass(frozen=True)
class FaceScore:
    score: float
    weight: float
    confidence: float
    eye_to_mouth_ratio: float | None = None


@dataclass(frozen=True)
class FrameScore:
    raw: float
    face_count: int
    usable_count: int
    confidence: float
    eye_to_mouth_ratio: float | None = None


@dataclass(frozen=True)
class ImageAnalysis:
    image: np.ndarray
    detections: np.ndarray | None
    frame_score: FrameScore


@dataclass(frozen=True)
class FrameAnalysis:
    frame: np.ndarray
    detections: np.ndarray | None
    frame_score: FrameScore


@dataclass
class AttentionState:
    baseline: float | None = None
    baseline_usable_count: int | None = None
    baseline_eye_to_mouth_ratio: float | None = None
    smoothed: float | None = None
    calibrating_until: float = 0.0
    calibration_seconds: float = 5.0
    calibration_samples: list[float] | None = None
    calibration_count_samples: list[int] | None = None
    calibration_eye_to_mouth_samples: list[float] | None = None
    recalibration_requested: bool = False


class CameraFrameSource:
    def __init__(self, cv: object, camera_index: int) -> None:
        self._cv = cv
        self._cap = cv.VideoCapture(camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_index}")
        self._lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def read_latest(self) -> tuple[bool, np.ndarray | None]:
        with self._lock:
            if self._latest_frame is None:
                return False, None
            return True, self._latest_frame.copy()

    def release(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._cap.release()

    def _capture_loop(self) -> None:
        while self._running:
            ok, frame = self._cap.read()
            if ok:
                with self._lock:
                    self._latest_frame = frame
            else:
                time.sleep(0.01)


class VideoFrameSource:
    def __init__(self, cv: object, path: str) -> None:
        self._cv = cv
        self._path = path
        self._cap = cv.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video {path}")

        self.fps = float(self._cap.get(cv.CAP_PROP_FPS) or 0.0)
        if self.fps <= 0:
            self.fps = 30.0
        self.frame_count = int(self._cap.get(cv.CAP_PROP_FRAME_COUNT) or 0)
        if self.frame_count <= 0:
            self._cap.release()
            raise RuntimeError(f"Could not determine frame count for video: {path}")

        self._playback_start = time.monotonic()

    def read_first_frame(self) -> tuple[bool, np.ndarray | None]:
        self._cap.set(self._cv.CAP_PROP_POS_FRAMES, 0)
        ok, frame = self._cap.read()
        self._playback_start = time.monotonic()
        return ok, frame

    def read_latest(self) -> tuple[bool, np.ndarray | None]:
        index = current_video_frame_index(
            time.monotonic() - self._playback_start,
            self.fps,
            self.frame_count,
        )
        self._cap.set(self._cv.CAP_PROP_POS_FRAMES, index)
        return self._cap.read()

    def release(self) -> None:
        self._cap.release()


def clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def ema(previous: float | None, current: float, alpha: float) -> float:
    current = clamp01(current)
    alpha = clamp01(alpha)
    if previous is None:
        return current
    return clamp01(previous * (1.0 - alpha) + current * alpha)


def detection_to_face_score(
    detection: np.ndarray,
    frame_area: float,
    min_confidence: float = 0.0,
    min_face_size: float = 8.0,
    baseline_eye_to_mouth_ratio: float | None = None,
) -> FaceScore | None:
    """Score one YuNet detection for coarse front-facingness.

    YuNet detections are expected to be:
    x, y, w, h, right_eye_x, right_eye_y, left_eye_x, left_eye_y,
    nose_x, nose_y, right_mouth_x, right_mouth_y, left_mouth_x,
    left_mouth_y, confidence.
    """
    if detection is None or len(detection) < 15:
        return None

    values = np.asarray(detection[:15], dtype=np.float32)
    if not np.all(np.isfinite(values)):
        return None

    x, y, w, h = map(float, values[0:4])
    confidence = float(values[14])
    if confidence < min_confidence or w < min_face_size or h < min_face_size:
        return None

    landmarks = values[4:14].reshape((5, 2)).astype(np.float32)
    if not _landmarks_are_plausible(landmarks, x, y, w, h):
        return None

    eye_a, eye_b, nose, mouth_a, mouth_b = landmarks
    eye_mid = (eye_a + eye_b) * 0.5
    mouth_mid = (mouth_a + mouth_b) * 0.5
    eye_dist = float(np.linalg.norm(eye_a - eye_b))
    mouth_dist = float(np.linalg.norm(mouth_a - mouth_b))
    if eye_dist < 2.0 or mouth_dist < 1.0:
        return None

    nose_shift = float(nose[0] - eye_mid[0]) / eye_dist
    mouth_shift = float(mouth_mid[0] - nose[0]) / eye_dist
    eye_tilt = float(eye_a[1] - eye_b[1]) / eye_dist
    mouth_tilt = float(mouth_a[1] - mouth_b[1]) / eye_dist

    eye_y = float(eye_mid[1])
    nose_y = float(nose[1])
    mouth_y = float(mouth_mid[1])
    face_h = max(h, 1.0)
    vertical_order = _soft_positive(nose_y - eye_y, face_h * 0.10) * _soft_positive(
        mouth_y - nose_y, face_h * 0.12
    )

    eye_width_ratio = eye_dist / max(w, 1.0)
    mouth_width_ratio = mouth_dist / max(w, 1.0)
    eye_to_mouth_ratio = float(mouth_mid[1] - eye_mid[1]) / eye_dist

    # These gates intentionally represent "clearly forward" rather than merely
    # "face-like." Side-facing faces often still have valid landmarks.
    linear_score = (
        0.38 * _gate_abs_ratio(nose_shift, 0.28)
        + 0.16 * _gate_abs_ratio(mouth_shift, 0.38)
        + 0.12 * _gate_abs_ratio(eye_tilt, 0.23)
        + 0.08 * _gate_abs_ratio(mouth_tilt, 0.32)
        + 0.08 * _gate_range(eye_width_ratio, 0.34, 0.54, 0.16)
        + 0.06 * _gate_range(mouth_width_ratio, 0.26, 0.46, 0.16)
        + 0.08 * _gate_range(eye_to_mouth_ratio, 0.75, 1.32, 0.55)
        + 0.04 * clamp01(vertical_order)
    )
    forward_score = clamp01(linear_score) ** 2.0
    score = forward_score * downward_attention_gate(
        eye_to_mouth_ratio,
        baseline_eye_to_mouth_ratio,
    )

    area_ratio = (w * h) / max(frame_area, 1.0)
    capped_area = min(area_ratio, 0.02) / 0.02
    weight = clamp01(confidence) * (0.35 + 0.65 * math.sqrt(clamp01(capped_area)))
    return FaceScore(
        score=clamp01(score),
        weight=max(weight, 1e-6),
        confidence=clamp01(confidence),
        eye_to_mouth_ratio=eye_to_mouth_ratio,
    )


def aggregate_frame_score(
    detections: np.ndarray | None,
    frame_shape: tuple[int, int] | tuple[int, int, int],
    min_confidence: float = 0.0,
    baseline_eye_to_mouth_ratio: float | None = None,
) -> FrameScore:
    height = int(frame_shape[0])
    width = int(frame_shape[1])
    frame_area = float(max(height * width, 1))

    if detections is None:
        return FrameScore(raw=0.0, face_count=0, usable_count=0, confidence=0.0)

    rows = np.asarray(detections, dtype=np.float32)
    if rows.size == 0:
        return FrameScore(raw=0.0, face_count=0, usable_count=0, confidence=0.0)
    if rows.ndim == 1:
        rows = rows.reshape((1, -1))

    face_scores = [
        face_score
        for row in rows
        if (
            face_score := detection_to_face_score(
                row,
                frame_area=frame_area,
                min_confidence=min_confidence,
                baseline_eye_to_mouth_ratio=baseline_eye_to_mouth_ratio,
            )
        )
        is not None
    ]
    if not face_scores:
        return FrameScore(raw=0.0, face_count=len(rows), usable_count=0, confidence=0.0)

    trimmed = _trim_low_scores(face_scores)
    weights = np.asarray([item.weight for item in trimmed], dtype=np.float32)
    scores = np.asarray([item.score for item in trimmed], dtype=np.float32)
    weighted_mean = float(np.average(scores, weights=weights))
    clear_forward = (scores >= CLEAR_FORWARD_THRESHOLD).astype(np.float32)
    weighted_clear_share = float(np.average(clear_forward, weights=weights))
    raw = 0.85 * weighted_clear_share + 0.15 * weighted_mean

    usable_count = len(face_scores)
    mean_detection_confidence = float(np.mean([item.confidence for item in face_scores]))
    confidence = clamp01(mean_detection_confidence)
    ratios = [
        item.eye_to_mouth_ratio
        for item in face_scores
        if item.eye_to_mouth_ratio is not None and math.isfinite(item.eye_to_mouth_ratio)
    ]
    frame_ratio = float(np.median(ratios)) if ratios else None
    return FrameScore(
        raw=clamp01(raw),
        face_count=len(rows),
        usable_count=usable_count,
        confidence=confidence,
        eye_to_mouth_ratio=frame_ratio,
    )


def normalize_attention(
    frame_score: FrameScore,
    baseline: float | None,
    baseline_usable_count: int | None = None,
) -> float:
    if frame_score.usable_count <= 0 or frame_score.confidence <= 0.0:
        return LOW_ATTENTION_FLOOR

    effective_baseline = max(float(baseline or MIN_BASELINE), MIN_BASELINE)
    normalized = clamp01(frame_score.raw / effective_baseline)
    return clamp01(normalized * coverage_damping(frame_score.usable_count, baseline_usable_count))


def coverage_damping(current_usable_count: int, baseline_usable_count: int | None) -> float:
    if baseline_usable_count is None or baseline_usable_count < 4:
        return 1.0
    coverage_ratio = current_usable_count / max(float(baseline_usable_count), 1.0)
    if coverage_ratio >= 0.25:
        return 1.0
    return clamp01(coverage_ratio / 0.25)


def downward_attention_gate(
    eye_to_mouth_ratio: float,
    baseline_eye_to_mouth_ratio: float | None,
) -> float:
    if baseline_eye_to_mouth_ratio is None or baseline_eye_to_mouth_ratio <= 0:
        return 1.0

    # Looking down tends to compress the eye-to-mouth geometry relative to a
    # forward calibration. Keep this broad so it remains crowd/person agnostic.
    relative_drop = (baseline_eye_to_mouth_ratio - eye_to_mouth_ratio) / baseline_eye_to_mouth_ratio
    if relative_drop <= 0.08:
        return 1.0
    if relative_drop >= 0.35:
        return 0.25
    return 1.0 - ((relative_drop - 0.08) / 0.27) * 0.75


def degrade_attention(previous: float | None) -> float:
    if previous is None:
        return LOW_ATTENTION_FLOOR
    return clamp01(previous * 0.92 + LOW_ATTENTION_FLOOR * 0.08)


def update_attention(
    state: AttentionState,
    frame_score: FrameScore,
    now: float,
    smoothing: float,
) -> float:
    if state.calibration_samples is None:
        state.calibration_samples = []
    if state.calibration_count_samples is None:
        state.calibration_count_samples = []
    if state.calibration_eye_to_mouth_samples is None:
        state.calibration_eye_to_mouth_samples = []

    if state.recalibration_requested:
        state.baseline = None
        state.baseline_usable_count = None
        state.baseline_eye_to_mouth_ratio = None
        state.calibration_samples.clear()
        state.calibration_count_samples.clear()
        state.calibration_eye_to_mouth_samples.clear()
        state.calibrating_until = now + max(0.0, state.calibration_seconds)
        state.recalibration_requested = False

    if state.baseline is None or now < state.calibrating_until:
        if frame_score.usable_count > 0 and frame_score.raw > 0:
            state.calibration_samples.append(frame_score.raw)
            state.calibration_count_samples.append(frame_score.usable_count)
            if frame_score.eye_to_mouth_ratio is not None:
                state.calibration_eye_to_mouth_samples.append(frame_score.eye_to_mouth_ratio)
        if now >= state.calibrating_until and state.calibration_samples:
            state.baseline = max(float(np.median(state.calibration_samples)), MIN_BASELINE)
            state.baseline_usable_count = int(round(float(np.median(state.calibration_count_samples))))
            if state.calibration_eye_to_mouth_samples:
                state.baseline_eye_to_mouth_ratio = float(
                    np.median(state.calibration_eye_to_mouth_samples)
                )
            state.calibration_samples.clear()
            state.calibration_count_samples.clear()
            state.calibration_eye_to_mouth_samples.clear()

    if frame_score.usable_count <= 0:
        current = degrade_attention(state.smoothed)
    else:
        current = normalize_attention(frame_score, state.baseline, state.baseline_usable_count)

    state.smoothed = ema(state.smoothed, current, smoothing)
    return state.smoothed


class YuNetDetector:
    def __init__(
        self,
        model_path: str,
        score_threshold: float,
        nms_threshold: float,
        top_k: int,
    ) -> None:
        import cv2 as cv

        model = Path(model_path)
        if not model.exists():
            raise FileNotFoundError(
                f"Model not found: {model}. Run scripts/download_yunet.py or pass --model."
            )

        self._cv = cv
        self._detector = cv.FaceDetectorYN.create(
            str(model),
            "",
            (320, 320),
            float(score_threshold),
            float(nms_threshold),
            int(top_k),
        )
        self._input_size: tuple[int, int] | None = None

    def detect(self, frame: np.ndarray) -> np.ndarray | None:
        height, width = frame.shape[:2]
        input_size = (int(width), int(height))
        if input_size != self._input_size:
            self._detector.setInputSize(input_size)
            self._input_size = input_size

        _, faces = self._detector.detect(frame)
        return faces


def analyze_image(path: str, detector: YuNetDetector, min_confidence: float) -> ImageAnalysis:
    import cv2 as cv

    image = cv.imread(path)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    detections = detector.detect(image)
    frame_score = aggregate_frame_score(detections, image.shape, min_confidence=min_confidence)
    return ImageAnalysis(image=image, detections=detections, frame_score=frame_score)


def analyze_image_with_baseline(
    path: str,
    detector: YuNetDetector,
    min_confidence: float,
    baseline_eye_to_mouth_ratio: float | None,
) -> ImageAnalysis:
    import cv2 as cv

    image = cv.imread(path)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    detections = detector.detect(image)
    frame_score = aggregate_frame_score(
        detections,
        image.shape,
        min_confidence=min_confidence,
        baseline_eye_to_mouth_ratio=baseline_eye_to_mouth_ratio,
    )
    return ImageAnalysis(image=image, detections=detections, frame_score=frame_score)


def process_image(path: str, detector: YuNetDetector, min_confidence: float) -> FrameScore:
    return analyze_image(path, detector, min_confidence).frame_score


def annotate_analysis(
    analysis: ImageAnalysis,
    output_path: str,
    label: str,
    min_confidence: float,
    attention: float | None = None,
    baseline: float | None = None,
    baseline_eye_to_mouth_ratio: float | None = None,
) -> None:
    import cv2 as cv

    annotated = render_annotated_frame(
        cv,
        analysis.image,
        analysis.detections,
        analysis.frame_score,
        label=label,
        min_confidence=min_confidence,
        attention=attention,
        baseline=baseline,
        baseline_eye_to_mouth_ratio=baseline_eye_to_mouth_ratio,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv.imwrite(str(output), annotated):
        raise RuntimeError(f"Could not write annotated image: {output}")


def run_debug(args: argparse.Namespace) -> int:
    if not args.calibration or not args.test:
        raise SystemExit("--debug requires --calibration and --test")

    detector = YuNetDetector(args.model, args.score_threshold, args.nms_threshold, args.top_k)
    calibration_analysis = analyze_image(args.calibration, detector, args.score_threshold)
    calibration = calibration_analysis.frame_score
    baseline = max(calibration.raw, MIN_BASELINE)
    baseline_usable_count = calibration.usable_count
    baseline_eye_to_mouth_ratio = calibration.eye_to_mouth_ratio
    test_analysis = analyze_image_with_baseline(
        args.test,
        detector,
        args.score_threshold,
        baseline_eye_to_mouth_ratio,
    )
    test = test_analysis.frame_score
    unsmoothed = normalize_attention(test, baseline, baseline_usable_count)
    smoothed = ema(None, unsmoothed, args.smoothing)

    if args.annotate_calibration:
        annotate_analysis(
            calibration_analysis,
            args.annotate_calibration,
            label="calibration",
            min_confidence=args.score_threshold,
            attention=normalize_attention(calibration, baseline, baseline_usable_count),
            baseline=baseline,
            baseline_eye_to_mouth_ratio=baseline_eye_to_mouth_ratio,
        )
    if args.annotate_test:
        annotate_analysis(
            test_analysis,
            args.annotate_test,
            label="test",
            min_confidence=args.score_threshold,
            attention=smoothed,
            baseline=baseline,
            baseline_eye_to_mouth_ratio=baseline_eye_to_mouth_ratio,
        )

    if args.verbose:
        print(
            json.dumps(
                {
                    "attention": smoothed,
                    "unsmoothed": unsmoothed,
                    "baseline": baseline,
                    "baseline_faces": baseline_usable_count,
                    "baseline_eye_to_mouth_ratio": baseline_eye_to_mouth_ratio,
                    "calibration_raw": calibration.raw,
                    "calibration_faces": calibration.usable_count,
                    "test_raw": test.raw,
                    "test_faces": test.usable_count,
                    "test_confidence": test.confidence,
                },
                sort_keys=True,
            )
        )
    else:
        print(f"{smoothed:.4f}")
    return 0


async def run_live(args: argparse.Namespace) -> int:
    import cv2 as cv
    from websockets.asyncio.server import serve

    detector = YuNetDetector(args.model, args.score_threshold, args.nms_threshold, args.top_k)
    source_label = f"video {args.video}" if args.video else f"camera {args.camera}"
    source = VideoFrameSource(cv, args.video) if args.video else CameraFrameSource(cv, args.camera)

    clients: set[object] = set()
    state = AttentionState(
        calibrating_until=time.monotonic() + max(0.0, args.calibration_seconds),
        calibration_seconds=max(0.0, args.calibration_seconds),
        calibration_samples=[],
    )

    if args.video:
        ok, first_frame = source.read_first_frame()
        if not ok:
            source.release()
            raise RuntimeError(f"Could not read first frame from video: {args.video}")
        first_frame_score = score_frame(first_frame, detector, args.score_threshold)
        state.baseline = max(first_frame_score.raw, MIN_BASELINE)
        state.baseline_usable_count = first_frame_score.usable_count
        state.baseline_eye_to_mouth_ratio = first_frame_score.eye_to_mouth_ratio
        state.calibrating_until = 0.0
        state.calibration_samples = []
        state.calibration_count_samples = []
        state.calibration_eye_to_mouth_samples = []
        state.smoothed = normalize_attention(
            first_frame_score,
            state.baseline,
            state.baseline_usable_count,
        )
    else:
        source.start()

    stop = asyncio.Event()

    def handle_stop() -> None:
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, handle_stop)
        except NotImplementedError:
            pass

    async def client_handler(websocket: object) -> None:
        clients.add(websocket)
        try:
            if state.smoothed is not None:
                await websocket.send(json.dumps({"attention": round(clamp01(state.smoothed), 4)}))
            async for message in websocket:
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    continue
                if data.get("command") == "calibrate" and not args.video:
                    state.recalibration_requested = True
        finally:
            clients.discard(websocket)

    async def capture_loop() -> None:
        frame_interval = 1.0 / max(float(args.fps), 0.1)

        while not stop.is_set():
            started = time.monotonic()
            ok, frame = source.read_latest()
            if not ok:
                current = degrade_attention(state.smoothed)
                state.smoothed = ema(state.smoothed, current, args.smoothing)
                await broadcast_payload({"attention": round(clamp01(state.smoothed), 4)})
            else:
                analysis = analyze_frame(
                    frame,
                    detector,
                    args.score_threshold,
                    state.baseline_eye_to_mouth_ratio,
                )
                attention = update_attention(
                    state,
                    analysis.frame_score,
                    time.monotonic(),
                    args.smoothing,
                )
                payload: dict[str, object] = {"attention": round(clamp01(attention), 4)}
                if args.preview:
                    payload["faces"] = analysis.frame_score.usable_count
                    payload["frame"] = encode_preview_frame(
                        cv,
                        analysis,
                        label="preview",
                        min_confidence=args.score_threshold,
                        attention=attention,
                        baseline=state.baseline,
                        baseline_eye_to_mouth_ratio=state.baseline_eye_to_mouth_ratio,
                        width=args.preview_width,
                        quality=args.preview_quality,
                    )
                await broadcast_payload(payload)

            elapsed = time.monotonic() - started
            await asyncio.sleep(max(0.0, frame_interval - elapsed))

    async def broadcast_payload(payload_data: dict[str, object]) -> None:
        if not clients:
            return
        payload = json.dumps(payload_data)
        dead_clients = []
        for websocket in list(clients):
            try:
                await websocket.send(payload)
            except Exception:
                dead_clients.append(websocket)
        for websocket in dead_clients:
            clients.discard(websocket)

    async with serve(client_handler, args.host, args.port):
        print(f"Attention sidecar listening on ws://{args.host}:{args.port}")
        if args.video:
            print(
                f"Using {source_label}; source is {source.fps:.2f} FPS, "
                f"{source.frame_count} frames; baseline from first frame is "
                f"{state.baseline:.3f}; wall-clock sampling without recalibration."
            )
        else:
            print(
                f"Calibrating for {args.calibration_seconds:.1f}s from camera {args.camera}; "
                "latest-frame capture active; send {\"command\":\"calibrate\"} to recalibrate."
            )
        capture_task = asyncio.create_task(capture_loop())
        await stop.wait()
        capture_task.cancel()
        try:
            await capture_task
        except asyncio.CancelledError:
            pass

    source.release()
    return 0


def score_frame(frame: np.ndarray, detector: YuNetDetector, min_confidence: float) -> FrameScore:
    return analyze_frame(frame, detector, min_confidence).frame_score


def analyze_frame(
    frame: np.ndarray,
    detector: YuNetDetector,
    min_confidence: float,
    baseline_eye_to_mouth_ratio: float | None = None,
) -> FrameAnalysis:
    detections = detector.detect(frame)
    frame_score = aggregate_frame_score(
        detections,
        frame.shape,
        min_confidence=min_confidence,
        baseline_eye_to_mouth_ratio=baseline_eye_to_mouth_ratio,
    )
    return FrameAnalysis(frame=frame, detections=detections, frame_score=frame_score)


def encode_preview_frame(
    cv: object,
    analysis: FrameAnalysis,
    label: str,
    min_confidence: float,
    attention: float | None,
    baseline: float | None,
    baseline_eye_to_mouth_ratio: float | None,
    width: int,
    quality: int,
) -> str:
    annotated = render_annotated_frame(
        cv,
        analysis.frame,
        analysis.detections,
        analysis.frame_score,
        label=label,
        min_confidence=min_confidence,
        attention=attention,
        baseline=baseline,
        baseline_eye_to_mouth_ratio=baseline_eye_to_mouth_ratio,
    )
    resized = resize_to_width(cv, annotated, width)
    ok, encoded = cv.imencode(
        ".jpg",
        resized,
        [int(cv.IMWRITE_JPEG_QUALITY), int(max(1, min(100, quality)))],
    )
    if not ok:
        raise RuntimeError("Could not encode preview JPEG")
    data = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{data}"


def current_video_frame_index(elapsed_seconds: float, fps: float, frame_count: int) -> int:
    if fps <= 0 or frame_count <= 0:
        return 0
    return int(max(0.0, elapsed_seconds) * fps) % frame_count


def render_annotated_frame(
    cv: object,
    frame: np.ndarray,
    detections: np.ndarray | None,
    frame_score: FrameScore,
    label: str,
    min_confidence: float,
    attention: float | None = None,
    baseline: float | None = None,
    baseline_eye_to_mouth_ratio: float | None = None,
) -> np.ndarray:
    annotated = frame.copy()
    frame_area = float(max(frame.shape[0] * frame.shape[1], 1))
    rows = _detection_rows(detections)

    for row in rows:
        face_score = detection_to_face_score(
            row,
            frame_area=frame_area,
            min_confidence=min_confidence,
            baseline_eye_to_mouth_ratio=baseline_eye_to_mouth_ratio,
        )
        x, y, w, h = [int(round(value)) for value in row[:4]]
        score = face_score.score if face_score is not None else 0.0
        color = _score_color(score, face_score is not None)

        cv.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        if len(row) >= 15:
            landmarks = np.asarray(row[4:14], dtype=np.float32).reshape((5, 2))
            _draw_landmarks(cv, annotated, landmarks)

        text = f"{score:.2f}" if face_score is not None else "skip"
        text_y = max(14, y - 5)
        cv.putText(
            annotated,
            text,
            (x, text_y),
            cv.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv.LINE_AA,
        )

    _draw_summary(cv, annotated, frame_score, label, attention, baseline)
    return annotated


def resize_to_width(cv: object, image: np.ndarray, width: int) -> np.ndarray:
    target_width = int(width)
    if target_width <= 0 or image.shape[1] <= target_width:
        return image
    scale = target_width / image.shape[1]
    target_height = max(1, int(round(image.shape[0] * scale)))
    return cv.resize(image, (target_width, target_height), interpolation=cv.INTER_AREA)


def _landmarks_are_plausible(
    landmarks: np.ndarray,
    x: float,
    y: float,
    w: float,
    h: float,
) -> bool:
    pad_x = w * 0.35
    pad_y = h * 0.35
    min_x = x - pad_x
    max_x = x + w + pad_x
    min_y = y - pad_y
    max_y = y + h + pad_y
    for point in landmarks:
        px, py = map(float, point)
        if px < min_x or px > max_x or py < min_y or py > max_y:
            return False
    return True


def _soft_positive(value: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return clamp01(value / scale)


def _gate_abs_ratio(value: float, limit: float) -> float:
    if limit <= 0:
        return 0.0
    return clamp01(1.0 - (abs(value) / limit) ** 2)


def _gate_range(value: float, low: float, high: float, falloff: float) -> float:
    if low <= value <= high:
        return 1.0
    if falloff <= 0:
        return 0.0
    if value < low:
        return clamp01(1.0 - (low - value) / falloff)
    return clamp01(1.0 - (value - high) / falloff)


def _detection_rows(detections: np.ndarray | None) -> np.ndarray:
    if detections is None:
        return np.empty((0, 15), dtype=np.float32)
    rows = np.asarray(detections, dtype=np.float32)
    if rows.size == 0:
        return np.empty((0, 15), dtype=np.float32)
    if rows.ndim == 1:
        rows = rows.reshape((1, -1))
    return rows


def _score_color(score: float, usable: bool) -> tuple[int, int, int]:
    if not usable:
        return (150, 150, 150)
    if score >= CLEAR_FORWARD_THRESHOLD:
        return (40, 190, 40)
    if score >= 0.45:
        return (0, 210, 255)
    return (40, 40, 230)


def _draw_landmarks(cv: object, image: np.ndarray, landmarks: np.ndarray) -> None:
    colors = [
        (255, 90, 90),
        (255, 90, 90),
        (80, 220, 255),
        (255, 120, 255),
        (255, 120, 255),
    ]
    for point, color in zip(landmarks, colors):
        x, y = [int(round(value)) for value in point]
        cv.circle(image, (x, y), 3, color, -1, cv.LINE_AA)

    eye_mid = tuple(np.round((landmarks[0] + landmarks[1]) * 0.5).astype(int))
    nose = tuple(np.round(landmarks[2]).astype(int))
    mouth_mid = tuple(np.round((landmarks[3] + landmarks[4]) * 0.5).astype(int))
    cv.line(image, eye_mid, nose, (80, 220, 255), 1, cv.LINE_AA)
    cv.line(image, nose, mouth_mid, (255, 120, 255), 1, cv.LINE_AA)


def _draw_summary(
    cv: object,
    image: np.ndarray,
    frame_score: FrameScore,
    label: str,
    attention: float | None,
    baseline: float | None,
) -> None:
    lines = [
        f"{label}",
        f"faces: {frame_score.usable_count}/{frame_score.face_count}",
        f"raw: {frame_score.raw:.3f}",
        f"confidence: {frame_score.confidence:.3f}",
    ]
    if baseline is not None:
        lines.append(f"baseline: {baseline:.3f}")
    if attention is not None:
        lines.append(f"attention: {clamp01(attention):.3f}")

    x, y = 16, 30
    line_height = 24
    width = 270
    height = 18 + line_height * len(lines)
    overlay = image.copy()
    cv.rectangle(overlay, (8, 8), (8 + width, 8 + height), (0, 0, 0), -1)
    cv.addWeighted(overlay, 0.58, image, 0.42, 0, image)
    for index, line in enumerate(lines):
        cv.putText(
            image,
            line,
            (x, y + index * line_height),
            cv.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            1,
            cv.LINE_AA,
        )


def _trim_low_scores(face_scores: list[FaceScore]) -> list[FaceScore]:
    if len(face_scores) < 5:
        return face_scores
    ordered = sorted(face_scores, key=lambda item: item.score)
    trim_count = max(1, int(len(ordered) * 0.1))
    return ordered[trim_count:]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate approximate aggregate forward-facing audience attention "
            "from YuNet face detections."
        )
    )
    parser.add_argument("--debug", action="store_true", help="Run image-based test mode.")
    parser.add_argument("--calibration", help="Calibration image path for --debug.")
    parser.add_argument("--test", help="Test image path for --debug.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index for live mode.")
    parser.add_argument("--video", help="Video file to use as a looping live-mode source.")
    parser.add_argument("--host", default="localhost", help="WebSocket bind host.")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket port.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to YuNet ONNX model.")
    parser.add_argument("--calibration-seconds", type=float, default=5.0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--score-threshold", type=float, default=0.6)
    parser.add_argument("--nms-threshold", type=float, default=0.3)
    parser.add_argument("--top-k", type=int, default=5000)
    parser.add_argument("--smoothing", type=float, default=0.2)
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Include annotated JPEG preview frames in WebSocket messages.",
    )
    parser.add_argument("--preview-width", type=int, default=960)
    parser.add_argument("--preview-quality", type=int, default=70)
    parser.add_argument(
        "--annotate-calibration",
        help="Write an annotated calibration image in --debug mode.",
    )
    parser.add_argument(
        "--annotate-test",
        help="Write an annotated test image in --debug mode.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.debug:
        return run_debug(args)
    return asyncio.run(run_live(args))


if __name__ == "__main__":
    raise SystemExit(main())
