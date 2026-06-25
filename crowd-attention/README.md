# Vision Attention Sidecar

Local Python sidecar for a browser-based presentation. It estimates an approximate aggregate forward-facing audience signal from a webcam and publishes one normalized value between `0` and `1`.

This is not gaze tracking, identity recognition, emotion inference, or individual analytics. It uses multi-face detection plus coarse landmark geometry to produce a room-level signal.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_yunet.py
```

The downloader stores OpenCV Zoo's YuNet model at:

```text
./models/face_detection_yunet_2023mar.onnx
```

## Debug Mode

Use a known-forward calibration image and a test image:

```bash
python sidecar.py --debug \
  --model ./models/face_detection_yunet_2023mar.onnx \
  --calibration ./calib.jpg \
  --test ./test.jpg
```

Default output is a single numeric attention value. Add `--verbose` to print JSON with face counts, baseline, and confidence diagnostics.

To visually inspect what the sidecar is using, write annotated copies of both images:

```bash
python sidecar.py --debug --verbose \
  --model ./models/face_detection_yunet_2023mar.onnx \
  --calibration ./calib.png \
  --test ./test.png \
  --annotate-calibration ./calib_annotated.png \
  --annotate-test ./test_annotated.png
```

Annotation colors:

- Green boxes: clearly forward-facing by the current heuristic.
- Yellow boxes: ambiguous.
- Red boxes: likely not forward-facing.
- Gray boxes: detected but skipped as unusable.

Landmark dots show the eye, nose, and mouth-corner points used by the scorer.

## Live Mode

```bash
python sidecar.py --camera 0 --port 8765 \
  --model ./models/face_detection_yunet_2023mar.onnx
```

Use a looping video file as the live source for testing:

```bash
python sidecar.py --video ./audience.mp4 --port 8765 \
  --model ./models/face_detection_yunet_2023mar.onnx
```

Video mode calibrates once from the first frame, then samples frames by wall-clock playback time without recalibrating. If processing is slower than the video, frames are skipped rather than played back slowly.

For annotated preview frames in `monitor.html`, add `--preview`:

```bash
python sidecar.py --video ./audience.mp4 --port 8765 --preview \
  --model ./models/face_detection_yunet_2023mar.onnx
```

Without `--preview`, WebSocket messages remain attention-only for production use.

Defaults:

- WebSocket: `ws://localhost:8765`
- Startup calibration: first `5` seconds
- Processing rate: `10` FPS
- Smoothing alpha: `0.2`

Camera mode keeps only the latest camera frame, so processing does not build up a stale-frame backlog.

Manual recalibration is available by sending:

```json
{ "command": "calibrate" }
```

Outgoing messages stay minimal:

```json
{ "attention": 0.72 }
```

## Browser Integration

For a barebones live display, open `monitor.html` in a browser while the sidecar is running.

```js
const socket = new WebSocket("ws://localhost:8765");

socket.addEventListener("message", (event) => {
  const { attention } = JSON.parse(event.data);
  const opacity = Math.max(0, Math.min(1, attention));
  document.documentElement.style.setProperty("--presentation-opacity", opacity);
});
```

## Tuning

Useful flags:

```bash
python sidecar.py \
  --camera 0 \
  --host localhost \
  --port 8765 \
  --fps 10 \
  --calibration-seconds 5 \
  --score-threshold 0.6 \
  --nms-threshold 0.3 \
  --top-k 5000 \
  --smoothing 0.2
```

For wide audience shots, lower `--score-threshold` can recover more small faces but may increase false positives. The sidecar weights detections by confidence and capped face area, normalizes against calibration, and degrades smoothly when faces disappear.

The published `attention` value is based on forward-facingness relative to calibration. A single-face calibration can still reach `1.0`; a large drop from a many-face calibration is damped as low coverage.

The scorer also applies a calibrated head-down penalty using the same eye/nose/mouth landmarks. This helps catch phone-looking posture without adding eye tracking or person-specific identity matching.
