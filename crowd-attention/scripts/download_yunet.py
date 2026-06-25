#!/usr/bin/env python3
"""Download the OpenCV Zoo YuNet ONNX model into ./models/."""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path


MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
DEFAULT_OUTPUT = Path("models/face_detection_yunet_2023mar.onnx")
MIN_EXPECTED_BYTES = 200_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_model(path: Path) -> None:
    size = path.stat().st_size
    prefix = path.read_bytes()[:80]
    if size < MIN_EXPECTED_BYTES or prefix.startswith(b"version https://git-lfs.github.com/spec"):
        raise RuntimeError(
            f"Downloaded file does not look like the YuNet ONNX model: {path} "
            f"({size} bytes). It may be a Git LFS pointer or failed download."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Download OpenCV YuNet face detector.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Destination ONNX path.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing model file.")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists() and not args.force:
        validate_model(output)
        print(f"Already exists: {output}")
        print(f"sha256: {sha256(output)}")
        return 0

    print(f"Downloading {MODEL_URL}")
    print(f"Writing {output}")
    urllib.request.urlretrieve(MODEL_URL, output)
    validate_model(output)
    print(f"sha256: {sha256(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
