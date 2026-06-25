import unittest

import numpy as np

from sidecar import (
    LOW_ATTENTION_FLOOR,
    FrameScore,
    aggregate_frame_score,
    clamp01,
    coverage_damping,
    current_video_frame_index,
    detection_to_face_score,
    degrade_attention,
    downward_attention_gate,
    normalize_attention,
)


def make_detection(
    *,
    right_eye=(42, 40),
    left_eye=(58, 40),
    nose=(50, 52),
    right_mouth=(44, 65),
    left_mouth=(56, 65),
    confidence=0.9,
):
    return np.asarray(
        [
            30,
            25,
            40,
            55,
            *right_eye,
            *left_eye,
            *nose,
            *right_mouth,
            *left_mouth,
            confidence,
        ],
        dtype=np.float32,
    )


class ScoringTests(unittest.TestCase):
    def test_clamp01_bounds_and_nan(self):
        self.assertEqual(clamp01(-1.0), 0.0)
        self.assertEqual(clamp01(2.0), 1.0)
        self.assertEqual(clamp01(0.5), 0.5)
        self.assertEqual(clamp01(float("nan")), 0.0)

    def test_invalid_landmarks_are_rejected(self):
        detection = make_detection(nose=(500, 500))
        self.assertIsNone(
            detection_to_face_score(detection, frame_area=640 * 480, min_confidence=0.6)
        )

    def test_forward_geometry_scores_higher_than_skewed_geometry(self):
        forward = detection_to_face_score(
            make_detection(),
            frame_area=640 * 480,
            min_confidence=0.6,
        )
        skewed = detection_to_face_score(
            make_detection(nose=(65, 52), right_mouth=(58, 65), left_mouth=(69, 68)),
            frame_area=640 * 480,
            min_confidence=0.6,
        )

        self.assertIsNotNone(forward)
        self.assertIsNotNone(skewed)
        self.assertGreater(forward.score, skewed.score)

    def test_downward_gate_penalizes_compressed_vertical_geometry(self):
        self.assertEqual(downward_attention_gate(1.0, 1.0), 1.0)
        self.assertEqual(downward_attention_gate(0.92, 1.0), 1.0)
        self.assertLess(downward_attention_gate(0.7, 1.0), 0.6)

    def test_downward_geometry_scores_lower_than_forward_with_baseline(self):
        forward = detection_to_face_score(
            make_detection(),
            frame_area=640 * 480,
            min_confidence=0.6,
        )
        downward = detection_to_face_score(
            make_detection(
                nose=(50, 50),
                right_mouth=(44, 58),
                left_mouth=(56, 58),
            ),
            frame_area=640 * 480,
            min_confidence=0.6,
            baseline_eye_to_mouth_ratio=forward.eye_to_mouth_ratio,
        )

        self.assertIsNotNone(forward)
        self.assertIsNotNone(downward)
        self.assertLess(downward.score, forward.score)

    def test_zero_usable_faces_degrades_gracefully(self):
        frame_score = aggregate_frame_score(None, (480, 640, 3), min_confidence=0.6)
        self.assertEqual(frame_score.usable_count, 0)
        self.assertEqual(normalize_attention(frame_score, baseline=0.8), LOW_ATTENTION_FLOOR)
        self.assertGreaterEqual(degrade_attention(0.9), 0.0)
        self.assertLessEqual(degrade_attention(0.9), 1.0)

    def test_normalization_handles_weak_baseline_and_clamps(self):
        score = FrameScore(raw=0.9, face_count=4, usable_count=4, confidence=1.0)
        self.assertEqual(normalize_attention(score, baseline=0.0), 1.0)
        self.assertEqual(normalize_attention(score, baseline=None), 1.0)

    def test_single_face_can_reach_full_attention_when_baseline_matches(self):
        score = FrameScore(raw=0.997, face_count=1, usable_count=1, confidence=0.482)
        self.assertEqual(normalize_attention(score, baseline=0.995, baseline_usable_count=1), 1.0)

    def test_coverage_damping_only_applies_after_large_baseline_drop(self):
        self.assertEqual(coverage_damping(current_usable_count=1, baseline_usable_count=1), 1.0)
        self.assertEqual(coverage_damping(current_usable_count=30, baseline_usable_count=100), 1.0)
        self.assertAlmostEqual(coverage_damping(current_usable_count=1, baseline_usable_count=100), 0.04)

    def test_aggregate_scores_detected_faces(self):
        detections = np.vstack([make_detection(), make_detection(confidence=0.8)])
        frame_score = aggregate_frame_score(detections, (480, 640, 3), min_confidence=0.6)
        self.assertEqual(frame_score.face_count, 2)
        self.assertEqual(frame_score.usable_count, 2)
        self.assertGreaterEqual(frame_score.raw, 0.0)
        self.assertLessEqual(frame_score.raw, 1.0)
        self.assertGreaterEqual(frame_score.confidence, 0.0)
        self.assertLessEqual(frame_score.confidence, 1.0)

    def test_current_video_frame_index_uses_wall_clock_and_loops(self):
        self.assertEqual(current_video_frame_index(0.0, fps=30.0, frame_count=100), 0)
        self.assertEqual(current_video_frame_index(0.5, fps=30.0, frame_count=100), 15)
        self.assertEqual(current_video_frame_index(4.0, fps=30.0, frame_count=100), 20)

    def test_current_video_frame_index_handles_bad_metadata(self):
        self.assertEqual(current_video_frame_index(1.0, fps=0.0, frame_count=100), 0)
        self.assertEqual(current_video_frame_index(1.0, fps=30.0, frame_count=0), 0)


if __name__ == "__main__":
    unittest.main()
