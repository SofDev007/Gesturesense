"""End-to-end gesture pipeline tests on synthetic landmark poses.

Each supported gesture has a synthetic pose fixture; the engine must pick
the right rule with solid confidence, and ambiguous poses must yield no
detection at all.
"""

from __future__ import annotations

import pytest

from gesturesense.gesture.base import registered_names
from gesturesense.gesture.engine import GestureEngine
from gesturesense.gesture.features import extract_features
from tests import fixtures


@pytest.fixture()
def engine() -> GestureEngine:
    return GestureEngine(min_confidence=0.55, history=7, min_votes=4)


@pytest.mark.parametrize("expected,builder", sorted(fixtures.GESTURE_POSES.items()))
def test_each_gesture_wins_on_its_pose(engine, expected, builder):
    """The correct rule must win with clear confidence on its own pose."""
    features = extract_features(builder(), "Right")
    match = engine.classify(features)
    assert match is not None, f"no gesture detected for {expected}"
    assert match.name == expected
    assert match.confidence > 0.7


def test_all_gestures_have_a_pose_fixture():
    """Every registered rule is covered by the parametrised test above."""
    assert set(fixtures.GESTURE_POSES) == set(registered_names())


def test_relaxed_hand_matches_nothing(engine):
    features = extract_features(fixtures.relaxed_hand(), "Right")
    assert engine.classify(features) is None


def test_process_hand_stabilises_over_frames(engine):
    """A steady pose becomes a RecognizedGesture within the vote window."""
    landmarks = fixtures.thumbs_up()
    result = None
    for _ in range(7):
        features, result = engine.process_hand(landmarks, "Right")
    assert features.handedness == "Right"
    assert result is not None
    assert result.name == "Thumbs Up"
    assert result.handedness == "Right"
    assert 0.0 < result.confidence <= 1.0


def test_hands_are_stabilised_independently(engine):
    """Left and right hands must not share temporal state."""
    up = fixtures.thumbs_up()
    palm = fixtures.open_palm()
    for _ in range(7):
        _, right = engine.process_hand(up, "Right")
        _, left = engine.process_hand(palm, "Left")
    assert right is not None and right.name == "Thumbs Up"
    assert left is not None and left.name == "Open Palm"


def test_hand_lost_resets_state(engine):
    landmarks = fixtures.peace_sign()
    for _ in range(7):
        engine.process_hand(landmarks, "Right")
    engine.hand_lost("Right")
    # Immediately after a reset one frame is not enough for a majority.
    _, result = engine.process_hand(landmarks, "Right")
    assert result is None
