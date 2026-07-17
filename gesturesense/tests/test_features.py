"""Tests for feature extraction and per-finger state detection."""

from __future__ import annotations

import numpy as np
import pytest

from gesturesense.gesture import topology as topo
from gesturesense.gesture.features import extract_features
from gesturesense.gesture.finger_state import FingerState, read_fingers
from gesturesense.utils import geometry as geo
from tests import fixtures


def test_hand_size_matches_fixture_scale():
    features = extract_features(fixtures.open_palm(), "Right")
    assert features.hand_size == pytest.approx(fixtures.HAND_SIZE, rel=1e-6)


def test_open_palm_reads_all_fingers_extended():
    features = extract_features(fixtures.open_palm(), "Right")
    for finger in topo.FINGER_NAMES:
        assert features.state(finger) is FingerState.EXTENDED, finger
        assert features.extension(finger) > 0.9, finger


def test_closed_fist_reads_all_fingers_curled():
    features = extract_features(fixtures.closed_fist(), "Right")
    for finger in topo.FINGER_NAMES:
        assert features.state(finger) is FingerState.CURLED, finger
        assert features.extension(finger) < 0.1, finger


def test_half_bent_finger_reads_half():
    landmarks = fixtures.build_hand(
        finger_folds={"index": fixtures.FOLD_HALF}, thumb="tucked"
    )
    readings = read_fingers(landmarks, fixtures.HAND_SIZE)
    assert readings["index"].state is FingerState.HALF


def test_pinch_distance_is_small_for_ok_sign():
    features = extract_features(fixtures.ok_sign(), "Right")
    assert features.pinch_distance < 0.1


def test_index_middle_spread_for_peace_sign():
    features = extract_features(fixtures.peace_sign(), "Right")
    assert features.index_middle_spread == pytest.approx(24.0, abs=1.0)


def test_finger_direction_and_screen_direction():
    features = extract_features(fixtures.pointing_left(), "Right")
    direction = features.direction("index")
    assert float(np.dot(direction, fixtures.LEFT)) > 0.99
    assert features.screen_direction("index") is geo.Direction.LEFT


def test_handedness_is_preserved():
    features = extract_features(fixtures.open_palm(), "Left")
    assert features.handedness == "Left"


def test_degenerate_hand_does_not_crash():
    """All landmarks collapsed to one point must not divide by zero."""
    landmarks = np.full((topo.NUM_LANDMARKS, 3), 0.5)
    features = extract_features(landmarks, "Right")
    assert features.hand_size > 0.0
    for finger in topo.FINGER_NAMES:
        assert 0.0 <= features.extension(finger) <= 1.0
