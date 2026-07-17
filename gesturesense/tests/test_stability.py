"""Tests for timing utilities, landmark smoothing and temporal stabilisation."""

from __future__ import annotations

import numpy as np
import pytest

from gesturesense.gesture.base import GestureMatch
from gesturesense.gesture.engine import GestureStabilizer
from gesturesense.tracking.smoother import LandmarkSmoother
from gesturesense.utils.timing import FpsCounter


# ---------------------------------------------------------------------------
# FpsCounter
# ---------------------------------------------------------------------------
def test_fps_counter_reports_zero_before_two_ticks():
    counter = FpsCounter()
    assert counter.fps == 0.0
    counter.tick(now=0.0)
    assert counter.fps == 0.0


def test_fps_counter_converges_to_steady_rate():
    counter = FpsCounter(alpha=0.5)
    for i in range(50):
        counter.tick(now=i / 30.0)  # exactly 30 FPS
    assert counter.fps == pytest.approx(30.0, rel=1e-3)


def test_fps_counter_reset():
    counter = FpsCounter()
    counter.tick(now=0.0)
    counter.tick(now=0.1)
    assert counter.fps > 0
    counter.reset()
    assert counter.fps == 0.0


# ---------------------------------------------------------------------------
# LandmarkSmoother
# ---------------------------------------------------------------------------
def _landmarks(offset: float) -> np.ndarray:
    return np.full((21, 3), offset, dtype=np.float64)


def test_smoother_first_frame_passes_through():
    smoother = LandmarkSmoother(alpha=0.5)
    first = _landmarks(0.4)
    out = smoother.smooth("Right", first)
    assert np.allclose(out, first)


def test_smoother_blends_towards_new_position():
    smoother = LandmarkSmoother(alpha=0.5)
    smoother.smooth("Right", _landmarks(0.0))
    out = smoother.smooth("Right", _landmarks(0.1))
    assert np.allclose(out, _landmarks(0.05))


def test_smoother_alpha_one_is_passthrough():
    smoother = LandmarkSmoother(alpha=1.0)
    smoother.smooth("Right", _landmarks(0.0))
    out = smoother.smooth("Right", _landmarks(0.9))
    assert np.allclose(out, _landmarks(0.9))


def test_smoother_teleport_resets_instead_of_dragging():
    smoother = LandmarkSmoother(alpha=0.2)
    smoother.smooth("Right", _landmarks(0.1))
    far = _landmarks(0.9)  # wrist jump ≫ threshold → re-detection
    out = smoother.smooth("Right", far)
    assert np.allclose(out, far)


def test_smoother_invalid_alpha_rejected():
    with pytest.raises(ValueError):
        LandmarkSmoother(alpha=0.0)
    with pytest.raises(ValueError):
        LandmarkSmoother(alpha=1.5)


def test_smoother_prune_returns_lost_hands():
    smoother = LandmarkSmoother(alpha=0.5)
    smoother.smooth("Right", _landmarks(0.1))
    smoother.smooth("Left", _landmarks(0.2))

    smoother.mark_all_stale()
    smoother.smooth("Right", _landmarks(0.11))  # only Right seen this frame
    assert smoother.prune() == ["Left"]

    smoother.mark_all_stale()
    assert sorted(smoother.prune()) == ["Right"]


# ---------------------------------------------------------------------------
# GestureStabilizer
# ---------------------------------------------------------------------------
def _match(name: str, confidence: float = 0.9) -> GestureMatch:
    return GestureMatch(name=name, confidence=confidence)


def test_stabilizer_requires_majority_before_reporting():
    stab = GestureStabilizer(history=7, min_votes=4)
    for i in range(3):
        name, _ = stab.update(_match("Fist"))
        assert name is None, f"reported too early at frame {i}"
    name, confidence = stab.update(_match("Fist"))
    assert name == "Fist"
    assert confidence > 0.0


def test_stabilizer_ignores_single_frame_flicker():
    stab = GestureStabilizer(history=7, min_votes=4)
    for _ in range(6):
        stab.update(_match("Fist"))
    name, _ = stab.update(_match("Open Palm"))  # one-frame glitch
    assert name == "Fist"


def test_stabilizer_switches_after_sustained_change():
    stab = GestureStabilizer(history=7, min_votes=4)
    for _ in range(7):
        stab.update(_match("Fist"))
    name = "Fist"
    for _ in range(7):
        name, _ = stab.update(_match("Open Palm"))
    assert name == "Open Palm"


def test_stabilizer_reset_clears_state():
    stab = GestureStabilizer(history=7, min_votes=4)
    for _ in range(7):
        stab.update(_match("Fist"))
    stab.reset()
    name, confidence = stab.update(_match("Fist"))
    assert name is None
    assert confidence == 0.0
