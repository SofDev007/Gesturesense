"""Tests for the shared vector-geometry helpers."""

from __future__ import annotations

import numpy as np
import pytest

from gesturesense.utils import geometry as geo


def test_normalize_unit_length_and_zero_vector():
    v = geo.normalize(np.array([3.0, 4.0, 0.0]))
    assert geo.norm(v) == pytest.approx(1.0)
    zero = geo.normalize(np.zeros(3))
    assert geo.norm(zero) == 0.0


def test_distance():
    assert geo.distance(np.zeros(3), np.array([3.0, 4.0, 0.0])) == pytest.approx(5.0)


@pytest.mark.parametrize(
    "v1,v2,expected",
    [
        ((1, 0, 0), (0, 1, 0), 90.0),
        ((1, 0, 0), (1, 0, 0), 0.0),
        ((1, 0, 0), (-1, 0, 0), 180.0),
    ],
)
def test_angle_between(v1, v2, expected):
    assert geo.angle_between(np.array(v1, float), np.array(v2, float)) == pytest.approx(
        expected, abs=1e-6
    )


def test_joint_angle_straight_and_folded():
    a, b, c = np.array([0.0, 0, 0]), np.array([1.0, 0, 0]), np.array([2.0, 0, 0])
    assert geo.joint_angle(a, b, c) == pytest.approx(180.0)
    c_folded = np.array([0.0, 0.001, 0.0])
    assert geo.joint_angle(a, b, c_folded) < 1.0


def test_smoothstep_endpoints_and_monotonicity():
    assert geo.smoothstep(0.0, 0.2, 0.8) == 0.0
    assert geo.smoothstep(1.0, 0.2, 0.8) == 1.0
    assert geo.smoothstep(0.5, 0.2, 0.8) == pytest.approx(0.5)
    values = [geo.smoothstep(x, 0.2, 0.8) for x in np.linspace(0, 1, 21)]
    assert values == sorted(values)


def test_inverse_smoothstep_complements():
    for x in (0.0, 0.3, 0.5, 0.9):
        total = geo.smoothstep(x, 0.2, 0.8) + geo.inverse_smoothstep(x, 0.2, 0.8)
        assert total == pytest.approx(1.0)


@pytest.mark.parametrize(
    "vector,expected",
    [
        ((0, -1, 0), geo.Direction.UP),
        ((0, 1, 0), geo.Direction.DOWN),
        ((-1, 0, 0), geo.Direction.LEFT),
        ((1, 0, 0), geo.Direction.RIGHT),
        ((1, -1, 0), geo.Direction.NONE),  # perfect diagonal is ambiguous
        ((0, 0, 0), geo.Direction.NONE),
    ],
)
def test_principal_direction(vector, expected):
    assert geo.principal_direction(np.array(vector, float)) is expected


def test_rotate_2d_quarter_turn():
    rotated = geo.rotate_2d(np.array([1.0, 0.0, 5.0]), 90.0)
    assert rotated[0] == pytest.approx(0.0, abs=1e-9)
    assert rotated[1] == pytest.approx(1.0)
    assert rotated[2] == 5.0  # z untouched
