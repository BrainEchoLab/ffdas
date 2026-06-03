"""Tests for cdist, rect_dist, and angle."""

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ffdas import cdist, rect_dist, angle


# cdist


def test_cdist_single_pair():
    a = np.array([[0.0, 0.0, 0.0]])
    b = np.array([[3.0, 4.0, 0.0]])
    assert_allclose(cdist(a, b), [[5.0]])


def test_cdist_identical_points():
    a = np.array([[1.0, 2.0, 3.0]])
    assert_allclose(cdist(a, a), [[0.0]], atol=1e-12)


def test_cdist_symmetry(rng):
    a = rng.standard_normal((5, 3))
    b = rng.standard_normal((7, 3))
    assert_allclose(cdist(a, b), cdist(b, a).T, atol=1e-12)


def test_cdist_known_distances():
    a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    b = np.array([[1.0, 0.0, 0.0]])
    assert_allclose(cdist(a, b), [[1.0], [0.0]], atol=1e-12)


def test_cdist_output_shape():
    assert cdist(np.zeros((4, 3)), np.zeros((6, 3))).shape == (4, 6)


def test_cdist_batched_shape():
    assert cdist(np.zeros((2, 5, 3)), np.zeros((7, 3))).shape == (2, 5, 7)


# rect_dist


def test_rect_dist_above_center():
    points = np.array([[0.0, 0.0, 1.0]])
    assert_allclose(rect_dist(points, np.array([2.0, 2.0])), [1.0])


def test_rect_dist_inside_projection():
    """xy projection inside rectangle — distance equals |z|."""
    points = np.array([[0.3, 0.3, 2.0]])
    assert_allclose(rect_dist(points, np.array([4.0, 4.0])), [2.0])


def test_rect_dist_outside():
    points = np.array([[2.0, 0.0, 0.0]])
    assert_allclose(rect_dist(points, np.array([2.0, 2.0])), [1.0])


def test_rect_dist_on_edge():
    points = np.array([[1.0, 0.0, 0.0]])
    assert_allclose(rect_dist(points, np.array([2.0, 2.0])), [0.0], atol=1e-12)


def test_rect_dist_batch_shape():
    assert rect_dist(np.zeros((3, 4, 3)), np.array([1.0, 1.0])).shape == (3, 4)


# angle


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ([1, 0, 0], [2, 0, 0], 0.0),
        ([1, 0, 0], [0, 1, 0], math.pi / 2),
        ([1, 0, 0], [-1, 0, 0], math.pi),
        ([1, 0, 0], [1, 1, 0], math.pi / 4),
    ],
)
def test_angle_known(a, b, expected):
    a = np.array([a], dtype="float64")
    b = np.array([b], dtype="float64")
    assert_allclose(angle(a, b), [expected], atol=1e-3)


def test_angle_batch_shape():
    a = np.ones((5, 3))
    result = angle(a, a)
    assert result.shape == (5,)
    assert_allclose(result, 0.0, atol=1e-3)
