"""문제 5 — 동차변환 inv_T 검증 (pytest). [학생 작성용 템플릿]

지시문이 요구하는 것은 `inv_T` 검증이지만,
점/방향 구분과 벡터화, 최소자승까지 함께 검증해 두면 이후 문제에서 안전하다.

실행: 프로젝트 루트에서  pytest -v
"""

import numpy as np
import pytest

from src.rotation import rot_x, rot_y, rot_z
from src.transform import (
    inv_T,
    least_squares_normal_equation,
    make_T,
    transform_direction,
    transform_point,
    transform_points,
)


@pytest.fixture
def T():
    """테스트에 쓸 대표 동차변환 하나."""
    R = rot_z(0.9) @ rot_y(-0.35) @ rot_x(1.3)
    return make_T(R, [0.35, -0.15, 0.55])


def test_inv_T_gives_identity(T):
    T_inv = inv_T(T)
    assert np.allclose(T_inv @ T, np.eye(4))
    assert np.allclose(T @ T_inv, np.eye(4))


def test_inv_T_matches_generic_inverse(T):
    # np.linalg.inv is used here only as a verification oracle.
    assert np.allclose(inv_T(T), np.linalg.inv(T))


def test_point_and_direction_differ(T):
    v = np.array([0.2, -0.6, 0.4])
    point = transform_point(T, v)
    direction = transform_direction(T, v)

    assert not np.allclose(point, direction)
    assert np.allclose(point - direction, T[:3, 3])
    assert np.isclose(np.linalg.norm(direction), np.linalg.norm(v))


def test_transform_points_is_vectorized(T):
    P = np.random.default_rng(42).uniform(-1.0, 1.0, size=(100, 3))
    actual = transform_points(T, P)
    expected = np.array([transform_point(T, p) for p in P])
    assert actual.shape == P.shape
    assert np.allclose(actual, expected)


def test_roundtrip_through_inverse(T):
    P = np.random.default_rng(42).uniform(-1.0, 1.0, size=(100, 3))
    assert np.allclose(transform_points(inv_T(T), transform_points(T, P)), P)


def test_least_squares_matches_lstsq():
    rng = np.random.default_rng(42)
    A = rng.normal(size=(36, 12))
    x_true = rng.normal(size=12)
    b = A @ x_true + 1e-3 * rng.normal(size=36)

    x, residual = least_squares_normal_equation(A, b)
    # np.linalg.lstsq is used here only as a verification oracle.
    x_ref, *_ = np.linalg.lstsq(A, b, rcond=None)

    assert np.allclose(x, x_ref, atol=1e-10)
    assert np.allclose(A.T @ residual, np.zeros(12), atol=1e-10)
