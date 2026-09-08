"""문제 3 — 회전 행렬의 수학적 성질 검증 (pytest). [학생 작성용 템플릿]

지시문이 요구하는 4가지를 각각 테스트 함수로 작성한다.

  1. 회전행렬의 열이 서로 직교하는 단위벡터인가   -> test_columns_are_orthonormal
  2. 행렬식이 1인가                               -> test_determinant_is_one
  3. 역행렬이 전치와 같은가                       -> test_inverse_equals_transpose
  4. 재직교화 결과가 직교행렬인가                 -> test_gram_schmidt_restores_orthogonality

작성 요령
--------
- `@pytest.mark.parametrize` 로 여러 축 x 여러 각도를 한 함수에서 검사하면
  테스트 하나가 여러 케이스를 담당한다 (아래 ANGLES / MAKERS 참고).
- 비교는 반드시 `np.isclose` / `np.allclose` 로 한다 (부동소수점).
- `np.linalg` 는 검산용으로만 쓰고, 쓸 때는 주석으로 검산임을 밝힌다.
- assert 에 실패 메시지를 붙이면 어디가 깨졌는지 바로 보인다.
- 4개는 **최소 개수**다. 반사 행렬 반례, 로드리게스 일치, 축·각 왕복 같은
  테스트를 더 붙이면 좋다.

실행: 프로젝트 루트에서  pytest -v
"""

import numpy as np
import pytest

from src.rotation import (
    axis_angle_from_matrix,
    gram_schmidt,
    is_rotation,
    orthogonality_error,
    rodrigues,
    rot_x,
    rot_y,
    rot_z,
)

ANGLES = [0.0, np.deg2rad(22.5), np.pi / 6, np.pi / 4, np.pi / 2, 2.0, np.pi, -1.234]
MAKERS = [rot_x, rot_y, rot_z]


@pytest.fixture
def rng():
    """난수는 반드시 시드를 고정한다."""
    return np.random.default_rng(42)


# --- 1. 열이 서로 직교하는 단위벡터인가 -------------------------------------

@pytest.mark.parametrize("maker", MAKERS)
@pytest.mark.parametrize("theta", ANGLES)
def test_columns_are_orthonormal(maker, theta):
    R = maker(theta)

    for i in range(3):
        assert np.isclose(
            np.linalg.norm(R[:, i]), 1.0
        ), f"column {i} is not a unit vector"

    for i in range(3):
        for j in range(i + 1, 3):
            assert np.isclose(
                np.dot(R[:, i], R[:, j]), 0.0
            ), f"columns {i} and {j} are not orthogonal"


# --- 2. 행렬식이 1인가 --------------------------------------------------------

@pytest.mark.parametrize("maker", MAKERS)
@pytest.mark.parametrize("theta", ANGLES)
def test_determinant_is_one(maker, theta):
    R = maker(theta)

    # np.linalg.det is used here only as a verification oracle.
    assert np.isclose(
        np.linalg.det(R), 1.0
    ), f"determinant is not 1: det(R)={np.linalg.det(R)}"


# --- 3. 역행렬 == 전치 --------------------------------------------------------

@pytest.mark.parametrize("maker", MAKERS)
@pytest.mark.parametrize("theta", ANGLES)
def test_inverse_equals_transpose(maker, theta):
    R = maker(theta)
    I = np.eye(3)

    # np.linalg.inv is used here only as a verification oracle.
    assert np.allclose(
        np.linalg.inv(R), R.T
    ), "the inverse of a rotation matrix must equal its transpose"
    assert np.allclose(
        R.T @ R, I
    ), "R.T @ R must be the identity matrix"


# --- 4. 재직교화 결과가 직교행렬인가 -----------------------------------------

def test_gram_schmidt_restores_orthogonality(rng):
    R = rot_z(0.4) @ rot_y(-0.7) @ rot_x(0.2)
    noisy = R + 1e-3 * rng.standard_normal((3, 3))
    fixed = gram_schmidt(noisy)

    assert orthogonality_error(fixed) < 1e-14, (
        "Gram-Schmidt did not restore orthogonality: "
        f"error={orthogonality_error(fixed)}"
    )
    assert np.isclose(np.linalg.det(fixed), 1.0), (
        f"reorthogonalized matrix has the wrong determinant: "
        f"det={np.linalg.det(fixed)}"
    )
    assert is_rotation(fixed), "reorthogonalized matrix is not a rotation"


# --- 여기부터는 추가 테스트 (권장) -------------------------------------------
#
# 예) def test_reflection_is_not_a_rotation():
#         """det = -1 인 반사 행렬은 직교여도 회전이 아니다."""
#
# 예) def test_rodrigues_matches_rot_z(theta): ...
# 예) def test_axis_angle_roundtrip(rng): ...
