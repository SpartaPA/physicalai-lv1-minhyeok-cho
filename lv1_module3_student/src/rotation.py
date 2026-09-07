"""문제 2·3 — 회전 행렬 모듈. (학생 작성용 템플릿)

축별 회전 행렬, 로드리게스 공식(임의 축 회전), Gram-Schmidt 재직교화,
회전행렬 판정과 고유값 분해 기반 축·각 복원을 직접 구현한다.

문제 1 에서 만든 `src/vectors.py` 를 그대로 재사용한다.
"""

from __future__ import annotations

import numpy as np

from .vectors import det, normalize, skew, project

__all__ = [
    "rot_x",
    "rot_y",
    "rot_z",
    "rodrigues",
    "gram_schmidt",
    "orthogonality_error",
    "is_rotation",
    "axis_angle_from_matrix",
    "quaternion_from_axis_angle",
]


# ------------------------------------------------------------ 축별 회전 행렬

def rot_x(theta: float) -> np.ndarray:
    """x축 기준 회전 행렬 (theta 는 **라디안**). x 성분은 보존된다."""
    # TODO: 문제 2-1
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0],
                    [0.0,   c,  -s],
                    [0.0,   s,   c]], dtype=float)


def rot_y(theta: float) -> np.ndarray:
    """y축 기준 회전 행렬 (theta 는 라디안). y 성분은 보존된다.

    부호 배치가 x·z 와 반대로 보이는 이유는 노트북 2-1 에서 설명한다.
    """
    # TODO: 문제 2-1
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c,    0.0, s],
                    [0.0,   1.0, 0],
                    [-s,    0.0, c]], dtype=float)


def rot_z(theta: float) -> np.ndarray:
    """z축 기준 회전 행렬 (theta 는 라디안). z 성분은 보존된다."""
    # TODO: 문제 2-1
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0],
                    [s,     c, 0.0],
                    [0.0, 0.0, 1.0]], dtype=float)


def rodrigues(axis, theta: float) -> np.ndarray:
    """로드리게스 공식으로 임의 축 회전 행렬을 만든다.

        R = I + sin(theta) * K + (1 - cos(theta)) * K @ K,   K = [k]_x

    - 축은 함수 안에서 단위벡터로 정규화한다
      (정규화되지 않은 축을 넣어도 같은 결과가 나와야 한다).
    - 문제 1 의 `skew` 를 반드시 사용한다.
    """
    # TODO: 문제 2-5
    k = np.asarray(axis, dtype=float).reshape(-1)
    if k.shape != (3,):
        raise ValueError(f"3 elements required. Input shape={k.shape}")

    I = np.eye(3)
    k = normalize(k)
    k_skew = skew(k)
    # b = np.array([[np.cos(theta)], [1 - np.cos(theta)], [np.sin(theta)]])
    return I + (k_skew @ k_skew) * (1 - np.cos(theta)) + k_skew * np.sin(theta)


# ------------------------------------------------------------- 재직교화 관련

def gram_schmidt(A) -> np.ndarray:
    """**열벡터**에 대해 Gram-Schmidt 직교정규화를 수행한다.

        q1 = a1 / |a1|
        vj = aj - sum_{i<j} (qi · aj) qi
        qj = vj / |vj|

    각 열에서 앞선 열 방향 성분(정사영)을 빼고 정규화하는 것이며,
    문제 1 의 project / reject 와 같은 연산의 반복이다.

    수치적으로는 성분을 빼자마자 갱신하는 modified Gram-Schmidt 가 더 안정적이다.
    앞선 열들에 종속인 열이 있으면 ValueError.
    """
    # TODO: 문제 3-2
    R = np.asarray(A, dtype=float)

    Q = np.zeros_like(R)

    for i in range(R.shape[1]):
        v = R[:, i].copy()

        for j in range(i):
            v -= project(v, Q[:, j])

        Q[:, i] = normalize(v)

    return Q

def orthogonality_error(R) -> float:
    """직교성 이탈 지표: || R^T R - I ||_F  (프로베니우스 노름).

    완전한 직교행렬이면 0 이고, 클수록 직교성이 무너진 것이다.
    """
    # TODO: 문제 3-1
    R = np.asarray(R, dtype=float)
    E = R.T @ R - np.eye(R.shape[0])
    return(np.sqrt(np.sum(E * E)))

def is_rotation(R, atol: float = 1e-8) -> bool:
    """회전행렬 판정: 직교(R^T R = I) **그리고** det(R) = +1 이면 True.

    det = -1 이면 직교이긴 하지만 반사가 섞여 있어 회전이 아니다.
    3x3 이 아니면 False.
    """
    # TODO: 문제 3-2
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        return False
    
    return np.allclose(R.T @ R, np.eye(R.shape[0]),atol=1.7e-8) and np.isclose(det(R), 1.0, atol=1.7e-8)

# --------------------------------------------------- 회전축·회전각·쿼터니언

def axis_angle_from_matrix(R, atol: float = 1e-8):
    """고유값 분해로 회전축을, 대각합으로 회전각을 복원한다.

    - 회전축은 고유값 1 에 대응하는 실수 고유벡터다 (R k = k).
      -> 여기서는 `np.linalg.eig` 를 써도 된다 (검산이 아니라 축 복원이 목적).
    - 회전각은 trace(R) = 1 + 2 cos(theta) 에서 구한다.
    - arccos 의 치역이 [0, pi] 라 '어느 쪽으로 도는지'는 알 수 없고,
      고유벡터도 부호가 정해지지 않는다. 반대칭 성분
      R - R^T = 2 sin(theta) [k]_x 를 이용해 부호를 맞춘다.
    - theta = 0 (회전 없음) 과 theta = pi (sin = 0) 는 따로 처리해야 한다.
      두 경우에 어떤 규약을 쓸지 정하고 주석으로 남긴다.

    Returns
    -------
    axis : 단위 회전축 (3,)
    angle : 회전각 [rad], 0 <= angle <= pi
    """
    # TODO: 문제 6-4
    R = np.asarray(R, dtype=float)

    if R.shape != (3, 3):
        raise ValueError("R must be a 3x3 matrix.")

    # cos(theta) = (trace(R) - 1) / 2
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    angle = np.arccos(cos_theta)

    # --------------------------------------------------
    # theta = 0
    # --------------------------------------------------
    if np.isclose(angle, 0.0, atol=atol):
        return np.array([1.0, 0.0, 0.0]), 0.0

    # --------------------------------------------------
    # theta = pi
    # --------------------------------------------------
    if np.isclose(angle, np.pi, atol=atol):

        A = (R + np.eye(3)) / 2.0

        axis = np.sqrt(np.maximum(np.diag(A), 0.0))

        # Recover signs
        if axis[0] > atol:
            axis[1] = np.copysign(
                axis[1],
                R[0, 1] + R[1, 0]
            )
            axis[2] = np.copysign(
                axis[2],
                R[0, 2] + R[2, 0]
            )

        elif axis[1] > atol:
            axis[0] = np.copysign(
                axis[0],
                R[0, 1] + R[1, 0]
            )
            axis[2] = np.copysign(
                axis[2],
                R[1, 2] + R[2, 1]
            )

        else:
            axis[0] = np.copysign(
                axis[0],
                R[0, 2] + R[2, 0]
            )
            axis[1] = np.copysign(
                axis[1],
                R[1, 2] + R[2, 1]
            )

        axis = normalize(axis, atol)

        return axis, angle

    # --------------------------------------------------
    # General case: 0 < theta < pi
    # --------------------------------------------------

    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ])

    axis /= 2.0 * np.sin(angle)

    axis = normalize(axis, atol)

    return axis, angle

def quaternion_from_axis_angle(axis, angle: float) -> np.ndarray:
    """축-각에서 단위 쿼터니언을 만든다.

        q = (k * sin(theta/2), cos(theta/2))

    반환 순서는 SciPy `Rotation.as_quat()` 와 같은 **(x, y, z, w)** 로 맞춘다
    (그래야 문제 6-5 에서 바로 비교할 수 있다).
    """
    # TODO: 문제 6-5
    axis = np.asarray(axis, dtype=float).reshape(-1)

    if axis.shape != (3,):
        raise ValueError(
            f"Axis must be a 3-element vector. Input shape={axis.shape}"
        )

    axis = normalize(axis)

    half_angle = angle / 2.0

    w = np.cos(half_angle)
    xyz = axis * np.sin(half_angle)

    q = np.concatenate(([w], xyz))

    # Optional numerical cleanup
    q = normalize(q)

    return q