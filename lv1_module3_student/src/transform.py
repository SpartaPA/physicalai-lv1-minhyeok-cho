"""문제 5 — 4x4 동차변환 모듈. (학생 작성용 템플릿)

동차변환 생성/역변환, 점과 방향의 구분, 벡터화된 점군 변환,
정규방정식 기반 최소자승법을 직접 구현한다.
"""

from __future__ import annotations

import numpy as np

from .vectors import as_vector, det, inverse_gauss_jordan

__all__ = [
    "make_T",
    "inv_T",
    "inv_T_batch",
    "to_homogeneous",
    "transform_point",
    "transform_direction",
    "transform_points",
    "least_squares_normal_equation",
    "rmse",
]


def make_T(R, t) -> np.ndarray:
    """Return a 4x4 homogeneous transformation matrix."""

    R = np.asarray(R, dtype=float)
    t = as_vector(t)

    if R.shape != (3, 3):
        raise ValueError(
            f"3x3 matrix for R is required. "
            f"Input shape={R.shape}."
        )

    if len(t) != 3:
        raise ValueError(
            f"3-element vector for t is required. "
            f"Input shape={t.shape}."
        )

    M = np.eye(4)

    M[:3, :3] = R
    M[:3, 3] = t

    return M

def inv_T(M, eps=1e-12) -> np.ndarray:
    """Return inverse of a 4x4 homogeneous transformation matrix M."""

    R, t = _validate_transform(M, eps)

    inv_M = np.eye(4)
    inv_M[:3, :3] = R.T
    inv_M[:3, 3] = -R.T @ t
    return inv_M


def to_homogeneous(t, w=1.0) -> np.ndarray:
    """Return [N x 4] matrix where t is [N x 3] matrix and w is either 0 or 1."""

    t = np.asarray(t, dtype=float)

    if t.size == 3:
        t = t.reshape(1, -1)
    elif t.ndim == 2:
        if t.shape[1] != 3:
            raise ValueError(f"Nx3 matrix required. Input shape = {t.shape}.")
    else:
        raise ValueError(f"2D matrix required. Input dimension = {t.ndim}.")


    w = float(w)
    if w not in (0.0, 1.0):
        raise ValueError(f"w shall be either 1 or 0. Input w = {w}.")

    w = np.full((t.shape[0], 1), w)

    return np.concatenate((t, w), axis=1)


def _validate_transform(M, eps=1e-12):
    M = np.asarray(M, dtype=float)
    if M.shape != (4, 4):
        raise ValueError(f"4x4 matrix required. Input shape = {M.shape}.")
    if not np.allclose(M[3, :], [0., 0., 0., 1.], atol=eps, rtol=0.0):
        raise ValueError(f"Homogeneous matrix required. Bottom row: {M[3,:]}")
    R = M[:3, :3]
    if not (
        np.allclose(R.T @ R, np.eye(3), atol=eps, rtol=0.0)
        and np.isclose(det(R), 1.0, atol=eps, rtol=0.0)
    ):
        raise ValueError(f"Proper rotation matrix required. R:\n{R}")
    return R, M[:3, 3]


def transform_point(M, p, eps=1e-12) -> np.ndarray:
    """Return point p transformed by a 4x4 homogeneous transformation."""

    R, t = _validate_transform(M, eps)

    p = as_vector(p)

    if len(p) != 3:
        raise ValueError(
            f"Require p of 3 elements. "
            f"Input shape = {p.shape}."
        )

    return R @ p + t


def transform_direction(M, p, eps=1e-12) -> np.ndarray:
    """Return direction p transformed by a 4x4 homogeneous transformation."""

    R, _ = _validate_transform(M, eps)

    p = as_vector(p)

    if len(p) != 3:
        raise ValueError(
            f"Require p of 3 elements. "
            f"Input shape = {p.shape}."
        )

    return R @ p


def transform_points(M, P, w=1.0, eps=1e-12) -> np.ndarray:
    """Transform a point or point cloud using a homogeneous transform.

    ``w=1`` applies rotation and translation; ``w=0`` applies rotation only.
    A single ``(3,)`` input returns ``(3,)`` and an ``(N, 3)`` input returns
    ``(N, 3)``.
    """

    R, t = _validate_transform(M, eps)

    P = np.asarray(P, dtype=float)
    single = False

    if P.ndim == 1 and P.shape == (3,):
        P = P.reshape(1, 3)
        single = True
    elif P.ndim != 2 or P.shape[0] == 0 or P.shape[1] != 3:
        raise ValueError(f"P must have shape (3,) or (N, 3). Input shape = {P.shape}")

    w = float(w)
    if w not in (0.0, 1.0):
        raise ValueError(f"w shall be either 1 or 0. Input w = {w}.")

    result = (R @ P.T + w * t[:, None]).T
    return result[0] if single else result


def inv_T_batch(T, eps=1e-12) -> np.ndarray:
    """Return inverse of homogeneous transformation matrices
    Input: N x (4 x 4) array"""
    T = np.asarray(T, dtype=float)

    # Confirm input
    if T.ndim != 3 or T.shape[1:] != (4, 4):
        raise ValueError(
            f"Input must have shape (N, 4, 4). "
            f"Input shape={T.shape}."
        )

    if not np.allclose(
        T[:,3, :],
        np.array([0., 0., 0., 1.]),
        atol=eps,
        rtol=0.0,
        ):
        raise ValueError(f"Homogeneous matrix required. Current bottom rows: {T[:,3,:]}")

    R = T[:, :3, :3]
    R_T = np.swapaxes(R, 1, 2)

    if R.shape[0] > 0:
        if np.max(np.abs(R_T @ R - np.eye(3))) > eps:
            raise ValueError(f"Proper rotation matrices are required. Rs:\n{R}")

        det_R = (
            R[:, 0, 0] * (R[:, 1, 1] * R[:, 2, 2] - R[:, 1, 2] * R[:, 2, 1])
            - R[:, 0, 1] * (R[:, 1, 0] * R[:, 2, 2] - R[:, 1, 2] * R[:, 2, 0])
            + R[:, 0, 2] * (R[:, 1, 0] * R[:, 2, 1] - R[:, 1, 1] * R[:, 2, 0])
        )
        if np.max(np.abs(det_R - 1.0)) > eps:
            raise ValueError(f"Proper rotation matrices are required. Rs:\n{R}")

    T_result = np.zeros_like(T)

    T_result[:, :3, :3] = R_T
    T_result[:, :3, 3] = -np.einsum("nij,nj->ni", R_T, T[:, :3, 3])
    T_result[:, 3, 3] = 1

    return T_result


def least_squares_normal_equation(A, b) -> tuple[np.ndarray, np.ndarray]:
    """Return the normal-equation solution and residual ``r = b - A @ x``."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    if A.ndim == 2 and b.ndim == 1:
        if A.shape[0] != b.shape[0] or A.shape[1] != 12 or b.shape[0] % 3 != 0:
            raise ValueError(f"A and b shall have shape of (3N, 12) and (3N,), respectively.\n"
                             f"Current input shape of A and b: {A.shape} and {b.shape}")

    else:
        raise ValueError(f"A and b shall be 2D matrix and 1D vector, respectively.\n\
                         Current input dimension of A and b: {A.shape} and {b.shape}")

    normal_matrix = A.T @ A
    x_approx = inverse_gauss_jordan(normal_matrix) @ (A.T @ b)
    r = b - A @ x_approx
    return x_approx, r


def rmse(r) -> float:
    r = np.asarray(r, dtype=float)
    if r.size == 0:
        raise ValueError("Cannot calculate RMSE of an empty residual.")
    return float(np.sqrt(np.mean(r**2)))
