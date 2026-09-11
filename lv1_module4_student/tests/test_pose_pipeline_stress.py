import numpy as np
import pytest

from your_package.pose_pipeline import PosePipeline
from your_package.rotation import rot_x, rot_y, rot_z
from your_package.transform import make_T


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def random_rotation(rng):
    """Generate a random proper rotation matrix."""
    angles = rng.uniform(-np.pi, np.pi, size=3)

    R = (
        rot_z(angles[2])
        @ rot_y(angles[1])
        @ rot_x(angles[0])
    )

    return R


def random_transform(rng):
    """Generate a random rigid homogeneous transform."""
    R = random_rotation(rng)
    t = rng.uniform(-10.0, 10.0, size=3)

    return make_T(R, t)


def random_points(rng, n):
    """Generate N random 3D points."""
    return rng.uniform(-10.0, 10.0, size=(n, 3))


# ---------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------

def test_constructor_accepts_valid_transforms():
    T_base_link = np.eye(4)
    T_link_camera = np.eye(4)

    pipe = PosePipeline(T_base_link, T_link_camera)

    assert pipe.joint_axis == "z"
    assert pipe.joint_angle == 0.0


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_all_joint_axes(axis):
    pipe = PosePipeline(
        np.eye(4),
        np.eye(4),
        joint_axis=axis,
    )

    assert pipe.joint_axis == axis
    assert pipe.joint_angle == 0.0


@pytest.mark.parametrize(
    "bad_shape",
    [
        (3, 3),
        (4, 3),
        (3, 4),
        (16,),
        (2, 4, 4),
        (0, 0),
    ],
)
def test_invalid_base_transform_shape(bad_shape):
    with pytest.raises(ValueError):
        PosePipeline(
            np.zeros(bad_shape),
            np.eye(4),
        )


@pytest.mark.parametrize(
    "bad_shape",
    [
        (3, 3),
        (4, 3),
        (3, 4),
        (16,),
        (2, 4, 4),
        (0, 0),
    ],
)
def test_invalid_camera_transform_shape(bad_shape):
    with pytest.raises(ValueError):
        PosePipeline(
            np.eye(4),
            np.zeros(bad_shape),
        )


@pytest.mark.parametrize("axis", ["a", "X", "1", "", None, 1])
def test_invalid_joint_axis(axis):
    with pytest.raises(ValueError):
        PosePipeline(
            np.eye(4),
            np.eye(4),
            joint_axis=axis,
        )


# ---------------------------------------------------------------------
# T_base_link behavior
# ---------------------------------------------------------------------

@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize(
    "theta",
    [
        0.0,
        0.1,
        -0.1,
        np.pi / 2,
        -np.pi / 2,
        np.pi,
        -np.pi,
        10.0,
        -10.0,
        100.0,
    ],
)
def test_joint_rotation(axis, theta):
    rng = np.random.default_rng(123)

    T0 = random_transform(rng)

    pipe = PosePipeline(
        T0,
        np.eye(4),
        joint_axis=axis,
    )

    pipe.set_joint_angle(theta)

    expected = T0 @ make_T(
        {
            "x": rot_x,
            "y": rot_y,
            "z": rot_z,
        }[axis](theta),
        [0.0, 0.0, 0.0],
    )

    np.testing.assert_allclose(
        pipe.T_base_link,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_zero_joint_angle_returns_original_transform():
    rng = np.random.default_rng(1)

    T0 = random_transform(rng)

    pipe = PosePipeline(T0, np.eye(4))

    np.testing.assert_allclose(
        pipe.T_base_link,
        T0,
        rtol=1e-12,
        atol=1e-12,
    )


# ---------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------

def test_base_camera_composition():
    rng = np.random.default_rng(2)

    T_base_link = random_transform(rng)
    T_link_camera = random_transform(rng)

    pipe = PosePipeline(
        T_base_link,
        T_link_camera,
    )

    expected = T_base_link @ T_link_camera

    np.testing.assert_allclose(
        pipe.T_base_camera,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


# ---------------------------------------------------------------------
# Inverse consistency
# ---------------------------------------------------------------------

def test_base_camera_inverse():
    rng = np.random.default_rng(3)

    pipe = PosePipeline(
        random_transform(rng),
        random_transform(rng),
    )

    T = pipe.T_base_camera
    T_inv = pipe.T_camera_base

    expected = np.eye(4)

    np.testing.assert_allclose(
        T @ T_inv,
        expected,
        rtol=1e-10,
        atol=1e-10,
    )

    np.testing.assert_allclose(
        T_inv @ T,
        expected,
        rtol=1e-10,
        atol=1e-10,
    )


# ---------------------------------------------------------------------
# Point-cloud round trip
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "n",
    [
        1,
        2,
        3,
        10,
        100,
        1000,
        10000,
    ],
)
def test_point_cloud_round_trip(n):
    rng = np.random.default_rng(4)

    pipe = PosePipeline(
        random_transform(rng),
        random_transform(rng),
    )

    P_cam = random_points(rng, n)

    P_base = pipe.camera_to_base(P_cam)
    P_cam_recovered = pipe.base_to_camera(P_base)

    np.testing.assert_allclose(
        P_cam_recovered,
        P_cam,
        rtol=1e-9,
        atol=1e-9,
    )


def test_single_point_round_trip():
    rng = np.random.default_rng(5)

    pipe = PosePipeline(
        random_transform(rng),
        random_transform(rng),
    )

    P_cam = rng.uniform(-10, 10, size=3)

    P_base = pipe.camera_to_base(P_cam)
    P_cam_recovered = pipe.base_to_camera(P_base)

    np.testing.assert_allclose(
        P_cam_recovered,
        P_cam,
        rtol=1e-9,
        atol=1e-9,
    )


# ---------------------------------------------------------------------
# Joint motion should change the transformed points
# ---------------------------------------------------------------------

@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_joint_motion_changes_pose(axis):
    rng = np.random.default_rng(6)

    pipe = PosePipeline(
        random_transform(rng),
        random_transform(rng),
        joint_axis=axis,
    )

    P_cam = random_points(rng, 100)

    pipe.set_joint_angle(0.0)
    P0 = pipe.camera_to_base(P_cam)

    pipe.set_joint_angle(np.pi / 2)
    P1 = pipe.camera_to_base(P_cam)

    assert not np.allclose(P0, P1)


# ---------------------------------------------------------------------
# set_joint_angle
# ---------------------------------------------------------------------

def test_set_joint_angle_returns_self():
    pipe = PosePipeline(
        np.eye(4),
        np.eye(4),
    )

    result = pipe.set_joint_angle(1.234)

    assert result is pipe
    assert pipe.joint_angle == pytest.approx(1.234)


def test_set_joint_angle_casts_to_float():
    pipe = PosePipeline(
        np.eye(4),
        np.eye(4),
    )

    pipe.set_joint_angle(np.float32(1.5))

    assert isinstance(pipe.joint_angle, float)
    assert pipe.joint_angle == pytest.approx(1.5)


# ---------------------------------------------------------------------
# object_pose_in_base
# ---------------------------------------------------------------------

def test_object_pose_composition():
    rng = np.random.default_rng(7)

    pipe = PosePipeline(
        random_transform(rng),
        random_transform(rng),
    )

    T_camera_object = random_transform(rng)

    expected = pipe.T_base_camera @ T_camera_object

    actual = pipe.object_pose_in_base(T_camera_object)

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "bad_shape",
    [
        (3, 3),
        (4, 3),
        (3, 4),
        (16,),
        (2, 4, 4),
    ],
)
def test_object_pose_rejects_invalid_shape(bad_shape):
    pipe = PosePipeline(
        np.eye(4),
        np.eye(4),
    )

    with pytest.raises(ValueError):
        pipe.object_pose_in_base(np.zeros(bad_shape))


# ---------------------------------------------------------------------
# Immutability / defensive copying
# ---------------------------------------------------------------------

def test_constructor_copies_input_matrices():
    T_base_link = np.eye(4)
    T_link_camera = np.eye(4)

    pipe = PosePipeline(
        T_base_link,
        T_link_camera,
    )

    T_base_link[0, 3] = 999.0
    T_link_camera[1, 3] = 999.0

    assert pipe.T_base_link[0, 3] != 999.0
    assert pipe.T_link_camera[1, 3] != 999.0


def test_T_link_camera_returns_copy():
    T = np.eye(4)

    pipe = PosePipeline(
        np.eye(4),
        T,
    )

    returned = pipe.T_link_camera

    returned[0, 3] = 999.0

    assert pipe.T_link_camera[0, 3] != 999.0


# ---------------------------------------------------------------------
# Repeated random stress test
# ---------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(20))
def test_randomized_pipeline(seed):
    rng = np.random.default_rng(seed)

    axis = rng.choice(["x", "y", "z"])
    theta = rng.uniform(-4 * np.pi, 4 * np.pi)

    T_base_link = random_transform(rng)
    T_link_camera = random_transform(rng)

    pipe = PosePipeline(
        T_base_link,
        T_link_camera,
        joint_axis=axis,
    )

    pipe.set_joint_angle(theta)

    P_cam = random_points(
        rng,
        rng.integers(1, 500),
    )

    P_base = pipe.camera_to_base(P_cam)
    P_cam_recovered = pipe.base_to_camera(P_base)

    np.testing.assert_allclose(
        P_cam_recovered,
        P_cam,
        rtol=1e-8,
        atol=1e-8,
    )


# ---------------------------------------------------------------------
# Repeated joint updates
# ---------------------------------------------------------------------

def test_repeated_joint_updates():
    rng = np.random.default_rng(100)

    T0 = random_transform(rng)
    Tc = random_transform(rng)

    pipe = PosePipeline(
        T0,
        Tc,
        joint_axis="z",
    )

    for theta in np.linspace(-10 * np.pi, 10 * np.pi, 101):
        pipe.set_joint_angle(theta)

        P_cam = random_points(rng, 20)

        P_base = pipe.camera_to_base(P_cam)
        P_recovered = pipe.base_to_camera(P_base)

        np.testing.assert_allclose(
            P_recovered,
            P_cam,
            rtol=1e-8,
            atol=1e-8,
        )


# ---------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------

def test_repr():
    pipe = PosePipeline(
        np.eye(4),
        np.eye(4),
        joint_axis="y",
    )

    pipe.set_joint_angle(1.23456)

    text = repr(pipe)

    assert "PosePipeline" in text
    assert "joint_axis='y'" in text
    assert "1.2346" in text