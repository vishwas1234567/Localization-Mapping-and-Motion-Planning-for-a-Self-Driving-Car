"""Planar (SE(2)) geometry helpers used throughout the project.

Poses are numpy arrays ``[x, y, theta]``.  ``compose(a, b)`` is the group
product  ``a (+) b``  (pose ``b`` expressed in frame ``a`` -> world frame).
"""
from __future__ import annotations

import numpy as np


def wrap(a):
    """Wrap angle(s) to (-pi, pi]."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def rot(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def compose(a, b) -> np.ndarray:
    """a (+) b."""
    c, s = np.cos(a[2]), np.sin(a[2])
    return np.array([a[0] + c * b[0] - s * b[1],
                     a[1] + s * b[0] + c * b[1],
                     wrap(a[2] + b[2])])


def inverse(a) -> np.ndarray:
    c, s = np.cos(a[2]), np.sin(a[2])
    return np.array([-(c * a[0] + s * a[1]),
                     s * a[0] - c * a[1],
                     wrap(-a[2])])


def between(a, b) -> np.ndarray:
    """Pose of ``b`` expressed in the frame of ``a``  (a^-1 (+) b)."""
    return compose(inverse(a), b)


def transform_points(pose, pts: np.ndarray) -> np.ndarray:
    """Map (N,2) points from the frame of ``pose`` into the parent frame."""
    c, s = np.cos(pose[2]), np.sin(pose[2])
    out = np.empty_like(pts, dtype=float)
    out[:, 0] = c * pts[:, 0] - s * pts[:, 1] + pose[0]
    out[:, 1] = s * pts[:, 0] + c * pts[:, 1] + pose[1]
    return out


def voxel_downsample(pts: np.ndarray, res: float) -> np.ndarray:
    """Keep one (averaged) point per voxel of size ``res``."""
    if len(pts) == 0:
        return pts
    ij = np.floor(pts / res).astype(np.int64)
    key = ij[:, 0] * 1_000_003 + ij[:, 1]
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    sx = np.bincount(inv, pts[:, 0])
    sy = np.bincount(inv, pts[:, 1])
    return np.stack([sx / cnt, sy / cnt], axis=1)


def pose_errors(est: np.ndarray, gt: np.ndarray):
    """Longitudinal / lateral / heading error of est w.r.t. gt, in the gt frame."""
    d = est[:, :2] - gt[:, :2]
    c, s = np.cos(gt[:, 2]), np.sin(gt[:, 2])
    lon = c * d[:, 0] + s * d[:, 1]
    lat = -s * d[:, 0] + c * d[:, 1]
    yaw = wrap(est[:, 2] - gt[:, 2])
    return lon, lat, yaw


def rmse(x) -> float:
    x = np.asarray(x)
    return float(np.sqrt(np.mean(x ** 2)))
