"""Point-to-point 2-D ICP (trimmed, KD-tree) -- the scan-to-scan baseline."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from ..geometry import compose, wrap


def _best_fit(A: np.ndarray, B: np.ndarray):
    """Rigid transform (x,y,theta) minimising |R A + t - B|^2 (closed form)."""
    ca, cb = A.mean(0), B.mean(0)
    Ac, Bc = A - ca, B - cb
    h = Ac.T @ Bc
    th = np.arctan2(h[0, 1] - h[1, 0], h[0, 0] + h[1, 1])
    c, s = np.cos(th), np.sin(th)
    t = cb - np.array([c * ca[0] - s * ca[1], s * ca[0] + c * ca[1]])
    return np.array([t[0], t[1], th])


def icp(src: np.ndarray, dst: np.ndarray, init, tree: cKDTree | None = None, max_iter: int = 40,
        max_dist: float = 2.0, trim: float = 0.85, tol: float = 1e-5):
    """Estimate the pose of the ``src`` frame in the ``dst`` frame.  Returns (pose, rms, n_corr)."""
    tree = tree or cKDTree(dst)
    pose = np.array(init, float)
    rms, n = np.inf, 0
    for _ in range(max_iter):
        c, s = np.cos(pose[2]), np.sin(pose[2])
        pw = np.stack([c * src[:, 0] - s * src[:, 1] + pose[0], s * src[:, 0] + c * src[:, 1] + pose[1]], axis=1)
        d, j = tree.query(pw, distance_upper_bound=max_dist)
        ok = np.isfinite(d)
        if ok.sum() < 10:
            break
        idx = np.where(ok)[0]
        if trim < 1.0:
            keep = idx[np.argsort(d[idx])[: int(trim * len(idx))]]
        else:
            keep = idx
        inc = _best_fit(pw[keep], dst[j[keep]])
        new = compose(inc, pose)
        rms = float(np.sqrt(np.mean(d[keep] ** 2)))
        n = len(keep)
        done = np.linalg.norm(inc[:2]) < tol and abs(inc[2]) < tol
        pose = new
        if done:
            break
    pose[2] = wrap(pose[2])
    return pose, rms, n


def _normals(pts: np.ndarray, k: int = 3) -> np.ndarray:
    """Unit normals of an angularly ordered 2-D scan from neighbouring points (PCA)."""
    n = len(pts)
    nrm = np.zeros((n, 2))
    pad = np.vstack([np.repeat(pts[:1], k, 0), pts, np.repeat(pts[-1:], k, 0)])
    for i in range(n):
        nb = pad[i:i + 2 * k + 1]
        c = nb - nb.mean(0)
        w, v = np.linalg.eigh(c.T @ c)
        nrm[i] = v[:, 0]
    return nrm


def icp_point_to_line(src, dst, init, max_iter: int = 40, max_dist: float = 1.5, trim: float = 0.85,
                      tol: float = 1e-5, dst_normals=None, tree=None):
    """Point-to-line ICP (Censi 2008 style linearisation) - the stronger scan-to-scan baseline."""
    tree = tree or cKDTree(dst)
    nrm = dst_normals if dst_normals is not None else _normals(dst)
    pose = np.array(init, float)
    for _ in range(max_iter):
        c, s = np.cos(pose[2]), np.sin(pose[2])
        pw = np.stack([c * src[:, 0] - s * src[:, 1] + pose[0], s * src[:, 0] + c * src[:, 1] + pose[1]], axis=1)
        d, j = tree.query(pw, distance_upper_bound=max_dist)
        ok = np.where(np.isfinite(d))[0]
        if len(ok) < 10:
            break
        if trim < 1.0:
            ok = ok[np.argsort(d[ok])[: int(trim * len(ok))]]
        nn = nrm[j[ok]]
        r = np.einsum("ij,ij->i", nn, pw[ok] - dst[j[ok]])
        Jt = nn
        Jr = nn[:, 0] * (-pw[ok][:, 1]) + nn[:, 1] * pw[ok][:, 0]     # n . perp(p')
        J = np.column_stack([Jt, Jr])
        delta = -np.linalg.lstsq(J.T @ J + 1e-9 * np.eye(3), J.T @ r, rcond=None)[0]
        # apply the increment in the world frame (rotation about the origin)
        cd, sd = np.cos(delta[2]), np.sin(delta[2])
        pose = np.array([cd * pose[0] - sd * pose[1] + delta[0],
                         sd * pose[0] + cd * pose[1] + delta[1],
                         wrap(pose[2] + delta[2])])
        if np.linalg.norm(delta[:2]) < tol and abs(delta[2]) < tol:
            break
    return pose
