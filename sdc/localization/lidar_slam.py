"""LiDAR SLAM in the spirit of HDL-Graph-SLAM:

  * front end : keyframe-based NDT *scan-to-submap* odometry (coarse-to-fine),
  * loop detection : distance-based candidates, multi-start NDT verification
                     with a KD-tree inlier-ratio test,
  * back end  : SE(2) pose-graph optimisation with a Huber kernel on loop edges.

A LOAM-style curvature feature extractor is provided as well (``extract_loam_features``);
it is used for diagnostics / visualisation of edge and planar points.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from ..geometry import between, compose, transform_points, voxel_downsample
from .icp import icp
from .ndt import NDTGrid, ndt_register_multires
from .pose_graph import Edge, optimize


# --------------------------------------------------------------------------
@dataclass
class Keyframe:
    idx: int
    pose: np.ndarray
    pts: np.ndarray
    dist: float


@dataclass
class SlamConfig:
    cells: tuple = (3.0, 1.5)
    kf_dist: float = 1.5
    kf_rot: float = 0.15
    submap_kfs: int = 20
    voxel: float = 0.25
    loop: bool = True
    loop_min_travel: float = 80.0
    loop_radius: float = 20.0
    loop_every: int = 3
    loop_inlier_thr: float = 0.65
    odom_sigma: tuple = (0.05, 0.05, 0.01)
    loop_sigma: tuple = (0.15, 0.15, 0.02)
    optimize_every: int = 10


class LidarSlam:
    def __init__(self, init_pose, cfg: SlamConfig = SlamConfig()):
        self.cfg = cfg
        self.kfs: list[Keyframe] = []
        self.edges: list[Edge] = []
        self.cur = np.array(init_pose, float)
        self.delta = np.zeros(3)
        self._grids = None
        self.scan_log: list[dict] = []
        self.n_loops = 0
        self._since_opt = 0
        self._kf_since_loop = 0
        self.loop_pairs: list[tuple[int, int]] = []

    # ------------------------------------------------------------------
    def _world_points(self, ids):
        pts = [transform_points(self.kfs[k].pose, self.kfs[k].pts) for k in ids]
        return voxel_downsample(np.concatenate(pts, axis=0), self.cfg.voxel)

    def _rebuild_submap(self):
        lo = max(0, len(self.kfs) - self.cfg.submap_kfs)
        wp = self._world_points(range(lo, len(self.kfs)))
        self._grids = [NDTGrid(wp, c, min_pts=3) for c in self.cfg.cells]

    def _add_keyframe(self, idx, pts, dist):
        prev = self.kfs[-1] if self.kfs else None
        self.kfs.append(Keyframe(idx, self.cur.copy(), pts, dist))
        if prev is not None:
            z = between(prev.pose, self.cur)
            om = np.diag(1.0 / np.array(self.cfg.odom_sigma) ** 2)
            self.edges.append(Edge(len(self.kfs) - 2, len(self.kfs) - 1, z, om, "odom"))
        self._rebuild_submap()

    # ------------------------------------------------------------------
    def process(self, pts: np.ndarray, idx: int):
        if not self.kfs:
            self._add_keyframe(idx, pts, 0.0)
            return self.cur
        guess = compose(self.cur, self.delta)
        res = ndt_register_multires(pts, self._grids, guess)
        self.delta = between(self.cur, res.pose)
        self.cur = res.pose
        self.scan_log.append(dict(idx=idx, pose=res.pose.copy(), deg=res.degeneracy(),
                                  fit=res.fitness, nm=res.n_matched))
        last = self.kfs[-1]
        d = between(last.pose, self.cur)
        if np.hypot(d[0], d[1]) > self.cfg.kf_dist or abs(d[2]) > self.cfg.kf_rot:
            self._add_keyframe(idx, pts, last.dist + float(np.hypot(d[0], d[1])))
            self._kf_since_loop += 1
            if self.cfg.loop and self._kf_since_loop >= self.cfg.loop_every:
                self._try_loop()
        return self.cur

    # ------------------------------------------------------------------
    def _try_loop(self):
        i = len(self.kfs) - 1
        kf = self.kfs[i]
        pos = np.array([k.pose[:2] for k in self.kfs[:i]])
        trav = kf.dist - np.array([k.dist for k in self.kfs[:i]])
        dist = np.hypot(*(pos - kf.pose[:2]).T)
        cand = np.where((trav > self.cfg.loop_min_travel) & (dist < self.cfg.loop_radius))[0]
        if len(cand) == 0:
            return
        self._kf_since_loop = 0
        j = int(cand[np.argmin(dist[cand])])
        ids = range(max(0, j - 6), min(i - 1, j + 6) + 1)
        wp = self._world_points(ids)
        tree = cKDTree(wp)
        grids = [NDTGrid(wp, c, min_pts=3) for c in self.cfg.cells]
        best, best_inl = None, 0.0
        for dyaw in (0.0, 0.08, -0.08):
            for dx, dy in ((0, 0), (2, 0), (-2, 0), (0, 2), (0, -2)):
                init = kf.pose + np.array([dx, dy, dyaw])
                r = ndt_register_multires(kf.pts, grids, init)
                pw = transform_points(r.pose, kf.pts)
                dd, _ = tree.query(pw, distance_upper_bound=0.3)
                inl = float(np.mean(np.isfinite(dd)))
                if inl > best_inl:
                    best, best_inl = r, inl
        if best is None or best_inl < self.cfg.loop_inlier_thr:
            return
        z = between(self.kfs[j].pose, best.pose)
        om = np.diag(1.0 / np.array(self.cfg.loop_sigma) ** 2)
        self.edges.append(Edge(j, i, z, om, "loop"))
        self.loop_pairs.append((j, i))
        self.n_loops += 1
        self._since_opt += 1
        if self._since_opt >= 1 and (self.n_loops == 1 or self.n_loops % self.cfg.optimize_every == 0):
            self.optimize()

    def optimize(self):
        poses = np.array([k.pose for k in self.kfs])
        new, chi2 = optimize(poses, self.edges)
        for k, p in zip(self.kfs, new):
            k.pose = p
        self.cur = self.kfs[-1].pose.copy()
        self._since_opt = 0
        self._rebuild_submap()
        return chi2

    # ------------------------------------------------------------------
    def keyframe_poses(self):
        return np.array([k.pose for k in self.kfs]), np.array([k.idx for k in self.kfs])

    def global_map(self, voxel: float = 0.15):
        return voxel_downsample(self._world_points(range(len(self.kfs))), voxel)


# --------------------------------------------------------------------------
def icp_odometry(scans, init_pose, priors=None):
    """Scan-to-scan point-to-line ICP odometry (open loop) - the baseline.

    ``priors[k-1]`` (optional) is an initial guess for the relative pose between
    scan k-1 and k (e.g. from visual odometry); otherwise constant velocity is used.
    """
    from .icp import icp_point_to_line, _normals
    from scipy.spatial import cKDTree
    poses = [np.array(init_pose, float)]
    delta = np.zeros(3)
    for k in range(1, len(scans)):
        init = delta if priors is None else priors[k - 1]
        rel = icp_point_to_line(scans[k], scans[k - 1], init, dst_normals=_normals(scans[k - 1]),
                                tree=cKDTree(scans[k - 1]))
        delta = rel
        poses.append(compose(poses[-1], rel))
    return np.array(poses)


def extract_loam_features(pts: np.ndarray, k: int = 5, n_edge: int = 20, n_planar: int = 40,
                          edge_thr: float = 0.08, planar_thr: float = 0.01):
    """LOAM-style curvature features on one (angularly ordered) scan line.

    c_i = | sum_{j in S}(p_i - p_j) | / (|S| |p_i|),  S = k neighbours on each side.
    High curvature -> edge points; low curvature -> planar points.  Points near
    range discontinuities (occlusions) and nearly parallel to the beam are rejected.
    """
    n = len(pts)
    if n < 4 * k:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(n)
    c = np.full(n, np.nan)
    r = np.hypot(pts[:, 0], pts[:, 1])
    for i in range(k, n - k):
        nb = np.r_[pts[i - k:i], pts[i + 1:i + k + 1]]
        c[i] = np.linalg.norm((pts[i] - nb).sum(0)) / (2 * k * r[i])
    bad = np.zeros(n, bool)
    jump = np.abs(np.diff(r)) > 0.5 + 0.1 * r[:-1]                  # range discontinuity
    for i in np.where(jump)[0]:
        bad[max(0, i - k):i + k + 1] = True
    bad |= np.isnan(c)
    order = np.argsort(-np.nan_to_num(c, nan=-1))
    edges, taken = [], np.zeros(n, bool)
    for i in order:
        if len(edges) >= n_edge or c[i] < edge_thr:
            break
        if bad[i] or taken[max(0, i - k):i + k + 1].any():
            continue
        edges.append(i)
        taken[max(0, i - k):i + k + 1] = True
    cc = np.where(bad, np.inf, c)
    planars = [i for i in np.argsort(cc)[:n_planar] if cc[i] < planar_thr]
    return np.array(edges, int), np.array(planars, int), c
