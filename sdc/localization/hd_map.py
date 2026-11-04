"""High-definition (HD) point-cloud map with the two lookup structures used for
localisation: an NDT grid (scan-to-map alignment) and a Euclidean distance
field (particle-filter likelihood)."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

from ..geometry import transform_points, voxel_downsample
from ..sim.drive import Drive, LidarParams, lidar_scan
from ..sim.world import World
from .ndt import NDTGrid


class HDMap:
    def __init__(self, points: np.ndarray, res: float = 0.1, cells=(3.0, 1.5), max_dist: float = 3.0):
        self.points = voxel_downsample(points, 0.1)
        self.res = res
        self.max_dist = max_dist
        lo = self.points.min(0) - 5.0
        hi = self.points.max(0) + 5.0
        self.origin = lo
        shape = np.ceil((hi - lo) / res).astype(int) + 1
        occ = np.zeros(shape, bool)
        ij = np.floor((self.points - lo) / res).astype(int)
        occ[ij[:, 0], ij[:, 1]] = True
        self.edt = (distance_transform_edt(~occ) * res).astype(np.float32)
        self.shape = shape
        self.grids = [NDTGrid(self.points, c, min_pts=4) for c in cells]

    def distance(self, pts: np.ndarray) -> np.ndarray:
        """Distance to the nearest map point (clipped); works for arbitrary-shaped input (...,2)."""
        ij = np.floor((pts - self.origin) / self.res).astype(int)
        ok = (ij[..., 0] >= 0) & (ij[..., 0] < self.shape[0]) & (ij[..., 1] >= 0) & (ij[..., 1] < self.shape[1])
        ii = np.clip(ij[..., 0], 0, self.shape[0] - 1)
        jj = np.clip(ij[..., 1], 0, self.shape[1] - 1)
        d = self.edt[ii, jj]
        return np.where(ok, np.minimum(d, self.max_dist), self.max_dist)


def survey_map(world: World, drive: Drive, rng: np.random.Generator, pose_sigma=(0.02, 0.02, 0.0015),
               lidar: LidarParams = LidarParams(), rate_div: int = 10) -> HDMap:
    """Build the HD map from a survey drive with (near) ground-truth poses - the stand-in for an
    RTK-GNSS/INS-referenced mapping vehicle."""
    pts = []
    for i in range(0, len(drive.t), rate_div):
        pose = drive.pose[i] + rng.normal(0, pose_sigma)
        scan = lidar_scan(world, drive.pose[i], rng, lidar)
        pts.append(transform_points(pose, scan))
    return HDMap(np.concatenate(pts, axis=0))
