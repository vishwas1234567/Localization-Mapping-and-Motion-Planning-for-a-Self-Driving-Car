"""Monte-Carlo localisation (particle filter) against the HD map's distance field,
and an NDT scan-to-map localiser driven by an odometry prediction."""
from __future__ import annotations

import numpy as np

from ..geometry import compose, wrap
from .hd_map import HDMap
from .ndt import ndt_register_multires


class ParticleFilter:
    def __init__(self, hdmap: HDMap, n: int, rng: np.random.Generator, sigma_hit: float = 0.2,
                 n_beams: int = 72, beam_power: float = 0.35, z_rand: float = 0.08,
                 motion_noise=(0.02, 0.04, 0.004, 0.01)):
        self.map, self.n, self.rng = hdmap, n, rng
        self.sigma_hit, self.n_beams, self.beam_power, self.z_rand = sigma_hit, n_beams, beam_power, z_rand
        self.motion_noise = motion_noise       # (trans const, trans prop., rot const, rot prop.)
        self.p = np.zeros((n, 3))
        self.logw = np.full(n, -np.log(n))
        self.neff = float(n)

    # ------------------------------------------------------------------
    def init_gaussian(self, mean, std):
        self.p = np.asarray(mean) + self.rng.normal(0, 1, (self.n, 3)) * np.asarray(std)
        self.logw[:] = -np.log(self.n)

    def init_uniform(self, sampler):
        """``sampler(n, rng)`` returns (n,3) poses - e.g. uniform over the drivable corridor."""
        self.p = sampler(self.n, self.rng)
        self.logw[:] = -np.log(self.n)

    # ------------------------------------------------------------------
    def predict(self, delta):
        """Apply an odometry increment (in the body frame) with noise to every particle."""
        a, b, c, d = self.motion_noise
        dist = np.hypot(delta[0], delta[1])
        n = self.n
        dx = delta[0] + self.rng.normal(0, a + b * dist, n)
        dy = delta[1] + self.rng.normal(0, a + b * dist, n)
        dth = delta[2] + self.rng.normal(0, c + d * abs(delta[2]), n)
        cs, sn = np.cos(self.p[:, 2]), np.sin(self.p[:, 2])
        self.p[:, 0] += cs * dx - sn * dy
        self.p[:, 1] += sn * dx + cs * dy
        self.p[:, 2] = wrap(self.p[:, 2] + dth)

    def update(self, scan: np.ndarray):
        if len(scan) == 0:
            return
        sel = np.linspace(0, len(scan) - 1, min(self.n_beams, len(scan))).astype(int)
        s = scan[sel]
        cs, sn = np.cos(self.p[:, 2])[:, None], np.sin(self.p[:, 2])[:, None]
        wx = self.p[:, 0:1] + cs * s[None, :, 0] - sn * s[None, :, 1]
        wy = self.p[:, 1:2] + sn * s[None, :, 0] + cs * s[None, :, 1]
        d = self.map.distance(np.stack([wx, wy], axis=-1))
        lik = (1 - self.z_rand) * np.exp(-0.5 * (d / self.sigma_hit) ** 2) + self.z_rand
        ll = self.beam_power * np.log(lik).sum(axis=1)
        self.logw += ll
        self.logw -= np.logaddexp.reduce(self.logw)
        w = np.exp(self.logw)
        self.neff = 1.0 / np.sum(w * w)
        if self.neff < 0.5 * self.n:
            self._resample(w)

    def _resample(self, w):
        n = self.n
        pos = (self.rng.random() + np.arange(n)) / n
        idx = np.minimum(np.searchsorted(np.cumsum(w), pos), n - 1)
        self.p = self.p[idx]
        self.p[:, :2] += self.rng.normal(0, 0.02, (n, 2))
        self.p[:, 2] = wrap(self.p[:, 2] + self.rng.normal(0, 0.003, n))
        self.logw[:] = -np.log(n)

    # ------------------------------------------------------------------
    def estimate(self):
        w = np.exp(self.logw)
        xy = (w[:, None] * self.p[:, :2]).sum(0)
        th = np.arctan2((w * np.sin(self.p[:, 2])).sum(), (w * np.cos(self.p[:, 2])).sum())
        spread = np.sqrt((w * ((self.p[:, :2] - xy) ** 2).sum(1)).sum())
        return np.array([xy[0], xy[1], th]), spread


class NdtLocalizer:
    """Scan-to-map NDT alignment seeded by an odometry prediction."""

    def __init__(self, hdmap: HDMap, init_pose):
        self.map = hdmap
        self.pose = np.array(init_pose, float)
        self.last_degeneracy = np.nan

    def step(self, scan: np.ndarray, odom_delta):
        pred = compose(self.pose, odom_delta)
        res = ndt_register_multires(scan, self.map.grids, pred)
        self.pose = res.pose
        self.last_degeneracy = res.degeneracy()
        return self.pose, res
