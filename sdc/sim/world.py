"""Synthetic urban environment: a closed rounded-rectangle road loop with
buildings, poles, trees, a curved tunnel (GPS denied, low-texture) and an
underpass (GPS denied).  Everything is planar, matching the 2-D LiDAR model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------
# Route (road centre line)
# --------------------------------------------------------------------------
def _line(p0, p1, ds):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    n = max(2, int(np.hypot(*(p1 - p0)) / ds) + 1)
    return p0 + np.linspace(0, 1, n)[:, None] * (p1 - p0)


def _arc(cx, cy, r, a0, a1, ds):
    n = max(2, int(abs(a1 - a0) * r / ds) + 1)
    a = np.linspace(a0, a1, n)
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], axis=1)


@dataclass
class Route:
    s: np.ndarray       # arclength samples
    xy: np.ndarray      # (N,2)
    theta: np.ndarray   # unwrapped heading
    kappa: np.ndarray   # curvature
    length: float

    def at(self, s):
        """Periodic evaluation: returns xy (...,2), theta, kappa."""
        s = np.mod(s, self.length)
        x = np.interp(s, self.s, self.xy[:, 0])
        y = np.interp(s, self.s, self.xy[:, 1])
        th = np.interp(s, self.s, self.theta)
        k = np.interp(s, self.s, self.kappa)
        return np.stack([x, y], axis=-1), th, k


def build_route(W=220.0, H=140.0, r=25.0, ds=0.1) -> Route:
    """Counter-clockwise rounded rectangle starting at the middle of the bottom edge."""
    pieces = [
        _line((W / 2, 0), (W - r, 0), ds),
        _arc(W - r, r, r, -np.pi / 2, 0.0, ds),
        _line((W, r), (W, H - r), ds),
        _arc(W - r, H - r, r, 0.0, np.pi / 2, ds),
        _line((W - r, H), (r, H), ds),
        _arc(r, H - r, r, np.pi / 2, np.pi, ds),
        _line((0, H - r), (0, r), ds),
        _arc(r, r, r, np.pi, 1.5 * np.pi, ds),
        _line((r, 0), (W / 2, 0), ds),
    ]
    pts = [pieces[0]] + [p[1:] for p in pieces[1:]]
    pts = np.concatenate(pts, axis=0)
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s_raw = np.concatenate([[0.0], np.cumsum(seg)])
    length = s_raw[-1]
    s = np.arange(0.0, length, ds)
    xy = np.stack([np.interp(s, s_raw, pts[:, 0]), np.interp(s, s_raw, pts[:, 1])], axis=1)
    d = np.gradient(xy, ds, axis=0)
    theta = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    # smooth curvature estimate
    kappa = np.gradient(theta, ds)
    k = 25
    kappa = np.convolve(np.pad(kappa, k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), mode="valid")
    # close the loop explicitly so that interpolation is continuous across s = L
    s = np.append(s, length)
    xy = np.vstack([xy, xy[0]])
    theta = np.append(theta, theta[0] + 2.0 * np.pi)
    kappa = np.append(kappa, kappa[0])
    return Route(s, xy, theta, kappa, float(length))


# --------------------------------------------------------------------------
# World
# --------------------------------------------------------------------------
def _rect_segments(corners):
    c = np.asarray(corners)
    return np.concatenate([c, np.roll(c, -1, axis=0)], axis=1)  # (4,4): x1 y1 x2 y2


class World:
    """Static map + metadata (tunnel / GPS-denied intervals)."""

    W, H, R = 220.0, 140.0, 25.0

    def __init__(self, seed: int = 7):
        self.rng = np.random.default_rng(seed)
        self.route = build_route(self.W, self.H, self.R)
        L = self.route.length
        self.length = L
        self.tunnel = (200.0, 330.0)
        self.underpass = (500.0, 525.0)
        half = self.W / 2 - self.R            # length of the first/last half of the bottom edge
        arc = np.pi * self.R / 2
        vert, horiz = self.H - 2 * self.R, self.W - 2 * self.R
        # s-ranges of the straight road pieces and of the corners
        s0 = half + arc
        self.straights, self.corners = [], [(half, half + arc)]
        for length_ in (vert, horiz, vert):
            self.straights.append((s0, s0 + length_))
            self.corners.append((s0 + length_, s0 + length_ + arc))
            s0 += length_ + arc
        self.straights.append((s0, s0 + half + half))       # bottom edge, wraps through s = L

        segs, circs = [], []
        self._add_buildings(segs)
        self._add_clutter(circs)
        self._add_tunnel(segs)
        self._add_underpass(circs)
        self.segments = np.concatenate(segs, axis=0)
        self.circles = np.asarray(circs, float).reshape(-1, 3)

    # ------------------------------------------------------------------
    def _frame(self, s):
        xy, th, _ = self.route.at(s)
        t = np.array([np.cos(th), np.sin(th)])
        n = np.array([-np.sin(th), np.cos(th)])
        return xy, t, n

    def in_interval(self, s, interval, margin=0.0):
        sm = np.mod(s, self.length)
        return (sm >= interval[0] - margin) & (sm <= interval[1] + margin)

    def gps_available(self, s) -> bool:
        return not (self.in_interval(s, self.tunnel) or self.in_interval(s, self.underpass))

    def vision_factor(self, s) -> float:
        """Visual-odometry noise multiplier (low texture / dark in the tunnel)."""
        return 2.5 if self.in_interval(s, self.tunnel) else 1.0

    # ------------------------------------------------------------------
    def _add_buildings(self, segs):
        rng = self.rng
        for (a, b) in self.straights:
            for side in (+1.0, -1.0):
                s = a + 6 + rng.uniform(0, 6)
                while True:
                    w = rng.uniform(12, 28)
                    if s + w > b - 6:
                        break
                    gap = rng.uniform(4, 12)
                    if not (self.in_interval(s, self.tunnel, 12) or self.in_interval(s + w, self.tunnel, 12)):
                        off, depth = rng.uniform(9.5, 12.0), rng.uniform(8, 14)
                        c, t, n = self._frame(s + w / 2)
                        p = [c - w / 2 * t + side * off * n,
                             c + w / 2 * t + side * off * n,
                             c + w / 2 * t + side * (off + depth) * n,
                             c - w / 2 * t + side * (off + depth) * n]
                        segs.append(_rect_segments(p))
                    s += w + gap

    def _add_clutter(self, circs):
        rng = self.rng
        # poles along the road
        s = 3.0
        while s < self.length:
            if not self.in_interval(s, self.tunnel, 5):
                for side in (+1.0, -1.0):
                    if rng.random() < 0.8:
                        c, t, n = self._frame(s + rng.uniform(-2, 2))
                        p = c + side * rng.uniform(5.0, 6.5) * n
                        circs.append([p[0], p[1], 0.25])
            s += rng.uniform(14, 20)
        # trees in the corners
        for (a, b) in self.corners:
            if self.in_interval(0.5 * (a + b), self.tunnel, 25):
                continue
            s = a + 2
            while s < b:
                for side in (+1.0, -1.0):
                    c, t, n = self._frame(s)
                    p = c + side * rng.uniform(7, 10) * n
                    circs.append([p[0], p[1], rng.uniform(0.8, 1.2)])
                s += rng.uniform(6, 9)

    def _add_tunnel(self, segs, half_width=4.5, step=1.5):
        a, b = self.tunnel
        ss = np.arange(a, b + step, step)
        for side in (+1.0, -1.0):
            pts = []
            for s in ss:
                c, t, n = self._frame(s)
                pts.append(c + side * half_width * n)
            pts = np.asarray(pts)
            segs.append(np.concatenate([pts[:-1], pts[1:]], axis=1))
        # structural ribs (cross-passage frames) every 20 m, alternating sides
        k = 0
        for s in np.arange(a + 10, b - 5, 20.0):
            side = +1.0 if k % 2 == 0 else -1.0
            c, t, n = self._frame(s)
            w = c + side * half_width * n
            p = [w - 0.4 * t, w - 0.4 * t - side * 0.9 * n, w + 0.4 * t - side * 0.9 * n, w + 0.4 * t]
            segs.append(np.concatenate([np.array(p[:-1]), np.array(p[1:])], axis=1))
            k += 1

    def _add_underpass(self, circs):
        a, b = self.underpass
        for s in (a, b):
            c, t, n = self._frame(s)
            for side in (+1.0, -1.0):
                p = c + side * 7.5 * n
                circs.append([p[0], p[1], 0.7])

    # ------------------------------------------------------------------
    def dynamic_objects(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Unmapped vehicles (parked / oncoming) -> (4n,4) segment array."""
        segs = []
        for _ in range(n):
            for _try in range(20):
                s = rng.uniform(0, self.length)
                if rng.random() < 0.5:
                    lat = -rng.uniform(4.8, 5.6)      # parked on the kerb side
                    if self.in_interval(s, self.tunnel, 3):
                        continue
                else:
                    lat = rng.uniform(1.4, 2.2)       # opposite lane
                c, t, nn = self._frame(s)
                ctr = c + lat * nn
                L, Wd = 4.5, 1.9
                p = [ctr - L / 2 * t - Wd / 2 * nn, ctr + L / 2 * t - Wd / 2 * nn,
                     ctr + L / 2 * t + Wd / 2 * nn, ctr - L / 2 * t + Wd / 2 * nn]
                segs.append(_rect_segments(p))
                break
        return np.concatenate(segs, axis=0) if segs else np.zeros((0, 4))
