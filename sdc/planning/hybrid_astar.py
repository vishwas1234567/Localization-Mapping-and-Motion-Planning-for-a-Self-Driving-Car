"""Hybrid A* (Dolgov et al., 2010) for car-like robots, plus a grid-A* baseline.

* Search state  : continuous (x, y, theta) stored in a discretised (0.5 m, 5 deg) grid -> every
                  expanded node keeps its *exact* continuous pose, so paths are kinematically feasible.
* Motion model  : kinematic bicycle, 5 steering angles x {forward, reverse}, 1.5 m arcs.
* Cost          : arc length, reverse and direction-switch penalties, steering and steering-change penalties.
* Heuristic     : holonomic-with-obstacles distance (Dijkstra from the goal over the clearance-inflated grid).
* Analytic shot : Dubins connections (forward and reverse) to the goal, collision-checked.
* Collision     : 3-disc footprint on a Euclidean distance transform of the occupancy grid.
"""
from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.ndimage import distance_transform_edt

from .dubins import dubins_path, dubins_shortest, reverse_dubins_path


# --------------------------------------------------------------------------
@dataclass
class Vehicle:
    wheelbase: float = 2.7
    length: float = 4.5
    width: float = 1.9
    rear_overhang: float = 0.9
    delta_max: float = 0.52          # rad

    @property
    def rho(self) -> float:
        return self.wheelbase / np.tan(self.delta_max)

    @property
    def kappa_max(self) -> float:
        return 1.0 / self.rho

    def disc_offsets_radius(self, n=3):
        """Centres (along the heading, from the rear axle) and common radius of n covering discs."""
        seg = self.length / n
        offs = -self.rear_overhang + seg * (np.arange(n) + 0.5)
        r = np.hypot(seg / 2, self.width / 2)
        return offs, r

    def corners(self, pose):
        x, y, th = pose
        c, s = np.cos(th), np.sin(th)
        xs = np.array([-self.rear_overhang, self.length - self.rear_overhang] * 2)
        ys = np.array([-self.width / 2, -self.width / 2, self.width / 2, self.width / 2])
        order = [0, 1, 3, 2]
        return np.stack([x + c * xs[order] - s * ys[order], y + s * xs[order] + c * ys[order]], axis=1)


class OccGrid:
    def __init__(self, x0, y0, x1, y1, res=0.1):
        self.res, self.x0, self.y0 = res, x0, y0
        self.nx, self.ny = int(np.ceil((x1 - x0) / res)), int(np.ceil((y1 - y0) / res))
        self.occ = np.zeros((self.nx, self.ny), bool)
        self.occ[0, :] = self.occ[-1, :] = self.occ[:, 0] = self.occ[:, -1] = True   # outer wall
        self._edt = None

    def add_polygon(self, poly):
        poly = np.asarray(poly, float)
        lo = np.floor((poly.min(0) - [self.x0, self.y0]) / self.res).astype(int) - 1
        hi = np.ceil((poly.max(0) - [self.x0, self.y0]) / self.res).astype(int) + 1
        lo, hi = np.maximum(lo, 0), np.minimum(hi, [self.nx - 1, self.ny - 1])
        ii, jj = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1), indexing="ij")
        cen = np.stack([self.x0 + (ii + 0.5) * self.res, self.y0 + (jj + 0.5) * self.res], axis=-1).reshape(-1, 2)
        inside = MplPath(poly).contains_points(cen).reshape(ii.shape)
        self.occ[ii[inside], jj[inside]] = True
        self._edt = None

    def add_rect(self, x0, y0, x1, y1):
        self.add_polygon([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])

    @property
    def edt(self):
        if self._edt is None:
            self._edt = (distance_transform_edt(~self.occ) * self.res).astype(np.float32)
        return self._edt

    def occupied(self, x, y):
        i = np.floor((np.asarray(x) - self.x0) / self.res).astype(int)
        j = np.floor((np.asarray(y) - self.y0) / self.res).astype(int)
        ok = (i >= 0) & (i < self.nx) & (j >= 0) & (j < self.ny)
        return np.where(ok, self.occ[np.clip(i, 0, self.nx - 1), np.clip(j, 0, self.ny - 1)], True)

    def dist(self, x, y):
        i = np.floor((np.asarray(x) - self.x0) / self.res).astype(int)
        j = np.floor((np.asarray(y) - self.y0) / self.res).astype(int)
        ok = (i >= 0) & (i < self.nx) & (j >= 0) & (j < self.ny)
        d = self.edt[np.clip(i, 0, self.nx - 1), np.clip(j, 0, self.ny - 1)]
        return np.where(ok, d, 0.0)


class FootprintChecker:
    """Collision check of the oriented car rectangle.

    The rectangle is sampled on a regular lattice (``spacing`` m, border included); a pose is free
    when every sample keeps at least ``margin`` clearance on the Euclidean distance field.  This is much
    tighter than a 3-disc cover, which over-extends ~0.5 m beyond the bumpers and makes tight slots
    infeasible.  ``exact_free`` re-checks a path against raw occupancy with a very dense raster.
    """

    def __init__(self, grid: OccGrid, veh: Vehicle, margin=0.08, spacing=0.3):
        self.g, self.veh, self.margin = grid, veh, margin
        nl = int(np.ceil(veh.length / spacing)) + 1
        nw = int(np.ceil(veh.width / spacing)) + 1
        lx = np.linspace(-veh.rear_overhang, veh.length - veh.rear_overhang, nl)
        ly = np.linspace(-veh.width / 2, veh.width / 2, nw)
        X, Y = np.meshgrid(lx, ly, indexing="ij")
        self.px, self.py = X.ravel(), Y.ravel()
        lx2 = np.arange(-veh.rear_overhang, veh.length - veh.rear_overhang + 1e-9, 0.04)
        ly2 = np.arange(-veh.width / 2, veh.width / 2 + 1e-9, 0.04)
        X2, Y2 = np.meshgrid(lx2, ly2, indexing="ij")
        self.dx, self.dy = X2.ravel(), Y2.ravel()

    def _world(self, x, y, th, px, py):
        c, s = np.cos(th)[..., None], np.sin(th)[..., None]
        return x[..., None] + c * px - s * py, y[..., None] + s * px + c * py

    def free(self, x, y, th):
        """Vectorised over pose arrays -> bool array (True = collision free)."""
        x, y, th = np.asarray(x, float), np.asarray(y, float), np.asarray(th, float)
        wx, wy = self._world(x, y, th, self.px, self.py)
        return np.all(self.g.dist(wx, wy) > self.margin, axis=-1)

    def exact_free(self, x, y, th):
        """Dense raster against the raw occupancy grid (no clearance)."""
        x, y, th = np.atleast_1d(x), np.atleast_1d(y), np.atleast_1d(th)
        out = np.ones(len(x), bool)
        for k in range(len(x)):
            wx, wy = self._world(x[k:k + 1], y[k:k + 1], th[k:k + 1], self.dx, self.dy)
            out[k] = not self.g.occupied(wx, wy).any()
        return out


# --------------------------------------------------------------------------
@dataclass
class PlanResult:
    success: bool
    path: np.ndarray | None = None       # (N,4): x, y, theta, direction (+1 fwd / -1 rev)
    time_s: float = 0.0
    expansions: int = 0
    cost: float = np.inf
    used_shot: bool = False


class HybridAStar:
    def __init__(self, grid: OccGrid, veh: Vehicle = Vehicle(), xy_res=0.5, n_theta=72, steps=(0.6, 1.5), sub=0.25,
                 n_steer=5, reverse_pen=2.0, switch_pen=6.0, steer_pen=0.4, steer_chg_pen=1.5,
                 shot_every=6, max_exp=40000, margin=0.08, nh_heuristic=True, h_weight=1.0, nh_radius=1e9):
        self.g, self.veh = grid, veh
        self.chk = FootprintChecker(grid, veh, margin=margin)
        self.xy_res, self.n_theta, self.steps, self.sub = xy_res, n_theta, tuple(steps), sub
        self.reverse_pen, self.switch_pen, self.steer_pen, self.steer_chg_pen = reverse_pen, switch_pen, steer_pen, steer_chg_pen
        self.shot_every, self.max_exp = shot_every, max_exp
        self.nh_heuristic, self.h_weight, self.nh_radius = nh_heuristic, h_weight, nh_radius
        self.steers = np.linspace(-veh.delta_max, veh.delta_max, n_steer)
        self._prims()

    # primitives in the local frame, one group per arc length: (A, K, 3)
    def _prims(self):
        self.groups = []
        for step in self.steps:
            ks = max(int(round(step / self.sub)), 1)
            d = np.arange(1, ks + 1) * (step / ks)
            A, act = [], []
            for dirn in (+1, -1):
                for si, dl in enumerate(self.steers):
                    kap = np.tan(dl) / self.veh.wheelbase
                    sd = dirn * d
                    if abs(kap) < 1e-9:
                        lx, ly, lt = sd, np.zeros_like(sd), np.zeros_like(sd)
                    else:
                        lx, ly, lt = np.sin(kap * sd) / kap, (1 - np.cos(kap * sd)) / kap, kap * sd
                    A.append(np.stack([lx, ly, lt], axis=-1))
                    act.append((dirn, si))
            self.groups.append((np.array(A), act, step))

    # ------------------------------------------------------------------
    def _key(self, x, y, th):
        return (int(np.floor(x / self.xy_res)), int(np.floor(y / self.xy_res)),
                int(np.floor(((th % (2 * np.pi)) / (2 * np.pi)) * self.n_theta)) % self.n_theta)

    def _heuristic_field(self, goal):
        """Dijkstra over the grid (8-connected) from the goal, cells blocked if closer than the disc radius."""
        res = 0.25
        nx, ny = int(self.g.nx * self.g.res / res), int(self.g.ny * self.g.res / res)
        xs = self.g.x0 + (np.arange(nx) + 0.5) * res
        ys = self.g.y0 + (np.arange(ny) + 0.5) * res
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        free = self.g.dist(X, Y) > self.veh.width / 2 + 0.05
        import scipy.sparse as sp
        from scipy.sparse.csgraph import dijkstra
        idx = -np.ones((nx, ny), int)
        idx[free] = np.arange(free.sum())
        rows, cols, w = [], [], []
        for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
            a = idx[0:nx - dx, max(0, -dy):ny - max(0, dy)]
            b = idx[dx:nx, max(0, dy):ny - max(0, -dy)]
            m = (a >= 0) & (b >= 0)
            rows.append(a[m]); cols.append(b[m]); w.append(np.full(m.sum(), res * np.hypot(dx, dy)))
        r, c, ww = np.concatenate(rows), np.concatenate(cols), np.concatenate(w)
        n = int(free.sum())
        G = sp.coo_matrix((ww, (r, c)), shape=(n, n)).tocsr()
        G = G + G.T
        gi = int(np.clip((goal[0] - self.g.x0) / res, 0, nx - 1)), int(np.clip((goal[1] - self.g.y0) / res, 0, ny - 1))
        if idx[gi] < 0:                                  # snap to nearest free cell
            fx, fy = np.where(free)
            k = np.argmin((fx - gi[0]) ** 2 + (fy - gi[1]) ** 2)
            gi = (fx[k], fy[k])
        dist = dijkstra(G, directed=False, indices=int(idx[gi]))
        field = np.full((nx, ny), np.inf)
        field[free] = dist
        return field, res

    def _h(self, field, x, y):
        f, res = field
        i = int(np.clip((x - self.g.x0) / res, 0, f.shape[0] - 1))
        j = int(np.clip((y - self.g.y0) / res, 0, f.shape[1] - 1))
        v = f[i, j]
        if not np.isfinite(v):                           # in a blocked cell: use the nearest finite neighbour
            sl = f[max(0, i - 2):i + 3, max(0, j - 2):j + 3]
            v = sl[np.isfinite(sl)].min() + 0.5 if np.isfinite(sl).any() else 1e4
        return v

    def _h_full(self, hf, x, y, th, goal):
        """max( holonomic-with-obstacles , non-holonomic-without-obstacles ) -- Dolgov's pair of heuristics.

        The non-holonomic term is the shorter of the forward Dubins length and the reverse Dubins length
        (weighted by the reverse penalty), so it also guides reversing manoeuvres.
        """
        h = self._h(hf, x, y)
        if self.nh_heuristic and np.hypot(goal[0] - x, goal[1] - y) <= self.nh_radius:   # Dubins term only near the goal
            f = dubins_shortest((x, y, th), goal, self.veh.rho)
            r = dubins_shortest(goal, (x, y, th), self.veh.rho)          # reverse drive = time-reversed forward drive
            c = min(f[0] if f else np.inf, (r[0] * (1.0 + self.reverse_pen)) if r else np.inf)
            if np.isfinite(c):
                h = max(h, c)
        return self.h_weight * h

    # ------------------------------------------------------------------
    def _try_shot(self, node, goal):
        q0 = (node[0], node[1], node[2])
        best = None
        for fn, dirn in ((dubins_path, +1), (reverse_dubins_path, -1)):
            r = fn(q0, goal, self.veh.rho, 0.3)
            if r is None:
                continue
            pts, L = r
            if L > 60:
                continue
            if self.chk.free(pts[:, 0], pts[:, 1], pts[:, 2]).all():
                cost = L * (1.0 if dirn > 0 else 1.0 + self.reverse_pen)
                if best is None or cost < best[0]:
                    best = (cost, pts, dirn)
        return best

    def plan(self, start, goal, time_limit=20.0) -> PlanResult:
        t0 = time.time()
        start, goal = np.array(start, float), np.array(goal, float)
        if not self.chk.free(*start)[()] or not self.chk.free(*goal)[()]:
            return PlanResult(False, time_s=time.time() - t0)
        hf = self._heuristic_field(goal)
        # node: (x,y,th,g,dir,steer_idx,parent,primitive_pts)
        nodes = [(start[0], start[1], start[2], 0.0, 0, 2, -1, None)]
        openq = [(self._h_full(hf, start[0], start[1], start[2], goal), 0)]
        best_g = {self._key(*start): 0.0}
        closed = set()
        n_exp, used_shot = 0, False
        final = None
        while openq and n_exp < self.max_exp and time.time() - t0 < time_limit:
            f, ni = heapq.heappop(openq)
            x, y, th, g, dirn, sidx, par, _ = nodes[ni]
            k = self._key(x, y, th)
            if k in closed:
                continue
            closed.add(k)
            n_exp += 1
            dgoal = np.hypot(goal[0] - x, goal[1] - y)
            dth = abs((th - goal[2] + np.pi) % (2 * np.pi) - np.pi)
            if dgoal < 0.30 and dth < 0.09:           # ~5 deg; the exact Dubins shot is preferred when it exists
                final = (ni, None)
                break
            if n_exp % self.shot_every == 0 or (dgoal < 15.0 and n_exp % 2 == 0):
                shot = self._try_shot((x, y, th), goal)
                if shot is not None:
                    final = (ni, shot)
                    used_shot = True
                    break
            # vectorised expansion, one batch per primitive length
            c, s_ = np.cos(th), np.sin(th)
            for P, act, step in self.groups:
                X = x + c * P[..., 0] - s_ * P[..., 1]
                Y = y + s_ * P[..., 0] + c * P[..., 1]
                T = th + P[..., 2]
                ok = self.chk.free(X, Y, T).all(axis=1)
                for a in np.where(ok)[0]:
                    d2, si2 = act[a]
                    nx_, ny_, nt_ = X[a, -1], Y[a, -1], T[a, -1]
                    nk = self._key(nx_, ny_, nt_)
                    if nk in closed:
                        continue
                    cost = step * (1.0 if d2 > 0 else 1.0 + self.reverse_pen)
                    cost += self.steer_pen * abs(self.steers[si2]) + self.steer_chg_pen * abs(self.steers[si2] - self.steers[sidx])
                    if dirn != 0 and d2 != dirn:
                        cost += self.switch_pen
                    g2 = g + cost
                    if g2 < best_g.get(nk, np.inf):
                        best_g[nk] = g2
                        nodes.append((nx_, ny_, nt_, g2, d2, si2, ni, np.stack([X[a], Y[a], T[a]], axis=1)))
                        heapq.heappush(openq, (g2 + self._h_full(hf, nx_, ny_, nt_, goal), len(nodes) - 1))
        dt = time.time() - t0
        if final is None:
            return PlanResult(False, time_s=dt, expansions=n_exp)
        # backtrack
        ni, shot = final
        segs, cur = [], ni
        while cur != -1:
            n = nodes[cur]
            if n[7] is not None:
                segs.append(np.column_stack([n[7], np.full(len(n[7]), n[4])]))
            cur = n[6]
        segs = segs[::-1]
        path = np.vstack([[start[0], start[1], start[2], 0.0]] + segs) if segs else np.array([[*start, 0.0]])
        if shot is not None:
            cost, pts, dirn = shot
            path = np.vstack([path, np.column_stack([pts[1:], np.full(len(pts) - 1, dirn)])])
        path[0, 3] = path[1, 3] if len(path) > 1 else 1.0
        path[:, 2] = (path[:, 2] + np.pi) % (2 * np.pi) - np.pi
        return PlanResult(True, path, dt, n_exp, nodes[ni][3], used_shot)


# --------------------------------------------------------------------------
def grid_astar(grid: OccGrid, start_xy, goal_xy, clearance: float, res: float = 0.25):
    """Classic 8-connected grid A* for a *point* robot on a clearance-inflated grid."""
    nx, ny = int(grid.nx * grid.res / res), int(grid.ny * grid.res / res)
    xs = grid.x0 + (np.arange(nx) + 0.5) * res
    ys = grid.y0 + (np.arange(ny) + 0.5) * res
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    free = grid.dist(X, Y) > clearance
    si = (int((start_xy[0] - grid.x0) / res), int((start_xy[1] - grid.y0) / res))
    gi = (int((goal_xy[0] - grid.x0) / res), int((goal_xy[1] - grid.y0) / res))
    if not (free[si] and free[gi]):
        return None, 0
    openq = [(0.0, si)]
    g = {si: 0.0}
    par = {}
    closed = set()
    moves = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    n = 0
    while openq:
        _, cur = heapq.heappop(openq)
        if cur in closed:
            continue
        closed.add(cur)
        n += 1
        if cur == gi:
            break
        for dx, dy in moves:
            nb = (cur[0] + dx, cur[1] + dy)
            if not (0 <= nb[0] < nx and 0 <= nb[1] < ny) or not free[nb] or nb in closed:
                continue
            g2 = g[cur] + res * np.hypot(dx, dy)
            if g2 < g.get(nb, np.inf):
                g[nb] = g2
                par[nb] = cur
                heapq.heappush(openq, (g2 + res * np.hypot(gi[0] - nb[0], gi[1] - nb[1]), nb))
    if gi not in par and gi != si:
        return None, n
    path, cur = [gi], gi
    while cur != si:
        cur = par[cur]
        path.append(cur)
    path = np.array(path[::-1], float)
    return np.stack([grid.x0 + (path[:, 0] + 0.5) * res, grid.y0 + (path[:, 1] + 0.5) * res], axis=1), n


def chaikin(path: np.ndarray, iters: int = 3) -> np.ndarray:
    for _ in range(iters):
        q = 0.75 * path[:-1] + 0.25 * path[1:]
        r = 0.25 * path[:-1] + 0.75 * path[1:]
        path = np.vstack([path[:1], np.stack([q, r], axis=1).reshape(-1, 2), path[-1:]])
    return path


def path_metrics(xy: np.ndarray, theta: np.ndarray | None = None, kappa_max: float | None = None):
    """Length, max |curvature|, number of heading jumps > 20 deg between consecutive samples (corners),
    and fraction of the path exceeding the curvature limit.  ``theta`` only used for cusp handling."""
    d = np.diff(xy, axis=0)
    seg = np.hypot(d[:, 0], d[:, 1])
    keep = seg > 1e-6
    d, seg = d[keep], seg[keep]
    hd = np.arctan2(d[:, 1], d[:, 0])
    dh = np.abs((np.diff(hd) + np.pi) % (2 * np.pi) - np.pi)
    ds = 0.5 * (seg[:-1] + seg[1:])
    kap = dh / np.maximum(ds, 1e-6)
    return dict(length=float(seg.sum()), max_kappa=float(kap.max()) if len(kap) else 0.0,
                corners=int(np.sum(dh > np.radians(20))),
                frac_over=float(np.mean(kap > kappa_max * 1.05)) if kappa_max else np.nan)
