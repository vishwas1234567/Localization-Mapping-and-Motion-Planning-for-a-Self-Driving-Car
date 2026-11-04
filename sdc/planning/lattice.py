"""Conformal state lattice in the Frenet frame of the road centre line.

Candidates are quintic lateral polynomials d(s) with a *spatial* parametrisation,

    d(s0)=d0, d'(s0)=d0', d''(s0)=d0''   ->   d(s0+S_f)=d_f, d'=0, d''=0,

for a lattice of terminal offsets d_f and horizons S_f.  They are C^2 by construction, so the
stitched receding-horizon path has continuous curvature.  Each candidate is scored by

    J = w_j * int (d''')^2 ds  +  w_o * ((d_f - d_ref)^2 + mean (d - d_ref)^2)  +  w_k * max(|kappa|-0.6 kmax,0)^2

and rejected if it leaves the road, exceeds the curvature limit or comes closer to an obstacle than the margin.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LatticeConfig:
    d_offsets: tuple = tuple(np.linspace(-3.0, 3.0, 13))
    horizons: tuple = (18.0, 26.0, 36.0)
    d_ref: float = -1.75                 # preferred lane centre
    d_min: float = -3.1
    d_max: float = 3.1                   # drivable corridor (no oncoming traffic in this test)
    ds: float = 0.5
    kappa_max: float = 0.2
    w_jerk: float = 40.0
    w_off: float = 1.0
    w_kappa: float = 200.0
    ego_discs: tuple = (-0.2, 1.5, 3.2)  # along the heading from the rear axle
    ego_radius: float = 1.15
    margin: float = 0.35


def quintic(d0, dp0, ddp0, df, T):
    A = np.array([[T ** 3, T ** 4, T ** 5], [3 * T ** 2, 4 * T ** 3, 5 * T ** 4], [6 * T, 12 * T ** 2, 20 * T ** 3]])
    b = np.array([df - (d0 + dp0 * T + 0.5 * ddp0 * T ** 2), -(dp0 + ddp0 * T), -ddp0])
    a3, a4, a5 = np.linalg.solve(A, b)
    return np.array([d0, dp0, 0.5 * ddp0, a3, a4, a5])


def eval_poly(c, x):
    d = c[0] + c[1] * x + c[2] * x ** 2 + c[3] * x ** 3 + c[4] * x ** 4 + c[5] * x ** 5
    dp = c[1] + 2 * c[2] * x + 3 * c[3] * x ** 2 + 4 * c[4] * x ** 3 + 5 * c[5] * x ** 4
    ddp = 2 * c[2] + 6 * c[3] * x + 12 * c[4] * x ** 2 + 20 * c[5] * x ** 3
    dddp = 6 * c[3] + 24 * c[4] * x + 60 * c[5] * x ** 2
    return d, dp, ddp, dddp


class Frenet:
    """Frenet <-> world conversions for a periodic ``Route``."""

    def __init__(self, route):
        self.route = route

    def to_world(self, s, d, dp=None):
        xy, th, kr = self.route.at(s)
        n = np.stack([-np.sin(th), np.cos(th)], axis=-1)
        pos = xy + d[..., None] * n
        if dp is None:
            return pos, th, kr
        dth = np.arctan2(dp, 1.0 - kr * d)
        return pos, th + dth, kr

    @staticmethod
    def curvature(kr, d, dp, ddp):
        """Curvature of the path d(s) on a road of curvature kr (kr' neglected)."""
        a = 1.0 - kr * d
        dth_ds = (ddp * a + dp * (kr * dp)) / (a * a + dp * dp)
        return (kr + dth_ds) / np.sqrt(a * a + dp * dp)


class StateLattice:
    def __init__(self, route, obstacles: np.ndarray, cfg: LatticeConfig = LatticeConfig(), profile: str = "quintic",
                 ramp: float = 12.0):
        """``obstacles``: (N,3) discs [x, y, r] in the world frame.

        ``profile='quintic'``: C^2 quintic lateral profiles (the state lattice).
        ``profile='linear'`` : piecewise-linear lateral ramps over ``ramp`` m -- the "connect-the-dots" baseline.
        Both use the same terminal-offset set, collision test and road limits, so only smoothness differs.
        """
        self.route, self.cfg, self.fr = route, cfg, Frenet(route)
        self.profile, self.ramp = profile, ramp
        self.obs = np.asarray(obstacles, float).reshape(-1, 3)

    def clearance(self, pos, th):
        """Minimum (disc-to-disc) clearance along a sampled path: pos (M,2), heading th (M,)."""
        if len(self.obs) == 0:
            return np.inf
        offs = np.array(self.cfg.ego_discs)
        cx = pos[:, 0:1] + np.cos(th)[:, None] * offs[None, :]
        cy = pos[:, 1:2] + np.sin(th)[:, None] * offs[None, :]
        dx = cx[:, :, None] - self.obs[None, None, :, 0]
        dy = cy[:, :, None] - self.obs[None, None, :, 1]
        gap = np.hypot(dx, dy) - self.cfg.ego_radius - self.obs[None, None, :, 2]
        return float(gap.min())

    def _profile(self, d0, dp0, ddp0, df, Sf):
        if self.profile == "quintic":
            c = quintic(d0, dp0, ddp0, df, Sf)
            return lambda x: eval_poly(c, x)
        L, k = self.ramp, (df - d0)
        def fn(x):
            u = np.clip(x / L, 0.0, 1.0)
            d = d0 + k * u
            dp = np.where(x < L, k / L, 0.0)
            z = np.zeros_like(x)
            return d, dp, z, z
        return fn

    def plan(self, s0, d0, dp0, ddp0):
        cfg = self.cfg
        best, cands = None, []
        horizons = cfg.horizons if self.profile == "quintic" else (max(cfg.horizons[0], self.ramp + 6.0),)
        for Sf in horizons:
            x = np.arange(0.0, Sf + 1e-9, cfg.ds)
            s = s0 + x
            xy_c, th_c, kr = self.route.at(s)
            for df in cfg.d_offsets:
                fn = self._profile(d0, dp0, ddp0, df, Sf)
                d, dp, ddp, dddp = fn(x)
                ok = (d.min() >= cfg.d_min) and (d.max() <= cfg.d_max)
                kap = Frenet.curvature(kr, d, dp, ddp)
                kmax = np.abs(kap).max()
                pos, th, _ = self.fr.to_world(s, d, dp)
                clr = self.clearance(pos, th) if ok else -np.inf
                feasible = ok and clr >= cfg.margin and (self.profile != "quintic" or kmax <= cfg.kappa_max)
                jerk = np.trapezoid(dddp ** 2, x)
                off = (df - cfg.d_ref) ** 2 + np.mean((d - cfg.d_ref) ** 2)
                cost = cfg.w_jerk * jerk + cfg.w_off * off
                if self.profile == "quintic":
                    cost += cfg.w_kappa * max(kmax - 0.6 * cfg.kappa_max, 0.0) ** 2
                cost += 4.0 * max(1.2 - clr, 0.0) if feasible else 0.0        # prefer wider clearance
                cands.append(dict(Sf=Sf, df=df, cost=cost, feasible=feasible, fn=fn, kmax=kmax, clr=clr))
                if feasible and (best is None or cost < best["cost"]):
                    best = cands[-1]
        return best, cands


def run_receding_horizon(route, planner, s_start, s_end, v=8.0, replan_every=4.0, d_init=-1.75):
    """Receding-horizon driving with perfect tracking; returns the executed path and its statistics.

    Every ``replan_every`` metres the planner is re-run from the current Frenet state (d, d', d'').  If no
    candidate is feasible the previous plan is kept (and counted as a failure).
    """
    fr = planner.fr
    s, d, dp, ddp = s_start, d_init, 0.0, 0.0
    S, D, DP, DDP, plan_fail, n_plans = [], [], [], [], 0, 0
    cur_fn, cur_s0, cur_Sf = None, None, None
    while s < s_end - 1e-9:
        best, _ = planner.plan(s, d, dp, ddp)
        n_plans += 1
        if best is None:
            plan_fail += 1
            if cur_fn is None:
                break
            fn, s_ref, Sf = cur_fn, cur_s0, cur_Sf
        else:
            fn, s_ref, Sf = best["fn"], s, best["Sf"]
        x_end = min(replan_every, s_end - s)
        xs = np.arange(0.0, x_end + 1e-9, planner.cfg.ds / 2)
        xx = np.clip(xs + (s - s_ref), 0.0, Sf)
        dd, ddp_, ddpp, _ = fn(xx)
        S.append(s + xs); D.append(dd); DP.append(ddp_); DDP.append(ddpp)
        cur_fn, cur_s0, cur_Sf = fn, s_ref, Sf
        s, d, dp, ddp = s + xs[-1], dd[-1], ddp_[-1], ddpp[-1]
    S, D, DP, DDP = map(np.concatenate, (S, D, DP, DDP))
    pos, th, kr = fr.to_world(S, D, DP)
    seg = np.hypot(*np.diff(pos, axis=0).T)
    keep = np.r_[True, seg > 1e-6]
    pos_k, th_k = pos[keep], np.unwrap(th[keep])
    sl = np.r_[0.0, np.cumsum(np.hypot(*np.diff(pos_k, axis=0).T))]
    kappa = np.gradient(th_k, sl)                     # same estimator for every planner
    return dict(s=S[keep], d=D[keep], pos=pos_k, theta=th_k, kappa=kappa, arclen=sl, fails=plan_fail, plans=n_plans)
