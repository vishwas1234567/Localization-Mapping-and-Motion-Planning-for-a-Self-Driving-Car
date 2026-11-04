"""2-D Normal Distributions Transform (NDT) scan registration.

Implements the P2D-NDT of Biber & Strasser (2003) with the Gaussian-plus-
uniform likelihood approximation of Magnusson (2009):

    cost(x) = d1 * exp(-d2/2 * (x'-mu)^T Sigma^-1 (x'-mu)),   x' = T(p) x

minimised with damped Newton + backtracking line search.  Four half-cell
shifted grids are used to remove discretisation artefacts.  The final Hessian
is returned so that registration *degeneracy* (e.g. a featureless tunnel) can be
analysed from its eigenvalues.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_OFF = 1 << 20
_K = 1 << 22


class NDTGrid:
    def __init__(self, points: np.ndarray, cell: float, min_pts: int = 4,
                 shifts=((0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5)),
                 min_ratio: float = 1e-3, min_var: float = 4e-4):
        self.cell = float(cell)
        self.grids = [self._build(points, shift, min_pts, min_ratio, min_var) for shift in shifts]
        self.shifts = shifts
        # Magnusson (2009) parameters of the Gaussian approximation
        p_o = 0.1
        c1, c2 = 10.0 * (1 - p_o), p_o / (self.cell ** 2)
        d3 = -np.log(c2)
        self.d1 = -np.log(c1 + c2) - d3
        self.d2 = -2.0 * np.log((-np.log(c1 * np.exp(-0.5) + c2) - d3) / self.d1)

    def _build(self, pts, shift, min_pts, min_ratio, min_var):
        ij = np.floor(pts / self.cell - np.asarray(shift)).astype(np.int64)
        key = (ij[:, 0] + _OFF) * _K + (ij[:, 1] + _OFF)
        uniq, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        x, y = pts[:, 0], pts[:, 1]
        sx, sy = np.bincount(inv, x), np.bincount(inv, y)
        sxx, sxy, syy = np.bincount(inv, x * x), np.bincount(inv, x * y), np.bincount(inv, y * y)
        mx, my = sx / cnt, sy / cnt
        cxx, cxy, cyy = sxx / cnt - mx * mx, sxy / cnt - mx * my, syy / cnt - my * my
        keep = cnt >= min_pts
        uniq, mx, my, cxx, cxy, cyy = uniq[keep], mx[keep], my[keep], cxx[keep], cxy[keep], cyy[keep]
        C = np.zeros((len(uniq), 2, 2))
        C[:, 0, 0], C[:, 0, 1], C[:, 1, 0], C[:, 1, 1] = cxx, cxy, cxy, cyy
        if len(uniq):
            lam, V = np.linalg.eigh(C)                       # ascending
            lam = np.maximum(lam, np.maximum(min_ratio * lam[:, 1:2], min_var))
            Cinv = np.einsum("nij,nj,nkj->nik", V, 1.0 / lam, V)
        else:
            Cinv = np.zeros((0, 2, 2))
        return uniq, np.stack([mx, my], axis=1), Cinv

    def _lookup(self, g, pts_w):
        uniq, mu, Cinv = self.grids[g]
        shift = self.shifts[g]
        ij = np.floor(pts_w / self.cell - np.asarray(shift)).astype(np.int64)
        key = (ij[:, 0] + _OFF) * _K + (ij[:, 1] + _OFF)
        if len(uniq) == 0:
            return np.zeros(len(pts_w), bool), None, None
        pos = np.searchsorted(uniq, key)
        pos = np.clip(pos, 0, len(uniq) - 1)
        hit = uniq[pos] == key
        return hit, mu[pos[hit]], Cinv[pos[hit]]


@dataclass
class NDTResult:
    pose: np.ndarray
    cost: float
    n_matched: int
    n_points: int
    iters: int
    hessian: np.ndarray

    @property
    def fitness(self) -> float:
        """Mean per-point cost (more negative = better)."""
        return self.cost / max(self.n_points, 1)

    def degeneracy(self) -> float:
        """Ratio  lambda_min / lambda_max  of the translational Hessian block (0 = degenerate)."""
        w = np.linalg.eigvalsh(self.hessian[:2, :2])
        return float(max(w[0], 0.0) / max(w[1], 1e-12))


def _cost_only(src, pose, grid: NDTGrid):
    c, s = np.cos(pose[2]), np.sin(pose[2])
    pw = np.empty_like(src)
    pw[:, 0] = c * src[:, 0] - s * src[:, 1] + pose[0]
    pw[:, 1] = s * src[:, 0] + c * src[:, 1] + pose[1]
    total, nm = 0.0, 0
    for g in range(len(grid.grids)):
        hit, mu, Ci = grid._lookup(g, pw)
        if hit.any():
            e = pw[hit] - mu
            q = np.einsum("ni,nij,nj->n", e, Ci, e)
            total += float(np.sum(grid.d1 * np.exp(-0.5 * grid.d2 * q)))
            nm += int(hit.sum())
    return total, nm


def ndt_register(src: np.ndarray, grid: NDTGrid, init, max_iter: int = 30, tol: float = 1e-5) -> NDTResult:
    """Find pose ``p`` (sensor -> map) minimising the NDT cost of ``src`` (sensor frame points)."""
    pose = np.array(init, float)
    cost, nm = _cost_only(src, pose, grid)
    H = np.eye(3)
    it = 0
    for it in range(1, max_iter + 1):
        c, s = np.cos(pose[2]), np.sin(pose[2])
        pw = np.empty_like(src)
        pw[:, 0] = c * src[:, 0] - s * src[:, 1] + pose[0]
        pw[:, 1] = s * src[:, 0] + c * src[:, 1] + pose[1]
        g = np.zeros(3)
        H = np.zeros((3, 3))
        for gi in range(len(grid.grids)):
            hit, mu, Ci = grid._lookup(gi, pw)
            if not hit.any():
                continue
            e = pw[hit] - mu
            q = np.einsum("ni,nij,nj->n", e, Ci, e)
            w = -grid.d1 * grid.d2 * np.exp(-0.5 * grid.d2 * q)     # >= 0
            sx, sy = src[hit, 0], src[hit, 1]
            J = np.zeros((len(e), 2, 3))
            J[:, 0, 0] = 1.0
            J[:, 1, 1] = 1.0
            J[:, 0, 2] = -s * sx - c * sy
            J[:, 1, 2] = c * sx - s * sy
            CiJ = np.einsum("nij,njk->nik", Ci, J)
            a = np.einsum("nik,ni->nk", J, np.einsum("nij,nj->ni", Ci, e))    # J^T Ci e
            JtCiJ = np.einsum("nik,nil->nkl", J, CiJ)
            g += (w[:, None] * a).sum(0)
            H += (w[:, None, None] * (JtCiJ - grid.d2 * a[:, :, None] * a[:, None, :])).sum(0)
            # second-order term of the rotation
            d2x = np.stack([-(c * sx - s * sy), -(s * sx + c * sy)], axis=1)   # -R x
            H[2, 2] += float((w * np.einsum("ni,nij,nj->n", e, Ci, d2x)).sum())
        # Levenberg-Marquardt style damping until the system is positive definite
        lam = 1e-6 * max(np.trace(H), 1.0)
        for _ in range(30):
            try:
                L = np.linalg.cholesky(H + lam * np.eye(3))
                break
            except np.linalg.LinAlgError:
                lam *= 10.0
        else:
            break
        delta = -np.linalg.solve(H + lam * np.eye(3), g)
        step = 1.0
        improved = False
        for _ in range(8):
            cand = pose + step * delta
            cand[2] = (cand[2] + np.pi) % (2 * np.pi) - np.pi
            c_new, nm_new = _cost_only(src, cand, grid)
            if c_new < cost:
                pose, cost, nm, improved = cand, c_new, nm_new, True
                break
            step *= 0.5
        if not improved or np.linalg.norm(step * delta) < tol:
            break
    return NDTResult(pose, cost, nm, len(src), it, H)


def ndt_register_multires(src, grids, init, **kw) -> NDTResult:
    """Coarse-to-fine registration over a list of ``NDTGrid`` (largest cell first)."""
    pose = init
    res = None
    for g in grids:
        res = ndt_register(src, g, pose, **kw)
        pose = res.pose
    return res
