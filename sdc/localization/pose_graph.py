"""SE(2) pose-graph optimisation (vectorised Gauss-Newton on a sparse system)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..geometry import wrap


@dataclass
class Edge:
    i: int
    j: int
    z: np.ndarray       # measured pose of j expressed in the frame of i
    omega: np.ndarray   # 3x3 information matrix
    kind: str = "odom"


def _linearize(x, I, J, Z):
    """Residuals e (E,3) and Jacobians A=de/dxi, B=de/dxj for all edges at once."""
    xi, xj = x[I], x[J]
    ci, si = np.cos(xi[:, 2]), np.sin(xi[:, 2])
    cz, sz = np.cos(Z[:, 2]), np.sin(Z[:, 2])
    dx, dy = xj[:, 0] - xi[:, 0], xj[:, 1] - xi[:, 1]
    zt0, zt1 = ci * dx + si * dy, -si * dx + ci * dy               # Ri^T (tj - ti)
    r0, r1 = zt0 - Z[:, 0], zt1 - Z[:, 1]
    e = np.stack([cz * r0 + sz * r1, -sz * r0 + cz * r1, wrap(xj[:, 2] - xi[:, 2] - Z[:, 2])], axis=1)
    E = len(I)
    RzT = np.zeros((E, 2, 2)); RzT[:, 0, 0] = cz; RzT[:, 0, 1] = sz; RzT[:, 1, 0] = -sz; RzT[:, 1, 1] = cz
    RiT = np.zeros((E, 2, 2)); RiT[:, 0, 0] = ci; RiT[:, 0, 1] = si; RiT[:, 1, 0] = -si; RiT[:, 1, 1] = ci
    dRiT_dt = np.stack([-si * dx + ci * dy, -ci * dx - si * dy], axis=1)   # d(Ri^T)/dth_i (tj - ti)
    RzRi = np.einsum("eij,ejk->eik", RzT, RiT)
    A = np.zeros((E, 3, 3)); B = np.zeros((E, 3, 3))
    A[:, :2, :2] = -RzRi
    A[:, :2, 2] = np.einsum("eij,ej->ei", RzT, dRiT_dt)
    A[:, 2, 2] = -1.0
    B[:, :2, :2] = RzRi
    B[:, 2, 2] = 1.0
    return e, A, B


def optimize(poses: np.ndarray, edges: list, iters: int = 10, tol: float = 1e-6, robust: float | None = 3.0):
    """Optimise ``poses`` (N,3) holding node 0 fixed.  Returns (poses, chi2)."""
    x = poses.copy()
    n = len(x)
    I = np.array([e.i for e in edges]); J = np.array([e.j for e in edges])
    Z = np.array([e.z for e in edges]); Om = np.array([e.omega for e in edges])
    is_loop = np.array([e.kind == "loop" for e in edges])
    ar = np.arange(3)
    chi2 = np.inf
    for _ in range(iters):
        e, A, B = _linearize(x, I, J, Z)
        c2 = np.einsum("ei,eij,ej->e", e, Om, e)
        w = np.ones(len(edges))
        if robust is not None:
            big = is_loop & (c2 > robust ** 2)
            w[big] = robust / np.sqrt(c2[big])
        chi2 = float(np.sum(w * c2))
        Omw = Om * w[:, None, None]
        Hii = np.einsum("eki,ekl,elj->eij", A, Omw, A)
        Hij = np.einsum("eki,ekl,elj->eij", A, Omw, B)
        Hjj = np.einsum("eki,ekl,elj->eij", B, Omw, B)
        bi = np.einsum("eki,ekl,el->ei", A, Omw, e)
        bj = np.einsum("eki,ekl,el->ei", B, Omw, e)
        rows, cols, vals = [], [], []
        for P_, Q_, M in ((I, I, Hii), (I, J, Hij), (J, I, np.transpose(Hij, (0, 2, 1))), (J, J, Hjj)):
            rows.append((3 * P_[:, None, None] + ar[None, :, None] + 0 * ar[None, None, :]).ravel())
            cols.append((3 * Q_[:, None, None] + ar[None, None, :] + 0 * ar[None, :, None]).ravel())
            vals.append(M.ravel())
        H = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(3 * n, 3 * n)).tocsc()
        b = np.zeros(3 * n)
        np.add.at(b, (3 * I[:, None] + ar[None, :]).ravel(), bi.ravel())
        np.add.at(b, (3 * J[:, None] + ar[None, :]).ravel(), bj.ravel())
        H = H + sp.identity(3 * n, format="csc") * 1e-9
        H = H + sp.csc_matrix((np.full(3, 1e9), (ar, ar)), shape=(3 * n, 3 * n))      # anchor node 0
        dx = spla.spsolve(H, -b)
        x += dx.reshape(n, 3)
        x[:, 2] = wrap(x[:, 2])
        if np.linalg.norm(dx) < tol:
            break
    return x, chi2
