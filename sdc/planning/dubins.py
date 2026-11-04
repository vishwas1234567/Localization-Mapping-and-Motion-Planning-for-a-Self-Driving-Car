"""Dubins shortest paths (forward-only, bounded curvature) -- the analytic "shot" used by Hybrid A*.

Reverse-driving connections are obtained from the same routine by computing the forward path
in the opposite direction (q1 -> q0) and traversing it backwards (time reversal).
"""
from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi


def _m(a):
    return a % TWO_PI


def _words(alpha, beta, d):
    """Candidate (t, p, q) for all six words in the normalised problem; None if infeasible."""
    sa, sb, ca, cb = np.sin(alpha), np.sin(beta), np.cos(alpha), np.cos(beta)
    c_ab = np.cos(alpha - beta)
    out = {}
    # LSL
    p2 = 2 + d * d - 2 * c_ab + 2 * d * (sa - sb)
    if p2 >= 0:
        t1 = np.arctan2(cb - ca, d + sa - sb)
        out["LSL"] = (_m(-alpha + t1), np.sqrt(p2), _m(beta - t1))
    # RSR
    p2 = 2 + d * d - 2 * c_ab + 2 * d * (sb - sa)
    if p2 >= 0:
        t1 = np.arctan2(ca - cb, d - sa + sb)
        out["RSR"] = (_m(alpha - t1), np.sqrt(p2), _m(-beta + t1))
    # LSR
    p2 = -2 + d * d + 2 * c_ab + 2 * d * (sa + sb)
    if p2 >= 0:
        p = np.sqrt(p2)
        t1 = np.arctan2(-ca - cb, d + sa + sb) - np.arctan2(-2.0, p)
        out["LSR"] = (_m(-alpha + t1), p, _m(-_m(beta) + t1))
    # RSL
    p2 = d * d - 2 + 2 * c_ab - 2 * d * (sa + sb)
    if p2 >= 0:
        p = np.sqrt(p2)
        t1 = np.arctan2(ca + cb, d - sa - sb) - np.arctan2(2.0, p)
        out["RSL"] = (_m(alpha - t1), p, _m(beta - t1))
    # RLR
    t0 = (6 - d * d + 2 * c_ab + 2 * d * (sa - sb)) / 8.0
    if abs(t0) <= 1:
        p = _m(TWO_PI - np.arccos(t0))
        t = _m(alpha - np.arctan2(ca - cb, d - sa + sb) + _m(p / 2))
        out["RLR"] = (t, p, _m(alpha - beta - t + _m(p)))
    # LRL
    t0 = (6 - d * d + 2 * c_ab + 2 * d * (sb - sa)) / 8.0
    if abs(t0) <= 1:
        p = _m(TWO_PI - np.arccos(t0))
        t = _m(-alpha - np.arctan2(ca - cb, d + sa - sb) + p / 2)
        out["LRL"] = (t, p, _m(_m(beta) - alpha - t + _m(p)))
    return out


def dubins_shortest(q0, q1, rho):
    """Return (length, word, (t,p,q)) of the shortest Dubins path, or None."""
    dx, dy = q1[0] - q0[0], q1[1] - q0[1]
    D = np.hypot(dx, dy)
    d = D / rho
    theta = _m(np.arctan2(dy, dx)) if D > 1e-12 else 0.0
    alpha, beta = _m(q0[2] - theta), _m(q1[2] - theta)
    best = None
    for w, (t, p, q) in _words(alpha, beta, d).items():
        L = (t + p + q) * rho
        if best is None or L < best[0]:
            best = (L, w, (t, p, q))
    return best


def dubins_sample(q0, rho, word, tpq, step):
    """Sample poses (x,y,theta) every <= ``step`` metres along the path (inclusive of both ends)."""
    x, y, th = float(q0[0]), float(q0[1]), float(q0[2])
    out = [np.array([[x, y, th]])]
    for typ, norm_len in zip(word, tpq):
        length = norm_len * rho                          # t, p, q are normalised by rho
        if length < 1e-9:
            continue
        n = max(int(np.ceil(length / step)), 1)
        sd = np.linspace(0.0, length, n + 1)[1:]
        if typ == "S":
            seg = np.stack([x + sd * np.cos(th), y + sd * np.sin(th), np.full(n, th)], axis=1)
        else:
            sgn = 1.0 if typ == "L" else -1.0
            dth = sgn * sd / rho
            seg = np.stack([x + sgn * rho * (np.sin(th + dth) - np.sin(th)),
                            y - sgn * rho * (np.cos(th + dth) - np.cos(th)), th + dth], axis=1)
        x, y, th = seg[-1]
        out.append(seg)
    pts = np.vstack(out)
    pts[:, 2] = (pts[:, 2] + np.pi) % TWO_PI - np.pi
    return pts


def dubins_path(q0, q1, rho, step=0.3):
    """Forward-only Dubins path poses and its length (None if no path)."""
    r = dubins_shortest(q0, q1, rho)
    if r is None:
        return None
    L, w, tpq = r
    return dubins_sample(q0, rho, w, tpq, step), L


def reverse_dubins_path(q0, q1, rho, step=0.3):
    """Path from q0 to q1 driven in *reverse* (headings are those of the car body).

    Driving backwards from q0 to q1 is the time-reversal of driving *forwards* from q1 to q0,
    so we compute the forward Dubins path q1 -> q0 and traverse it in the opposite order.
    """
    r = dubins_path(q1, q0, rho, step)
    if r is None:
        return None
    pts, L = r
    return pts[::-1].copy(), L
