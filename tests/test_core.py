"""Fast unit tests for the core algorithms (run with `pytest -q`)."""
import numpy as np
import pytest

from sdc.behavior.bt import Action, Condition, Fallback, Sequence, Status
from sdc.geometry import between, compose, inverse, wrap
from sdc.localization.ndt import NDTGrid, ndt_register_multires
from sdc.localization.pose_graph import Edge, optimize
from sdc.planning.dubins import dubins_path, reverse_dubins_path
from sdc.sim.drive import lidar_scan
from sdc.sim.world import World


def test_se2_algebra():
    a, b = np.array([1.0, 2.0, 0.7]), np.array([-3.0, 0.5, -2.0])
    assert np.allclose(compose(a, between(a, b)), b, atol=1e-12)
    assert np.allclose(compose(a, inverse(a)), 0, atol=1e-12)
    assert abs(wrap(3 * np.pi) - np.pi) < 1e-12 or abs(wrap(3 * np.pi) + np.pi) < 1e-12


@pytest.mark.parametrize("fn", [dubins_path, reverse_dubins_path])
def test_dubins_reaches_goal_and_respects_curvature(fn):
    rng, rho = np.random.default_rng(0), 4.7
    for _ in range(100):
        q0 = np.r_[rng.uniform(-20, 20, 2), rng.uniform(-np.pi, np.pi)]
        q1 = np.r_[rng.uniform(-20, 20, 2), rng.uniform(-np.pi, np.pi)]
        pts, L = fn(q0, q1, rho, 0.05)
        assert np.hypot(*(pts[-1, :2] - q1[:2])) < 0.06 and np.hypot(*(pts[0, :2] - q0[:2])) < 0.06
        d = np.diff(np.unwrap(pts[:, 2])); ds = np.hypot(*np.diff(pts[:, :2], axis=0).T)
        assert np.abs(d / np.maximum(ds, 1e-9)).max() < 1.05 / rho


def test_pose_graph_loop_closure_reduces_drift():
    rng, n = np.random.default_rng(0), 80
    gt = np.zeros((n, 3))
    for k in range(1, n):
        gt[k] = compose(gt[k - 1], [1.5, 0, 2 * np.pi / 70])
    om = np.diag([1 / 0.05 ** 2, 1 / 0.05 ** 2, 1 / 0.01 ** 2])
    est, edges = np.zeros((n, 3)), []
    for k in range(1, n):
        z = between(gt[k - 1], gt[k]) + rng.normal(0, [0.03, 0.03, 0.006])
        edges.append(Edge(k - 1, k, z, om)); est[k] = compose(est[k - 1], z)
    loop = Edge(0, n - 1, between(gt[0], gt[n - 1]), np.diag([1 / 0.1 ** 2, 1 / 0.1 ** 2, 1 / 0.02 ** 2]), "loop")
    opt, _ = optimize(est, edges + [loop])
    err = lambda p: np.hypot(*(p[:, :2] - gt[:, :2]).T).max()
    assert err(opt) < 0.5 * err(est)


def test_ndt_recovers_known_offset():
    w, rng = World(seed=7), np.random.default_rng(0)
    A, B = np.array([60.0, 0.0, 0.0]), np.array([61.0, 0.2, 0.02])
    from sdc.geometry import transform_points
    sA, sB = lidar_scan(w, A, rng), lidar_scan(w, B, rng)
    grids = [NDTGrid(transform_points(A, sA), 3.0), NDTGrid(transform_points(A, sA), 1.5)]
    res = ndt_register_multires(sB, grids, B + np.array([0.4, -0.3, 0.05]))
    assert np.hypot(*(res.pose[:2] - B[:2])) < 0.3 and abs(wrap(res.pose[2] - B[2])) < 0.02


def test_bt_priority_and_preemption():
    bb = {"flag": False, "log": []}
    hi = Sequence("hi", [Condition("c", lambda b: b["flag"]), Action("high", lambda b: (b["log"].append("H"), Status.RUNNING)[1])])
    lo = Action("low", lambda b: (b["log"].append("L"), Status.RUNNING)[1])
    root = Fallback("root", [hi, lo])
    root.tick(bb); bb["flag"] = True; root.tick(bb)
    assert bb["log"] == ["L", "H"] and lo.status == Status.IDLE        # low-priority branch was halted
