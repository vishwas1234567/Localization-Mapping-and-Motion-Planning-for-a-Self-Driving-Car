"""Loosely-coupled visual-inertial odometry as an EKF with *stochastic cloning*.

State (8):  p=(x,y) world position, theta heading, v=(vx,vy) world velocity,
            b_g gyro bias, b_a=(bax,bay) body-frame accelerometer bias.
Clone (3):  the pose at the previous camera frame.  A visual-odometry (VO)
            measurement is the *relative* pose between the clone and the
            current pose; keeping the clone in the state keeps the filter
            consistent (the same information is never used twice).

The same class is used for the three estimators compared in the experiments:
    imu_only : propagation only
    gps_imu  : propagation + GPS position updates
    vio      : propagation + GPS + visual-odometry relative-pose updates
"""
from __future__ import annotations

import numpy as np

from ..geometry import wrap
from ..sim.drive import Drive, GpsMeas, ImuParams, VoMeas

NX = 8
IDX_P, IDX_TH, IDX_V, IDX_BG, IDX_BA = slice(0, 2), 2, slice(3, 5), 5, slice(6, 8)


def _R(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


def _dR(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[-s, -c], [c, -s]])


class VioEKF:
    def __init__(self, pose0, v0, imu: ImuParams, P0_diag=None, use_vo=True):
        self.x = np.zeros(NX)
        self.x[0:2] = pose0[:2]
        self.x[2] = pose0[2]
        self.x[3:5] = v0
        self.use_vo = use_vo
        n = NX + 3
        self.P = np.zeros((n, n))
        if P0_diag is None:
            P0_diag = [0.6 ** 2, 0.6 ** 2, 0.03 ** 2, 0.3 ** 2, 0.3 ** 2,
                       (1.5 * imu.gyro_bias0) ** 2, (1.5 * imu.accel_bias0) ** 2, (1.5 * imu.accel_bias0) ** 2]
        self.P[:NX, :NX] = np.diag(P0_diag)
        self.imu = imu
        self._clone()
        self.last_nis = np.nan

    # ------------------------------------------------------------------
    def _clone(self):
        """(Re)create the clone of the current pose; marginalises the old one."""
        A = np.zeros((NX + 3, NX))
        A[:NX, :] = np.eye(NX)
        A[NX:, :3] = np.eye(3)[:, [0, 1, 2]]
        Pxx = self.P[:NX, :NX]
        self.P = A @ Pxx @ A.T
        self.clone = self.x[:3].copy()

    # ------------------------------------------------------------------
    def propagate(self, imu_meas, dt):
        gz, ax, ay = imu_meas
        x = self.x
        th = x[IDX_TH]
        w = gz - x[IDX_BG]
        ab = np.array([ax - x[6], ay - x[7]])
        R, dR = _R(th), _dR(th)
        aw = R @ ab
        # state update
        x[0:2] += x[3:5] * dt + 0.5 * aw * dt * dt
        x[3:5] += aw * dt
        x[IDX_TH] = wrap(th + w * dt)
        # Jacobian (additive error state)
        F = np.eye(NX)
        F[0:2, 3:5] = np.eye(2) * dt
        F[0:2, 2] = 0.5 * dt * dt * (dR @ ab)
        F[0:2, 6:8] = -0.5 * dt * dt * R
        F[3:5, 2] = dt * (dR @ ab)
        F[3:5, 6:8] = -dt * R
        F[2, 5] = -dt
        q = self.imu
        Q = np.zeros((NX, NX))
        Q[2, 2] = (q.gyro_noise * dt) ** 2
        Q[3:5, 3:5] = np.eye(2) * (q.accel_noise * dt) ** 2
        Q[0:2, 0:2] = np.eye(2) * (0.5 * q.accel_noise * dt * dt) ** 2
        Q[5, 5] = q.gyro_bias_rw ** 2 * dt
        Q[6:8, 6:8] = np.eye(2) * q.accel_bias_rw ** 2 * dt
        P = self.P
        Pxx = F @ P[:NX, :NX] @ F.T + Q
        P[:NX, :NX] = 0.5 * (Pxx + Pxx.T)                      # keep P symmetric (round-off)
        P[:NX, NX:] = F @ P[:NX, NX:]
        P[NX:, :NX] = P[:NX, NX:].T

    # ------------------------------------------------------------------
    def _update(self, y, H, Rm, gate=None):
        S = H @ self.P @ H.T + Rm
        nis = float(y @ np.linalg.solve(S, y))
        self.last_nis = nis
        if gate is not None and nis > gate:
            return False
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ y
        self.x += dx[:NX]
        self.x[IDX_TH] = wrap(self.x[IDX_TH])
        self.clone = self.clone + dx[NX:]
        self.clone[2] = wrap(self.clone[2])
        I = np.eye(self.P.shape[0])
        P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ Rm @ K.T
        self.P = 0.5 * (P + P.T)
        return True

    def update_gps(self, z, sigma):
        H = np.zeros((2, NX + 3))
        H[0, 0] = H[1, 1] = 1.0
        y = z - self.x[0:2]
        return self._update(y, H, np.eye(2) * sigma ** 2, gate=13.8)   # chi2(2) 99.9 %

    def update_vo(self, z, sigma_t, sigma_r, infl=1.3):
        """Relative-pose update between the clone and the current pose, then re-clone."""
        pc, thc = self.clone[:2], self.clone[2]
        p, th = self.x[0:2], self.x[IDX_TH]
        Rc = _R(thc)
        h = np.concatenate([Rc.T @ (p - pc), [wrap(th - thc)]])
        y = z - h
        y[2] = wrap(y[2])
        H = np.zeros((3, NX + 3))
        H[0:2, 0:2] = Rc.T
        H[2, 2] = 1.0
        H[0:2, NX:NX + 2] = -Rc.T
        H[0:2, NX + 2] = _dR(thc).T @ (p - pc)
        H[2, NX + 2] = -1.0
        Rm = np.diag([(infl * sigma_t) ** 2, (infl * sigma_t) ** 2, (infl * sigma_r) ** 2])
        ok = self._update(y, H, Rm, gate=16.3)                       # chi2(3) 99.9 %
        self._clone()
        return ok


# --------------------------------------------------------------------------
def run_estimator(drive: Drive, imu: np.ndarray, imu_params: ImuParams, gps: GpsMeas,
                  vo: VoMeas | None, mode: str, rng: np.random.Generator):
    """Run one estimator over the whole drive.  Returns (N,3) pose estimates and sigma_xy."""
    assert mode in ("imu_only", "gps_imu", "vio", "vo_only")
    n, dt = len(drive.t), drive.dt
    # initialisation from a short GPS fix and approximate heading / speed
    pose0 = drive.pose[0].copy()
    pose0[:2] += rng.normal(0, 0.5, 2)
    pose0[2] += rng.normal(0, 0.02)
    v0 = drive.v[0] * np.array([np.cos(drive.pose[0, 2]), np.sin(drive.pose[0, 2])]) + rng.normal(0, 0.2, 2)
    est = np.zeros((n, 3))
    sig = np.zeros(n)
    if mode == "vo_only":                       # pure visual dead reckoning (interpolated between frames)
        from ..geometry import compose
        pose = pose0.copy()
        key_idx, key_pose = [0], [pose.copy()]
        for j in range(vo.idx.size):
            pose = compose(pose, vo.z[j])
            key_idx.append(int(vo.idx[j]))
            key_pose.append(pose.copy())
        key_idx, key_pose = np.array(key_idx), np.array(key_pose)
        th = np.unwrap(key_pose[:, 2])
        t_all = np.arange(n)
        est[:, 0] = np.interp(t_all, key_idx, key_pose[:, 0])
        est[:, 1] = np.interp(t_all, key_idx, key_pose[:, 1])
        est[:, 2] = (np.interp(t_all, key_idx, th) + np.pi) % (2 * np.pi) - np.pi
        return est, sig
    f = VioEKF(pose0, v0, imu_params, use_vo=(mode == "vio"))
    gps_ptr, vo_ptr = 0, 0
    gps_idx = set(gps.idx.tolist()) if mode in ("gps_imu", "vio") else set()
    gps_map = {int(i): k for k, i in enumerate(gps.idx)}
    vo_map = {int(i): k for k, i in enumerate(vo.idx)} if mode == "vio" else {}
    est[0] = f.x[[0, 1, 2]]
    for i in range(1, n):
        f.propagate(imu[i], dt)
        if i in vo_map:
            k = vo_map[i]
            f.update_vo(vo.z[k], vo.sigma[k, 0], vo.sigma[k, 1])
        if i in gps_idx and i > 0:
            f.update_gps(gps.z[gps_map[i]], gps.sigma)
        est[i] = f.x[[0, 1, 2]]
        sig[i] = np.sqrt(f.P[0, 0] + f.P[1, 1])
    return est, sig
