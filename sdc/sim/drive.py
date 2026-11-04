"""Ground-truth vehicle trajectory along the route + sensor data generation."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geometry import between, wrap
from .world import World


@dataclass
class Drive:
    t: np.ndarray        # (N,) time stamps @ imu rate
    pose: np.ndarray     # (N,3) ground-truth x,y,theta
    v: np.ndarray        # speed
    omega: np.ndarray    # yaw rate
    a_long: np.ndarray   # longitudinal acceleration
    s: np.ndarray        # arclength along the route (monotone, not wrapped)
    dt: float


def simulate_drive(world: World, laps: float = 1.25, dt: float = 0.01, seed: int = 0,
                   lane_offset: float = -1.75, wander: float = 0.5, vmax: float = 11.0,
                   a_lat: float = 2.2, a_acc: float = 1.5, a_dec: float = 2.0,
                   v0: float = 2.0) -> Drive:
    """Drive the lane (road centre + lateral offset + gentle weaving) with a speed
    profile limited by lateral acceleration, longitudinal acceleration and the
    tunnel speed limit.  The profile is computed along the *vehicle path* so that
    speed and acceleration are continuous."""
    rng = np.random.default_rng(seed)
    route, L = world.route, world.length
    ds = 0.1
    s_c = np.arange(0.0, laps * L + 60.0, ds)               # road-centre arclength
    xy_c, th_c, _ = route.at(s_c)
    phi = rng.uniform(0, 2 * np.pi)
    d_lat = lane_offset + wander * np.sin(2 * np.pi * s_c / 90.0 + phi)
    n_vec = np.stack([-np.sin(th_c), np.cos(th_c)], axis=1)
    xy_p = xy_c + d_lat[:, None] * n_vec                     # vehicle path
    kw = 40                                                  # +-4 m low-pass: curvature is continuous
    ker_p = np.ones(2 * kw + 1) / (2 * kw + 1)
    xy_p = np.stack([np.convolve(np.pad(xy_p[:, i], kw, mode="edge"), ker_p, mode="valid")
                     for i in range(2)], axis=1)
    seg = np.hypot(*np.diff(xy_p, axis=0).T)
    sp = np.concatenate([[0.0], np.cumsum(seg)])             # path arclength
    d = np.gradient(xy_p, axis=0)
    theta_p = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    kap = np.gradient(theta_p, sp)
    k = 25
    kap = np.convolve(np.pad(kap, k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), mode="valid")

    v_lim = np.minimum(vmax, np.sqrt(a_lat / (np.abs(kap) + 1e-6)))
    in_tun = world.in_interval(s_c, world.tunnel)
    v_lim = np.where(in_tun, np.minimum(v_lim, 9.0), v_lim)
    v2 = v_lim ** 2
    v2[0] = min(v2[0], v0 ** 2)
    for i in range(1, len(v2)):
        v2[i] = min(v2[i], v2[i - 1] + 2 * a_acc * seg[i - 1])
    for i in range(len(v2) - 2, -1, -1):
        v2[i] = min(v2[i], v2[i + 1] + 2 * a_dec * seg[i])
    v_prof = np.sqrt(v2)

    s_end = laps * L
    sp_list, cur = [0.0], 0.0
    while np.interp(cur, sp, s_c) < s_end:
        cur += np.interp(cur, sp, v_prof) * dt
        sp_list.append(cur)
    sp_t = np.asarray(sp_list)
    t = np.arange(len(sp_t)) * dt
    x = np.interp(sp_t, sp, xy_p[:, 0])
    y = np.interp(sp_t, sp, xy_p[:, 1])
    theta = np.interp(sp_t, sp, theta_p)
    speed = np.interp(sp_t, sp, v_prof)
    kk = 4                                                   # 9-sample moving average
    ker = np.ones(2 * kk + 1) / (2 * kk + 1)
    sm = lambda a: np.convolve(np.pad(a, kk, mode="edge"), ker, mode="valid")
    omega = sm(np.gradient(theta, dt))
    a_long = sm(np.gradient(speed, dt))
    s_arr = np.interp(sp_t, sp, s_c)
    pose = np.column_stack([x, y, wrap(theta)])
    return Drive(t, pose, speed, omega, a_long, s_arr, dt)


# --------------------------------------------------------------------------
# Sensors
# --------------------------------------------------------------------------
@dataclass
class ImuParams:
    gyro_noise: float = 0.002      # rad/s   (white, per sample)
    accel_noise: float = 0.05      # m/s^2   (white, per sample)
    gyro_bias0: float = 0.005      # rad/s   (1-sigma initial bias)
    accel_bias0: float = 0.05      # m/s^2
    gyro_bias_rw: float = 1e-4     # rad/s / sqrt(s)
    accel_bias_rw: float = 2e-3    # m/s^2 / sqrt(s)


def simulate_imu(drive: Drive, p: ImuParams, rng: np.random.Generator):
    """Planar IMU: gyro-z and body-frame (x,y) specific force.  Returns (N,3) and true biases."""
    n, dt = len(drive.t), drive.dt
    bg = np.cumsum(np.r_[rng.normal(0, p.gyro_bias0), rng.normal(0, p.gyro_bias_rw * np.sqrt(dt), n - 1)])
    bax = np.cumsum(np.r_[rng.normal(0, p.accel_bias0), rng.normal(0, p.accel_bias_rw * np.sqrt(dt), n - 1)])
    bay = np.cumsum(np.r_[rng.normal(0, p.accel_bias0), rng.normal(0, p.accel_bias_rw * np.sqrt(dt), n - 1)])
    gz = drive.omega + bg + rng.normal(0, p.gyro_noise, n)
    ax = drive.a_long + bax + rng.normal(0, p.accel_noise, n)
    ay = drive.v * drive.omega + bay + rng.normal(0, p.accel_noise, n)
    return np.column_stack([gz, ax, ay]), np.column_stack([bg, bax, bay])


@dataclass
class VoParams:
    """Stereo visual-odometry error model (relative pose between consecutive frames)."""
    sigma_t0: float = 0.010
    sigma_t1: float = 0.010       # proportional to travelled distance
    sigma_r0: float = 0.0015
    sigma_r1: float = 0.002       # proportional to |dtheta|
    scale_err_std: float = 0.004  # systematic scale error (calibration)
    rate_div: int = 10            # camera frame every rate_div IMU samples (10 Hz)


@dataclass
class VoMeas:
    idx: np.ndarray    # IMU sample index of the *current* frame
    z: np.ndarray      # (M,3) relative pose [dx,dy,dtheta] in previous body frame
    sigma: np.ndarray  # (M,2) reported (sigma_t, sigma_r)


def simulate_vo(drive: Drive, world: World, p: VoParams, rng: np.random.Generator) -> VoMeas:
    idx = np.arange(p.rate_div, len(drive.t), p.rate_div)
    scale = 1.0 + rng.normal(0, p.scale_err_std)
    zs, sig = [], []
    for i in idx:
        j = i - p.rate_div
        d = between(drive.pose[j], drive.pose[i])
        f = world.vision_factor(drive.s[i])
        dist = np.hypot(d[0], d[1])
        st = f * (p.sigma_t0 + p.sigma_t1 * dist)
        sr = f * (p.sigma_r0 + p.sigma_r1 * abs(d[2]))
        z = np.array([d[0] * scale + rng.normal(0, st), d[1] * scale + rng.normal(0, st),
                      d[2] + rng.normal(0, sr)])
        zs.append(z)
        sig.append([st, sr])
    return VoMeas(idx, np.asarray(zs), np.asarray(sig))


@dataclass
class GpsMeas:
    idx: np.ndarray
    z: np.ndarray       # (M,2)
    sigma: float


def simulate_gps(drive: Drive, world: World, rng: np.random.Generator, sigma: float = 0.6,
                 rate_div: int = 100) -> GpsMeas:
    idx = np.arange(0, len(drive.t), rate_div)
    keep = np.array([world.gps_available(drive.s[i]) for i in idx])
    idx = idx[keep]
    z = drive.pose[idx, :2] + rng.normal(0, sigma, (len(idx), 2))
    return GpsMeas(idx, z, sigma)


# --------------------------------------------------------------------------
# LiDAR (2-D, 360 beams) with vectorised ray casting
# --------------------------------------------------------------------------
def raycast(origin, angles, segs, circs, max_range):
    d = np.stack([np.cos(angles), np.sin(angles)], axis=1)     # (B,2)
    best = np.full(len(angles), np.inf)
    if len(segs):
        mid = 0.5 * (segs[:, :2] + segs[:, 2:])
        rad = 0.5 * np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
        keep = np.hypot(mid[:, 0] - origin[0], mid[:, 1] - origin[1]) < max_range + rad
        sg = segs[keep]
        if len(sg):
            p = sg[:, :2]
            e = sg[:, 2:] - sg[:, :2]
            w = p - origin[:2]
            denom = d[:, 0:1] * e[None, :, 1] - d[:, 1:2] * e[None, :, 0]
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (w[:, 0] * e[:, 1] - w[:, 1] * e[:, 0])[None, :] / denom
                u = (w[:, 0][None, :] * d[:, 1:2] - w[:, 1][None, :] * d[:, 0:1]) / denom
            ok = (np.abs(denom) > 1e-9) & (t > 0) & (u >= 0) & (u <= 1)
            t = np.where(ok, t, np.inf)
            best = np.minimum(best, t.min(axis=1))
    if len(circs):
        keep = np.hypot(circs[:, 0] - origin[0], circs[:, 1] - origin[1]) < max_range + circs[:, 2]
        cc = circs[keep]
        if len(cc):
            oc = origin[:2] - cc[:, :2]                       # (K,2)
            b = d @ oc.T                                       # (B,K)
            cterm = (oc ** 2).sum(1) - cc[:, 2] ** 2
            disc = b ** 2 - cterm[None, :]
            with np.errstate(invalid="ignore"):
                t = -b - np.sqrt(disc)
            ok = (disc >= 0) & (t > 0)
            t = np.where(ok, t, np.inf)
            best = np.minimum(best, t.min(axis=1))
    best[best > max_range] = np.inf
    return best


@dataclass
class LidarParams:
    n_beams: int = 360
    max_range: float = 40.0
    sigma0: float = 0.02
    sigma1: float = 0.001     # range proportional noise
    dropout: float = 0.02


def lidar_scan(world: World, pose, rng: np.random.Generator, p: LidarParams = LidarParams(),
               extra_segments: np.ndarray | None = None, noise: bool = True) -> np.ndarray:
    """Return the (M,2) point cloud in the sensor (= body) frame."""
    segs = world.segments if extra_segments is None or len(extra_segments) == 0 else \
        np.concatenate([world.segments, extra_segments], axis=0)
    ang_body = np.linspace(-np.pi, np.pi, p.n_beams, endpoint=False)
    r = raycast(pose, ang_body + pose[2], segs, world.circles, p.max_range)
    ok = np.isfinite(r)
    if noise:
        ok &= rng.random(len(r)) > p.dropout
        r = r + rng.normal(0, 1, len(r)) * (p.sigma0 + p.sigma1 * np.where(ok, r, 0))
    a, r = ang_body[ok], r[ok]
    return np.stack([r * np.cos(a), r * np.sin(a)], axis=1)


def simulate_lidar(drive: Drive, world: World, rng: np.random.Generator, rate_div: int = 10,
                   extra_segments=None, p: LidarParams = LidarParams()):
    """Scans at 10 Hz.  Returns (indices, list of point clouds)."""
    idx = np.arange(0, len(drive.t), rate_div)
    scans = [lidar_scan(world, drive.pose[i], rng, p, extra_segments) for i in idx]
    return idx, scans
