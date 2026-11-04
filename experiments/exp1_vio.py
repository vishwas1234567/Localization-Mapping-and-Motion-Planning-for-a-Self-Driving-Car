"""Experiment 1 -- Visual-inertial odometry through GPS-denied sections (tunnel, underpass)."""
from common import *  # noqa
from sdc.geometry import pose_errors
from sdc.localization.vio import run_estimator
from sdc.sim.drive import GpsMeas, ImuParams, VoParams, simulate_drive, simulate_gps, simulate_imu, simulate_vo
from sdc.sim.world import World

MODES = ["imu_only", "vo_only", "gps_imu", "vio"]
LABEL = {"imu_only": "IMU only", "vo_only": "Visual odometry only", "gps_imu": "GPS + IMU", "vio": "GPS + IMU + VO (VIO)",
         "vio_nogps": "VIO, GPS only for first 10 s"}


def setup(seed, vo_scale=1.0):
    w = World(seed=7)
    d = simulate_drive(w, laps=1.25, seed=0)
    rng = np.random.default_rng(1000 + seed)
    ip = ImuParams()
    imu, bias = simulate_imu(d, ip, rng)
    vp = VoParams(sigma_t0=0.010 * vo_scale, sigma_t1=0.010 * vo_scale, sigma_r0=0.0015 * vo_scale,
                  sigma_r1=0.002 * vo_scale)
    vo = simulate_vo(d, w, vp, rng)
    gps = simulate_gps(d, w, rng)
    return w, d, ip, imu, vo, gps


def metrics(w, d, est, sig=None):
    err = np.hypot(*(est[:, :2] - d.pose[:, :2]).T)
    lon, lat, yaw = pose_errors(est, d.pose)
    tun, und = w.in_interval(d.s, w.tunnel), w.in_interval(d.s, w.underpass)
    it = np.where(tun)[0]
    out = dict(rmse=float(np.sqrt(np.mean(err ** 2))), rmse_tunnel=float(np.sqrt(np.mean(err[tun] ** 2))),
               max_tunnel=float(err[tun].max()), exit_tunnel=float(err[it[-1]]),
               max_underpass=float(err[und].max()), final=float(err[-1]),
               yaw_rmse_deg=float(np.degrees(np.sqrt(np.mean(yaw ** 2)))),
               lon_tunnel=float(np.sqrt(np.mean(lon[tun] ** 2))), lat_tunnel=float(np.sqrt(np.mean(lat[tun] ** 2))))
    if sig is not None and sig.any():
        out["tunnel_sigma_mean"] = float(sig[tun].mean())
        out["within_3sigma_tunnel"] = float(np.mean(err[tun] < 3 * np.maximum(sig[tun], 1e-6)))
    return out, err


def one_run(seed):
    w, d, ip, imu, vo, gps = setup(seed)
    res = {}
    for m in MODES:
        est, sig = run_estimator(d, imu, ip, gps, vo, m, np.random.default_rng(2000 + seed))
        res[m] = metrics(w, d, est, sig)[0]
    g0 = GpsMeas(gps.idx[gps.idx < 1000], gps.z[gps.idx < 1000], gps.sigma)
    est, sig = run_estimator(d, imu, ip, g0, vo, "vio", np.random.default_rng(2000 + seed))
    r, err = metrics(w, d, est, sig)
    r["drift_pct"] = float(100 * err[-1] / d.s[-1])        # final error / distance driven
    res["vio_nogps"] = r
    return res


def sweep_run(args):
    seed, k = args
    w, d, ip, imu, vo, gps = setup(seed, k)
    est, sig = run_estimator(d, imu, ip, gps, vo, "vio", np.random.default_rng(2000 + seed))
    r = metrics(w, d, est, sig)[0]
    return k, r["exit_tunnel"], r["max_tunnel"]


def main():
    N = 20
    runs = pool_map(one_run, list(range(N)))
    keys = list(runs[0]["vio"].keys())
    table = {}
    for m in MODES + ["vio_nogps"]:
        ks = [k for k in runs[0][m].keys()]
        table[m] = {k: dict(zip(("mean", "std"), ms([r[m][k] for r in runs]))) for k in ks}
    # VO noise sweep
    scales = [0.5, 1.0, 2.0, 4.0]
    sw = pool_map(sweep_run, [(s, k) for k in scales for s in range(6)])
    sweep = {str(k): dict(exit=ms([x[1] for x in sw if x[0] == k]), maxt=ms([x[2] for x in sw if x[0] == k])) for k in scales}
    save_json("exp1_vio", dict(n_runs=N, table=table, vo_noise_sweep=sweep,
                               tunnel_len=float(w_len()), raw_exit={m: [r[m]["exit_tunnel"] for r in runs] for m in MODES}))

    # ------------------ figures (representative run: seed 0) ------------------
    w, d, ip, imu, vo, gps = setup(0)
    ests, errs = {}, {}
    for m in MODES:
        ests[m], sg = run_estimator(d, imu, ip, gps, vo, m, np.random.default_rng(2000))
        errs[m] = np.hypot(*(ests[m][:, :2] - d.pose[:, :2]).T)
        if m == "vio":
            sig_vio = sg
    col = {"imu_only": MUTED, "vo_only": YELLOW, "gps_imu": ORANGE, "vio": BLUE}
    ls = {"imu_only": ":", "vo_only": "-.", "gps_imu": "--", "vio": "-"}
    fig, ax = plt.subplots(1, 2, figsize=(10.2, 3.5), gridspec_kw=dict(width_ratios=[1.0, 1.45]))
    a = ax[0]
    a.plot(*w.route.xy.T, color=GRID, lw=7, solid_capstyle="round", zorder=0)
    tmask = w.in_interval(d.s, w.tunnel)
    a.plot(d.pose[:, 0], d.pose[:, 1], color=GT, lw=1.0, label="ground truth")
    a.plot(d.pose[tmask, 0], d.pose[tmask, 1], color=INK, lw=4.0, alpha=0.18, label="tunnel")
    for m in ("gps_imu", "vio"):
        a.plot(ests[m][:, 0], ests[m][:, 1], color=col[m], ls=ls[m], lw=1.4, label=LABEL[m])
    a.set_aspect("equal"); a.set_xlabel("x [m]"); a.set_ylabel("y [m]"); a.set_title("Route and estimates")
    a.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.20), fontsize=7.5)
    b = ax[1]
    t = d.t
    for m in MODES:
        b.plot(t, np.maximum(errs[m], 1e-2), color=col[m], ls=ls[m], label=LABEL[m])
    b.set_yscale("log"); b.set_ylim(1e-2, 2e2)
    tun_t = t[tmask][[0, -1]]; und_t = t[w.in_interval(d.s, w.underpass)][[0, -1]]
    shade(b, [tun_t, und_t])
    b.text(tun_t.mean(), 1.4e2, "tunnel", ha="center", fontsize=7.5, color=INK2)
    b.text(und_t.mean(), 1.4e2, "underpass", ha="center", fontsize=7.5, color=INK2)
    b.set_xlabel("time [s]"); b.set_ylabel("position error [m]"); b.set_title("Position error (log scale)")
    b.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.20), fontsize=7.5)
    savefig(fig, "fig_vio_overview")

    # exit-error distribution across the Monte-Carlo runs
    fig, a = plt.subplots(figsize=(5.2, 3.0))
    data = [np.array([r[m]["exit_tunnel"] for r in runs]) for m in ("vo_only", "gps_imu", "vio")]
    bp = a.boxplot(data, orientation="horizontal", widths=0.5, patch_artist=True, medianprops=dict(color=INK, lw=1.4),
                   boxprops=dict(lw=0.8, edgecolor=INK2), whiskerprops=dict(color=INK2, lw=0.8), capprops=dict(color=INK2),
                   flierprops=dict(marker="o", ms=3, mfc="none", mec=INK2))
    for p, c in zip(bp["boxes"], (YELLOW, ORANGE, BLUE)):
        p.set_facecolor(c); p.set_alpha(0.55)
    a.set_yticklabels(["VO only", "GPS + IMU", "VIO"]); a.set_xlabel("position error at tunnel exit [m]")
    a.set_title(f"Tunnel-exit error, {N} Monte-Carlo runs")
    savefig(fig, "fig_vio_exit_box")
    print("exp1 done")


def w_len():
    return 130.0


if __name__ == "__main__":
    main()
