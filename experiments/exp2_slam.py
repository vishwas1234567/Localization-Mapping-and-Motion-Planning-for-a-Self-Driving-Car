"""Experiment 2 -- LiDAR SLAM: ICP odometry vs NDT scan-to-submap vs NDT + loop closure (pose graph)."""
from common import *  # noqa
from scipy.spatial import cKDTree
from sdc.geometry import compose
from sdc.localization.lidar_slam import LidarSlam, SlamConfig, icp_odometry
from sdc.sim.drive import lidar_scan, simulate_drive, simulate_lidar
from sdc.sim.world import World

SEEDS = [0, 1, 2]


def ate(est, gt):
    e = np.hypot(*(est[:, :2] - gt[:, :2]).T)
    return dict(rmse=float(np.sqrt(np.mean(e ** 2))), max=float(e.max()), final=float(e[-1])), e


def map_error(points, ref_tree):
    d, _ = ref_tree.query(points)
    return dict(mean=float(d.mean()), p95=float(np.percentile(d, 95)))


def one(seed):
    w = World(seed=7)
    d = simulate_drive(w, laps=1.25, seed=seed)
    rng = np.random.default_rng(seed)
    idx, scans = simulate_lidar(d, w, rng)
    gt = d.pose[idx]
    ref = np.concatenate([lidar_scan(w, d.pose[i], rng, noise=False) @ np.array([[np.cos(d.pose[i, 2]), np.sin(d.pose[i, 2])],
                          [-np.sin(d.pose[i, 2]), np.cos(d.pose[i, 2])]]) + d.pose[i, :2] for i in idx[::2]])
    tree = cKDTree(ref)
    out, keep = {}, {}
    ic = icp_odometry(scans, gt[0])
    out["icp"], e = ate(ic, gt)
    out["icp"]["yaw_final_deg"] = float(np.degrees(abs((ic[-1, 2] - gt[-1, 2] + np.pi) % (2 * np.pi) - np.pi)))
    keep["icp"] = ic
    for name, loop in (("ndt_odom", False), ("ndt_loop", True)):
        slam = LidarSlam(gt[0], SlamConfig(loop=loop))
        t0 = time.time()
        for k, s in enumerate(scans):
            slam.process(s, k)
        if loop:
            slam.optimize()
        dt = (time.time() - t0) / len(scans) * 1e3
        kp, ki = slam.keyframe_poses()
        out[name], e = ate(kp, gt[ki])
        out[name]["ms_per_scan"] = float(dt)
        out[name]["loops"] = int(slam.n_loops)
        out[name]["keyframes"] = int(len(kp))
        out[name]["map"] = map_error(slam.global_map(0.2), tree)
        keep[name] = (kp, ki)
        if name == "ndt_odom":
            deg = np.array([r["deg"] for r in slam.scan_log])
            sidx = np.array([r["idx"] for r in slam.scan_log])
            keep["deg"] = (np.mod(d.s[idx][sidx], w.length), deg)
        else:
            keep["map"] = slam.global_map(0.25)
            keep["loops"] = [(slam.kfs[a].pose[:2].copy(), slam.kfs[b].pose[:2].copy()) for a, b in slam.loop_pairs]
    keep["gt"], keep["idx"], keep["s"] = gt, idx, d.s[idx]
    return out, (keep if seed == 0 else None)


def main():
    res = pool_map(one, SEEDS)
    table = {}
    for m in ("icp", "ndt_odom", "ndt_loop"):
        table[m] = {k: (ms([r[0][m][k] for r in res]) if k != "map" else
                        {kk: ms([r[0][m]["map"][kk] for r in res]) for kk in ("mean", "p95")})
                    for k in res[0][0][m]}
    k0 = res[0][1]
    L = World(seed=7).length
    sdeg, deg = k0["deg"]
    tun = (sdeg >= 200) & (sdeg <= 330)
    table["degeneracy"] = dict(tunnel_median=float(np.median(deg[tun])), outside_median=float(np.median(deg[~tun])))
    save_json("exp2_slam", dict(seeds=SEEDS, table=table))

    gt = k0["gt"]
    kp, ki = k0["ndt_loop"]
    ko, kio = k0["ndt_odom"]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4), gridspec_kw=dict(width_ratios=[1.1, 1.1, 1.2]))
    a = ax[0]
    a.plot(gt[:, 0], gt[:, 1], color=GT, lw=1.0, label="ground truth")
    a.plot(k0["icp"][:, 0], k0["icp"][:, 1], color=YELLOW, ls="-.", label="ICP odometry")
    a.plot(ko[:, 0], ko[:, 1], color=ORANGE, ls="--", label="NDT odometry")
    a.plot(kp[:, 0], kp[:, 1], color=BLUE, label="NDT + loop closure")
    a.set_aspect("equal"); a.set_title("Estimated trajectories"); a.set_xlabel("x [m]"); a.set_ylabel("y [m]")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=7)
    b = ax[1]
    mp = k0["map"]
    b.scatter(mp[:, 0], mp[:, 1], s=0.15, color=INK2, alpha=0.6, rasterized=True)
    for p, q in k0["loops"][:60]:
        b.plot([p[0], q[0]], [p[1], q[1]], color=BLUE, lw=0.4, alpha=0.7)
    b.set_aspect("equal"); b.set_title("SLAM map and loop-closure edges"); b.set_xlabel("x [m]")
    c = ax[2]
    c.plot(k0["s"], np.hypot(*(k0["icp"][:, :2] - gt[:, :2]).T), color=YELLOW, ls="-.", label="ICP odometry")
    c.plot(k0["s"][kio], np.hypot(*(ko[:, :2] - gt[kio, :2]).T), color=ORANGE, ls="--", label="NDT odometry")
    c.plot(k0["s"][ki], np.hypot(*(kp[:, :2] - gt[ki, :2]).T), color=BLUE, label="NDT + loop closure")
    c.set_yscale("log"); c.set_xlabel("distance driven [m]"); c.set_ylabel("position error [m]")
    c.set_title("Absolute trajectory error"); c.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=7)
    savefig(fig, "fig_slam_overview")

    fig, a = plt.subplots(figsize=(5.4, 2.6))
    a.plot(sdeg, deg, color=BLUE, lw=0.8)
    shade(a, [(200, 330)]); a.set_ylim(0, max(0.7, deg.max()))
    a.text(265, a.get_ylim()[1] * 0.93, "tunnel", ha="center", fontsize=7.5, color=INK2)
    a.set_xlabel("position along route [m]"); a.set_ylabel(r"$\lambda_{min}/\lambda_{max}$")
    a.set_title("NDT Hessian conditioning (low = weakly constrained)")
    savefig(fig, "fig_slam_degeneracy")
    print("exp2 done")


if __name__ == "__main__":
    import time
    main()
