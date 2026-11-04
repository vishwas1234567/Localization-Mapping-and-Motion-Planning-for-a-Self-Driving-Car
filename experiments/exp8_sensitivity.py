"""Experiment 8 -- 20-seed statistics with bootstrap CIs: E2 (SLAM) and E3 (HD-map localisation) under LiDAR degradation."""
from common import *  # noqa
import sys
import exp2_slam as e2
import exp3_hdmap as e3
from sdc.sim.drive import LidarParams, VoParams, simulate_drive, simulate_lidar, simulate_vo

N = 20
LP = {"nominal": LidarParams(), "noise x4": LidarParams(sigma0=0.08, sigma1=0.004),
      "dropout 40%": LidarParams(dropout=0.40)}


def boot(x, n=5000, seed=0):
    """Mean and 95% percentile-bootstrap CI of the mean."""
    x = np.asarray(x, float)
    m = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return dict(mean=float(x.mean()), lo=float(np.percentile(m, 2.5)), hi=float(np.percentile(m, 97.5)),
                median=float(np.median(x)), std=float(x.std(ddof=1)), n=len(x))


def slam_job(seed):
    out, _ = e2.one(seed)
    return {m: out[m]["rmse"] for m in ("icp", "ndt_odom", "ndt_loop")}


def loc_job(args):
    cfg, seed = args
    w = e3._W
    d = simulate_drive(w, laps=1.25, seed=1 + seed)
    rng = np.random.default_rng(10 + seed)
    dyn = w.dynamic_objects(15, np.random.default_rng(20 + seed))
    idx, scans = simulate_lidar(d, w, rng, extra_segments=dyn, p=LP[cfg])
    vo = simulate_vo(d, w, VoParams(), rng)
    gt = d.pose[idx]
    est, _ = e3.run_ndt(scans, vo, gt)
    sn, _ = e3.stats(est, gt)
    est2, _, _ = e3.run_pf(scans, vo, gt)
    sp, _ = e3.stats(est2, gt)
    r = lambda s: float(np.hypot(s["lon_rms"], s["lat_rms"]))
    return dict(cfg=cfg, seed=seed, ndt=r(sn), pf=r(sp), ndt_max=sn["max"], pf_max=sp["max"])


def main():
    if "--replot" in sys.argv:                          # redraw the figure from results/exp8_sensitivity.json
        e8 = load_json("exp8_sensitivity")
        slam, s_stats, l_stats = e8["slam"]["raw"], e8["slam"], e8["loc"]
    else:
        e3.hdmap()
        slam = pool_map(slam_job, list(range(N)))
        s_stats = {m: boot([r[m] for r in slam]) for m in ("icp", "ndt_odom", "ndt_loop")}
        s_stats["raw"] = slam
        loc = pool_map(loc_job, [(c, s) for c in LP for s in range(N)])
        l_stats = {}
        for c in LP:
            sub = [r for r in loc if r["cfg"] == c]
            l_stats[c] = {m: boot([r[m] for r in sub]) for m in ("ndt", "pf")}
            l_stats[c]["ndt_fail"] = int(sum(r["ndt_max"] > 1.0 for r in sub))
            l_stats[c]["pf_fail"] = int(sum(r["pf_max"] > 1.0 for r in sub))
    if "--replot" not in sys.argv:
        save_json("exp8_sensitivity", dict(n=N, slam=s_stats, loc=l_stats, loc_raw=loc))
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.0))
    a = ax[0]
    for j, (m, lab, c) in enumerate((("icp", "ICP odometry", YELLOW), ("ndt_odom", "NDT odometry", ORANGE), ("ndt_loop", "NDT + loop", BLUE))):
        v = [r[m] for r in slam]
        a.scatter(np.full(len(v), j) + np.random.default_rng(j).uniform(-0.12, 0.12, len(v)), v, s=10, color=c, alpha=0.7, zorder=3)
        a.errorbar(j, s_stats[m]["mean"], yerr=[[s_stats[m]["mean"] - s_stats[m]["lo"]], [s_stats[m]["hi"] - s_stats[m]["mean"]]],
                   color=INK, capsize=4, marker="_", ms=14, zorder=4)
    a.set_yscale("log"); a.set_xticks(range(3)); a.set_xticklabels(["ICP", "NDT odom.", "NDT + loop"])
    a.set_ylabel("ATE RMSE [m]"); a.set_title(f"E2, {N} seeds (mean, 95% CI)")
    b = ax[1]
    for j, (m, lab, c) in enumerate((("ndt", "NDT", BLUE), ("pf", "PF (500)", ORANGE))):
        for i, cfg in enumerate(LP):
            s = l_stats[cfg][m]
            b.errorbar(i + (j - 0.5) * 0.18, 100 * s["mean"], yerr=[[100 * (s["mean"] - s["lo"])], [100 * (s["hi"] - s["mean"])]],
                       color=c, marker="o", ms=4, capsize=3, label=lab if i == 0 else None)
    b.set_ylim(0, 5.5); b.set_xticks(range(len(LP))); b.set_xticklabels(list(LP)); b.set_ylabel("position RMS [cm]")
    b.set_title(f"E3 under LiDAR degradation, {N} seeds"); b.legend()
    savefig(fig, "fig_sensitivity")
    print("exp8 done")


if __name__ == "__main__":
    import time  # noqa  (exp2_slam.one uses `time` from the script namespace)
    e2.time = time
    main()
