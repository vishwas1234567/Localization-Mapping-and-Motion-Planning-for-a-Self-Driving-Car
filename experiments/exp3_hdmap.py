"""Experiment 3 -- Localisation against a pre-built HD map: NDT alignment vs particle filter."""
from common import *  # noqa
import time
from sdc.geometry import compose, pose_errors
from sdc.localization.hd_map import survey_map
from sdc.localization.particle_filter import NdtLocalizer, ParticleFilter
from sdc.sim.drive import VoParams, simulate_drive, simulate_lidar, simulate_vo
from sdc.sim.world import World

_W = World(seed=7)
_MAP = None


def hdmap():
    global _MAP
    if _MAP is None:
        sd = simulate_drive(_W, laps=1.0, seed=100, lane_offset=-1.5, wander=1.0)
        _MAP = survey_map(_W, sd, np.random.default_rng(100))
    return _MAP


def data(seed, n_dyn, laps=1.25):
    d = simulate_drive(_W, laps=laps, seed=1 + seed)
    rng = np.random.default_rng(10 + seed)
    dyn = _W.dynamic_objects(n_dyn, np.random.default_rng(20 + seed)) if n_dyn else None
    idx, scans = simulate_lidar(d, _W, rng, extra_segments=dyn)
    vo = simulate_vo(d, _W, VoParams(), rng)
    return d, idx, scans, d.pose[idx], vo


def stats(est, gt):
    lon, lat, yaw = pose_errors(est, gt)
    e = np.hypot(lon, lat)
    return dict(lon_rms=float(np.sqrt(np.mean(lon ** 2))), lat_rms=float(np.sqrt(np.mean(lat ** 2))),
                yaw_rms_deg=float(np.degrees(np.sqrt(np.mean(yaw ** 2)))), p95=float(np.percentile(e, 95)),
                max=float(e.max())), e


def run_ndt(scans, vo, gt):
    nl, est, t0 = NdtLocalizer(hdmap(), gt[0]), [gt[0].copy()], time.time()
    for k in range(1, len(scans)):
        est.append(nl.step(scans[k], vo.z[k - 1])[0].copy())
    return np.array(est), (time.time() - t0) / len(scans) * 1e3


def run_pf(scans, vo, gt, n=500, seed=4, init=None):
    pf = ParticleFilter(hdmap(), n, np.random.default_rng(seed))
    if init is None:
        pf.init_gaussian(gt[0], [1.0, 1.0, 0.05])
    else:
        pf.init_uniform(init)
    est, spread, t0 = [], [], time.time()
    for k in range(len(scans)):
        if k:
            pf.predict(vo.z[k - 1])
        pf.update(scans[k])
        p, s = pf.estimate()
        est.append(p); spread.append(s)
    return np.array(est), np.array(spread), (time.time() - t0) / len(scans) * 1e3


def dead_reckon(vo, gt):
    p, out = gt[0].copy(), [gt[0].copy()]
    for k in range(len(gt) - 1):
        p = compose(p, vo.z[k]); out.append(p.copy())
    return np.array(out)


def job(args):
    seed, n_dyn = args
    d, idx, scans, gt, vo = data(seed, n_dyn)
    r = {}
    est, ms_ndt = run_ndt(scans, vo, gt); r["ndt"], e_ndt = stats(est, gt); r["ndt"]["ms"] = ms_ndt
    est2, _, ms_pf = run_pf(scans, vo, gt); r["pf"], e_pf = stats(est2, gt); r["pf"]["ms"] = ms_pf
    r["vo"], _ = stats(dead_reckon(vo, gt), gt)
    return (seed, n_dyn, r, (d.t[idx], e_ndt, e_pf) if (seed == 0 and n_dyn == 15) else None)


def global_job(seed):
    d, idx, scans, gt, vo = data(seed, 0, laps=0.6)
    sc = scans[:450]; gt = gt[:450]
    w = _W

    def sampler(n, rng):
        s = rng.uniform(0, w.length, n)
        xy, th, _ = w.route.at(s)
        nrm = np.stack([-np.sin(th), np.cos(th)], 1)
        xy = xy + rng.uniform(-3.5, 3.5, n)[:, None] * nrm
        return np.column_stack([xy, rng.uniform(-np.pi, np.pi, n)])
    est, spread, _ = run_pf(sc, vo, gt, n=20000, seed=seed, init=sampler)
    e = np.hypot(*(est[:, :2] - gt[:, :2]).T)
    ok = np.where((e < 0.5) & (spread < 1.0))[0]
    conv = None
    for k in ok:                          # first time it converges and stays converged
        if np.all(e[k:] < 1.0):
            conv = float(k * 0.1); break
    return dict(seed=seed, converge_s=conv, final=float(e[-1])), (d.t[idx][:450], e)


def main():
    hdmap()
    jobs = [(s, n) for s in range(3) for n in (0, 15, 40)]
    out = pool_map(job, jobs)
    table = {}
    for n in (0, 15, 40):
        sub = [o[2] for o in out if o[1] == n]
        table[str(n)] = {m: {k: ms([x[m][k] for x in sub]) for k in sub[0][m]} for m in ("ndt", "pf", "vo")}
    # particle-count sweep (seed 0, 15 unmapped vehicles)
    d, idx, scans, gt, vo = data(0, 15)
    sweep = {}
    for n in (50, 200, 1000, 3000):
        est, _, t = run_pf(scans, vo, gt, n=n); s, _ = stats(est, gt); s["ms"] = t; sweep[str(n)] = s
    gl = pool_map(global_job, [0, 1, 2])
    save_json("exp3_hdmap", dict(table=table, pf_sweep=sweep, global_loc=[g[0] for g in gl],
                                 map_points=int(len(hdmap().points))))
    # figures
    t, e_ndt, e_pf = [o[3] for o in out if o[3] is not None][0]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.0))
    a = ax[0]
    a.plot(t, e_ndt * 100, color=BLUE, lw=1.0, label="NDT"); a.plot(t, e_pf * 100, color=ORANGE, lw=1.0, ls="--", label="particle filter")
    a.set_xlabel("time [s]"); a.set_ylabel("position error [cm]"); a.set_title("Error along the drive"); a.legend(fontsize=7.5)
    b = ax[1]
    lv = [0, 15, 40]
    for j, (m, c, lab) in enumerate((("ndt", BLUE, "NDT"), ("pf", ORANGE, "PF (500)"))):
        b.bar(np.arange(3) + (j - 0.5) * 0.36, [100 * np.hypot(table[str(n)][m]["lon_rms"][0], table[str(n)][m]["lat_rms"][0]) for n in lv],
              0.34, color=c, label=lab)
    b.set_xticks(range(3)); b.set_xticklabels([str(n) for n in lv]); b.set_xlabel("unmapped vehicles in scene")
    b.set_ylabel("position RMS [cm]"); b.set_title("Robustness to map changes"); b.legend(fontsize=7.5)
    c = ax[2]
    for g, col in zip(gl, (BLUE, ORANGE, AQUA)):
        tt, ee = g[1]; c.plot(tt, np.maximum(ee, 1e-2), color=col, lw=1.0, label=f"seed {g[0]['seed']}")
    c.set_yscale("log"); c.set_xlabel("time [s]"); c.set_ylabel("position error [m]")
    c.set_title("Global localisation (20k particles)"); c.legend(fontsize=7.5)
    savefig(fig, "fig_hdmap_overview")
    print("exp3 done")


if __name__ == "__main__":
    main()
