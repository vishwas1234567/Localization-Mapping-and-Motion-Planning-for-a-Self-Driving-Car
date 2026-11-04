"""Experiment 6 -- Hybrid A* parallel parking: slot-length sweep and search-speed study (heuristic weight)."""
from common import *  # noqa
import sys
from exp5_planning import VEH, jitter, kappa_pose_path
from sdc.geometry import wrap
from sdc.planning.hybrid_astar import FootprintChecker, HybridAStar
from sdc.planning.scenarios import parallel_parking

SLOTS = [5.5, 6.0, 6.5, 7.0, 7.5]
CONFIGS = {"baseline": (1.0, 1e9), "w = 1.5": (1.5, 1e9), "w = 2.0": (2.0, 1e9), "Dubins h <= 10 m": (1.0, 10.0)}
STARTS = range(6)                       # same 6 start poses as E5 (index 0 = nominal)


def run(args):
    slot, ti, cfg = args
    hw, rad = CONFIGS[cfg]
    sc = parallel_parking(VEH, slot=slot)
    chk = FootprintChecker(sc.grid, VEH)
    start = sc.start if ti == 0 else jitter(sc, np.random.default_rng(100 + ti), chk)
    r = HybridAStar(sc.grid, VEH, max_exp=60000, h_weight=hw, nh_radius=rad).plan(start, sc.goal, time_limit=40)
    out = dict(slot=slot, ti=ti, cfg=cfg, found=bool(r.success), feasible=False, time=float(r.time_s), exp=int(r.expansions))
    if r.success:
        p = r.path
        k = kappa_pose_path(p)
        coll = bool((~chk.exact_free(p[:, 0], p[:, 1], p[:, 2])).any())
        end = float(np.hypot(*(p[-1, :2] - sc.goal[:2])))
        eth = float(np.degrees(abs(wrap(p[-1, 2] - sc.goal[2]))))
        out.update(feasible=bool((not coll) and k.max() <= 1.05 * VEH.kappa_max and end < 0.35 and eth < 6),
                   length=float(np.sum(np.hypot(*np.diff(p[:, :2], axis=0).T))),
                   cusps=int(np.sum(p[1:, 3] * p[:-1, 3] < 0)), end_err=end)
    return out


def agg(rows):
    ok = [r for r in rows if r["feasible"]]
    return dict(n=len(rows), feasible=len(ok),
                time_med=float(np.median([r["time"] for r in ok])) if ok else None,
                time_mean=float(np.mean([r["time"] for r in ok])) if ok else None,
                exp_mean=float(np.mean([r["exp"] for r in ok])) if ok else None,
                length_mean=float(np.mean([r["length"] for r in ok])) if ok else None,
                cusps_mean=float(np.mean([r["cusps"] for r in ok])) if ok else None)


def main():
    speed_jobs = [(6.2, ti, c) for c in CONFIGS for ti in STARTS]
    if "--replot" in sys.argv:                          # redraw the figure from results/exp6_hybrid.json
        rows = load_json("exp6_hybrid")["rows"]
    elif "--speed-only" in sys.argv:                      # keep the sweep rows from the previous run
        old = [dict(r, cfg="baseline") for r in load_json("exp6_hybrid")["rows"] if r.get("hw", 1.0) == 1.0 and r.get("cfg", "baseline") == "baseline" and r["slot"] in SLOTS]
        rows = old + pool_map(run, speed_jobs)
    else:
        rows = pool_map(run, [(sl, ti, "baseline") for sl in SLOTS for ti in STARTS] + speed_jobs)
    sweep = {str(sl): agg([r for r in rows if r["slot"] == sl and r["cfg"] == "baseline"]) for sl in SLOTS}
    speed = {c: agg([r for r in rows if r["slot"] == 6.2 and r["cfg"] == c and r in rows[-len(speed_jobs):]]) for c in CONFIGS}
    save_json("exp6_hybrid", dict(slots=SLOTS, configs=list(CONFIGS), sweep=sweep, speed=speed, rows=rows))
    fig, ax = plt.subplots(1, 3, figsize=(11, 2.9))
    a = ax[0]
    a.plot(SLOTS, [sweep[str(s)]["feasible"] for s in SLOTS], "o-", color=BLUE)
    a.set_ylim(-0.3, 6.3); a.set_xlabel("slot length [m] (car 4.5 m)"); a.set_ylabel("feasible starts of 6")
    a.set_title("Success vs slot length")
    b = ax[1]
    for sl in SLOTS:
        t = [r["time"] for r in rows if r["slot"] == sl and r["cfg"] == "baseline" and r["feasible"]]
        b.plot([sl] * len(t), t, "o", color=BLUE, alpha=0.6, ms=4)
    b.set_xticks(SLOTS); b.set_xlabel("slot length [m]"); b.set_ylabel("solve time [s] (feasible runs)"); b.set_title("Search time")
    c = ax[2]
    last = rows[-len(speed_jobs):]
    for j, cf in enumerate(CONFIGS):
        sub = [r for r in last if r["cfg"] == cf]
        c.scatter([j] * len(sub), [r["time"] for r in sub], c=[BLUE if r["feasible"] else ORANGE for r in sub], s=18, zorder=3)
        c.text(j, 44, f"{sum(r['feasible'] for r in sub)}/6", ha="center", fontsize=8, color=INK2)
    c.set_xticks(range(len(CONFIGS))); c.set_xticklabels(list(CONFIGS), fontsize=7, rotation=12); c.set_ylim(0, 48)
    c.set_ylabel("time [s]"); c.set_title("Search speed-up (slot 6.2 m)")
    c.text(0.02, 0.04, "blue = feasible, orange = not", transform=c.transAxes, fontsize=7, color=INK2)
    savefig(fig, "fig_hybrid_sweep")
    print("exp6 done")


if __name__ == "__main__":
    main()
