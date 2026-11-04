"""Experiment 5 -- Motion planning: Hybrid A* vs grid A* (parking, K-turn); Frenet state lattice vs linear ramps."""
from common import *  # noqa
import time
from matplotlib.patches import Polygon
from sdc.planning.hybrid_astar import *  # noqa
from sdc.planning.lattice import Frenet, StateLattice, run_receding_horizon
from sdc.planning.scenarios import k_turn, parallel_parking, perpendicular_parking
from sdc.sim.world import World

VEH = Vehicle()
MAKERS = [perpendicular_parking, parallel_parking, k_turn]
NTRIALS = 6


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def kappa_pose_path(p):
    """|curvature| along a pose path, ignoring cusps (direction changes)."""
    ds = np.hypot(*np.diff(p[:, :2], axis=0).T)
    dth = np.abs(wrap(np.diff(p[:, 2])))
    ok = (ds > 1e-3) & (p[1:, 3] == p[:-1, 3] if p.shape[1] > 3 else True)
    return dth[ok] / ds[ok]


def jitter(sc, rng, chk):
    for _ in range(50):
        s = sc.start + rng.uniform(-1, 1, 3) * np.array(sc.start_jitter)
        if chk.free(*s) and chk.exact_free(*[np.array([v]) for v in s])[0]:
            return s
    return sc.start


def evaluate_grid(sc, start, chk, smooth):
    xy, nexp = grid_astar(sc.grid, start[:2], sc.goal[:2], clearance=VEH.width / 2 + 0.05)
    if xy is None:
        return dict(found=False, feasible=False)
    if smooth:
        xy = chaikin(xy, 3)
    d = np.diff(xy, axis=0)
    th = np.arctan2(d[:, 1], d[:, 0]); th = np.r_[th, th[-1]]
    m = path_metrics(xy, kappa_max=VEH.kappa_max)
    coll = float(np.mean(~chk.exact_free(xy[:, 0], xy[:, 1], th)))
    gerr = float(np.degrees(abs(wrap(th[-1] - sc.goal[2])))); serr = float(np.degrees(abs(wrap(th[0] - start[2]))))
    feas = coll == 0 and m["max_kappa"] <= 1.05 * VEH.kappa_max and gerr < 10 and serr < 10
    return dict(found=True, feasible=bool(feas), length=m["length"], max_kappa=m["max_kappa"], corners=m["corners"],
                collision_frac=coll, goal_heading_err_deg=gerr)


def trial(args):
    mi, ti = args
    sc = MAKERS[mi](VEH)
    chk = FootprintChecker(sc.grid, VEH)
    rng = np.random.default_rng(100 * mi + ti)
    start = sc.start if ti == 0 else jitter(sc, rng, chk)
    hp = HybridAStar(sc.grid, VEH, max_exp=60000)
    r = hp.plan(start, sc.goal, time_limit=40)
    out = dict(mi=mi, ti=ti, grid=evaluate_grid(sc, start, chk, False), grid_smooth=evaluate_grid(sc, start, chk, True))
    if r.success:
        p = r.path
        k = kappa_pose_path(p)
        dirs = p[1:, 3]
        coll = bool((~chk.exact_free(p[:, 0], p[:, 1], p[:, 2])).any())
        end = float(np.hypot(*(p[-1, :2] - sc.goal[:2]))); eth = float(np.degrees(abs(wrap(p[-1, 2] - sc.goal[2]))))
        out["hybrid"] = dict(found=True, feasible=bool((not coll) and k.max() <= 1.05 * VEH.kappa_max and end < 0.35 and eth < 6),
                             time=r.time_s, expansions=r.expansions, length=float(np.sum(np.hypot(*np.diff(p[:, :2], axis=0).T))),
                             cusps=int(np.sum(dirs[1:] * dirs[:-1] < 0)), max_kappa=float(k.max()), end_err=end, end_heading_err_deg=eth,
                             collision=coll, shot=bool(r.used_shot))
        if ti == 0:
            out["path"] = p
    else:
        out["hybrid"] = dict(found=False, feasible=False, time=r.time_s, expansions=r.expansions)
    return out


def lattice_mc(n=20):
    w = World(seed=7); fr = Frenet(w.route)
    res = {"quintic": [], "linear": []}
    for seed in range(n):
        rng = np.random.default_rng(seed)
        obs = []
        for s in np.sort(rng.uniform(60, 330, rng.integers(5, 9))):
            d = rng.choice([-1.75, -1.75, -1.0, -2.6, 1.75])
            if rng.random() < 0.6:
                for o in (-1.5, 0.0, 1.5):
                    obs.append([*fr.to_world(np.array([s + o]), np.array([d]))[0][0], 1.0])
            else:
                obs.append([*fr.to_world(np.array([s]), np.array([d]))[0][0], rng.uniform(0.3, 0.5)])
        obs = np.array(obs)
        for prof in res:
            pl = StateLattice(w.route, obs, profile=prof)
            r = run_receding_horizon(w.route, pl, 30.0, 360.0)
            cl = pl.clearance(r["pos"], r["theta"])
            res[prof].append(dict(clearance=float(cl), collision_free=bool(cl > 0), fails=int(r["fails"]),
                                  max_kappa=float(np.abs(r["kappa"]).max()),
                                  max_dkappa=float(np.max(np.abs(np.diff(r["kappa"]) / np.maximum(np.diff(r["arclen"]), 1e-3)))),
                                  ok=bool(cl > 0 and np.abs(r["kappa"]).max() <= 1.05 * VEH.kappa_max)))
    return res


def main():
    out = pool_map(trial, [(m, t) for m in range(3) for t in range(NTRIALS)])
    table = {}
    for m, mk in enumerate(MAKERS):
        sub = [o for o in out if o["mi"] == m]
        row = {}
        for meth in ("hybrid", "grid", "grid_smooth"):
            rs = [o[meth] for o in sub]
            row[meth] = dict(found=sum(r["found"] for r in rs), feasible=sum(r["feasible"] for r in rs), n=len(rs))
            for k in ("time", "expansions", "length", "cusps", "max_kappa", "corners", "collision_frac", "goal_heading_err_deg", "end_err"):
                v = [r[k] for r in rs if r.get("found") and k in r]
                if v:
                    row[meth][k] = float(np.mean(v))
        table[mk(VEH).name] = row
    mc = lattice_mc()
    lat = {p: dict(ok=sum(r["ok"] for r in rs), n=len(rs), collision_free=sum(r["collision_free"] for r in rs),
                   max_kappa=ms([r["max_kappa"] for r in rs]), max_dkappa=ms([r["max_dkappa"] for r in rs]),
                   plan_fails=sum(r["fails"] for r in rs)) for p, rs in mc.items()}
    save_json("exp5_planning", dict(parking=table, lattice=lat, kappa_limit=VEH.kappa_max, rho=VEH.rho))

    # figure: Hybrid A* solutions (nominal start) with grid A* overlay
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.3))
    for m, mk in enumerate(MAKERS):
        sc = mk(VEH); a = ax[m]
        a.imshow(sc.grid.occ.T, origin="lower", extent=[sc.grid.x0, sc.grid.x0 + sc.grid.nx * sc.grid.res, sc.grid.y0, sc.grid.y0 + sc.grid.ny * sc.grid.res],
                 cmap="Greys", alpha=0.35, interpolation="nearest")
        xy, _ = grid_astar(sc.grid, sc.start[:2], sc.goal[:2], VEH.width / 2 + 0.05)
        if xy is not None:
            a.plot(xy[:, 0], xy[:, 1], color=ORANGE, ls="--", lw=1.0, label="grid A*")
        o = [x for x in out if x["mi"] == m and x["ti"] == 0][0]
        if "path" in o:
            p = o["path"]; a.plot(p[:, 0], p[:, 1], color=BLUE, lw=1.4, label="Hybrid A*")
            for k in np.linspace(0, len(p) - 1, 9).astype(int):
                a.add_patch(Polygon(VEH.corners(p[k, :3]), fill=False, ec=BLUE, lw=0.5, alpha=0.7))
            a.add_patch(Polygon(VEH.corners(sc.goal), fill=False, ec=INK, lw=1.0, ls="--"))
        a.set_aspect("equal"); a.set_title(sc.name); a.grid(False)
        xs0, ys0 = sc.start[0], sc.start[1]
        a.set_xlim(max(0, (sc.goal[0] + xs0) / 2 - 17), (sc.goal[0] + xs0) / 2 + 17); a.set_ylim(sc.grid.y0, sc.grid.y0 + sc.grid.ny * sc.grid.res)
        if m == 0:
            a.legend(loc="lower left", fontsize=7)
    savefig(fig, "fig_parking")

    # figure: lattice vs linear
    w = World(seed=7); fr = Frenet(w.route)
    def cd(s, d): return [[*fr.to_world(np.array([s + o]), np.array([d]))[0][0], 1.0] for o in (-1.5, 0.0, 1.5)]
    def dc(s, d, r): return [[*fr.to_world(np.array([s]), np.array([d]))[0][0], r]]
    obs = np.array(cd(80, -1.75) + dc(120, -1.0, 0.3) + cd(150, -1.75) + dc(190, -2.6, 0.4) + cd(235, -1.75) + cd(300, -1.75) + cd(318, 1.75))
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.0), gridspec_kw=dict(width_ratios=[1.3, 1]))
    for prof, col, ls, lab in (("quintic", BLUE, "-", "quintic lattice"), ("linear", ORANGE, "--", "linear ramps")):
        pl = StateLattice(w.route, obs, profile=prof); r = run_receding_horizon(w.route, pl, 30.0, 360.0)
        ax[0].plot(r["s"], r["d"], color=col, ls=ls, label=lab)
        ax[1].plot(r["arclen"], r["kappa"], color=col, ls=ls, lw=1.0, label=lab)
    sob = []
    for o in obs:
        i = np.argmin(np.hypot(*(w.route.xy - o[:2]).T)); sob.append((w.route.s[i], o[2]))
    ax[0].set_xlabel("route position s [m]"); ax[0].set_ylabel("lateral offset d [m]"); ax[0].set_title("Executed lateral profile")
    ax[0].axhline(-1.75, color=GRID, lw=0.8); ax[0].legend(fontsize=7.5)
    ax[1].axhline(VEH.kappa_max, color=INK2, lw=0.8, ls=":"); ax[1].axhline(-VEH.kappa_max, color=INK2, lw=0.8, ls=":")
    ax[1].text(5, VEH.kappa_max + 0.02, r"$\kappa_{max}$", fontsize=7.5, color=INK2)
    ax[1].set_xlabel("path length [m]"); ax[1].set_ylabel(r"curvature [m$^{-1}$]"); ax[1].set_title("Path curvature"); ax[1].legend(fontsize=7.5)
    savefig(fig, "fig_lattice")
    print("exp5 done")


if __name__ == "__main__":
    main()
