"""Experiment 7 -- why does the quintic lattice fail in some random layouts?  Solvability (Frenet DP) and variants."""
from common import *  # noqa
from exp5_planning import VEH
from sdc.planning.lattice import Frenet, LatticeConfig, StateLattice, run_receding_horizon
from sdc.sim.world import World

N, S0, S1 = 20, 30.0, 360.0


def make_layout(seed, fr):                      # identical to exp5_planning.lattice_mc
    rng = np.random.default_rng(seed)
    obs = []
    for s in np.sort(rng.uniform(60, 330, rng.integers(5, 9))):
        d = rng.choice([-1.75, -1.75, -1.0, -2.6, 1.75])
        if rng.random() < 0.6:
            for o in (-1.5, 0.0, 1.5):
                obs.append([*fr.to_world(np.array([s + o]), np.array([d]))[0][0], 1.0])
        else:
            obs.append([*fr.to_world(np.array([s]), np.array([d]))[0][0], rng.uniform(0.3, 0.5)])
    return np.array(obs)


def gap_map(pl, fr, ss, dd):
    """Clearance of the 3-disc ego body at road heading for every (s,d) cell -> (len(ss), len(dd))."""
    S, D = np.meshgrid(ss, dd, indexing="ij")
    pos, th, _ = fr.to_world(S.ravel(), D.ravel())
    offs = np.array(pl.cfg.ego_discs)
    cx = pos[:, 0:1] + np.cos(th)[:, None] * offs
    cy = pos[:, 1:2] + np.sin(th)[:, None] * offs
    g = np.hypot(cx[:, :, None] - pl.obs[None, None, :, 0], cy[:, :, None] - pl.obs[None, None, :, 1]) \
        - pl.cfg.ego_radius - pl.obs[None, None, :, 2]
    return g.min(axis=(1, 2)).reshape(S.shape)


def point_gap(pl, pos, th):
    offs = np.array(pl.cfg.ego_discs)
    cx, cy = pos[:, 0:1] + np.cos(th)[:, None] * offs, pos[:, 1:2] + np.sin(th)[:, None] * offs
    g = np.hypot(cx[:, :, None] - pl.obs[None, None, :, 0], cy[:, :, None] - pl.obs[None, None, :, 1]) \
        - pl.cfg.ego_radius - pl.obs[None, None, :, 2]
    return g.min(axis=(1, 2))


def reachable(valid, d0_idx, max_step=2):
    """Forward reachability over s with |d-index change| <= max_step per column (lateral slope <= 0.4)."""
    r = np.zeros(valid.shape[1], bool); r[d0_idx] = valid[0, d0_idx]
    for i in range(1, valid.shape[0]):
        nr = np.zeros_like(r)
        for k in range(-max_step, max_step + 1):
            nr |= np.roll(r, k)
        r = nr & valid[i]
        if not r.any():
            return False, i
    return True, valid.shape[0]


def evaluate(w, obs, cfg=None, replan=4.0):
    pl = StateLattice(w.route, obs, cfg or LatticeConfig())
    r = run_receding_horizon(w.route, pl, S0, S1, replan_every=replan)
    cl = pl.clearance(r["pos"], r["theta"])
    return dict(ok=bool(cl > 0 and np.abs(r["kappa"]).max() <= 1.05 * VEH.kappa_max), clearance=float(cl),
                fails=int(r["fails"]), max_kappa=float(np.abs(r["kappa"]).max())), r, pl


def one(seed):
    w = World(seed=7); fr = Frenet(w.route)
    obs = make_layout(seed, fr)
    nom, r, pl = evaluate(w, obs)
    out = dict(seed=seed, n_obs=int(len(obs)), nominal=nom)
    ss, dd = np.arange(S0, S1 + 0.5, 0.5), np.arange(-3.1, 3.1001, 0.1)
    gm = gap_map(pl, fr, ss, dd)
    d0 = int(np.argmin(abs(dd + 1.75)))
    out["solvable_collision_free"] = reachable(gm > 0.0, d0)[0]
    out["solvable_margin"] = reachable(gm >= pl.cfg.margin, d0)[0]
    if not nom["ok"]:
        dense = LatticeConfig(d_offsets=tuple(np.linspace(-3.0, 3.0, 25)), horizons=(12.0, 18.0, 26.0, 36.0, 50.0))
        out["dense"] = evaluate(w, obs, dense)[0]
        out["frequent"] = evaluate(w, obs, None, replan=2.0)[0]
        out["dense_frequent"] = evaluate(w, obs, dense, replan=2.0)[0]
        out["keep"] = dict(ss=ss, dd=dd, gm=gm, path_s=r["s"], path_d=r["d"], obs=obs, fail_s=float(r["s"][int(np.argmax(point_gap(pl, r["pos"], r["theta"]) <= 0))]))
    return out


def main():
    res = pool_map(one, list(range(N)))
    bad = [r for r in res if not r["nominal"]["ok"]]
    summary = dict(n=N, nominal_ok=N - len(bad), failing=len(bad),
                   failing_solvable_collision_free=sum(r["solvable_collision_free"] for r in bad),
                   failing_solvable_margin=sum(r["solvable_margin"] for r in bad),
                   dense_ok=sum(r["dense"]["ok"] for r in bad), frequent_ok=sum(r["frequent"]["ok"] for r in bad),
                   dense_frequent_ok=sum(r["dense_frequent"]["ok"] for r in bad),
                   ok_layouts_unsolvable=sum(1 for r in res if r["nominal"]["ok"] and not r["solvable_margin"]))
    rows = [{k: v for k, v in r.items() if k != "keep"} for r in res]
    save_json("exp7_lattice", dict(summary=summary, rows=rows))
    fig, ax = plt.subplots(1, max(len(bad), 1), figsize=(2.9 * max(len(bad), 1), 2.6), squeeze=False)
    fr = Frenet(World(seed=7).route)
    for a, r in zip(ax[0], bad):
        k = r["keep"]
        a.imshow((k["gm"] > 0.0).T, origin="lower", aspect="auto", cmap="Greys_r", vmin=-0.4, vmax=1.4,
                 extent=[k["ss"][0], k["ss"][-1], k["dd"][0] - 0.05, k["dd"][-1] + 0.05])
        a.plot(k["path_s"], k["path_d"], color=BLUE, lw=1.2)
        a.axvline(k["fail_s"], color=ORANGE, lw=0.8, ls="--")
        a.set_title(f"layout {r['seed']}", fontsize=8.5)
        a.set_xlabel("s [m]"); a.grid(False)
        tag = "solvable" if r["solvable_margin"] else ("solvable only w/o margin" if r["solvable_collision_free"] else "no path")
        a.text(0.02, 0.04, tag, transform=a.transAxes, fontsize=7, color=ORANGE if not r["solvable_margin"] else AQUA,
               bbox=dict(fc=SURF, ec="none", pad=1))
    ax[0][0].set_ylabel("lateral offset d [m]")
    savefig(fig, "fig_lattice_failures")
    print("exp7 done", summary)


if __name__ == "__main__":
    main()
