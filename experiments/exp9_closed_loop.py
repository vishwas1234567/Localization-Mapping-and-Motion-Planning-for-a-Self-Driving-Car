"""Experiment 9 -- closed loop: localisation -> behaviour tree -> state lattice -> pure pursuit -> kinematic bicycle.

Three pose sources feed the same stack: ground truth, NDT against the HD map (with simulated VO prior) and VO dead reckoning.
Perception of the 7 obstacles is perfect (known world positions) but the obstacles are *absent from the HD map*, so
the NDT localizer sees them as unmapped objects.
"""
from common import *  # noqa
from exp3_hdmap import hdmap, _W
from exp5_planning import VEH
from sdc.behavior.bt import Action, Condition, Fallback, Sequence, Status
from sdc.geometry import between, compose, wrap
from sdc.localization.particle_filter import NdtLocalizer
from sdc.planning.lattice import Frenet, StateLattice
from sdc.sim.drive import lidar_scan

S_START, S_END, DT, L = 30.0, 360.0, 0.05, VEH.wheelbase
MODES = ["gt", "ndt", "dr"]
LABEL = {"gt": "ground truth", "ndt": "NDT on HD map", "dr": "VO dead reckoning"}


def layout(fr):
    def cd(s, d): return [[*fr.to_world(np.array([s + o]), np.array([d]))[0][0], 1.0] for o in (-1.5, 0.0, 1.5)]
    def dc(s, d, r): return [[*fr.to_world(np.array([s]), np.array([d]))[0][0], r]]
    obs = np.array(cd(80, -1.75) + dc(120, -1.0, 0.3) + cd(150, -1.75) + dc(190, -2.6, 0.4) + cd(235, -1.75)
                   + cd(300, -1.75) + cd(318, 1.75))
    s_obs = np.array([80, 120, 150, 190, 235, 300, 318], float)
    return obs, s_obs


def octagons(obs):
    a = np.linspace(0, 2 * np.pi, 9)
    segs = []
    for x, y, r in obs:
        p = np.stack([x + r * np.cos(a), y + r * np.sin(a)], 1)
        segs.append(np.hstack([p[:-1], p[1:]]))
    return np.concatenate(segs)


def to_frenet(route, xy, th, s_hint):
    idx = np.where(np.abs(route.s - s_hint) < 25)[0]
    k = idx[np.argmin(np.hypot(*(route.xy[idx] - xy).T))]
    t = route.theta[k]
    dxy = xy - route.xy[k]
    d = -np.sin(t) * dxy[0] + np.cos(t) * dxy[1]
    s = route.s[k] + np.cos(t) * dxy[0] + np.sin(t) * dxy[1]
    return float(s), float(d), float((1 - route.kappa[k] * d) * np.tan(wrap(th - t)))


def build_bt():
    def setv(v):
        def f(bb):
            bb["v_ref"] = v
            return Status.RUNNING
        return f
    return Fallback("drive", [
        Sequence("blocked", [Condition("no_feasible_plan", lambda bb: bb["no_plan"]), Action("stop", setv(0.0))]),
        Sequence("avoid", [Condition("obstacle_ahead", lambda bb: bb["obs_ahead"] < 45.0), Action("slow_nudge", setv(5.0))]),
        Action("cruise", setv(8.0))])


def vo_measure(prev, cur, rng, s, scale):
    rel = between(prev, cur)
    k = _W.vision_factor(s)
    d = np.hypot(rel[0], rel[1])
    st, sr = (0.01 + 0.01 * d) * k, (0.0015 + 0.002 * abs(rel[2])) * k
    return np.array([rel[0] * scale, rel[1] * scale, rel[2]]) + rng.normal(0, 1, 3) * np.array([st, st, sr])


def run(args):
    mode, seed = args
    rng = np.random.default_rng(500 + seed)
    fr = Frenet(_W.route)
    obs, s_obs = layout(fr)
    segs = octagons(obs)
    pl = StateLattice(_W.route, obs)
    bt, bb = build_bt(), {"no_plan": False, "obs_ahead": 99.0, "v_ref": 8.0}
    p0, th0, _ = fr.to_world(np.array([S_START]), np.array([-1.75]), np.array([0.0]))
    true = np.array([p0[0, 0], p0[0, 1], th0[0]])
    est, prev, v, delta, scale = true.copy(), true.copy(), 0.0, 0.0, 1.0 + rng.normal(0, 0.004)
    loc = NdtLocalizer(hdmap(), true) if mode == "ndt" else None
    plan_xy, s_est, n_ctl = None, S_START, 0
    tr, e_loc, cte, tags, collided, min_clr, no_plan_ct = [], [], [], [], False, np.inf, 0
    t = 0.0
    while t < 120.0:
        if n_ctl % 2 == 0:                                         # 10 Hz pose sample
            if mode == "gt":
                est = true.copy()
            else:
                z = vo_measure(prev, true, rng, s_est, scale)
                if mode == "ndt":
                    est = loc.step(lidar_scan(_W, true, rng, extra_segments=segs), z)[0].copy()
                else:
                    est = compose(est, z)
            prev = true.copy()
            e_loc.append(float(np.hypot(*(est[:2] - true[:2]))))
        if n_ctl % 10 == 0:                                        # 2 Hz decision + planning
            s_est, d_est, dp_est = to_frenet(_W.route, est[:2], est[2], s_est)
            ahead = s_obs[s_obs > s_est - 3.0] - s_est
            bb["obs_ahead"] = float(ahead.min()) if len(ahead) else 99.0
            bt.tick(bb)
            best, _ = pl.plan(s_est, d_est, dp_est, 0.0)
            bb["no_plan"] = best is None
            no_plan_ct += best is None
            if best is not None:
                x = np.arange(0.0, best["Sf"] + 1e-9, 0.5)
                dd, dpp, _, _ = best["fn"](x)
                plan_xy = fr.to_world(s_est + x, dd, dpp)[0]
            tags.append(bb.get("active_leaf", "?"))
        # pure pursuit on the estimated pose
        if plan_xy is not None:
            k0 = int(np.argmin(np.hypot(*(plan_xy - est[:2]).T)))
            Ld = float(np.clip(1.0 * v + 3.0, 4.0, 10.0))
            cum = np.r_[0, np.cumsum(np.hypot(*np.diff(plan_xy, axis=0).T))]
            tgt = plan_xy[min(np.searchsorted(cum, cum[k0] + Ld), len(plan_xy) - 1)]
            al = wrap(np.arctan2(tgt[1] - est[1], tgt[0] - est[0]) - est[2])
            d_cmd = float(np.clip(np.arctan2(2 * L * np.sin(al), Ld), -VEH.delta_max, VEH.delta_max))
            delta += float(np.clip(d_cmd - delta, -0.6 * DT, 0.6 * DT))
            cte.append(float(np.min(np.hypot(*(plan_xy - true[:2]).T))))
        a = float(np.clip((bb["v_ref"] - v) / 1.0, -4.0, 2.0))
        v = max(0.0, v + a * DT)
        true = true + DT * v * np.array([np.cos(true[2]), np.sin(true[2]), np.tan(delta) / L])
        true[2] = wrap(true[2])
        t += DT; n_ctl += 1
        clr = pl.clearance(true[None, :2], np.array([true[2]])) + (1.15 - VEH.width / 2)   # physical body radius = half width
        min_clr = min(min_clr, float(clr))
        s_t, d_t, _ = to_frenet(_W.route, true[:2], true[2], s_est)
        tr.append((s_t, d_t))
        if clr < 0:
            collided = True; break
        if s_t >= S_END or abs(d_t) > 6.0:
            break
    tr = np.array(tr)
    reached = bool(tr[-1, 0] >= S_END)
    out = dict(mode=mode, seed=seed, success=bool(reached and not collided), collided=bool(collided), reached=reached,
               min_clearance=float(min_clr), cte_rms=float(np.sqrt(np.mean(np.square(cte)))) if cte else None,
               cte_max=float(np.max(cte)) if cte else None, loc_rms=float(np.sqrt(np.mean(np.square(e_loc)))),
               loc_max=float(np.max(e_loc)), no_plan=int(no_plan_ct), time=float(t),
               tags={k: tags.count(k) for k in set(tags)})
    return out, (tr, np.array(e_loc), t) if seed == 0 else None


def main():
    hdmap()
    res = pool_map(run, [(m, s) for m in MODES for s in range(10)])
    agg = {}
    for m in MODES:
        r = [x[0] for x in res if x[0]["mode"] == m]
        agg[m] = dict(n=len(r), success=sum(x["success"] for x in r), collisions=sum(x["collided"] for x in r),
                      min_clearance=ms([x["min_clearance"] for x in r]), loc_rms=ms([x["loc_rms"] for x in r]),
                      loc_max=ms([x["loc_max"] for x in r]),
                      cte_rms=ms([x["cte_rms"] for x in r if x["cte_rms"] is not None]),
                      no_plan=ms([x["no_plan"] for x in r]))
    save_json("exp9_closed_loop", dict(agg=agg, runs=[x[0] for x in res]))
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.0), gridspec_kw=dict(width_ratios=[1.5, 1]))
    a = ax[0]
    for sc_, d_, r_ in ((80, -1.75, 1.0), (120, -1.0, 0.3), (150, -1.75, 1.0), (190, -2.6, 0.4), (235, -1.75, 1.0), (300, -1.75, 1.0), (318, 1.75, 1.0)):
        a.add_patch(plt.Circle((sc_, d_), r_, color=SHADE, zorder=1))
        if r_ >= 1.0:
            for o in (-1.5, 1.5):
                a.add_patch(plt.Circle((sc_ + o, d_), r_, color=SHADE, zorder=1))
    for m, col, ls in (("gt", INK, "-"), ("ndt", BLUE, "--"), ("dr", ORANGE, "-.")):
        tr = [x[1][0] for x in res if x[0]["mode"] == m and x[0]["seed"] == 0][0]
        a.plot(tr[:, 0], tr[:, 1], color=col, ls=ls, lw=1.3, label=LABEL[m])
    a.set_xlim(S_START, S_END); a.set_ylim(-4.2, 3.2); a.set_xlabel("route position s [m]"); a.set_ylabel("lateral offset d [m]")
    a.set_title("True path, seed 0 (grey = obstacles)"); a.legend(fontsize=7, ncol=3, loc="lower center")
    b = ax[1]
    for m, col, ls in (("ndt", BLUE, "--"), ("dr", ORANGE, "-.")):
        e = [x[1][1] for x in res if x[0]["mode"] == m and x[0]["seed"] == 0][0]
        b.plot(np.arange(len(e)) * 0.1, np.maximum(e, 1e-3), color=col, ls=ls, label=LABEL[m])
    b.set_yscale("log"); b.set_xlabel("time [s]"); b.set_ylabel("localisation error [m]"); b.set_title("Localisation error in the loop")
    b.legend(fontsize=7.5)
    savefig(fig, "fig_closed_loop")
    print("exp9 done")


if __name__ == "__main__":
    main()
