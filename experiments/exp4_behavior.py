"""Experiment 4 -- Behaviour tree vs finite-state machines on randomised traffic scenarios."""
from common import *  # noqa
import collections
from sdc.behavior.bt import draw_tree
from sdc.behavior.scenario import *  # noqa

N = 300
CTRL = [("BT", BTController), ("FSM-basic", lambda: FSMController("basic")), ("FSM-full", lambda: FSMController("full")),
        ("FSM-full, no RB re-check", lambda: FSMController("full", rb_recheck=False))]


def summarise(res):
    col = collections.Counter(r["collision"] for r in res if r["collision"])
    ok = [r for r in res if r["reached"] and not r["collision"]]
    emg = sum(sum(1 for k in range(1, len(r["tags"])) if r["tags"][k] == "EMERGENCY" and r["tags"][k - 1] != "EMERGENCY") for r in res)
    return dict(success=len(ok), n=len(res), collisions=dict(col), timeouts=sum(1 for r in res if not r["reached"] and not r["collision"]),
                time=float(np.mean([r["time"] for r in ok])) if ok else None, overtakes=sum(r["overtook"] for r in res),
                aborts=sum(any(t.startswith("ABORT") for t in r["tags"]) for r in res),
                hard_brakes=float(np.mean([r["hard_brakes"] for r in res])), emergency=emg,
                switches=float(np.mean([r["switches"] for r in res])))


def main():
    rng = np.random.default_rng(123)
    scs = [sample_scenario(rng) for _ in range(N)]
    out, runs = {}, {}
    for name, mk in CTRL:
        runs[name] = [run_scenario(sc, mk()) for sc in scs]
        out[name] = summarise(runs[name])
    surprise = [sc.surprise for sc in scs]
    for name in runs:
        out[name]["success_surprise"] = sum(1 for r, s in zip(runs[name], surprise) if s and r["reached"] and not r["collision"])
        out[name]["n_surprise"] = sum(surprise)
    # structure
    root = build_bt()
    nodes = list(root.walk())
    kinds = collections.Counter(n.kind for n, _ in nodes)
    full, basic = FSMController("full"), FSMController("basic")
    n_states = len(full.T)
    out["structure"] = dict(bt_nodes=len(nodes), bt_conditions=kinds["condition"], bt_actions=kinds["action"],
                            bt_composites=kinds["sequence"] + kinds["fallback"], fsm_states=n_states,
                            fsm_transitions_basic=basic.n_transitions, fsm_transitions_full=full.n_transitions,
                            bt_add_behaviour=3, fsm_add_behaviour=n_states + 1)
    save_json("exp4_behavior", out)
    draw_tree(root, str(FIG / "fig_bt_tree.png"), "Behaviour tree (priority order: top to bottom)")
    draw_tree(root, str(FIG / "fig_bt_tree.pdf"))

    # timeline of a scenario where the BT aborts an overtake
    pick = next((i for i, r in enumerate(runs["BT"]) if any(t.startswith("ABORT") for t in r["tags"]) and r["overtook"] is False
                 and not r["collision"]), None)
    if pick is None:
        pick = next(i for i, r in enumerate(runs["BT"]) if any(t.startswith("ABORT") for t in r["tags"]))
    sc, r = scs[pick], runs["BT"][pick]
    L = r["log"]; t = L[:, 0]
    fig, ax = plt.subplots(3, 1, figsize=(7.2, 5.4), sharex=True, gridspec_kw=dict(height_ratios=[2, 1, 1]))
    ax[0].plot(t, L[:, 1], color=BLUE, label="ego")
    ax[0].plot(t, np.minimum(sc.s_lead0 + sc.v_lead * t, sc.s_turn + 5), color=ORANGE, ls="--", label="lead vehicle")
    for o in sc.oncoming:
        s0, v, ts = o
        m = (t >= ts) & (s0 - v * (t - ts) > 0)
        if m.any() and abs(s0 - v * (t[m][0] - ts) - L[m, 1][0]) < 500:
            ax[0].plot(t[m], s0 - v * (t[m] - ts), color=MUTED, lw=0.8, ls=":")
    ax[0].plot([], [], color=MUTED, ls=":", label="oncoming vehicles")
    ax[0].set_ylabel("position s [m]"); ax[0].set_ylim(0, 540); ax[0].legend(ncol=3, fontsize=7.5, loc="upper left", frameon=True, facecolor=SURF, edgecolor="none", framealpha=0.95)
    ax[0].axhspan(RB_ZONE[0], RB_ZONE[1], color=SHADE, lw=0); ax[0].text(t[0] + 1, RB_CONF + 8, "roundabout", ha="left", fontsize=7, color=INK2)
    ax[0].axhspan(CW_ZONE[0], CW_ZONE[1], color=SHADE, lw=0); ax[0].text(t[0] + 1, CW_S + 8, "crosswalk", ha="left", fontsize=7, color=INK2)
    ax[1].plot(t, L[:, 2], color=BLUE); ax[1].set_ylabel("lateral d [m]"); ax[1].axhline(LANE_W, color=GRID, lw=0.8)
    ax[2].plot(t, L[:, 3], color=BLUE); ax[2].set_ylabel("speed [m/s]"); ax[2].set_xlabel("time [s]")
    tags = r["tags"]
    start, row = 0, 0
    for k in range(1, len(tags) + 1):
        if k == len(tags) or tags[k] != tags[start]:
            tg, t1 = tags[start], t[min(k, len(t) - 1)]
            if tg not in ("CRUISE", "FOLLOW") and (t1 - t[start]) > 0.8:
                ax[2].annotate(tg.replace("_", " "), xy=((t[start] + t1) / 2, L[start, 3]), xytext=((t[start] + t1) / 2, 8.6 + 1.9 * (row % 3)),
                               ha="center", fontsize=6.2, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))
                row += 1
            start = k
    ax[2].set_ylim(0, 15.5)
    ax[0].set_title("Behaviour-tree run with an aborted overtake (labels = active leaf)")
    savefig(fig, "fig_bt_timeline")

    fig, a = plt.subplots(figsize=(5.4, 2.8))
    names = [n for n, _ in CTRL]
    succ = [100 * out[n]["success"] / N for n in names]
    a.barh(range(4), succ, color=[BLUE, YELLOW, ORANGE, MUTED], height=0.55)
    for i, v in enumerate(succ):
        a.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=8)
    a.set_yticks(range(4)); a.set_yticklabels(names, fontsize=8); a.invert_yaxis(); a.set_xlim(0, 112)
    a.set_xlabel(f"scenarios completed without collision [%] (N={N})"); a.set_title("Controller outcomes")
    savefig(fig, "fig_bt_outcomes")
    print("exp4 done")


if __name__ == "__main__":
    main()
