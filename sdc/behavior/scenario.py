"""Frenet-frame traffic scenario used to compare behaviour trees with finite-state machines.

Road (s = longitudinal, d = lateral, d=0 own lane, d=+3.5 opposite lane):

    s=0 ...... slow lead vehicle ........ [no-overtake zone] roundabout ...... crosswalk ..... s=520

* A slow lead vehicle (turns off at a random junction) invites an overtake.
* Oncoming vehicles are only perceived within ``SENSE_RANGE`` -> an unseen vehicle can
  appear *during* the manoeuvre and force an abort.
* A roundabout with a random stream of circulating vehicles (ego must yield).
* A pedestrian who starts crossing when the ego gets close (ego must yield).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bt import Action, Condition, Fallback, Inverter, Node, Sequence, Status

DT = 0.1
ROUTE_END = 520.0
CAR_L, CAR_W = 4.5, 1.9
LANE_W = 3.5
SENSE_RANGE = 90.0
RB_LINE, RB_CONF = 255.0, 272.0        # yield line, conflict point (s)
RB_ZONE = (RB_CONF - 4.5, RB_CONF + 4.5)
CW_S = 430.0                           # crosswalk
CW_LINE, CW_ZONE = CW_S - 7.0, (CW_S - 2.5 - 2.25, CW_S + 2.5 + 2.25)
NO_OVERTAKE = (RB_LINE - 70.0, RB_CONF + 20.0)
V_CRUISE, V_OVERTAKE, V_RB = 12.0, 15.0, 6.0
A_COMF, B_COMF, B_MAX = 1.8, 2.5, 8.0


# --------------------------------------------------------------------------
@dataclass
class Scenario:
    v_lead: float
    s_lead0: float
    s_turn: float
    oncoming: list            # [(s0, v, t_spawn)]  vehicles driving in -s direction in the opposite lane
    rb_times: list            # absolute times at which a circulating vehicle is at the conflict point
    ped_trigger: float        # ego distance to the crosswalk at which the pedestrian starts to cross
    ped_dur: float
    v0: float = 8.0
    surprise: bool = False


def sample_scenario(rng: np.random.Generator, surprise_prob: float = 0.5) -> Scenario:
    oncoming = []
    t = rng.uniform(0, 8)
    for _ in range(4):                                   # stream of oncoming vehicles
        oncoming.append((520.0 + rng.uniform(0, 30), rng.uniform(9, 13), t))
        t += rng.uniform(5, 22)
    surprise = rng.random() < surprise_prob
    if surprise:                                         # an unseen vehicle that arrives mid-overtake
        oncoming.append((rng.uniform(205, 235) + 0.0, rng.uniform(10, 14), 0.0))
    rb, t = [], rng.uniform(15, 25)
    for _ in range(14):
        rb.append(t)
        t += rng.exponential(5.0) + 2.2
    return Scenario(v_lead=rng.uniform(3.0, 6.0), s_lead0=rng.uniform(45, 70), s_turn=rng.uniform(170, 330),
                    oncoming=oncoming, rb_times=rb, ped_trigger=rng.uniform(28, 55), ped_dur=rng.uniform(5, 9),
                    surprise=surprise)


# --------------------------------------------------------------------------
class Sim:
    """Ground truth + perception."""

    def __init__(self, sc: Scenario):
        self.sc = sc
        self.t = 0.0
        self.s, self.v, self.a = 0.0, sc.v0, 0.0
        self.d = 0.0
        self.lc = None                                    # (t0, d0, d1, dur)
        self.s_lead = sc.s_lead0
        self.lead_alive = True
        self.onc = [dict(s=s0, v=v, t_spawn=ts) for (s0, v, ts) in sc.oncoming]
        self.ped_t0 = None
        self.collision = None
        self.log = []

    # ---- perception --------------------------------------------------
    def perceive(self) -> dict:
        sc = self.sc
        lead = None
        if self.lead_alive and self.s_lead > self.s - CAR_L and self.s_lead - self.s < SENSE_RANGE:
            lead = (self.s_lead, sc.v_lead)
        onc = [(o["s"], o["v"]) for o in self.onc if self.t >= o["t_spawn"] and 0 < o["s"] - self.s < SENSE_RANGE]
        rb = [t_ for t_ in sc.rb_times if 0.0 <= t_ - self.t < 9.0 or (self.t - t_ < 1.2 and self.t >= t_)]
        ped = "none"
        if self.ped_t0 is not None:
            ped = "waiting" if self.t < self.ped_t0 else ("crossing" if self.t < self.ped_t0 + sc.ped_dur else "none")
        return dict(t=self.t, s=self.s, v=self.v, d=self.d, lead=lead, oncoming=onc, rb=rb, ped=ped,
                    ped_clear_t=(None if self.ped_t0 is None else self.ped_t0 + sc.ped_dur))

    # ---- dynamics ----------------------------------------------------
    def step(self, cmd: dict):
        sc = self.sc
        # lateral
        tgt = cmd.get("d_target", 0.0)
        if self.lc is None or abs(self.lc[2] - tgt) > 1e-6:
            self.lc = (self.t, self.d, tgt, 3.0)
        t0, d0, d1, dur = self.lc
        u = min(max((self.t - t0) / dur, 0.0), 1.0)
        self.d = d0 + (d1 - d0) * (u * u * u * (u * (6 * u - 15) + 10))
        # longitudinal: IDM towards v_des with an optional virtual obstacle (stop line / lead)
        v_des = cmd.get("v_des", V_CRUISE)
        a = A_COMF * (1.0 - (self.v / max(v_des, 0.1)) ** 4)
        for gap, vr in cmd.get("obstacles", []):          # (distance to obstacle front, obstacle speed)
            if gap is None:
                continue
            gap = max(gap, 0.05)
            s_star = 2.0 + max(0.0, self.v * 1.2 + self.v * (self.v - vr) / (2 * np.sqrt(A_COMF * B_COMF)))
            a = min(a, A_COMF * (1.0 - (self.v / max(v_des, 0.1)) ** 4 - (s_star / gap) ** 2))
        if cmd.get("emergency"):
            a = -B_MAX
        a = float(np.clip(a, -B_MAX, 2.5))
        a_prev = self.a
        self.a = a
        self.v = max(0.0, self.v + a * DT)
        self.s += self.v * DT
        # world
        self.t += DT
        if self.lead_alive:
            self.s_lead += sc.v_lead * DT
            if self.s_lead > sc.s_turn:
                self.lead_alive = False
        for o in self.onc:
            if self.t >= o["t_spawn"]:
                o["s"] -= o["v"] * DT
        if self.ped_t0 is None and CW_S - self.s < sc.ped_trigger:
            self.ped_t0 = self.t + 1.5                    # waits 1.5 s at the kerb, then crosses
        self._check_collisions()
        self.log.append((self.t, self.s, self.d, self.v, a, a_prev, cmd.get("tag", "")))

    def _check_collisions(self):
        if self.collision:
            return
        sc = self.sc
        if self.lead_alive and abs(self.s - self.s_lead) < CAR_L and abs(self.d) < CAR_W:
            self.collision = "lead"
        for o in self.onc:
            if self.t >= o["t_spawn"] and abs(self.s - o["s"]) < CAR_L and abs(self.d - LANE_W) < CAR_W:
                self.collision = "oncoming"
        for tr in sc.rb_times:
            if abs(self.t - tr) < 1.1 and RB_ZONE[0] <= self.s <= RB_ZONE[1]:
                self.collision = "roundabout"
        if self.ped_t0 is not None and self.ped_t0 <= self.t < self.ped_t0 + sc.ped_dur \
                and CW_ZONE[0] <= self.s <= CW_ZONE[1] and self.d < 2.5:
            self.collision = "pedestrian"

    @property
    def done(self):
        return self.s >= ROUTE_END or self.collision is not None or self.t > 140.0


# --------------------------------------------------------------------------
# Shared decision primitives (used by both the BT and the FSM so the comparison is about structure)
# --------------------------------------------------------------------------
def brake_dist(v, b=B_COMF):
    return v * v / (2 * b)


def ttc_ahead(p: dict) -> float:
    """Smallest time-to-collision with a perceived object in the lane currently occupied."""
    best = np.inf
    if p["lead"] is not None and abs(p["d"]) < 1.9:
        gap = p["lead"][0] - p["s"] - CAR_L
        cl = p["v"] - p["lead"][1]
        if cl > 0.1:
            best = min(best, max(gap, 0.0) / cl)
    if abs(p["d"] - LANE_W) < 1.9:
        for so, vo in p["oncoming"]:
            gap = so - p["s"] - CAR_L
            best = min(best, max(gap, 0.0) / (p["v"] + vo))
    return best


def overtake_time_needed(p: dict) -> float:
    """Time until the ego is back in its lane, ahead of the lead, if it starts/continues now."""
    sl, vl = p["lead"]
    rel = (sl - p["s"]) + CAR_L + 10.0
    t_pass = max(rel, 0.0) / max(V_OVERTAKE - vl, 1.0)
    t_out = 0.0 if p["d"] > 3.0 else 3.0 * (1 - p["d"] / LANE_W)
    return t_out + t_pass + 3.0


def oncoming_clear(p: dict, margin=14.0) -> bool:
    """No perceived oncoming vehicle can reach the overtaking corridor before we are back in lane."""
    if p["lead"] is None:
        return True
    tn = overtake_time_needed(p)
    for so, vo in p["oncoming"]:
        if so - p["s"] - (V_OVERTAKE + vo) * tn < margin:
            return False
    return True


def in_no_overtake_zone(p: dict, lookahead_t: float = 0.0) -> bool:
    end = p["s"] + overtake_time_needed(p) * V_OVERTAKE if p["lead"] else p["s"]
    return (end > NO_OVERTAKE[0] and p["s"] < NO_OVERTAKE[1]) or (p["s"] > CW_S - 90 and p["s"] < CW_S + 10)


def rb_gap_ok(p: dict) -> bool:
    """Can the ego traverse the conflict zone without meeting a circulating vehicle?"""
    dist = RB_ZONE[0] - 2.25 - p["s"]
    if p["s"] > RB_ZONE[1] + 2.25:
        return True
    v0 = max(p["v"], 0.3)
    # accelerate at 1.5 m/s^2 up to V_RB
    def t_for(x):
        if x <= 0:
            return 0.0
        x_acc = (V_RB ** 2 - v0 ** 2) / (2 * 1.5) if V_RB > v0 else 0.0
        if x <= x_acc:
            return (-v0 + np.sqrt(v0 ** 2 + 2 * 1.5 * x)) / 1.5
        return max((V_RB - v0) / 1.5, 0.0) + (x - x_acc) / V_RB
    t_in = t_for(dist)
    t_out = t_for(dist + (RB_ZONE[1] + 2.25) - (RB_ZONE[0] - 2.25))
    lo, hi = p["t"] + t_in - 1.2, p["t"] + t_out + 1.2
    return not any(lo - 1.1 < tr < hi + 1.1 for tr in p["rb"])


def stop_obstacle(line_s: float, p: dict):
    return (line_s - p["s"], 0.0)


def lead_obstacle(p: dict):
    if p["lead"] is None or abs(p["d"]) > 1.9:
        return (None, 0.0)
    return (p["lead"][0] - p["s"] - CAR_L, p["lead"][1])


def speed_limit(p: dict) -> float:
    v = V_CRUISE
    if RB_LINE - 40 < p["s"] < RB_ZONE[1] + 5:
        v = min(v, V_RB + 2.0)
    return v


# --------------------------------------------------------------------------
# Behaviour tree
# --------------------------------------------------------------------------
def build_bt() -> Node:
    def cmd(bb, **kw):
        c = bb["cmd"]
        c.update(kw)

    # ---- conditions --------------------------------------------------
    c_imminent = Condition("imminent_collision", lambda bb: ttc_ahead(bb["p"]) < 1.6)
    c_ped = Condition("pedestrian_at_crosswalk",
                      lambda bb: bb["p"]["ped"] != "none" and bb["p"]["s"] < CW_ZONE[1] and CW_S - bb["p"]["s"] < 70)
    c_rb = Condition("approaching_roundabout", lambda bb: RB_LINE - 60 < bb["p"]["s"] < RB_ZONE[1] + 2.25)
    c_gap = Condition("roundabout_gap_ok",
                      lambda bb: rb_gap_ok(bb["p"]) or (RB_LINE - bb["p"]["s"] < brake_dist(bb["p"]["v"], 4.0) and bb["p"]["s"] > RB_LINE - 30))
    c_unsafe = Condition("overtake_unsafe", lambda bb: bb["p"]["d"] > 0.4 and bb["ot_phase"] is not None
                         and not (bb["p"]["lead"] is None or oncoming_clear(bb["p"], 6.0)))
    c_need = Condition("slow_lead_ahead", lambda bb: bb["ot_phase"] is not None or (
        bb["p"]["lead"] is not None and bb["p"]["lead"][1] < V_CRUISE - 3.0
        and 0 < bb["p"]["lead"][0] - bb["p"]["s"] < 40.0))
    c_legal = Condition("overtake_legal", lambda bb: not in_no_overtake_zone(bb["p"]) or bb["ot_phase"] is not None)
    c_feasible = Condition("oncoming_clear", lambda bb: bb["ot_phase"] is not None or oncoming_clear(bb["p"]))
    c_follow = Condition("lead_close", lambda bb: lead_obstacle(bb["p"])[0] is not None and lead_obstacle(bb["p"])[0] < 60)

    # ---- actions -----------------------------------------------------
    def a_emergency(bb):
        cmd(bb, emergency=True, tag="EMERGENCY")
        return Status.RUNNING

    def a_stop_ped(bb):
        p = bb["p"]
        cmd(bb, obstacles=[stop_obstacle(CW_LINE, p), lead_obstacle(p)], v_des=speed_limit(p), d_target=0.0, tag="YIELD_PED")
        return Status.RUNNING

    def a_roundabout_hold(bb):
        p = bb["p"]
        cmd(bb, obstacles=[stop_obstacle(RB_LINE, p), lead_obstacle(p)], v_des=V_RB + 2, d_target=0.0, tag="YIELD_RB")
        return Status.RUNNING

    def a_roundabout_go(bb):
        p = bb["p"]
        cmd(bb, v_des=V_RB if p["s"] > RB_LINE - 15 else V_RB + 2, d_target=0.0,
            obstacles=[lead_obstacle(p)], tag="RB_TRAVERSE")
        return Status.RUNNING

    def a_abort(bb):
        p = bb["p"]
        if p["lead"] is None or p["s"] - p["lead"][0] > 6.0:          # already ahead: merge back in front
            cmd(bb, d_target=0.0, v_des=V_OVERTAKE, tag="ABORT_MERGE_AHEAD")
        else:                                                           # fall back behind the lead
            cmd(bb, v_des=max(p["lead"][1] - 2.5, 0.5), d_target=0.0 if p["lead"][0] - p["s"] > CAR_L + 6 else p["d"],
                obstacles=[], tag="ABORT_FALL_BACK")
        if p["d"] < 0.3:
            bb["ot_phase"] = None
        else:
            bb["ot_phase"] = "abort"
        return Status.RUNNING

    def a_overtake(bb):
        p = bb["p"]
        sl, vl = p["lead"] if p["lead"] else (p["s"] - 100, 0.0)
        if bb["ot_phase"] in (None, "abort"):
            bb["ot_phase"] = "out"
        ph = bb["ot_phase"]
        ahead = p["s"] - sl
        if ph == "out" and p["d"] > 3.0:
            bb["ot_phase"] = ph = "pass"
        if ph in ("out", "pass") and ahead > 10.0 + 0:
            bb["ot_phase"] = ph = "in"
        if ph == "in" and p["d"] < 0.25:
            bb["ot_phase"] = None
            return Status.SUCCESS
        if ph == "out":
            cmd(bb, d_target=LANE_W, v_des=V_OVERTAKE, obstacles=[(lead_obstacle(p)[0], lead_obstacle(p)[1])] if p["d"] < 1.2 else [], tag="OT_OUT")
        elif ph == "pass":
            cmd(bb, d_target=LANE_W, v_des=V_OVERTAKE, obstacles=[], tag="OT_PASS")
        else:
            cmd(bb, d_target=0.0, v_des=V_OVERTAKE, obstacles=[], tag="OT_IN")
        return Status.RUNNING

    def a_follow(bb):
        p = bb["p"]
        cmd(bb, obstacles=[lead_obstacle(p)], v_des=speed_limit(p), d_target=0.0, tag="FOLLOW")
        return Status.RUNNING

    def a_cruise(bb):
        p = bb["p"]
        cmd(bb, v_des=speed_limit(p), d_target=0.0, obstacles=[], tag="CRUISE")
        return Status.RUNNING

    root = Fallback("drive", [
        Sequence("emergency", [c_imminent, Action("emergency_brake", a_emergency)]),
        Sequence("yield_pedestrian", [c_ped, Action("stop_at_crosswalk", a_stop_ped)]),
        Sequence("roundabout", [c_rb, Fallback("enter_or_yield", [
            Sequence("enter", [c_gap, Action("traverse_roundabout", a_roundabout_go)]),
            Action("hold_at_yield_line", a_roundabout_hold)])]),
        Sequence("abort_overtake", [c_unsafe, Action("abort_and_return", a_abort)]),
        Sequence("overtake", [c_need, c_legal, c_feasible, Action("overtake_manoeuvre", a_overtake)]),
        Sequence("follow", [c_follow, Action("follow_lead", a_follow)]),
        Action("cruise", a_cruise),
    ])
    return root


class BTController:
    name = "BT"

    def __init__(self):
        self.root = build_bt()
        self.bb = {"ot_phase": None, "cmd": {}}
        self.switches = 0
        self._last = None

    def decide(self, p: dict) -> dict:
        self.bb["p"] = p
        self.bb["cmd"] = {}
        self.root.tick(self.bb)
        c = self.bb["cmd"]
        if c.get("tag") != self._last:
            self.switches += 1
            self._last = c.get("tag")
        return c


# --------------------------------------------------------------------------
# Finite-state machine baselines
# --------------------------------------------------------------------------
class FSMController:
    """Hand-written FSM.

    variant "basic": the transitions a designer writes for the nominal scenarios
                     (cruise <-> follow -> overtake out/pass/in, yield states from cruise/follow only)
                     plus a global emergency brake.
    variant "full" : adds the pre-emption transitions (abort overtake, yield from every state ...)
                     -- what is needed to match the reactivity of the BT.
    """

    def __init__(self, variant="basic", rb_recheck=True):
        self.variant, self.rb_recheck = variant, rb_recheck
        self.name = f"FSM-{variant}" + ("" if (variant == "basic" or rb_recheck) else "-norecheck")
        self.state = "CRUISE"
        self.switches = 0
        T = {
            "CRUISE": ["FOLLOW", "OT_OUT", "YIELD_RB", "YIELD_PED"],
            "FOLLOW": ["CRUISE", "OT_OUT", "YIELD_RB", "YIELD_PED"],
            "OT_OUT": ["OT_PASS", "OT_IN"], "OT_PASS": ["OT_IN"], "OT_IN": ["CRUISE"],
            "YIELD_RB": ["RB_TRAVERSE"], "RB_TRAVERSE": ["CRUISE"], "YIELD_PED": ["CRUISE"],
            "EMERGENCY": ["CRUISE"],
        }
        if variant == "full":
            T["OT_OUT"] += ["ABORT", "YIELD_RB", "YIELD_PED"]
            T["OT_PASS"] += ["ABORT", "YIELD_RB", "YIELD_PED"]
            T["OT_IN"] += ["YIELD_RB", "YIELD_PED"]
            T["ABORT"] = ["CRUISE", "YIELD_RB", "YIELD_PED"]
            T["RB_TRAVERSE"] += ["YIELD_PED"] + (["YIELD_RB"] if rb_recheck else [])
        for s in list(T):
            if s != "EMERGENCY":
                T[s] = T[s] + ["EMERGENCY"]
        self.T = T

    @property
    def n_transitions(self):
        return sum(len(v) for v in self.T.values())

    def _go(self, new):
        if new != self.state:
            assert new in self.T[self.state], f"illegal transition {self.state}->{new}"
            self.state = new

    def decide(self, p: dict) -> dict:
        st, full = self.state, self.variant == "full"
        before = st
        ph_ahead = None if p["lead"] is None else p["s"] - p["lead"][0]
        # global emergency
        if ttc_ahead(p) < 1.6 and st != "EMERGENCY":
            self._go("EMERGENCY")
        st = self.state
        ped = p["ped"] != "none" and p["s"] < CW_ZONE[1] and CW_S - p["s"] < 70
        rb = RB_LINE - 60 < p["s"] < RB_ZONE[1] + 2.25
        cmd = {}
        if st == "EMERGENCY":
            if ttc_ahead(p) > 2.5:
                self._go("CRUISE")
            cmd = dict(emergency=True)
        elif st in ("CRUISE", "FOLLOW"):
            if ped:
                self._go("YIELD_PED")
            elif rb:
                self._go("YIELD_RB")
            elif p["lead"] is not None and p["lead"][1] < V_CRUISE - 3 and 0 < p["lead"][0] - p["s"] < 40 \
                    and not in_no_overtake_zone(p) and oncoming_clear(p):
                self._go("OT_OUT")
            elif lead_obstacle(p)[0] is not None and lead_obstacle(p)[0] < 60:
                self._go("FOLLOW")
            else:
                self._go("CRUISE")
        elif st in ("OT_OUT", "OT_PASS", "OT_IN"):
            if full and ped:
                self._go("YIELD_PED")
            elif full and rb and st == "OT_IN":
                self._go("YIELD_RB")
            elif full and st in ("OT_OUT", "OT_PASS") and rb and p["d"] > 0.4:
                self._go("ABORT")
            elif full and st in ("OT_OUT", "OT_PASS") and p["lead"] is not None and not oncoming_clear(p, 6.0):
                self._go("ABORT")
            elif st == "OT_OUT" and p["d"] > 3.0:
                self._go("OT_PASS")
            elif st in ("OT_OUT", "OT_PASS") and (p["lead"] is None or ph_ahead > 10.0):
                self._go("OT_IN" if p["lead"] is not None else "OT_IN")
            elif st == "OT_IN" and p["d"] < 0.25:
                self._go("CRUISE")
        elif st == "ABORT":
            if p["d"] < 0.25:
                self._go("CRUISE")
        elif st == "YIELD_PED":
            if not ped:
                self._go("CRUISE")
        elif st == "YIELD_RB":
            if rb_gap_ok(p) or (RB_LINE - p["s"] < brake_dist(p["v"], 4.0) and p["s"] > RB_LINE - 30):
                self._go("RB_TRAVERSE")
            elif full and ped:
                self._go("YIELD_PED")
        elif st == "RB_TRAVERSE":
            committed = RB_LINE - p["s"] < brake_dist(p["v"], 4.0) and p["s"] > RB_LINE - 30
            if p["s"] > RB_ZONE[1] + 2.25:
                self._go("CRUISE")
            elif full and ped and p["s"] > RB_ZONE[1]:
                self._go("YIELD_PED")
            elif full and self.rb_recheck and not (rb_gap_ok(p) or committed):
                self._go("YIELD_RB")
        st = self.state
        if st != "EMERGENCY":
            cmd = self._command(st, p)
        cmd["tag"] = st
        if st != before:
            self.switches += 1
        return cmd

    def _command(self, st, p):
        if st == "CRUISE":
            return dict(v_des=speed_limit(p), d_target=0.0, obstacles=[])
        if st == "FOLLOW":
            return dict(v_des=speed_limit(p), d_target=0.0, obstacles=[lead_obstacle(p)])
        if st == "YIELD_PED":
            return dict(v_des=speed_limit(p), d_target=0.0, obstacles=[stop_obstacle(CW_LINE, p), lead_obstacle(p)])
        if st == "YIELD_RB":
            return dict(v_des=V_RB + 2, d_target=0.0, obstacles=[stop_obstacle(RB_LINE, p), lead_obstacle(p)])
        if st == "RB_TRAVERSE":
            return dict(v_des=V_RB, d_target=0.0, obstacles=[lead_obstacle(p)])
        if st == "OT_OUT":
            return dict(v_des=V_OVERTAKE, d_target=LANE_W, obstacles=[lead_obstacle(p)] if p["d"] < 1.2 else [])
        if st == "OT_PASS":
            return dict(v_des=V_OVERTAKE, d_target=LANE_W, obstacles=[])
        if st == "OT_IN":
            return dict(v_des=V_OVERTAKE, d_target=0.0, obstacles=[])
        if st == "ABORT":
            if p["lead"] is None or p["s"] - p["lead"][0] > 6.0:
                return dict(v_des=V_OVERTAKE, d_target=0.0, obstacles=[])
            return dict(v_des=max(p["lead"][1] - 2.5, 0.5), d_target=0.0 if p["lead"][0] - p["s"] > CAR_L + 6 else p["d"], obstacles=[])
        raise KeyError(st)


# --------------------------------------------------------------------------
def run_scenario(sc: Scenario, ctrl):
    sim = Sim(sc)
    while not sim.done:
        sim.step(ctrl.decide(sim.perceive()))
    log = np.array([r[:6] for r in sim.log])
    tags = [r[6] for r in sim.log]
    reached = sim.s >= ROUTE_END
    return dict(sim=sim, log=log, tags=tags, collision=sim.collision, reached=reached, time=sim.t,
                switches=ctrl.switches,
                hard_brakes=int(np.sum((log[1:, 4] < -4.0) & (log[:-1, 4] >= -4.0))),
                max_jerk=float(np.max(np.abs(np.diff(log[:, 4])) / DT)) if len(log) > 2 else 0.0,
                overtook=any(t.startswith("OT_PASS") for t in tags))
