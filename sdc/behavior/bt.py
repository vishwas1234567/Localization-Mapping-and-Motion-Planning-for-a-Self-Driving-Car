"""A small, dependency-free behaviour-tree library (tick-based, reactive, with a blackboard).

Node semantics follow the usual BT conventions (Colledanchise & Ogren, 2018):
  * Sequence  ("->")  : ticks children left to right, returns the first non-SUCCESS status.
  * Fallback  ("?")   : ticks children left to right, returns the first non-FAILURE status.
  * With ``memory=True`` a Sequence resumes at the child that returned RUNNING instead of
    re-evaluating earlier children (used for committed multi-step manoeuvres).
  * When a higher-priority child takes over, lower-priority children that were RUNNING are
    *halted* (their state is reset) -- this is what gives BTs their reactivity / pre-emption.
"""
from __future__ import annotations

from enum import Enum
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch


class Status(Enum):
    IDLE = 0
    SUCCESS = 1
    FAILURE = 2
    RUNNING = 3


class Node:
    kind = "node"

    def __init__(self, name: str, children=None):
        self.name = name
        self.children = list(children or [])
        self.status = Status.IDLE
        self.ticks = 0

    def tick(self, bb: dict) -> Status:
        self.ticks += 1
        self.status = self._tick(bb)
        return self.status

    def _tick(self, bb) -> Status:
        raise NotImplementedError

    def halt(self, bb: dict):
        for c in self.children:
            if c.status == Status.RUNNING:
                c.halt(bb)
        self.on_halt(bb)
        self.status = Status.IDLE

    def on_halt(self, bb):
        pass

    def walk(self, depth=0):
        yield self, depth
        for c in self.children:
            yield from c.walk(depth + 1)


class Sequence(Node):
    kind = "sequence"

    def __init__(self, name, children, memory=False):
        super().__init__(name, children)
        self.memory = memory
        self._idx = 0

    def on_halt(self, bb):
        self._idx = 0

    def _tick(self, bb):
        start = self._idx if self.memory else 0
        for i in range(start, len(self.children)):
            s = self.children[i].tick(bb)
            if s == Status.RUNNING:
                self._idx = i
                self._halt_after(i, bb)
                return s
            if s == Status.FAILURE:
                self._idx = 0
                self._halt_after(i, bb)
                return s
        self._idx = 0
        return Status.SUCCESS

    def _halt_after(self, i, bb):
        for c in self.children[i + 1:]:
            if c.status == Status.RUNNING:
                c.halt(bb)


class Fallback(Node):
    kind = "fallback"

    def _tick(self, bb):
        for i, c in enumerate(self.children):
            s = c.tick(bb)
            if s != Status.FAILURE:
                for later in self.children[i + 1:]:
                    if later.status == Status.RUNNING:
                        later.halt(bb)          # pre-emption of lower-priority branches
                return s
        return Status.FAILURE


class Condition(Node):
    kind = "condition"

    def __init__(self, name, fn: Callable[[dict], bool]):
        super().__init__(name)
        self.fn = fn

    def _tick(self, bb):
        return Status.SUCCESS if self.fn(bb) else Status.FAILURE


class Action(Node):
    kind = "action"

    def __init__(self, name, fn: Callable[[dict], Status], on_halt: Callable[[dict], None] | None = None):
        super().__init__(name)
        self.fn = fn
        self._on_halt = on_halt

    def _tick(self, bb):
        s = self.fn(bb)
        if s in (Status.RUNNING, Status.SUCCESS):
            bb["active_leaf"] = self.name
        return s

    def on_halt(self, bb):
        if self._on_halt:
            self._on_halt(bb)


class Inverter(Node):
    kind = "decorator"

    def _tick(self, bb):
        s = self.children[0].tick(bb)
        return {Status.SUCCESS: Status.FAILURE, Status.FAILURE: Status.SUCCESS}.get(s, s)


# --------------------------------------------------------------------------
# Visualisation
# --------------------------------------------------------------------------
def draw_tree(root: Node, path: str, title: str | None = None, active: set | None = None):
    """Render the tree left-to-right with matplotlib (rounded box = fallback/sequence/action,
    ellipse = condition). Root on the left, one row per leaf, so labels stay readable at page width.
    Data units equal inches."""
    import textwrap
    XSTEP, W, H, YSTEP = 1.42, 1.16, 0.36, 0.44
    pos, nxt = {}, [0]

    def layout(n, depth):
        if not n.children:
            pos[id(n)] = (depth * XSTEP, -nxt[0] * YSTEP)
            nxt[0] += 1
        else:
            for c in n.children:
                layout(c, depth + 1)
            ys = [pos[id(c)][1] for c in n.children]
            pos[id(n)] = (depth * XSTEP, sum(ys) / len(ys))

    layout(root, 0)
    nodes = list(root.walk())
    maxd = max(d for _, d in nodes)
    fig, ax = plt.subplots(figsize=(maxd * XSTEP + W + 0.3, YSTEP * nxt[0] + 0.3))
    colors = {"fallback": "#d9e8fb", "sequence": "#fde7c8", "condition": "#e3f4e1", "action": "#f7d9d9",
              "decorator": "#ece0f5"}
    gap = (XSTEP - W) / 2
    for n, d in nodes:
        for c in n.children:
            (x0, y0), (x1, y1) = pos[id(n)], pos[id(c)]
            xm = x0 + W / 2 + gap
            ax.plot([x0 + W / 2, xm, xm, x1 - W / 2], [y0, y0, y1, y1], color="#666", lw=0.8, zorder=1)
    for n, d in nodes:
        x, y = pos[id(n)]
        col = colors[n.kind]
        label = {"fallback": "?", "sequence": "->"}.get(n.kind, "")
        if n.kind == "condition":
            ax.add_patch(Ellipse((x, y), W, H, fc=col, ec="#2f7d32", lw=1.0, zorder=2))
        else:
            ec = "#b03a3a" if n.kind == "action" else "#555"
            ax.add_patch(FancyBboxPatch((x - W / 2, y - H / 2), W, H, boxstyle="round,pad=0.01",
                                        fc=col, ec=ec, lw=1.0, zorder=2))
        txt = ("\u2192 " if n.kind == "sequence" else "? " if n.kind == "fallback" else "") + n.name.replace("_", " ")
        ax.text(x, y, textwrap.fill(txt, 17), ha="center", va="center", fontsize=6.8, linespacing=0.95, zorder=3)
    ax.set_xlim(-W / 2 - 0.05, maxd * XSTEP + W / 2 + 0.05)
    ax.set_ylim(-(nxt[0] - 1) * YSTEP - H, H)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout(pad=0.1)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path
