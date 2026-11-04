"""Tight-space planning scenarios (metres).  Poses are the *rear axle* centre (x, y, theta)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hybrid_astar import OccGrid, Vehicle


@dataclass
class PlanScenario:
    name: str
    grid: OccGrid
    start: np.ndarray
    goal: np.ndarray
    cars: list            # obstacle polygons (for drawing)
    start_jitter: tuple = (1.0, 0.4, 0.15)


def _car_poly(cx, cy, th, L=4.5, W=1.9):
    c, s = np.cos(th), np.sin(th)
    p = np.array([[-L / 2, -W / 2], [L / 2, -W / 2], [L / 2, W / 2], [-L / 2, W / 2]])
    return np.stack([cx + c * p[:, 0] - s * p[:, 1], cy + s * p[:, 0] + c * p[:, 1]], axis=1)


def perpendicular_parking(veh: Vehicle = Vehicle()) -> PlanScenario:
    g = OccGrid(0, 0, 40, 22)
    cars = []
    slot_w, y_lo, y_hi = 2.7, 12.5, 17.5
    target = 6
    for k in range(10):
        xc = 4.0 + slot_w * (k + 0.5)
        if k == target:
            continue
        poly = _car_poly(xc, 15.0, np.pi / 2)
        g.add_polygon(poly); cars.append(poly)
    g.add_rect(0, 17.6, 40, 22)                 # building behind the slots
    g.add_rect(0, 0, 40, 5.6)                   # opposite kerb / wall: aisle y in [5.6, 12.5]
    xc = 4.0 + slot_w * (target + 0.5)
    goal = np.array([xc, 15.0 - (veh.length / 2 - veh.rear_overhang), np.pi / 2])
    return PlanScenario("Perpendicular parking", g, np.array([3.0, 8.4, 0.0]), goal, cars)


def parallel_parking(veh: Vehicle = Vehicle(), slot: float = 6.2) -> PlanScenario:
    """Parallel slot of ``slot`` = 6.2 m by default (car 4.5 m): parked cars and the goal sit 0.2 m off the kerb."""
    g = OccGrid(0, 0, 40, 12)
    cars = []
    yc = 0.3 + 0.2 + 0.95                        # kerb top + 0.2 m + half width
    for xc in (11.0, 11.0 + 4.5 + slot):
        poly = _car_poly(xc, yc, 0.0)
        g.add_polygon(poly); cars.append(poly)
    g.add_rect(0, 0, 40, 0.3)                    # kerb
    g.add_rect(0, 8.0, 40, 12)                   # opposite side of the street (lane y in [0.3, 8.0])
    xc = 11.0 + 2.25 + slot / 2                       # slot centre
    goal = np.array([xc - (veh.length / 2 - veh.rear_overhang), yc, 0.0])
    return PlanScenario("Parallel parking", g, np.array([20.0, 5.0, 0.0]), goal, cars, (1.5, 0.3, 0.1))


def k_turn(veh: Vehicle = Vehicle()) -> PlanScenario:
    g = OccGrid(0, 0, 36, 12)
    g.add_rect(0, 0, 36, 2.0)
    g.add_rect(0, 8.6, 36, 12)                   # street width 6.6 m < 2 * rho + width: needs a 3-point turn
    g.add_rect(33.0, 0, 36, 12)                  # dead end
    return PlanScenario("K-turn in a narrow street", g, np.array([8.0, 4.0, 0.0]), np.array([8.0, 6.6, np.pi]), [],
                        (0.8, 0.2, 0.1))
