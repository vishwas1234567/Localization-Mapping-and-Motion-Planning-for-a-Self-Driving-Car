# Future work

Same list as Section "Future work" of `main.pdf`. P1-P8 were planned; the status column says what E6-E9 covered.
Hypotheses are expectations, not results. The ideas below the table have not been run.

## Planned experiments (this week)

| ID | Topic | Experiment | Hypothesis | Status |
|----|-------|-----------|-----------|--------|
| P1 | LOAM features | Edge/planar-feature scan-to-map odometry (`extract_loam_features`) vs NDT | Faster per scan, similar ATE in structured areas, worse in the tunnel | not run |
| P2 | Root causes | Inspect the 4 BT failures; ICP rotation bias vs turn rate; NDT step at tunnel exit (ablate tunnel ribs) | Unknown for BT; ICP bias grows with turn rate | not run |
| P3 | Sensitivity, statistics | LiDAR noise/dropout/range; map age; VO/IMU noise; 20 seeds for E2/E3 with bootstrap CIs | Graceful degradation, then a cliff | partly run (E8: E2/E3, 20 seeds, LiDAR noise/dropout) |
| P4 | Integration | VIO/NDT pose -> BT -> lattice/Hybrid A* -> tracking controller, closed loop | Localization error is small against lane margins | partly run (E9: NDT in loop; no VIO filter) |
| P5 | Hybrid A* | Slot-length sweep 5.5-7.5 m, goal refinement, numba/C++ port | Success threshold near the geometric minimum slot | partly run (E6: slot sweep, heuristic variants; no compiled port) |
| P6 | Global localization | More particles, heading prior, place recognition; 20 seeds | Place-recognition seeding fixes most failures | not run |
| P7 | Lattice | Solvability of the 4 failing layouts; speed dimension; moving obstacles | Failures are infeasible layouts, not planner faults | partly run (E7: solvability only) |
| P8 | Real data | NDT/SLAM on a public LiDAR odometry dataset (e.g. KITTI) | Same ranking ICP < NDT < NDT+loop | not run |

## Research ideas

**Estimation and mapping**
1. Real visual front end: KLT/ORB tracking with an MSCKF or sliding-window back end and IMU preintegration; validate on EuRoC.
2. Tightly coupled LiDAR-inertial odometry with IMU de-skewing (LIO-SAM / FAST-LIO2 style) and 3-D LOAM features.
3. Degeneracy-aware fusion: feed the Hessian ratio or eigenvectors into the EKF covariance; constrain only well-observed directions.
4. Online calibration: IMU-camera extrinsics and time offset; observability for planar motion; wheel odometry and non-holonomic constraints.
5. Robust loop closure: Scan Context place recognition, switchable constraints or max-mixtures, injected false-loop tests.
6. Consistency analysis: NEES/NIS tests for the EKF, covariance recovery from the pose graph.
7. Map maintenance: change detection, long-term updating, Lanelet2/OpenDRIVE export.
8. Global localization and recovery: multi-hypothesis NDT, GNSS-seeded priors, kidnapped-robot tests, integrity monitoring.

**Decision making**
9. Closed-loop BT in CARLA or ROS 2 with OpenSCENARIO scenarios; compare BT, hierarchical FSM and a learned policy.
10. Formal verification of BTs (model checking, runtime monitors) to make the "missing edge" failure a checkable property.
11. Behaviour planning under uncertainty: POMDP / multi-policy leaves; learned condition nodes with calibrated confidence.

**Motion planning and control**
12. Hybrid A* acceleration: compiled implementation, Reeds-Shepp expansions, learned heuristics, anytime search.
13. Trajectory refinement: optimization-based smoothing and collision avoidance on the Hybrid A* path.
14. Space-time lattice: speed dimension, moving obstacles, jerk limits; compare with RRT* and optimization-based planners.
15. Tracking control: pure pursuit / Stanley / MPC on a dynamic bicycle model with tyre slip.

**System and validation**
16. Closed-loop integration with latency budget and fault injection (sensor dropout, delayed GPS, map mismatch).
17. Public data: KITTI, nuScenes, Newer College or Oxford RobotCar; ATE/RPE protocol.
18. Statistical protocol: more seeds, bootstrap CIs, paired comparisons, pre-registered hypotheses.
19. Embedded deployment: real-time profiling on an automotive-class computer; C++/ROS 2 port of slow modules.
20. Safety argument: SOTIF-style hazard analysis for the failure modes found here (global-localization failure, missed roundabout re-check, unsolved parking).
