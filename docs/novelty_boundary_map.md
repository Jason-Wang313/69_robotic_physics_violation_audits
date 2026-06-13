# Novelty Boundary Map

## Crowded Territory
- Bigger data/model scaling.
- New benchmark only.
- Generic active learning or uncertainty.
- Combining a planner with a learned policy without a new state/action object.

## Claimed Boundary
Robotic physics violation audits keeps action-critical alternatives explicit until a physical observation collapses them.

## What Would Falsify The Claim
If observed-only baselines match the adverse-mode coverage and closed-loop success of the proposed branch-aware mechanism, the paper should be revised or killed.

## v4 Falsification
The real MuJoCo rebuild falsifies the current main-track claim. On combined violation shift, `physics_violation_audit` reaches F1 1.000, but so do kinematic residuals, energy residuals, ensemble uncertainty, autoencoder reconstruction, and supervised classifier baselines. On nominal valid traces, the explicit audit false-flags 23.3% of rollouts.
