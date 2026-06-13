# ICLR Main Gate

Paper: 69 robotic_physics_violation_audits

Existing v2 decision: KILL_ARCHIVE

Gate verdict: KILL_ARCHIVE

Evidence digest: 44d060481900fe1c

Fatal blockers:
- Explicit physics audit is matched by kinematic residual, energy residual, learned uncertainty, autoencoder reconstruction, and supervised classifier baselines on combined violation shift.
- Explicit audit false-flags 23.3% of nominal valid MuJoCo rollouts.
- Ablations show several explicit checks can be removed without hurting combined-shift F1.
- No real-robot validation or public benchmark replication.
- No manual exhaustive related-work synthesis.

Real high-fidelity evidence now exists, but it supports archive rather than submission.
