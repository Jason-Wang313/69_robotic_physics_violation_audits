# Claims

- Mechanism claim under test: explicit contact, support, energy, friction, actuator, and causality checks should detect physics violations that robot policies silently tolerate better than generic residual or learned anomaly baselines.
- Real-evidence result: the v4 MuJoCo benchmark falsifies the novelty claim. On combined violation shift, `physics_violation_audit` reaches F1 1.000, but so do kinematic residual, energy residual, ensemble uncertainty, autoencoder reconstruction, and supervised classifier baselines.
- False-positive result: on nominal valid MuJoCo traces, the explicit audit false-flags 23.3% of rollouts.
- Scope claim: results support archiving this specific audit mechanism, not deployment.
- Unsupported claim explicitly avoided: no claim of SOTA robot performance.
