# Final Audit

1. Chosen thesis: Robotic Physics Violation Audits explores `Audit robot policies by the physics violations they silently tolerate.` for foundation-model evaluation for embodied agents.
2. ICLR-main decision: KILL_ARCHIVE.
3. Submission-hardening version: v4 real MuJoCo rebuild.
4. Reason: real MuJoCo evidence falsifies the novelty claim. The explicit physics audit reaches F1 1.000 on combined violations, but kinematic residuals, energy residuals, ensemble uncertainty, autoencoder reconstruction, and supervised classifiers also reach F1 1.000; the audit false-flags 23.3% of nominal valid traces.
5. Closest hostile prior work: see `docs/hostile_prior_work.md`, `docs/hostile_prior_work_100_cards.csv`, and `docs/hostile_reviewer_response.md`.
6. Reproducibility: `python src\run_experiment.py` reproduces the MuJoCo rollouts, corruption injections, learned baselines, CSVs, figures, ablations, pairwise stats, stress sweep, and negative cases.
7. Claim-validity status: main-conference claims killed by direct empirical evidence; archive retained as a negative result.
8. Exact Downloads PDF path: `C:/Users/wangz/Downloads/69.pdf`
9. GitHub URL: https://github.com/Jason-Wang313/69_robotic_physics_violation_audits
10. Confirmation: no visible Desktop copy was requested or made.
