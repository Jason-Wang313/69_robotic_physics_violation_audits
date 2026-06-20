# Plan

Build paper 69 `robotic_physics_violation_audits` from the shared pool, compile PDF to Downloads only, and publish the exact-name public repo.

2026-06-20 expanded-standard plan:

1. Use `docs/paper69_expanded_submission_plan_20260620.md` as the controlling plan.
2. Expand the benchmark, theory, baselines, ablations, stress tests, validation scripts, and manuscript before finalizing any terminal decision.
3. Freeze the protocol before the full run; after freeze, only fix execution, serialization, plotting, or PDF defects that do not alter the method or decision gates.
4. Keep CPU-only execution and bounded RAM, but do not weaken the empirical or theoretical standard for convenience.
5. Preserve `KILL_ARCHIVE` unless explicit physics audits clear residual, learned, calibrated, ablation, false-positive, and maximum-stress gates.

2026-06-15 continuation plan:

1. Re-audit the real MuJoCo physics-violation evidence before changing the terminal decision.
2. Verify code compilation, training/rollout/main/ablation/stress CSV integrity, residual and learned baselines, uncertainty, PDF rebuild, Downloads-only artifact placement, and public GitHub availability.
3. Keep `KILL_ARCHIVE` unless explicit physics audits beat residual and learned baselines while maintaining low false-positive rates on valid traces.
4. Preserve the paper as a negative result because the current evidence shows perfect combined-shift F1 is matched by simpler baselines and nominal false positives remain too high.
