# Paper 69 Development Log

Date: 2026-06-20

Policy: development runs may repair execution defects and obvious pre-freeze modeling bugs. No result-chasing is allowed after protocol freeze.

## Starting Point

The v4 archive is real but too small for the expanded submission-hardening standard.

- 5 seeds.
- 12 episodes per seed.
- 7 main splits.
- 9 methods.
- 3,780 main method-evaluation rows.
- 480 ablation rows.
- 1,200 stress rows.
- Terminal decision: KILL_ARCHIVE.
- Primary reason: explicit physics audit is matched by residual and learned baselines on combined violation shift and false-flags nominal valid traces.

## Planned Development Changes

- Add CLI configurability and output isolation.
- Add valid rare-contact, clock-skew, low-friction, subtle-corruption, timestamp-corruption, adversarial, and near-threshold splits.
- Add stronger baselines: max residual, logistic residual stack, HGB classifier, random-forest ensemble, IsolationForest, PCA reconstruction, conformal residual ensemble, and calibrated learned stack.
- Add `physics_violation_audit_v5` with rare-valid and timestamp guards.
- Add aggregate metrics, fixed-FPR recall, all-split paired statistics, expanded ablations, and stress splits.
- Add generated table/PDF/validation scripts after the frozen run.

## Development Runs

### Dev1: expanded runner bootstrap

Command scale: 2 seeds, 2 main episodes, 2 ablation episodes, 2 stress episodes, 30 training scenes, 7 main splits, 2 ablation splits, and 2 stress levels.

Output: `results/dev_20260620_2337`

Initial issue:

- The preliminary training path attempted to score the calibrated learned stack before the stacker had been trained.
- Repair: preliminary scores use the HGB classifier as a placeholder until the stacker and stack-scaler exist.

### Dev2: paired-stat and ablation-gate hygiene

Same command scale and output directory.

Repairs:

- Seed-pair lookup now keeps numeric seed keys.
- Zero-variance paired statistics now report finite sentinel t-statistics rather than unbounded values.
- The mechanism-ablation gate ignores valid-only splits and evaluates component necessity on corruption splits.

Result:

- Main rows: 420.
- Metrics rows: 105.
- Terminal decision: KILL_ARCHIVE.
- v5 does not beat `calibrated_learned_stack` on hard-corruption or combined/adversarial aggregate metrics.
- v5 recall at 5% FPR trails `logistic_residual_stack`.
- Multiple ablations match or beat full v5 on the corruption split.

## Pre-freeze Choice

The frozen implementation uses the repaired expanded runner. This is not tuned to make the method pass. The dev runs show the central negative result persists under stronger baselines and harder splits.
