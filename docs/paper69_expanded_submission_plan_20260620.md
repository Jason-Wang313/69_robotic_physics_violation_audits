# Paper 69 Expanded Submission Plan

Date: 2026-06-20

Target: rebuild Paper 69 into a 25+ page ICLR-style frozen submission archive under the expanded standard. The goal is not to make the result look good. The goal is to determine whether explicit robotic physics-violation audits survive hostile review against strong residual, learned, calibrated, and stress-test baselines.

## Starting Point

- Current version: v4 real MuJoCo rebuild.
- Current PDF: 4 pages at `C:/Users/wangz/Downloads/69.pdf`.
- Current terminal state: `KILL_ARCHIVE`.
- Current evidence scale: 420 main rollout summaries, 3,780 main method-evaluation rows, 480 ablation rows, and 1,200 stress rows.
- Current blocker: on combined violation shift, `physics_violation_audit` reaches F1 1.000 but is matched by kinematic residual, energy residual, ensemble uncertainty, autoencoder reconstruction, supervised classifier, and oracle labels.
- Current false-positive blocker: the explicit audit false-flags 23.3% of nominal valid MuJoCo traces.
- Current manuscript blocker: short paper, placeholder bibliography, no generated appendix tables, no expanded theory, and no validator for the final 25+ page artifact.

## Submission Claim To Test

The strongest defensible claim is:

> Explicit physics-audit structure can detect physically impossible robotic contact rollouts under subtle and mixed corruptions better than generic residual or learned anomaly baselines, while maintaining low false positives on valid but rare contact dynamics.

This claim must be falsified if either condition fails:

- Strong residual/learned/calibrated baselines match or beat the explicit audit on hard corruptions.
- The explicit audit cannot keep false positives low on valid, rare, or timestamp-skewed traces.

## Theory Additions

Add a formal section defining a rollout audit as a calibrated binary test over trajectory features:

- Rollout notation: states, controls, contact impulses, support indicators, work proxies, and sensor timestamps.
- Explicit audit energy: contact, support, work-energy, friction-slip, actuator, and causality terms.
- False-positive control proposition: if each term is calibrated at quantile `1 - alpha_j`, the union audit has a worst-case false-positive bound at `sum_j alpha_j` unless dependencies are modeled.
- Identifiability theorem: when injected corruptions are separable by a scalar residual already available to baselines, explicit multi-term physics audits are not identifiable as the necessary mechanism.
- Component-necessity criterion: a component is supported only if removing it lowers hard-split F1 or recall at fixed false-positive budget by a pre-registered margin.
- Negative theorem: benchmark designs with only high-severity synthetic corruptions can reward detectors that are not deployable on rare valid contact regimes.

## Experiment Expansion

### Implementation

- Add argparse CLI for seeds, episodes, training size, stress levels, workers, results dir, and figures dir.
- Keep CPU-only execution and bounded RAM by using compact sklearn models and streaming CSV writes.
- Isolate development runs under `results/dev_*`; preserve final frozen outputs under `results/`.
- Record a development log, then freeze protocol before the full run.

### Data Splits

Retain the original seven splits and add harder falsification splits:

- `rare_valid_bounce`: valid high-impulse contacts that should not be flagged.
- `valid_clock_skew`: asynchronous but physically valid sensor traces; should be handled or explicitly abstained.
- `valid_low_friction_slip`: valid slip under low friction; checks false-positive control.
- `subtle_contact_corruption`: lower-severity contact/acceleration mismatch.
- `subtle_energy_corruption`: lower-severity work-energy mismatch.
- `timestamp_noncausal_corruption`: true causality corruption that resembles clock skew.
- `adversarial_compensated_violation`: combined corruptions designed to hide from scalar residuals.
- `mixed_near_threshold`: corruptions near the calibration threshold.

### Methods

Evaluate the proposed method against uncomfortable baselines:

- Random flagger.
- Scalar kinematic residual.
- Scalar energy residual.
- Scalar contact impulse residual.
- Max-of-residuals detector.
- Logistic residual stack.
- HistGradientBoosting classifier.
- Random-forest ensemble uncertainty.
- IsolationForest anomaly detector.
- PCA reconstruction anomaly detector.
- Conformal residual ensemble.
- Calibrated learned stack.
- Old v4 explicit physics audit.
- New `physics_violation_audit_v5`.
- Oracle violation labels.

### Proposed v5 Development

Before freeze only:

- Calibrate per-term thresholds on valid and rare-valid training traces.
- Add separate rare-valid false-positive penalties.
- Add timestamp-skew handling: flag true noncausal jumps but avoid treating simple clock skew as impossible physics.
- Add a conformal aggregation option that reports low-confidence/abstain as non-submission evidence, not as a free success.
- Compare v5 against v4 and keep any improvement only if it is methodologically justified, not result-chasing.

### Metrics

Report all predefined metrics:

- F1, precision, recall, accuracy, flag rate, and false-positive rate by split.
- AUROC and AUPRC by split where score outputs exist.
- Recall at fixed false-positive budgets: 1%, 5%, and 10%.
- False-positive rate on nominal, rare-valid, clock-skew-valid, and low-friction-valid splits.
- Seed-level paired deltas against every baseline.
- Aggregate metrics across all hard corruptions and all valid regimes.
- Stress sweep over corruption severity, sensor noise, timestamp skew, friction, and impact rarity.

### Decision Gates

Use frozen gates after development:

- `KILL_ARCHIVE` if v5 fails to beat the strongest non-oracle aggregate hard-corruption baseline by at least 0.030 F1.
- `KILL_ARCHIVE` if v5 fails to beat the strongest non-oracle combined/adversarial split baseline by at least 0.030 F1.
- `KILL_ARCHIVE` if v5 exceeds 0.100 false-positive rate on any rare-valid split.
- `KILL_ARCHIVE` if v5 recall at 5% false-positive budget is not best or within 0.030 of best non-oracle.
- `KILL_ARCHIVE` if any removed-component ablation matches or beats full v5 within 0.020 F1 or fixed-FPR recall.
- `KILL_ARCHIVE` if v5 loses the maximum-stress gate to any non-oracle baseline.
- `STRONG_REVISE` only if all local gates pass; ICLR-main readiness still requires hardware or public benchmark validation.

## Manuscript Expansion

The final paper must be at least 25 pages without filler:

- Main paper: problem setup, mechanism, theory, calibration protocol, experimental design, main result, false-positive analysis, ablations, stress tests, failure cases, and limitations.
- Appendix: full metrics table, full paired statistics, all seed metrics, all ablations, stress sweep, negative cases, protocol freeze, and reproducibility checklist.
- Replace placeholder references with verified primary sources for MuJoCo, physics-informed modeling, robotic anomaly detection, safety monitoring, conformal prediction, and embodied-agent evaluation.
- Use bright boxed clickable in-text citations and cross-references through `hyperref`.

## Validation And Artifact Rules

- Build final numbered PDF only at `C:/Users/wangz/Downloads/69.pdf`.
- Never copy any PDF to the visible Desktop.
- Add `scripts/render_submission_assets.py`, `scripts/build_submission_pdf.ps1`, and `scripts/validate_submission_artifacts.py`.
- Validator must check expected row counts, generated figures, generated tables, citation-box settings, 25+ page PDF, Downloads-only output, and Desktop hygiene.
- Render representative PDF pages to PNG and visually inspect title/citations, figures, dense tables, appendix pages, and references.
- Commit and push the public GitHub repo after validation.
- Update `GLOBAL_POOL_STATUS.md`, `BATCH_STATUS.md`, `MASTER_REPORT.md`, `SUBMISSION_STATUS.md`, `MASTER_SUBMISSION_REPORT.md`, and `SUBMISSION_AUDIT_MATRIX.csv`.
