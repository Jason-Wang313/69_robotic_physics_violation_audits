# Paper 69 Expanded Terminal Decision

Decision: KILL_ARCHIVE

Reason: v5 does not beat strongest non-oracle hard-corruption baseline calibrated_learned_stack by 0.030; v5 does not beat strongest combined/adversarial baseline calibrated_learned_stack by 0.030; v5 false-positive rate on at least one rare-valid split is 0.286 > 0.100; v5 recall at 5% FPR trails hgb_failure_classifier by more than 0.030; ablation gate fails because no_actuator_check, no_causality_check, no_conformal_aggregation, no_contact_check, no_energy_check, no_friction_slip_check, no_rare_valid_guard, no_support_check, no_timestamp_guard, old_v4_physics_audit, scalar_residual_only matches or beats full v5

Training rows: 192
Main rollout rows: 1680
Main method-evaluation rows: 25200
Ablation rows: 3840
Stress rows: 9216

This decision is generated from frozen CSV artifacts, not hand-transcribed table values.

Final PDF: C:/Users/wangz/Downloads/69.pdf

Final PDF pages: 46

Final PDF SHA256: 395FE00C4D5222E0BC0A4B58E434EAE11C6F7D7079B29E11FE0383BC07666546

Validation: `python scripts\validate_submission_artifacts.py` passes, and rendered-page visual QA passed for the title/citation page, main result pages, appendix tables, and references.

Desktop hygiene: no PDF copy exists at C:/Users/wangz/Desktop/69.pdf.
