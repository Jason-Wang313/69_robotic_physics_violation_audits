# Submission Readiness Decision

Decision: KILL_ARCHIVE

ICLR main-conference readiness: NO.

Reason: v5 expands the MuJoCo physics-violation audit benchmark under a frozen hostile-review protocol, but the evidence is still negative. `physics_violation_audit_v5` does not beat the strongest non-oracle hard-corruption baseline by the preregistered 0.030 F1 margin, does not beat the strongest combined/adversarial baseline, exceeds the rare-valid false-positive gate on at least one split, trails learned baselines at the fixed 5 percent false-positive budget, and fails the component-necessity ablation gate.

Honest terminal action: archive/kill for ICLR main. Do not submit this paper to ICLR main in its current form.

Revival condition: build a harder public or hardware benchmark where explicit physics audits detect policy failures that strong learned, residual, reconstruction, and conformal baselines miss while maintaining low false positives on valid rare-contact rollouts.

Final artifact: `C:/Users/wangz/Downloads/69.pdf`, 46 pages, SHA256 `395FE00C4D5222E0BC0A4B58E434EAE11C6F7D7079B29E11FE0383BC07666546`.
