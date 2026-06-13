# Submission Readiness Decision

Decision: KILL_ARCHIVE

ICLR main-conference readiness: NO.

Reason: v4 adds a real MuJoCo physics-violation audit benchmark, but the evidence is negative. The proposed explicit audit is matched by simple residual thresholds and learned uncertainty/reconstruction/classifier baselines on combined violation shift, and it false-flags 23.3% of nominal valid traces.

Honest terminal action: archive/kill for ICLR main. Do not submit this paper to ICLR main in its current form.

Revival condition: build a harder public or hardware benchmark where explicit physics audits detect policy failures that strong learned and residual baselines miss while maintaining low false positives on valid rollouts.
