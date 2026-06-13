# Reviewer Attacks

- This is only uncertainty with different words.
- The evidence is synthetic and may not transfer to real hardware.
- The hostile prior work already contains contact-aware world models.
- The proposed mechanism may be too specialized for broad ICLR interest.

Response: the v4 MuJoCo rebuild confirms the harshest review. Explicit physics checks detect the injected violations, but so do simple residual and learned baselines, and false positives on valid traces are too high for an ICLR-main claim. The paper is killed/archived.
