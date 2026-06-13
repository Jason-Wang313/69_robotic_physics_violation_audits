# 69 Robotic Physics Violation Audits

Submission-hardening version: v4 real MuJoCo rebuild

Terminal decision: KILL_ARCHIVE for ICLR main conference.

The repository is retained as an archive of a falsified physics-audit mechanism. The v4 rebuild replaces the synthetic probability scaffold with MuJoCo contact rollouts, controlled physics-violation corruptions, explicit physics audits, learned uncertainty/reconstruction/classifier baselines, ablations, stress sweeps, and negative cases.

The proposed `physics_violation_audit` does not survive the ICLR-main gate. On combined violation shift it reaches perfect F1, but so do kinematic residual, energy residual, ensemble uncertainty, autoencoder reconstruction, and supervised classifier baselines. On nominal valid rollouts it also false-flags 23.3% of traces.

## Reproduce Real Evidence

```powershell
python src\run_experiment.py
```

The run writes MuJoCo rollout summaries, raw method evaluations, metrics, seed metrics, pairwise comparisons, ablations, stress sweeps, negative cases, learned-baseline summaries, and figures into `results/` and `figures/`.

## Rebuild Archive PDF

```powershell
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Canonical local PDF: `C:/Users/wangz/Downloads/69.pdf`
