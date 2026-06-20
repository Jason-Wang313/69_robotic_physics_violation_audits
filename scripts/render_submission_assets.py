"""Render Paper 69 CSV evidence into LaTeX assets and audit summaries."""

from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PAPER = ROOT / "paper"
GENERATED = PAPER / "generated"
DOCS = ROOT / "docs"

PROPOSED = "physics_violation_audit_v5"

METHOD_ORDER = [
    "random_flagger",
    "kinematic_residual_threshold",
    "energy_residual_threshold",
    "contact_impulse_threshold",
    "max_residual_detector",
    "logistic_residual_stack",
    "hgb_failure_classifier",
    "random_forest_ensemble",
    "isolation_forest_detector",
    "pca_reconstruction_audit",
    "conformal_residual_ensemble",
    "calibrated_learned_stack",
    "physics_violation_audit",
    PROPOSED,
    "oracle_violation_labels",
]

SELECTED_METHODS = [
    "kinematic_residual_threshold",
    "energy_residual_threshold",
    "max_residual_detector",
    "logistic_residual_stack",
    "hgb_failure_classifier",
    "calibrated_learned_stack",
    "physics_violation_audit",
    PROPOSED,
    "oracle_violation_labels",
]

HOSTILE_SPLITS = [
    "combined_violation_shift",
    "adversarial_compensated_violation",
    "mixed_near_threshold",
    "subtle_contact_corruption",
    "timestamp_noncausal_corruption",
]

VALID_SPLITS = [
    "nominal_valid",
    "rare_valid_bounce",
    "valid_clock_skew",
    "valid_low_friction_slip",
]

LABELS = {
    "random_flagger": "Random",
    "kinematic_residual_threshold": "Kin. residual",
    "energy_residual_threshold": "Energy residual",
    "contact_impulse_threshold": "Contact residual",
    "max_residual_detector": "Max residual",
    "logistic_residual_stack": "Logistic stack",
    "hgb_failure_classifier": "HGB classifier",
    "random_forest_ensemble": "RF ensemble",
    "isolation_forest_detector": "IsolationForest",
    "pca_reconstruction_audit": "PCA recon.",
    "conformal_residual_ensemble": "Conformal residual",
    "calibrated_learned_stack": "Calibrated stack",
    "physics_violation_audit": "Old audit",
    PROPOSED: "Audit v5",
    "oracle_violation_labels": "Oracle labels",
    "full_physics_violation_audit_v5": "Full v5",
    "no_contact_check": "No contact",
    "no_support_check": "No support",
    "no_energy_check": "No energy",
    "no_friction_slip_check": "No friction",
    "no_actuator_check": "No actuator",
    "no_causality_check": "No causality",
    "no_timestamp_guard": "No timestamp",
    "no_rare_valid_guard": "No rare-valid",
    "no_conformal_aggregation": "No conformal agg.",
    "old_v4_physics_audit": "Old v4",
    "scalar_residual_only": "Scalar only",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _ in reader)


def f3(value: object) -> str:
    return f"{float(value):.3f}"


def esc(text: object) -> str:
    out = str(text)
    return (
        out.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("~", "\\textasciitilde{}")
        .replace("^", "\\textasciicircum{}")
    )


def label(name: str) -> str:
    return LABELS.get(name, name.replace("_", " "))


def method_rank(name: str) -> int:
    return METHOD_ORDER.index(name) if name in METHOD_ORDER else len(METHOD_ORDER)


def write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def parse_summary() -> dict[str, str]:
    text = (RESULTS / "summary.txt").read_text(encoding="utf-8")
    out: dict[str, str] = {}
    decision = re.search(r"Terminal decision:\s*(.+)", text)
    reason = re.search(r"Terminal reason:\s*(.+)", text)
    counts = re.search(
        r"Main raw rollouts:\s*(\d+);\s*main eval rows:\s*(\d+);\s*ablation rows:\s*(\d+);\s*stress rows:\s*(\d+)",
        text,
    )
    training = re.search(r"Training scenes:\s*(\d+)", text)
    out["decision"] = decision.group(1).strip() if decision else "UNKNOWN"
    out["reason"] = reason.group(1).strip() if reason else "missing"
    out["training_rows"] = training.group(1) if training else str(count_rows(RESULTS / "training_audit_rollouts.csv"))
    if counts:
        out["rollout_rows"], out["main_rows"], out["ablation_rows"], out["stress_rows"] = counts.groups()
    else:
        out["rollout_rows"] = str(count_rows(RESULTS / "physics_audit_rollouts.csv"))
        out["main_rows"] = str(count_rows(RESULTS / "physics_audit_raw.csv"))
        out["ablation_rows"] = str(count_rows(RESULTS / "physics_audit_ablation_raw.csv"))
        out["stress_rows"] = str(count_rows(RESULTS / "stress_sweep.csv"))
    return out


def render_macros() -> str:
    summary = parse_summary()
    return "\n".join(
        [
            f"\\newcommand{{\\PaperDecision}}{{{esc(summary['decision'])}}}",
            f"\\newcommand{{\\PaperDecisionReason}}{{{esc(summary['reason'])}}}",
            f"\\newcommand{{\\TrainingRows}}{{{summary['training_rows']}}}",
            f"\\newcommand{{\\RolloutRows}}{{{summary['rollout_rows']}}}",
            f"\\newcommand{{\\MainRows}}{{{summary['main_rows']}}}",
            f"\\newcommand{{\\AblationRows}}{{{summary['ablation_rows']}}}",
            f"\\newcommand{{\\StressRows}}{{{summary['stress_rows']}}}",
            f"\\newcommand{{\\SeedMetricRows}}{{{count_rows(RESULTS / 'raw_seed_metrics.csv')}}}",
            f"\\newcommand{{\\PairwiseRows}}{{{count_rows(RESULTS / 'physics_audit_pairwise.csv')}}}",
            "",
        ]
    )


def render_aggregate(rows: list[dict[str, str]]) -> str:
    by_key = {(row["aggregate_group"], row["method"]): row for row in rows}
    body = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Frozen aggregate metrics. F1/recall are higher-is-better; valid-regime FPR is lower-is-better.}",
        "\\label{tab:aggregate-main}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.5pt}",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Method & Hard F1 & Hard Rec. & Comb. F1 & Comb. Rec. & Valid FPR & Fixed-5\\% Rec. \\\\",
        "\\midrule",
    ]
    fixed = {
        row["method"]: row
        for row in read_csv(RESULTS / "fixed_fpr_metrics.csv")
        if row["fpr_budget"] == "0.05"
    }
    for method in METHOD_ORDER:
        hard = by_key[("hard_corruptions", method)]
        combined = by_key[("combined_and_adversarial", method)]
        valid = by_key[("valid_regimes", method)]
        body.append(
            f"{esc(label(method))} & {f3(hard['f1'])} & {f3(hard['recall'])} & "
            f"{f3(combined['f1'])} & {f3(combined['recall'])} & {f3(valid['false_positive_rate'])} & "
            f"{f3(fixed[method]['hard_recall_at_budget'])} \\\\"
        )
    body += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(body)


def render_selected_splits(metrics: list[dict[str, str]]) -> str:
    by_key = {(row["split"], row["method"]): row for row in metrics}
    body = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Hostile split F1. These are the splits most likely to falsify explicit physics-audit claims.}",
        "\\label{tab:selected-splits}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{1.8pt}",
        "\\begin{tabular}{lrrrrrrrrr}",
        "\\toprule",
        "Split & Kin. & Energy & Max & Logit & HGB & Stack & Old & v5 & Oracle \\\\",
        "\\midrule",
    ]
    for split in HOSTILE_SPLITS:
        vals = [f3(by_key[(split, method)]["f1"]) for method in SELECTED_METHODS]
        body.append(f"{esc(split)} & " + " & ".join(vals) + " \\\\")
    body += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(body)


def render_valid_fpr(metrics: list[dict[str, str]]) -> str:
    by_key = {(row["split"], row["method"]): row for row in metrics}
    methods = [
        "kinematic_residual_threshold",
        "energy_residual_threshold",
        "max_residual_detector",
        "hgb_failure_classifier",
        "calibrated_learned_stack",
        "physics_violation_audit",
        PROPOSED,
    ]
    body = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{False-positive rates on valid regimes. Rare valid contacts and clock skew are the hostile reviewer tests.}",
        "\\label{tab:valid-fpr}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.0pt}",
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "Valid split & Kin. & Energy & Max & HGB & Stack & Old & v5 \\\\",
        "\\midrule",
    ]
    for split in VALID_SPLITS:
        vals = [f3(by_key[(split, method)]["false_positive_rate"]) for method in methods]
        body.append(f"{esc(split)} & " + " & ".join(vals) + " \\\\")
    body += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(body)


def render_ablation(rows: list[dict[str, str]]) -> str:
    selected = [row for row in rows if row["split"] == "combined_violation_shift"]
    body = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Combined-shift ablations. A mechanism is not identified if removed components match full v5.}",
        "\\label{tab:ablation-combined}",
        "\\scriptsize",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Variant & Episodes & F1 & Precision & Recall \\\\",
        "\\midrule",
    ]
    for row in sorted(selected, key=lambda r: (-float(r["f1"]), label(r["method"]))):
        body.append(
            f"{esc(label(row['method']))} & {row['episodes']} & {f3(row['f1'])} & "
            f"{f3(row['precision'])} & {f3(row['recall'])} \\\\"
        )
    body += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(body)


def render_fixed_fpr(rows: list[dict[str, str]]) -> str:
    selected = [row for row in rows if row["fpr_budget"] == "0.05"]
    body = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Recall at a 5\\% false-positive budget. Thresholds are calibrated from valid-regime scores.}",
        "\\label{tab:fixed-fpr}",
        "\\scriptsize",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Method & Threshold & Hard recall & Valid scores \\\\",
        "\\midrule",
    ]
    for row in sorted(selected, key=lambda r: method_rank(r["method"])):
        body.append(
            f"{esc(label(row['method']))} & {f3(row['threshold'])} & {f3(row['hard_recall_at_budget'])} & {row['valid_scores']} \\\\"
        )
    body += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(body)


def render_stress(rows: list[dict[str, str]]) -> str:
    levels = ["0.00", "0.20", "0.40", "0.60", "0.80", "1.00"]
    split = "combined_violation_shift"
    by_key = {(row["method"], row["stress_level"], row["split"]): row for row in rows}
    body = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Maximum-stress trajectory. Entries are F1 on the combined-violation stress split.}",
        "\\label{tab:stress}",
        "\\scriptsize",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Method & 0.00 & 0.20 & 0.40 & 0.60 & 0.80 & 1.00 \\\\",
        "\\midrule",
    ]
    for method in [
        "kinematic_residual_threshold",
        "energy_residual_threshold",
        "max_residual_detector",
        "hgb_failure_classifier",
        "random_forest_ensemble",
        "calibrated_learned_stack",
        PROPOSED,
    ]:
        vals = []
        for level in levels:
            key = (method, level, f"stress_{level}_{split}")
            vals.append(f3(by_key[key]["f1"]))
        body.append(f"{esc(label(method))} & " + " & ".join(vals) + " \\\\")
    body += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(body)


def render_negative_cases(rows: list[dict[str, str]]) -> str:
    body = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Pre-specified negative cases beyond the local benchmark scope.}",
        "\\label{tab:negative-cases}",
        "\\scriptsize",
        "\\begin{tabular}{p{0.22\\linewidth}p{0.34\\linewidth}p{0.34\\linewidth}}",
        "\\toprule",
        "Case & Observed failure mode & Submission implication \\\\",
        "\\midrule",
    ]
    for row in rows:
        case_name = row["case"].replace("_", " ")
        body.append(
            f"{esc(case_name)} & {esc(row['observed_failure_mode'])} & {esc(row['submission_implication'])} \\\\"
        )
    body += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(body)


def longtable_metrics(rows: list[dict[str, str]], caption: str, label_name: str) -> str:
    body = [
        "{\\scriptsize",
        "\\setlength{\\tabcolsep}{2.5pt}",
        "\\begin{longtable}{llrrrrrr}",
        f"\\caption{{{esc(caption)}}}\\label{{{label_name}}}\\\\",
        "\\toprule",
        "Split & Method & Episodes & F1 & Prec. & Rec. & FPR & AUROC \\\\",
        "\\midrule",
        "\\endfirsthead",
        "\\toprule",
        "Split & Method & Episodes & F1 & Prec. & Rec. & FPR & AUROC \\\\",
        "\\midrule",
        "\\endhead",
    ]
    for row in sorted(rows, key=lambda r: (r.get("split", ""), method_rank(r["method"]))):
        body.append(
            f"{esc(row.get('split', 'all'))} & {esc(label(row['method']))} & {row['episodes']} & "
            f"{f3(row['f1'])} & {f3(row['precision'])} & {f3(row['recall'])} & "
            f"{f3(row['false_positive_rate'])} & {f3(row['auroc'])} \\\\"
        )
    body += ["\\bottomrule", "\\end{longtable}", "}", ""]
    return "\n".join(body)


def longtable_pairwise(rows: list[dict[str, str]]) -> str:
    body = [
        "{\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{longtable}{llrrr}",
        "\\caption{All split-level paired seed comparisons versus audit v5. Positive values favor audit v5.}\\label{tab:full-pairwise}\\\\",
        "\\toprule",
        "Split & Baseline & Diff & t approx. & p approx. \\\\",
        "\\midrule",
        "\\endfirsthead",
        "\\toprule",
        "Split & Baseline & Diff & t approx. & p approx. \\\\",
        "\\midrule",
        "\\endhead",
    ]
    for row in rows:
        body.append(
            f"{esc(row['split'])} & {esc(label(row['baseline']))} & {f3(row['mean_f1_diff_vs_v5'])} & "
            f"{f3(row['paired_t_approx'])} & {f3(row['normal_approx_p'])} \\\\"
        )
    body += ["\\bottomrule", "\\end{longtable}", "}", ""]
    return "\n".join(body)


def write_terminal_doc() -> None:
    summary = parse_summary()
    text = "\n".join(
        [
            "# Paper 69 Expanded Terminal Decision",
            "",
            f"Decision: {summary['decision']}",
            "",
            f"Reason: {summary['reason']}",
            "",
            f"Training rows: {summary['training_rows']}",
            f"Main rollout rows: {summary['rollout_rows']}",
            f"Main method-evaluation rows: {summary['main_rows']}",
            f"Ablation rows: {summary['ablation_rows']}",
            f"Stress rows: {summary['stress_rows']}",
            "",
            "This decision is generated from frozen CSV artifacts, not hand-transcribed table values.",
            "",
        ]
    )
    write(DOCS / "paper69_expanded_terminal_decision_20260620.md", text)


def main() -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    metrics = read_csv(RESULTS / "physics_audit_metrics.csv")
    aggregate = read_csv(RESULTS / "aggregate_metrics.csv")
    ablation = read_csv(RESULTS / "physics_audit_ablation.csv")
    stress = read_csv(RESULTS / "stress_sweep.csv")
    pairwise = read_csv(RESULTS / "pairwise_stats.csv")
    negatives = read_csv(RESULTS / "negative_cases.csv")
    seed_rows = read_csv(RESULTS / "raw_seed_metrics.csv")
    fixed_rows = read_csv(RESULTS / "fixed_fpr_metrics.csv")

    write(GENERATED / "result_macros.tex", render_macros())
    write(GENERATED / "aggregate_metrics_table.tex", render_aggregate(aggregate))
    write(GENERATED / "selected_split_table.tex", render_selected_splits(metrics))
    write(GENERATED / "valid_fpr_table.tex", render_valid_fpr(metrics))
    write(GENERATED / "ablation_table.tex", render_ablation(ablation))
    write(GENERATED / "fixed_fpr_table.tex", render_fixed_fpr(fixed_rows))
    write(GENERATED / "stress_table.tex", render_stress(stress))
    write(GENERATED / "negative_cases_table.tex", render_negative_cases(negatives))
    write(GENERATED / "full_metrics_longtable.tex", longtable_metrics(metrics, "Full split-level main metrics.", "tab:full-metrics"))
    write(GENERATED / "full_aggregate_longtable.tex", longtable_metrics(aggregate, "Full aggregate metrics.", "tab:full-aggregate"))
    write(GENERATED / "full_ablation_longtable.tex", longtable_metrics(ablation, "Full ablation metrics.", "tab:full-ablation"))
    write(GENERATED / "full_stress_longtable.tex", longtable_metrics(stress, "Full stress-sweep metrics.", "tab:full-stress"))
    write(GENERATED / "full_pairwise_longtable.tex", longtable_pairwise(pairwise))
    write(
        GENERATED / "seed_metrics_selected_longtable.tex",
        longtable_metrics(
            [row for row in seed_rows if row["method"] in {"kinematic_residual_threshold", "hgb_failure_classifier", "calibrated_learned_stack", PROPOSED}],
            "Selected seed-level metrics for reproducibility.",
            "tab:seed-metrics",
        ),
    )
    write(
        GENERATED / "all_seed_metrics_longtable.tex",
        longtable_metrics(seed_rows, "All method/split/seed metrics from the frozen run.", "tab:all-seed-metrics"),
    )
    write_terminal_doc()
    print(f"Rendered Paper 69 generated assets in {GENERATED}")


if __name__ == "__main__":
    main()
