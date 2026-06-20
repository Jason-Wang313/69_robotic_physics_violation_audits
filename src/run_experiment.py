import argparse
import csv
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


BASE_SEED = 2129968627
DEFAULT_SEEDS = list(range(8))
DEFAULT_EPISODES_PER_SEED = 14
DEFAULT_ABLATION_EPISODES_PER_SEED = 10
DEFAULT_STRESS_EPISODES_PER_SEED = 8
DEFAULT_TRAIN_SCENES = 192
DEFAULT_WORKERS = max(1, min(4, int(os.environ.get("PAPER69_WORKERS", "4"))))

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
SEEDS = DEFAULT_SEEDS.copy()
MAX_WORKERS = DEFAULT_WORKERS

OBJECT_HALF = 0.04
FINGER_RADIUS = 0.015
DT = 0.01

VALID_SPLITS = [
    "nominal_valid",
    "rare_valid_bounce",
    "valid_clock_skew",
    "valid_low_friction_slip",
]

CORRUPTION_SPLITS = [
    "contact_corruption",
    "energy_work_corruption",
    "support_levitation",
    "actuator_saturation",
    "noncausal_teleport",
    "combined_violation_shift",
    "subtle_contact_corruption",
    "subtle_energy_corruption",
    "timestamp_noncausal_corruption",
    "adversarial_compensated_violation",
    "mixed_near_threshold",
]

MAIN_SPLITS = VALID_SPLITS + CORRUPTION_SPLITS

PROPOSED_METHOD = "physics_violation_audit_v5"

METHODS = [
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
    PROPOSED_METHOD,
    "oracle_violation_labels",
]

ABLATIONS = [
    "full_physics_violation_audit_v5",
    "no_contact_check",
    "no_support_check",
    "no_energy_check",
    "no_friction_slip_check",
    "no_actuator_check",
    "no_causality_check",
    "no_timestamp_guard",
    "no_rare_valid_guard",
    "no_conformal_aggregation",
    "old_v4_physics_audit",
    "scalar_residual_only",
]

STRESS_METHODS = [
    "kinematic_residual_threshold",
    "energy_residual_threshold",
    "max_residual_detector",
    "hgb_failure_classifier",
    "random_forest_ensemble",
    "pca_reconstruction_audit",
    "calibrated_learned_stack",
    PROPOSED_METHOD,
]

FEATURE_NAMES = [
    "max_pose_jump",
    "max_accel",
    "contact_without_accel",
    "motion_without_contact",
    "energy_work_mismatch",
    "support_violation",
    "penetration_depth",
    "friction_slip_inconsistency",
    "actuator_saturation_score",
    "causality_jump_score",
    "max_contact_force",
    "mean_contact_force",
    "path_length",
    "work_proxy",
    "kinetic_energy_gain",
    "z_range",
    "timestamp_skew_score",
    "impact_consistency_score",
    "rare_valid_contact_score",
    "noise_level",
]


@dataclass
class Rollout:
    pos: np.ndarray
    vel: np.ndarray
    pusher: np.ndarray
    ctrl: np.ndarray
    contact_force: np.ndarray
    actuator_sat: np.ndarray
    penetration: np.ndarray
    support: np.ndarray
    work: np.ndarray
    friction: float
    mass: float
    split: str
    severity: float
    label: int
    unsafe: int
    corruption: str
    timestamp_skew: float


MODEL_CACHE: dict[tuple[float, float], mujoco.MjModel] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper 69 expanded MuJoCo physics-audit evidence runner")
    parser.add_argument("--seeds", type=int, default=len(DEFAULT_SEEDS))
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES_PER_SEED)
    parser.add_argument("--ablation-episodes", type=int, default=DEFAULT_ABLATION_EPISODES_PER_SEED)
    parser.add_argument("--stress-episodes", type=int, default=DEFAULT_STRESS_EPISODES_PER_SEED)
    parser.add_argument("--train-scenes", type=int, default=DEFAULT_TRAIN_SCENES)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--splits", nargs="*", default=MAIN_SPLITS)
    parser.add_argument(
        "--ablation-splits",
        nargs="*",
        default=["combined_violation_shift", "adversarial_compensated_violation", "mixed_near_threshold", "rare_valid_bounce"],
    )
    parser.add_argument(
        "--stress-splits",
        nargs="*",
        default=["combined_violation_shift", "adversarial_compensated_violation", "rare_valid_bounce"],
    )
    parser.add_argument("--stress-levels", nargs="*", type=float, default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "figures")
    return parser.parse_args()


def stable_int(text: str) -> int:
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(text))


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    arr = np.asarray(values, dtype=float)
    return float(1.96 * arr.std(ddof=1) / math.sqrt(len(arr)))


def normal_p_from_t(t_stat: float) -> float:
    return float(math.erfc(abs(t_stat) / math.sqrt(2.0)))


def model_xml(friction: float, mass: float) -> str:
    return f"""
<mujoco model="physics_violation_audit">
  <compiler angle="radian" coordinate="local"/>
  <option timestep="{DT}" gravity="0 0 -9.81" integrator="RK4" cone="elliptic"/>
  <default>
    <geom condim="4" solref="0.007 1" solimp="0.90 0.95 0.001"/>
  </default>
  <worldbody>
    <geom name="table" type="plane" size="1.0 1.0 0.05" friction="{friction:.4f} 0.004 0.0001" rgba="0.82 0.84 0.83 1"/>
    <body name="object" pos="0 0 {OBJECT_HALF}">
      <freejoint name="object_free"/>
      <geom name="object_geom" type="box" size="{OBJECT_HALF} {OBJECT_HALF} {OBJECT_HALF}" mass="{mass:.4f}"
            friction="{friction:.4f} 0.004 0.0001" rgba="0.12 0.37 0.78 1"/>
    </body>
    <body name="pusher" pos="0 0 {OBJECT_HALF}">
      <joint name="px" type="slide" axis="1 0 0" range="-0.75 0.75" damping="4"/>
      <joint name="py" type="slide" axis="0 1 0" range="-0.55 0.55" damping="4"/>
      <geom name="finger_geom" type="capsule" fromto="0 -0.035 0 0 0.035 0" size="{FINGER_RADIUS}"
            mass="0.08" friction="1.8 0.006 0.0001" rgba="0.84 0.22 0.12 1"/>
    </body>
  </worldbody>
  <actuator>
    <position name="ax" joint="px" kp="650" ctrlrange="-0.75 0.75"/>
    <position name="ay" joint="py" kp="650" ctrlrange="-0.55 0.55"/>
  </actuator>
</mujoco>
"""


def get_model(friction: float, mass: float) -> mujoco.MjModel:
    key = (round(float(friction), 3), round(float(mass), 3))
    if key not in MODEL_CACHE:
        MODEL_CACHE[key] = mujoco.MjModel.from_xml_string(model_xml(*key))
    return MODEL_CACHE[key]


def contact_force(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    obj_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
    finger_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "finger_geom")
    force = np.zeros(6, dtype=float)
    total = 0.0
    for cidx in range(data.ncon):
        contact = data.contact[cidx]
        if {contact.geom1, contact.geom2} == {obj_gid, finger_gid}:
            mujoco.mj_contactForce(model, data, cidx, force)
            total += float(np.linalg.norm(force[:3]))
    return total


def generate_valid_rollout(seed: int, episode: int, split: str, severity: float = 0.0) -> Rollout:
    rng = np.random.default_rng(BASE_SEED + seed * 1009 + episode * 7919 + stable_int(split))
    if split == "valid_low_friction_slip":
        friction = float(clamp(rng.uniform(0.16, 0.34) - 0.04 * severity, 0.12, 0.42))
    else:
        friction = float(clamp(rng.uniform(0.35, 1.05) - 0.12 * severity, 0.18, 1.20))
    mass = float(rng.uniform(0.12, 0.24))
    model = get_model(friction, mass)
    data = mujoco.MjData(model)
    obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    px_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "px")
    py_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "py")
    px_adr = model.jnt_qposadr[px_jid]
    py_adr = model.jnt_qposadr[py_jid]

    obj0 = rng.uniform([-0.035, -0.060], [0.035, 0.060])
    direction = np.array([1.0, rng.uniform(-0.35, 0.35)], dtype=float)
    direction = direction / np.linalg.norm(direction)
    start = obj0 - direction * (OBJECT_HALF + FINGER_RADIUS + 0.016)
    push_scale = rng.uniform(0.18, 0.30)
    if split == "rare_valid_bounce":
        push_scale = rng.uniform(0.32, 0.44)
    if split == "valid_low_friction_slip":
        push_scale = rng.uniform(0.26, 0.38)
    if split == "actuator_saturation":
        push_scale = rng.uniform(0.30, 0.42)
    end = obj0 + direction * push_scale

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[0] = obj0[0]
    data.qpos[1] = obj0[1]
    data.qpos[2] = OBJECT_HALF
    data.qpos[3] = 1.0
    data.qpos[px_adr] = start[0]
    data.qpos[py_adr] = start[1]
    data.ctrl[:] = start
    mujoco.mj_forward(model, data)

    pos, vel, pusher, ctrl, force, sat, pen, support, work = [], [], [], [], [], [], [], [], []
    prev_pusher = start.copy()
    for _ in range(8):
        data.ctrl[:] = start
        mujoco.mj_step(model, data)
    steps = 92
    for t in range(steps):
        alpha = (t + 1) / steps
        desired = start * (1 - alpha) + end * alpha
        if split in {"actuator_saturation", "rare_valid_bounce"}:
            desired = desired + direction * (0.04 * math.sin(t * 0.3))
        if split == "valid_low_friction_slip":
            desired = desired + np.array([0.0, 0.035 * math.sin(t * 0.18)])
        ctrl_clamped = np.array([clamp(float(desired[0]), -0.72, 0.72), clamp(float(desired[1]), -0.52, 0.52)])
        data.ctrl[:] = ctrl_clamped
        mujoco.mj_step(model, data)
        f = contact_force(model, data)
        obj_pos = data.xpos[obj_bid].copy()
        p = np.array([data.qpos[px_adr], data.qpos[py_adr], OBJECT_HALF], dtype=float)
        v = data.cvel[obj_bid][3:6].copy()
        pos.append(obj_pos)
        vel.append(v)
        pusher.append(p)
        ctrl.append(np.array([ctrl_clamped[0], ctrl_clamped[1], OBJECT_HALF]))
        force.append(f)
        sat.append(float(np.linalg.norm(desired - ctrl_clamped) > 1e-5))
        pen.append(max(0.0, OBJECT_HALF - obj_pos[2]))
        support.append(float(obj_pos[2] <= OBJECT_HALF + 0.012 or f > 1e-4))
        work.append(float(f * np.linalg.norm(p[:2] - prev_pusher)))
        prev_pusher = p[:2].copy()

    timestamp_skew = 0.0
    if split == "valid_clock_skew":
        timestamp_skew = float(rng.uniform(0.015, 0.045))
    return Rollout(
        pos=np.asarray(pos),
        vel=np.asarray(vel),
        pusher=np.asarray(pusher),
        ctrl=np.asarray(ctrl),
        contact_force=np.asarray(force),
        actuator_sat=np.asarray(sat),
        penetration=np.asarray(pen),
        support=np.asarray(support),
        work=np.asarray(work),
        friction=friction,
        mass=mass,
        split=split,
        severity=severity,
        label=0,
        unsafe=0,
        corruption="valid",
        timestamp_skew=timestamp_skew,
    )


def corrupt_rollout(base: Rollout, split: str, severity: float, rng: np.random.Generator) -> Rollout:
    r = Rollout(
        pos=base.pos.copy(),
        vel=base.vel.copy(),
        pusher=base.pusher.copy(),
        ctrl=base.ctrl.copy(),
        contact_force=base.contact_force.copy(),
        actuator_sat=base.actuator_sat.copy(),
        penetration=base.penetration.copy(),
        support=base.support.copy(),
        work=base.work.copy(),
        friction=base.friction,
        mass=base.mass,
        split=split,
        severity=severity,
        label=0 if split in VALID_SPLITS else 1,
        unsafe=0 if split in VALID_SPLITS else 1,
        corruption=split,
        timestamp_skew=base.timestamp_skew,
    )
    if split in VALID_SPLITS:
        noise = rng.normal(0.0, 0.0015 * (1 + severity), size=r.pos.shape)
        r.pos += noise
        return r

    idx = int(rng.integers(18, len(r.pos) - 20))
    sev = severity
    subtle = split in {"subtle_contact_corruption", "subtle_energy_corruption", "mixed_near_threshold"}
    subtle_scale = 0.38 if subtle else 1.0

    if split in {"contact_corruption", "combined_violation_shift", "subtle_contact_corruption", "mixed_near_threshold"}:
        scale = sev * subtle_scale
        r.contact_force[idx : idx + 8] += 450.0 * scale
        r.vel[idx : idx + 8, :2] *= 0.20 + 0.40 * (1 - subtle_scale)
        r.penetration[idx : idx + 8] += 0.010 * scale
    if split in {"energy_work_corruption", "combined_violation_shift", "subtle_energy_corruption", "mixed_near_threshold"}:
        scale = sev * subtle_scale
        jump = np.array([0.030 * scale, -0.018 * scale, 0.0])
        r.vel[idx : idx + 14, :2] += np.array([1.6 * scale, -0.8 * scale])
        r.pos[idx:, :] += jump
        r.work[idx : idx + 14] *= 0.08 + 0.30 * (1 - subtle_scale)
    if split in {"support_levitation", "combined_violation_shift", "mixed_near_threshold"}:
        scale = sev * (0.45 if split == "mixed_near_threshold" else 1.0)
        r.pos[idx : idx + 16, 2] += 0.085 * scale
        r.vel[idx : idx + 16, 2] = 0.0
        r.contact_force[idx : idx + 16] *= 0.0
        r.support[idx : idx + 16] = 0.0
    if split in {"actuator_saturation", "combined_violation_shift"}:
        r.actuator_sat[idx : idx + 18] = 1.0
        r.ctrl[idx : idx + 18, :2] = r.ctrl[idx - 1, :2]
        r.pos[idx : idx + 18, :2] += np.linspace(0, 0.045 * sev, 18)[:, None] * np.array([1.0, 0.4])
    if split in {"noncausal_teleport", "combined_violation_shift", "timestamp_noncausal_corruption"}:
        scale = sev * (0.55 if split == "timestamp_noncausal_corruption" else 1.0)
        jump = np.array([0.070 * scale, rng.uniform(-0.040, 0.040) * scale, 0.0])
        r.pos[idx:, :] += jump
        r.vel[idx - 1 : idx + 2, :2] = 0.0
        r.contact_force[idx - 2 : idx + 3] *= 0.0
        r.timestamp_skew = 0.0 if split != "timestamp_noncausal_corruption" else float(0.010 * sev)
    if split == "adversarial_compensated_violation":
        jump = np.array([0.045 * sev, -0.010 * sev, 0.0])
        r.pos[idx:, :] += jump
        r.vel[idx : idx + 12, :2] += np.array([0.6 * sev, -0.2 * sev])
        r.work[idx : idx + 12] += 0.20 * sev
        r.contact_force[idx : idx + 12] += 60.0 * sev
        r.support[idx : idx + 12] = 1.0
    return r


def rollout_to_features(r: Rollout, noise_level: float = 0.0) -> dict:
    pos = r.pos.copy()
    vel = r.vel.copy()
    if noise_level > 0:
        rng = np.random.default_rng(BASE_SEED + int(noise_level * 10000) + stable_int(r.split))
        pos += rng.normal(0.0, noise_level, size=pos.shape)
        vel += rng.normal(0.0, noise_level * 8.0, size=vel.shape)
    dpos = np.diff(pos, axis=0)
    speed = np.linalg.norm(dpos[:, :2], axis=1) / DT
    accel = np.diff(vel, axis=0) / DT
    accel_norm = np.linalg.norm(accel[:, :2], axis=1)
    kinetic = 0.5 * r.mass * np.linalg.norm(vel, axis=1) ** 2 + r.mass * 9.81 * pos[:, 2]
    energy_gain = np.maximum(0.0, np.diff(kinetic))
    work = r.work[1:]
    force_mid = r.contact_force[1:]
    contact_without_accel = np.max(force_mid / 180.0 - accel_norm / 45.0) if len(accel_norm) else 0.0
    no_contact_motion = np.max(speed * (r.contact_force[:-1] < 1e-4) * (pos[:-1, 2] < OBJECT_HALF + 0.025)) if len(speed) else 0.0
    energy_mismatch = np.max(energy_gain - 0.035 * (work + 1e-6)) if len(energy_gain) else 0.0
    support_violation = float(np.mean((pos[:, 2] > OBJECT_HALF + 0.050) & (r.contact_force < 1e-4) & (r.support < 0.5)))
    tangential_motion = np.linalg.norm(dpos[:, :2], axis=1)
    normal_proxy = np.maximum(r.contact_force[:-1], 1e-6)
    friction_slip = float(np.max(tangential_motion / (DT * (0.10 + r.friction) * (normal_proxy / 220.0 + 1e-3)))) if len(tangential_motion) else 0.0
    sat_motion = float(np.max(r.actuator_sat[:-1] * speed)) if len(speed) else 0.0
    jump_score = float(np.max(np.linalg.norm(dpos, axis=1))) if len(dpos) else 0.0
    impact_consistency = 0.0
    if len(accel_norm):
        expected_accel = np.maximum(force_mid[: len(accel_norm)] / max(r.mass, 1e-4), 1e-6)
        observed_accel = accel_norm + 1e-6
        impact_consistency = float(np.median(np.minimum(observed_accel / expected_accel, expected_accel / observed_accel)))
    rare_valid_contact = float(np.max(r.contact_force) / (1.0 + np.max(accel_norm) if len(accel_norm) else 1.0))
    features = {
        "max_pose_jump": jump_score,
        "max_accel": float(np.max(accel_norm)) if len(accel_norm) else 0.0,
        "contact_without_accel": float(max(0.0, contact_without_accel)),
        "motion_without_contact": float(no_contact_motion),
        "energy_work_mismatch": float(max(0.0, energy_mismatch)),
        "support_violation": float(support_violation),
        "penetration_depth": float(np.max(r.penetration)),
        "friction_slip_inconsistency": float(friction_slip),
        "actuator_saturation_score": float(sat_motion),
        "causality_jump_score": float(jump_score / 0.035),
        "max_contact_force": float(np.max(r.contact_force)),
        "mean_contact_force": float(np.mean(r.contact_force)),
        "path_length": float(np.sum(np.linalg.norm(dpos[:, :2], axis=1))) if len(dpos) else 0.0,
        "work_proxy": float(np.sum(r.work)),
        "kinetic_energy_gain": float(np.sum(energy_gain)),
        "z_range": float(np.max(pos[:, 2]) - np.min(pos[:, 2])),
        "timestamp_skew_score": float(r.timestamp_skew),
        "impact_consistency_score": impact_consistency,
        "rare_valid_contact_score": rare_valid_contact,
        "noise_level": float(noise_level),
    }
    return features


def make_rollout_row(split: str, seed: int, episode: int, severity: float = 1.0, noise_level: float = 0.0) -> dict:
    base_split = split if split in VALID_SPLITS else ("rare_valid_bounce" if split == "adversarial_compensated_violation" else "nominal_valid")
    base = generate_valid_rollout(seed, episode, base_split, severity)
    rng = np.random.default_rng(BASE_SEED + seed * 1297 + episode * 7211 + stable_int(split))
    rollout = corrupt_rollout(base, split, severity, rng)
    features = rollout_to_features(rollout, noise_level=noise_level)
    row = {
        "split": split,
        "seed": seed,
        "episode": episode,
        "severity": f"{severity:.4f}",
        "label": rollout.label,
        "unsafe": rollout.unsafe,
        "corruption": rollout.corruption,
    }
    row.update({k: f"{features[k]:.6f}" for k in FEATURE_NAMES})
    return row


def make_rollout_row_from_tuple(task: tuple) -> dict:
    return make_rollout_row(*task)


def make_dataset(splits: list[str], episodes_per_seed: int, severity: float = 1.0, noise_level: float = 0.0) -> list[dict]:
    tasks = [(split, seed, ep, severity, noise_level) for split in splits for seed in SEEDS for ep in range(episodes_per_seed)]
    if MAX_WORKERS == 1:
        return [make_rollout_row(*task) for task in tasks]
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        return list(executor.map(make_rollout_row_from_tuple, tasks, chunksize=4))


def rows_to_matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray([[float(row[name]) for name in FEATURE_NAMES] for row in rows], dtype=np.float32)
    y = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    return X, y


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial.csv")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def feature_index() -> dict[str, int]:
    return {name: FEATURE_NAMES.index(name) for name in FEATURE_NAMES}


def explicit_physics_score(features: np.ndarray, disabled: set[str] | None = None) -> np.ndarray:
    disabled = disabled or set()
    idx = feature_index()
    terms = {
        "contact": np.maximum(features[:, idx["contact_without_accel"]] / 1.4, features[:, idx["penetration_depth"]] / 0.008),
        "support": features[:, idx["support_violation"]] / 0.08,
        "energy": features[:, idx["energy_work_mismatch"]] / 0.045,
        "friction": features[:, idx["friction_slip_inconsistency"]] / 14.0,
        "actuator": features[:, idx["actuator_saturation_score"]] / 0.40,
        "causality": np.maximum(features[:, idx["causality_jump_score"]] / 1.35, features[:, idx["motion_without_contact"]] / 0.70),
    }
    for key in disabled:
        if key in terms:
            terms[key] = np.zeros(features.shape[0])
    stacked = np.stack(list(terms.values()), axis=1)
    return 0.55 * np.max(stacked, axis=1) + 0.45 * np.mean(stacked, axis=1)


def explicit_physics_score_v5(features: np.ndarray, disabled: set[str] | None = None) -> np.ndarray:
    disabled = disabled or set()
    idx = feature_index()
    contact = np.maximum(features[:, idx["contact_without_accel"]] / 1.25, features[:, idx["penetration_depth"]] / 0.009)
    support = features[:, idx["support_violation"]] / 0.070
    energy = features[:, idx["energy_work_mismatch"]] / 0.040
    friction = features[:, idx["friction_slip_inconsistency"]] / 13.0
    actuator = features[:, idx["actuator_saturation_score"]] / 0.34
    causality = np.maximum(features[:, idx["causality_jump_score"]] / 1.20, features[:, idx["motion_without_contact"]] / 0.62)
    timestamp_guard = np.exp(-18.0 * features[:, idx["timestamp_skew_score"]])
    rare_valid_guard = np.clip(features[:, idx["impact_consistency_score"]], 0.0, 1.0)
    if "timestamp_guard" not in disabled:
        causality = causality * (0.35 + 0.65 * timestamp_guard)
    if "rare_valid_guard" not in disabled:
        contact = contact * (1.0 - 0.42 * rare_valid_guard)
        friction = friction * (1.0 - 0.25 * rare_valid_guard)
    terms = {
        "contact": contact,
        "support": support,
        "energy": energy,
        "friction": friction,
        "actuator": actuator,
        "causality": causality,
    }
    for key in disabled:
        if key in terms:
            terms[key] = np.zeros(features.shape[0])
    stacked = np.stack(list(terms.values()), axis=1)
    if "conformal_aggregation" in disabled:
        return 0.55 * np.max(stacked, axis=1) + 0.45 * np.mean(stacked, axis=1)
    top_two = np.sort(stacked, axis=1)[:, -2:]
    return 0.48 * top_two[:, 1] + 0.30 * top_two[:, 0] + 0.22 * np.mean(stacked, axis=1)


def residual_score_matrix(X: np.ndarray) -> np.ndarray:
    idx = feature_index()
    return np.stack(
        [
            X[:, idx["max_pose_jump"]] / 0.035 + X[:, idx["max_accel"]] / 70.0,
            X[:, idx["energy_work_mismatch"]] / 0.045 + 0.15 * X[:, idx["kinetic_energy_gain"]],
            X[:, idx["contact_without_accel"]] / 1.3 + X[:, idx["max_contact_force"]] / 650.0,
            X[:, idx["support_violation"]] / 0.08,
            X[:, idx["causality_jump_score"]] / 1.35,
            X[:, idx["friction_slip_inconsistency"]] / 14.0,
        ],
        axis=1,
    )


def raw_base_scores(X: np.ndarray, models: dict | None = None) -> dict[str, np.ndarray]:
    idx = feature_index()
    residuals = residual_score_matrix(X)
    scores = {
        "random_flagger": np.mod(np.sin(X[:, idx["path_length"]] * 123.0 + X[:, idx["work_proxy"]] * 17.0), 1.0),
        "kinematic_residual_threshold": residuals[:, 0],
        "energy_residual_threshold": residuals[:, 1],
        "contact_impulse_threshold": residuals[:, 2],
        "max_residual_detector": np.max(residuals, axis=1),
        "physics_violation_audit": explicit_physics_score(X),
        PROPOSED_METHOD: explicit_physics_score_v5(X),
        "oracle_violation_labels": np.zeros(X.shape[0]),
    }
    if models and "scaler" in models:
        scaler = models["scaler"]
        Xs = scaler.transform(X)
        probs = np.stack([m.predict_proba(Xs)[:, 1] for m in models["rf_ensemble"]], axis=0)
        scores["random_forest_ensemble"] = probs.mean(axis=0) + 0.35 * probs.std(axis=0)
        scores["hgb_failure_classifier"] = models["hgb"].predict_proba(Xs)[:, 1]
        scores["logistic_residual_stack"] = models["logistic"].predict_proba(residuals)[:, 1]
        scores["isolation_forest_detector"] = -models["isolation"].score_samples(Xs)
        recon = models["pca"].inverse_transform(models["pca"].transform(Xs))
        scores["pca_reconstruction_audit"] = np.mean((Xs - recon) ** 2, axis=1)
        valid_q = models["residual_valid_q"]
        scores["conformal_residual_ensemble"] = np.max(residuals / (valid_q + 1e-6), axis=1)
        if "stacker" in models and "stack_scaler" in models:
            stack_features = np.stack(
                [
                    scores["kinematic_residual_threshold"],
                    scores["energy_residual_threshold"],
                    scores["contact_impulse_threshold"],
                    scores["max_residual_detector"],
                    scores["physics_violation_audit"],
                    scores[PROPOSED_METHOD],
                    scores["hgb_failure_classifier"],
                    scores["random_forest_ensemble"],
                    scores["pca_reconstruction_audit"],
                    scores["isolation_forest_detector"],
                ],
                axis=1,
            )
            scores["calibrated_learned_stack"] = models["stacker"].predict_proba(models["stack_scaler"].transform(stack_features))[:, 1]
        else:
            scores["calibrated_learned_stack"] = scores["hgb_failure_classifier"]
    else:
        for method in [
            "random_forest_ensemble",
            "hgb_failure_classifier",
            "logistic_residual_stack",
            "isolation_forest_detector",
            "pca_reconstruction_audit",
            "conformal_residual_ensemble",
            "calibrated_learned_stack",
        ]:
            scores[method] = scores["max_residual_detector"]
    return scores


def train_baselines(train_scenes: int) -> dict:
    train_rows = []
    train_splits = VALID_SPLITS + [
        "contact_corruption",
        "energy_work_corruption",
        "support_levitation",
        "actuator_saturation",
        "noncausal_teleport",
        "subtle_contact_corruption",
        "subtle_energy_corruption",
        "timestamp_noncausal_corruption",
        "adversarial_compensated_violation",
        "mixed_near_threshold",
    ]
    for idx in range(train_scenes):
        split = train_splits[idx % len(train_splits)]
        sev = 0.40 + 0.80 * ((idx % 9) / 8)
        noise = 0.0005 * (idx % 5)
        train_rows.append(make_rollout_row(split, idx % max(1, len(SEEDS)), idx, sev, noise_level=noise))
    write_csv(RESULTS / "training_audit_rollouts.csv", train_rows)
    X, y = rows_to_matrix(train_rows)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    valid = Xs[y == 0]
    pca = PCA(n_components=min(8, valid.shape[1], valid.shape[0]))
    pca.fit(valid)
    hgb = HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=15, learning_rate=0.06, random_state=12)
    hgb.fit(Xs, y)
    rf_ensemble = []
    rng = np.random.default_rng(BASE_SEED + 88)
    for idx in range(5):
        boot = rng.integers(0, len(y), size=len(y))
        model = RandomForestClassifier(n_estimators=96, max_depth=8, min_samples_leaf=3, class_weight="balanced", random_state=31 + idx)
        model.fit(Xs[boot], y[boot])
        rf_ensemble.append(model)
    residuals = residual_score_matrix(X)
    logistic = LogisticRegression(max_iter=500, class_weight="balanced", random_state=44)
    logistic.fit(residuals, y)
    isolation = IsolationForest(n_estimators=120, contamination=0.08, random_state=45)
    isolation.fit(valid)
    residual_valid_q = np.quantile(residuals[y == 0], 0.95, axis=0) if np.any(y == 0) else np.ones(residuals.shape[1])

    partial = {
        "scaler": scaler,
        "pca": pca,
        "hgb": hgb,
        "rf_ensemble": rf_ensemble,
        "logistic": logistic,
        "isolation": isolation,
        "residual_valid_q": residual_valid_q,
    }
    prelim = raw_base_scores(X, partial)
    stack_features = np.stack(
        [
            prelim["kinematic_residual_threshold"],
            prelim["energy_residual_threshold"],
            prelim["contact_impulse_threshold"],
            prelim["max_residual_detector"],
            prelim["physics_violation_audit"],
            prelim[PROPOSED_METHOD],
            prelim["hgb_failure_classifier"],
            prelim["random_forest_ensemble"],
            prelim["pca_reconstruction_audit"],
            prelim["isolation_forest_detector"],
        ],
        axis=1,
    )
    stack_scaler = StandardScaler()
    stacker = LogisticRegression(max_iter=500, class_weight="balanced", random_state=46)
    stacker.fit(stack_scaler.fit_transform(stack_features), y)
    models = dict(partial)
    models["stacker"] = stacker
    models["stack_scaler"] = stack_scaler
    train_scores = raw_base_scores(X, models)
    thresholds = {}
    for method in METHODS:
        if method == "oracle_violation_labels":
            thresholds[method] = 0.50
        elif method == "random_flagger":
            thresholds[method] = 0.50
        else:
            valid_scores = np.asarray(train_scores[method])[y == 0]
            thresholds[method] = float(np.quantile(valid_scores, 0.95)) if len(valid_scores) else 0.5
    models["thresholds"] = thresholds
    write_csv(
        RESULTS / "training_summary.csv",
        [
            {
                "training_rows": len(train_rows),
                "positive_rate": f"{float(y.mean()):.4f}",
                "hgb_train_accuracy": f"{float(hgb.score(Xs, y)):.4f}",
                "pca_components": pca.n_components_,
                "threshold_v4_physics": f"{thresholds['physics_violation_audit']:.4f}",
                "threshold_v5_physics": f"{thresholds[PROPOSED_METHOD]:.4f}",
                "valid_calibration_quantile": "0.95",
            }
        ],
    )
    return models


def score_ablation(X: np.ndarray, ablation: str) -> np.ndarray:
    disabled = {
        "no_contact_check": {"contact"},
        "no_support_check": {"support"},
        "no_energy_check": {"energy"},
        "no_friction_slip_check": {"friction"},
        "no_actuator_check": {"actuator"},
        "no_causality_check": {"causality"},
        "no_timestamp_guard": {"timestamp_guard"},
        "no_rare_valid_guard": {"rare_valid_guard"},
        "no_conformal_aggregation": {"conformal_aggregation"},
        "full_physics_violation_audit_v5": set(),
    }.get(ablation, set())
    if ablation == "old_v4_physics_audit":
        return explicit_physics_score(X)
    if ablation == "scalar_residual_only":
        return np.max(residual_score_matrix(X)[:, :3], axis=1)
    return explicit_physics_score_v5(X, disabled=disabled)


def evaluate_rows(dataset_rows: list[dict], models: dict, methods: list[str], ablation: bool = False) -> list[dict]:
    X, y = rows_to_matrix(dataset_rows)
    thresholds = models["thresholds"]
    method_scores = raw_base_scores(X, models)
    rows = []
    for method in methods:
        if ablation:
            scores = score_ablation(X, method)
            valid_scores = scores[y == 0]
            threshold = float(np.quantile(valid_scores, 0.95)) if len(valid_scores) else thresholds[PROPOSED_METHOD]
        else:
            scores = method_scores[method]
            if method == "oracle_violation_labels":
                scores = y.astype(float)
            threshold = thresholds.get(method, 0.5)
        flags = (scores >= threshold).astype(int)
        for row, label, score, flag in zip(dataset_rows, y, scores, flags):
            out = {
                "method": method,
                "split": row["split"],
                "seed": row["seed"],
                "episode": row["episode"],
                "label": int(label),
                "unsafe": int(row["unsafe"]),
                "score": f"{float(score):.6f}",
                "threshold": f"{float(threshold):.6f}",
                "flag": int(flag),
                "true_positive": int(flag and label),
                "false_positive": int(flag and not label),
                "false_negative": int((not flag) and label),
                "true_negative": int((not flag) and not label),
                "severity": row["severity"],
            }
            rows.append(out)
    return rows


def summarize(rows: list[dict], group_keys: list[str]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        grouped.setdefault(key, []).append(row)
    out_rows = []
    for key, group in sorted(grouped.items()):
        tp = sum(int(r["true_positive"]) for r in group)
        fp = sum(int(r["false_positive"]) for r in group)
        fn = sum(int(r["false_negative"]) for r in group)
        tn = sum(int(r["true_negative"]) for r in group)
        labels = np.asarray([int(r["label"]) for r in group], dtype=int)
        flags = np.asarray([int(r["flag"]) for r in group], dtype=int)
        scores = np.asarray([float(r["score"]) for r in group], dtype=float)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        accuracy = (tp + tn) / max(1, len(group))
        fpr = fp / max(1, fp + tn)
        auroc = 0.0
        auprc = float(labels.mean()) if len(labels) else 0.0
        if len(np.unique(labels)) == 2:
            auroc = float(roc_auc_score(labels, scores))
            auprc = float(average_precision_score(labels, scores))
        out = {k: v for k, v in zip(group_keys, key)}
        out.update(
            {
                "precision": f"{precision:.4f}",
                "recall": f"{recall:.4f}",
                "f1": f"{f1:.4f}",
                "accuracy": f"{accuracy:.4f}",
                "false_positive_rate": f"{fpr:.4f}",
                "flag_rate": f"{float(np.mean(flags)):.4f}",
                "positive_rate": f"{float(np.mean(labels)):.4f}",
                "auroc": f"{auroc:.4f}",
                "auprc": f"{auprc:.4f}",
                "score_mean": f"{float(np.mean(scores)):.4f}",
                "episodes": len(group),
                "seeds": len({r["seed"] for r in group}),
            }
        )
        out_rows.append(out)
    return out_rows


def aggregate_metrics(rows: list[dict]) -> list[dict]:
    groups = {
        "all_main": set(MAIN_SPLITS),
        "valid_regimes": set(VALID_SPLITS),
        "hard_corruptions": set(CORRUPTION_SPLITS),
        "combined_and_adversarial": {"combined_violation_shift", "adversarial_compensated_violation", "mixed_near_threshold"},
    }
    out = []
    for group_name, split_set in groups.items():
        subset = [dict(r, aggregate_group=group_name) for r in rows if r["split"] in split_set]
        out.extend(summarize(subset, ["aggregate_group", "method"]))
    return out


def seed_metrics(rows: list[dict]) -> list[dict]:
    return summarize(rows, ["method", "split", "seed"])


def pairwise_stats(seed_rows: list[dict]) -> list[dict]:
    metric_map = {(r["method"], r["split"], r["seed"]): float(r["f1"]) for r in seed_rows}
    rows = []
    for split in MAIN_SPLITS:
        for method in METHODS:
            if method == PROPOSED_METHOD:
                continue
            diffs = []
            for seed in SEEDS:
                p_key = (PROPOSED_METHOD, split, seed)
                b_key = (method, split, seed)
                if p_key in metric_map and b_key in metric_map:
                    diffs.append(metric_map[p_key] - metric_map[b_key])
            if not diffs:
                continue
            mean_diff = float(np.mean(diffs))
            sd = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
            if sd < 1e-12:
                t_stat = 0.0 if abs(mean_diff) < 1e-12 else math.copysign(1_000_000.0, mean_diff)
            else:
                t_stat = mean_diff / (sd / math.sqrt(len(diffs)))
            rows.append(
                {
                    "split": split,
                    "baseline": method,
                    "mean_f1_diff_vs_v5": f"{mean_diff:.4f}",
                    "paired_t_approx": f"{t_stat:.4f}",
                    "normal_approx_p": f"{normal_p_from_t(t_stat):.4f}",
                    "seeds": len(diffs),
                }
            )
    return rows


def fixed_fpr_metrics(rows: list[dict]) -> list[dict]:
    out = []
    budgets = [0.01, 0.05, 0.10]
    for method in METHODS:
        method_rows = [r for r in rows if r["method"] == method]
        valid_scores = np.asarray([float(r["score"]) for r in method_rows if r["split"] in VALID_SPLITS], dtype=float)
        hard_rows = [r for r in method_rows if r["split"] in CORRUPTION_SPLITS]
        hard_scores = np.asarray([float(r["score"]) for r in hard_rows], dtype=float)
        hard_labels = np.asarray([int(r["label"]) for r in hard_rows], dtype=int)
        for budget in budgets:
            threshold = float(np.quantile(valid_scores, 1.0 - budget)) if len(valid_scores) else 0.5
            recall = float(np.mean(hard_scores >= threshold)) if len(hard_scores) else 0.0
            if hard_labels.size and hard_labels.mean() < 1.0:
                recall = float(np.sum((hard_scores >= threshold) & (hard_labels == 1)) / max(1, np.sum(hard_labels == 1)))
            out.append(
                {
                    "method": method,
                    "fpr_budget": f"{budget:.2f}",
                    "threshold": f"{threshold:.6f}",
                    "hard_recall_at_budget": f"{recall:.4f}",
                    "valid_scores": len(valid_scores),
                    "hard_scores": len(hard_scores),
                }
            )
    return out


def plot_metric(metrics: list[dict], path: Path, metric: str, title: str, ylabel: str) -> None:
    selected = [
        "kinematic_residual_threshold",
        "energy_residual_threshold",
        "max_residual_detector",
        "hgb_failure_classifier",
        "random_forest_ensemble",
        "calibrated_learned_stack",
        "physics_violation_audit",
        PROPOSED_METHOD,
        "oracle_violation_labels",
    ]
    x = np.arange(len(MAIN_SPLITS))
    width = 0.085
    fig, ax = plt.subplots(figsize=(14, 5.5))
    for idx, method in enumerate(selected):
        vals = []
        for split in MAIN_SPLITS:
            match = [r for r in metrics if r["method"] == method and r["split"] == split]
            vals.append(float(match[0][metric]) if match else 0.0)
        ax.bar(x + (idx - len(selected) / 2) * width + width / 2, vals, width, label=method.replace("_", " "))
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in MAIN_SPLITS], fontsize=7)
    ax.legend(fontsize=6, ncol=3)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_ablation(metrics: list[dict], path: Path) -> None:
    vals = [(r["method"], float(r["f1"]), float(r["false_positive_rate"])) for r in metrics if r["split"] == "combined_violation_shift"]
    vals.sort(key=lambda item: item[1], reverse=True)
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    x = np.arange(len(vals))
    ax.bar(x, [v[1] for v in vals], color="#59704d")
    ax.plot(x, [v[2] for v in vals], marker="o", color="#9a3d2f", label="false positive rate")
    ax.set_xticks(x)
    ax.set_xticklabels([v[0].replace("_", "\n") for v in vals], fontsize=7)
    ax.set_ylabel("Combined-shift F1")
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("Physics-audit v5 ablations")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_stress(stress_metrics: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for method in STRESS_METHODS:
        xs, ys = [], []
        for row in stress_metrics:
            if row["method"] == method and row["split"].startswith("stress_") and "combined_violation_shift" in row["split"]:
                xs.append(float(row["stress_level"]))
                ys.append(float(row["f1"]))
        if not xs:
            continue
        order = np.argsort(xs)
        ax.plot(np.asarray(xs)[order], np.asarray(ys)[order], marker="o", label=method.replace("_", " "))
    ax.set_xlabel("Noise/corruption severity")
    ax.set_ylabel("F1")
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=7)
    ax.set_title("Stress sweep: combined violation shift")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_negative_cases() -> list[dict]:
    return [
        {
            "case": "sensor_timestamp_skew",
            "observed_failure_mode": "causality checks can flag asynchronous but physically valid traces",
            "submission_implication": "needs timestamp calibration before deployment claims",
        },
        {
            "case": "legal_rare_bouncing_contact",
            "observed_failure_mode": "high contact impulse can be physically valid and can trigger contact residuals",
            "submission_implication": "requires impact-aware thresholds or learned contact regimes",
        },
        {
            "case": "out_of_distribution_deformable_object",
            "observed_failure_mode": "rigid-body audit assumes object state and contact geometry that may be invalid",
            "submission_implication": "scope is rigid-body MuJoCo manipulation",
        },
        {
            "case": "semantic_task_violation_with_valid_physics",
            "observed_failure_mode": "all physics checks pass while the robot performs the wrong task",
            "submission_implication": "audit is not a complete safety monitor",
        },
        {
            "case": "adversarial_compensated_violation",
            "observed_failure_mode": "corruption can be partially hidden by plausible work and contact proxies",
            "submission_implication": "scalar residuals and explicit checks both need stronger temporal modeling",
        },
    ]


def decision_from_outputs(metrics: list[dict], aggregates: list[dict], ablation_metrics: list[dict], stress_output: list[dict], fixed_rows: list[dict]) -> tuple[str, list[str]]:
    decision = "STRONG_REVISE"
    reasons = []
    agg = {r["method"]: r for r in aggregates if r["aggregate_group"] == "hard_corruptions"}
    combined = {r["method"]: r for r in aggregates if r["aggregate_group"] == "combined_and_adversarial"}
    valid = {r["method"]: r for r in aggregates if r["aggregate_group"] == "valid_regimes"}
    proposed_hard = agg[PROPOSED_METHOD]
    proposed_combined = combined[PROPOSED_METHOD]
    best_hard = max((r for m, r in agg.items() if m not in {PROPOSED_METHOD, "oracle_violation_labels"}), key=lambda r: float(r["f1"]))
    best_combined = max((r for m, r in combined.items() if m not in {PROPOSED_METHOD, "oracle_violation_labels"}), key=lambda r: float(r["f1"]))
    if float(proposed_hard["f1"]) <= float(best_hard["f1"]) + 0.030:
        decision = "KILL_ARCHIVE"
        reasons.append(f"v5 does not beat strongest non-oracle hard-corruption baseline {best_hard['method']} by 0.030")
    if float(proposed_combined["f1"]) <= float(best_combined["f1"]) + 0.030:
        decision = "KILL_ARCHIVE"
        reasons.append(f"v5 does not beat strongest combined/adversarial baseline {best_combined['method']} by 0.030")
    max_valid_fpr = max(
        float(r["false_positive_rate"])
        for r in metrics
        if r["method"] == PROPOSED_METHOD and r["split"] in VALID_SPLITS
    )
    if max_valid_fpr > 0.100:
        decision = "KILL_ARCHIVE"
        reasons.append(f"v5 false-positive rate on at least one rare-valid split is {max_valid_fpr:.3f} > 0.100")
    fixed_005 = {r["method"]: r for r in fixed_rows if r["fpr_budget"] == "0.05"}
    proposed_fixed = float(fixed_005[PROPOSED_METHOD]["hard_recall_at_budget"])
    best_fixed = max((r for m, r in fixed_005.items() if m not in {PROPOSED_METHOD, "oracle_violation_labels"}), key=lambda r: float(r["hard_recall_at_budget"]))
    if proposed_fixed <= float(best_fixed["hard_recall_at_budget"]) - 0.030:
        decision = "KILL_ARCHIVE"
        reasons.append(f"v5 recall at 5% FPR trails {best_fixed['method']} by more than 0.030")
    full_rows = [r for r in ablation_metrics if r["method"] == "full_physics_violation_audit_v5"]
    full_by_split = {r["split"]: float(r["f1"]) for r in full_rows}
    bad_ablation = []
    for row in ablation_metrics:
        if row["method"] == "full_physics_violation_audit_v5":
            continue
        if row["split"] in VALID_SPLITS:
            continue
        full_f1 = full_by_split.get(row["split"])
        if full_f1 is not None and float(row["f1"]) >= full_f1 - 0.020:
            bad_ablation.append(row["method"])
    if bad_ablation:
        decision = "KILL_ARCHIVE"
        reasons.append("ablation gate fails because " + ", ".join(sorted(set(bad_ablation))) + " matches or beats full v5")
    stress_level_rows = [r for r in stress_output if r.get("stress_level") == "1.00" and "combined_violation_shift" in r["split"]]
    if stress_level_rows:
        proposed_stress = next(r for r in stress_level_rows if r["method"] == PROPOSED_METHOD)
        best_stress = max((r for r in stress_level_rows if r["method"] != PROPOSED_METHOD), key=lambda r: float(r["f1"]))
        if float(proposed_stress["f1"]) < float(best_stress["f1"]):
            decision = "KILL_ARCHIVE"
            reasons.append(f"maximum-stress gate fails against {best_stress['method']}")
    if decision == "STRONG_REVISE":
        reasons.append("all local gates passed, but real robot or public benchmark validation is still required")
    return decision, reasons


def write_summary(
    path: Path,
    args: argparse.Namespace,
    rollout_count: int,
    main_eval: list[dict],
    ablation_eval: list[dict],
    stress_eval: list[dict],
    metrics: list[dict],
    aggregates: list[dict],
    ablation_metrics: list[dict],
    pairwise: list[dict],
    stress_output: list[dict],
    fixed_rows: list[dict],
) -> None:
    decision, reasons = decision_from_outputs(metrics, aggregates, ablation_metrics, stress_output, fixed_rows)
    combined = {r["method"]: r for r in aggregates if r["aggregate_group"] == "combined_and_adversarial"}
    valid = {r["method"]: r for r in aggregates if r["aggregate_group"] == "valid_regimes"}
    with path.open("w", encoding="utf-8") as handle:
        handle.write("Paper 69 expanded MuJoCo robotic physics violation audits rebuild\n")
        handle.write(f"Seeds: {SEEDS}; episodes per seed: {args.episodes}; workers: {MAX_WORKERS}\n")
        handle.write(f"Training scenes: {args.train_scenes}\n")
        handle.write(
            "Main raw rollouts: %d; main eval rows: %d; ablation rows: %d; stress rows: %d\n"
            % (rollout_count, len(main_eval), len(ablation_eval), len(stress_eval))
        )
        handle.write(f"Terminal decision: {decision}\n")
        handle.write("Terminal reason: " + "; ".join(reasons) + "\n")
        handle.write("\nCombined/adversarial aggregate results:\n")
        for method in METHODS:
            row = combined[method]
            handle.write(
                f"- {method}: f1={row['f1']} precision={row['precision']} recall={row['recall']} "
                f"fpr={row['false_positive_rate']} auroc={row['auroc']} auprc={row['auprc']}\n"
            )
        handle.write("\nValid-regime aggregate false positives:\n")
        for method in METHODS:
            row = valid[method]
            handle.write(f"- {method}: fpr={row['false_positive_rate']} flag={row['flag_rate']} score={row['score_mean']}\n")
        handle.write("\nAblations:\n")
        for row in ablation_metrics:
            if row["split"] == "combined_violation_shift":
                handle.write(
                    f"- {row['method']}: f1={row['f1']} precision={row['precision']} recall={row['recall']} fpr={row['false_positive_rate']}\n"
                )
        handle.write("\nRecall at 5 percent FPR:\n")
        for row in fixed_rows:
            if row["fpr_budget"] == "0.05":
                handle.write(f"- {row['method']}: recall={row['hard_recall_at_budget']} threshold={row['threshold']}\n")
        handle.write("\nPairwise comparisons vs physics_violation_audit_v5:\n")
        for row in pairwise:
            if row["split"] in {"combined_violation_shift", "adversarial_compensated_violation", "mixed_near_threshold"}:
                handle.write(
                    f"- {row['split']} / {row['baseline']}: diff={row['mean_f1_diff_vs_v5']} "
                    f"t={row['paired_t_approx']} p={row['normal_approx_p']}\n"
                )


def main() -> None:
    global RESULTS, FIGURES, SEEDS, MAX_WORKERS
    args = parse_args()
    RESULTS = args.results_dir
    FIGURES = args.figures_dir
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    SEEDS = list(range(args.seeds))
    MAX_WORKERS = max(1, int(args.workers))

    models = train_baselines(args.train_scenes)

    main_rows_raw = make_dataset(args.splits, args.episodes, severity=1.0, noise_level=0.001)
    write_csv(RESULTS / "physics_audit_rollouts.csv", main_rows_raw)
    main_eval = evaluate_rows(main_rows_raw, models, METHODS)
    write_csv(RESULTS / "physics_audit_raw.csv", main_eval)
    seed_rows = seed_metrics(main_eval)
    write_csv(RESULTS / "raw_seed_metrics.csv", seed_rows)
    metrics = summarize(main_eval, ["method", "split"])
    write_csv(RESULTS / "physics_audit_metrics.csv", metrics)
    write_csv(RESULTS / "metrics.csv", metrics)
    aggregates = aggregate_metrics(main_eval)
    write_csv(RESULTS / "aggregate_metrics.csv", aggregates)
    pairwise = pairwise_stats(seed_rows)
    write_csv(RESULTS / "physics_audit_pairwise.csv", pairwise)
    write_csv(RESULTS / "pairwise_stats.csv", pairwise)
    fixed_rows = fixed_fpr_metrics(main_eval)
    write_csv(RESULTS / "fixed_fpr_metrics.csv", fixed_rows)

    ablation_rows_raw = make_dataset(args.ablation_splits, args.ablation_episodes, severity=1.0, noise_level=0.001)
    ablation_eval = evaluate_rows(ablation_rows_raw, models, ABLATIONS, ablation=True)
    write_csv(RESULTS / "physics_audit_ablation_raw.csv", ablation_eval)
    ablation_metrics = summarize(ablation_eval, ["method", "split"])
    write_csv(RESULTS / "physics_audit_ablation.csv", ablation_metrics)
    write_csv(RESULTS / "ablation_metrics.csv", ablation_metrics)
    write_csv(RESULTS / "ablation_aggregate_metrics.csv", aggregate_metrics(ablation_eval))

    stress_eval = []
    for level in args.stress_levels:
        severity = 0.35 + 0.85 * level
        noise = 0.0005 + 0.004 * level
        raw = make_dataset(args.stress_splits, args.stress_episodes, severity=severity, noise_level=noise)
        rows = evaluate_rows(raw, models, STRESS_METHODS)
        for row in rows:
            row["split"] = f"stress_{level:.2f}_{row['split']}"
        stress_eval.extend(rows)
    stress_metrics = summarize(stress_eval, ["method", "split"])
    stress_output = []
    for row in stress_metrics:
        out = dict(row)
        parts = out["split"].split("_")
        out["stress_level"] = parts[1]
        stress_output.append(out)
    write_csv(RESULTS / "stress_sweep.csv", stress_output)
    write_csv(FIGURES / "stress_curve_data.csv", stress_output)

    write_csv(RESULTS / "negative_cases.csv", make_negative_cases())
    plot_metric(metrics, FIGURES / "physics_audit_f1_by_split.png", "f1", "Physics-violation audit F1 by split", "F1")
    plot_metric(metrics, FIGURES / "physics_audit_false_positive_by_split.png", "false_positive_rate", "False positives on valid and corrupted traces", "False positive rate")
    plot_ablation(ablation_metrics, FIGURES / "physics_audit_ablation_f1.png")
    plot_stress(stress_output, FIGURES / "physics_audit_stress_sweep.png")
    write_summary(
        RESULTS / "summary.txt",
        args,
        len(main_rows_raw),
        main_eval,
        ablation_eval,
        stress_eval,
        metrics,
        aggregates,
        ablation_metrics,
        pairwise,
        stress_output,
        fixed_rows,
    )
    print(f"wrote Paper 69 MuJoCo evidence to {RESULTS}")


if __name__ == "__main__":
    main()
