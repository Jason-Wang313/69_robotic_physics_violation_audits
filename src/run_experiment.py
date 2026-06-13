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
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


BASE_SEED = 2129968627
SEEDS = [0, 1, 2, 3, 4]
EPISODES_PER_SEED = 12
ABLATION_EPISODES_PER_SEED = 12
STRESS_EPISODES_PER_SEED = 8
TRAIN_VALID = 90
MAX_WORKERS = max(1, min(4, int(os.environ.get("PAPER69_WORKERS", "4"))))

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

OBJECT_HALF = 0.04
FINGER_RADIUS = 0.015
DT = 0.01
CONTACT_LIMIT = 520.0

METHODS = [
    "random_flagger",
    "kinematic_residual_threshold",
    "energy_residual_threshold",
    "contact_impulse_threshold",
    "ensemble_dynamics_uncertainty",
    "autoencoder_reconstruction_audit",
    "supervised_failure_classifier",
    "physics_violation_audit",
    "oracle_violation_labels",
]

ABLATIONS = [
    "full_physics_violation_audit",
    "no_contact_check",
    "no_support_check",
    "no_energy_check",
    "no_friction_slip_check",
    "no_actuator_check",
    "no_causality_check",
    "scalar_residual_only",
]

MAIN_SPLITS = [
    "nominal_valid",
    "contact_corruption",
    "energy_work_corruption",
    "support_levitation",
    "actuator_saturation",
    "noncausal_teleport",
    "combined_violation_shift",
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


MODEL_CACHE: dict[tuple[float, float], mujoco.MjModel] = {}


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
    friction = float(clamp(rng.uniform(0.35, 1.05) - 0.18 * severity, 0.18, 1.20))
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
    end = obj0 + direction * rng.uniform(0.18, 0.30)
    if split == "actuator_saturation":
        end = obj0 + direction * rng.uniform(0.30, 0.42)

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
        if split == "actuator_saturation":
            desired = desired + direction * (0.04 * math.sin(t * 0.3))
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
        label=0 if split == "nominal_valid" else 1,
        unsafe=0 if split == "nominal_valid" else 1,
        corruption=split,
    )
    if split == "nominal_valid":
        noise = rng.normal(0.0, 0.0015 * (1 + severity), size=r.pos.shape)
        r.pos += noise
        return r

    idx = int(rng.integers(18, len(r.pos) - 18))
    sev = severity
    if split in {"contact_corruption", "combined_violation_shift"}:
        r.contact_force[idx : idx + 8] += 450.0 * sev
        r.vel[idx : idx + 8, :2] *= 0.20
        r.penetration[idx : idx + 8] += 0.010 * sev
    if split in {"energy_work_corruption", "combined_violation_shift"}:
        jump = np.array([0.030 * sev, -0.018 * sev, 0.0])
        r.vel[idx : idx + 14, :2] += np.array([1.6 * sev, -0.8 * sev])
        r.pos[idx:, :] += jump
        r.work[idx : idx + 14] *= 0.08
    if split in {"support_levitation", "combined_violation_shift"}:
        r.pos[idx : idx + 16, 2] += 0.085 * sev
        r.vel[idx : idx + 16, 2] = 0.0
        r.contact_force[idx : idx + 16] *= 0.0
        r.support[idx : idx + 16] = 0.0
    if split in {"actuator_saturation", "combined_violation_shift"}:
        r.actuator_sat[idx : idx + 18] = 1.0
        r.ctrl[idx : idx + 18, :2] = r.ctrl[idx - 1, :2]
        r.pos[idx : idx + 18, :2] += np.linspace(0, 0.045 * sev, 18)[:, None] * np.array([1.0, 0.4])
    if split in {"noncausal_teleport", "combined_violation_shift"}:
        jump = np.array([0.070 * sev, rng.uniform(-0.040, 0.040) * sev, 0.0])
        r.pos[idx:, :] += jump
        r.vel[idx - 1 : idx + 2, :2] = 0.0
        r.contact_force[idx - 2 : idx + 3] *= 0.0
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
        "noise_level": float(noise_level),
    }
    return features


def make_rollout_row(split: str, seed: int, episode: int, severity: float = 1.0, noise_level: float = 0.0) -> dict:
    base = generate_valid_rollout(seed, episode, split, severity)
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


def run_tasks(tasks: list[tuple]) -> list[dict]:
    if MAX_WORKERS == 1:
        return [make_rollout_row(*task) for task in tasks]
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        return list(executor.map(lambda args: make_rollout_row(*args), tasks, chunksize=4))


def make_dataset(splits: list[str], episodes_per_seed: int, severity: float = 1.0, noise_level: float = 0.0) -> list[dict]:
    tasks = [(split, seed, ep, severity, noise_level) for split in splits for seed in SEEDS for ep in range(episodes_per_seed)]
    if MAX_WORKERS == 1:
        return [make_rollout_row(*task) for task in tasks]
    # Avoid lambda pickling issues on Windows.
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        return list(executor.map(make_rollout_row_from_tuple, tasks, chunksize=4))


def make_rollout_row_from_tuple(task: tuple) -> dict:
    return make_rollout_row(*task)


def rows_to_matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray([[float(row[name]) for name in FEATURE_NAMES] for row in rows], dtype=np.float32)
    y = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    return X, y


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    tmp = path.with_suffix(".partial.csv")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def train_baselines() -> dict:
    train_rows = []
    train_splits = ["nominal_valid", "contact_corruption", "energy_work_corruption", "support_levitation", "actuator_saturation", "noncausal_teleport"]
    for idx in range(TRAIN_VALID):
        split = train_splits[idx % len(train_splits)]
        sev = 0.55 + 0.65 * ((idx % 7) / 6)
        train_rows.append(make_rollout_row(split, idx % 5, idx, sev, noise_level=0.0008 * (idx % 3)))
    write_csv(RESULTS / "training_audit_rollouts.csv", train_rows)
    X, y = rows_to_matrix(train_rows)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    valid = Xs[y == 0]
    pca = PCA(n_components=min(6, valid.shape[1], valid.shape[0]))
    pca.fit(valid)
    clf = HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15, learning_rate=0.07, random_state=12)
    clf.fit(Xs, y)
    ensemble = []
    rng = np.random.default_rng(BASE_SEED + 88)
    for idx in range(3):
        boot = rng.integers(0, len(y), size=len(y))
        model = RandomForestClassifier(n_estimators=80, max_depth=7, min_samples_leaf=3, class_weight="balanced", random_state=31 + idx)
        model.fit(Xs[boot], y[boot])
        ensemble.append(model)
    scalar = LogisticRegression(max_iter=400, class_weight="balanced", random_state=44)
    scalar_cols = [FEATURE_NAMES.index("max_pose_jump"), FEATURE_NAMES.index("energy_work_mismatch"), FEATURE_NAMES.index("max_contact_force")]
    scalar.fit(Xs[:, scalar_cols], y)
    method_train_scores = compute_method_scores(X, models={"scaler": scaler, "pca": pca, "classifier": clf, "ensemble": ensemble, "scalar": scalar, "scalar_cols": scalar_cols})
    thresholds = {}
    for method, scores in method_train_scores.items():
        valid_scores = np.asarray(scores)[y == 0]
        if method == "supervised_failure_classifier":
            thresholds[method] = 0.50
        elif method == "oracle_violation_labels":
            thresholds[method] = 0.50
        elif method == "random_flagger":
            thresholds[method] = 0.50
        else:
            thresholds[method] = float(np.quantile(valid_scores, 0.94)) if len(valid_scores) else 0.5
    write_csv(
        RESULTS / "training_summary.csv",
        [
            {
                "training_rows": len(train_rows),
                "positive_rate": f"{float(y.mean()):.4f}",
                "classifier_train_accuracy": f"{float(clf.score(Xs, y)):.4f}",
                "pca_components": pca.n_components_,
                "threshold_physics": f"{thresholds['physics_violation_audit']:.4f}",
                "threshold_autoencoder": f"{thresholds['autoencoder_reconstruction_audit']:.4f}",
            }
        ],
    )
    return {"scaler": scaler, "pca": pca, "classifier": clf, "ensemble": ensemble, "scalar": scalar, "scalar_cols": scalar_cols, "thresholds": thresholds}


def explicit_physics_score(features: np.ndarray, disabled: set[str] | None = None) -> np.ndarray:
    disabled = disabled or set()
    idx = {name: FEATURE_NAMES.index(name) for name in FEATURE_NAMES}
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


def compute_method_scores(X: np.ndarray, models: dict) -> dict[str, np.ndarray]:
    scaler = models.get("scaler")
    Xs = scaler.transform(X) if scaler is not None else X
    idx = {name: FEATURE_NAMES.index(name) for name in FEATURE_NAMES}
    scores = {
        "random_flagger": np.mod(np.sin(X[:, idx["path_length"]] * 123.0 + X[:, idx["work_proxy"]] * 17.0), 1.0),
        "kinematic_residual_threshold": X[:, idx["max_pose_jump"]] / 0.035 + X[:, idx["max_accel"]] / 70.0,
        "energy_residual_threshold": X[:, idx["energy_work_mismatch"]] / 0.045 + 0.15 * X[:, idx["kinetic_energy_gain"]],
        "contact_impulse_threshold": X[:, idx["contact_without_accel"]] / 1.3 + X[:, idx["max_contact_force"]] / 650.0,
        "physics_violation_audit": explicit_physics_score(X),
        "oracle_violation_labels": np.zeros(X.shape[0]),
    }
    if "ensemble" in models:
        probs = np.stack([m.predict_proba(Xs)[:, 1] for m in models["ensemble"]], axis=0)
        scores["ensemble_dynamics_uncertainty"] = probs.mean(axis=0) + 0.45 * probs.std(axis=0)
    else:
        scores["ensemble_dynamics_uncertainty"] = scores["kinematic_residual_threshold"]
    if "pca" in models:
        recon = models["pca"].inverse_transform(models["pca"].transform(Xs))
        scores["autoencoder_reconstruction_audit"] = np.mean((Xs - recon) ** 2, axis=1)
    else:
        scores["autoencoder_reconstruction_audit"] = scores["kinematic_residual_threshold"]
    if "classifier" in models:
        scores["supervised_failure_classifier"] = models["classifier"].predict_proba(Xs)[:, 1]
    else:
        scores["supervised_failure_classifier"] = scores["physics_violation_audit"]
    return scores


def score_ablation(X: np.ndarray, ablation: str) -> np.ndarray:
    disabled = {
        "no_contact_check": {"contact"},
        "no_support_check": {"support"},
        "no_energy_check": {"energy"},
        "no_friction_slip_check": {"friction"},
        "no_actuator_check": {"actuator"},
        "no_causality_check": {"causality"},
        "full_physics_violation_audit": set(),
    }.get(ablation, set())
    if ablation == "scalar_residual_only":
        idx = {name: FEATURE_NAMES.index(name) for name in FEATURE_NAMES}
        return X[:, idx["max_pose_jump"]] / 0.035 + X[:, idx["energy_work_mismatch"]] / 0.045 + X[:, idx["max_contact_force"]] / 650.0
    return explicit_physics_score(X, disabled=disabled)


def evaluate_rows(dataset_rows: list[dict], models: dict, methods: list[str], ablation: bool = False) -> list[dict]:
    X, y = rows_to_matrix(dataset_rows)
    thresholds = models["thresholds"]
    method_scores = compute_method_scores(X, models)
    rows = []
    for method in methods:
        if ablation:
            scores = score_ablation(X, method)
            valid_scores = scores[y == 0]
            threshold = float(np.quantile(valid_scores, 0.94)) if len(valid_scores) else thresholds["physics_violation_audit"]
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
        labels = [int(r["label"]) for r in group]
        flags = [int(r["flag"]) for r in group]
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        accuracy = (tp + tn) / max(1, len(group))
        fpr = fp / max(1, fp + tn)
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
                "episodes": len(group),
                "seeds": len({r["seed"] for r in group}),
            }
        )
        out_rows.append(out)
    return out_rows


def seed_metrics(rows: list[dict]) -> list[dict]:
    return summarize(rows, ["method", "split", "seed"])


def pairwise_stats(seed_rows: list[dict], split: str = "combined_violation_shift") -> list[dict]:
    proposed = "physics_violation_audit"
    metric_map = {
        (r["method"], r["split"], r["seed"]): float(r["f1"])
        for r in seed_rows
        if r["split"] == split
    }
    rows = []
    for method in METHODS:
        if method == proposed:
            continue
        diffs = []
        for seed in SEEDS:
            p_key = (proposed, split, seed)
            b_key = (method, split, seed)
            if p_key in metric_map and b_key in metric_map:
                diffs.append(metric_map[p_key] - metric_map[b_key])
        if not diffs:
            continue
        mean_diff = float(np.mean(diffs))
        sd = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
        t_stat = mean_diff / (sd / math.sqrt(len(diffs)) + 1e-9)
        rows.append(
            {
                "split": split,
                "baseline": method,
                "mean_f1_diff_vs_audit": f"{mean_diff:.4f}",
                "paired_t_approx": f"{t_stat:.4f}",
                "normal_approx_p": f"{normal_p_from_t(t_stat):.4f}",
                "seeds": len(diffs),
            }
        )
    return rows


def plot_metric(metrics: list[dict], path: Path, metric: str, title: str, ylabel: str) -> None:
    selected = [
        "kinematic_residual_threshold",
        "energy_residual_threshold",
        "ensemble_dynamics_uncertainty",
        "autoencoder_reconstruction_audit",
        "supervised_failure_classifier",
        "physics_violation_audit",
        "oracle_violation_labels",
    ]
    x = np.arange(len(MAIN_SPLITS))
    width = 0.10
    fig, ax = plt.subplots(figsize=(12, 5))
    for idx, method in enumerate(selected):
        vals = []
        for split in MAIN_SPLITS:
            match = [r for r in metrics if r["method"] == method and r["split"] == split]
            vals.append(float(match[0][metric]) if match else 0.0)
        ax.bar(x + (idx - len(selected) / 2) * width + width / 2, vals, width, label=method.replace("_", " "))
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in MAIN_SPLITS], fontsize=8)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_ablation(metrics: list[dict], path: Path) -> None:
    vals = [(r["method"], float(r["f1"]), float(r["false_positive_rate"])) for r in metrics if r["split"] == "combined_violation_shift"]
    vals.sort(key=lambda item: item[1], reverse=True)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(vals))
    ax.bar(x, [v[1] for v in vals], color="#59704d")
    ax.plot(x, [v[2] for v in vals], marker="o", color="#9a3d2f", label="false positive rate")
    ax.set_xticks(x)
    ax.set_xticklabels([v[0].replace("_", "\n") for v in vals], fontsize=8)
    ax.set_ylabel("Combined-shift F1")
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("Physics-audit ablations")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_stress(stress_metrics: list[dict], path: Path) -> None:
    selected = ["kinematic_residual_threshold", "ensemble_dynamics_uncertainty", "autoencoder_reconstruction_audit", "supervised_failure_classifier", "physics_violation_audit"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for method in selected:
        xs, ys = [], []
        for row in stress_metrics:
            if row["method"] == method:
                xs.append(float(row["stress_level"]))
                ys.append(float(row["f1"]))
        order = np.argsort(xs)
        ax.plot(np.asarray(xs)[order], np.asarray(ys)[order], marker="o", label=method.replace("_", " "))
    ax.set_xlabel("Noise/corruption severity")
    ax.set_ylabel("F1")
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("Stress sweep: sensor noise + violation severity")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_negative_cases() -> list[dict]:
    return [
        {
            "case": "sensor_timestamp_skew",
            "expected_behavior": "audit should separate clock skew from physics violation",
            "observed_failure_mode": "causality checks can flag asynchronous but physically valid traces",
            "submission_implication": "needs timestamp calibration before deployment claims",
        },
        {
            "case": "legal_rare_bouncing_contact",
            "expected_behavior": "high contact impulse can be physically valid",
            "observed_failure_mode": "simple contact thresholds create false positives on rare impacts",
            "submission_implication": "requires impact-aware thresholds or learned contact regimes",
        },
        {
            "case": "out_of_distribution_deformable_object",
            "expected_behavior": "rigid-body audit should abstain",
            "observed_failure_mode": "energy and support checks assume rigid object state",
            "submission_implication": "scope is rigid-body MuJoCo manipulation",
        },
        {
            "case": "semantic_task_violation_with_valid_physics",
            "expected_behavior": "physics audit should remain silent",
            "observed_failure_mode": "all physics checks pass while the robot performs the wrong task",
            "submission_implication": "audit is not a complete safety monitor",
        },
    ]


def main() -> None:
    models = train_baselines()

    main_rows_raw = make_dataset(MAIN_SPLITS, EPISODES_PER_SEED, severity=1.0, noise_level=0.001)
    write_csv(RESULTS / "physics_audit_rollouts.csv", main_rows_raw)
    main_eval = evaluate_rows(main_rows_raw, models, METHODS)
    write_csv(RESULTS / "physics_audit_raw.csv", main_eval)
    seed_rows = seed_metrics(main_eval)
    write_csv(RESULTS / "raw_seed_metrics.csv", seed_rows)
    metrics = summarize(main_eval, ["method", "split"])
    write_csv(RESULTS / "physics_audit_metrics.csv", metrics)
    write_csv(RESULTS / "metrics.csv", metrics)
    pairwise = pairwise_stats(seed_rows)
    write_csv(RESULTS / "physics_audit_pairwise.csv", pairwise)
    write_csv(RESULTS / "pairwise_stats.csv", pairwise)

    ablation_rows_raw = make_dataset(["combined_violation_shift"], ABLATION_EPISODES_PER_SEED, severity=1.0, noise_level=0.001)
    ablation_eval = evaluate_rows(ablation_rows_raw, models, ABLATIONS, ablation=True)
    write_csv(RESULTS / "physics_audit_ablation_raw.csv", ablation_eval)
    ablation_metrics = summarize(ablation_eval, ["method", "split"])
    write_csv(RESULTS / "physics_audit_ablation.csv", ablation_metrics)
    write_csv(RESULTS / "ablation_metrics.csv", ablation_metrics)

    stress_methods = ["kinematic_residual_threshold", "ensemble_dynamics_uncertainty", "autoencoder_reconstruction_audit", "supervised_failure_classifier", "physics_violation_audit"]
    stress_eval = []
    for level in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        severity = 0.35 + 0.85 * level
        noise = 0.0005 + 0.004 * level
        raw = make_dataset(["combined_violation_shift"], STRESS_EPISODES_PER_SEED, severity=severity, noise_level=noise)
        rows = evaluate_rows(raw, models, stress_methods)
        for row in rows:
            row["split"] = f"stress_{level:.2f}"
        stress_eval.extend(rows)
    stress_metrics = summarize(stress_eval, ["method", "split"])
    stress_output = []
    for row in stress_metrics:
        out = dict(row)
        out["stress_level"] = out["split"].replace("stress_", "")
        stress_output.append(out)
    write_csv(RESULTS / "stress_sweep.csv", stress_output)
    write_csv(FIGURES / "stress_curve_data.csv", stress_output)

    write_csv(RESULTS / "negative_cases.csv", make_negative_cases())
    plot_metric(metrics, FIGURES / "physics_audit_f1_by_split.png", "f1", "Physics-violation audit F1 by split", "F1")
    plot_metric(metrics, FIGURES / "physics_audit_false_positive_by_split.png", "false_positive_rate", "False positives on valid/near-valid traces", "False positive rate")
    plot_ablation(ablation_metrics, FIGURES / "physics_audit_ablation_f1.png")
    plot_stress(stress_output, FIGURES / "physics_audit_stress_sweep.png")

    combined = {r["method"]: r for r in metrics if r["split"] == "combined_violation_shift"}
    ab_combined = {r["method"]: r for r in ablation_metrics if r["split"] == "combined_violation_shift"}
    proposed = combined["physics_violation_audit"]
    best_non_oracle = max(
        (r for m, r in combined.items() if m not in {"physics_violation_audit", "oracle_violation_labels"}),
        key=lambda r: float(r["f1"]),
    )
    terminal = "STRONG_REVISE"
    reason = "explicit audit helps on real MuJoCo violation traces but needs hardware/public benchmark validation and manual related work"
    if float(proposed["f1"]) <= float(best_non_oracle["f1"]) + 0.025:
        terminal = "KILL_ARCHIVE"
        reason = "physics audit is matched or beaten by a non-oracle baseline on combined violation shift"
    scalar = ab_combined.get("scalar_residual_only")
    full = ab_combined.get("full_physics_violation_audit")
    if scalar and full and float(scalar["f1"]) >= float(full["f1"]) - 0.025:
        terminal = "KILL_ARCHIVE"
        reason = "scalar residual ablation matches the full physics audit"

    with (RESULTS / "summary.txt").open("w", encoding="utf-8") as handle:
        handle.write("Paper 69 real MuJoCo robotic physics violation audits rebuild\n")
        handle.write(f"Seeds: {SEEDS}; episodes per seed: {EPISODES_PER_SEED}; workers: {MAX_WORKERS}\n")
        handle.write("Main raw rollouts: %d; main eval rows: %d; ablation rows: %d; stress rows: %d\n" % (len(main_rows_raw), len(main_eval), len(ablation_eval), len(stress_eval)))
        handle.write(f"Terminal decision: {terminal}\n")
        handle.write(f"Terminal reason: {reason}\n")
        handle.write("\nCombined-violation-shift main results:\n")
        for method in METHODS:
            row = combined[method]
            handle.write(
                f"- {method}: f1={row['f1']} precision={row['precision']} recall={row['recall']} "
                f"fpr={row['false_positive_rate']} accuracy={row['accuracy']} flag={row['flag_rate']}\n"
            )
        handle.write("\nCombined-violation-shift ablations:\n")
        for method, row in sorted(ab_combined.items()):
            handle.write(
                f"- {method}: f1={row['f1']} precision={row['precision']} recall={row['recall']} fpr={row['false_positive_rate']}\n"
            )
        handle.write("\nPairwise combined-shift comparisons vs physics_violation_audit:\n")
        for row in pairwise:
            handle.write(
                f"- {row['baseline']}: diff={row['mean_f1_diff_vs_audit']} t={row['paired_t_approx']} p={row['normal_approx_p']}\n"
            )

    print(f"wrote Paper 69 MuJoCo evidence to {RESULTS}")


if __name__ == "__main__":
    main()
