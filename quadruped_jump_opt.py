"""
~/quadruped_jump_opt.py
Oct 2025; Jiri Blaha, Giulia Cortelazzo, Semih Zaman
École Polytechnique Fédérale de Lausanne,
CH-1015 Lausanne,
Switzerland
"""

import numpy as np
import sys
import math
import optuna
import importlib
import json
import random
import matplotlib.pyplot as plt
from collections import OrderedDict
from functools import partial
from optuna.trial import Trial
from typing import Optional, Tuple

from env.simulation import QuadSimulator, SimulationOptions
from profiles import FootForceProfile
from jump_params_opt import DECISION_VARS
from jump_params import JUMP_PARAMS
import quadruped_jump as qj
from quadruped_jump import (
    nominal_position,
    gravity_compensation,
    apply_force_profile,
    virtual_model,
)

# === Global constants and controller gains ===
# Quadruped model constants.
N_LEGS = 4
N_JOINTS = 3

# Cartesian PD gains.
KP_XY = 280.0
KP_Z = 1250.0
KD_FLIGHT_XY = 8.0
KD_FLIGHT_Z = 8.0
KD_STANCE_XY = 40.0
KD_STANCE_Z = 150.0

# VMC for hopping
K_VMC_HOP = 36.0

# === Optimization parameters and constraints ===
N_TRIALS = 40                 # trials per Optuna study/seed
SETTLE_T = 1.0                # wait cap after impulse (all feet touch)
MAX_WAIT_AFTER_IMPULSE = 5.0  # absolute wait cap after impulse (possible fall)
HEIGHT_THRESH = 0.08          # minimum allowed base height
HEIGHT_PENALTY = 2.0          # height penalty
TILT_THRESH_DEG = 30.0        # maximum allowed tilt
TILT_PENALTY = 1.5            # tilt penalty
HEIGHT_THRESH_HOP = 0.18      # minimum allowed base height (hopping)
HEIGHT_PENALTY_HOP = 15.0     # height penalty (hopping)
TILT_THRESH_DEG_HOP = 7.0     # maximum allowed tilt (hopping)
TILT_PENALTY_HOP = 15.0       # tilt penalty (hopping)


def _yaw_from_R(R: np.ndarray) -> float:
    """
    Extract yaw from a 3x3 rotation matrix.

    Args:
        R: 3x3 rotation matrix.

    Returns:
        Yaw angle in radians.
    """
    return math.atan2(R[1, 0], R[0, 0])


def _round_num(x):
    """
    Number rounding utility.

    Args:
        x: Value to round.

    Returns:
        Rounded value of the same broad type.
    """
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(f"{x:.12f}")
    return x


def _ordered_jump_dict(src: dict) -> OrderedDict:
    """
    Produce a jump param dict with a consistent key order.

    Args:
        src: Source dictionary of jump parameters.

    Returns:
        OrderedDict with specified key ordering.
    """
    order = [
        "IMPULSE_F0",
        "IDLE_F1",
        "FORCE_FX",
        "FORCE_FY",
        "FORCE_FZ",
        "N_JUMPS",
        "K_VMC",
        "Z_OFFSET",
        "Y_OFFSET",
        "X_OFFSET",
    ]
    out = OrderedDict()
    for k in order:
        if k in src:
            out[k] = _round_num(src[k])
    for k, v in src.items():
        if k not in out:
            out[k] = _round_num(v)
    return out


def _write_optimals(jt: str, best_params: dict) -> None:
    """
    Writes best parameters into optimals.py for later use.

    Args:
        jt: Jump type (forward, lateral_left, etc.).
        best_params: Dict containing force profile params.

    Returns:
        None.
    """
    try:
        opt_mod = importlib.import_module("optimals")
        data = dict(opt_mod.OPTIMALS)
    except Exception:
        data = {}

    jt_base = "forward" if jt == "hopping" else jt
    base = dict(JUMP_PARAMS[jt_base])
    base["IMPULSE_F0"] = best_params["f0"]
    base["IDLE_F1"] = best_params["f1"]
    base["FORCE_FX"] = best_params["Fx"]
    base["FORCE_FY"] = best_params["Fy"]
    base["FORCE_FZ"] = best_params["Fz"]

    if jt == "hopping":
        base["K_VMC"] = K_VMC_HOP
    else:
        base["K_VMC"] = JUMP_PARAMS[jt]["K_VMC"]

    data[jt] = _ordered_jump_dict(base)

    ordered_all = OrderedDict()
    for k in sorted(data.keys()):
        ordered_all[k] = data[k]

    with open("optimals.py", "w") as f:
        f.write("OPTIMALS = ")
        f.write(json.dumps(ordered_all, indent=4))
        f.write("\n")


def _apply_step(
    simulator: QuadSimulator,
    profile: Optional[FootForceProfile],
    jt_apply: str,
    k_vmc: float
) -> Tuple[np.ndarray, float]:
    """
    Advance the simulation by one timestep.

    Args:
        simulator: Physics/simulation wrapper.
        profile: FootForceProfile from ~/profiles.py or None.
        jt_apply: Jump type (forward, lateral_left, etc.).
        k_vmc: Scalar VMC gain.

    Returns:
        p: (3,) np.ndarray with base position after step.
        tilt: Scalar tilt
    """
    tau = np.zeros(N_JOINTS * N_LEGS)
    tau += nominal_position(
        simulator,
        KP_XY,
        KP_Z,
        KD_FLIGHT_XY,
        KD_FLIGHT_Z,
        KD_STANCE_XY,
        KD_STANCE_Z,
    )
    if profile is not None:
        tau += apply_force_profile(simulator, profile, jt_apply)
    tau += gravity_compensation(simulator)
    if np.any(simulator.get_foot_contacts()):
        tau += virtual_model(simulator, k_vmc)
    simulator.set_motor_targets(tau)
    simulator.step()
    p = np.array(simulator.get_base_position(), dtype=float)
    R = simulator.get_base_orientation_matrix()
    up_dot = float(np.clip(R[2, 2], -1.0, 1.0))
    tilt = math.acos(up_dot)
    return p, tilt


def quadruped_jump_opt(jt: str, seed: int):
    """
    Run a SINGLE-seed Optuna optimization for given jt.

    Args:
        jt: Jump type (forward, lateral_left, etc.).
        seed: Integer seed.

    Returns:
        (best_value, best_params) from the study.
    """
    # Pull nominal offsets for this jt.
    jt_offsets = "forward" if jt == "hopping" else jt
    offsets = JUMP_PARAMS[jt_offsets]
    qj.X_OFFSET = offsets["X_OFFSET"]
    qj.Y_OFFSET = offsets["Y_OFFSET"]
    qj.Z_OFFSET = offsets["Z_OFFSET"]

    sim_options = SimulationOptions(
        on_rack=False,
        render=True,
        record_video=False,
        tracking_camera=True
    )
    simulator = QuadSimulator(sim_options)

    objective = partial(evaluate_jumping, simulator=simulator, jt=jt)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        study_name=f"Quadruped Jumping Optimization ({jt}) [seed={seed}]",
        sampler=sampler,
        direction="maximize",
    )
    study.optimize(objective, n_trials=N_TRIALS)

    simulator.close()
    return study.best_value, study.best_params


def evaluate_jumping(trial: Trial, simulator: QuadSimulator, jt: str) -> float:
    """
    Simulate jump and return a objective value of score - penalty.

    Args:
        trial: Optuna Trial object.
        simulator: Physics/simulation wrapper.
        jt: Jump type (forward, lateral_left, etc.).

    Returns:
        Scalar objective value of score - penalty.
    """
    space = DECISION_VARS[jt]
    f0 = trial.suggest_float("f0", space["f0"]["low"], space["f0"]["high"])
    f1 = trial.suggest_float("f1", space["f1"]["low"], space["f1"]["high"])
    Fx = trial.suggest_float("Fx", space["Fx"]["low"], space["Fx"]["high"])
    Fy = trial.suggest_float("Fy", space["Fy"]["low"], space["Fy"]["high"])
    Fz = trial.suggest_float("Fz", space["Fz"]["low"], space["Fz"]["high"])

    K_VMC = K_VMC_HOP if jt == "hopping" else JUMP_PARAMS[jt]["K_VMC"]

    simulator.reset()
    dt = simulator.options.timestep
    profile = FootForceProfile(f0=f0, f1=f1, Fx=Fx, Fy=Fy, Fz=Fz)

    p0 = np.array(simulator.get_base_position(), dtype=float)
    yaw0 = _yaw_from_R(simulator.get_base_orientation_matrix())
    min_height = p0[2]
    tilt_worst = 0.0

    jump_T = profile.impulse_duration() + profile.idle_duration()

    if jt == "hopping":
        total_T = max(10 * jump_T, 10.0)
        n_steps = int(np.ceil(total_T / dt))
        for _ in range(n_steps):
            profile.step(dt)
            p, tilt = _apply_step(simulator, profile, jt, K_VMC)
            min_height = min(min_height, p[2])
            tilt_worst = max(tilt_worst, tilt)
        pend = np.array(simulator.get_base_position(), dtype=float)
        dx = float(pend[0] - p0[0])
        dy = float(pend[1] - p0[1])
        dyaw = _yaw_from_R(simulator.get_base_orientation_matrix()) - yaw0
        total_time = n_steps * dt
    else:
        # No hopping -> one try to go as far as possible
        n_imp = int(np.ceil(jump_T / dt))
        for _ in range(n_imp):
            profile.step(dt)
            p, tilt = _apply_step(simulator, profile, jt, K_VMC)
            min_height = min(min_height, p[2])
            tilt_worst = max(tilt_worst, tilt)

        n_extra = int(np.ceil(SETTLE_T / dt))
        started_wait = False
        waited = 0
        t_since_impulse = 0.0

        # Post-impulse settling:
        # - start counting settle window once all feet are on ground
        # - stop after SETTLE_T worth of steps or MAX_WAIT cap
        while True:
            p, tilt = _apply_step(simulator, None, jt, K_VMC)
            min_height = min(min_height, p[2])
            tilt_worst = max(tilt_worst, tilt)
            t_since_impulse += dt

            contacts = simulator.get_foot_contacts()
            all_on_ground = all(contacts)
            if all_on_ground and not started_wait:
                started_wait = True
                waited = 0
            if started_wait:
                waited += 1
                if waited >= n_extra:
                    break
            if t_since_impulse >= MAX_WAIT_AFTER_IMPULSE:
                break

        pend = np.array(simulator.get_base_position(), dtype=float)
        dx = float(pend[0] - p0[0])
        dy = float(pend[1] - p0[1])
        dyaw = _yaw_from_R(simulator.get_base_orientation_matrix()) - yaw0
        total_time = None

    # Wrap yaw for consistency.
    while dyaw > math.pi:
        dyaw -= 2 * math.pi
    while dyaw < -math.pi:
        dyaw += 2 * math.pi

    score_map = {
        "forward": dx,
        "lateral_left": dy,
        "lateral_right": -dy,
        "twist_ccw": dyaw,
        "twist_cw": -dyaw,
        "hopping": dx / (total_time if total_time is not None else dt),
    }
    score = score_map[jt]

    # Guard against NaN/Inf.
    if not np.isfinite(score):
        return -1e3

    # Be stricter on hopping.
    if jt == "hopping":
        height_thresh = HEIGHT_THRESH_HOP
        tilt_thresh_deg = TILT_THRESH_DEG_HOP
        height_penalty = HEIGHT_PENALTY_HOP
        tilt_penalty = TILT_PENALTY_HOP
    else:
        height_thresh = HEIGHT_THRESH
        tilt_thresh_deg = TILT_THRESH_DEG
        height_penalty = HEIGHT_PENALTY
        tilt_penalty = TILT_PENALTY

    # Penalty calculation.
    penalty = 0.0
    if min_height < height_thresh:
        penalty += height_penalty
    if tilt_worst > math.radians(tilt_thresh_deg):
        penalty += tilt_penalty

    return score - penalty


if __name__ == "__main__":
    # Minimal CLI:
    #   python quadruped_jump_opt.py <jt>
    jt = sys.argv[1] if len(sys.argv) > 1 else "forward"
    seeds = [10 * i for i in range(10)]

    all_scores = []
    best_values = []
    best_params_by_seed = []

    # What follows is basically quadruped_jump_opt()
    for s in seeds:
        # Ensure reproducibility.
        np.random.seed(s)
        random.seed(s)

        # Pull nominal offsets for this jt.
        jt_offsets = "forward" if jt == "hopping" else jt
        offsets = JUMP_PARAMS[jt_offsets]
        qj.X_OFFSET = offsets["X_OFFSET"]
        qj.Y_OFFSET = offsets["Y_OFFSET"]
        qj.Z_OFFSET = offsets["Z_OFFSET"]

        sim_options = SimulationOptions(
            on_rack=False,
            render=True,
            record_video=False,
            tracking_camera=True
        )
        simulator = QuadSimulator(sim_options)

        objective = partial(evaluate_jumping, simulator=simulator, jt=jt)
        sampler = optuna.samplers.TPESampler(seed=s)
        study = optuna.create_study(
            study_name=f"Quadruped Jumping Optimization ({jt}) [seed={s}]",
            sampler=sampler,
            direction="maximize",
        )
        study.optimize(objective, n_trials=N_TRIALS)

        simulator.close()

        # Collection of results.
        scores = [t.value for t in study.trials if t.value is not None]
        all_scores.append(scores)
        best_values.append(study.best_value)
        best_params_by_seed.append(study.best_params)

    # Eval results accross all seeds.
    best_values = np.asarray(best_values, dtype=float)
    mean_val = float(
        np.mean(best_values)
    ) if len(best_values) else float("nan")
    std_val = float(
        np.std(best_values, ddof=0)
    ) if len(best_values) else float("nan")

    print("Seeds:", seeds)
    print("Best values per seed:", best_values.tolist())
    print("Mean best value:", mean_val)
    print("Std best value:", std_val)

    # Write overall best parameters to optimals.py.
    best_idx = int(np.argmax(best_values)) if len(best_values) else 0
    _write_optimals(jt, best_params_by_seed[best_idx])

    # === Results plot ===
    plt.figure(figsize=(9, 6))
    for s, scores in zip(seeds, all_scores):
        if scores:
            plt.scatter(
                [s] * len(scores),
                scores,
                s=36,
                alpha=0.7,
            )
    plt.axhline(
        mean_val,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"Mean best value: ({mean_val:.3f})"
    )

    best_seed = seeds[best_idx]
    best_score = best_values[best_idx]
    plt.scatter(
        best_seed,
        best_score,
        s=36,
        marker="s",
        color="black",
        label=f"Best overall: ({best_score:.3f})"
    )

    plt.title(f"All trial scores per seed [{jt}]")
    plt.xlabel("Seed")
    plt.ylabel("Score")
    plt.xticks(seeds)
    plt.grid(False)
    plt.xlim(min(seeds) - 5, max(seeds) + 5)
    plt.legend()
    plt.tight_layout()
    plt.show()
