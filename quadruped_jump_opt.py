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

# Quadruped parameters
N_LEGS = 4
N_JOINTS = 3

# Cartesian PD gains
KP_XY = 280.0
KP_Z = 1250.0
KD_FLIGHT_XY = 8.0
KD_FLIGHT_Z = 8.0
KD_STANCE_XY = 40.0
KD_STANCE_Z = 150.0

# Optimization parameters
N_TRIALS = 35
SETTLE_T = 1.0
MAX_WAIT_AFTER_IMPULSE = 5.0
HEIGHT_THRESH = 0.08
HEIGHT_PENALTY = 2.0
TILT_THRESH_DEG = 30.0
TILT_PENALTY = 1.5


def _yaw_from_R(R: np.ndarray) -> float:
    return math.atan2(R[1, 0], R[0, 0])


def _round_num(x):
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(f"{x:.12f}")
    return x


def _ordered_jump_dict(src: dict) -> OrderedDict:
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
    try:
        opt_mod = importlib.import_module("optimals")
        data = dict(opt_mod.OPTIMALS)
    except Exception:
        data = {}

    jt_apply = jt
    base = dict(JUMP_PARAMS[jt_apply])
    base["IMPULSE_F0"] = best_params["f0"]
    base["IDLE_F1"] = best_params["f1"]
    base["FORCE_FX"] = best_params["Fx"]
    base["FORCE_FY"] = best_params["Fy"]
    base["FORCE_FZ"] = best_params["Fz"]

    data[jt_apply] = _ordered_jump_dict(base)

    ordered_all = OrderedDict()
    for k in sorted(data.keys()):
        ordered_all[k] = data[k]

    with open("optimals.py", "w") as f:
        f.write("OPTIMALS = ")
        f.write(json.dumps(ordered_all, indent=4))
        f.write("\n")


def quadruped_jump_opt(jt: str, seed: int):
    jt_offsets = "forward" if jt == "hopping" else jt
    offsets = JUMP_PARAMS[jt_offsets]
    qj.X_OFFSET = offsets["X_OFFSET"]
    qj.Y_OFFSET = offsets["Y_OFFSET"]
    qj.Z_OFFSET = offsets["Z_OFFSET"]

    sim_options = SimulationOptions(
        on_rack=False,
        render=True,
        record_video=False,
        tracking_camera=True,
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
    space = DECISION_VARS[jt]
    f0 = trial.suggest_float("f0", space["f0"]["low"], space["f0"]["high"])
    f1 = trial.suggest_float("f1", space["f1"]["low"], space["f1"]["high"])
    Fx = trial.suggest_float("Fx", space["Fx"]["low"], space["Fx"]["high"])
    Fy = trial.suggest_float("Fy", space["Fy"]["low"], space["Fy"]["high"])
    Fz = trial.suggest_float("Fz", space["Fz"]["low"], space["Fz"]["high"])

    jt_apply = "forward" if jt == "hopping" else jt
    K_VMC = JUMP_PARAMS[jt_apply]["K_VMC"]

    simulator.reset()

    sim_options = simulator.options
    profile = FootForceProfile(f0=f0, f1=f1, Fx=Fx, Fy=Fy, Fz=Fz)

    jump_T = profile.impulse_duration() + profile.idle_duration()

    if jt == "hopping":
        total_T = max(6 * jump_T, 2.0)
        n_steps = int(np.ceil(total_T / sim_options.timestep))
        p0 = np.array(simulator.get_base_position(), dtype=float)
        R0 = simulator.get_base_orientation_matrix()
        yaw0 = _yaw_from_R(R0)
        min_height = p0[2]
        tilt_worst = 0.0

        for _ in range(n_steps):
            profile.step(sim_options.timestep)

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
            tau += apply_force_profile(simulator, profile, jt_apply)
            tau += gravity_compensation(simulator)

            if np.any(simulator.get_foot_contacts()):
                tau += virtual_model(simulator, K_VMC)

            simulator.set_motor_targets(tau)
            simulator.step()

            p = np.array(simulator.get_base_position(), dtype=float)
            R = simulator.get_base_orientation_matrix()
            min_height = min(min_height, p[2])
            up_dot = float(R[2, 2])
            up_dot = max(min(up_dot, 1.0), -1.0)
            tilt = math.acos(up_dot)
            tilt_worst = max(tilt_worst, tilt)

        pend = np.array(simulator.get_base_position(), dtype=float)
        dx = float(pend[0] - p0[0])
        dy = float(pend[1] - p0[1])
        yaw_end = _yaw_from_R(simulator.get_base_orientation_matrix())
        dyaw = yaw_end - yaw0
    else:
        n_imp = int(np.ceil(jump_T / sim_options.timestep))
        n_extra = int(np.ceil(SETTLE_T / sim_options.timestep))

        p0 = np.array(simulator.get_base_position(), dtype=float)
        R0 = simulator.get_base_orientation_matrix()
        yaw0 = _yaw_from_R(R0)
        min_height = p0[2]
        tilt_worst = 0.0

        for _ in range(n_imp):
            profile.step(sim_options.timestep)

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
            tau += apply_force_profile(simulator, profile, jt_apply)
            tau += gravity_compensation(simulator)

            if np.any(simulator.get_foot_contacts()):
                tau += virtual_model(simulator, K_VMC)

            simulator.set_motor_targets(tau)
            simulator.step()

            p = np.array(simulator.get_base_position(), dtype=float)
            R = simulator.get_base_orientation_matrix()
            min_height = min(min_height, p[2])
            up_dot = float(R[2, 2])
            up_dot = max(min(up_dot, 1.0), -1.0)
            tilt = math.acos(up_dot)
            tilt_worst = max(tilt_worst, tilt)

        started_wait = False
        waited = 0
        t_since_impulse = 0.0

        while True:
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
            tau += gravity_compensation(simulator)

            if np.any(simulator.get_foot_contacts()):
                tau += virtual_model(simulator, K_VMC)

            simulator.set_motor_targets(tau)
            simulator.step()

            t_since_impulse += sim_options.timestep

            p = np.array(simulator.get_base_position(), dtype=float)
            R = simulator.get_base_orientation_matrix()
            min_height = min(min_height, p[2])
            up_dot = float(R[2, 2])
            up_dot = max(min(up_dot, 1.0), -1.0)
            tilt = math.acos(up_dot)
            tilt_worst = max(tilt_worst, tilt)

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
        yaw_end = _yaw_from_R(simulator.get_base_orientation_matrix())
        dyaw = yaw_end - yaw0

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
        "hopping": dx / (max(1, int(np.ceil(max(6 * jump_T, 2.0) / sim_options.timestep))) * sim_options.timestep),
    }
    score = score_map[jt]

    if not np.isfinite(score):
        return -1e3

    penalty = 0.0
    if min_height < HEIGHT_THRESH:
        penalty += HEIGHT_PENALTY
    if tilt_worst > math.radians(TILT_THRESH_DEG):
        penalty += TILT_PENALTY

    return score - penalty


if __name__ == "__main__":
    jt = sys.argv[1] if len(sys.argv) > 1 else "forward"
    seeds = [10 * i for i in range(10)]
    best_values = []
    best_params_by_seed = []
    for s in seeds:
        np.random.seed(s)
        random.seed(s)
        val, params = quadruped_jump_opt(jt, seed=s)
        best_values.append(val)
        best_params_by_seed.append(params)

    best_values = np.asarray(best_values, dtype=float)
    mean_val = float(np.mean(best_values))
    std_val = float(np.std(best_values, ddof=0))

    print("Seeds:", seeds)
    print("Best values per seed:", best_values.tolist())
    print("Mean best value:", mean_val)
    print("Std best value:", std_val)

    best_idx = int(np.argmax(best_values))
    _write_optimals(jt, best_params_by_seed[best_idx])

    plt.figure()
    plt.title(f"Best score per seed [{jt}]")
    plt.xlabel("seed")
    plt.ylabel("best score")
    plt.plot(seeds, best_values, marker="o")
    plt.grid(True)

    plt.show()
