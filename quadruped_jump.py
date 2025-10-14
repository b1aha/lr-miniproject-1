"""
~/quadruped_jump.py
Oct 2025; Jiri Blaha, Giulia Cortelazzo, Semih Zaman
École Polytechnique Fédérale de Lausanne,
CH-1015 Lausanne,
Switzerland
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

from env.simulation import QuadSimulator, SimulationOptions
from profiles import FootForceProfile
from jump_params import JUMP_PARAMS

N_LEGS = 4
N_JOINTS = 3

KP_XY = 280.0
KP_Z = 1250.0
KD_FLIGHT_XY = 8.0
KD_FLIGHT_Z = 8.0
KD_STANCE_XY = 40.0
KD_STANCE_Z = 150.0

LEG_SIGN_X = np.array([+1, +1, -1, -1], dtype=float)
LEG_SIGN_Y = np.array([-1, +1, -1, +1], dtype=float)


def quadruped_jump(jump_type: str = "forward"):
    params = JUMP_PARAMS[jump_type]
    IMPULSE_F0 = params["IMPULSE_F0"]
    IDLE_F1 = params["IDLE_F1"]
    FORCE_FX = params["FORCE_FX"]
    FORCE_FY = params["FORCE_FY"]
    FORCE_FZ = params["FORCE_FZ"]
    N_JUMPS = params["N_JUMPS"]
    K_VMC = params["K_VMC"]

    global X_OFFSET, Y_OFFSET, Z_OFFSET
    X_OFFSET = params["X_OFFSET"]
    Y_OFFSET = params["Y_OFFSET"]
    Z_OFFSET = params["Z_OFFSET"]

    sim_options = SimulationOptions(
        on_rack=False,
        render=True,
        record_video=False,
        tracking_camera=True,
    )
    simulator = QuadSimulator(sim_options)

    force_profile = FootForceProfile(
        f0=IMPULSE_F0, f1=IDLE_F1, Fx=FORCE_FX, Fy=FORCE_FY, Fz=FORCE_FZ
    )

    jump_T = force_profile.impulse_duration() + force_profile.idle_duration()
    n_steps = int(N_JUMPS * jump_T / sim_options.timestep)

    force_log, time_log, pos_log = [], [], []
    tau_log = []

    for _ in range(n_steps):
        if not simulator.is_connected():
            break
        force_profile.step(sim_options.timestep)
        F = force_profile.force()
        force_log.append(F.copy())
        time_log.append(simulator.time())
        pos_log.append(np.array(simulator.get_base_position()))

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
        tau += apply_force_profile(simulator, force_profile, jump_type)
        tau += gravity_compensation(simulator)
        if np.any(simulator.get_foot_contacts()):
            tau += virtual_model(simulator, K_VMC)

        tau_log.append(tau.copy())

        simulator.set_motor_targets(tau)
        simulator.step()

    simulator.close()

    force_log = np.array(force_log)
    tau_log = np.array(tau_log)
    time_log = np.array(time_log)

    if len(force_log) > 0:
        plt.figure(figsize=(10, 5))
        plt.plot(time_log, force_log[:, 0], label="Fx")
        plt.plot(time_log, force_log[:, 1], label="Fy")
        plt.plot(time_log, force_log[:, 2], label="Fz")
        plt.xlabel("Time (s)")
        plt.ylabel("Force (N)")
        plt.title("Foot Force Profile Over Time")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    if len(tau_log) > 0:
        fig, axs = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
        leg_names = ["FR", "FL", "RR", "RL"]
        joint_names = ["Hip", "Thigh", "Calf"]
        for leg_id in range(N_LEGS):
            ax = axs.flat[leg_id]
            s = leg_id * N_JOINTS
            ax.plot(time_log, tau_log[:, s + 0], label=joint_names[0])
            ax.plot(time_log, tau_log[:, s + 1], label=joint_names[1])
            ax.plot(time_log, tau_log[:, s + 2], label=joint_names[2])
            ax.set_title(f"Leg {leg_names[leg_id]}")
            ax.set_ylabel("Torque (Nm)")
            ax.grid(True)
            if leg_id in (2, 3):
                ax.set_xlabel("Time (s)")
            ax.legend(fontsize=8)
        plt.tight_layout()
        plt.show()

    pos_log = np.array(pos_log)
    if len(pos_log) > 0:
        xy = pos_log[:, :2] - pos_log[0, :2]
        plt.figure(figsize=(6, 6))
        plt.plot(xy[:, 0], xy[:, 1])
        plt.scatter([0], [0], s=60)
        plt.scatter([xy[-1, 0]], [xy[-1, 1]], s=60)
        plt.axis("equal")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.title("Top-Down Base Trajectory (start at [0, 0])")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def _leg_target_offsets(leg_id: int) -> np.ndarray:
    z_offset = Z_OFFSET
    y_offset = Y_OFFSET
    x_offset = X_OFFSET
    if leg_id == 0:
        return np.array([+x_offset, -y_offset, z_offset])
    elif leg_id == 1:
        return np.array([+x_offset, +y_offset, z_offset])
    elif leg_id == 2:
        return np.array([-x_offset, -y_offset, z_offset])
    else:
        return np.array([-x_offset, +y_offset, z_offset])


def nominal_position(
    simulator: "QuadSimulator",
    KP_XY: float,
    KP_Z: float,
    KD_FLIGHT_XY: float,
    KD_FLIGHT_Z: float,
    KD_STANCE_XY: float,
    KD_STANCE_Z: float,
) -> np.ndarray:
    tau = np.zeros(N_JOINTS * N_LEGS)
    foot_contacts = simulator.get_foot_contacts()
    Kp = np.diag([KP_XY, KP_XY, KP_Z])
    Kd_flight = np.diag([KD_FLIGHT_XY, KD_FLIGHT_XY, KD_FLIGHT_Z])
    Kd_stance = np.diag([KD_STANCE_XY, KD_STANCE_XY, KD_STANCE_Z])
    for leg_id in range(N_LEGS):
        dq = simulator.get_motor_velocities(leg_id)
        J, p = simulator.get_jacobian_and_position(leg_id)
        v = J @ dq
        pd = _leg_target_offsets(leg_id)
        vd = np.zeros(3)
        Kd = Kd_stance if foot_contacts[leg_id] else Kd_flight
        F = Kp @ (pd - p) + Kd @ (vd - v)
        tau_i = J.T @ F
        start = leg_id * N_JOINTS
        tau[start: start + N_JOINTS] = tau_i
    return tau


def virtual_model(simulator: "QuadSimulator", k_vmc: float) -> np.ndarray:
    tau = np.zeros(N_JOINTS * N_LEGS)
    contacts = simulator.get_foot_contacts()
    if not np.any(contacts):
        return tau
    R = simulator.get_base_orientation_matrix()
    P_body = np.array([[1, 1, -1, -1], [-1, 1, -1, 1], [0, 0, 0, 0]])
    z = (R @ P_body)[2, :]
    F_world = np.zeros((3, N_LEGS))
    F_world[2, :] = k_vmc * z
    F_leg = R.T @ F_world
    for leg_id in range(N_LEGS):
        if not contacts[leg_id]:
            continue
        J, _ = simulator.get_jacobian_and_position(leg_id)
        tau_i = J.T @ F_leg[:, leg_id]
        start = leg_id * N_JOINTS
        tau[start: start + N_JOINTS] = tau_i
    return tau


def gravity_compensation(simulator: "QuadSimulator") -> np.ndarray:
    tau = np.zeros(N_JOINTS * N_LEGS)
    m_total = simulator.get_mass()
    g = 9.81
    Fg = np.array([0.0, 0.0, -(m_total / 4.0) * g])
    for leg_id in range(N_LEGS):
        J, _ = simulator.get_jacobian_and_position(leg_id)
        tau_i = J.T @ Fg
        start = leg_id * N_JOINTS
        tau[start: start + N_JOINTS] = tau_i
    return tau


def per_leg_force(
    F_base: np.ndarray,
    leg_id: int,
    jump_type: str
) -> np.ndarray:
    Fx, Fy, Fz = F_base
    sx, sy = LEG_SIGN_X[leg_id], LEG_SIGN_Y[leg_id]
    if jump_type in ("twist_ccw", "twist_cw"):
        s = +1.0 if jump_type == "twist_ccw" else -1.0
        Fx_i = s * sy * Fx
        Fy_i = s * sx * Fy
        return np.array([Fx_i, Fy_i, Fz], dtype=float)
    return F_base


def apply_force_profile(
    simulator: "QuadSimulator", force_profile: FootForceProfile, jump_type: str
) -> np.ndarray:
    tau = np.zeros(N_JOINTS * N_LEGS)
    F = force_profile.force()
    foot_contacts = simulator.get_foot_contacts()
    front_sym = foot_contacts[0] and foot_contacts[1]
    rear_sym = foot_contacts[2] and foot_contacts[3]
    fx_allowed = front_sym and rear_sym
    for leg_id in range(N_LEGS):
        if foot_contacts[leg_id]:
            F_i = per_leg_force(F, leg_id, jump_type).copy()
            if not fx_allowed:
                F_i[0] = 0.0
            J, _ = simulator.get_jacobian_and_position(leg_id)
            tau_i = J.T @ F_i
        else:
            tau_i = np.zeros(3)
        start = leg_id * N_JOINTS
        tau[start: start + N_JOINTS] = tau_i
    return tau


if __name__ == "__main__":
    jt = sys.argv[1] if len(sys.argv) > 1 else "forward"
    quadruped_jump(jt)
