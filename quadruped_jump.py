# ~/quadruped_jump.py
# Oct 2025; Jiri Blaha, Giulia Cortelazzo, Semih Zaman
# École Polytechnique Fédérale de Lausanne,
# CH-1015 Lausanne,
# Switzerland
# ================================================

import numpy as np
from env.simulation import QuadSimulator, SimulationOptions
from profiles import FootForceProfile
import matplotlib.pyplot as plt

N_LEGS = 4
N_JOINTS = 3

# === TUNABLE PARAMETERS ===
IMPULSE_F0 = 3.8
IDLE_F1 = 1.0
FORCE_FX = 140.0
FORCE_FY = 0.0
FORCE_FZ = 370.0

N_JUMPS = 10

Z_OFFSET = -0.22
X_OFFSET = 0.0011

KP_XY = 280.0
KP_Z = 1200.0
KD_FLIGHT_XY = 8.0
KD_FLIGHT_Z = 8.0
KD_STANCE_XY = 40.0
KD_STANCE_Z = 150.0
# === TUNABLE PARAMETERS ===


def quadruped_jump():
    sim_options = SimulationOptions(
        on_rack=False,
        render=True,
        record_video=False,
        tracking_camera=True,
    )
    simulator = QuadSimulator(sim_options)

    force_profile = FootForceProfile(
        f0=IMPULSE_F0,
        f1=IDLE_F1,
        Fx=FORCE_FX,
        Fy=FORCE_FY,
        Fz=FORCE_FZ,
    )

    jump_T = force_profile.impulse_duration() + force_profile.idle_duration()
    n_steps = int(N_JUMPS * jump_T / sim_options.timestep)

    force_log, time_log = [], []
    pos_log = []

    for _ in range(n_steps):
        if not simulator.is_connected():
            break

        force_profile.step(sim_options.timestep)
        F = force_profile.force()
        force_log.append(F.copy())
        time_log.append(simulator.time())
        pos_log.append(np.array(simulator.get_base_position()))

        tau = np.zeros(N_JOINTS * N_LEGS)
        tau += nominal_position(simulator)
        tau += gravity_compensation(simulator)
        tau += apply_force_profile(simulator, force_profile)

        simulator.set_motor_targets(tau)
        simulator.step()

    simulator.close()

    force_log = np.array(force_log)
    time_log = np.array(time_log)
    plt.figure(figsize=(10, 5))
    plt.plot(time_log, force_log[:, 0], label='Fx')
    plt.plot(time_log, force_log[:, 1], label='Fy')
    plt.plot(time_log, force_log[:, 2], label='Fz')
    plt.xlabel('Time (s)')
    plt.ylabel('Force (N)')
    plt.title('Foot Force Profile Over Time')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    pos_log = np.array(pos_log)
    if len(pos_log) > 0:
        xy = pos_log[:, :2] - pos_log[0, :2]
        plt.figure(figsize=(6, 6))
        plt.plot(xy[:, 0], xy[:, 1])
        plt.scatter([0], [0], s=60)
        plt.scatter([xy[-1, 0]], [xy[-1, 1]], s=60)
        plt.axis('equal')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title('Top-Down Base Trajectory (start at [0, 0])')
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def _leg_target_offsets(leg_id: int) -> np.ndarray:
    z_offset = Z_OFFSET
    y_offset = 0.0838
    x_offset = X_OFFSET
    if leg_id == 0:
        return np.array([+x_offset, -y_offset, z_offset])
    elif leg_id == 1:
        return np.array([+x_offset, +y_offset, z_offset])
    elif leg_id == 2:
        return np.array([-x_offset, -y_offset, z_offset])
    else:
        return np.array([-x_offset, +y_offset, z_offset])


def nominal_position(simulator: "QuadSimulator") -> np.ndarray:
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
        tau[start:start + N_JOINTS] = tau_i
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
        tau[start:start + N_JOINTS] = tau_i
    return tau


def apply_force_profile(simulator: "QuadSimulator",
                        force_profile: FootForceProfile) -> np.ndarray:
    tau = np.zeros(N_JOINTS * N_LEGS)
    F = force_profile.force()
    foot_contacts = simulator.get_foot_contacts()
    front_sym = foot_contacts[0] and foot_contacts[1]
    rear_sym = foot_contacts[2] and foot_contacts[3]
    fx_allowed = front_sym and rear_sym
    F_use = F.copy()
    if not fx_allowed:
        F_use[0] = 0.0
    for leg_id in range(N_LEGS):
        if foot_contacts[leg_id]:
            J, _ = simulator.get_jacobian_and_position(leg_id)
            tau_i = J.T @ F_use
        else:
            tau_i = np.zeros(3)
        start = leg_id * N_JOINTS
        tau[start:start + N_JOINTS] = tau_i
    return tau


if __name__ == "__main__":
    quadruped_jump()
