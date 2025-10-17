"""
~/quadruped_jump.py
Oct 2025; Jiri Blaha, Giulia Cortelazzo, Semih Zaman
École Polytechnique Fédérale de Lausanne,
CH-1015 Lausanne,
Switzerland
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

from env.simulation import QuadSimulator, SimulationOptions
from profiles import FootForceProfile
from jump_params import JUMP_PARAMS

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

# Sign patterns encode the leg frames’ lateral (Y) and sagittal (X)
# directions for each leg, ordered as FR=0, FL=1, RR=2, RL=3.
# This lets us mirror targets/forces across the body without hardcoding.
LEG_SIGN_X = np.array([+1, +1, -1, -1], dtype=float)
LEG_SIGN_Y = np.array([-1, +1, -1, +1], dtype=float)

# Init nominal leg position.
# Overwritten inside quadruped_jump().
X_OFFSET = 0.0
Y_OFFSET = 0.0
Z_OFFSET = 0.0


def _leg_target_offsets(leg_id: int) -> np.ndarray:
    """
    Compute the nominal Cartesian target offset for a given leg.

    The per-leg target position is derived from the global offsets
    (X_OFFSET, Y_OFFSET, Z_OFFSET) and the sign pattern for that leg.

    Args:
        leg_id: Integer in [0..3] (FR=0, FL=1, RR=2, RL=3).

    Returns:
        (3,) np.ndarray [x, y, z] nominal target offset for this leg.
    """
    return np.array(
        [
            LEG_SIGN_X[leg_id] * X_OFFSET,  # mirror X for front/back
            LEG_SIGN_Y[leg_id] * Y_OFFSET,  # mirror Y for left/right
            Z_OFFSET,                       # same Z for all
        ],
        dtype=float,
    )


def nominal_position(
    simulator: "QuadSimulator",
    KP_XY: float,
    KP_Z: float,
    KD_FLIGHT_XY: float,
    KD_FLIGHT_Z: float,
    KD_STANCE_XY: float,
    KD_STANCE_Z: float,
) -> np.ndarray:
    """
    Cartesian PD controller that keeps each foot near a nominal pose.

    Args:
        simulator: Physics/simulation wrapper.
        KP_XY, KP_Z: Proportional gains for (x,y) and z.
        KD_FLIGHT_XY, KD_FLIGHT_Z: Derivative gains in flight.
        KD_STANCE_XY, KD_STANCE_Z: Derivative gains in stance.

    Returns:
        (N_JOINTS*N_LEGS,) np.ndarray of joint torques from NP.
    """
    tau = np.zeros(N_JOINTS * N_LEGS)
    contacts = simulator.get_foot_contacts()  # boolean array per leg
    Kp = np.diag([KP_XY, KP_XY, KP_Z])
    Kd_flight = np.diag([KD_FLIGHT_XY, KD_FLIGHT_XY, KD_FLIGHT_Z])
    Kd_stance = np.diag([KD_STANCE_XY, KD_STANCE_XY, KD_STANCE_Z])
    for leg_id in range(N_LEGS):
        dq = simulator.get_motor_velocities(leg_id)
        J, p = simulator.get_jacobian_and_position(leg_id)
        v = J @ dq
        pd = _leg_target_offsets(leg_id)
        vd = np.zeros(3)
        Kd = Kd_stance if contacts[leg_id] else Kd_flight
        F = Kp @ (pd - p) + Kd @ (vd - v)  # Cartesian force
        tau_i = J.T @ F                    # map to joint torques
        s = leg_id * N_JOINTS
        tau[s: s + N_JOINTS] = tau_i
    return tau


def virtual_model(simulator: "QuadSimulator", k_vmc: float) -> np.ndarray:
    """
    VMC to stabilize base roll/pitch via virtual springs.

    Args:
        simulator: Physics/simulation wrapper.
        k_vmc: Scalar VMC gain.

    Returns:
        (N_JOINTS*N_LEGS,) np.ndarray of joint torques from VMC.
    """
    tau = np.zeros(N_JOINTS * N_LEGS)
    contacts = simulator.get_foot_contacts()
    if not np.any(contacts):
        # No contact -> no VMC
        return tau
    R = simulator.get_base_orientation_matrix()  # rotation matrix from sim
    P_body = np.array([[1, 1, -1, -1], [-1, 1, -1, 1], [0, 0, 0, 0]])
    z = (R @ P_body)[2, :]
    F_world = np.zeros((3, N_LEGS))
    F_world[2, :] = k_vmc * z
    F_leg = R.T @ F_world
    for leg_id in range(N_LEGS):
        if not contacts[leg_id]:
            # No contact -> no VMC
            continue
        J, _ = simulator.get_jacobian_and_position(leg_id)
        tau_i = J.T @ F_leg[:, leg_id]
        s = leg_id * N_JOINTS
        tau[s: s + N_JOINTS] = tau_i
    return tau


def gravity_compensation(simulator: "QuadSimulator") -> np.ndarray:
    """
    Distribute gravity compensation equally.

    Args:
        simulator: Physics/simulation wrapper.

    Returns:
        (N_JOINTS*N_LEGS,) np.ndarray of joint torques from GC.
    """
    tau = np.zeros(N_JOINTS * N_LEGS)
    m_total = simulator.get_mass()
    g = 9.81
    Fg = np.array([0.0, 0.0, -(m_total / N_LEGS) * g])
    for leg_id in range(N_LEGS):
        J, _ = simulator.get_jacobian_and_position(leg_id)
        tau_i = J.T @ Fg
        s = leg_id * N_JOINTS
        tau[s: s + N_JOINTS] = tau_i
    return tau


def per_leg_force(F_base: np.ndarray, leg_id: int, jt: str) -> np.ndarray:
    """
    Per-leg force shaping for twist jumps.

    Args:
        F_base: (3,) np.ndarray, force profile from current step.
        leg_id: Integer in [0..3] (FR=0, FL=1, RR=2, RL=3).
        jt: Jump type (forward, lateral_left, etc.).

    Returns:
        (3,) np.ndarray of forces per leg from PLF.
    """
    Fx, Fy, Fz = F_base
    sx, sy = LEG_SIGN_X[leg_id], LEG_SIGN_Y[leg_id]
    if jt in ("twist_ccw", "twist_cw"):
        s = +1.0 if jt == "twist_ccw" else -1.0
        Fx_i = s * sy * Fx
        Fy_i = s * sx * Fy
        return np.array([Fx_i, Fy_i, Fz], dtype=float)
    return F_base


def apply_force_profile(
    simulator: "QuadSimulator", profile: FootForceProfile, jt: str
) -> np.ndarray:
    """
    Apply force profile and compute torques.

    Args:
        simulator: Physics/simulation wrapper.
        profile: FootForceProfile from ~/profiles.py.
        jt: Jump type (forward, lateral_left, etc.).

    Returns:
        (N_JOINTS*N_LEGS,) np.ndarray of joint torques AFP.
    """
    tau = np.zeros(N_JOINTS * N_LEGS)
    F = profile.force()
    contacts = simulator.get_foot_contacts()
    front_sym = contacts[0] and contacts[1]
    rear_sym = contacts[2] and contacts[3]
    fx_allowed = front_sym and rear_sym    # do we have full support?
    for leg_id in range(N_LEGS):
        if contacts[leg_id]:
            F_i = per_leg_force(F, leg_id, jt).copy()
            if not fx_allowed:
                F_i[0] = 0.0
            J, _ = simulator.get_jacobian_and_position(leg_id)
            tau_i = J.T @ F_i
        else:
            tau_i = np.zeros(3)
        s = leg_id * N_JOINTS
        tau[s: s + N_JOINTS] = tau_i
    return tau


def _compute_tau(
    simulator: "QuadSimulator",
    profile: FootForceProfile,
    jt: str,
    k_vmc: float,
) -> np.ndarray:
    """
    Assemble total joint torques from components.

    Args:
        simulator: Physics/simulation wrapper.
        profile: FootForceProfile from ~/profiles.py.
        jt: Jump type (forward, lateral_left, etc.).
        k_vmc: Scalar VMC gain.

    Returns:
        (N_JOINTS*N_LEGS,) np.ndarray of total torques for current step.
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
    tau += apply_force_profile(simulator, profile, jt)
    tau += gravity_compensation(simulator)
    if np.any(simulator.get_foot_contacts()):
        tau += virtual_model(simulator, k_vmc)
    return tau


def quadruped_jump(jt: str) -> None:
    """
    Entry point. Run a jumping session for the specified jump type.

    Args:
        jt: Jump type (forward, lateral_left, etc.).

    Returns:
        None.
    """
    # Pull jump parameters (except offsets).
    params = JUMP_PARAMS[jt]
    IMPULSE_F0 = params["IMPULSE_F0"]
    IDLE_F1 = params["IDLE_F1"]
    FORCE_FX = params["FORCE_FX"]
    FORCE_FY = params["FORCE_FY"]
    FORCE_FZ = params["FORCE_FZ"]
    N_JUMPS = params["N_JUMPS"]
    K_VMC = params["K_VMC"]

    # Update nominal offsets for this jt.
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

    profile = FootForceProfile(
        f0=IMPULSE_F0, f1=IDLE_F1, Fx=FORCE_FX, Fy=FORCE_FY, Fz=FORCE_FZ
    )

    jump_T = profile.impulse_duration() + profile.idle_duration()
    n_steps = int(N_JUMPS * jump_T / sim_options.timestep)

    force_log, time_log, pos_log, tau_log = [], [], [], []  # for plots

    for _ in range(n_steps):
        if not simulator.is_connected():
            # No sim -> exit
            break

        profile.step(sim_options.timestep)
        F = profile.force()
        force_log.append(F.copy())
        time_log.append(simulator.time())
        pos_log.append(np.array(simulator.get_base_position()))

        tau = _compute_tau(simulator, profile, jt, K_VMC)
        tau_log.append(tau.copy())

        simulator.set_motor_targets(tau)
        simulator.step()

    simulator.close()

    force_log = np.array(force_log)
    tau_log = np.array(tau_log)
    time_log = np.array(time_log)

    # === Force profile plot ===
    if len(force_log) > 0:
        plt.figure()
        plt.plot(time_log, force_log[:, 0], label="Fx")
        plt.plot(time_log, force_log[:, 1], label="Fy")
        plt.plot(time_log, force_log[:, 2], label="Fz")
        plt.xlabel("Time (s)")
        plt.ylabel("Force (N)")
        plt.title("Foot Force Profile Over Time")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"forces_{jt}.pdf", format="pdf")
        plt.show()

    # === Torques plot ===
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
        plt.savefig(f"torques_{jt}.pdf", format="pdf")
        plt.show()

    # === Trajectory plot===
    pos_log = np.array(pos_log)
    if len(pos_log) > 0:
        # Subtract initial XY for relative plot.
        xy = pos_log[:, :2] - pos_log[0, :2]
        plt.figure()
        plt.plot(xy[:, 0], xy[:, 1])
        plt.scatter([0], [0], s=60)
        plt.scatter([xy[-1, 0]], [xy[-1, 1]], s=60)
        plt.axis("equal")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.title("Top-Down Base Trajectory (start at [0, 0])")
        plt.grid(True)
        plt.savefig(f"traj_{jt}.pdf", format="pdf")
        plt.show()


if __name__ == "__main__":
    # Minimal CLI:
    #   python quadruped_jump.py <jt> [--compare-vmc]
    # If --compare-vmc is passed, the script runs the same jump once with VMC
    # disabled (K_VMC=0) and once with the configured K_VMC (jump_params.py).
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    compare = "--compare-vmc" in sys.argv[1:]
    jt = args[0] if len(args) > 0 else "forward"

    if not compare:
        quadruped_jump(jt)
    else:
        def run_once(jt, k_vmc):
            """
            Run a single simulation pass but with vmc off or on.

            Args:
                jt: Jump type (forward, lateral_left, etc.).
                k_vmc: Scalar VMC gain.

            Returns:
                t, r, p: np.ndarrays of time, roll, and pitch.
            """
            params = JUMP_PARAMS[jt]
            global X_OFFSET, Y_OFFSET, Z_OFFSET
            X_OFFSET = params["X_OFFSET"]
            Y_OFFSET = params["Y_OFFSET"]
            Z_OFFSET = params["Z_OFFSET"]

            sim_options = SimulationOptions(
                on_rack=False,
                render=True,
                record_video=False,
                tracking_camera=True
            )
            simulator = QuadSimulator(sim_options)

            profile = FootForceProfile(
                f0=params["IMPULSE_F0"],
                f1=params["IDLE_F1"],
                Fx=params["FORCE_FX"],
                Fy=params["FORCE_FY"],
                Fz=params["FORCE_FZ"],
            )
            jump_T = profile.impulse_duration() + profile.idle_duration()
            n_steps = int(params["N_JUMPS"] * jump_T / sim_options.timestep)

            t, r, p = [], [], []
            for _ in range(n_steps):
                if not simulator.is_connected():
                    break
                profile.step(sim_options.timestep)
                tau = _compute_tau(simulator, profile, jt, k_vmc)
                simulator.set_motor_targets(tau)
                simulator.step()

                R = simulator.get_base_orientation_matrix()
                roll = np.arctan2(R[2, 1], R[2, 2])  # extract roll
                pitch = np.arctan2(
                    -R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)
                )  # extract pitch
                t.append(simulator.time())
                r.append(roll)
                p.append(pitch)

            simulator.close()
            return np.array(t), np.array(r), np.array(p)

        t0, r0, p0 = run_once(jt, 0.0)  # run w/ VMC off
        t1, r1, p1 = run_once(jt, JUMP_PARAMS[jt]["K_VMC"])  # run w/ VMC on

        # === VMC comparison plot ===
        plt.figure()
        plt.plot(t0, r0, "-", color="black", label="roll (VMC off)")
        plt.plot(t0, p0, "-", color="red", label="pitch (VMC off)")
        plt.plot(t1, r1, "-", color="green", label="roll (VMC on)")
        plt.plot(t1, p1, "-", color="blue", label="pitch (VMC on)")
        plt.title(f"Base roll/pitch with and without VMC [{jt}]")
        plt.xlabel("time (s)")
        plt.ylabel("angle (rad)")
        plt.legend()
        plt.savefig(f"vmc_{jt}.pdf", format="pdf")
        plt.show()
