"""
~/optimals_demo.py
Oct 2025; Jiri Blaha, Giulia Cortelazzo, Semih Zaman
École Polytechnique Fédérale de Lausanne,
CH-1015 Lausanne,
Switzerland
"""

import sys
import numpy as np

from env.simulation import QuadSimulator, SimulationOptions
from profiles import FootForceProfile
from optimals import OPTIMALS
import quadruped_jump as qj
from quadruped_jump import (
    nominal_position,
    gravity_compensation,
    apply_force_profile,
    virtual_model,
)

N_LEGS = 4
N_JOINTS = 3

KP_XY = 280.0
KP_Z = 1250.0
KD_FLIGHT_XY = 8.0
KD_FLIGHT_Z = 8.0
KD_STANCE_XY = 40.0
KD_STANCE_Z = 150.0

EXTRA_WAIT_T = 1.0


def run_single(jt: str):
    params = OPTIMALS[jt]
    qj.X_OFFSET = params["X_OFFSET"]
    qj.Y_OFFSET = params["Y_OFFSET"]
    qj.Z_OFFSET = params["Z_OFFSET"]
    K_VMC = params["K_VMC"]

    sim_options = SimulationOptions(
        on_rack=False, render=True, record_video=False, tracking_camera=True
    )
    simulator = QuadSimulator(sim_options)

    try:
        while True:
            simulator.reset()
            profile = FootForceProfile(
                f0=params["IMPULSE_F0"],
                f1=params["IDLE_F1"],
                Fx=params["FORCE_FX"],
                Fy=params["FORCE_FY"],
                Fz=params["FORCE_FZ"],
            )

            jump_T = profile.impulse_duration() + profile.idle_duration()
            n_imp = int(np.ceil(jump_T / sim_options.timestep))
            n_extra = int(np.ceil(EXTRA_WAIT_T / sim_options.timestep))

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
                tau += apply_force_profile(simulator, profile, jt)
                tau += gravity_compensation(simulator)

                if np.any(simulator.get_foot_contacts()):
                    tau += virtual_model(simulator, K_VMC)

                simulator.set_motor_targets(tau)
                simulator.step()

            started_wait = False
            waited = 0
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

                contacts = simulator.get_foot_contacts()
                if np.any(contacts):
                    tau += virtual_model(simulator, K_VMC)

                simulator.set_motor_targets(tau)
                simulator.step()

                all_on_ground = all(contacts)
                if all_on_ground and not started_wait:
                    started_wait = True
                    waited = 0
                if started_wait:
                    waited += 1
                    if waited >= n_extra:
                        break
    except KeyboardInterrupt:
        pass
    finally:
        simulator.close()


def run_hopping():
    jt = "hopping"
    params = OPTIMALS[jt]
    qj.X_OFFSET = params["X_OFFSET"]
    qj.Y_OFFSET = params["Y_OFFSET"]
    qj.Z_OFFSET = params["Z_OFFSET"]
    K_VMC = params["K_VMC"]

    sim_options = SimulationOptions(
        on_rack=False, render=True, record_video=False, tracking_camera=True
    )
    simulator = QuadSimulator(sim_options)

    try:
        while True:
            profile = FootForceProfile(
                f0=params["IMPULSE_F0"],
                f1=params["IDLE_F1"],
                Fx=params["FORCE_FX"],
                Fy=params["FORCE_FY"],
                Fz=params["FORCE_FZ"],
            )
            jump_T = profile.impulse_duration() + profile.idle_duration()
            total_T = jump_T + EXTRA_WAIT_T
            n_steps = int(np.ceil(total_T / sim_options.timestep))

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
                tau += apply_force_profile(simulator, profile, "forward")
                tau += gravity_compensation(simulator)

                if np.any(simulator.get_foot_contacts()):
                    tau += virtual_model(simulator, K_VMC)

                simulator.set_motor_targets(tau)
                simulator.step()
    except KeyboardInterrupt:
        pass
    finally:
        simulator.close()


if __name__ == "__main__":
    jt = sys.argv[1] if len(sys.argv) > 1 else "forward"
    if jt == "hopping":
        run_hopping()
    else:
        run_single(jt)
