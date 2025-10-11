import numpy as np
from env.simulation import QuadSimulator, SimulationOptions

from profiles import FootForceProfile
import matplotlib.pyplot as plt
import pybullet
N_LEGS = 4
N_JOINTS = 3

def quadruped_jump():
    # Initialize simulation
    # Feel free to change these options! (except for control_mode and timestep)
    sim_options = SimulationOptions(
        on_rack=False,  # Whether to suspend the robot in the air (helpful for debugging)
        render=True,  # Whether to use the GUI visualizer (slower than running in the background)
        record_video=False,  # Whether to record a video to file (needs render=True)
        tracking_camera=True,  # Whether the camera follows the robot (instead of free)
    )
    simulator = QuadSimulator(sim_options)

    # Determine number of jumps to simulate
    n_jumps = 10  # Feel free to change this number
    jump_duration = 5.0  # TODO: determine how long a jump takes
    n_steps = int(n_jumps * jump_duration / sim_options.timestep)

    # TODO: set parameters for the foot force profile here
    force_profile = FootForceProfile(f0=10.0, f1= 0.1, Fx=0.0, Fy=0.0, Fz=200.0)
    force_log = []
    time_log = []
    n_steps = 100000

    for _ in range(n_steps):
        # If the simulator is closed, stop the loop
        if not simulator.is_connected():
            break

        # Step the oscillator
        force_profile.step(sim_options.timestep)

        # Log force and time
        F = force_profile.force()
        force_log.append(F.copy())
        time_log.append(simulator.time())

        # Compute torques as motor targets
        # The convention is as follows:
        # - A 1D array where the torques for the 3 motors follow each other for each leg
        # - The first 3 elements are the hip, thigh, calf torques for the FR leg.
        # - The order of the legs is FR, FL, RR, RL (front/rear,right/left)
        # - The resulting torque array is therefore structured as follows:
        # [FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf, RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf]
        tau = np.zeros(N_JOINTS * N_LEGS)

        # TODO: implement the functions below, and add potential controller parameters as function parameters here
        tau += nominal_position(simulator)
        tau += apply_force_profile(simulator, force_profile)
        tau += gravity_compensation(simulator)
        tau += virtual_model(simulator)

        # If touching the ground, add virtual model
        #foot_contacts = simulator.get_foot_contacts()
        #on_ground = np.any(foot_contacts)   # TODO: how do we know we're on the ground?
        #if on_ground:
        #    tau += virtual_model(simulator)
    
        # Set the motor commands and step the simulation
        simulator.set_motor_targets(tau)
        simulator.step()

    # Close the simulation
    simulator.close()

    # --- Plot the force profile after simulation ends
    force_log = np.array(force_log)
    time_log = np.array(time_log)

    plt.figure(figsize=(10, 5))
    plt.plot(time_log, force_log[:, 0], label='Fx')
    plt.plot(time_log, force_log[:, 1], label='Fy')
    plt.plot(time_log, force_log[:, 2], label='Fz')
    plt.xlabel('Time (s)')
    plt.ylabel('Force (N)')
    plt.title('Foot Force Profile Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # OPTIONAL: add additional functions here (e.g., plotting)


def nominal_position(
    simulator: QuadSimulator,
    # OPTIONAL: add potential controller parameters here (e.g., gains)
) -> np.ndarray:
    # All motor torques are in a single array
    tau = np.zeros(N_JOINTS * N_LEGS)
    for leg_id in range(N_LEGS):

        # TODO: compute nominal position torques for leg_id
        q = simulator.get_motor_angles(leg_id)
        dq = simulator.get_motor_velocities(leg_id)
        J, p = simulator.get_jacobian_and_position(leg_id)

        #p = simulator.get_world_foot_position(leg_id)
        v = J @ dq 

        y_offset = 0.0838  # adjust based on hip offset
        z_offset = -0.2  # standing height
        x_offset = 0
        if leg_id == 0:  # FR
            pd = np.array([x_offset, -y_offset, z_offset])
        elif leg_id == 1:  # FL
            pd = np.array([x_offset, y_offset, z_offset])
        elif leg_id == 2:  # RR
            pd = np.array([-x_offset, -y_offset, z_offset])
        elif leg_id == 3:  # RL
            pd = np.array([-x_offset, y_offset, z_offset])

        vd = np.zeros(3)
        tau_i = np.zeros(3)

        Kp = np.diag([300, 300, 3000])
        Kd = np.diag([10, 10, 10])

        F = Kp @ (pd - p) + Kd @ (vd - v)
        tau_i = J.T @ F

        # Store in torques array
        tau[leg_id * N_JOINTS : leg_id * N_JOINTS + N_JOINTS] = tau_i
    return tau


def virtual_model(
    simulator: QuadSimulator,
    # OPTIONAL: add potential controller parameters here (e.g., gains)
) -> np.ndarray:
    # All motor torques are in a single array
    tau = np.zeros(N_JOINTS * N_LEGS)
    R = simulator.get_base_orientation_matrix()
    P_local = np.array([[1, 1, -1, -1], [-1, 1, -1, 1], [0, 0, 0, 0]])
    P_world = R @ P_local
    k_vmc = 200
    foot_contacts = simulator.get_foot_contacts()
    for leg_id in range(N_LEGS):
        if foot_contacts[leg_id] :
            # TODO: compute virtual model torques for leg_id
            pz = P_world[2, leg_id] # z-component only
            F_vmc = np.array([0.0, 0.0, k_vmc * pz])
            J, _ = simulator.get_jacobian_and_position(leg_id)
            tau_i = J.T @ F_vmc

            # Store in torques array
            tau[leg_id * N_JOINTS : leg_id * N_JOINTS + N_JOINTS] = tau_i
        else:
            tau[leg_id * N_JOINTS : leg_id * N_JOINTS + N_JOINTS] = np.zeros(3) 
    return tau


def gravity_compensation(
    simulator: QuadSimulator,
    # OPTIONAL: add potential controller parameters here (e.g., gains)
) -> np.ndarray:
    # All motor torques are in a single array
    tau = np.zeros(N_JOINTS * N_LEGS)
    m = simulator.get_mass() / 4
    g = 9.81
    Fg = np.array([0, 0, -m * g])
    for leg_id in range(N_LEGS):

        # TODO: compute gravity compensation torques for leg_id
        J, _ = simulator.get_jacobian_and_position(leg_id)
        tau_i = J.T @ Fg

        # Store in torques array
        tau[leg_id * N_JOINTS : leg_id * N_JOINTS + N_JOINTS] = tau_i

    return tau


def apply_force_profile(
    simulator: QuadSimulator,
    force_profile: FootForceProfile,
    # OPTIONAL: add potential controller parameters here (e.g., gains)
) -> np.ndarray:
    # All motor torques are in a single array
    tau = np.zeros(N_JOINTS * N_LEGS)
    F = force_profile.force()
    for leg_id in range(N_LEGS):

        # TODO: compute force profile torques for leg_id
        J, _ = simulator.get_jacobian_and_position(leg_id)
        tau_i = J.T @ F

        # Store in torques array
        tau[leg_id * N_JOINTS : leg_id * N_JOINTS + N_JOINTS] = tau_i

    return tau



def test_nominal_position_standing():
    sim_options = SimulationOptions(
        on_rack=False,
        render=True,
        record_video=False,
        tracking_camera=True,
    )
    simulator = QuadSimulator(sim_options)

    run_time = 5.0  # seconds
    n_steps = int(run_time / sim_options.timestep)

    for _ in range(n_steps):
        if not simulator.is_connected():
            break

        tau = np.zeros(N_JOINTS * N_LEGS)
        tau += nominal_position(simulator)
        tau += gravity_compensation(simulator)

        simulator.set_motor_targets(tau)
        simulator.step()
    simulator.close()





if __name__ == "__main__":
    quadruped_jump()
    #test_nominal_position_standing()
