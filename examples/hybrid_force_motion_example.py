"""
Demonstrates the use of the HybridForceMotion controller to move the
end-effector along a trajectory loaded from a .npy file while applying a
constant force.
"""
import sys
import numpy as np
import panda_py
from panda_py import controllers

if __name__ == '__main__':
    if len(sys.argv) < 5:
        raise RuntimeError(
            f'Usage: python {sys.argv[0]} <robot-hostname> <path-to-npy> <primitive-name> <force-z>'
        )

    # Arguments
    hostname = sys.argv[1]
    npy_path = sys.argv[2]
    primitive_name = sys.argv[3]
    force_z = float(sys.argv[4])

    # Connect to the robot
    panda = panda_py.Panda(hostname)
    panda.move_to_start()

    # Load trajectory from .npy file
    try:
        loaded_data = np.load(npy_path, allow_pickle=True).item()
        trajectory_q = loaded_data[primitive_name]['q']
        trajectory_dq = loaded_data[primitive_name]['dq']
        print(f'Loaded {len(trajectory_q)} waypoints for primitive '{primitive_name}' from {npy_path}')
    except (IOError, KeyError) as e:
        print(f'Error: Could not find, read or parse trajectory file at {npy_path}: {e}')
        sys.exit(1)

    # Configure the HybridForceMotion controller
    ctrl = controllers.HybridForceMotion()

    # Define the desired force
    force = np.zeros(6)
    force[2] = force_z

    # Start the controller
    print("Starting HybridForceMotion controller...")
    panda.start_controller(ctrl)

    # Loop through the trajectory
    with panda.create_context(frequency=1000) as ctx:
        for i in range(len(trajectory_q)):
            if not ctx.ok():
                break
            
            q_d = trajectory_q[i]
            dq_d = trajectory_dq[i]
            ctrl.set_control(q_d, force, dq_d)

            # Optional: print progress
            if i % 100 == 0:
                print(f'Waypoint {i}/{len(trajectory_q)}')

    print("Trajectory finished. Stopping controller.")
    panda.stop_controller()
