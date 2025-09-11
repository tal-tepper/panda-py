"""
Demonstrates the use of the HybridForceMotion controller to move the
end-effector along a trajectory loaded from a CSV file while applying a
constant force.
"""
import sys
import numpy as np
import panda_py
from panda_py import controllers

if __name__ == '__main__':
    if len(sys.argv) < 4:
        raise RuntimeError(
            f'Usage: python {sys.argv[0]} <robot-hostname> <path-to-csv> <force-z>'
        )

    # Arguments
    hostname = sys.argv[1]
    csv_path = sys.argv[2]
    force_z = float(sys.argv[3])

    # --- Create a dummy trajectory file for demonstration ---
    try:
        panda_conn = panda_py.Panda(hostname)
        panda_conn.move_to_start()
        start_pose = panda_conn.get_pose()
        
        # 10cm downwards trajectory
        trajectory_poses = []
        for i in range(100):
            pose = start_pose.copy()
            pose[2, 3] -= 0.1 * (i / 99.0)
            trajectory_poses.append(pose)

        # Convert to position and quaternion
        trajectory_data = []
        for pose in trajectory_poses:
            pos = pose[:3, 3]
            quat = panda_py.get_orientation_from_matrix(pose)
            trajectory_data.append(np.concatenate([pos, quat]))

        np.savetxt(csv_path, trajectory_data, delimiter=',')
        print(f'Created dummy trajectory at {csv_path}')
    except Exception as e:
        print(f'Could not create dummy trajectory file: {e}')
        print('Please ensure the robot is connected and the path is valid.')

    # ------------------------------------------------------

    # Connect to the robot
    panda = panda_py.Panda(hostname)
    panda.move_to_start()

    # Load trajectory from CSV
    try:
        trajectory = np.loadtxt(csv_path, delimiter=',')
        print(f'Loaded {len(trajectory)} waypoints from {csv_path}')
    except IOError:
        print(f'Error: Could not find or read trajectory file at {csv_path}')
        sys.exit(1)

    # Configure the HybridForceMotion controller
    # Enable force control along the Z-axis
    selection = np.array([0, 0, 1, 0, 0, 0], dtype=bool)
    ctrl = controllers.HybridForceMotion(selection=selection)

    # Define the desired force
    force = np.zeros(6)
    force[2] = force_z

    # Start the controller
    print("Starting HybridForceMotion controller...")
    panda.start_controller(ctrl)

    # Loop through the trajectory
    with panda.create_context(frequency=1000) as ctx:
        for i, waypoint in enumerate(trajectory):
            if not ctx.ok():
                break
            
            position = waypoint[:3]
            orientation = waypoint[3:]
            
            ctrl.set_control(position, orientation, force)

            # Optional: print progress
            if i % 100 == 0:
                print(f'Waypoint {i}/{len(trajectory)}')

    print("Trajectory finished. Stopping controller.")
    panda.stop_controller()