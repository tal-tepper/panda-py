"""
Demonstrates the use of the PrimitiveTrajectory controller to move the
end-effector along a trajectory by setting position and orientation
step-by-step.
"""
import sys
from time import sleep
import numpy as np
import panda_py
from panda_py import controllers

if __name__ == '__main__':
    # Arguments
    npy_path = '/home/robot-lab/repos/tactile_panda/primitive_files/primitive_poses_hard_shell_delicate.npy'
    primitive_name = 'try1'
    hostname = '172.16.0.2'
    username = 'tamarlab'
    password = 'Panda468#'
    
    # Load trajectory from .npy file
    try:
        loaded_data = np.load(npy_path, allow_pickle=True).item()
        print(f'Available primitives in the file: {list(loaded_data.keys())}')
        trajectory_q = loaded_data[primitive_name]['q']
        trajectory_position = loaded_data[primitive_name]['position']
        trajectory_orientation = loaded_data[primitive_name]['orientation']
        print(f'Loaded {len(trajectory_q)} waypoints for primitive {primitive_name} from {npy_path}')
    except (IOError, KeyError) as e:
        print(f'Error: Could not find, read or parse trajectory file at {npy_path}: {e}')
        sys.exit(1)
    
    try:
        # Connect to the robot
        desk = panda_py.Desk(hostname, username, password)
        desk.activate_fci()
        panda = panda_py.Panda(hostname)
        panda.move_to_joint_position(trajectory_q[0])
        print(f'q0: {trajectory_q[0]}')
        print(f'q-: {trajectory_q[-1]}')
        sleep(2)
        
        # Configure the PrimitiveTrajectory controller
        ctrl = controllers.PrimitiveTrajectory(q_init=trajectory_q[0])
        
        # Store actual positions during execution
        actual_positions = []
        
        # Start the controller
        print("Starting PrimitiveTrajectory controller...")
        panda.start_controller(ctrl)
        
        # Loop through the trajectory
        with panda.create_context(frequency=1000) as ctx:
            for i in range(len(trajectory_q)):
                if not ctx.ok():
                    break
                
                # Get desired position and orientation for this step
                position = trajectory_position[i]
                orientation = trajectory_orientation[i]
                
                # Set control for this step
                ctrl.set_control(position, orientation)
                
                # Record actual position
                actual_positions.append(panda.get_position())
                
                # Optional: print progress
                if i % 100 == 0:
                    print(f'Waypoint {i}/{len(trajectory_q)}')
        
        panda.move_to_joint_position(trajectory_q[0])
        print("Trajectory finished. Stopping controller.")
        
    finally:
        desk.deactivate_fci()
        desk.release_control()
    
    # Convert to numpy array
    actual_positions = np.array(actual_positions)
    
    # Calculate and print trajectory tracking error statistics
    position_errors = np.linalg.norm(trajectory_position - actual_positions, axis=1)
    print(f"\nTrajectory Tracking Statistics:")
    print(f"  Mean position error: {np.mean(position_errors)*1000:.3f} mm")
    print(f"  Max position error: {np.max(position_errors)*1000:.3f} mm")
    print(f"  Std position error: {np.std(position_errors)*1000:.3f} mm")
