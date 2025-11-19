"""
Demonstrates the use of the HybridForceMotion controller to move the
end-effector along a trajectory loaded from a .npy file while applying a
constant force.
"""
import sys
from time import sleep
import numpy as np
import panda_py
from panda_py import controllers
import plotly.graph_objects as go

if __name__ == '__main__':
    # if len(sys.argv) < 5:
    #     raise RuntimeError(
    #         f'Usage: python {sys.argv[0]} <robot-hostname> <path-to-npy> <primitive-name> <force-z>'
    #     )

    # Arguments
    # hostname = sys.argv[1]
    npy_path = '/home/robot-lab/repos/tactile_panda/primitive_files/primitive_poses_hard_shell_delicate.npy'#sys.argv[1]
    primitive_name = 'try1'#'back_and_forth_1'#'circular_1'#'try'#'circles_1'#'roll'#'line'#sys.argv[2]
    force_z = float(-3.8)
    hostname = '172.16.0.2'
    username = 'tamarlab'
    password = 'Panda468#'
    
    # Load trajectory from .npy file
    try:
        loaded_data = np.load(npy_path, allow_pickle=True).item()
        print(f'Available primitives in the file: {list(loaded_data.keys())}')
        trajectory_q = loaded_data[primitive_name]['q']
        trajectory_dq = loaded_data[primitive_name]['dq']
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
        print(f'q0:{trajectory_q[0]}')
        print(f'q-:{trajectory_q[-1]}')
        sleep(2)
        # Configure the HybridForceMotion controller
        ctrl = controllers.HybridForceMotion()

        # Define the desired force
        force = np.zeros(6)
        force[0] = force_z
        force[1] = force_z
        force[2] = force_z
        
        # Store actual joint positions during execution
        actual_trajectory_q = []

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
                
                # Record actual joint position
                actual_trajectory_q.append(panda.q.copy())
                # print(f'q:{panda.q}')
                # Optional: print progress
                if i % 100 == 0:
                    print(f'Waypoint {i}/{len(trajectory_q)}')
        panda.move_to_joint_position(trajectory_q[0])
        print("Trajectory finished. Stopping controller.")
    finally:
        desk.deactivate_fci()
        desk.release_control()
    
    

  
    print("Trajectory finished. Stopping controller.")
    # panda.stop_controller()
    
    # Convert actual trajectory to numpy array
    actual_trajectory_q = np.array(actual_trajectory_q)
    
    # Compute forward kinematics for both trajectories to get Cartesian positions
    print("Computing forward kinematics for visualization...")
    
    # Get end-effector positions for planned trajectory
    planned_positions = []
    for q in trajectory_q[:len(actual_trajectory_q)]:
        pose = panda.get_pose(q)  # Get 4x4 transformation matrix
        planned_positions.append(pose[:3, 3])  # Extract position (x, y, z)
    planned_positions = np.array(planned_positions)
    
    # Get end-effector positions for actual trajectory
    actual_positions = []
    for q in actual_trajectory_q:
        pose = panda.get_pose(q)  # Get 4x4 transformation matrix
        actual_positions.append(pose[:3, 3])  # Extract position (x, y, z)
    actual_positions = np.array(actual_positions)
    
    # Create 3D plot with Plotly
    print("Creating 3D visualization...")
    fig = go.Figure()
    
    # Add planned trajectory (green)
    fig.add_trace(go.Scatter3d(
        x=planned_positions[:, 0],
        y=planned_positions[:, 1],
        z=planned_positions[:, 2],
        mode='lines',
        name='Planned Trajectory',
        line=dict(color='green', width=4)
    ))
    
    # Add actual trajectory (red)
    fig.add_trace(go.Scatter3d(
        x=actual_positions[:, 0],
        y=actual_positions[:, 1],
        z=actual_positions[:, 2],
        mode='lines',
        name='Actual Trajectory',
        line=dict(color='red', width=4)
    ))
    
    # Add start point
    fig.add_trace(go.Scatter3d(
        x=[planned_positions[0, 0]],
        y=[planned_positions[0, 1]],
        z=[planned_positions[0, 2]],
        mode='markers',
        name='Start',
        marker=dict(size=8, color='blue', symbol='diamond')
    ))
    
    # Add end point
    fig.add_trace(go.Scatter3d(
        x=[planned_positions[-1, 0]],
        y=[planned_positions[-1, 1]],
        z=[planned_positions[-1, 2]],
        mode='markers',
        name='End',
        marker=dict(size=8, color='orange', symbol='diamond')
    ))
    
    # Update layout
    fig.update_layout(
        title=f'Trajectory Comparison - {primitive_name}<br>Applied Force Z: {force_z} N',
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data'
        ),
        width=1000,
        height=800,
        showlegend=True
    )
    
    # Show the plot
    fig.show()
    
    # Calculate and print trajectory tracking error statistics
    position_errors = np.linalg.norm(planned_positions - actual_positions, axis=1)
    print(f"\nTrajectory Tracking Statistics:")
    print(f"  Mean position error: {np.mean(position_errors)*1000:.3f} mm")
    print(f"  Max position error: {np.max(position_errors)*1000:.3f} mm")
    print(f"  Std position error: {np.std(position_errors)*1000:.3f} mm")
