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
import datetime

def check_boundaries(position, lower_bounds, upper_bounds):
    """Check if the given position is within the specified joint boundaries."""
    for i in range(len(position)):
        if position[i] < lower_bounds[i] or position[i] > upper_bounds[i]:
            return False
    return True

if __name__ == '__main__':
    # if len(sys.argv) < 5:
    #     raise RuntimeError(
    #         f'Usage: python {sys.argv[0]} <robot-hostname> <path-to-npy> <primitive-name> <force-z>'
    #     )

    # Arguments
    # hostname = sys.argv[1]
    npy_path = '/home/robot-lab/repos/tactile_panda/primitive_files/primitive_poses_soft_shell_delicate.npy'#sys.argv[1]
    # primitive_name = 'try1'#'back_and_forth_1'#'circular_1'#'try'#'circles_1'#'roll'#'line'#sys.argv[2]
    force_z = float(-3.8)
    hostname = '172.16.0.2'
    username = 'tamarlab'
    password = 'Panda468#'
    
    # Load trajectory from .npy file
    try:
        loaded_data = np.load(npy_path, allow_pickle=True).item()
        # ['jab', 'feel', 'sense', 'back_and_forth', 'sides', 'back_and_forth_2']
        primitive_name = list(loaded_data.keys())[2] # 2 is problematic
        print(f'Available primitives in the file: {list(loaded_data.keys())}')
        trajectory_q = loaded_data[primitive_name]['q']
        trajectory_dq = loaded_data[primitive_name]['dq']
        print(f'Loaded {len(trajectory_q)} waypoints for primitive {primitive_name} from {npy_path}')
    except (IOError, KeyError) as e:
        print(f'Error: Could not find, read or parse trajectory file at {npy_path}: {e}')
        sys.exit(1)
    # Define the desired force
    force = np.zeros(6)
    force[0] = force_z
    force[1] = force_z
    force[2] = force_z
    
    # Store actual joint positions and velocities during execution
    actual_trajectory_q = []
    actual_joint_positions = []
    actual_joint_velocities = []
    
    error_occurred = False
    error_message = ""
    
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
        
        first_position = panda.get_position()
        lower_bounds = first_position - 0.06
        upper_bounds = first_position + 0.06
        

        # Start the controller
        print("Starting HybridForceMotion controller...")
        panda.start_controller(ctrl)

        try:
            # Loop through the trajectory
            with panda.create_context(frequency=1000) as ctx:
                for i in range(len(trajectory_q)):
                    if not ctx.ok():
                        break
                    
                    q_d = trajectory_q[i]
                    dq_d = trajectory_dq[i]
                    ctrl.set_control(q_d, force, dq_d)
                    # panda.update_robot_state()
                    # Record actual joint position and state
                    state = panda.get_state()
                    new_pos = panda.get_position()
                    actual_trajectory_q.append(new_pos)
                    actual_joint_positions.append(state.q)
                    actual_joint_velocities.append(state.dq)
                    if not check_boundaries(new_pos, lower_bounds, upper_bounds):
                        print(f"Position out of bounds at waypoint {i}: {new_pos}")
                        raise RuntimeError("Joint position exceeded safety boundaries. Stopping execution.")
                    # print(f'q:{panda.q}')
                    # Optional: print progress
                    # if i % 100 == 0:
                    now = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    print(f'i:{i} time:{now} new_pos: {new_pos}')#Waypoint {i}/{len(trajectory_q)}
        except Exception as e:
            error_occurred = True
            error_message = str(e)
            print(f"\n*** Error during trajectory execution: {e} ***")
            print(f"*** Collected {len(actual_trajectory_q)} data points before error ***\n")
        
        try:
            panda.move_to_joint_position(trajectory_q[0])
            print("Returned to start position.")
        except Exception as e:
            print(f"Warning: Could not return to start position: {e}")
            
    finally:
        try:
            desk.deactivate_fci()
            desk.release_control()
        except:
            pass
    
    
    # Check if we have any data to visualize
    if len(actual_trajectory_q) == 0:
        print("No data collected. Exiting without visualization.")
        sys.exit(1)
    
    if error_occurred:
        print(f"\nVisualization will show partial trajectory (up to error point)")
        print(f"Error was: {error_message}\n")
    else:
        print("Trajectory finished successfully.")
    
    # Convert actual trajectory to numpy array
    actual_positions = np.array(actual_trajectory_q)
    
    # Compute forward kinematics for both trajectories to get Cartesian positions
    print("Computing forward kinematics for visualization...")
    
    # Get end-effector positions for planned trajectory (truncate to match actual data length)
    all_planned_positions = np.array(loaded_data[primitive_name]['position'])
    planned_positions = all_planned_positions[:len(actual_positions)]
    
    # Truncate planned trajectory arrays to match actual data length
    trajectory_q = trajectory_q[:len(actual_positions)]
    trajectory_dq = trajectory_dq[:len(actual_positions)]
    
       
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
    
    # Create time vs position plot with subplots for x, y, z
    print("Creating time series visualization...")
    
    # Create time array (assuming 1000 Hz control frequency)
    dt = 0.001  # 1 ms per sample
    time_planned = np.arange(len(planned_positions)) * dt
    time_actual = np.arange(len(actual_positions)) * dt
    
    # Create figure with 3 subplots
    from plotly.subplots import make_subplots
    
    fig2 = make_subplots(
        rows=3, cols=1,
        subplot_titles=('X Position vs Time', 'Y Position vs Time', 'Z Position vs Time'),
        vertical_spacing=0.08,
        x_title='Time (s)',
    )
    
    # X position subplot
    fig2.add_trace(
        go.Scatter(x=time_planned, y=planned_positions[:, 0], 
                   mode='lines', name='Planned X', 
                   line=dict(color='green', width=2)),
        row=1, col=1
    )
    fig2.add_trace(
        go.Scatter(x=time_actual, y=actual_positions[:, 0], 
                   mode='lines', name='Actual X', 
                   line=dict(color='red', width=2)),
        row=1, col=1
    )
    
    # Y position subplot
    fig2.add_trace(
        go.Scatter(x=time_planned, y=planned_positions[:, 1], 
                   mode='lines', name='Planned Y', 
                   line=dict(color='green', width=2)),
        row=2, col=1
    )
    fig2.add_trace(
        go.Scatter(x=time_actual, y=actual_positions[:, 1], 
                   mode='lines', name='Actual Y', 
                   line=dict(color='red', width=2)),
        row=2, col=1
    )
    
    # Z position subplot
    fig2.add_trace(
        go.Scatter(x=time_planned, y=planned_positions[:, 2], 
                   mode='lines', name='Planned Z', 
                   line=dict(color='green', width=2)),
        row=3, col=1
    )
    fig2.add_trace(
        go.Scatter(x=time_actual, y=actual_positions[:, 2], 
                   mode='lines', name='Actual Z', 
                   line=dict(color='red', width=2)),
        row=3, col=1
    )
    
    # Update axes labels
    fig2.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig2.update_yaxes(title_text="X (m)", row=1, col=1)
    fig2.update_yaxes(title_text="Y (m)", row=2, col=1)
    fig2.update_yaxes(title_text="Z (m)", row=3, col=1)
    
    # Update layout
    fig2.update_layout(
        title=f'Position Tracking vs Time - {primitive_name}<br>Applied Force: [{force[0]:.1f}, {force[1]:.1f}, {force[2]:.1f}] N',
        height=900,
        width=1200,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Show the plot
    fig2.show()
    
    # Convert joint data to numpy arrays
    actual_joint_positions = np.array(actual_joint_positions)
    actual_joint_velocities = np.array(actual_joint_velocities)
    
    # Create figure with 7 subplots for joint positions and velocities on same plot
    print("Creating joint tracking visualization...")
    fig3 = make_subplots(
        rows=7, cols=1,
        subplot_titles=[f'Joint {i+1} - Position and Velocity vs Time' for i in range(7)],
        vertical_spacing=0.02,
        specs=[[{"secondary_y": True}] for _ in range(7)]
    )
    
    # Add traces for each joint
    for joint_idx in range(7):
        # Position - Planned (primary y-axis)
        fig3.add_trace(
            go.Scatter(x=time_planned, y=trajectory_q[:, joint_idx], 
                       mode='lines', name=f'Planned q{joint_idx+1}',
                       line=dict(color='green', width=2, dash='solid'),
                       showlegend=(joint_idx == 0)),
            row=joint_idx+1, col=1,
            secondary_y=False
        )
        # Position - Actual (primary y-axis)
        fig3.add_trace(
            go.Scatter(x=time_actual, y=actual_joint_positions[:, joint_idx], 
                       mode='lines', name=f'Actual q{joint_idx+1}',
                       line=dict(color='red', width=2, dash='solid'),
                       showlegend=(joint_idx == 0)),
            row=joint_idx+1, col=1,
            secondary_y=False
        )
        
        # Velocity - Planned (secondary y-axis)
        fig3.add_trace(
            go.Scatter(x=time_planned, y=trajectory_dq[:, joint_idx], 
                       mode='lines', name=f'Planned dq{joint_idx+1}',
                       line=dict(color='lightgreen', width=1.5, dash='dash'),
                       showlegend=(joint_idx == 0)),
            row=joint_idx+1, col=1,
            secondary_y=True
        )
        # Velocity - Actual (secondary y-axis)
        fig3.add_trace(
            go.Scatter(x=time_actual, y=actual_joint_velocities[:, joint_idx], 
                       mode='lines', name=f'Actual dq{joint_idx+1}',
                       line=dict(color='orange', width=1.5, dash='dash'),
                       showlegend=(joint_idx == 0)),
            row=joint_idx+1, col=1,
            secondary_y=True
        )
        
        # Update y-axis labels
        fig3.update_yaxes(title_text=f"Position (rad)", row=joint_idx+1, col=1, secondary_y=False)
        fig3.update_yaxes(title_text=f"Velocity (rad/s)", row=joint_idx+1, col=1, secondary_y=True)
    
    # Update x-axis label for bottom subplot
    fig3.update_xaxes(title_text="Time (s)", row=7, col=1)
    
    # Update layout
    fig3.update_layout(
        title=f'Joint Position & Velocity Tracking - {primitive_name}<br>Applied Force: [{force[0]:.1f}, {force[1]:.1f}, {force[2]:.1f}] N',
        height=1400,
        width=1200,
        showlegend=True,
    )
    
    # Show the plot
    fig3.show()



 #[ 0.56795613 -0.05610449  0.1748564 ]
 #[ 0.57543462 -0.02768491  0.17638216