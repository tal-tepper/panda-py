"""
Example: Transform and Execute Recorded Trajectory

This example demonstrates how to:
1. Load a recorded trajectory from guided mode
2. Apply translation and rotation transformations
3. Generate a valid trajectory for execution
4. Execute the transformed trajectory on the robot

NOTE: This example expects to be connected to the robot.
If you're just testing the transformation without robot access,
comment out the execution section at the bottom.
"""

import numpy as np
import sys
from time import sleep
from trajectory_transformer import TrajectoryTransformer


def example_transform_and_execute():
    """
    Transform a recorded trajectory and execute it on the robot.
    """
    
    # ============================================
    # CONFIGURATION
    # ============================================
    
    # Input trajectory
    npy_path = '/home/robot-lab/repos/tactile_panda/primitive_files/primitive_poses_soft_shell_delicate_slow_3.npy'
    primitive_name = 'left_and_right'
    
    # Transformation parameters
    translation = np.array([0.001, 0.003, 0.0])  # Move 10cm in Y direction
    rotation_axis = 'z'  # Rotate about Z axis
    rotation_angle = np.radians(40)  # 15 degrees rotation
    
    # Or use a custom rotation matrix (comment out axis/angle if using this)
    # rotation_matrix = np.array([
    #     [1, 0, 0],
    #     [0, np.cos(np.pi/4), -np.sin(np.pi/4)],
    #     [0, np.sin(np.pi/4), np.cos(np.pi/4)]
    # ])
    rotation_matrix = None
    
    # Trajectory generation parameters
    max_waypoints = 150  # More waypoints = smoother but slower computation
    min_distance = 0.005  # Minimum 0.005 rad between waypoints (more permissive)
    speed_factor = 0.05  # Slower for safety (0.05 = 5% of max speed)
    
    # Robot connection (only needed for execution)
    hostname = '172.16.0.2'
    username = 'tamarlab'
    password = 'Panda468#'
    
    # ============================================
    # TRAJECTORY TRANSFORMATION
    # ============================================
    
    print("\n" + "="*70)
    print(" TRAJECTORY TRANSFORMATION EXAMPLE")
    print("="*70)
    print(f"\nInput File: {npy_path}")
    print(f"Primitive: {primitive_name}")
    print(f"Translation: {translation} m")
    if rotation_matrix is not None:
        print(f"Rotation: Custom matrix")
    elif rotation_axis:
        print(f"Rotation: {np.degrees(rotation_angle):.1f}° about {rotation_axis}-axis")
    print(f"Max waypoints: {max_waypoints}")
    print(f"Speed factor: {speed_factor}")
    
    # Create transformer
    transformer = TrajectoryTransformer(verbose=True)
    
    # Transform the trajectory
    result = transformer.transform_trajectory(
        npy_path=npy_path,
        primitive_name=primitive_name,
        translation=translation,
        rotation_matrix=rotation_matrix,
        rotation_axis=rotation_axis,
        rotation_angle=rotation_angle,
        rotate_orientation=True,  # Rotate end-effector orientation by the same angle
        max_waypoints=max_waypoints,
        min_distance=min_distance,
        speed_factor=speed_factor
    )
    
    # Save transformed trajectory
    output_path = f'transformed_{primitive_name}.npy'
    transformer.save_transformed_trajectory(result, output_path)
    
    print("\n" + "="*70)
    print(" TRANSFORMATION SUMMARY")
    print("="*70)
    print(f"\nOriginal trajectory length: {len(result['q_original'])} waypoints")
    print(f"Transformed trajectory: {len(result['q_transformed_full'])} valid waypoints")
    print(f"Downsampled for execution: {len(result['q_waypoints'])} waypoints")
    print(f"\nSuccess rate: {len(result['valid_indices'])/len(result['q_original'])*100:.1f}%")
    
    # Print some statistics about the transformation
    original_pos = result['positions_original']
    transformed_pos = result['positions_transformed']
    
    print(f"\nOriginal workspace:")
    print(f"  X: [{original_pos[:, 0].min():.3f}, {original_pos[:, 0].max():.3f}] m")
    print(f"  Y: [{original_pos[:, 1].min():.3f}, {original_pos[:, 1].max():.3f}] m")
    print(f"  Z: [{original_pos[:, 2].min():.3f}, {original_pos[:, 2].max():.3f}] m")
    
    print(f"\nTransformed workspace:")
    print(f"  X: [{transformed_pos[:, 0].min():.3f}, {transformed_pos[:, 0].max():.3f}] m")
    print(f"  Y: [{transformed_pos[:, 1].min():.3f}, {transformed_pos[:, 1].max():.3f}] m")
    print(f"  Z: [{transformed_pos[:, 2].min():.3f}, {transformed_pos[:, 2].max():.3f}] m")
    
    print(f"\nTransformed trajectory saved to: {output_path}")
    print("="*70 + "\n")
    
    # ============================================
    # TRAJECTORY EXECUTION (requires robot connection)
    # ============================================
    
    # COMMENT OUT THIS SECTION IF NOT CONNECTED TO ROBOT
    execute_on_robot = True  # Set to True to execute on robot
    
    if execute_on_robot:
        import panda_py
        import scipy.linalg
        import matplotlib.pyplot as plt
        
        print("\n" + "="*70)
        print(" EXECUTING TRANSFORMED TRAJECTORY")
        print("="*70 + "\n")
        
        # Storage for logged data
        logged_data = {
            'time': [],
            'tau_J': [],
            'gravity': [],
            'coriolis': [],
            'jacobian': [],
            'q': [],
            'dq': []
        }
        
        try:
            # Connect to robot
            print("Connecting to robot...")
            desk = panda_py.Desk(hostname, username, password)
            desk.activate_fci()
            panda = panda_py.Panda(hostname)
            
            # Move to start position
            print(f"Moving to start position...")
            q_start = result['q_waypoints'][0]
            panda.move_to_joint_position(q_start)
            sleep(2)
            
            # Estimate trajectory duration
            # Calculate approximate time based on waypoint distances and speed factor
            q_waypoints = result['q_waypoints']
            total_distance = 0
            for i in range(1, len(q_waypoints)):
                total_distance += np.linalg.norm(q_waypoints[i] - q_waypoints[i-1])
            
            # Rough estimate: assume average velocity and account for speed_factor
            # Typical joint velocity ~1 rad/s, with acceleration/deceleration overhead
            estimated_duration = total_distance / (speed_factor * 1.0) * 1.5  # Add 50% margin
            num_samples = int(1000 * estimated_duration)  # 1000 Hz sampling
            
            print(f"Estimated trajectory duration: {estimated_duration:.1f}s ({num_samples} samples)")
            
            # Enable logging after reaching start position
            print("Enabling data logging...")
            panda.enable_logging(num_samples)
            
            # Execute transformed trajectory
            print(f"Executing trajectory with {len(result['q_waypoints'])} waypoints...")
            success = panda.move_to_joint_position(
                result['q_waypoints'].tolist(),
                speed_factor=speed_factor
            )
            
            if success:
                print("✓ Trajectory executed successfully!")
            else:
                print("⚠ Trajectory execution incomplete")
            
            # Get logged data
            print("Retrieving logged data...")
            log = panda.get_log()
            
            # Return to start
            print("Returning to start position...")
            panda.move_to_joint_position(q_start)
            
            # Process logged data to calculate forces
            print("\nProcessing logged data to calculate forces...")
            print(f"Log keys: {log.keys()}")
            print(f"Number of samples: {len(log['q'])}")
            
            calculated_forces = []
            p_model = panda.get_model()
            
            # Get arrays from log
            q_log = np.array(log['q'])
            dq_log = np.array(log['dq'])
            tau_J_log = np.array(log['tau_J'])
            
            # Process each sample
            for i in range(len(q_log)):
                q = q_log[i]
                dq = dq_log[i]
                tau_j = tau_J_log[i]
                
                # Calculate gravity (pass q directly with default mass and gravity)
                # Assuming default end-effector mass and center of mass
                m_total = 0.73  # Default EE mass in kg
                F_x_Ctotal = [0.01, 0.0, 0.03]  # Default EE center of mass
                gravity = np.array(p_model.gravity(q, m_total, F_x_Ctotal))
                
                # Calculate coriolis (pass q and dq)
                coriolis = np.array(p_model.coriolis(q, dq, m_total, F_x_Ctotal))
                
                # Calculate jacobian (pass q)
                jacobian = np.array(p_model.zero_jacobian(q))
                
                # Calculate joint torques without gravity and coriolis
                tau = tau_j - gravity - coriolis
                
                # Calculate Cartesian forces using least squares
                # F = (J^T)^+ * tau, where ^+ is pseudoinverse
                calced_force, _, _, _ = scipy.linalg.lstsq(jacobian.T, tau, lapack_driver='gelsy')
                
                calculated_forces.append(calced_force[:3])  # Only X, Y, Z forces
            
            calculated_forces = np.array(calculated_forces)
            times = np.arange(len(calculated_forces)) / 1000.0  # Time in seconds (1000 Hz sampling)
            
            # Plot force vs time
            print("Creating force plot...")
            fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
            
            axes[0].plot(times, calculated_forces[:, 0], 'r-', linewidth=2)
            axes[0].set_ylabel('Force X (N)', fontsize=12)
            axes[0].grid(True, alpha=0.3)
            axes[0].set_title('Calculated Cartesian Forces', fontsize=14, fontweight='bold')
            
            axes[1].plot(times, calculated_forces[:, 1], 'g-', linewidth=2)
            axes[1].set_ylabel('Force Y (N)', fontsize=12)
            axes[1].grid(True, alpha=0.3)
            
            axes[2].plot(times, calculated_forces[:, 2], 'b-', linewidth=2)
            axes[2].set_ylabel('Force Z (N)', fontsize=12)
            axes[2].set_xlabel('Time (s)', fontsize=12)
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('trajectory_forces.png', dpi=150)
            print("✓ Force plot saved to trajectory_forces.png")
            plt.show()
            
            # Print force statistics
            print(f"\nForce Statistics:")
            print(f"  X: mean={calculated_forces[:, 0].mean():.2f}N, std={calculated_forces[:, 0].std():.2f}N, max={np.abs(calculated_forces[:, 0]).max():.2f}N")
            print(f"  Y: mean={calculated_forces[:, 1].mean():.2f}N, std={calculated_forces[:, 1].std():.2f}N, max={np.abs(calculated_forces[:, 1]).max():.2f}N")
            print(f"  Z: mean={calculated_forces[:, 2].mean():.2f}N, std={calculated_forces[:, 2].std():.2f}N, max={np.abs(calculated_forces[:, 2]).max():.2f}N")
            
        except Exception as e:
            print(f"✗ Error during execution: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            desk.deactivate_fci()
            desk.release_control()
            print("\n✓ Robot control released")
        
        print("="*70 + "\n")
    else:
        print("\n[INFO] Execution skipped (execute_on_robot=False)")
        print("[INFO] Set execute_on_robot=True to run on the robot\n")
    
    return result


def example_multiple_transformations():
    """
    Example: Create multiple transformed versions of the same trajectory.
    """
    
    print("\n" + "="*70)
    print(" BATCH TRANSFORMATION EXAMPLE")
    print("="*70 + "\n")
    
    # Input configuration
    npy_path = '/home/robot-lab/repos/tactile_panda/primitive_files/primitive_poses_hard_shell_delicate.npy'
    primitive_name = 'try1'
    
    # Create transformer
    transformer = TrajectoryTransformer(verbose=False)  # Less verbose for batch processing
    
    # Define multiple transformations
    transformations = [
        {'name': 'offset_x_pos', 'translation': np.array([0.05, 0.0, 0.0])},
        {'name': 'offset_x_neg', 'translation': np.array([-0.05, 0.0, 0.0])},
        {'name': 'offset_y_pos', 'translation': np.array([0.0, 0.05, 0.0])},
        {'name': 'offset_y_neg', 'translation': np.array([0.0, -0.05, 0.0])},
        {'name': 'rotate_z_15deg', 'translation': np.zeros(3), 
         'rotation_axis': 'z', 'rotation_angle': np.radians(15)},
        {'name': 'rotate_z_neg15deg', 'translation': np.zeros(3), 
         'rotation_axis': 'z', 'rotation_angle': np.radians(-15)},
    ]
    
    results = {}
    
    for tf in transformations:
        print(f"\nProcessing: {tf['name']}")
        print("-" * 70)
        
        transformer.verbose = True  # Enable verbose for this transformation
        
        try:
            result = transformer.transform_trajectory(
                npy_path=npy_path,
                primitive_name=primitive_name,
                translation=tf.get('translation', np.zeros(3)),
                rotation_axis=tf.get('rotation_axis'),
                rotation_angle=tf.get('rotation_angle', 0.0),
                max_waypoints=100,
                min_distance=0.01,
                speed_factor=0.1
            )
            
            # Save result
            output_path = f'transformed_{primitive_name}_{tf["name"]}.npy'
            transformer.save_transformed_trajectory(result, output_path)
            
            results[tf['name']] = result
            
            print(f"✓ {tf['name']}: {len(result['q_waypoints'])} waypoints - Saved to {output_path}")
            
        except Exception as e:
            print(f"✗ {tf['name']}: Failed - {e}")
        
        transformer.verbose = False
    
    print("\n" + "="*70)
    print(f" BATCH COMPLETE: {len(results)}/{len(transformations)} successful")
    print("="*70 + "\n")
    
    return results


if __name__ == '__main__':
    # Run the main example
    result = example_transform_and_execute()
    
    # Uncomment to run batch transformations instead
    # results = example_multiple_transformations()
