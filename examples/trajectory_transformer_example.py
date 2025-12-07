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
    npy_path = '/home/robot-lab/repos/tactile_panda/primitive_files/primitive_poses_hard_shell_delicate.npy'
    primitive_name = 'try1'
    
    # Transformation parameters
    translation = np.array([0.0, 0.1, 0.0])  # Move 10cm in Y direction
    rotation_axis = 'z'  # Rotate about Z axis
    rotation_angle = np.radians(0)  # 15 degrees rotation
    
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
    execute_on_robot = False  # Set to True to execute on robot
    
    if execute_on_robot:
        import panda_py
        
        print("\n" + "="*70)
        print(" EXECUTING TRANSFORMED TRAJECTORY")
        print("="*70 + "\n")
        
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
            
            # Return to start
            print("Returning to start position...")
            panda.move_to_joint_position(q_start)
            
        except Exception as e:
            print(f"✗ Error during execution: {e}")
            
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
