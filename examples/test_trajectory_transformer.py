"""
Trajectory Transformer - Test Script (No Robot Required)

This script demonstrates trajectory transformation without requiring
robot connection. Perfect for testing transformations offline.
"""

import numpy as np
from trajectory_transformer import TrajectoryTransformer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_trajectories(original_pos, transformed_pos, title="Trajectory Comparison"):
    """
    Visualize original and transformed trajectories in 3D.
    
    Args:
        original_pos: Original positions [n, 3]
        transformed_pos: Transformed positions [m, 3]
        title: Plot title
    """
    fig = plt.figure(figsize=(12, 5))
    
    # 3D plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(original_pos[:, 0], original_pos[:, 1], original_pos[:, 2], 
             'g-', linewidth=2, label='Original', alpha=0.7)
    ax1.plot(transformed_pos[:, 0], transformed_pos[:, 1], transformed_pos[:, 2], 
             'r-', linewidth=2, label='Transformed', alpha=0.7)
    ax1.scatter(original_pos[0, 0], original_pos[0, 1], original_pos[0, 2],
                c='blue', s=100, marker='o', label='Start')
    ax1.scatter(original_pos[-1, 0], original_pos[-1, 1], original_pos[-1, 2],
                c='orange', s=100, marker='s', label='End')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title(f'{title} - 3D View')
    ax1.legend()
    ax1.grid(True)
    
    # Top view (X-Y plane)
    ax2 = fig.add_subplot(122)
    ax2.plot(original_pos[:, 0], original_pos[:, 1], 
             'g-', linewidth=2, label='Original', alpha=0.7)
    ax2.plot(transformed_pos[:, 0], transformed_pos[:, 1], 
             'r-', linewidth=2, label='Transformed', alpha=0.7)
    ax2.scatter(original_pos[0, 0], original_pos[0, 1],
                c='blue', s=100, marker='o', label='Start')
    ax2.scatter(original_pos[-1, 0], original_pos[-1, 1],
                c='orange', s=100, marker='s', label='End')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title(f'{title} - Top View (X-Y)')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')
    
    plt.tight_layout()
    plt.savefig(f'trajectory_comparison.png', dpi=150)
    print(f"✓ Visualization saved to trajectory_comparison.png")
    plt.show()


def test_transformation(
    npy_path: str,
    primitive_name: str,
    translation: np.ndarray = np.zeros(3),
    rotation_axis: str = None,
    rotation_angle: float = 0.0,
    max_waypoints: int = 100,
    visualize: bool = True
):
    """
    Test trajectory transformation without robot connection.
    
    Args:
        npy_path: Path to trajectory file
        primitive_name: Name of primitive
        translation: Translation vector [x, y, z]
        rotation_axis: 'x', 'y', or 'z'
        rotation_angle: Angle in radians
        max_waypoints: Maximum waypoints
        visualize: Whether to create visualization
    """
    
    print("\n" + "="*70)
    print(" TRAJECTORY TRANSFORMATION TEST (NO ROBOT)")
    print("="*70)
    
    # Create transformer
    transformer = TrajectoryTransformer(verbose=True)
    
    # Transform trajectory
    result = transformer.transform_trajectory(
        npy_path=npy_path,
        primitive_name=primitive_name,
        translation=translation,
        rotation_axis=rotation_axis,
        rotation_angle=rotation_angle,
        rotate_orientation=True,        
        max_waypoints=max_waypoints,
        min_distance=0.01,
        speed_factor=0.1
    )
    
    # Print summary
    print("\n" + "="*70)
    print(" RESULTS SUMMARY")
    print("="*70)
    
    print(f"\nWaypoints:")
    print(f"  Original:     {len(result['q_original'])}")
    print(f"  Valid IK:     {len(result['q_transformed_full'])} ({len(result['valid_indices'])/len(result['q_original'])*100:.1f}%)")
    print(f"  Downsampled:  {len(result['q_waypoints'])}")
    
    print(f"\nWorkspace - Original:")
    orig_pos = result['positions_original']
    print(f"  X: [{orig_pos[:, 0].min():.3f}, {orig_pos[:, 0].max():.3f}] m  (range: {orig_pos[:, 0].max()-orig_pos[:, 0].min():.3f} m)")
    print(f"  Y: [{orig_pos[:, 1].min():.3f}, {orig_pos[:, 1].max():.3f}] m  (range: {orig_pos[:, 1].max()-orig_pos[:, 1].min():.3f} m)")
    print(f"  Z: [{orig_pos[:, 2].min():.3f}, {orig_pos[:, 2].max():.3f}] m  (range: {orig_pos[:, 2].max()-orig_pos[:, 2].min():.3f} m)")
    
    print(f"\nWorkspace - Transformed:")
    trans_pos = result['positions_transformed']
    print(f"  X: [{trans_pos[:, 0].min():.3f}, {trans_pos[:, 0].max():.3f}] m  (range: {trans_pos[:, 0].max()-trans_pos[:, 0].min():.3f} m)")
    print(f"  Y: [{trans_pos[:, 1].min():.3f}, {trans_pos[:, 1].max():.3f}] m  (range: {trans_pos[:, 1].max()-trans_pos[:, 1].min():.3f} m)")
    print(f"  Z: [{trans_pos[:, 2].min():.3f}, {trans_pos[:, 2].max():.3f}] m  (range: {trans_pos[:, 2].max()-trans_pos[:, 2].min():.3f} m)")
    
    print(f"\nJoint Range Changes:")
    orig_q = result['q_original']
    trans_q = result['q_transformed_full']
    for i in range(7):
        orig_range = orig_q[:, i].max() - orig_q[:, i].min()
        trans_range = trans_q[:, i].max() - trans_q[:, i].min()
        print(f"  Joint {i+1}: {np.degrees(orig_range):6.2f}° → {np.degrees(trans_range):6.2f}° (Δ {np.degrees(trans_range-orig_range):+.2f}°)")
    
    # Check if trajectory is safe
    print(f"\nSafety Checks:")
    from panda_py.constants import JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER
    joint_limits_lower = np.array(JOINT_LIMITS_LOWER)
    joint_limits_upper = np.array(JOINT_LIMITS_UPPER)
    
    within_limits = True
    for i in range(7):
        q_min = trans_q[:, i].min()
        q_max = trans_q[:, i].max()
        margin_lower = q_min - joint_limits_lower[i]
        margin_upper = joint_limits_upper[i] - q_max
        
        status = "✓" if margin_lower >= 0 and margin_upper >= 0 else "✗"
        print(f"  {status} Joint {i+1}: margin [{np.degrees(margin_lower):+6.2f}°, {np.degrees(margin_upper):+6.2f}°]")
        
        if margin_lower < 0 or margin_upper < 0:
            within_limits = False
    
    if within_limits:
        print(f"\n✓ All joints within limits - Safe to execute")
    else:
        print(f"\n⚠ Some joints exceed limits - Adjust transformation")
    
    print("="*70 + "\n")
    
    # Visualize if requested
    if visualize:
        try:
            visualize_trajectories(
                result['positions_original'],
                result['positions_transformed'],
                title=f"{primitive_name} - {translation} - {np.degrees(rotation_angle):.0f}°"
            )
        except Exception as e:
            print(f"⚠ Visualization failed: {e}")
            print("  (matplotlib may not be available)")
    
    # Save result
    output_path = f'test_transformed_{primitive_name}.npy'
    transformer.save_transformed_trajectory(result, output_path)
    
    return result


def test_multiple_transformations():
    """
    Test multiple transformations to see which ones are feasible.
    """
    
    npy_path = '/home/robot-lab/repos/tactile_panda/primitive_files/primitive_poses_hard_shell_delicate.npy'
    primitive_name = 'try1'
    
    print("\n" + "="*70)
    print(" TESTING MULTIPLE TRANSFORMATIONS")
    print("="*70 + "\n")
    
    tests = [
        {'name': 'No Change', 'translation': np.zeros(3), 'rotation_axis': None, 'rotation_angle': 0},
        {'name': 'X +5cm', 'translation': np.array([0.05, 0, 0]), 'rotation_axis': None, 'rotation_angle': 0},
        {'name': 'Y +10cm', 'translation': np.array([0, 0.1, 0]), 'rotation_axis': None, 'rotation_angle': 0},
        {'name': 'Z +5cm', 'translation': np.array([0, 0, 0.05]), 'rotation_axis': None, 'rotation_angle': 0},
        {'name': 'Rotate Z +15°', 'translation': np.zeros(3), 'rotation_axis': 'z', 'rotation_angle': np.radians(15)},
        {'name': 'Rotate Z +30°', 'translation': np.zeros(3), 'rotation_axis': 'z', 'rotation_angle': np.radians(30)},
        {'name': 'Combined', 'translation': np.array([0, 0.05, 0]), 'rotation_axis': 'z', 'rotation_angle': np.radians(15)},
    ]
    
    results = {}
    
    for test in tests:
        print(f"\nTest: {test['name']}")
        print("-" * 70)
        
        try:
            transformer = TrajectoryTransformer(verbose=False)
            result = transformer.transform_trajectory(
                npy_path=npy_path,
                primitive_name=primitive_name,
                translation=test['translation'],
                rotation_axis=test['rotation_axis'],
                rotation_angle=test['rotation_angle'],
                max_waypoints=100,
                min_distance=0.01,
                speed_factor=0.1
            )
            
            success_rate = len(result['valid_indices']) / len(result['q_original']) * 100
            n_waypoints = len(result['q_waypoints'])
            
            status = "✓ PASS" if success_rate > 95 and n_waypoints > 50 else "⚠ MARGINAL" if success_rate > 80 else "✗ FAIL"
            
            print(f"{status}: {success_rate:.1f}% valid, {n_waypoints} waypoints")
            
            results[test['name']] = {
                'success_rate': success_rate,
                'n_waypoints': n_waypoints,
                'result': result
            }
            
        except Exception as e:
            print(f"✗ FAIL: {e}")
            results[test['name']] = None
    
    print("\n" + "="*70)
    print(" SUMMARY")
    print("="*70)
    
    for name, data in results.items():
        if data is not None:
            print(f"{name:20s}: {data['success_rate']:5.1f}% valid, {data['n_waypoints']:3d} waypoints")
        else:
            print(f"{name:20s}: FAILED")
    
    print("="*70 + "\n")
    
    return results


if __name__ == '__main__':
    
    # Example 1: Single transformation test
    print("Example 1: Testing single transformation\n")
    
    result = test_transformation(
        npy_path='/home/robot-lab/repos/tactile_panda/primitive_files/primitive_poses_soft_shell_delicate_slow_3.npy',
        primitive_name='jab',
        translation=np.array([0.005, 0.01, 0.0]),  # 10cm in Y
        rotation_axis='z',
        rotation_angle=np.radians(40),
        max_waypoints=150,
        visualize=True
    )
    
    # Example 2: Test multiple transformations
    # Uncomment to run:
    # print("\n\nExample 2: Testing multiple transformations\n")
    # results = test_multiple_transformations()
