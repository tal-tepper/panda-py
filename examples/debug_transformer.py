"""
Debug script for trajectory transformer - tests without transformation first

This helps identify whether the issue is with IK or with the transformation.
"""

import numpy as np
from trajectory_transformer import TrajectoryTransformer

# Test 1: NO transformation - should work perfectly
print("="*70)
print("TEST 1: NO TRANSFORMATION (Identity test)")
print("="*70 + "\n")

transformer = TrajectoryTransformer(verbose=True)

result = transformer.transform_trajectory(
    npy_path='/home/robot-lab/repos/tactile_panda/primitive_files/primitive_poses_soft_shell_delicate_slow_3.npy',
    primitive_name='jab',
    translation=np.array([0.0, 0.0, 0.0]),  # NO translation
    rotation_axis=None,
    rotation_angle=0.0,  # NO rotation
    max_waypoints=50,  # Use fewer for faster testing
    min_distance=0.02,
    speed_factor=0.1
)

print(f"\n{'='*70}")
print(f"TEST 1 RESULT: {len(result['q_waypoints'])} waypoints generated")
print(f"Success rate: {len(result['valid_indices'])/len(result['q_original'])*100:.1f}%")
print(f"{'='*70}\n")

if len(result['q_waypoints']) == 0:
    print("ERROR: Even with NO transformation, IK is failing!")
    print("This suggests a problem with the IK function itself or data format.")
    print("\nDEBUGGING INFO:")
    print(f"  Original trajectory shape: {result['q_original'].shape}")
    print(f"  First joint position: {result['q_original'][0]}")
    print(f"  First cartesian position: {result['positions_original'][0]}")
    print(f"  First orientation:\n{result['orientations_original'][0]}")
else:
    print("✓ TEST 1 PASSED: Base functionality works\n")
    
    # Test 2: Minimal transformation
    print("="*70)
    print("TEST 2: MINIMAL TRANSFORMATION")
    print("="*70 + "\n")
    
    result2 = transformer.transform_trajectory(
        npy_path='/home/robot-lab/repos/tactile_panda/primitive_files/primitive_poses_soft_shell_delicate_slow_3.npy',
        primitive_name='jab',
        translation=np.array([0.0, 0.001, 0.0]),  # Only 1mm translation
        rotation_axis=None,
        rotation_angle=0.0,
        max_waypoints=50,
        min_distance=0.02,
        speed_factor=0.1
    )
    
    print(f"\n{'='*70}")
    print(f"TEST 2 RESULT: {len(result2['q_waypoints'])} waypoints generated")
    print(f"Success rate: {len(result2['valid_indices'])/len(result2['q_original'])*100:.1f}%")
    print(f"{'='*70}\n")
