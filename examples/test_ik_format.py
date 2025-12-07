#!/usr/bin/env python3
"""
Test to verify IK function behavior and quaternion format.
"""
import numpy as np
import panda_py
from scipy.spatial.transform import Rotation

# Load a sample trajectory
trajectory_file = '/home/robot-lab/repos/tactile_panda/primitive_files/primitive_poses_soft_shell_delicate_slow_3.npy'
trajectory = np.load(trajectory_file)
print(f"Loaded trajectory shape: {trajectory.shape}")

# Take the first waypoint
q_original = trajectory[0]
print(f"\nOriginal joint positions: {q_original}")

# Get FK for this configuration
T = panda_py.fk(q_original)
print(f"\nFK result T (4x4 matrix):\n{T}")

# Extract position and orientation
position = T[:3, 3]
rotation_matrix = T[:3, :3]
print(f"\nExtracted position: {position}")
print(f"\nExtracted rotation matrix:\n{rotation_matrix}")

# Convert rotation matrix to quaternion using scipy
rot = Rotation.from_matrix(rotation_matrix)
quat_scipy = rot.as_quat()  # Returns [x, y, z, w]
print(f"\nScipy quaternion [x,y,z,w]: {quat_scipy}")

# Try different quaternion formats for IK
position_col = position.reshape(3, 1)
q_col = q_original.reshape(7, 1)

print("\n" + "="*70)
print("Testing different quaternion formats:")
print("="*70)

# Format 1: [w, x, y, z]
quat_wxyz = np.array([[quat_scipy[3]], [quat_scipy[0]], [quat_scipy[1]], [quat_scipy[2]]])
print(f"\n1. Format [w,x,y,z]: {quat_wxyz.T}")
try:
    q_result = panda_py.ik(position_col, quat_wxyz, q_col)
    print(f"   Result: {q_result.flatten()}")
    print(f"   Contains NaN: {np.any(np.isnan(q_result))}")
    if not np.any(np.isnan(q_result)):
        print(f"   ✓ SUCCESS - Error from original: {np.linalg.norm(q_result.flatten() - q_original):.6f}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Format 2: [x, y, z, w]
quat_xyzw = np.array([[quat_scipy[0]], [quat_scipy[1]], [quat_scipy[2]], [quat_scipy[3]]])
print(f"\n2. Format [x,y,z,w]: {quat_xyzw.T}")
try:
    q_result = panda_py.ik(position_col, quat_xyzw, q_col)
    print(f"   Result: {q_result.flatten()}")
    print(f"   Contains NaN: {np.any(np.isnan(q_result))}")
    if not np.any(np.isnan(q_result)):
        print(f"   ✓ SUCCESS - Error from original: {np.linalg.norm(q_result.flatten() - q_original):.6f}")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "="*70)
print("Testing with T matrix directly:")
print("="*70)

# Test using the 4x4 transformation matrix directly
try:
    q_result = panda_py.ik(T, q_col)
    print(f"Result: {q_result.flatten()}")
    print(f"Contains NaN: {np.any(np.isnan(q_result))}")
    if not np.any(np.isnan(q_result)):
        print(f"✓ SUCCESS - Error from original: {np.linalg.norm(q_result.flatten() - q_original):.6f}")
except Exception as e:
    print(f"✗ Error: {e}")
