"""
Simple test to debug HeightConstrainedJointTrajectory segfault.
Tests with minimal waypoints to isolate the issue.
"""

import numpy as np
import panda_py
from panda_py._core import JointTrajectory, HeightConstrainedJointTrajectory

print("=" * 70)
print("TEST 1: Create simple JointTrajectory with 3 waypoints")
print("=" * 70)

# Create a minimal trajectory with just 3 waypoints
q1 = np.array([[0.0], [-0.785], [0.0], [-2.356], [0.0], [1.571], [0.785]])
q2 = np.array([[0.1], [-0.785], [0.0], [-2.356], [0.0], [1.571], [0.785]])
q3 = np.array([[0.2], [-0.785], [0.0], [-2.356], [0.0], [1.571], [0.785]])

waypoints = [q1, q2, q3]

try:
    traj = JointTrajectory(
        waypoints=waypoints,
        speed_factor=0.2,
        max_deviation=0.0,
        timeout=60.0
    )
    print(f"✓ JointTrajectory created")
    print(f"  Duration: {traj.get_duration():.2f}s")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Try to create HeightConstrainedJointTrajectory with minimal parameters
print("\n" + "=" * 70)
print("TEST 2: Create HeightConstrainedJointTrajectory with default params")
print("=" * 70)

try:
    constrained = HeightConstrainedJointTrajectory(
        base_trajectory=traj,
        height_limit=0.15,
        dt=0.01  # Larger dt = fewer samples = faster
    )
    print(f"✓ HeightConstrainedJointTrajectory created")
    print(f"  Duration: {constrained.get_duration():.2f}s")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Try with explicit parameters
print("\n" + "=" * 70)
print("TEST 3: Create HeightConstrainedJointTrajectory with explicit params")
print("=" * 70)

try:
    constrained = HeightConstrainedJointTrajectory(
        base_trajectory=traj,
        height_limit=0.15,
        dt=0.01,
        max_deviation=0.001,
        speed_factor=0.2,
        timeout=10.0,
        max_waypoints=100
    )
    print(f"✓ HeightConstrainedJointTrajectory created")
    print(f"  Duration: {constrained.get_duration():.2f}s")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Try with a longer trajectory (more waypoints)
print("\n" + "=" * 70)
print("TEST 4: Create HeightConstrainedJointTrajectory with 10 waypoints")
print("=" * 70)

waypoints_long = []
for i in range(10):
    q = np.array([
        [0.0 + 0.05 * i],
        [-0.785],
        [0.0],
        [-2.356],
        [0.0],
        [1.571],
        [0.785]
    ])
    waypoints_long.append(q)

try:
    traj_long = JointTrajectory(
        waypoints=waypoints_long,
        speed_factor=0.2,
        max_deviation=0.0,
        timeout=60.0
    )
    print(f"✓ JointTrajectory with 10 waypoints created")
    print(f"  Duration: {traj_long.get_duration():.2f}s")
    
    constrained_long = HeightConstrainedJointTrajectory(
        base_trajectory=traj_long,
        height_limit=0.15,
        dt=0.05,  # Larger dt
        max_deviation=0.0001,
        speed_factor=0.2,
        timeout=10.0,
        max_waypoints=500
    )
    print(f"✓ HeightConstrainedJointTrajectory created")
    print(f"  Duration: {constrained_long.get_duration():.2f}s")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
