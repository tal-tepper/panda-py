"""
Sweep test for trajectory pipeline across tilt/rotation angles and primitives.

Tests all combinations of:
  - 3 primitives (poke, left_and_right, back_and_forth)
  - Tilt angles: 0° to 30° (step 10°)
  - Rotation angles: -45° to 45° (step 15°)

Pipeline per combination:
  Original joints → FK → tilt/rotate → IK → JointTrajectory → HeightConstrained

Only plots cases with problems. Prints summary table at the end.
Use debug_height_limit_trajectory.py to investigate specific problem cases.

Usage:
    python sweep_trajectory_test.py [path_to_npy_file]
"""

import sys
import os
import time
import numpy as np

# Shared utilities
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trajectory_test_utils import (
    load_trajectory_data,
    make_transformer,
    test_combination,
    is_problem,
    plot_problem,
    print_summary,
    DEFAULT_HEIGHT_LIMIT,
    DEFAULT_MAX_DEVIATION,
)

# =============================================================================
# Configuration
# =============================================================================
NPY_PATH = sys.argv[1] if len(sys.argv) > 1 else None

PRIMITIVES = ['poke', 'left_and_right', 'back_and_forth']
TILT_ANGLES_DEG = list(range(0, 35, 10))        # 0, 10, 20, 30
ROTATION_ANGLES_DEG = list(range(-45, 50, 15))   # -45 … 45

HEIGHT_LIMIT  = DEFAULT_HEIGHT_LIMIT
MAX_DEVIATION = DEFAULT_MAX_DEVIATION

# =============================================================================
# Main sweep
# =============================================================================
print("=" * 70)
print("TRAJECTORY PIPELINE SWEEP TEST")
print("=" * 70)

# Load data
raw_data, npy_path = load_trajectory_data(NPY_PATH)
print(f"Loaded: {npy_path}")
print(f"Primitives available: {list(raw_data.keys())}")
print(f"Tilt angles:    {TILT_ANGLES_DEG}°")
print(f"Rotation angles: {ROTATION_ANGLES_DEG}°")
print(f"Height limit:   {HEIGHT_LIMIT*100:.0f} cm")
print(f"Max deviation:  {MAX_DEVIATION}")

# Set up transformer
transformer = make_transformer()

total = len(PRIMITIVES) * len(TILT_ANGLES_DEG) * len(ROTATION_ANGLES_DEG)
count = 0
all_results = {}
problems = []

for prim_name in PRIMITIVES:
    if prim_name not in raw_data:
        print(f"\n⚠ Primitive '{prim_name}' not found in file, skipping")
        continue

    q_full = np.array(raw_data[prim_name]['q'])

    print(f"\n{'=' * 70}")
    print(f"Primitive: {prim_name}  ({len(q_full)} waypoints)")
    print(f"{'=' * 70}")

    for tilt_deg in TILT_ANGLES_DEG:
        for rot_deg in ROTATION_ANGLES_DEG:
            count += 1
            tag = f"{prim_name}/tilt={tilt_deg}°/rot={rot_deg}°"

            t0 = time.time()
            result = test_combination(transformer, q_full, tilt_deg, rot_deg)
            elapsed = time.time() - t0

            # Build status string
            parts = []
            if result['ik_success_rate'] < 100:
                parts.append(f"IK={result['ik_success_rate']:.0f}%")
            if result['joint_traj_ok']:
                wp_note = (f"↓{result['joint_traj_n_waypoints']}"
                           if result['joint_traj_reduced'] else
                           f"{result['joint_traj_n_waypoints']}")
                parts.append(f"JT={wp_note}wp/{result['joint_traj_duration']:.1f}s")
            if result['height_constrained_ok']:
                parts.append(f"HC={result['height_constrained_duration']:.1f}s")

            flagged = is_problem(result)
            marker = "✗" if flagged else "✓"
            info = " | ".join(parts) if parts else "—"
            err_short = ""
            if result['error']:
                err_short = f"  [{result['error'][:60]}]"

            print(f"  [{count:3d}/{total}] {marker} tilt={tilt_deg:3d}° "
                  f"rot={rot_deg:4d}°  {info}{err_short}  ({elapsed:.1f}s)")

            result['tag'] = tag
            all_results[tag] = result
            if flagged:
                problems.append(result)

# =============================================================================
# Plot problematic cases
# =============================================================================
if problems:
    import matplotlib
    matplotlib.use('Agg')   # non-interactive for batch saving

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'sweep_problems')
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"Saving {len(problems)} problem plot(s) to {save_dir}/")
    print(f"{'=' * 70}")

    for prob in problems:
        fpath = plot_problem(prob, save_dir)
        print(f"  📊 {os.path.basename(fpath)}")

# =============================================================================
# Summary
# =============================================================================
print_summary(problems, count)

print(f"\n{'=' * 70}")
print("DONE")
print(f"{'=' * 70}")
