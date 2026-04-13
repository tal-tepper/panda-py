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
    downsample_by_distance,
    suggest_downsample_distances,
    DEFAULT_HEIGHT_LIMIT,
    DEFAULT_MAX_DEVIATION,
)

# =============================================================================
# Configuration
# =============================================================================

# --- CLI ---
import argparse
_parser = argparse.ArgumentParser(
    description='Sweep test for trajectory pipeline.',
    formatter_class=argparse.RawDescriptionHelpFormatter)
_parser.add_argument('npy_path', nargs='?', default=None,
                     help='Path to .npy trajectory file')
_parser.add_argument('--min-distance', '-d', type=float, default=None,
                     help='Min joint-space L2 distance between consecutive '
                          'waypoints for downsampling (0 = no downsampling)')
_args = _parser.parse_args()

NPY_PATH = _args.npy_path
MIN_DISTANCE = _args.min_distance

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

# -----------------------------------------------------------------------------
# Suggest downsample distances
# -----------------------------------------------------------------------------
print(f"\n--- Downsample distance suggestions (joint-space L2) ---")
for prim_name in PRIMITIVES:
    if prim_name not in raw_data:
        continue
    q_tmp = np.array(raw_data[prim_name]['q'])
    suggestions = suggest_downsample_distances(q_tmp, factors=(2, 4, 8))
    parts = [f"{f}x → d={d:.6f} ({n} wp)" for f, (d, n) in suggestions.items()]
    print(f"  {prim_name:20s} ({len(q_tmp):5d} wp)  {' | '.join(parts)}")

if MIN_DISTANCE is not None and MIN_DISTANCE > 0:
    print(f"\n  ► Using --min-distance={MIN_DISTANCE:.6f}")
else:
    print(f"\n  ► No downsampling (use --min-distance / -d to enable)")

# -----------------------------------------------------------------------------
# Compute min height after IK for all primitives (for use as height limit)
# -----------------------------------------------------------------------------
print("\nComputing minimum height after IK for all primitives...")
min_heights = []
for prim_name in PRIMITIVES:
    if prim_name not in raw_data:
        continue
    q_full = np.array(raw_data[prim_name]['q'])
    # Use default tilt/rot (0,0) for min height check
    result = test_combination(make_transformer(), q_full, 0, 0, use_pink=True)
    if result.get('q_transformed') is not None:
        from trajectory_test_utils import _compute_heights
        h = _compute_heights(result['q_transformed'])
        min_heights.append(h.min())
if min_heights:
    HEIGHT_LIMIT = min(min_heights) - 0.001  # Add small margin
    print(f"  New height limit (min after IK): {HEIGHT_LIMIT*100:.1f} cm")
else:
    print("  Could not compute min height, using default.")

print(f"Height limit:   {HEIGHT_LIMIT*100:.0f} cm")
print(f"Max deviation:  {MAX_DEVIATION}")



# Set up transformer
transformer = make_transformer()

total = len(PRIMITIVES) * len(TILT_ANGLES_DEG) * len(ROTATION_ANGLES_DEG)
count = 0
all_results = {}
problems = []
primitive_times = {}  # prim_name → list of elapsed times

sweep_t0 = time.time()

for prim_name in PRIMITIVES:
    if prim_name not in raw_data:
        print(f"\n⚠ Primitive '{prim_name}' not found in file, skipping")
        continue

    q_full = np.array(raw_data[prim_name]['q'])

    # Optional distance-based downsampling
    if MIN_DISTANCE is not None and MIN_DISTANCE > 0:
        q_full = downsample_by_distance(q_full, MIN_DISTANCE)

    print(f"\n{'=' * 70}")
    print(f"Primitive: {prim_name}  ({len(q_full)} waypoints)")
    print(f"{'=' * 70}")

    prim_t0 = time.time()
    prim_elapsed_list = []

    for tilt_deg in TILT_ANGLES_DEG:
        for rot_deg in ROTATION_ANGLES_DEG:
            count += 1
            tag = f"{prim_name}/tilt={tilt_deg}°/rot={rot_deg}°"

            t0 = time.time()
            result = test_combination(transformer, q_full, tilt_deg, rot_deg, use_pink=True)
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
            prim_elapsed_list.append(elapsed)
            if flagged:
                problems.append(result)

    # Per-primitive timing summary
    prim_total_time = time.time() - prim_t0
    n_combos = len(prim_elapsed_list)
    if n_combos:
        avg_t = np.mean(prim_elapsed_list)
        med_t = np.median(prim_elapsed_list)
        min_t = np.min(prim_elapsed_list)
        max_t = np.max(prim_elapsed_list)
        print(f"  ┌─ {prim_name} timing: {n_combos} combos in {prim_total_time:.1f}s "
              f"(avg {avg_t:.1f}s, med {med_t:.1f}s, min {min_t:.1f}s, max {max_t:.1f}s)")
    primitive_times[prim_name] = {
        'total': prim_total_time,
        'count': n_combos,
        'times': prim_elapsed_list,
    }

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

sweep_elapsed = time.time() - sweep_t0
print(f"\n--- Timing summary ---")
print(f"  {'Primitive':<20s} {'Combos':>6s} {'Total':>8s} {'Avg':>7s} {'Med':>7s} {'Min':>7s} {'Max':>7s}")
print(f"  {'-'*20} {'-'*6} {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
for prim_name, pt in primitive_times.items():
    ts = pt['times']
    if ts:
        print(f"  {prim_name:<20s} {pt['count']:6d} {pt['total']:7.1f}s "
              f"{np.mean(ts):6.1f}s {np.median(ts):6.1f}s "
              f"{np.min(ts):6.1f}s {np.max(ts):6.1f}s")
print(f"  {'TOTAL':<20s} {count:6d} {sweep_elapsed:7.1f}s")

print(f"\n{'=' * 70}")
print("DONE")
print(f"{'=' * 70}")
