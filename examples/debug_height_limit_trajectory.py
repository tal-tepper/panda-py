"""
Detailed single-run debug script for the trajectory pipeline.

Tests one specific (primitive, tilt, rotation) combination with full
step-by-step output and figures. Use this script to deep-dive into
a specific case — typically one flagged by sweep_trajectory_test.py.

Usage:
    python debug_height_limit_trajectory.py [primitive] [tilt_deg] [rot_deg] [npy_path]

Examples:
    python debug_height_limit_trajectory.py poke 30 -15
    python debug_height_limit_trajectory.py left_and_right 20 45 /path/to/file.npy
    python debug_height_limit_trajectory.py   # defaults: poke 30 0
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
    transform_trajectory,
    build_joint_trajectory,
    build_height_constrained,
    is_problem,
    plot_full_pipeline,
    _compute_heights,
    _sample_trajectory,
    DEFAULT_HEIGHT_LIMIT,
    DEFAULT_SPEED_FACTOR,
    DEFAULT_MAX_DEVIATION,
)
import panda_py

# =============================================================================
# Parse arguments
# =============================================================================
PRIMITIVE  = sys.argv[1] if len(sys.argv) > 1 else 'poke'
TILT_DEG   = int(sys.argv[2]) if len(sys.argv) > 2 else 30
ROT_DEG    = int(sys.argv[3]) if len(sys.argv) > 3 else 0
NPY_PATH   = sys.argv[4] if len(sys.argv) > 4 else None

HEIGHT_LIMIT   = DEFAULT_HEIGHT_LIMIT
SPEED_FACTOR   = DEFAULT_SPEED_FACTOR
MAX_DEVIATION  = DEFAULT_MAX_DEVIATION


def print_section(title):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# =============================================================================
# Main
# =============================================================================
print("=" * 70)
print("TRAJECTORY PIPELINE — DETAILED DEBUG")
print("=" * 70)

# --- Step 1: Load data ---
print_section("Step 1: Load data")
raw_data, path = load_trajectory_data(NPY_PATH)
print(f"  File:       {path}")
print(f"  Primitives: {list(raw_data.keys())}")

if PRIMITIVE not in raw_data:
    print(f"\n  ✗ Primitive '{PRIMITIVE}' not found! Available: {list(raw_data.keys())}")
    sys.exit(1)

q_full = np.array(raw_data[PRIMITIVE]['q'])
print(f"  Primitive:  {PRIMITIVE}")
print(f"  Waypoints:  {len(q_full)}")

# Compute original heights
h_orig = _compute_heights(q_full)
print(f"  Height range: {h_orig.min()*100:.1f} cm – {h_orig.max()*100:.1f} cm")

# --- Step 2: Transform (FK → tilt/rotate → IK) ---
print_section(f"Step 2: Transform (tilt={TILT_DEG}°, rot={ROT_DEG}°)")
transformer = make_transformer()

t0 = time.time()
q_transformed, ik_valid, ik_total = transform_trajectory(
    transformer, q_full, TILT_DEG, ROT_DEG)
t_transform = time.time() - t0

ik_rate = ik_valid / ik_total * 100 if ik_total else 0
print(f"  IK success: {ik_valid}/{ik_total} ({ik_rate:.1f}%)")
print(f"  Time:       {t_transform:.2f}s")

if ik_valid < 2:
    print(f"\n  ✗ IK failed — only {ik_valid} valid poses. Cannot continue.")
    sys.exit(1)

h_trans = _compute_heights(q_transformed)
print(f"  Height range after transform: {h_trans.min()*100:.1f} cm – {h_trans.max()*100:.1f} cm")

# --- Step 3: JointTrajectory (with gradual reduction) ---
print_section("Step 3: JointTrajectory")
print(f"  Speed factor:  {SPEED_FACTOR}")
print(f"  Max deviation: {MAX_DEVIATION}")

t0 = time.time()
try:
    jt, n_used, n_before, was_reduced = build_joint_trajectory(
        q_transformed, speed_factor=SPEED_FACTOR, max_deviation=MAX_DEVIATION)
    t_jt = time.time() - t0

    print(f"  ✓ Success!")
    if was_reduced:
        print(f"  Waypoints:      {n_before} → {n_used} (REDUCED)")
    else:
        print(f"  Waypoints:      {n_used}")
    print(f"  Duration:       {jt.get_duration():.2f}s")
    print(f"  Time:           {t_jt:.2f}s")

    # Sample and show height range
    ts_jt, q_jt = _sample_trajectory(jt, n_samples=500)
    h_jt = _compute_heights(q_jt)
    print(f"  Height range:   {h_jt.min()*100:.1f} cm – {h_jt.max()*100:.1f} cm")

except RuntimeError as e:
    t_jt = time.time() - t0
    print(f"  ✗ FAILED after {t_jt:.1f}s: {e}")
    jt = None
    n_used = 0
    was_reduced = False

# --- Step 5: HeightConstrainedJointTrajectory ---
ct = None
if jt is not None:
    print_section("Step 4: HeightConstrainedJointTrajectory")
    print(f"  Height limit: {HEIGHT_LIMIT*100:.0f} cm")

    t0 = time.time()
    try:
        ct = build_height_constrained(jt, height_limit=HEIGHT_LIMIT)
        t_hc = time.time() - t0

        print(f"  ✓ Success!")
        print(f"  Duration: {ct.get_duration():.2f}s")
        print(f"  Time:     {t_hc:.2f}s")

        ts_ct, q_ct = _sample_trajectory(ct, n_samples=500)
        h_ct = _compute_heights(q_ct)
        print(f"  Height range: {h_ct.min()*100:.1f} cm – {h_ct.max()*100:.1f} cm")
        violations = np.sum(h_ct > HEIGHT_LIMIT)
        if violations > 0:
            print(f"  ⚠ {violations} samples exceed height limit!")
        else:
            print(f"  ✓ All samples within height limit")

    except Exception as e:
        t_hc = time.time() - t0
        print(f"  ✗ FAILED after {t_hc:.1f}s: {e}")

# --- Step 5: Figures ---
print_section("Step 5: Plotting")

# Build a result dict compatible with plot_full_pipeline
result = dict(
    tag=f"{PRIMITIVE}/tilt={TILT_DEG}°/rot={ROT_DEG}°",
    tilt_deg=TILT_DEG, rot_deg=ROT_DEG,
    ik_success_rate=ik_rate, ik_valid=ik_valid, ik_total=ik_total,
    joint_traj_ok=(jt is not None),
    joint_traj_n_waypoints=n_used,
    joint_traj_duration=jt.get_duration() if jt else 0,
    joint_traj_reduced=was_reduced,
    height_constrained_ok=(ct is not None),
    height_constrained_duration=ct.get_duration() if ct else 0,
    error=None,
    q_original=q_full, q_transformed=q_transformed,
    joint_traj=jt, constrained_traj=ct,
)

# Save figure
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'debug_figures')
os.makedirs(save_dir, exist_ok=True)
safe_tag = result['tag'].replace('/', '_').replace('°', '').replace('=', '')
save_path = os.path.join(save_dir, f'debug_{safe_tag}.png')

plot_full_pipeline(result, height_limit=HEIGHT_LIMIT,
                   show=True, save_path=save_path)

# --- Summary ---
print_section("Summary")
flagged = is_problem(result)
print(f"  Primitive:    {PRIMITIVE}")
print(f"  Tilt:         {TILT_DEG}°")
print(f"  Rotation:     {ROT_DEG}°")
print(f"  IK rate:      {ik_rate:.1f}%")
if jt:
    print(f"  JT waypoints: {n_used}" + (" (reduced)" if was_reduced else ""))
    print(f"  JT duration:  {jt.get_duration():.2f}s")
if ct:
    print(f"  HC duration:  {ct.get_duration():.2f}s")
print(f"  Problem:      {'YES' if flagged else 'NO'}")
print(f"\n{'=' * 70}")
print("DONE")
print(f"{'=' * 70}")

# =============================================================================
# Optionally run on robot
# =============================================================================
if ct is not None and not flagged:
    try:
        run_robot = input("\nWould you like to run this trajectory on the robot? [y/N]: ").strip().lower()
        if run_robot == 'y':
            print("\nEnter Desk connection details:")
            host = input("  Desk IP/hostname: ").strip()
            user = input("  Desk username: ").strip()
            pw = input("  Desk password: ").strip()
            print("\nConnecting to Desk and unlocking robot...")
            from panda_py import Desk, Panda
            desk = Desk(host, user, pw)
            desk.unlock()
            desk.activate_fci()
            print("\nConnecting to Panda robot...")
            panda = Panda(host)
            print("\nSending trajectory to robot (height-constrained)...")
            # Extract waypoints from the HeightConstrainedJointTrajectory
            waypoints = [ct.get_joint_positions(t) for t in np.linspace(0, ct.get_duration(), num=200)]
            # Use the move_to_joint_position_with_height_limit method
            ok = panda.move_to_joint_position_with_height_limit(
                waypoints, height_limit=HEIGHT_LIMIT, speed_factor=SPEED_FACTOR)
            if ok:
                print("\n✓ Trajectory executed successfully on robot.")
            else:
                print("\n✗ Robot reported failure during execution.")
    except Exception as e:
        print(f"\n✗ Exception during robot execution: {e}")
