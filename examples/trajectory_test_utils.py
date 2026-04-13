"""
Shared utilities for trajectory testing scripts.

Used by:
  - debug_height_limit_trajectory.py  (detailed single-run with full figures)
  - sweep_trajectory_test.py          (batch sweep across many parameters)
"""

import os
import sys
import time
import logging
import numpy as np
import panda_py
from panda_py._core import JointTrajectory, HeightConstrainedJointTrajectory
from panda_py.motion import (
    prefilter_waypoints,
    build_joint_trajectory,
    build_height_constrained,
    DEFAULT_HEIGHT_LIMIT,
    DEFAULT_SPEED_FACTOR,
    DEFAULT_MAX_DEVIATION,
)

# Ensure examples/ is on sys.path so trajectory_transformer is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trajectory_transformer import TrajectoryTransformer

# =============================================================================
# Default configuration (scripts may override individual values)
# =============================================================================
# DEFAULT_NPY_PATH = '/home/talte/repos/tactile_panda/primitive_files/pose_34.npy'
DEFAULT_NPY_PATH = '/home/talte/repos/tactile_panda/primitive_files/pose_34_downsampled_d0.0005.npy'
DEFAULT_N_DOWNSAMPLE = 200
DEFAULT_TILT_DIRECTION = np.array([1.0, 0.0])
DEFAULT_ROTATION_AXIS = 'z'


# =============================================================================
# Data loading
# =============================================================================

def load_trajectory_data(npy_path=None):
    """Load the .npy trajectory file and return the raw dict."""
    path = npy_path or DEFAULT_NPY_PATH
    raw = np.load(path, allow_pickle=True).item()
    return raw, path


def downsample_trajectory(q_full, n_downsample=None):
    """Uniformly downsample a trajectory to *n_downsample* waypoints."""
    n = n_downsample or DEFAULT_N_DOWNSAMPLE
    if len(q_full) <= n:
        return q_full.copy()
    idx = np.linspace(0, len(q_full) - 1, n, dtype=int)
    return q_full[idx]


def downsample_by_distance(q_full, min_distance):
    """
    Downsample a trajectory by keeping only waypoints that are at least
    *min_distance* (joint-space L2 norm) away from the last kept waypoint.

    Always keeps the first and last waypoints.
    Returns the filtered array.
    """
    if len(q_full) < 2 or min_distance <= 0:
        return q_full.copy()
    filtered = [q_full[0]]
    for i in range(1, len(q_full)):
        if np.linalg.norm(q_full[i] - filtered[-1]) >= min_distance:
            filtered.append(q_full[i])
    # Always keep the last point
    if not np.allclose(filtered[-1], q_full[-1]):
        filtered.append(q_full[-1])
    return np.array(filtered)


def suggest_downsample_distances(q_full, factors=(2, 4, 8)):
    """
    Compute the min_distance values that would roughly downsample *q_full*
    by the given factors.

    Uses binary search: for each target count N/factor, find the distance
    threshold that yields approximately that many points.

    Returns a dict {factor: (distance, resulting_count)}.
    """
    n_orig = len(q_full)
    if n_orig < 3:
        return {f: (0.0, n_orig) for f in factors}

    # Precompute consecutive distances
    dists = np.array([np.linalg.norm(q_full[i] - q_full[i - 1])
                      for i in range(1, n_orig)])

    def count_at_distance(d):
        """How many waypoints survive with threshold *d*."""
        if d <= 0:
            return n_orig
        kept = 1
        acc = 0.0
        for di in dists:
            acc += di
            if acc >= d:
                kept += 1
                acc = 0.0
        return kept + 1  # +1 for last point always kept

    result = {}
    for factor in factors:
        target = max(n_orig // factor, 2)
        lo, hi = 0.0, float(dists.sum())
        # Binary search for threshold
        for _ in range(60):
            mid = (lo + hi) / 2
            c = count_at_distance(mid)
            if c > target:
                lo = mid
            else:
                hi = mid
        dist_val = (lo + hi) / 2
        final_count = count_at_distance(dist_val)
        result[factor] = (dist_val, final_count)
    return result


# =============================================================================
# Transformer setup
# =============================================================================

def make_transformer(log_level=logging.WARNING):
    """Create a TrajectoryTransformer with quiet logging."""
    logger = logging.getLogger('trajectory_transformer')
    logger.setLevel(log_level)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    return TrajectoryTransformer(logger=logger, panda=None)


# =============================================================================
# Pipeline helpers
# =============================================================================


def transform_trajectory(transformer, q_trajectory, tilt_deg, rot_deg,
                         tilt_direction=None, rotation_axis=None,
                         use_pink=False):
    """
    FK → Cartesian transform → IK.

    Returns (q_transformed, ik_valid, ik_total).
    *q_transformed* may be shorter than input if IK fails for some poses.

    If *use_pink* is True, uses the pink differential-IK backend with
    interpolation disabled (only successfully solved waypoints are returned).
    """
    tilt_dir = tilt_direction if tilt_direction is not None else DEFAULT_TILT_DIRECTION
    rot_axis = rotation_axis or DEFAULT_ROTATION_AXIS

    tilt_rad = np.radians(tilt_deg)
    rot_rad = np.radians(rot_deg)

    # FK
    positions, orientations = transformer.joints_to_cartesian(q_trajectory)

    # Cartesian transform
    if tilt_deg == 0 and rot_deg == 0:
        trans_pos, trans_ori = positions, orientations
    else:
        trans_pos, trans_ori = transformer.apply_transformation(
            positions, orientations,
            translation=np.zeros(3),
            rotation_axis=rot_axis,
            rotation_angle=rot_rad,
            rotate_orientation=True,
            tilt_angle=tilt_rad,
            tilt_direction=tilt_dir,
            tilt_trajectory=False,
        )

    # IK
    q_init = q_trajectory[0]
    if use_pink:
        q_transformed, valid_idx = transformer.cartesian_to_joints_pink(
            trans_pos, trans_ori, q_init=q_init, skip_interpolation=True,
        )
    else:
        q_transformed, valid_idx = transformer.cartesian_to_joints(
            trans_pos, trans_ori, q_init=q_init, use_hqp=False,
        )

    return q_transformed, len(q_transformed), len(trans_pos)



# =============================================================================
# Full pipeline: test one combination
# =============================================================================

def test_combination(transformer, q_trajectory, tilt_deg, rot_deg,
                     height_limit=None, speed_factor=None,
                     max_deviation=None, use_pink=False):
    """
    Run the full pipeline for one tilt/rotation combo.
    Returns a result dict with status flags, data for plotting, and errors.

    If *use_pink* is True, use the pink differential-IK backend with
    interpolation disabled.
    """
    hl = height_limit if height_limit is not None else DEFAULT_HEIGHT_LIMIT
    sf = speed_factor if speed_factor is not None else DEFAULT_SPEED_FACTOR
    md = max_deviation if max_deviation is not None else DEFAULT_MAX_DEVIATION

    result = dict(
        tilt_deg=tilt_deg, rot_deg=rot_deg,
        ik_success_rate=0.0, ik_valid=0, ik_total=0,
        joint_traj_ok=False, joint_traj_n_waypoints_before=0,
        joint_traj_n_waypoints=0,
        joint_traj_duration=0.0, joint_traj_reduced=False,
        height_constrained_ok=False, height_constrained_duration=0.0,
        error=None,
        q_original=q_trajectory, q_transformed=None,
        joint_traj=None, constrained_traj=None,
    )

    # --- Transform (FK → Cartesian → IK) ---
    try:
        q_transformed, ik_valid, ik_total = transform_trajectory(
            transformer, q_trajectory, tilt_deg, rot_deg, use_pink=use_pink)
    except Exception as e:
        result['error'] = f'Transform failed: {e}'
        return result

    result['ik_total'] = ik_total
    result['ik_valid'] = ik_valid
    result['ik_success_rate'] = ik_valid / ik_total * 100 if ik_total else 0

    if ik_valid < 2:
        result['error'] = f'IK failed: only {ik_valid} valid'
        return result
    result['q_transformed'] = q_transformed

    # --- JointTrajectory (gradual reduction) ---
    try:
        jt, n_used, n_before, reduced = build_joint_trajectory(
            q_transformed, speed_factor=sf, max_deviation=md)
        result['joint_traj_ok'] = True
        result['joint_traj_n_waypoints_before'] = n_before
        result['joint_traj_n_waypoints'] = n_used
        result['joint_traj_duration'] = jt.get_duration()
        result['joint_traj_reduced'] = reduced
        result['joint_traj'] = jt
    except RuntimeError as e:
        result['error'] = str(e)
        return result

    # --- HeightConstrainedJointTrajectory ---
    try:
        ct, ct_type = build_height_constrained(jt, height_limit=hl, max_deviation=md)
        result['height_constrained_ok'] = True
        result['height_constrained_duration'] = ct.get_duration()
        result['height_constrained_type'] = ct_type
        result['constrained_traj'] = ct
    except Exception as e:
        result['error'] = f'HeightConstrained failed: {e}'

    return result


# =============================================================================
# Problem detection
# =============================================================================

def is_problem(result):
    """Return True if this result should be flagged as a problem."""
    if result['error'] is not None:
        return True
    if result['ik_success_rate'] < 95:
        return True
    return False


# =============================================================================
# Plotting
# =============================================================================

def _sample_trajectory(traj, n_samples=200):
    """Sample joint positions from a trajectory object at uniform time steps."""
    ts = np.linspace(0, traj.get_duration(), n_samples)
    return ts, np.array([traj.get_joint_positions(t).flatten() for t in ts])


def _compute_heights(q_array):
    """Compute end-effector heights for an array of joint configurations."""
    return np.array([panda_py.fk(q)[2, 3] for q in q_array])


def plot_problem(result, save_dir, height_limit=None):
    """
    Save a compact diagnostic figure for one problematic combination.
    4×2 grid: 7 joint plots + height profile.
    Returns the saved file path.
    """
    import matplotlib.pyplot as plt

    hl = height_limit if height_limit is not None else DEFAULT_HEIGHT_LIMIT
    tag = result['tag']
    err = result.get('error') or 'reduced waypoints / low IK rate'

    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    fig.suptitle(f"PROBLEM: {tag}\n{err}",
                 fontsize=11, fontweight='bold', color='red')

    q_orig = result['q_original']
    prog_orig = np.linspace(0, 1, len(q_orig))
    h_orig = _compute_heights(q_orig)

    for j in range(7):
        ax = axes[j // 2, j % 2]
        ax.plot(prog_orig, np.degrees(q_orig[:, j]),
                'b-', alpha=0.5, lw=1, label='Original')

        if result.get('q_transformed') is not None:
            q_t = result['q_transformed']
            ax.plot(np.linspace(0, 1, len(q_t)), np.degrees(q_t[:, j]),
                    'r-', alpha=0.7, lw=1, label='Transformed')

        if result.get('joint_traj') is not None:
            ts, q_jt = _sample_trajectory(result['joint_traj'])
            ax.plot(ts / result['joint_traj'].get_duration(),
                    np.degrees(q_jt[:, j]),
                    color='orange', lw=1.3, label='JointTraj')

        if result.get('constrained_traj') is not None:
            ts, q_ct = _sample_trajectory(result['constrained_traj'])
            ax.plot(ts / result['constrained_traj'].get_duration(),
                    np.degrees(q_ct[:, j]),
                    'g--', lw=1.3, label='HeightConstr')

        ax.set_ylabel(f'J{j+1} [°]')
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(fontsize=6)

    # Height profile
    ax_h = axes[3, 1]
    ax_h.plot(prog_orig, h_orig * 100, 'b-', alpha=0.5, lw=1, label='Original')

    if result.get('q_transformed') is not None:
        q_t = result['q_transformed']
        h_t = _compute_heights(q_t)
        ax_h.plot(np.linspace(0, 1, len(q_t)), h_t * 100,
                  'r-', alpha=0.7, lw=1, label='Transformed')

    if result.get('joint_traj') is not None:
        ts, q_jt = _sample_trajectory(result['joint_traj'])
        h_jt = _compute_heights(q_jt)
        ax_h.plot(ts / result['joint_traj'].get_duration(), h_jt * 100,
                  color='orange', lw=1.3, label='JointTraj')

    if result.get('constrained_traj') is not None:
        ts, q_ct = _sample_trajectory(result['constrained_traj'])
        h_ct = _compute_heights(q_ct)
        ax_h.plot(ts / result['constrained_traj'].get_duration(), h_ct * 100,
                  'g--', lw=1.3, label='HeightConstr')

    ax_h.axhline(y=hl * 100, color='r', ls=':', lw=1.5,
                 label=f'Limit ({hl*100:.0f} cm)')
    ax_h.set_ylabel('Height [cm]')
    ax_h.set_xlabel('Progress [0–1]')
    ax_h.legend(fontsize=6)
    ax_h.grid(True, alpha=0.3)

    plt.tight_layout()
    safe_name = tag.replace('/', '_').replace('°', '').replace('=', '')
    fpath = os.path.join(save_dir, f'problem_{safe_name}.png')
    plt.savefig(fpath, dpi=120)
    plt.close()
    return fpath


def plot_full_pipeline(result, height_limit=None, show=True, save_path=None):
    """
    Detailed 4×2 figure for one pipeline result, showing all 4 trajectory
    stages (original → transformed → JointTraj → HeightConstrained) plus
    height profile.  Suitable for the single-run debug script.
    """
    import matplotlib.pyplot as plt

    hl = height_limit if height_limit is not None else DEFAULT_HEIGHT_LIMIT
    tag = result.get('tag', f"tilt={result['tilt_deg']}° rot={result['rot_deg']}°")

    fig, axes = plt.subplots(4, 2, figsize=(16, 14))

    # Title with status
    status_parts = []
    status_parts.append(f"IK: {result['ik_success_rate']:.0f}%")
    if result['joint_traj_ok']:
        jt_info = f"{result['joint_traj_n_waypoints']} wp"
        if result['joint_traj_reduced']:
            jt_info += " (reduced!)"
        jt_info += f", {result['joint_traj_duration']:.1f}s"
        status_parts.append(f"JT: {jt_info}")
    if result['height_constrained_ok']:
        status_parts.append(f"HC: {result['height_constrained_duration']:.1f}s")
    if result.get('error'):
        status_parts.append(f"Error: {result['error']}")

    fig.suptitle(f"{tag}\n{' | '.join(status_parts)}",
                 fontsize=12, fontweight='bold')

    q_orig = result['q_original']
    prog_orig = np.linspace(0, 1, len(q_orig))
    h_orig = _compute_heights(q_orig)

    for j in range(7):
        ax = axes[j // 2, j % 2]

        # 1. Original
        ax.plot(prog_orig, np.degrees(q_orig[:, j]),
                'b-', alpha=0.4, lw=1, label='Original')

        # 2. Transformed
        if result.get('q_transformed') is not None:
            q_t = result['q_transformed']
            ax.plot(np.linspace(0, 1, len(q_t)), np.degrees(q_t[:, j]),
                    'r-', alpha=0.6, lw=1, label='Transformed (IK)')

        # 3. JointTrajectory
        if result.get('joint_traj') is not None:
            ts, q_jt = _sample_trajectory(result['joint_traj'])
            ax.plot(ts / result['joint_traj'].get_duration(),
                    np.degrees(q_jt[:, j]),
                    color='orange', lw=1.5, label='JointTrajectory')

        # 4. HeightConstrained
        if result.get('constrained_traj') is not None:
            ts, q_ct = _sample_trajectory(result['constrained_traj'])
            ax.plot(ts / result['constrained_traj'].get_duration(),
                    np.degrees(q_ct[:, j]),
                    'g--', lw=1.5, label='HeightConstrained')

        ax.set_ylabel(f'Joint {j+1} [°]')
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(fontsize=7, loc='best')

    # Height profile
    ax_h = axes[3, 1]
    ax_h.plot(prog_orig, h_orig * 100, 'b-', alpha=0.4, lw=1, label='Original')

    if result.get('q_transformed') is not None:
        q_t = result['q_transformed']
        h_t = _compute_heights(q_t)
        ax_h.plot(np.linspace(0, 1, len(q_t)), h_t * 100,
                  'r-', alpha=0.6, lw=1, label='Transformed')

    if result.get('joint_traj') is not None:
        ts, q_jt = _sample_trajectory(result['joint_traj'])
        h_jt = _compute_heights(q_jt)
        ax_h.plot(ts / result['joint_traj'].get_duration(), h_jt * 100,
                  color='orange', lw=1.5, label='JointTrajectory')

    if result.get('constrained_traj') is not None:
        ts, q_ct = _sample_trajectory(result['constrained_traj'])
        h_ct = _compute_heights(q_ct)
        ax_h.plot(ts / result['constrained_traj'].get_duration(), h_ct * 100,
                  'g--', lw=1.5, label='HeightConstrained')

    ax_h.axhline(y=hl * 100, color='r', ls=':', lw=1.5,
                 label=f'Limit ({hl*100:.0f} cm)')
    ax_h.set_ylabel('Height [cm]')
    ax_h.set_xlabel('Progress [0–1]')
    ax_h.legend(fontsize=7, loc='best')
    ax_h.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved figure → {save_path}")
    if show:
        plt.show()
    else:
        plt.close()

    return fig


# =============================================================================
# Summary printing
# =============================================================================

def _get_problem_reason(p):
    """Return a short human-readable reason why this result is a problem."""
    reasons = []
    if p.get('error'):
        reasons.append(p['error'][:40])
    if p['ik_success_rate'] < 95:
        reasons.append(f"IK low ({p['ik_success_rate']:.0f}%)")
    if p.get('joint_traj_reduced'):
        reasons.append(
            f"JT reduced {p['joint_traj_n_waypoints_before']}"
            f"→{p['joint_traj_n_waypoints']}")
    if not p.get('height_constrained_ok') and p.get('joint_traj_ok'):
        reasons.append("HC failed")
    return ' + '.join(reasons) if reasons else '?'


def print_summary(problems, total):
    """Print a summary table of problem cases."""
    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total combinations tested: {total}")
    print(f"Passed: {total - len(problems)}")
    print(f"Problems: {len(problems)}")

    if problems:
        hdr = (f"  {'Tag':<40s} {'IK%':>5s} {'JT wp':>14s} "
               f"{'JT dur':>7s} {'HC':>4s}  Reason")
        print(f"\n{hdr}")
        print("  " + "-" * 110)
        for p in problems:
            ik_str = f"{p['ik_success_rate']:.0f}%"
            if p['joint_traj_ok']:
                if p['joint_traj_reduced']:
                    jt_wp = (f"{p['joint_traj_n_waypoints_before']}"
                             f"→{p['joint_traj_n_waypoints']}")
                else:
                    jt_wp = str(p['joint_traj_n_waypoints'])
                jt_dur = f"{p['joint_traj_duration']:.1f}s"
            else:
                jt_wp = "FAIL"
                jt_dur = "—"
            hc_str = "OK" if p['height_constrained_ok'] else "FAIL"
            reason = _get_problem_reason(p)
            print(f"  {p['tag']:<40s} {ik_str:>5s} {jt_wp:>14s} "
                  f"{jt_dur:>7s} {hc_str:>4s}  {reason}")
    else:
        print("\n✓ All combinations passed!")
