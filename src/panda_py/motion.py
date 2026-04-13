"""
Motion generation for the Panda robot. These are also directly
integrated as convenience methods of the :py:class:`panda_py.Panda` class.

This module also provides Python-level helpers for robust trajectory
construction, including prefiltering, gradual waypoint reduction, and
an optional Pink-based IK fallback for height-constrained trajectories.
"""

import logging
import numpy as np

# pylint: disable=no-name-in-module
from ._core import JointTrajectory, CartesianTrajectory, HeightConstrainedJointTrajectory

# Pink IK imports (optional – only needed for fallback correction)
try:
    import pink
    import pinocchio as pin
    from robot_descriptions.loaders.pinocchio import load_robot_description
    _PINK_AVAILABLE = True
except ImportError:
    _PINK_AVAILABLE = False

__all__ = [
    'JointTrajectory',
    'CartesianTrajectory',
    'HeightConstrainedJointTrajectory',
    'prefilter_waypoints',
    'build_joint_trajectory',
    'build_height_constrained',
]

# ── Default configuration ────────────────────────────────────────────────────
DEFAULT_HEIGHT_LIMIT = 0.15       # metres
DEFAULT_SPEED_FACTOR = 0.2
DEFAULT_MAX_DEVIATION = 0.0001

# ── Prefiltering ─────────────────────────────────────────────────────────────

def prefilter_waypoints(waypoints, min_distance):
    """
    Remove consecutive waypoints closer than *min_distance* (L2 norm).

    Always keeps the first and last waypoint.
    """
    waypoints = np.asarray(waypoints)
    if len(waypoints) < 2:
        return waypoints
    filtered = [waypoints[0]]
    for i in range(1, len(waypoints)):
        if np.linalg.norm(waypoints[i] - filtered[-1]) >= min_distance:
            filtered.append(waypoints[i])
    if not np.allclose(filtered[-1], waypoints[-1]):
        filtered.append(waypoints[-1])
    return np.array(filtered)


# ── Robust JointTrajectory builder ───────────────────────────────────────────

def build_joint_trajectory(q_waypoints, speed_factor=None, max_deviation=None,
                           timeout=60.0, min_waypoints=10):
    """
    Build a :class:`JointTrajectory` with prefiltering and gradual waypoint
    reduction.

    Near-duplicate waypoints are removed first (threshold = 2.5× *max_deviation*).
    If the time-optimal planner fails, the waypoints are halved repeatedly down
    to *min_waypoints*.

    Parameters
    ----------
    q_waypoints : array-like, shape (N, 7)
        Joint-space waypoints.
    speed_factor : float, optional
        Speed factor (default ``DEFAULT_SPEED_FACTOR``).
    max_deviation : float, optional
        Max blending deviation (default ``DEFAULT_MAX_DEVIATION``).
    timeout : float
        Planner timeout in seconds.
    min_waypoints : int
        Minimum number of waypoints to try.

    Returns
    -------
    (JointTrajectory, n_used, n_before, was_reduced)
    """
    sf = speed_factor if speed_factor is not None else DEFAULT_SPEED_FACTOR
    md = max_deviation if max_deviation is not None else DEFAULT_MAX_DEVIATION
    log = logging.getLogger('motion')

    q_filtered = prefilter_waypoints(q_waypoints, 2.5 * md)
    n_full = len(q_filtered)
    log.info('build_joint_trajectory: %d input → %d after prefilter '
             '(threshold=%.6f), sf=%.2f, md=%.6f',
             len(q_waypoints), n_full, 2.5 * md, sf, md)

    # Build candidate counts: full, half, quarter, … down to min_waypoints
    candidates = []
    n = n_full
    while n >= min_waypoints:
        candidates.append(n)
        n = n // 2

    for n_try in candidates:
        if n_try == n_full:
            q_try = q_filtered
        else:
            idx = np.linspace(0, len(q_filtered) - 1, n_try, dtype=int)
            q_try = q_filtered[idx]
        wps = [q.reshape(7, 1) for q in q_try]
        try:
            jt = JointTrajectory(
                waypoints=wps, speed_factor=sf,
                max_deviation=md, timeout=timeout,
            )
            log.info('build_joint_trajectory: succeeded with %d waypoints '
                     '(duration=%.2f s%s)',
                     n_try, jt.get_duration(),
                     ', REDUCED' if n_try < n_full else '')
            return jt, n_try, n_full, (n_try < n_full)
        except Exception:
            log.info('build_joint_trajectory: failed with %d waypoints, '
                     'trying fewer...', n_try)
            continue

    raise RuntimeError(
        f'JointTrajectory failed at all counts ({n_full}\u2192{candidates[-1]})')


# ── HeightConstrainedJointTrajectory builder (with Pink fallback) ────────────

def build_height_constrained(joint_traj, height_limit=None, dt=0.01,
                             max_deviation=0.0001, speed_factor=None):
    """
    Build a :class:`HeightConstrainedJointTrajectory` from *joint_traj*.

    Uses the same parameters (dt, max_deviation, speed_factor) as the C++
    ``Panda::moveToJointPositionWithHeightLimit`` for consistent behaviour.

    If the C++ planner fails **and** Pink is available, sample the base
    trajectory, fix height-violating waypoints with Pink differential IK,
    and build a regular JointTrajectory from the corrected waypoints.
    """
    hl = height_limit if height_limit is not None else DEFAULT_HEIGHT_LIMIT
    sf = speed_factor if speed_factor is not None else DEFAULT_SPEED_FACTOR
    log = logging.getLogger('motion')
    try:
        hc = HeightConstrainedJointTrajectory(
            base_trajectory=joint_traj,
            height_limit=hl,
            dt=dt,
            max_deviation=max_deviation,
            speed_factor=sf,
        )
        log.info('build_height_constrained: C++ HeightConstrainedJointTrajectory succeeded '
                 '(duration=%.2f s)', hc.get_duration())
        return hc, 'HeightConstrainedJointTrajectory'
    except Exception as first_err:
        log.warning(
            'HeightConstrainedJointTrajectory failed: %s – attempting Pink fallback.', first_err)

        if not _PINK_AVAILABLE:
            raise RuntimeError(
                f'HeightConstrained failed ({first_err}) and Pink is not available for fallback.'
            ) from first_err

        corrected = _correct_height_with_pink(
            joint_traj, height_limit=hl, dt=dt)

        # Build a regular JointTrajectory from the corrected waypoints
        wps = [q.reshape(7, 1) for q in corrected]
        try:
            jt = JointTrajectory(
                waypoints=wps,
                speed_factor=sf,
                max_deviation=max_deviation,
                timeout=60.0,
            )
            log.info(
                'Pink fallback succeeded: %d waypoints, duration %.2f s.',
                len(corrected), jt.get_duration())
            return jt, 'JointTrajectory (Pink height-corrected)'
        except Exception as fallback_err:
            raise RuntimeError(
                f'HeightConstrained failed ({first_err}) and Pink fallback '
                f'also failed ({fallback_err}).'
            ) from fallback_err


# ── Pink-based height correction (private) ───────────────────────────────────

def _correct_height_with_pink(joint_traj, height_limit, dt=0.01):
    """
    Sample *joint_traj*, detect waypoints whose EE height < *height_limit*,
    and re-solve those with Pink differential IK (lifting the target to
    *height_limit*).  Returns an ndarray (N, 7) of corrected waypoints.
    """
    from . import fk as panda_fk  # avoid circular import at module level

    duration = joint_traj.get_duration()
    n_samples = max(int(np.ceil(duration / dt)) + 1, 2)
    ts = np.linspace(0, duration, n_samples)

    # Load Pinocchio robot
    robot = load_robot_description('panda_description')
    model, data = robot.model, robot.data
    frames = [f.name for f in model.frames]
    ee_frame = 'panda_hand_tcp' if 'panda_hand_tcp' in frames else 'panda_link8'
    fid = model.getFrameId(ee_frame)

    corrected = []
    q_prev_7 = np.array(joint_traj.get_joint_positions(0.0)).flatten()
    num_fixed = 0

    for t in ts:
        q7 = np.array(joint_traj.get_joint_positions(t)).flatten()
        T = panda_fk(q7)
        z = T[2, 3]

        if z >= height_limit:
            corrected.append(q7)
            q_prev_7 = q7
            continue

        # Lift target to height_limit
        target_pos = T[:3, 3].copy()
        target_pos[2] = height_limit
        target_rot = T[:3, :3]
        target_se3 = pin.SE3(target_rot, target_pos)

        q_new = _pink_ik_solve(
            model, data, fid, target_se3, q_prev_7, ee_frame)
        if q_new is not None:
            corrected.append(q_new)
            q_prev_7 = q_new
            num_fixed += 1
        else:
            corrected.append(q7)
            q_prev_7 = q7

    logging.getLogger('motion').info(
        'Pink height correction: %d/%d waypoints fixed.', num_fixed, len(ts))
    return np.array(corrected)


def _pink_ik_solve(model, data, frame_id, target_se3, q_init_7,
                   ee_frame='panda_hand_tcp', max_iter=100, dt_ik=1e-1,
                   tol=1e-4):
    """
    Solve one IK problem with Pink differential IK.
    Returns (7,) ndarray or None on failure.
    """
    nq = model.nq  # typically 9 for panda (7 arm + 2 fingers)
    q_full = np.zeros(nq)
    q_full[:7] = q_init_7

    configuration = pink.Configuration(model, data, q_full)

    ee_task = pink.tasks.FrameTask(
        ee_frame, position_cost=1.0, orientation_cost=1.0)
    ee_task.set_target(target_se3)

    for _ in range(max_iter):
        velocity = pink.solve_ik(
            configuration, [ee_task], dt_ik, solver='quadprog')
        configuration.integrate_inplace(velocity, dt_ik)

        err = ee_task.compute_error(configuration)
        if np.linalg.norm(err) < tol:
            return configuration.q[:7].copy()

    return None
