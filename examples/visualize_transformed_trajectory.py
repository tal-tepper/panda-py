"""
Visualize Transformed Trajectory

This script:
1. Takes an elevator index and transformation parameters (dx, dy, rotation_angle, tilt_angle, tilt_direction)
2. Reads lift data from panda_config_local.py
3. Loads the primitive trajectory
4. Applies transformation using TrajectoryTransformer.apply_transformation
5. Computes trajectory with height limit using panda_py.compute_trajectory_with_height_limit
6. Creates figures comparing original and transformed joint positions (q)

Usage:
    python visualize_transformed_trajectory.py <elevator_index> --dx <dx> --dy <dy> --rotation <angle_deg> --tilt <angle_deg> --tilt_dir <dx> <dy>

Example:
    python visualize_transformed_trajectory.py 0 --dx 0.05 --dy 0.02 --rotation 30 --tilt 10 --tilt_dir 1.0 0.0
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
import os
import logging

from panda_config_local import get_config
from trajectory_transformer import TrajectoryTransformer


def setup_logger() -> logging.Logger:
    """Set up a logger for the script."""
    logger = logging.getLogger('visualize_trajectory')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    return logger


class MockPanda:
    """Mock Panda class for offline trajectory visualization (no robot connection)."""
    def __init__(self, q_init: np.ndarray):
        self._q = q_init
    
    @property
    def q(self) -> np.ndarray:
        return self._q


def visualize_joint_trajectories(
    q_original: np.ndarray,
    q_transformed: np.ndarray,
    q_height_limited: Optional[np.ndarray] = None,
    q_base: Optional[np.ndarray] = None,
    time_original: Optional[np.ndarray] = None,
    time_transformed: Optional[np.ndarray] = None,
    time_height_limited: Optional[np.ndarray] = None,
    time_base: Optional[np.ndarray] = None,
    title: str = "Joint Trajectory Comparison",
    joint_names: Optional[list] = None
) -> plt.Figure:
    """
    Create a figure comparing original, transformed, and height-limited joint trajectories.
    
    Args:
        q_original: Original joint trajectory [n_waypoints, 7]
        q_transformed: Transformed joint trajectory [n_waypoints, 7]
        q_height_limited: Height-limited joint trajectory [n_waypoints, 7] (optional)
        q_base: Base JointTrajectory (before height constraint) [n_waypoints, 7] (optional)
        time_original: Time array for original trajectory (optional, defaults to normalized [0,1])
        time_transformed: Time array for transformed trajectory (optional, defaults to normalized [0,1])
        time_height_limited: Time array for height-limited trajectory (optional)
        time_base: Time array for base trajectory (optional)
        title: Figure title
        joint_names: Optional list of joint names
        
    Returns:
        matplotlib Figure object
    """
    if joint_names is None:
        joint_names = [f'Joint {i+1}' for i in range(7)]
    
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    # Use provided time arrays or create normalized time [0, 1]
    if time_original is None:
        time_original = np.linspace(0, 1, len(q_original))
    if time_transformed is None:
        time_transformed = np.linspace(0, 1, len(q_transformed))
    
    for i in range(7):
        ax = axes[i]
        ax.plot(time_original, np.degrees(q_original[:, i]), 'b-', label='Original', linewidth=1.5)
        ax.plot(time_transformed, np.degrees(q_transformed[:, i]), 'r--', label='Transformed (IK)', linewidth=1.5)
        
        # Plot base trajectory if available
        if q_base is not None:
            if time_base is None:
                time_base = np.linspace(0, 1, len(q_base))
            ax.plot(time_base, np.degrees(q_base[:, i]), 'm:', 
                   label='Base JointTrajectory', linewidth=1.5, alpha=0.8)
        
        # Plot height-limited trajectory if available
        if q_height_limited is not None:
            if time_height_limited is None:
                time_height_limited = np.linspace(0, 1, len(q_height_limited))
            ax.plot(time_height_limited, np.degrees(q_height_limited[:, i]), 'g-.', 
                   label='Height Constrained', linewidth=1.5, alpha=0.8)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (degrees)')
        ax.set_title(joint_names[i])
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide the last subplot (8th) since we only have 7 joints
    axes[7].axis('off')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def visualize_cartesian_trajectory(
    positions_original: np.ndarray,
    positions_transformed: np.ndarray,
    positions_height_limited: Optional[np.ndarray] = None,
    positions_base: Optional[np.ndarray] = None,
    height_limit: Optional[float] = None,
    title: str = "Cartesian Trajectory Comparison"
) -> plt.Figure:
    """
    Create a figure comparing original, transformed, and height-limited Cartesian trajectories.
    
    Args:
        positions_original: Original positions [n_waypoints, 3]
        positions_transformed: Transformed positions [n_waypoints, 3]
        positions_height_limited: Height-limited positions [n_waypoints, 3] (optional)
        positions_base: Base JointTrajectory positions [n_waypoints, 3] (optional)
        height_limit: Height limit value to show on plot (optional)
        title: Figure title
        
    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=(14, 5))
    
    # 3D view
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(positions_original[:, 0], positions_original[:, 1], positions_original[:, 2],
             'b-', label='Original', linewidth=2)
    ax1.plot(positions_transformed[:, 0], positions_transformed[:, 1], positions_transformed[:, 2],
             'r-', label='Transformed', linewidth=2)
    
    # Plot base trajectory if available
    if positions_base is not None:
        ax1.plot(positions_base[:, 0], positions_base[:, 1], positions_base[:, 2],
                 'm:', label='Base JointTrajectory', linewidth=2, alpha=0.8)
        ax1.scatter(*positions_base[0], c='magenta', s=100, marker='o')
    
    # Plot height-limited trajectory if available
    if positions_height_limited is not None:
        ax1.plot(positions_height_limited[:, 0], positions_height_limited[:, 1], positions_height_limited[:, 2],
                 'g--', label='Height Constrained', linewidth=2, alpha=0.8)
        ax1.scatter(*positions_height_limited[0], c='green', s=100, marker='o')
    
    ax1.scatter(*positions_original[0], c='blue', s=100, marker='o', label='Original Start')
    ax1.scatter(*positions_transformed[0], c='red', s=100, marker='o', label='Transformed Start')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D View')
    ax1.legend(fontsize=8)
    
    # XY view (top-down)
    ax2 = fig.add_subplot(122)
    ax2.plot(positions_original[:, 0], positions_original[:, 1], 'b-', label='Original', linewidth=2)
    ax2.plot(positions_transformed[:, 0], positions_transformed[:, 1], 'r-', label='Transformed', linewidth=2)
    
    if positions_base is not None:
        ax2.plot(positions_base[:, 0], positions_base[:, 1], 'm:', 
                label='Base JointTrajectory', linewidth=2, alpha=0.8)
        ax2.scatter(positions_base[0, 0], positions_base[0, 1], c='magenta', s=100, marker='o')
    
    if positions_height_limited is not None:
        ax2.plot(positions_height_limited[:, 0], positions_height_limited[:, 1], 'g--', 
                label='Height Constrained', linewidth=2, alpha=0.8)
        ax2.scatter(positions_height_limited[0, 0], positions_height_limited[0, 1], c='green', s=100, marker='o')
    
    ax2.scatter(positions_original[0, 0], positions_original[0, 1], c='blue', s=100, marker='o')
    ax2.scatter(positions_transformed[0, 0], positions_transformed[0, 1], c='red', s=100, marker='o')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Top-Down View (XY)')
    ax2.legend(fontsize=8)
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def visualize_z_profile(
    positions_original: np.ndarray,
    positions_transformed: np.ndarray,
    positions_height_limited: Optional[np.ndarray] = None,
    positions_base: Optional[np.ndarray] = None,
    time_original: Optional[np.ndarray] = None,
    time_transformed: Optional[np.ndarray] = None,
    time_height_limited: Optional[np.ndarray] = None,
    time_base: Optional[np.ndarray] = None,
    height_limit: Optional[float] = None,
    title: str = "Z Height Profile"
) -> plt.Figure:
    """
    Create a figure showing Z height profiles over time.
    
    Args:
        positions_original: Original positions [n_waypoints, 3]
        positions_transformed: Transformed positions [n_waypoints, 3]
        positions_height_limited: Height-limited positions [n_waypoints, 3] (optional)
        positions_base: Base JointTrajectory positions [n_waypoints, 3] (optional)
        time_original: Time array for original trajectory (optional, defaults to normalized [0,1])
        time_transformed: Time array for transformed trajectory (optional, defaults to normalized [0,1])
        time_height_limited: Time array for height-limited trajectory (optional)
        time_base: Time array for base trajectory (optional)
        height_limit: Height limit value to show as horizontal line (optional)
        title: Figure title
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Use provided time arrays or create normalized time [0, 1]
    if time_original is None:
        time_original = np.linspace(0, 1, len(positions_original))
    if time_transformed is None:
        time_transformed = np.linspace(0, 1, len(positions_transformed))
    
    ax.plot(time_original, positions_original[:, 2], 'b-', label='Original', linewidth=2)
    ax.plot(time_transformed, positions_transformed[:, 2], 'r--', label='Transformed', linewidth=2)
    
    if positions_base is not None:
        if time_base is None:
            time_base = np.linspace(0, 1, len(positions_base))
        ax.plot(time_base, positions_base[:, 2], 'm:', 
               label='Base JointTrajectory', linewidth=2, alpha=0.8)
    
    if positions_height_limited is not None:
        if time_height_limited is None:
            time_height_limited = np.linspace(0, 1, len(positions_height_limited))
        ax.plot(time_height_limited, positions_height_limited[:, 2], 'g-.', 
               label='Height Constrained', linewidth=2, alpha=0.8)
    
    # Show height limit line if provided
    if height_limit is not None:
        ax.axhline(y=height_limit, color='orange', linestyle=':', linewidth=2, 
                  label=f'Height Limit ({height_limit:.3f} m)')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Z Height (m)')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize transformed trajectory')
    parser.add_argument('elevator_index', type=int, help='Index of elevator/lift data from panda_config_local (0-based)')
    parser.add_argument('--dx', type=float, default=0.0, help='Translation in X direction (meters)')
    parser.add_argument('--dy', type=float, default=0.0, help='Translation in Y direction (meters)')
    parser.add_argument('--dz', type=float, default=0.0, help='Translation in Z direction (meters)')
    parser.add_argument('--rotation', type=float, default=0.0, help='Rotation angle about Z-axis (degrees)')
    parser.add_argument('--tilt', type=float, default=0.0, help='Tilt angle (degrees)')
    parser.add_argument('--tilt_dir', type=float, nargs=2, default=[1.0, 0.0], help='Tilt direction [dx, dy] in XY plane')
    parser.add_argument('--height_limit', type=float, default=0.05, help='Height limit for trajectory computation (meters)')
    parser.add_argument('--speed_factor', type=float, default=0.2, help='Speed factor for trajectory computation')
    parser.add_argument('--max_deviation', type=float, default=0.01, help='Max deviation for height-constrained trajectory smoothing (default: 0.01)')
    parser.add_argument('--timeout', type=float, default=60.0, help='Timeout for trajectory computation (seconds, default: 60)')
    parser.add_argument('--primitive', type=str, default=None, help='Primitive name to use (default: auto-detect poke)')
    parser.add_argument('--save', type=str, default=None, help='Path to save figures (without extension)')
    parser.add_argument('--no_robot', action='store_true', help='Run without connecting to robot (offline mode)')
    
    args = parser.parse_args()
    
    logger = setup_logger()
    
    # Get configuration
    config = get_config()
    lift_data_list = config['lift_data']
    
    # Validate elevator index
    if args.elevator_index < 0 or args.elevator_index >= len(lift_data_list):
        logger.error(f"Invalid elevator_index. Must be between 0 and {len(lift_data_list) - 1}")
        logger.info("Available lifts:")
        for i, lift in enumerate(lift_data_list):
            logger.info(f"  {i}: {lift['lift_name']} - {lift['primitive_file']}")
        return
    
    # Get the selected lift data
    lift_data = lift_data_list[args.elevator_index]
    position = np.array(lift_data['position'])
    primitive_file = lift_data['primitive_file']
    lift_name = lift_data['lift_name']
    
    logger.info("=" * 60)
    logger.info(f"Selected lift: {lift_name}")
    logger.info(f"Position: {position}")
    logger.info(f"Primitive file: {primitive_file}")
    logger.info("=" * 60)
    
    # Construct full path to primitive file
    primitive_path = os.path.join('/home/talte/repos/tactile_panda/', 'primitive_files', primitive_file)
    
    if not os.path.exists(primitive_path):
        logger.error(f"Primitive file not found: {primitive_path}")
        return
    
    # Load primitive file to check available primitives
    primitive_data = np.load(primitive_path, allow_pickle=True).item()
    available_primitives = list(primitive_data.keys())
    logger.info(f"Available primitives: {available_primitives}")
    
    # Find primitive to use
    if args.primitive:
        primitive_name = args.primitive
        if primitive_name not in available_primitives:
            logger.error(f"Primitive '{primitive_name}' not found. Available: {available_primitives}")
            return
    else:
        # Auto-detect poke primitive
        poke_primitive = None
        for prim_name in available_primitives:
            if 'poke' in prim_name.lower():
                poke_primitive = prim_name
                break
        primitive_name = poke_primitive if poke_primitive else available_primitives[0]
    
    logger.info(f"Using primitive: '{primitive_name}'")
    
    # Transformation parameters
    translation = np.array([args.dx, args.dy, args.dz])
    rotation_angle = np.deg2rad(args.rotation)
    tilt_angle = np.deg2rad(args.tilt)
    tilt_direction = np.array(args.tilt_dir)
    
    logger.info(f"\nTransformation parameters:")
    logger.info(f"  Translation: [{args.dx:.4f}, {args.dy:.4f}, {args.dz:.4f}] m")
    logger.info(f"  Rotation: {args.rotation:.1f}° about Z-axis")
    logger.info(f"  Tilt: {args.tilt:.1f}° towards [{args.tilt_dir[0]:.2f}, {args.tilt_dir[1]:.2f}]")
    logger.info(f"  Height limit: {args.height_limit:.3f} m")
    logger.info(f"  Speed factor: {args.speed_factor:.2f}")
    
    # Initialize transformer (with mock panda for offline mode or real panda)
    if args.no_robot:
        logger.info("\nRunning in offline mode (no robot connection)")
        mock_panda = MockPanda(position)
        transformer = TrajectoryTransformer(logger, mock_panda)
    else:
        try:
            import panda_py
            logger.info(f"\nConnecting to robot at {config['hostname']}...")
            desk = panda_py.Desk(config['hostname'], config['username'], config['password'])
            desk.activate_fci()
            panda = panda_py.Panda(config['hostname'])
            transformer = TrajectoryTransformer(logger, panda)
        except Exception as e:
            logger.error(f"Failed to connect to robot: {e}")
            logger.info("Try running with --no_robot flag for offline mode")
            return
    
    # Use transform_trajectory to get the full transformation result
    q_init = position  # Use lift position as initial joint config
    result = transformer.transform_trajectory(
        npy_path=primitive_path,
        primitive_name=primitive_name,
        translation=translation,
        rotation_axis='z',
        rotation_angle=rotation_angle,
        rotate_orientation=False,
        tilt_angle=tilt_angle,
        tilt_direction=tilt_direction,
        tilt_trajectory=False,
        q_init=q_init,
        skip_downsampling=True,  # Keep all waypoints for visualization
        collect_ik_failure_details=True  # Collect detailed IK failure info
    )
    
    if result is None:
        logger.error("Trajectory transformation failed")
        return
    
    # Extract data from result
    q_original = result['q_original']
    q_transformed = result['q_transformed_full']
    positions_original = result['positions_original']
    positions_tf = result['positions_transformed']
    
    # Log IK failure details if any
    if 'ik_failure_details' in result and result['ik_failure_details']:
        failures = result['ik_failure_details']
        logger.info(f"\n{'='*60}")
        logger.info(f"IK FAILURE DETAILS: {len(failures)} failures ({result['ik_success_rate']:.1f}% success rate)")
        logger.info(f"{'='*60}")
        
        # Group failures by reason
        failure_by_reason = {}
        for f in failures:
            reason = f['reason']
            if reason not in failure_by_reason:
                failure_by_reason[reason] = []
            failure_by_reason[reason].append(f)
        
        for reason, reason_failures in failure_by_reason.items():
            logger.info(f"\n  {reason.upper()}: {len(reason_failures)} failures")
            # Show first few examples
            for i, f in enumerate(reason_failures[:5]):
                pos = f['position']
                logger.info(f"    [{f['index']:4d}] pos=[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}] - {f['details']}")
            if len(reason_failures) > 5:
                logger.info(f"    ... and {len(reason_failures) - 5} more")
        
        # Show positions of failed IK for visualization reference
        if 'positions_transformed_all' in result:
            positions_all = result['positions_transformed_all']
            failed_indices = [f['index'] for f in failures]
            logger.info(f"\n  Failed waypoints span:")
            failed_positions = positions_all[failed_indices]
            logger.info(f"    X: [{failed_positions[:, 0].min():.4f}, {failed_positions[:, 0].max():.4f}] m")
            logger.info(f"    Y: [{failed_positions[:, 1].min():.4f}, {failed_positions[:, 1].max():.4f}] m")
            logger.info(f"    Z: [{failed_positions[:, 2].min():.4f}, {failed_positions[:, 2].max():.4f}] m")
        logger.info(f"{'='*60}\n")
    
    # Prefilter waypoints and create JointTrajectory (if robot is connected)
    q_filtered = None
    positions_filtered = None
    joint_trajectory = None
    times_filtered = None
    q_base_sampled = None
    positions_base = None
    times_base = None
    
    def prefilter_waypoints(waypoints, min_distance):
        """
        Pre-filter waypoints to ensure minimum distance between consecutive points.
        This prevents blending circle overlap issues with max_deviation > 0.
        
        Args:
            waypoints: numpy array of shape (N, 7) with joint positions
            min_distance: minimum distance between consecutive waypoints (should be > 2*max_deviation)
        
        Returns:
            Filtered waypoints array
        """
        if len(waypoints) < 2:
            return waypoints
        
        filtered = [waypoints[0]]  # Always keep first waypoint
        
        for i in range(1, len(waypoints)):
            dist = np.linalg.norm(waypoints[i] - filtered[-1])
            if dist >= min_distance:
                filtered.append(waypoints[i])
        
        # Always keep last waypoint if it's not already included
        if not np.allclose(filtered[-1], waypoints[-1]):
            filtered.append(waypoints[-1])
        
        return np.array(filtered)
    
    if not args.no_robot:
        try:
            from panda_py._core import JointTrajectory, HeightConstrainedJointTrajectory
            import time as time_module
            
            max_dev = 0.0001  # Very small deviation to keep more waypoints
            min_dist_required = 2.5 * max_dev  # Ensure waypoints are far enough apart
            
            logger.info(f"\nPre-filtering waypoints for JointTrajectory...")
            logger.info(f"  max_deviation = {max_dev}")
            logger.info(f"  min_distance >= {min_dist_required:.6f} rad (rule: > 2 * max_deviation)")
            
            # Original trajectory stats
            original_distances = [np.linalg.norm(q_transformed[i+1] - q_transformed[i]) 
                                 for i in range(len(q_transformed)-1)]
            logger.info(f"\n  Original trajectory:")
            logger.info(f"    Waypoints: {len(q_transformed)}")
            logger.info(f"    Min distance: {min(original_distances):.6f} rad")
            logger.info(f"    Max distance: {max(original_distances):.6f} rad")
            logger.info(f"    Mean distance: {np.mean(original_distances):.6f} rad")
            
            # Pre-filter the trajectory for minimum distance
            q_prefiltered = prefilter_waypoints(q_transformed, min_dist_required)
            
            if len(q_prefiltered) > 1:
                filtered_distances = [np.linalg.norm(q_prefiltered[i+1] - q_prefiltered[i]) 
                                     for i in range(len(q_prefiltered)-1)]
                logger.info(f"\n  After distance pre-filtering:")
                logger.info(f"    Waypoints: {len(q_prefiltered)} (removed {len(q_transformed) - len(q_prefiltered)})")
                logger.info(f"    Min distance: {min(filtered_distances):.6f} rad")
                logger.info(f"    Max distance: {max(filtered_distances):.6f} rad")
                logger.info(f"    Mean distance: {np.mean(filtered_distances):.6f} rad")
            
            # Convert to list of (7, 1) arrays
            waypoints_filtered = [q.reshape(7, 1) for q in q_prefiltered]
            
            logger.info(f"\n  Creating base JointTrajectory with {len(waypoints_filtered)} waypoints...")
            logger.info(f"    speed_factor={args.speed_factor}, max_deviation={max_dev}, timeout=60s")
            
            start_time = time_module.time()
            try:
                base_trajectory = JointTrajectory(
                    waypoints=waypoints_filtered,
                    speed_factor=args.speed_factor,
                    max_deviation=max_dev,
                    timeout=10.0
                )
                elapsed = time_module.time() - start_time
                logger.info(f"  ✓ Base JointTrajectory created in {elapsed:.3f}s")
                logger.info(f"    Duration: {base_trajectory.get_duration():.2f}s")
            except RuntimeError as e:
                logger.warning(f"  JointTrajectory creation failed: {e}")
                logger.info("  Retrying with max_deviation=0...")
                max_dev = 0.0
                base_trajectory = JointTrajectory(
                    waypoints=waypoints_filtered,
                    speed_factor=args.speed_factor,
                    max_deviation=0.0,
                    timeout=10.0
                )
                elapsed = time_module.time() - start_time
                logger.info(f"  ✓ Base JointTrajectory created (max_deviation=0) in {elapsed:.3f}s")
                logger.info(f"    Duration: {base_trajectory.get_duration():.2f}s")
            
            # Sample the base trajectory for visualization IMMEDIATELY after creation
            # (before attempting height-constrained trajectory which may fail)
            try:
                logger.info(f"\n  Sampling base trajectory...")
                base_duration = base_trajectory.get_duration()
                n_base_samples = int(base_duration * 1000)  # 1000Hz sampling
                times_base_raw = np.linspace(0, base_duration, n_base_samples)
                
                q_base_sampled = np.array([base_trajectory.get_joint_positions(t).flatten() for t in times_base_raw])
                
                # Scale time to match original recording duration for visualization
                original_duration = len(q_original) / 1000.0
                times_base = times_base_raw * (original_duration / base_duration)
                
                logger.info(f"  Sampled {len(q_base_sampled)} points (duration: {times_base[-1]:.3f}s scaled)")
                
                # Convert base q to Cartesian for visualization
                positions_base, _ = transformer.joints_to_cartesian(q_base_sampled)
                
                # Log base trajectory Z range
                z_base_max = positions_base[:, 2].max()
                z_base_min = positions_base[:, 2].min()
                logger.info(f"  Base Z range: [{z_base_min:.4f}, {z_base_max:.4f}] m")
            except Exception as e:
                logger.warning(f"Could not sample base trajectory: {e}")
            
            # Create height-constrained trajectory with retry logic
            logger.info(f"\n  Creating HeightConstrainedJointTrajectory...")
            logger.info(f"    height_limit = {args.height_limit} m")
            logger.info(f"    max_deviation = {args.max_deviation}")
            logger.info(f"    speed_factor = {args.speed_factor}")
            logger.info(f"    timeout = {args.timeout}s")
            
            # Try with increasing max_deviation and dt values if initial attempt fails
            constrained_trajectory = None
            attempt_params = [
                {'dt': 0.001, 'max_deviation': args.max_deviation, 'max_waypoints': 2000},
                {'dt': 0.002, 'max_deviation': args.max_deviation * 2, 'max_waypoints': 1500},
                {'dt': 0.005, 'max_deviation': args.max_deviation * 5, 'max_waypoints': 1000},
                {'dt': 0.01, 'max_deviation': args.max_deviation * 10, 'max_waypoints': 500},
            ]
            
            for attempt_idx, params in enumerate(attempt_params):
                try:
                    logger.info(f"    Attempt {attempt_idx + 1}: dt={params['dt']}, max_deviation={params['max_deviation']:.4f}, max_waypoints={params['max_waypoints']}")
                    constrained_trajectory = HeightConstrainedJointTrajectory(
                        base_trajectory=base_trajectory,
                        height_limit=args.height_limit,
                        dt=params['dt'],
                        max_deviation=params['max_deviation'],
                        speed_factor=args.speed_factor,
                        timeout=args.timeout,
                        max_waypoints=params['max_waypoints']
                    )
                    logger.info(f"  ✓ HeightConstrainedJointTrajectory created on attempt {attempt_idx + 1}!")
                    break
                except RuntimeError as e:
                    logger.warning(f"    Attempt {attempt_idx + 1} failed: {e}")
                    if attempt_idx == len(attempt_params) - 1:
                        logger.warning("All attempts to create HeightConstrainedJointTrajectory failed")
                        # Don't raise - continue without constrained trajectory
            
            if constrained_trajectory is not None:
                logger.info(f"    Duration: {constrained_trajectory.get_duration():.2f}s")
                
                # Compare base vs constrained at a few time points
                logger.info(f"\n  Comparing base vs constrained trajectory:")
                for t in [0.0, base_trajectory.get_duration() / 2, base_trajectory.get_duration()]:
                    q_base_pt = base_trajectory.get_joint_positions(t).flatten()
                    q_const = constrained_trajectory.get_joint_positions(t).flatten()
                    
                    pose_base = panda_py.fk(q_base_pt)
                    pose_const = panda_py.fk(q_const)
                    
                    h_base = pose_base[2, 3]
                    h_const = pose_const[2, 3]
                    
                    logger.info(f"    t={t:.2f}s: base_height={h_base:.4f}m, constrained_height={h_const:.4f}m")
            
            # Sample the constrained trajectory for visualization (if it was created)
            if constrained_trajectory is not None:
                try:
                    logger.info(f"\n  Sampling constrained trajectory...")
                    duration = constrained_trajectory.get_duration()
                    n_samples = int(duration * 1000)  # 1000Hz sampling
                    times = np.linspace(0, duration, n_samples)
                    
                    q_sampled = np.array([constrained_trajectory.get_joint_positions(t).flatten() for t in times])
                    
                    # Scale time to match original recording duration for visualization
                    original_duration = len(q_original) / 1000.0
                    times_scaled = times * (original_duration / duration)
                    
                    logger.info(f"  Sampled {len(q_sampled)} points (duration: {times_scaled[-1]:.3f}s scaled)")
                    
                    # Convert sampled q to Cartesian for visualization
                    positions_filtered, _ = transformer.joints_to_cartesian(q_sampled)
                    
                    # Verify height constraint
                    z_max = positions_filtered[:, 2].max()
                    z_min = positions_filtered[:, 2].min()
                    logger.info(f"  Constrained Z range: [{z_min:.4f}, {z_max:.4f}] m (limit: {args.height_limit} m)")
                    
                    # Use sampled trajectory for visualization
                    q_filtered = q_sampled
                    times_filtered = times_scaled
                except Exception as e:
                    logger.warning(f"Could not sample constrained trajectory: {e}")
            else:
                logger.warning("HeightConstrainedJointTrajectory was not created, skipping constrained trajectory visualization")
                
        except Exception as e:
            logger.warning(f"Could not create trajectories: {e}")
            import traceback
            traceback.print_exc()
    
    # Create visualizations
    logger.info("\nGenerating visualizations...")
    
    # All trajectories recorded/sampled at 1000Hz, so time = index / 1000
    time_original = np.arange(len(q_original)) / 1000.0  # Recording at 1000Hz
    
    # Transformed uses valid_indices to map back to original recording time
    valid_indices = result['valid_indices']
    time_transformed = np.array(valid_indices) / 1000.0  # Same time base as original
    
    # times_filtered is already in seconds (set above from get_joint_trajectory at 1000Hz)
    
    # Joint trajectory comparison (q)
    fig_joints = visualize_joint_trajectories(
        q_original,
        q_transformed,
        q_height_limited=q_filtered,
        q_base=q_base_sampled,
        time_original=time_original,
        time_transformed=time_transformed,
        time_height_limited=times_filtered,
        time_base=times_base,
        title=f"{lift_name} - {primitive_name}\n"
              f"Translation: [{args.dx:.3f}, {args.dy:.3f}, {args.dz:.3f}]m, "
              f"Rotation: {args.rotation:.1f}°, Tilt: {args.tilt:.1f}°"
    )
    
    # Cartesian trajectory comparison
    fig_cartesian = visualize_cartesian_trajectory(
        positions_original,
        positions_tf,
        positions_height_limited=positions_filtered,
        positions_base=positions_base,
        height_limit=args.height_limit,
        title=f"{lift_name} - {primitive_name} (Cartesian)"
    )
    
    # Z height profile
    fig_z_profile = visualize_z_profile(
        positions_original,
        positions_tf,
        positions_height_limited=positions_filtered,
        positions_base=positions_base,
        time_original=time_original,
        time_transformed=time_transformed,
        time_height_limited=times_filtered,
        time_base=times_base,
        height_limit=args.height_limit,
        title=f"{lift_name} - {primitive_name} (Z Height Profile)"
    )
    
    # Save figures if requested
    if args.save:
        fig_joints.savefig(f"{args.save}_joints.png", dpi=150, bbox_inches='tight')
        fig_cartesian.savefig(f"{args.save}_cartesian.png", dpi=150, bbox_inches='tight')
        fig_z_profile.savefig(f"{args.save}_z_profile.png", dpi=150, bbox_inches='tight')
        logger.info(f"✓ Figures saved to {args.save}_joints.png, {args.save}_cartesian.png, and {args.save}_z_profile.png")
    
    # Show figures
    plt.show()
    
    logger.info("\n✓ Visualization complete")


if __name__ == '__main__':
    main()
