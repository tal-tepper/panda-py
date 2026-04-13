"""
Trajectory Transformer for Panda Robot

This module provides utilities to transform recorded robot trajectories (joint positions)
by applying translation and rotation offsets in Cartesian space, then converting back
to valid joint space trajectories.

The main workflow:
1. Load recorded trajectory (joint positions at 1000Hz)
2. Convert to Cartesian space (end-effector positions and orientations)
3. Apply translation and rotation transformations
4. Convert back to joint space using inverse kinematics
5. Downsample intelligently to create valid trajectory waypoints
6. Validate the trajectory can be executed

Author: GitHub Copilot
"""

import numpy as np
import panda_py
from panda_py.constants import JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER
from typing import Tuple, Optional, List
import time
import scipy
from scipy.spatial.transform import Rotation as R
import logging


class TrajectoryTransformer:
    """
    Transform recorded robot trajectories by applying Cartesian transformations.
    """
    
    def __init__(self, logger: logging.Logger, panda):
        """
        Initialize the transformer.
        
        Args:
            logger: Logger instance for logging messages
        """
        self.logger = logger
        self.joint_limits_lower = np.array(JOINT_LIMITS_LOWER)
        self.joint_limits_upper = np.array(JOINT_LIMITS_UPPER)
        self.panda = panda
        
    def load_trajectory(self, npy_path: str, primitive_name: str) -> dict:
        """
        Load a trajectory from a .npy file.
        
        Args:
            npy_path: Path to the .npy file containing trajectory data
            primitive_name: Name of the primitive to load
            
        Returns:
            Dictionary containing trajectory data (q, dq, position, orientation, etc.)
        """
        loaded_data = np.load(npy_path, allow_pickle=True).item()
        
        if primitive_name not in loaded_data:
            available = list(loaded_data.keys())
            raise KeyError(f"Primitive '{primitive_name}' not found. Available: {available}")
        
        trajectory_data = loaded_data[primitive_name]
        
        self.logger.info(f"✓ Loaded primitive '{primitive_name}' with {len(trajectory_data['q'])} waypoints")
            
        return trajectory_data
    
    def joints_to_cartesian(self, q_trajectory: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert joint space trajectory to Cartesian space (positions and orientations).
        
        Args:
            q_trajectory: Array of joint positions [n_waypoints, 7]
            
        Returns:
            positions: Array of 3D positions [n_waypoints, 3]
            orientations: Array of 3x3 rotation matrices [n_waypoints, 3, 3]
        """
        n_waypoints = len(q_trajectory)
        positions = np.zeros((n_waypoints, 3))
        orientations = np.zeros((n_waypoints, 3, 3))
        
        self.logger.info(f"Converting {n_waypoints} waypoints to Cartesian space...")
        start_time = time.time()
        
        for i, q in enumerate(q_trajectory):
            pose = panda_py.fk(q)  # 4x4 homogeneous transformation matrix
            positions[i] = pose[:3, 3]  # Extract translation
            orientations[i] = pose[:3, :3]  # Extract rotation matrix
            
            # if i % 100 == 0 and i > 0:
            #     elapsed = time.time() - start_time
            #     remaining = (elapsed / i) * (n_waypoints - i)
            #     self.logger.debug(f"  Progress: {i}/{n_waypoints} ({i/n_waypoints*100:.1f}%) - "
            #           f"Est. remaining: {remaining:.1f}s")
        
        elapsed = time.time() - start_time
        self.logger.info(f"✓ Conversion complete in {elapsed:.2f}s")
            
        return positions, orientations
    
    def apply_transformation(
        self,
        positions: np.ndarray,
        orientations: np.ndarray,
        translation: np.ndarray = np.zeros(3),
        rotation_matrix: Optional[np.ndarray] = None,
        rotation_axis: Optional[str] = None,
        rotation_angle: float = 0.0,
        rotate_orientation: bool = False,
        tilt_angle: float = 0.0,
        tilt_direction: Optional[np.ndarray] = None,
        tilt_trajectory: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply translation and rotation to Cartesian trajectory.
        
        Args:
            positions: Original positions [n_waypoints, 3]
            orientations: Original 3x3 rotation matrices [n_waypoints, 3, 3]
            translation: Translation vector [x, y, z] in meters
            rotation_matrix: 3x3 rotation matrix to apply (optional)
            rotation_axis: Axis to rotate about ('x', 'y', or 'z') in base frame (optional)
            rotation_angle: Angle in radians (used with rotation_axis)
            rotate_orientation: If True, also rotate end-effector orientations.
                               If False (default), only rotate trajectory positions around start point.
            tilt_angle: Additional tilt angle in radians towards tilt_direction (applied after translation and rotation)
            tilt_direction: Direction vector [dx, dy] in XY plane to tilt towards (optional)
            tilt_trajectory: If True, also tilt trajectory positions; if False (default), only tilt end-effector orientation
            
        Returns:
            transformed_positions: Transformed positions
            transformed_orientations: Transformed orientations
        """
        self.logger.info(f"Applying transformation:")
        self.logger.info(f"  Translation: {translation}")
        if rotation_matrix is not None:
            self.logger.info(f"  Using provided rotation matrix")
        elif rotation_axis:
            self.logger.info(f"  Rotation: {np.degrees(rotation_angle):.1f}° about {rotation_axis}-axis (base frame)")
            self.logger.info(f"  Rotation center: end-effector start position")
        self.logger.info(f"  Rotate orientation: {rotate_orientation}")
        if tilt_angle != 0.0 and tilt_direction is not None:
            self.logger.info(f"  Tilt: {np.degrees(tilt_angle):.1f}° towards direction [{tilt_direction[0]:.4f}, {tilt_direction[1]:.4f}]")
        
        # Create rotation matrix if not provided
        if rotation_matrix is None:
            if rotation_axis and rotation_angle != 0.0:
                if rotation_axis == 'x':
                    rotation_matrix = R.from_rotvec([rotation_angle, 0, 0]).as_matrix()
                elif rotation_axis == 'y':
                    rotation_matrix = R.from_rotvec([0, rotation_angle, 0]).as_matrix()
                elif rotation_axis == 'z':
                    rotation_matrix = R.from_rotvec([0, 0, rotation_angle]).as_matrix()
                else:
                    raise ValueError("rotation_axis must be 'x', 'y', or 'z'")
            else:
                rotation_matrix = np.eye(3)
        
        # Rotate positions around the first waypoint (end-effector start)
        if rotation_angle != 0.0 or not np.allclose(rotation_matrix, np.eye(3)):
            rotation_center = positions[0]  # First waypoint as rotation center
            
            # For Z-axis rotation, only rotate X-Y plane, keep Z unchanged
            if rotation_axis == 'z' and rotation_angle != 0.0:
                # Rotate only in X-Y plane
                xy_positions = positions[:, :2].copy()  # Extract X,Y
                xy_center = rotation_center[:2]
                centered_xy = xy_positions - xy_center
                # Apply 2D rotation
                cos_a = np.cos(rotation_angle)
                sin_a = np.sin(rotation_angle)
                rot_2d = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                rotated_xy = centered_xy @ rot_2d.T
                
                # Reconstruct 3D positions with original Z values
                transformed_positions = positions.copy()
                transformed_positions[:, :2] = rotated_xy + xy_center
            else:
                # For X or Y axis rotation, or no rotation, rotate in 3D
                centered_positions = positions - rotation_center
                rotated_positions = centered_positions @ rotation_matrix.T
                transformed_positions = rotated_positions + rotation_center
            
            # Apply translation after rotation
            transformed_positions = transformed_positions + translation
        else:
            transformed_positions = positions + translation
        
        # Apply tilt transformation if specified (towards tilt_direction)
        tilt_rot = None
        if tilt_angle != 0.0 and tilt_direction is not None:
            # Create axis perpendicular to tilt_direction in XY plane
            # If tilting towards (dx, dy), rotate around axis perpendicular to it
            dx, dy = tilt_direction[0], tilt_direction[1]
            direction_norm = np.sqrt(dx**2 + dy**2)
            
            if direction_norm > 1e-9:
                # Normalize direction
                dx_norm = dx / direction_norm
                dy_norm = dy / direction_norm
                
                # Rotation axis is perpendicular to direction in XY plane
                # Perpendicular to (dx, dy, 0) is (dy, -dx, 0) for correct tilt direction
                tilt_axis = np.array([dy_norm, -dx_norm, 0.0])
                
                # Create rotation matrix for tilt
                tilt_rot = R.from_rotvec(tilt_axis * tilt_angle).as_matrix()
                
                # Optionally apply tilt rotation to trajectory positions
                if tilt_trajectory:
                    tilt_center = transformed_positions[0]
                    centered_positions = transformed_positions - tilt_center
                    tilted_positions = centered_positions @ tilt_rot.T
                    transformed_positions = tilted_positions + tilt_center
        
        # Apply orientation transformations
        # Start with original orientations
        transformed_orientations = orientations.copy()
        
        # Apply main rotation to orientations if rotate_orientation is True
        # This is applied BEFORE tilt
        if rotate_orientation and not np.allclose(rotation_matrix, np.eye(3)):
            for i in range(len(transformed_orientations)):
                transformed_orientations[i] = rotation_matrix @ transformed_orientations[i]
        
        # Apply tilt rotation to orientations if tilt was specified (AFTER main rotation)
        if tilt_rot is not None:
            # Apply the same tilt rotation to all end-effector orientations
            for i in range(len(transformed_orientations)):
                transformed_orientations[i] = tilt_rot @ transformed_orientations[i]
        
        self.logger.info(f"✓ Transformation applied")
            
        return transformed_positions, transformed_orientations
    
    def cartesian_to_joints(
        self,
        positions: np.ndarray,
        orientations: np.ndarray,
        q_init: Optional[np.ndarray] = None,
        debug_info: Optional[dict] = None,
        ee_rotation_angle: float = 0.0,
        collect_failure_details: bool = False,
        use_hqp: bool = True,
        z_min: Optional[float] = None
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Convert Cartesian trajectory to joint space using inverse kinematics.
        
        Uses Hierarchical Quadratic Programming (HQP) based IK by default, which ensures
        continuous joint trajectories by always seeding from the previous solution.
        
        The HQP solver:
            min_{dq} 0.5 * ||dq||^2 + regularization
        Subject to:
            Primary Task: J(q) * dq = dx  (Cartesian trajectory tracking)
            Height Constraint: z(q) + dz/dq * dq >= z_min  (floor avoidance)
            Joint Limits: q_min <= q + dq <= q_max
        
        Args:
            positions: Array of 3D positions [n_waypoints, 3]
            orientations: Array of 3x3 rotation matrices [n_waypoints, 3, 3]
            q_init: Initial joint configuration - REQUIRED for continuous trajectories.
                    This is the starting configuration from which the trajectory begins.
            ee_rotation_angle: Additional rotation to add to joint 7 (EE flange rotation)
            collect_failure_details: If True, collect detailed info about each IK failure
            use_hqp: If True (default), use HQP-based IK with height constraint.
                     If False, use standard analytical IK.
            z_min: Minimum allowed end-effector height. If None, computed as
                   min(positions[:,2]) - 0.001 (trajectory min minus small margin).
            
        Returns:
            q_trajectory: Joint positions [n_valid_waypoints, 7]
            valid_indices: Indices of waypoints that had valid IK solutions
            
        If collect_failure_details is True, also sets self.ik_failure_details with list of dicts:
            - 'index': waypoint index
            - 'reason': 'nan', 'joint_limits', 'hqp_failed', or 'exception'
            - 'position': Cartesian position
            - 'details': Additional details (e.g., joint values, exception message)
        """
        n_waypoints = len(positions)
        q_trajectory = []
        valid_indices = []
        failed_count = 0
        hqp_used_count = 0
        
        # Collect failure details if requested
        if collect_failure_details:
            self.ik_failure_details = []
        
        # Compute minimum z constraint from trajectory if not provided
        if z_min is None:
            z_min = np.min(positions[:, 2]) - 0.001
        
        self.logger.info(f"Computing inverse kinematics for {n_waypoints} waypoints...")
        if use_hqp:
            self.logger.info(f"  Using HQP-based IK with height constraint z_min={z_min:.4f}m")
        
        start_time = time.time()
        
        # Get initial configuration - critical for continuous trajectories
        if q_init is None:
            # Try to get from first waypoint using analytical IK as seed
            rot = R.from_matrix(orientations[0])
            quat = rot.as_quat()
            position_col = positions[0].reshape(3, 1)
            orientation_quat = quat.reshape(4, 1)
            q_seed = panda_py.ik(position_col, orientation_quat)
            if np.any(np.isnan(q_seed)):
                self.logger.warning("Could not find initial IK solution. Using default configuration.")
                q_seed = np.array([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4])
            q_init = q_seed.flatten()
        
        # Current joint configuration - updated after each successful IK
        q_current = q_init.copy()
        
        for i in range(n_waypoints):
            # Convert rotation matrix to quaternion (x, y, z, w format)
            rot = R.from_matrix(orientations[i])
            quat = rot.as_quat()  # Returns [x, y, z, w]
            
            try:
                q_flat = None
                ik_success = False
                
                if use_hqp:
                    # Use HQP-based IK with height constraint
                    # Always seed from the previous solution for continuity
                    hqp_result = panda_py.ik_hqp(
                        positions[i],
                        quat,
                        q_current,
                        z_min,
                        dt=0.001,
                        max_iterations=100,
                        position_tolerance=1e-4,
                        orientation_tolerance=1e-3,
                        damping=0.05,
                        step_size=0.5
                    )
                    
                    if hqp_result.success:
                        q_flat = np.array(hqp_result.q).flatten()
                        ik_success = True
                        hqp_used_count += 1
                    else:
                        # For trajectory tracking, accept solutions that are close enough
                        # This prevents gaps in the trajectory
                        if hqp_result.position_error < 0.01 and hqp_result.orientation_error < 0.1:
                            q_flat = np.array(hqp_result.q).flatten()
                            ik_success = True
                            hqp_used_count += 1
                            if i < 3:
                                self.logger.debug(f"  Waypoint {i}: Accepting relaxed solution "
                                                f"(pos_err={hqp_result.position_error:.4f}, "
                                                f"ori_err={hqp_result.orientation_error:.4f})")
                        elif collect_failure_details:
                            self.ik_failure_details.append({
                                'index': i,
                                'reason': 'hqp_failed',
                                'position': positions[i].copy(),
                                'details': f"pos_err={hqp_result.position_error:.4f}, "
                                          f"ori_err={hqp_result.orientation_error:.4f}, "
                                          f"iters={hqp_result.iterations}"
                            })
                else:
                    # Standard analytical IK
                    position_col = positions[i].reshape(3, 1)
                    orientation_quat = quat.reshape(4, 1)
                    q = panda_py.ik(position_col, orientation_quat, q_current.reshape(7, 1), q_init[6])
                    q_flat = q.flatten()
                    if not np.any(np.isnan(q_flat)):
                        ik_success = True
                
                # Check for NaN (IK failure)
                if not ik_success or q_flat is None or np.any(np.isnan(q_flat)):
                    failed_count += 1
                    if failed_count < 5:
                        self.logger.debug(f"  Debug: Waypoint {i} IK failed")
                        self.logger.debug(f"    Target position: {positions[i]}, z_min={z_min:.4f}")
                    if collect_failure_details and use_hqp:
                        pass  # Already added above
                    elif collect_failure_details:
                        self.ik_failure_details.append({
                            'index': i,
                            'reason': 'nan',
                            'position': positions[i].copy(),
                            'details': f"IK returned NaN - no solution found"
                        })
                    continue
                
                # Add EE rotation to joint 7 (flange rotation)
                if ee_rotation_angle != 0.0:
                    q_flat[6] -= ee_rotation_angle  # Joint 7 is index 6
                
                # Check joint limits
                if np.all(q_flat >= self.joint_limits_lower) and np.all(q_flat <= self.joint_limits_upper):
                    q_trajectory.append(q_flat)
                    valid_indices.append(i)
                    # Update current configuration for next iteration (key for continuity!)
                    q_current = q_flat.copy()
                else:
                    failed_count += 1
                    violated_joints = []
                    for j in range(7):
                        if q_flat[j] < self.joint_limits_lower[j] or q_flat[j] > self.joint_limits_upper[j]:
                            violated_joints.append(f"J{j}: {np.degrees(q_flat[j]):.2f}° "
                                                  f"(limits: [{np.degrees(self.joint_limits_lower[j]):.2f}°, "
                                                  f"{np.degrees(self.joint_limits_upper[j]):.2f}°])")
                    if failed_count < 5:
                        self.logger.debug(f"  Debug: Waypoint {i} failed joint limits check")
                        for vj in violated_joints:
                            self.logger.debug(f"    {vj}")
                    if collect_failure_details:
                        self.ik_failure_details.append({
                            'index': i,
                            'reason': 'joint_limits',
                            'position': positions[i].copy(),
                            'details': '; '.join(violated_joints),
                            'q_solution': q_flat.copy()
                        })
            except Exception as e:
                failed_count += 1
                if failed_count < 5:
                    self.logger.debug(f"  Debug: Waypoint {i} IK exception: {type(e).__name__}: {e}")
                if collect_failure_details:
                    self.ik_failure_details.append({
                        'index': i,
                        'reason': 'exception',
                        'position': positions[i].copy(),
                        'details': f"{type(e).__name__}: {e}"
                    })
        
        elapsed = time.time() - start_time
        success_rate = len(valid_indices) / n_waypoints * 100 if n_waypoints > 0 else 0
        self.logger.info(f"✓ IK computation complete in {elapsed:.2f}s")
        self.logger.info(f"  Valid waypoints: {len(valid_indices)}/{n_waypoints} ({success_rate:.1f}%)")
        if use_hqp:
            self.logger.info(f"  HQP IK used: {hqp_used_count}/{n_waypoints}")
        if failed_count > 0:
            self.logger.warning(f"  ⚠ Failed waypoints: {failed_count} (outside joint limits or no solution)")

        # Interpolate to fill in gaps left by failed IK waypoints
        if len(q_trajectory) > 1 and len(valid_indices) < n_waypoints:
            q_full = np.empty((n_waypoints, 7))
            q_arr = np.array(q_trajectory)
            vi = np.array(valid_indices)
            for j in range(7):
                q_full[:, j] = np.interp(
                    np.arange(n_waypoints), vi, q_arr[:, j])
            interp_count = n_waypoints - len(valid_indices)
            self.logger.info(f"  Interpolated {interp_count} failed waypoints in joint space")
            return q_full, list(range(n_waypoints))

        return np.array(q_trajectory), valid_indices

    def cartesian_to_joints_pink(
        self,
        positions: np.ndarray,
        orientations: np.ndarray,
        q_init: Optional[np.ndarray] = None,
        collect_failure_details: bool = False,
        pink_kwargs: Optional[dict] = None,
        skip_interpolation: bool = False,
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Alternate IK pipeline that uses the external `pink` solver if available.

        This function attempts to import `pink` and detect a sensible IK entry
        point such as `inverse_kinematics`, `ik_solve` or `solve`. It will
        iterate over waypoints, seeding the solver with the previous solution
        to encourage continuity. If `pink` is not available or does not expose
        a known API, an ImportError is raised.

        The return values mirror `cartesian_to_joints`: (q_trajectory, valid_indices).
        If some waypoints fail, the function will interpolate joint-space values
        across the missing indices to provide a full-length trajectory (same
        behaviour as `cartesian_to_joints`).
        """
        try:
            import pink
        except Exception as e:  # pragma: no cover - pink may not be installed in test env
            raise ImportError("pink module is required for cartesian_to_joints_pink") from e

        solver = None
        # Prefer high-level convenience API if present
        if hasattr(pink, 'solve_ik'):
            solver = getattr(pink, 'solve_ik')
            solver_name = 'solve_ik'
        else:
            # Try common solver function names used by different pink versions
            solver_name = None
            for name in ('inverse_kinematics', 'ik_solve', 'solve', 'ik'):
                if hasattr(pink, name):
                    solver = getattr(pink, name)
                    solver_name = name
                    break

        if solver is None:
            raise ImportError("Could not find a compatible IK solver in 'pink' (tried solve_ik, inverse_kinematics, ik_solve, solve, ik)")

        # Robot loader (only load if needed)
        robot = None
        load_robot_description = None
        try:
            from robot_descriptions.loaders.pinocchio import load_robot_description
            load_robot_description = load_robot_description
        except Exception:
            load_robot_description = None

        n_waypoints = len(positions)
        q_trajectory = []
        valid_indices = []
        failed_count = 0

        if collect_failure_details:
            self.ik_failure_details = []

        pink_kwargs = pink_kwargs or {}

        # Seed initial configuration
        if q_init is None:
            # Try to get initial from panda if present
            q_init = getattr(self.panda, 'q', None)
        if q_init is None:
            q_init = np.array([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4])

        q_current = q_init.copy()

        # If we can load a pinocchio robot description and pink exposes Configuration
        # and FrameTask/PostureTask, prefer constructing tasks and calling pink.solve_ik
        use_task_api = False
        try:
            if load_robot_description is not None and hasattr(pink, 'Configuration') and hasattr(pink, 'FrameTask'):
                # Attempt to load robot and verify Configuration works
                try:
                    robot = load_robot_description('panda_description')
                    # Choose a sensible end-effector frame
                    ee_frame = 'panda_hand_tcp' if 'panda_hand_tcp' in [f.name for f in robot.model.frames] else 'panda_link8'
                    use_task_api = True
                except Exception:
                    robot = None
                    use_task_api = False
        except Exception:
            robot = None
            use_task_api = False

        for i in range(n_waypoints):
            rot = R.from_matrix(orientations[i])
            quat = rot.as_quat()  # [x, y, z, w]

            try:
                q_flat = None

                if use_task_api and robot is not None:
                    # Iterative differential-IK via pink
                    # solve_ik returns a velocity dq; we integrate q += dq*dt
                    # and repeat until the Cartesian error is small enough.
                    try:
                        import pinocchio as pin

                        # Build full-dimension config (model.nq may include fingers)
                        q_cfg = np.array(robot.q0).copy() if hasattr(robot, 'q0') else np.zeros(robot.model.nq)
                        q_cfg[:len(q_current)] = q_current.copy()

                        # Target SE3 for this waypoint
                        se3_target = pin.SE3(orientations[i], positions[i])

                        # Tasks (re-created per waypoint for clarity)
                        frame_task = pink.FrameTask(ee_frame, position_cost=1.0, orientation_cost=1.0)
                        frame_task.set_target(se3_target)

                        posture_task = pink.PostureTask(cost=0.01)

                        # Convergence loop
                        dt = 0.01
                        max_iters = 50
                        pos_tol = 1e-3   # 1 mm
                        ori_tol = 1e-2   # ~0.6°
                        solver_name_qp = 'quadprog'

                        for _iter in range(max_iters):
                            configuration = pink.Configuration(
                                robot.model, robot.data, q_cfg)

                            # Posture target = current config (regularise around seed)
                            posture_task.set_target_from_configuration(configuration)

                            velocity = pink.solve_ik(
                                configuration, [frame_task, posture_task],
                                dt, solver=solver_name_qp, damping=1e-4)

                            # Integrate
                            q_cfg = pin.integrate(
                                robot.model, q_cfg, velocity * dt)

                            # Check convergence
                            pin.forwardKinematics(robot.model, robot.data, q_cfg)
                            pin.updateFramePlacements(robot.model, robot.data)
                            fid = robot.model.getFrameId(ee_frame)
                            current_se3 = robot.data.oMf[fid]
                            pos_err = np.linalg.norm(
                                current_se3.translation - se3_target.translation)
                            ori_err = np.linalg.norm(
                                pin.log3(current_se3.rotation.T @ se3_target.rotation))

                            if pos_err < pos_tol and ori_err < ori_tol:
                                break

                        # Extract arm joints (first 7)
                        q_flat = np.array(q_cfg[:7]).flatten()
                    except Exception as e:
                        q_flat = None
                        if collect_failure_details:
                            self.ik_failure_details.append({
                                'index': i,
                                'reason': 'pink_task_error',
                                'position': positions[i].copy(),
                                'details': f"{type(e).__name__}: {e}"
                            })

                # Fallback: try convenience solver signatures (older pink variants)
                if q_flat is None:
                    tried = []
                    res = None

                    # 1) solver(position, quaternion, seed=..., **kwargs)
                    try:
                        tried.append('pos,quat,seed')
                        res = solver(positions[i], quat, seed=q_current, **pink_kwargs)
                    except TypeError:
                        res = None

                    # 2) solver(position, quaternion, q_seed, **kwargs)
                    if res is None:
                        try:
                            tried.append('pos,quat,q_seed')
                            res = solver(positions[i], quat, q_current, **pink_kwargs)
                        except TypeError:
                            res = None

                    # 3) solver that includes robot/model
                    if res is None and robot is not None:
                        try:
                            tried.append('robot,pos,quat,seed')
                            res = solver(robot, positions[i], quat, q_current, **pink_kwargs)
                        except Exception:
                            res = None

                    # 4) solver(pos, quat)
                    if res is None:
                        try:
                            tried.append('pos,quat')
                            res = solver(positions[i], quat)
                        except TypeError:
                            res = None

                    if res is None:
                        # If nothing worked, record failure
                        if collect_failure_details:
                            self.ik_failure_details.append({
                                'index': i,
                                'reason': 'pink_signature_mismatch',
                                'position': positions[i].copy(),
                                'details': f"Tried signatures: {tried}"
                            })
                        failed_count += 1
                        continue

                    # Handle possible return types
                    if isinstance(res, dict):
                        if not res.get('success', True):
                            q_flat = None
                        else:
                            q_flat = np.array(res.get('q') or res.get('solution'))
                    elif isinstance(res, (list, tuple, np.ndarray)):
                        q_flat = np.array(res).flatten()
                    else:
                        q_flat = None

                if q_flat is None or np.any(np.isnan(q_flat)):
                    failed_count += 1
                    if collect_failure_details and q_flat is not None:
                        self.ik_failure_details.append({
                            'index': i,
                            'reason': 'pink_failed',
                            'position': positions[i].copy(),
                            'details': f'return={type(q_flat).__name__}'
                        })
                    continue

                # Check joint limits
                if np.all(q_flat >= self.joint_limits_lower) and np.all(q_flat <= self.joint_limits_upper):
                    q_trajectory.append(q_flat)
                    valid_indices.append(i)
                    q_current = q_flat.copy()
                else:
                    failed_count += 1
                    if collect_failure_details:
                        self.ik_failure_details.append({
                            'index': i,
                            'reason': 'joint_limits',
                            'position': positions[i].copy(),
                            'q_solution': q_flat.copy()
                        })
            except Exception as e:
                failed_count += 1
                if collect_failure_details:
                    self.ik_failure_details.append({
                        'index': i,
                        'reason': 'exception',
                        'position': positions[i].copy(),
                        'details': f"{type(e).__name__}: {e}"
                    })

        # If we have some valid points but not all, interpolate missing waypoints in joint space
        if not skip_interpolation and len(q_trajectory) > 1 and len(valid_indices) < n_waypoints:
            q_full = np.empty((n_waypoints, 7))
            q_arr = np.array(q_trajectory)
            vi = np.array(valid_indices)
            for j in range(7):
                q_full[:, j] = np.interp(np.arange(n_waypoints), vi, q_arr[:, j])
            interp_count = n_waypoints - len(valid_indices)
            self.logger.info(f"  Interpolated {interp_count} failed waypoints (pink) in joint space")
            return q_full, list(range(n_waypoints))

        return np.array(q_trajectory), valid_indices
    
    def downsample_trajectory(
        self,
        q_trajectory: np.ndarray,
        positions: np.ndarray,
        max_waypoints: int = 100,
        min_distance: float = 0.01
    ) -> np.ndarray:
        """
        Downsample trajectory intelligently to reduce waypoints while preserving shape.
        
        Two-stage approach:
        1. First pass: Keep points that are at least min_distance apart (greedy)
        2. Second pass: If fewer than max_waypoints, add farthest points to fill up
        
        Args:
            q_trajectory: Full joint trajectory [n_waypoints, 7]
            positions: Cartesian positions [n_waypoints, 3] corresponding to q_trajectory
            max_waypoints: Maximum number of waypoints to keep
            min_distance: Minimum joint-space distance between waypoints (radians)
            
        Returns:
            downsampled_trajectory: Reduced trajectory [n_reduced, 7]
        """
        if len(q_trajectory) <= max_waypoints:
            self.logger.info(f"Trajectory already has {len(q_trajectory)} waypoints (≤ {max_waypoints})")
            return q_trajectory
        
        self.logger.info(f"Downsampling trajectory from {len(q_trajectory)} to {max_waypoints} waypoints...")
        
        # First pass: Greedy selection with min_distance criterion
        selected_indices = [0]  # Always start with first point
        
        for i in range(1, len(q_trajectory)):
            # Check distance to last selected point
            dist = np.linalg.norm(q_trajectory[i] - q_trajectory[selected_indices[-1]])
            if dist >= min_distance:
                selected_indices.append(i)
        
        # Ensure last point is included
        if selected_indices[-1] != len(q_trajectory) - 1:
            selected_indices.append(len(q_trajectory) - 1)
        
        self.logger.debug(f"  After min_distance filtering: {len(selected_indices)} waypoints")
        
        # Second pass: If we have fewer than max_waypoints, add farthest points
        if len(selected_indices) < max_waypoints:
            remaining_indices = set(range(len(q_trajectory))) - set(selected_indices)
            
            # Calculate distances from each remaining point to nearest selected point
            def get_min_distance_to_selected(idx):
                min_dist = float('inf')
                for sel_idx in selected_indices:
                    dist = np.linalg.norm(positions[idx] - positions[sel_idx])
                    min_dist = min(min_dist, dist)
                return min_dist
            
            # Add points iteratively, always choosing the farthest one
            while len(selected_indices) < max_waypoints and remaining_indices:
                # Find the remaining point that is farthest from all selected points
                farthest_idx = max(remaining_indices, key=get_min_distance_to_selected)
                selected_indices.append(farthest_idx)
                remaining_indices.remove(farthest_idx)
            
            self.logger.debug(f"  After adding farthest points: {len(selected_indices)} waypoints")
        
        # If we have more than max_waypoints, uniformly subsample
        if len(selected_indices) > max_waypoints:
            # Sort first to maintain order
            selected_indices.sort()
            # Uniformly subsample
            step = len(selected_indices) / max_waypoints
            final_indices = [selected_indices[int(i * step)] for i in range(max_waypoints)]
            # Ensure first and last are included
            if final_indices[0] != 0:
                final_indices[0] = 0
            if final_indices[-1] != len(q_trajectory) - 1:
                final_indices[-1] = len(q_trajectory) - 1
            selected_indices = final_indices
            self.logger.debug(f"  After uniform subsampling: {len(selected_indices)} waypoints")
        
        # Sort indices to maintain trajectory order
        selected_indices.sort()
        
        reduced = q_trajectory[selected_indices]
        
        self.logger.info(f"✓ Downsampling complete: {len(q_trajectory)} → {len(reduced)} waypoints")
        
        return reduced
    
    def estimate_computation_time(
        self,
        n_waypoints: int,
        max_downsampled: int = 100
    ) -> dict:
        """
        Estimate computation time for trajectory transformation.
        
        Args:
            n_waypoints: Number of waypoints in original trajectory
            max_downsampled: Maximum waypoints after downsampling
            
        Returns:
            Dictionary with time estimates for each step
        """
        # Empirical estimates (will vary by hardware)
        fk_time_per_waypoint = 0.0002  # 0.2ms per FK computation
        ik_time_per_waypoint = 0.002   # 2ms per IK computation
        trajectory_gen_base = 0.5      # 0.5s base time for trajectory generation
        trajectory_gen_per_wp = 0.01   # 10ms per waypoint
        
        estimates = {
            'forward_kinematics': n_waypoints * fk_time_per_waypoint,
            'inverse_kinematics': n_waypoints * ik_time_per_waypoint,
            'downsampling': 0.1,
            'trajectory_generation': trajectory_gen_base + max_downsampled * trajectory_gen_per_wp,
        }
        
        estimates['total'] = sum(estimates.values())
        
        return estimates
    
    def transform_trajectory(
        self,
        npy_path: str,
        primitive_name: str,
        translation: np.ndarray = np.zeros(3),
        rotation_matrix: Optional[np.ndarray] = None,
        rotation_axis: Optional[str] = None,
        rotation_angle: float = 0.0,
        rotate_orientation: bool = False,
        max_waypoints: int = 100,
        min_distance: float = 0.01,
        speed_factor: float = 0.1,
        phantom_boundaries: Optional[dict] = None,
        skip_downsampling: bool = False,
        q_init = None,
        tilt_angle: float = 0.0,
        tilt_direction: Optional[np.ndarray] = None,
        tilt_trajectory: bool = False,
        collect_ik_failure_details: bool = False
    ) -> dict:
        """
        Complete pipeline to transform a recorded trajectory.
        
        Args:
            npy_path: Path to trajectory file
            primitive_name: Name of primitive to load
            translation: Translation offset [x, y, z] in meters
            rotation_matrix: 3x3 rotation matrix (optional)
            rotation_axis: Rotation axis 'x', 'y', or 'z' in base frame (optional)
            rotation_angle: Rotation angle in radians (optional)
            rotate_orientation: If True, also rotate end-effector orientations.
                               If False (default), only rotate trajectory positions.
            max_waypoints: Maximum waypoints after downsampling
            min_distance: Minimum joint distance between waypoints
            speed_factor: Speed factor for trajectory generation (0.0-1.0)
            phantom_boundaries: Dict with {'x': [min, max], 'y': [min, max], 'z': [min, max]}
                               for boundary checking. If provided, trajectory will be validated
                               and trimmed if needed (requires at least 50% valid waypoints).
            skip_downsampling: If True, skip downsampling step (use pre-downsampled primitives)
            tilt_angle: Additional tilt angle in radians towards tilt_direction (0-10 degrees recommended)
            tilt_direction: Direction vector [dx, dy] in XY plane to tilt towards (e.g., [-grid_x, -grid_y] to tilt towards center)
            tilt_trajectory: If True, also tilt trajectory positions; if False (default), only tilt end-effector orientation
            collect_ik_failure_details: If True, include detailed IK failure info in result
            
        Returns:
            Dictionary with transformed trajectory data (includes 'ik_failure_details' if collect_ik_failure_details=True)
        """
        # Load trajectory
        trajectory_data = self.load_trajectory(npy_path, primitive_name)
        q_original = np.array(trajectory_data['q'])
        
        # Always compute positions and orientations from joints using FK
        # This accounts for what the robot is holding (load parameters)
        positions, orientations = self.joints_to_cartesian(q_original)
        
        # Apply transformation
        positions_tf, orientations_tf = self.apply_transformation(
            positions, orientations,
            translation=translation,
            rotation_matrix=rotation_matrix,
            rotation_axis=rotation_axis,
            rotation_angle=rotation_angle,
            rotate_orientation=rotate_orientation,
            tilt_angle=tilt_angle,
            tilt_direction=tilt_direction,
            tilt_trajectory=tilt_trajectory
        )
        
        # Check boundaries if provided (relative to trajectory start)
        is_valid = True
        trimmed_index = None
        if phantom_boundaries is not None:
            self.logger.info(f"Checking trajectory boundaries (relative to start position)...")
            
            # Get trajectory start position for relative checking
            trajectory_start = positions_tf[0]
            
            for i, pos in enumerate(positions_tf):
                # Check position relative to trajectory start
                relative_pos = pos
                if not (phantom_boundaries['x'][0] <= relative_pos[0] <= phantom_boundaries['x'][1] and
                        phantom_boundaries['y'][0] <= relative_pos[1] <= phantom_boundaries['y'][1] and
                        phantom_boundaries['z'][0] <= relative_pos[2] <= phantom_boundaries['z'][1]):
                    is_valid = False
                    trimmed_index = i
                    self.logger.warning(f"  ⚠ Trajectory out of boundaries at step {i}/{len(positions_tf)}. relative_pos:{relative_pos},phantom_boundaries:{phantom_boundaries}")
                    break
            
            # Handle invalid trajectories
            if not is_valid:
                if trimmed_index is not None and trimmed_index / len(positions_tf) >= 0.5:
                    # Trim trajectory if more than 50% is valid
                    self.logger.info(f"  ✓ Trimming trajectory to {trimmed_index} steps (>{50}% valid)")
                    positions_tf = positions_tf[:trimmed_index]
                    orientations_tf = orientations_tf[:trimmed_index]
                else:
                    # More than 50% is out of bounds, reject the trajectory
                    self.logger.error(f"  ✗ Trajectory invalid: less than 50% waypoints within boundaries. Aborting transformation.:{trimmed_index}/{len(positions_tf)}")
                    return None
        
        # Convert back to joint space
        # If rotate_orientation is True, add rotation to joint 7 after IK
        ee_rotation = rotation_angle if (rotate_orientation and rotation_axis == 'z') else 0.0
        self.logger.info(f"min z is :{np.min(positions_tf[:,2])}, max z is:{np.max(positions_tf[:,2])} z boundaries:{phantom_boundaries['z'] if phantom_boundaries else 'N/A'}")
        if q_init is None:
            q_init = self.panda.q
        q_transformed, valid_indices = self.cartesian_to_joints(
            positions_tf, orientations_tf, q_init=q_init,
            debug_info={'translation': translation},
            ee_rotation_angle=ee_rotation,
            collect_failure_details=collect_ik_failure_details
        )
        
        # Get IK failure details if collected
        ik_failure_details = getattr(self, 'ik_failure_details', []) if collect_ik_failure_details else []
        
        if len(q_transformed) == 0:
            self.logger.error("✗ No valid IK solutions found for transformed trajectory. Aborting transformation.")
            return None
        
        # Downsample (optional, skip if using pre-downsampled primitives)
        if skip_downsampling:
            self.logger.info("Skipping downsampling (using pre-downsampled primitive)")
            q_downsampled = q_transformed
        else:
            q_downsampled = self.downsample_trajectory(
                q_transformed, positions_tf[valid_indices], max_waypoints=max_waypoints, min_distance=min_distance
            )
        
        # Create result dictionary
        result = {
            'q_original': q_original,
            'q_transformed_full': q_transformed,
            'q_waypoints': q_downsampled,
            'valid_indices': valid_indices,
            'positions_original': positions,
            'orientations_original': orientations,
            'positions_transformed': positions_tf[valid_indices],
            'orientations_transformed': orientations_tf[valid_indices],
            'translation': translation,
            'rotation_matrix': rotation_matrix if rotation_matrix is not None else np.eye(3),
            'primitive_name': primitive_name,
            'positions_transformed_all': positions_tf,  # All positions before IK filtering
            'orientations_transformed_all': orientations_tf,  # All orientations before IK filtering
        }
        
        # Add IK failure details if collected
        if collect_ik_failure_details:
            result['ik_failure_details'] = ik_failure_details
            result['ik_success_rate'] = len(valid_indices) / len(positions_tf) * 100
            result['ik_failed_count'] = len(ik_failure_details)
        
        return result
    
    def transform_trajectory_cartesian(
        self,
        npy_path: str,
        primitive_name: str,
        translation: np.ndarray = np.zeros(3),
        rotation_matrix: Optional[np.ndarray] = None,
        rotation_axis: Optional[str] = None,
        rotation_angle: float = 0.0,
        rotate_orientation: bool = True,
        max_waypoints: int = 100,
        min_distance: float = 0.01,
        speed_factor: float = 0.1,
        phantom_boundaries: Optional[dict] = None,
        skip_downsampling: bool = True,
        q_init: Optional[np.ndarray] = None,
        tilt_angle: float = 0.0,
        tilt_direction: Optional[np.ndarray] = None,
        tilt_trajectory: bool = False
    ) -> dict:
        """
        Transform a recorded trajectory using Cartesian data directly from the primitives file.
        
        Unlike transform_trajectory which converts from joint space to Cartesian using FK,
        this function uses the position/orientation data stored directly in the primitives file.
        
        Args:
            npy_path: Path to trajectory file
            primitive_name: Name of primitive to load
            translation: Translation offset [x, y, z] in meters
            rotation_matrix: 3x3 rotation matrix (optional)
            rotation_axis: Rotation axis 'x', 'y', or 'z' in base frame (optional)
            rotation_angle: Rotation angle in radians (optional)
            rotate_orientation: If True, also rotate end-effector orientations.
                               If False (default), only rotate trajectory positions.
            max_waypoints: Maximum waypoints after downsampling
            min_distance: Minimum joint distance between waypoints
            speed_factor: Speed factor for trajectory generation (0.0-1.0)
            phantom_boundaries: Dict with {'x': [min, max], 'y': [min, max], 'z': [min, max]}
                               for boundary checking. If provided, trajectory will be validated
                               and trimmed if needed (requires at least 50% valid waypoints).
            skip_downsampling: If True, skip downsampling step (use pre-downsampled primitives)
            q_init: Initial joint configuration for IK
            tilt_angle: Additional tilt angle in radians towards tilt_direction (0-10 degrees recommended)
            tilt_direction: Direction vector [dx, dy] in XY plane to tilt towards
            tilt_trajectory: If True, also tilt trajectory positions; if False (default), only tilt end-effector orientation
            
        Returns:
            Dictionary with transformed trajectory data, or None if transformation fails
        """
        # Load trajectory
        trajectory_data = self.load_trajectory(npy_path, primitive_name)
        
        # Use Cartesian data directly from the primitives file
        if 'position' not in trajectory_data or 'orientation' not in trajectory_data:
            self.logger.error("✗ Primitive file does not contain 'position' or 'orientation' data. "
                            "Use transform_trajectory() instead to compute from joint data.")
            return None
        
        positions = np.array(trajectory_data['position'])
        orientations = np.array(trajectory_data['orientation'])
        
        self.logger.info(f"✓ Using Cartesian data directly: {len(positions)} waypoints")
        
        # Apply transformation
        positions_tf, orientations_tf = self.apply_transformation(
            positions, orientations,
            translation=translation,
            rotation_matrix=rotation_matrix,
            rotation_axis=rotation_axis,
            rotation_angle=rotation_angle,
            rotate_orientation=rotate_orientation,
            tilt_angle=tilt_angle,
            tilt_direction=tilt_direction,
            tilt_trajectory=tilt_trajectory
        )
        
        # Check boundaries if provided (relative to trajectory start)
        is_valid = True
        trimmed_index = None
        if phantom_boundaries is not None:
            self.logger.info(f"Checking trajectory boundaries (relative to start position)...")
            
            for i, pos in enumerate(positions_tf):
                if not (phantom_boundaries['x'][0] <= pos[0] <= phantom_boundaries['x'][1] and
                        phantom_boundaries['y'][0] <= pos[1] <= phantom_boundaries['y'][1] and
                        phantom_boundaries['z'][0] <= pos[2] <= phantom_boundaries['z'][1]):
                    is_valid = False
                    trimmed_index = i
                    self.logger.warning(f"  ⚠ Trajectory out of boundaries at step {i}/{len(positions_tf)}. "
                                       f"pos:{pos}, phantom_boundaries:{phantom_boundaries}")
                    break
            
            # Handle invalid trajectories
            if not is_valid:
                if trimmed_index is not None and trimmed_index / len(positions_tf) >= 0.5:
                    # Trim trajectory if more than 50% is valid
                    self.logger.info(f"  ✓ Trimming trajectory to {trimmed_index} steps (>50% valid)")
                    positions_tf = positions_tf[:trimmed_index]
                    orientations_tf = orientations_tf[:trimmed_index]
                else:
                    # More than 50% is out of bounds, reject the trajectory
                    self.logger.error(f"  ✗ Trajectory invalid: less than 50% waypoints within boundaries. "
                                     f"Aborting transformation: {trimmed_index}/{len(positions_tf)}")
                    return None
        
        # Convert to joint space using IK
        # Note: ee_rotation is now applied directly to orientations in apply_transformation
        # so we don't need to apply it again here
        self.logger.info(f"min z: {np.min(positions_tf[:,2])}, max z: {np.max(positions_tf[:,2])} "
                        f"z boundaries: {phantom_boundaries['z'] if phantom_boundaries else 'N/A'}")
        
        if q_init is None:
            q_init = self.panda.q
            
      
        q_transformed = q_init
        if len(q_transformed) == 0:
            self.logger.error("✗ No valid IK solutions found for transformed trajectory. Aborting transformation.")
            return None
        
      
        q_downsampled = q_transformed
      
        
        # Create result dictionary
        result = {
            'q_transformed_full': q_transformed,
            'q_waypoints': q_downsampled,
            'valid_indices': 0,
            'positions_original': positions,
            'orientations_original': orientations,
            'positions_transformed': positions_tf,
            'orientations_transformed': orientations_tf,
            'translation': translation,
            'rotation_matrix': rotation_matrix if rotation_matrix is not None else np.eye(3),
            'primitive_name': primitive_name,
        }
        
        return result
    
    def save_transformed_trajectory(self, result: dict, output_path: str):
        """
        Save transformed trajectory to a .npy file.
        
        Args:
            result: Result dictionary from transform_trajectory
            output_path: Path to save the transformed trajectory
        """
        np.save(output_path, result)
        self.logger.info(f"✓ Saved transformed trajectory to {output_path}")
    
    def execute_trajectory_with_logging(
        self,
        panda,
        writer,
        q_waypoints: np.ndarray,
        speed_factor: float = 0.1,
        height_limit: float = 0.15,
    ) -> bool:
        """
        Execute a height-constrained trajectory from joint waypoints and log data to HDF5.

        Uses :py:func:`panda.move_to_joint_position_with_height_limit` which
        performs the entire pipeline in C++ (prepend current position → build
        time-optimal JointTrajectory → enforce height limit →  execute).

        Logging is enabled before execution and processed afterwards.  The
        robot returns to the first waypoint after execution.

        Args:
            panda: Panda robot instance.
            writer: HDF5Writer instance.
            q_waypoints: Joint-space waypoints [n_waypoints, 7].
            speed_factor: Speed factor for trajectory execution (0.0–1.0).
            height_limit: Minimum allowed end-effector height (metres).

        Returns:
            True on success, False on failure.
        """
        self.logger.info("=" * 60)
        self.logger.info("TRAJECTORY EXECUTION WITH LOGGING")
        self.logger.info("=" * 60)

        try:
            # --- diagnostics before execution ---
            q_robot = np.array(panda.q)
            q_start = q_waypoints[0]
            pos_err = np.linalg.norm(q_robot - q_start)
            self.logger.info(f"Waypoints:       {len(q_waypoints)}")
            self.logger.info(f"Height limit:    {height_limit * 100:.1f} cm")
            self.logger.info(f"Speed factor:    {speed_factor}")
            self.logger.info(f"Position error to first waypoint: {pos_err:.6f} rad "
                             f"({np.degrees(pos_err):.3f}°)")

            # Move to the first waypoint (no logging yet)
            self.logger.info("Moving to first waypoint...")
            panda.move_to_joint_position(q_start, speed_factor=speed_factor)

            # Estimate logging buffer from trajectory length
            # Rough upper bound: ~1000 samples/s, assume duration ≈ n_waypoints * 0.01 / speed_factor
            estimated_duration = len(q_waypoints) * 0.01 / max(speed_factor, 0.01)
            num_samples = int(estimated_duration * 1000) + 2000  # generous margin
            self.logger.info(f"Enabling data logging ({num_samples} samples buffer)...")
            panda.enable_logging(num_samples)

            # Enable external sensor writing during execution
            writer.start_writing()

            # Execute — everything (trajectory build + height constraint + control) in C++
            self.logger.info("Executing height-constrained trajectory...")
            wps = [q.tolist() for q in q_waypoints]
            t0 = time.time()
            success = panda.move_to_joint_position_with_height_limit(
                wps, height_limit, speed_factor=speed_factor,
            )
            exec_time = time.time() - t0

            # Stop external sensor writing
            writer.stop_writing()
            self.logger.info(f"Execution time: {exec_time:.2f}s  success={success}")

            # Retrieve logged data
            self.logger.info("Retrieving logged data...")
            log = panda.get_log()
            self.logger.info(f"Logged samples: {len(log['q'])}")

            # Process log → HDF5
            self.logger.info("Processing logged data to calculate forces...")
            self._bulk_add_logged_data(panda, writer, log)

            # Return to start position
            self.logger.info("Returning to start position...")
            panda.move_to_joint_position(q_start, speed_factor=speed_factor)
            self.logger.info("=" * 60)

            return success

        except Exception as e:
            self.logger.error(f"✗ Execution failed: {e}")
            writer.stop_writing()
            writer.clear_data()
            writer.to_file_enabled = False
            return False
    
    def _bulk_add_logged_data(self, panda, writer, log: dict):
        """
        Process panda log data and bulk-add to HDF5 writer.
        
        Args:
            panda: Panda robot instance (for model)
            writer: HDF5Writer instance
            log: Log dictionary from panda.get_log()
        """
        self.logger.info("Calculating forces from logged data...")
        t0 = time.time()
        # Get arrays from log
        q_log = np.array(log['q'])[::10]
        dq_log = np.array(log['dq'])[::10]
        tau_J_log = np.array(log['tau_J'])[::10]
        O_T_EE_log = np.array(log['O_T_EE'])[::10]
        F_T_EE_log = np.array(log['F_T_EE'])[::10]
        EE_T_K_log = np.array(log['EE_T_K'])[::10]
        m_total_log = np.array(log['m_total'])[::10]
        F_x_Ctotal_log = np.array(log['F_x_Ctotal'])[::10]
        I_total_log = np.array(log['I_total'])[::10]
        # m_total_log = log.get('m_total')
        # F_x_Ctotal_log = log.get('F_x_Ctotal')
        # I_total_log = log.get('I_total')
      
        # print(f'loaded log data: q_log.shape={q_log.shape}, dq_log.shape={dq_log.shape}, tau_J_log.shape={tau_J_log.shape}, O_T_EE_log.shape={O_T_EE_log.shape}, F_T_EE_log.shape={F_T_EE_log.shape}, EE_T_K_log.shape={EE_T_K_log.shape}')
        n_samples = len(q_log)
        
        # Get model once (constant during execution)
        p_model = panda.get_model()
        
        # Calculate forces for all samples
        calculated_forces = []
        
        self.logger.info(f"Calculating forces for {n_samples} samples... dt:{time.time()-t0:.2f}s")
        start_time = time.time()
        try:
            for i in range(n_samples):
                # Create a minimal state object for calc_force
                # class State:
                #     def __init__(self, q, dq, tau_J):
                #         self.q = q
                #         self.dq = dq
                #         self.tau_J = tau_J
                
                # state = State(q_log[i], dq_log[i], tau_J_log[i])
                
                # Calculate Jacobian
                jacobian = np.array(
                    p_model.zero_jacobian(panda_py.libfranka.Frame.kEndEffector, q_log[i],F_T_EE_log[i] ,EE_T_K_log[i])
                ).reshape(6, 7, order='F')
                
                # Calculate force (same as in panda_sampler.calc_force)
                tau_j = tau_J_log[i]
                gravity = np.array(p_model.gravity(q_log[i],m_total_log[i], F_x_Ctotal_log[i]))
                coriolis = np.array(p_model.coriolis(q_log[i], dq_log[i],I_total_log[i],m_total_log[i], F_x_Ctotal_log[i]))
                tau = tau_j - gravity - coriolis
                
                calced_force, _, _, _ = scipy.linalg.lstsq(
                    jacobian.T, tau, lapack_driver='gelsy'
                )
                calculated_forces.append(calced_force)
                
                # if (i + 1) % 100 == 0:
                #     elapsed = time.time() - start_time
                    # remaining = (elapsed / (i + 1)) * (n_samples - (i + 1))
                    # self.logger.debug(f"  Progress: {i+1}/{n_samples} ({(i+1)/n_samples*100:.1f}%) - "
                    #     f"Est. remaining: {remaining:.1f}s")
        except Exception as e:
            self.logger.error(f"✗ Force calculation failed: {e}")
            return
        elapsed = time.time() - start_time
        self.logger.info(f"✓ Force calculation complete in {elapsed:.2f}s")
        
        # Generate timestamps (use indices as timestamps if not in log)
        timestamps = log['time'][::10]
        
        
        # Bulk add all data to writer
        t1 = time.time()
        self.logger.info(f"Bulk adding {n_samples} samples to HDF5...")
        self.logger.debug(f"Sample of first timestamp: {timestamps[0]} shape: {np.array(timestamps).shape}")
        for i in range(n_samples):
            # Reshape O_T_EE to 4x4 matrix
            my_pose_4 = O_T_EE_log[i].reshape(4, 4).T
            
            writer.add_robot_data(
                q_log[i],
                dq_log[i],
                calculated_forces[i],
                tau_J_log[i],
                my_pose_4,
                timestamps[i][0]/1e3
            )
        
        self.logger.info(f"✓ Bulk add complete dt:{time.time()-t1:.2f}s")
