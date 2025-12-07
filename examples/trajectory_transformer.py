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
from scipy.spatial.transform import Rotation as R


class TrajectoryTransformer:
    """
    Transform recorded robot trajectories by applying Cartesian transformations.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the transformer.
        
        Args:
            verbose: If True, print progress information
        """
        self.verbose = verbose
        self.joint_limits_lower = np.array(JOINT_LIMITS_LOWER)
        self.joint_limits_upper = np.array(JOINT_LIMITS_UPPER)
        
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
        
        if self.verbose:
            print(f"✓ Loaded primitive '{primitive_name}' with {len(trajectory_data['q'])} waypoints")
            
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
        
        if self.verbose:
            print(f"Converting {n_waypoints} waypoints to Cartesian space...")
        
        start_time = time.time()
        
        for i, q in enumerate(q_trajectory):
            pose = panda_py.fk(q)  # 4x4 homogeneous transformation matrix
            positions[i] = pose[:3, 3]  # Extract translation
            orientations[i] = pose[:3, :3]  # Extract rotation matrix
            
            if self.verbose and i % 100 == 0 and i > 0:
                elapsed = time.time() - start_time
                remaining = (elapsed / i) * (n_waypoints - i)
                print(f"  Progress: {i}/{n_waypoints} ({i/n_waypoints*100:.1f}%) - "
                      f"Est. remaining: {remaining:.1f}s")
        
        if self.verbose:
            elapsed = time.time() - start_time
            print(f"✓ Conversion complete in {elapsed:.2f}s")
            
        return positions, orientations
    
    def apply_transformation(
        self,
        positions: np.ndarray,
        orientations: np.ndarray,
        translation: np.ndarray = np.zeros(3),
        rotation_matrix: Optional[np.ndarray] = None,
        rotation_axis: Optional[str] = None,
        rotation_angle: float = 0.0,
        rotate_orientation: bool = False
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
            
        Returns:
            transformed_positions: Transformed positions
            transformed_orientations: Transformed orientations
        """
        if self.verbose:
            print(f"Applying transformation:")
            print(f"  Translation: {translation}")
            if rotation_matrix is not None:
                print(f"  Using provided rotation matrix")
            elif rotation_axis:
                print(f"  Rotation: {np.degrees(rotation_angle):.1f}° about {rotation_axis}-axis (base frame)")
                print(f"  Rotation center: end-effector start position")
            print(f"  Rotate orientation: {rotate_orientation}")
        
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
            centered_positions = positions - rotation_center
            rotated_positions = centered_positions @ rotation_matrix.T
            transformed_positions = rotated_positions + rotation_center + translation
        else:
            transformed_positions = positions + translation
        
        # Apply rotation to orientations only if requested
        if rotate_orientation and (rotation_angle != 0.0 or not np.allclose(rotation_matrix, np.eye(3))):
            # Rotate EE around its own local Z-axis (right multiply for local frame rotation)
            # O_new = O_original * R means "rotate around EE's own axes"
            # For Z-axis rotation, we want to rotate around the EE's local Z
            # Since rotation_matrix is in base frame, we need to apply it in local frame
            transformed_orientations = np.einsum('nij,jk->nik', orientations, rotation_matrix)
        else:
            # Keep orientations fixed in base frame (EE orientation doesn't change)
            transformed_orientations = orientations.copy()
        
        if self.verbose:
            print(f"✓ Transformation applied")
            
        return transformed_positions, transformed_orientations
    
    def cartesian_to_joints(
        self,
        positions: np.ndarray,
        orientations: np.ndarray,
        q_init: Optional[np.ndarray] = None,
        debug_info: Optional[dict] = None
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Convert Cartesian trajectory to joint space using inverse kinematics.
        
        Args:
            positions: Array of 3D positions [n_waypoints, 3]
            orientations: Array of 3x3 rotation matrices [n_waypoints, 3, 3]
            q_init: Initial joint configuration for IK (uses first solution if None)
            
        Returns:
            q_trajectory: Joint positions [n_valid_waypoints, 7]
            valid_indices: Indices of waypoints that had valid IK solutions
        """
        n_waypoints = len(positions)
        q_trajectory = []
        valid_indices = []
        failed_count = 0
        
        if self.verbose:
            print(f"Computing inverse kinematics for {n_waypoints} waypoints...")
        
        start_time = time.time()
        
        for i in range(n_waypoints):
            # Convert rotation matrix to quaternion (x, y, z, w format for panda_py.ik)
            rot = R.from_matrix(orientations[i])
            quat = rot.as_quat()  # Returns [x, y, z, w]
            orientation_quat = quat.reshape(4, 1)  # Keep as [x, y, z, w] column vector [4, 1]
            
            # Prepare position as column vector [3, 1]
            position_col = positions[i].reshape(3, 1)
            
            # Use previous solution as initial guess if available
            if len(q_trajectory) > 0:
                q_prev = q_trajectory[-1].reshape(7, 1)
            elif q_init is not None:
                q_prev = q_init.reshape(7, 1)
            else:
                # First call, use default
                q_prev = panda_py.ik(position_col, orientation_quat)
            
            try:
                # Compute IK
                q = panda_py.ik(position_col, orientation_quat, q_prev)
                
                # Flatten to 1D array for storage
                q_flat = q.flatten()
                
                # Check for NaN (IK failure)
                if np.any(np.isnan(q_flat)):
                    failed_count += 1
                    if i < 3 and self.verbose:
                        print(f"  Debug: Waypoint {i} IK returned NaN (no solution found)")
                        print(f"    Transformed position: {positions[i]}")
                        if debug_info and 'translation' in debug_info:
                            print(f"    Translation applied: {debug_info['translation']}")
                    continue
                
                # Check joint limits
                if np.all(q_flat >= self.joint_limits_lower) and np.all(q_flat <= self.joint_limits_upper):
                    q_trajectory.append(q_flat)
                    valid_indices.append(i)
                else:
                    failed_count += 1
                    if i < 3 and self.verbose:  # Print first few failures for debugging
                        print(f"  Debug: Waypoint {i} failed joint limits check")
                        for j in range(7):
                            if q_flat[j] < self.joint_limits_lower[j] or q_flat[j] > self.joint_limits_upper[j]:
                                print(f"    Joint {j}: {np.degrees(q_flat[j]):.2f}° (limits: [{np.degrees(self.joint_limits_lower[j]):.2f}°, {np.degrees(self.joint_limits_upper[j]):.2f}°])")
            except Exception as e:
                failed_count += 1
                if i < 3 and self.verbose:  # Print first few exceptions for debugging
                    print(f"  Debug: Waypoint {i} IK exception: {type(e).__name__}: {e}")
            
            if self.verbose and i % 100 == 0 and i > 0:
                elapsed = time.time() - start_time
                remaining = (elapsed / i) * (n_waypoints - i)
                success_rate = len(valid_indices) / i * 100
                print(f"  Progress: {i}/{n_waypoints} ({i/n_waypoints*100:.1f}%) - "
                      f"Success rate: {success_rate:.1f}% - "
                      f"Est. remaining: {remaining:.1f}s")
        
        if self.verbose:
            elapsed = time.time() - start_time
            success_rate = len(valid_indices) / n_waypoints * 100
            print(f"✓ IK computation complete in {elapsed:.2f}s")
            print(f"  Valid waypoints: {len(valid_indices)}/{n_waypoints} ({success_rate:.1f}%)")
            if failed_count > 0:
                print(f"  ⚠ Failed waypoints: {failed_count} (outside joint limits or no solution)")
        
        return np.array(q_trajectory), valid_indices
    
    def downsample_trajectory(
        self,
        q_trajectory: np.ndarray,
        max_waypoints: int = 100,
        min_distance: float = 0.01
    ) -> np.ndarray:
        """
        Downsample trajectory intelligently to reduce waypoints while preserving shape.
        
        Uses a combination of:
        1. Minimum distance criterion (skip waypoints too close together)
        2. Maximum waypoint limit (uniform downsampling if needed)
        
        Args:
            q_trajectory: Full joint trajectory [n_waypoints, 7]
            max_waypoints: Maximum number of waypoints to keep
            min_distance: Minimum joint-space distance between waypoints (radians)
            
        Returns:
            downsampled_trajectory: Reduced trajectory [n_reduced, 7]
        """
        if len(q_trajectory) <= max_waypoints:
            if self.verbose:
                print(f"Trajectory already has {len(q_trajectory)} waypoints (≤ {max_waypoints})")
            return q_trajectory
        
        if self.verbose:
            print(f"Downsampling trajectory from {len(q_trajectory)} to ~{max_waypoints} waypoints...")
        
        # First pass: remove waypoints that are too close
        reduced = [q_trajectory[0]]
        for i in range(1, len(q_trajectory)):
            dist = np.linalg.norm(q_trajectory[i] - reduced[-1])
            if dist >= min_distance:
                reduced.append(q_trajectory[i])
        
        reduced = np.array(reduced)
        
        if self.verbose:
            print(f"  After distance filtering: {len(reduced)} waypoints")
        
        # Second pass: uniform downsampling if still too many
        if len(reduced) > max_waypoints:
            indices = np.linspace(0, len(reduced) - 1, max_waypoints, dtype=int)
            reduced = reduced[indices]
            if self.verbose:
                print(f"  After uniform downsampling: {len(reduced)} waypoints")
        
        if self.verbose:
            print(f"✓ Downsampling complete: {len(q_trajectory)} → {len(reduced)} waypoints")
        
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
    
    def print_time_estimate(self, n_waypoints: int, max_downsampled: int = 100):
        """Print formatted time estimate."""
        estimates = self.estimate_computation_time(n_waypoints, max_downsampled)
        
        print("\n" + "="*60)
        print("ESTIMATED COMPUTATION TIME")
        print("="*60)
        print(f"Forward Kinematics:      {estimates['forward_kinematics']:>6.2f}s")
        print(f"Inverse Kinematics:      {estimates['inverse_kinematics']:>6.2f}s")
        print(f"Downsampling:            {estimates['downsampling']:>6.2f}s")
        print(f"Trajectory Generation:   {estimates['trajectory_generation']:>6.2f}s")
        print("-"*60)
        print(f"TOTAL ESTIMATED TIME:    {estimates['total']:>6.2f}s")
        print("="*60 + "\n")
    
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
        speed_factor: float = 0.1
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
            
        Returns:
            Dictionary with transformed trajectory data
        """
        print("\n" + "="*60)
        print("TRAJECTORY TRANSFORMATION PIPELINE")
        print("="*60 + "\n")
        
        # Load trajectory
        trajectory_data = self.load_trajectory(npy_path, primitive_name)
        q_original = np.array(trajectory_data['q'])
        
        # Print time estimate
        self.print_time_estimate(len(q_original), max_waypoints)
        
        # Convert to Cartesian
        positions, orientations = self.joints_to_cartesian(q_original)
        
        # Apply transformation
        positions_tf, orientations_tf = self.apply_transformation(
            positions, orientations,
            translation=translation,
            rotation_matrix=rotation_matrix,
            rotation_axis=rotation_axis,
            rotation_angle=rotation_angle,
            rotate_orientation=rotate_orientation
        )
        
        # Convert back to joint space
        q_transformed, valid_indices = self.cartesian_to_joints(
            positions_tf, orientations_tf, q_init=q_original[0],
            debug_info={'translation': translation}
        )
        
        if len(q_transformed) == 0:
            raise RuntimeError("No valid IK solutions found! Transformation may be invalid.")
        
        # Downsample
        q_downsampled = self.downsample_trajectory(
            q_transformed, max_waypoints=max_waypoints, min_distance=min_distance
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
        }
        
        print("\n" + "="*60)
        print("TRANSFORMATION COMPLETE")
        print("="*60)
        print(f"Original waypoints:     {len(q_original)}")
        print(f"Valid IK solutions:     {len(q_transformed)} ({len(q_transformed)/len(q_original)*100:.1f}%)")
        print(f"Downsampled waypoints:  {len(q_downsampled)}")
        print("="*60 + "\n")
        
        return result
    
    def save_transformed_trajectory(self, result: dict, output_path: str):
        """
        Save transformed trajectory to a .npy file.
        
        Args:
            result: Result dictionary from transform_trajectory
            output_path: Path to save the transformed trajectory
        """
        np.save(output_path, result)
        if self.verbose:
            print(f"✓ Saved transformed trajectory to {output_path}")
