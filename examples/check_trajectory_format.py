"""
Trajectory Data Format Checker

This script helps you understand the expected format for recorded trajectories
and validates that your .npy files are compatible with the transformer.
"""

import numpy as np
import sys


def check_trajectory_file(npy_path: str):
    """
    Check if trajectory file has the expected format.
    
    Expected format:
    - Dictionary with primitive names as keys
    - Each primitive has: 'q', 'dq', 'position', 'orientation', etc.
    - 'q': joint positions [n_waypoints, 7]
    - 'dq': joint velocities [n_waypoints, 7] (optional)
    - 'position': Cartesian positions [n_waypoints, 3] (optional)
    - 'orientation': Rotation matrices [n_waypoints, 3, 3] (optional)
    """
    
    print("\n" + "="*70)
    print(" TRAJECTORY FILE FORMAT CHECKER")
    print("="*70 + "\n")
    
    print(f"Checking file: {npy_path}\n")
    
    try:
        # Load file
        data = np.load(npy_path, allow_pickle=True).item()
        print("✓ File loaded successfully")
        
        if not isinstance(data, dict):
            print("✗ Error: File should contain a dictionary")
            return False
        
        print(f"✓ File contains a dictionary with {len(data)} primitives\n")
        
        # List primitives
        print("Available primitives:")
        for i, name in enumerate(data.keys(), 1):
            print(f"  {i}. {name}")
        
        print("\n" + "-"*70 + "\n")
        
        # Check each primitive
        all_valid = True
        for prim_name, prim_data in data.items():
            print(f"Checking primitive: '{prim_name}'")
            
            if not isinstance(prim_data, dict):
                print(f"  ✗ Primitive data should be a dictionary")
                all_valid = False
                continue
            
            # Check required field: 'q'
            if 'q' not in prim_data:
                print(f"  ✗ Missing required field: 'q' (joint positions)")
                all_valid = False
                continue
            
            q = prim_data['q']
            
            if not isinstance(q, np.ndarray):
                print(f"  ✗ 'q' should be a numpy array")
                all_valid = False
                continue
            
            if q.ndim != 2 or q.shape[1] != 7:
                print(f"  ✗ 'q' should have shape [n_waypoints, 7], got {q.shape}")
                all_valid = False
                continue
            
            n_waypoints = len(q)
            print(f"  ✓ 'q': {n_waypoints} waypoints, shape {q.shape}")
            
            # Check joint limits
            joint_limits_lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
            joint_limits_upper = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
            
            within_limits = True
            for i in range(7):
                if np.any(q[:, i] < joint_limits_lower[i]) or np.any(q[:, i] > joint_limits_upper[i]):
                    within_limits = False
                    print(f"  ⚠ Joint {i+1} exceeds limits: [{np.degrees(q[:, i].min()):.1f}°, {np.degrees(q[:, i].max()):.1f}°]")
            
            if within_limits:
                print(f"  ✓ All joint values within limits")
            
            # Check optional fields
            optional_fields = {
                'dq': (n_waypoints, 7, "joint velocities"),
                'position': (n_waypoints, 3, "Cartesian positions"),
                'orientation': (n_waypoints, 3, 3, "rotation matrices"),
            }
            
            for field_name, expected in optional_fields.items():
                if field_name in prim_data:
                    field_data = prim_data[field_name]
                    if isinstance(field_data, np.ndarray):
                        expected_shape = expected[:-1]
                        if field_data.shape == expected_shape:
                            print(f"  ✓ '{field_name}': {field_data.shape} ({expected[-1]})")
                        else:
                            print(f"  ⚠ '{field_name}': shape {field_data.shape}, expected {expected_shape}")
                    else:
                        print(f"  ⚠ '{field_name}': not a numpy array")
                else:
                    print(f"  - '{field_name}': not present (optional)")
            
            # Print statistics
            print(f"\n  Statistics:")
            print(f"    Duration: {n_waypoints / 1000:.2f}s (assuming 1000 Hz)")
            print(f"    Joint ranges (degrees):")
            for i in range(7):
                q_range = q[:, i].max() - q[:, i].min()
                print(f"      Joint {i+1}: {np.degrees(q_range):6.2f}°")
            
            if 'position' in prim_data:
                pos = prim_data['position']
                print(f"    Workspace:")
                print(f"      X: [{pos[:, 0].min():.3f}, {pos[:, 0].max():.3f}] m")
                print(f"      Y: [{pos[:, 1].min():.3f}, {pos[:, 1].max():.3f}] m")
                print(f"      Z: [{pos[:, 2].min():.3f}, {pos[:, 2].max():.3f}] m")
            
            print()
        
        print("-"*70)
        
        if all_valid:
            print("\n✓ All primitives are valid and compatible with transformer\n")
        else:
            print("\n⚠ Some primitives have issues (see above)\n")
        
        print("="*70 + "\n")
        
        return all_valid
        
    except FileNotFoundError:
        print(f"✗ Error: File not found: {npy_path}")
        return False
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_example_trajectory_file(output_path: str = 'example_trajectory.npy'):
    """
    Create an example trajectory file showing the expected format.
    """
    
    print("\n" + "="*70)
    print(" CREATING EXAMPLE TRAJECTORY FILE")
    print("="*70 + "\n")
    
    # Create a simple circular trajectory example
    n_waypoints = 2000  # 2 seconds at 1000 Hz
    t = np.linspace(0, 2*np.pi, n_waypoints)
    
    # Example joint trajectory (circular motion in joint space)
    q = np.zeros((n_waypoints, 7))
    q[:, 0] = 0.0 + 0.2 * np.sin(t)
    q[:, 1] = -np.pi/4 + 0.1 * np.cos(t)
    q[:, 2] = 0.0
    q[:, 3] = -3*np.pi/4
    q[:, 4] = 0.0
    q[:, 5] = np.pi/2 + 0.1 * np.sin(t)
    q[:, 6] = np.pi/4
    
    # Example velocity (derivative of position)
    dq = np.zeros((n_waypoints, 7))
    dq[:, 0] = 0.2 * np.cos(t)
    dq[:, 1] = -0.1 * np.sin(t)
    dq[:, 5] = 0.1 * np.cos(t)
    
    # Note: In real data, you would compute position/orientation from FK
    # Here we create dummy data for demonstration
    position = np.zeros((n_waypoints, 3))
    position[:, 0] = 0.5 + 0.1 * np.cos(t)
    position[:, 1] = 0.0 + 0.1 * np.sin(t)
    position[:, 2] = 0.4
    
    # Rotation matrices (identity for simplicity)
    orientation = np.tile(np.eye(3), (n_waypoints, 1, 1))
    
    # Create data dictionary
    data = {
        'example_circle': {
            'q': q,
            'dq': dq,
            'position': position,
            'orientation': orientation,
        },
        'example_line': {
            'q': q[:1000],  # First half only
            'dq': dq[:1000],
            'position': position[:1000],
            'orientation': orientation[:1000],
        }
    }
    
    # Save file
    np.save(output_path, data)
    
    print(f"✓ Created example trajectory file: {output_path}")
    print(f"  Contains 2 primitives: 'example_circle' and 'example_line'")
    print(f"  Format: Compatible with trajectory transformer")
    print("\n" + "="*70 + "\n")
    
    return output_path


def show_usage():
    """Show usage instructions."""
    print("""
Usage:
    python check_trajectory_format.py <path_to_trajectory.npy>
    
    or
    
    python check_trajectory_format.py --create-example
    
Examples:
    # Check existing file
    python check_trajectory_format.py /path/to/recorded_trajectory.npy
    
    # Create example file
    python check_trajectory_format.py --create-example
    
Expected Format:
    The .npy file should contain a dictionary:
    {
        'primitive_name_1': {
            'q': np.array([n_waypoints, 7]),         # Required: joint positions
            'dq': np.array([n_waypoints, 7]),        # Optional: joint velocities
            'position': np.array([n_waypoints, 3]),  # Optional: Cartesian positions
            'orientation': np.array([n_waypoints, 3, 3])  # Optional: rotation matrices
        },
        'primitive_name_2': { ... },
        ...
    }
    
    The transformer only requires 'q' (joint positions). Other fields are optional
    but will be recomputed if not present.
""")


if __name__ == '__main__':
    
    if len(sys.argv) < 2:
        show_usage()
        sys.exit(1)
    
    if sys.argv[1] == '--create-example':
        output_path = sys.argv[2] if len(sys.argv) > 2 else 'example_trajectory.npy'
        created_path = create_example_trajectory_file(output_path)
        print(f"Now run: python check_trajectory_format.py {created_path}")
    elif sys.argv[1] in ['--help', '-h']:
        show_usage()
    else:
        npy_path = sys.argv[1]
        valid = check_trajectory_file(npy_path)
        
        if valid:
            print("✓ File is ready to use with trajectory transformer!")
            print("\nNext step:")
            print(f"  python test_trajectory_transformer.py")
            sys.exit(0)
        else:
            print("⚠ File has issues. Please check the format.")
            sys.exit(1)
