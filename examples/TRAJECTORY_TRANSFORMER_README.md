# Trajectory Transformation for Panda Robot

This module provides tools to transform recorded robot trajectories by applying spatial transformations (translation and rotation) in Cartesian space.

## Overview

When you record a trajectory using guided mode at 1000Hz, you get a dense set of joint positions. This tool allows you to:

1. **Transform** the trajectory in Cartesian space (translate and/or rotate)
2. **Validate** that the transformed trajectory is kinematically feasible
3. **Downsample** intelligently to create efficient waypoints
4. **Execute** the transformed trajectory on the robot

## Files

- `trajectory_transformer.py` - Main transformation module
- `trajectory_transformer_example.py` - Example usage scripts

## Requirements

```bash
pip install numpy scipy panda-py
```

## Quick Start

### Basic Usage

```python
from trajectory_transformer import TrajectoryTransformer
import numpy as np

# Create transformer
transformer = TrajectoryTransformer(verbose=True)

# Transform trajectory
result = transformer.transform_trajectory(
    npy_path='path/to/recorded_trajectory.npy',
    primitive_name='my_primitive',
    translation=np.array([0.0, 0.1, 0.0]),  # Move 10cm in Y
    rotation_axis='z',                       # Rotate about Z
    rotation_angle=np.radians(15),          # 15 degrees
    max_waypoints=100,                      # Limit waypoints
    speed_factor=0.1                        # 10% of max speed
)

# Access transformed trajectory
q_waypoints = result['q_waypoints']  # Ready to execute
```

### Run the Example

```bash
python trajectory_transformer_example.py
```

Edit the configuration section in the example to match your setup.

## Key Features

### 1. Intelligent Downsampling

The transformer automatically reduces the 1000Hz recorded trajectory to a manageable number of waypoints while preserving the trajectory shape:

- **Distance-based filtering**: Removes waypoints that are too close together
- **Uniform downsampling**: Further reduces to fit within max_waypoints limit
- **Configurable**: Adjust `min_distance` and `max_waypoints` for your needs

### 2. Transformation Options

**Translation:**
```python
translation=np.array([0.1, 0.0, 0.05])  # X, Y, Z in meters
```

**Rotation (Axis-Angle):**
```python
rotation_axis='z'                    # 'x', 'y', or 'z'
rotation_angle=np.radians(45)       # Angle in radians
```

**Rotation (Custom Matrix):**
```python
rotation_matrix=np.array([
    [1, 0, 0],
    [0, np.cos(theta), -np.sin(theta)],
    [0, np.sin(theta), np.cos(theta)]
])
```

### 3. Validation

The transformer automatically:
- Checks joint limits for all waypoints
- Removes invalid IK solutions
- Reports success rate
- Provides detailed progress information

### 4. Time Estimation

Before processing, you get an estimate of computation time:

```
==============================================================
ESTIMATED COMPUTATION TIME
==============================================================
Forward Kinematics:        0.40s
Inverse Kinematics:        4.00s
Downsampling:              0.10s
Trajectory Generation:     2.00s
--------------------------------------------------------------
TOTAL ESTIMATED TIME:      6.50s
==============================================================
```

## Parameters Guide

### Transformation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `translation` | array [3] | `[0,0,0]` | Translation offset in meters [x, y, z] |
| `rotation_matrix` | array [3,3] | `None` | Custom 3x3 rotation matrix |
| `rotation_axis` | str | `None` | Rotation axis ('x', 'y', or 'z') |
| `rotation_angle` | float | `0.0` | Rotation angle in radians |

### Processing Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_waypoints` | int | `100` | Maximum waypoints after downsampling |
| `min_distance` | float | `0.01` | Minimum joint distance between waypoints (rad) |
| `speed_factor` | float | `0.1` | Speed factor for trajectory generation (0.0-1.0) |

### Tuning Recommendations

**For smoother trajectories:**
- Increase `max_waypoints` (e.g., 150-200)
- Decrease `min_distance` (e.g., 0.005)
- Slower `speed_factor` (e.g., 0.05)

**For faster computation:**
- Decrease `max_waypoints` (e.g., 50-75)
- Increase `min_distance` (e.g., 0.02)
- Accept some trajectory deviation

**For permissive parameters (you can wait):**
```python
result = transformer.transform_trajectory(
    npy_path=npy_path,
    primitive_name=primitive_name,
    translation=translation,
    rotation_axis='z',
    rotation_angle=np.radians(15),
    max_waypoints=200,      # More waypoints
    min_distance=0.003,     # Closer spacing
    speed_factor=0.05       # Slower, safer
)
```

## Output Data Structure

The `transform_trajectory()` function returns a dictionary with:

```python
{
    'q_original': np.ndarray,              # Original joint trajectory [n, 7]
    'q_transformed_full': np.ndarray,      # All valid transformed joints [m, 7]
    'q_waypoints': np.ndarray,             # Downsampled waypoints [k, 7]
    'valid_indices': list,                 # Indices of valid solutions
    'positions_original': np.ndarray,      # Original Cartesian positions [n, 3]
    'orientations_original': np.ndarray,   # Original orientations [n, 3, 3]
    'positions_transformed': np.ndarray,   # Transformed positions [m, 3]
    'orientations_transformed': np.ndarray,# Transformed orientations [m, 3, 3]
    'translation': np.ndarray,             # Applied translation [3]
    'rotation_matrix': np.ndarray,         # Applied rotation [3, 3]
    'primitive_name': str,                 # Name of primitive
}
```

## Common Issues

### "Trajectory generation failed"

This happens when:
- Waypoints are too far apart (increase `max_waypoints`)
- Speed is too high (decrease `speed_factor`)
- Invalid transformations (check joint limits)

**Solution**: Use more permissive parameters:
```python
max_waypoints=200,
min_distance=0.002,
speed_factor=0.05
```

### Low success rate (< 90%)

This indicates many waypoints have no valid IK solution:
- Transformation moves trajectory outside workspace
- Orientation changes too extreme
- Near singularities

**Solution**: Reduce transformation magnitude or change approach.

### Computation too slow

For 2000 waypoints at 1000Hz:
- FK: ~0.4s
- IK: ~4s
- Downsampling: ~0.1s
- Total: ~5-7s

This is normal. For faster processing, load only a subset of the trajectory.

## Example: Batch Processing

Create multiple transformed versions:

```python
transformations = [
    {'name': 'offset_x', 'translation': np.array([0.05, 0, 0])},
    {'name': 'offset_y', 'translation': np.array([0, 0.05, 0])},
    {'name': 'rotate_15', 'rotation_axis': 'z', 
     'rotation_angle': np.radians(15)},
]

for tf in transformations:
    result = transformer.transform_trajectory(
        npy_path=npy_path,
        primitive_name=primitive_name,
        **tf
    )
    transformer.save_transformed_trajectory(
        result, f'transformed_{tf["name"]}.npy'
    )
```

## Safety Notes

⚠️ **Always test with low speed_factor first** (e.g., 0.05)

⚠️ **Verify transformed workspace** is within robot reach

⚠️ **Check for collisions** in the transformed path

⚠️ **Have emergency stop ready** during first execution

## Advanced Usage

### Custom Transformation Function

```python
def custom_transform(positions, orientations):
    # Apply time-varying transformation
    n = len(positions)
    for i in range(n):
        t = i / n  # Normalized time [0, 1]
        # Spiral transformation
        angle = 2 * np.pi * t
        R = scipy.spatial.transform.Rotation.from_rotvec(
            [0, 0, angle]
        ).as_matrix()
        positions[i] = R @ positions[i]
        orientations[i] = R @ orientations[i]
    return positions, orientations
```

### Extract Trajectory Subset

```python
# Only transform middle portion of trajectory
start_idx = 500
end_idx = 1500
q_subset = trajectory_data['q'][start_idx:end_idx]

# Continue with transformation...
```

## Citation

If you use this tool in your research, please cite the panda-py library.

## License

Same as panda-py project.
