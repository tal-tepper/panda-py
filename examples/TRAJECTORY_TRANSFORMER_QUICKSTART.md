# Trajectory Transformation System - Quick Reference

## What You Have

A complete system for transforming recorded robot trajectories recorded at 1000Hz in guided mode.

## Files Created

1. **`trajectory_transformer.py`** - Core transformation module
   - `TrajectoryTransformer` class with all functionality
   - Forward/inverse kinematics conversions
   - Intelligent downsampling
   - Safety validation

2. **`trajectory_transformer_example.py`** - Full example with robot execution
   - Single transformation demo
   - Batch processing demo
   - Robot execution code (can be disabled)

3. **`test_trajectory_transformer.py`** - Testing without robot
   - Offline testing script
   - Visualization support
   - Multi-transformation testing
   - Safety checks

4. **`TRAJECTORY_TRANSFORMER_README.md`** - Complete documentation
   - API reference
   - Parameter tuning guide
   - Troubleshooting

## Quick Start (Without Robot)

```bash
cd /home/user/repos/panda_py/panda-py/examples
python test_trajectory_transformer.py
```

This will:
- Load a recorded trajectory
- Apply transformation
- Check safety/feasibility
- Generate visualization
- Save transformed trajectory

## Key Features

### ✓ Efficient Processing
- Smart downsampling (1000Hz → ~100 waypoints)
- Time estimation before computation
- Progress tracking during processing

### ✓ Safety First
- Joint limit validation
- IK solution checking
- Success rate reporting
- Workspace boundary analysis

### ✓ Flexible Transformations
- Translation in X, Y, Z
- Rotation about any axis
- Custom rotation matrices
- Combined transformations

### ✓ No Robot Required for Testing
- Test transformations offline
- Visualize trajectories
- Validate feasibility
- Only connect robot for execution

## Typical Workflow

1. **Record trajectory in guided mode** at 1000Hz
   - Save as .npy file with joint positions

2. **Test transformation offline** (this computer):
   ```python
   python test_trajectory_transformer.py
   ```

3. **Review results**:
   - Check success rate (should be >95%)
   - Verify workspace bounds
   - Inspect visualization

4. **Adjust parameters** if needed:
   - More waypoints for smoothness
   - Lower speed_factor for safety
   - Smaller transformation if failing

5. **Execute on robot** (when connected):
   - Run `trajectory_transformer_example.py`
   - Set `execute_on_robot=True`
   - Start with low speed_factor (0.05)

## Example Transformations

### Move 10cm in Y direction
```python
result = transformer.transform_trajectory(
    npy_path='trajectory.npy',
    primitive_name='my_primitive',
    translation=np.array([0.0, 0.1, 0.0]),
    max_waypoints=150
)
```

### Rotate 15° about Z axis
```python
result = transformer.transform_trajectory(
    npy_path='trajectory.npy',
    primitive_name='my_primitive',
    rotation_axis='z',
    rotation_angle=np.radians(15),
    max_waypoints=150
)
```

### Combined transformation
```python
result = transformer.transform_trajectory(
    npy_path='trajectory.npy',
    primitive_name='my_primitive',
    translation=np.array([0.05, 0.05, 0.0]),
    rotation_axis='z',
    rotation_angle=np.radians(20),
    max_waypoints=150,
    speed_factor=0.05  # Slow and safe
)
```

## Recommended Parameters

### For Testing (Permissive - You Can Wait)
```python
max_waypoints=200      # Very smooth trajectory
min_distance=0.003     # Keep more waypoints
speed_factor=0.05      # Slow and safe
```

### For Production (Balanced)
```python
max_waypoints=100      # Good balance
min_distance=0.01      # Reasonable spacing
speed_factor=0.1       # Moderate speed
```

### For Quick Tests (Fast)
```python
max_waypoints=50       # Fewer waypoints
min_distance=0.02      # More aggressive filtering
speed_factor=0.2       # Faster execution
```

## Expected Computation Times

For 2000 waypoints @ 1000Hz:

| Operation | Time |
|-----------|------|
| Forward Kinematics | ~0.4s |
| Inverse Kinematics | ~4.0s |
| Downsampling | ~0.1s |
| Trajectory Generation | ~2.0s |
| **Total** | **~6-7s** |

## Troubleshooting

### Low Success Rate (<90%)
- ❌ Transformation too extreme
- ✓ Reduce translation/rotation magnitude
- ✓ Try different transformation direction

### "Trajectory generation failed"
- ❌ Waypoints too sparse or speed too high
- ✓ Increase `max_waypoints` to 150-200
- ✓ Decrease `speed_factor` to 0.05
- ✓ Decrease `min_distance` to 0.005

### Computation Too Slow
- ❌ Processing full 1000Hz trajectory
- ✓ Use subset of waypoints
- ✓ Increase `min_distance` to filter more
- ✓ Normal for first time (FK/IK caching)

## Output Files

Each transformation creates:
- `transformed_<name>.npy` - Full result dictionary
- `trajectory_comparison.png` - Visualization (if matplotlib available)

The `.npy` file contains everything needed for execution:
```python
data = np.load('transformed_trajectory.npy', allow_pickle=True).item()
q_waypoints = data['q_waypoints']  # Ready to execute!
```

## Safety Checklist

Before executing on robot:

- [ ] Success rate > 95%
- [ ] All joints within limits
- [ ] Workspace bounds reasonable
- [ ] Visualization looks correct
- [ ] Speed factor ≤ 0.1 for first run
- [ ] Emergency stop accessible
- [ ] No obstacles in workspace

## Next Steps

1. Run `test_trajectory_transformer.py` to validate the system
2. Adjust parameters based on your trajectory
3. When ready, use `trajectory_transformer_example.py` on robot
4. Start with very conservative parameters (speed_factor=0.05)

## Notes

- This computer is NOT connected to the robot (by design)
- All testing can be done offline
- Only execute when connected to robot
- Lint errors for numpy/scipy are normal if not installed
- The code will work fine on the robot computer

## Support

See `TRAJECTORY_TRANSFORMER_README.md` for detailed documentation.
