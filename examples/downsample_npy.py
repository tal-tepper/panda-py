"""
Downsample a trajectory .npy file using joint-space distance filtering.

Loads the original file, downsamples each primitive's joint trajectory
using a minimum distance threshold, then recomputes Cartesian positions
and orientations via FK. Joint velocities (dq) are approximated by
finite differences.

Usage:
    python downsample_npy.py [--input FILE] [--output FILE] [--min-distance D]
"""

import sys
import os
import argparse
import numpy as np
import panda_py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trajectory_test_utils import downsample_by_distance

DEFAULT_INPUT = '/home/talte/repos/tactile_panda/primitive_files/pose_34.npy'
DEFAULT_MIN_DISTANCE = 0.0005

parser = argparse.ArgumentParser(description='Downsample trajectory .npy file.')
parser.add_argument('--input', '-i', default=DEFAULT_INPUT,
                    help='Input .npy file path')
parser.add_argument('--output', '-o', default=None,
                    help='Output .npy file path (default: auto-generated)')
parser.add_argument('--min-distance', '-d', type=float, default=DEFAULT_MIN_DISTANCE,
                    help='Min joint-space L2 distance between consecutive waypoints')
args = parser.parse_args()

# Auto-generate output name
if args.output is None:
    base, ext = os.path.splitext(args.input)
    d_str = f"{args.min_distance:.6f}".rstrip('0').rstrip('.')
    args.output = f"{base}_downsampled_d{d_str}{ext}"

print(f"Input:        {args.input}")
print(f"Output:       {args.output}")
print(f"Min distance: {args.min_distance}")
print()

# Load
raw = np.load(args.input, allow_pickle=True).item()

result = {}

for prim_name, data in raw.items():
    q_full = np.array(data['q'])
    n_orig = len(q_full)

    # Downsample by distance
    q_ds = downsample_by_distance(q_full, args.min_distance)
    n_ds = len(q_ds)
    ratio = n_orig / n_ds if n_ds > 0 else float('inf')

    print(f"{prim_name}: {n_orig} → {n_ds} waypoints ({ratio:.1f}x reduction)")

    # Compute FK for positions and orientations
    positions = np.zeros((n_ds, 3))
    orientations = np.zeros((n_ds, 3, 3))
    for i, q in enumerate(q_ds):
        pose = panda_py.fk(q)  # 4x4 matrix
        positions[i] = pose[:3, 3]
        orientations[i] = pose[:3, :3]

    # Approximate dq by finite differences
    # Use the same dt=0.001 as the original recording (1 kHz control loop)
    dt = 0.001
    dq = np.zeros_like(q_ds)
    if n_ds > 1:
        # Central differences for interior points
        dq[1:-1] = (q_ds[2:] - q_ds[:-2]) / (2 * dt)
        # Forward/backward for endpoints
        dq[0] = (q_ds[1] - q_ds[0]) / dt
        dq[-1] = (q_ds[-1] - q_ds[-2]) / dt

    result[prim_name] = {
        'position': positions,
        'orientation': orientations,
        'q': q_ds,
        'dq': dq,
    }

# Save
np.save(args.output, result)
print(f"\nSaved → {args.output}")

# Verify
check = np.load(args.output, allow_pickle=True).item()
print("\nVerification:")
for prim_name, data in check.items():
    print(f"  {prim_name}:")
    for k, v in data.items():
        arr = np.array(v)
        print(f"    {k}: shape={arr.shape}, dtype={arr.dtype}")
