"""
02_preprocessed_marching_cubes.py

Generate a brain surface mesh from the WHS SD rat atlas using Marching Cubes
after voxel preprocessing, including hole filling, morphological closing,
and post-extraction component filtering.

Purpose
-------
This script represents the preprocessed Marching Cubes branch of the pipeline.
Compared with the raw Marching Cubes baseline, it applies voxel-level cleanup
before surface extraction and keeps only the largest connected mesh component.

Input
-----
- data/WHS_SD_rat_atlas_v4.nii.gz

Output
------
- results/02_preprocessed_marching_cubes.stl
"""

import nibabel as nib
import numpy as np
from skimage import measure
from scipy import ndimage
from scipy.ndimage import binary_fill_holes, binary_closing
import trimesh

# Configuration
NII_FILE = "data/WHS_SD_rat_atlas_v4.nii.gz"
DOWNSAMPLE = 4  # chosen to reduce memory usage and mesh size
OUTPUT_STL = "results/02_preprocessed_marching_cubes.stl"

# 1. Load atlas
img = nib.load(NII_FILE)
data = img.get_fdata()
voxel_size = np.array(img.header.get_zooms())

print(f"Voxel sizes      : {voxel_size} mm")
print(f"Original shape   : {data.shape}")
print(f"Brain voxels     : {(data > 0).sum()}")

# 2. Create binary mask
brain_mask = (data > 0).astype(np.uint8)

# 3. Downsample the mask
brain_downsampled = brain_mask[::DOWNSAMPLE, ::DOWNSAMPLE, ::DOWNSAMPLE]
effective_voxel_size = voxel_size * DOWNSAMPLE

print(f"Downsampled shape: {brain_downsampled.shape}")
print(f"Effective voxel  : {effective_voxel_size} mm")

# 4. Fill enclosed cavities
brain_filled = binary_fill_holes(brain_downsampled).astype(np.uint8)
filled_voxels = int(brain_filled.sum() - brain_downsampled.sum())
print(f"Voxels filled by fill_holes : {filled_voxels}")

# 5. Morphological closing
# This step helps seal the spinal cord opening and bridge small gaps in the mask.
structure = ndimage.generate_binary_structure(3, 2)
brain_closed = binary_closing(brain_filled, structure=structure, iterations=5).astype(
    np.uint8
)
closed_voxels = int(brain_closed.sum() - brain_filled.sum())
print(f"Voxels added by closing     : {closed_voxels}")

# 6. Run Marching Cubes
verts, faces, normals, values = measure.marching_cubes(
    brain_closed, level=0.5, spacing=tuple(effective_voxel_size)
)

print(f"\n=== Marching Cubes output ===")
print(f"Vertices : {len(verts)}")
print(f"Faces    : {len(faces)}")

# 7. Build trimesh
mesh = trimesh.Trimesh(vertices=verts, faces=faces)
mesh.process(validate=True)

# 8. Remove small disconnected components
components = mesh.split(only_watertight=False)
print(f"\n=== Before filtering ===")
print(f"Components   : {len(components)}")

mesh = max(components, key=lambda c: len(c.faces))
print(f"\n=== After filtering ===")
print(f"Kept largest component")
print(f"Vertices     : {len(mesh.vertices)}")
print(f"Faces        : {len(mesh.faces)}")
print(f"Is watertight: {mesh.is_watertight}")
print(f"Euler number : {mesh.euler_number}")
if mesh.is_watertight:
    print(f"Volume       : {mesh.volume:.2f} mm³")
else:
    print(f"Volume       : N/A (not watertight)")
print(f"Bounding box :\n{mesh.bounds}")

# 9. Save output mesh
mesh.export(OUTPUT_STL)
print(f"\n✓ Saved: {OUTPUT_STL}")
