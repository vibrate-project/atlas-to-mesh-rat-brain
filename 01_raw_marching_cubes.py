"""
01_raw_marching_cubes.py

Generate a raw brain surface mesh from the WHS SD rat atlas using standard
Marching Cubes after simple 4x voxel downsampling.

Purpose
-------
This script represents the baseline Marching Cubes branch of the pipeline.
It uses only binarization and downsampling, without additional voxel
preprocessing such as hole filling or morphological closing.

Input
-----
- data/WHS_SD_rat_atlas_v4.nii.gz

Output
------
- results/01_raw_marching_cubes.stl
"""

import nibabel as nib
import numpy as np
from skimage import measure
import trimesh

# Configuration
NII_FILE = "data/WHS_SD_rat_atlas_v4.nii.gz"
DOWNSAMPLE = 4  # Factor 4 to fit in VM RAM (12 GB limit)
OUTPUT_STL = "results/01_raw_marching_cubes.stl"

# 1. Load atlas
img = nib.load(NII_FILE)
data = img.get_fdata()
voxel_size = np.array(img.header.get_zooms())

print(f"Voxel sizes      : {voxel_size} mm")
print(f"Original shape   : {data.shape}")
print(f"Brain voxels     : {(data > 0).sum()}")

# 2. Create binary mask
brain_mask = (data > 0).astype(np.uint8)  # 1 = any brain region, 0 = background

# 3. Downsample the mask
brain_downsampled = brain_mask[::DOWNSAMPLE, ::DOWNSAMPLE, ::DOWNSAMPLE]
effective_voxel_size = voxel_size * DOWNSAMPLE

print(f"Downsampled shape: {brain_downsampled.shape}")
print(f"Effective voxel  : {effective_voxel_size} mm")

# 4. Run Marching Cubes
verts, faces, normals, values = measure.marching_cubes(
    brain_downsampled,
    level=0.5,  # threshold between background (0) and brain (1)
    spacing=tuple(effective_voxel_size),
)

print(f"\n=== Marching Cubes output ===")
print(f"Vertices : {len(verts)}")
print(f"Faces    : {len(faces)}")

# 5. Build trimesh and report mesh statistics
mesh = trimesh.Trimesh(vertices=verts, faces=faces)
mesh.process(validate=True)  # Merge duplicates, remove degenerate faces

components = mesh.split(only_watertight=False)

print(f"\n=== Mesh statistics ===")
print(f"Components  : {len(components)}")
print(f"Is watertight: {mesh.is_watertight}")
print(f"Euler number: {mesh.euler_number}")
if mesh.is_watertight:
    print(f"Volume      : {mesh.volume:.2f} mm³")
else:
    print(f"Volume      : N/A (not watertight)")
print(f"Bounding box:\n{mesh.bounds}")
print(f"Center of mass: {mesh.center_mass}")

# 6. Save output mesh
mesh.export(OUTPUT_STL)
print(f"\n✓ Saved: {OUTPUT_STL}")
