"""
03_raw_surfacenets.py

Generate a raw surface mesh of the WHS SD rat atlas brain mask using
VTK SurfaceNets after 4x voxel downsampling and hole filling.

Purpose
-------
This script represents the raw SurfaceNets branch of the pipeline.
It is intended as a checkpoint script before mesh repair with PyMeshFix.

Input
-----
- data/WHS_SD_rat_atlas_v4.nii.gz

Output
------
- results/03_raw_surfacenets.stl
"""

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_fill_holes
import vtk
from vtk.util.numpy_support import numpy_to_vtk
import trimesh

# Configuration
NII_FILE = "data/WHS_SD_rat_atlas_v4.nii.gz"
OUTPUT_STL = "results/03_raw_surfacenets.stl"
DOWNSAMPLE_FACTOR = 4  # chosen to keep memory and mesh size manageable

# 1. Load atlas
img = nib.load(NII_FILE)
data = img.get_fdata()
voxel_sizes = np.array(img.header.get_zooms())
print(f"Original voxel sizes: {voxel_sizes} mm")

# 2. Create binary brain mask
brain_mask = (data > 0).astype(np.uint8)
print(f"Original brain voxels: {np.sum(brain_mask)}")

# 3. Downsample the mask
brain_mask_downsampled = ndimage.zoom(brain_mask, 1 / DOWNSAMPLE_FACTOR, order=0)
effective_voxel_size = voxel_sizes * DOWNSAMPLE_FACTOR
print(f"Downsampled to: {brain_mask_downsampled.shape}")
print(f"Effective voxel size: {effective_voxel_size} mm")

# 4. Fill enclosed cavities
brain_filled = binary_fill_holes(brain_mask_downsampled).astype(np.uint8)
print(f"Filled {np.sum(brain_filled) - np.sum(brain_mask_downsampled)} voxels")

# 5. Convert binary volume to VTK image
vtk_data = numpy_to_vtk(
    brain_filled.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR
)
vtk_image = vtk.vtkImageData()
vtk_image.SetDimensions(brain_filled.shape)
vtk_image.SetSpacing(effective_voxel_size)
vtk_image.GetPointData().SetScalars(vtk_data)

# 6. Run SurfaceNets
print("\n=== Using vtkSurfaceNets3D ===")
surface_nets = vtk.vtkSurfaceNets3D()
surface_nets.SetInputData(vtk_image)
surface_nets.SetValue(0, 1)
surface_nets.SetBackgroundLabel(0)
surface_nets.SetSmoothing(True)
surface_nets.SetNumberOfIterations(25)
surface_nets.SetRelaxationFactor(0.4)
surface_nets.SetConstraintDistance(0.8)
surface_nets.SetOutputMeshTypeToTriangles()
surface_nets.Update()
poly_data = surface_nets.GetOutput()

# 7. Convert VTK polydata to trimesh
points = np.array([poly_data.GetPoint(i) for i in range(poly_data.GetNumberOfPoints())])
faces = [
    [
        poly_data.GetCell(i).GetPointId(j)
        for j in range(poly_data.GetCell(i).GetNumberOfPoints())
    ]
    for i in range(poly_data.GetNumberOfCells())
]

mesh = trimesh.Trimesh(vertices=points, faces=faces)
print(f"Raw output: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
print(f"Watertight: {mesh.is_watertight}, Euler: {mesh.euler_number}")

components = mesh.split(only_watertight=False)
print(f"Components: {len(components)}")

# 8. Save output mesh
mesh.export(OUTPUT_STL)
print(f"\n✓ Saved: {OUTPUT_STL}")
