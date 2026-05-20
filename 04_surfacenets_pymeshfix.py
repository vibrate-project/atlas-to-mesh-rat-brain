"""
04_surfacenets_pymeshfix.py

Generate a SurfaceNets mesh from the WHS SD rat atlas brain mask and repair it
with PyMeshFix to obtain a watertight final STL.

Purpose
-------
This script represents the final selected surface extraction branch of the
pipeline: SurfaceNets + PyMeshFix.

Input
-----
- data/WHS_SD_rat_atlas_v4.nii.gz

Output
------
- results/04_surfacenets_pymeshfix.stl
"""

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_fill_holes
import vtk
from vtk.util.numpy_support import numpy_to_vtk
import trimesh
import pymeshfix

# Configuration
NII_FILE = "data/WHS_SD_rat_atlas_v4.nii.gz"
OUTPUT_STL = "results/04_surfacenets_pymeshfix.stl"
DOWNSAMPLE_FACTOR = 4

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

print("\n=== Using vtkSurfaceNets3D ===")

# 6. Run SurfaceNets
surface_nets = vtk.vtkSurfaceNets3D()
surface_nets.SetInputData(vtk_image)
surface_nets.SetValue(0, 1)
surface_nets.SetBackgroundLabel(0)
surface_nets.SetSmoothing(True)
surface_nets.SetNumberOfIterations(25)
surface_nets.SetRelaxationFactor(0.4)
surface_nets.SetConstraintDistance(0.8)
surface_nets.SetOutputMeshTypeToTriangles()

print("Processing SurfaceNets...")
surface_nets.Update()

poly_data = surface_nets.GetOutput()
print(
    f"Raw output: {poly_data.GetNumberOfPoints()} vertices, {poly_data.GetNumberOfCells()} faces"
)

# 7. Convert VTK polydata to trimesh
points = np.array([poly_data.GetPoint(i) for i in range(poly_data.GetNumberOfPoints())])
faces = []
for i in range(poly_data.GetNumberOfCells()):
    cell = poly_data.GetCell(i)
    face = [cell.GetPointId(j) for j in range(cell.GetNumberOfPoints())]
    faces.append(face)

mesh = trimesh.Trimesh(vertices=points, faces=faces)
print(f"Before repair - Watertight: {mesh.is_watertight}, Euler: {mesh.euler_number}")

# 8. Repair mesh with PyMeshFix
print("\n=== Repairing mesh with pymeshfix ===")
meshfix = pymeshfix.MeshFix(mesh.vertices, mesh.faces)
meshfix.repair(verbose=True)

# Get repaired mesh
mesh_repaired = trimesh.Trimesh(vertices=meshfix.v, faces=meshfix.f)

print(f"\nAfter repair:")
print(f"Vertices: {len(mesh_repaired.vertices)}")
print(f"Faces: {len(mesh_repaired.faces)}")
print(f"Is watertight: {mesh_repaired.is_watertight}")
print(f"Euler number: {mesh_repaired.euler_number}")

if mesh_repaired.is_watertight:
    print(f"Volume: {mesh_repaired.volume:.2f} mm³")

# 9. Save repaired mesh
mesh_repaired.export(OUTPUT_STL)
print("\n✓ Saved!")
