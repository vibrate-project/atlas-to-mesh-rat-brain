# From Atlas to Mesh: Rat Brain Surface and Volume Meshing Pipeline

This repository contains a stepwise pipeline for extracting a 3D rat brain surface from the WHS SD rat atlas and converting the resulting geometry into a tetrahedral volume mesh.

The project is organized as a sequence of checkpoint scripts. Each script represents a distinct branch or stage of the workflow, from baseline surface extraction to the final repaired surface and volumetric meshing.

## Pipeline overview

The repository includes five main scripts:

1. `01_raw_marching_cubes.py`  
   Baseline Marching Cubes extraction on a 4× downsampled binary brain mask.

2. `02_preprocessed_marching_cubes.py`  
   Marching Cubes with voxel preprocessing, including hole filling, morphological closing, and largest-component filtering.

3. `03_raw_surfacenets.py`  
   Raw SurfaceNets extraction after hole filling.

4. `04_surfacenets_pymeshfix.py`  
   SurfaceNets extraction followed by PyMeshFix repair to obtain a watertight final surface mesh.

5. `05_step_to_tetrahedral_mesh.py`  
   Import of a repaired STEP model into Gmsh, OCC healing, and tetrahedral volume meshing.

## Input data

The scripts expect the following input files:

- `data/WHS_SD_rat_atlas_v4.nii.gz` — the WHS SD rat atlas volume used for surface extraction
- `data/rat_brain.step` — a repaired STEP model used for final tetrahedral meshing

## Dependencies

This project uses Python and several scientific/geometry-processing libraries.

Typical Python dependencies include:

- `nibabel`
- `numpy`
- `scipy`
- `scikit-image`
- `trimesh`
- `vtk`
- `pymeshfix`
- `gmsh`

In practice, scripts `01` to `04` represent alternative or progressive surface-extraction branches, while `05` is the downstream volumetric meshing stage after a valid repaired solid has been obtained.

## Notes

- All scripts assume relative paths under `data/` and `results/`.
- The `results/` directory should be created before running the scripts, or created automatically in future revisions.
- Depending on your system, VTK, PyMeshFix, and Gmsh may require extra installation steps.
- `05_step_to_tetrahedral_mesh.py` is intended for use after a valid STEP solid has been generated upstream.

## Tested environment

The scripts were developed and tested on:

- Ubuntu 24.04 LTS
- Python 3.12.3
- nibabel 5.2.1
- numpy 1.26.4
- scipy 1.11.4
- scikit-image 0.22.0
- trimesh 4.10.1
- pymeshfix 0.17.2
- gmsh 4.12.1

VTK is required by the SurfaceNets scripts, but it was not available in the Python environment used to record the package versions above.
