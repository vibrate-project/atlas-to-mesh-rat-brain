"""
05_step_to_tetrahedral_mesh.py

Import a repaired STEP model into Gmsh, apply OCC healing operations, and
generate a tetrahedral volume mesh.

Purpose
-------
This script represents the final volumetric meshing stage of the pipeline.
It assumes that a valid STEP solid has already been produced upstream.

Input
-----
- data/rat_brain.step

Outputs
-------
- results/05_brain_healed.msh
- results/05_brain_healed.stl
"""

import gmsh
import time

# Configuration
STEP_FILE = "data/rat_brain.step"
OUTPUT_MSH = "results/05_brain_healed.msh"
OUTPUT_STL = "results/05_brain_healed.stl"

# 1. Initialize Gmsh
gmsh.initialize()
gmsh.option.setNumber("General.Verbosity", 5)
gmsh.model.add("BrainSimplified")

# 2. Import STEP geometry
print("Loading STEP file...")
start = time.time()
gmsh.model.occ.importShapes(STEP_FILE)
gmsh.model.occ.synchronize()
print(f"✓ Loaded in {(time.time() - start)/60:.1f} minutes")

volumes = gmsh.model.occ.getEntities(3)
print(f"Initial volumes: {volumes}")

print("\nSimplifying & healing (since volume exists, focus on quality)...")

# 3. OCC healing options
gmsh.option.setNumber("Geometry.OCCFixDegenerated", 1)
gmsh.option.setNumber("Geometry.OCCFixSmallEdges", 1)
gmsh.option.setNumber("Geometry.OCCFixSmallFaces", 1)
gmsh.option.setNumber("Geometry.Tolerance", 1e-4)
gmsh.option.setNumber("Geometry.ToleranceBoolean", 1e-4)

surfaces = gmsh.model.occ.getEntities(2)
print(f"Surfaces: {len(surfaces)} (skipping fuse - use heal globally)")

print("Global healing...")
gmsh.model.occ.healShapes()  # Heals all shapes: fixes degens, small feats, sews where possible
gmsh.model.occ.synchronize()

# Optional: remove duplicates again post-heal
gmsh.model.occ.removeAllDuplicates()
gmsh.model.occ.synchronize()

# 4. Verify solid volume still exists
volumes = gmsh.model.occ.getEntities(3)
print(f"Volumes post-heal: {volumes}")
if len(volumes) == 0:
    raise ValueError("Lost volume during heal - STEP too defective")

# 5. Optional visualization aids
gmsh.option.setNumber("Geometry.Curves", 1)
gmsh.option.setNumber("Geometry.CurveWidth", 0.5)  # Thin curves

# 6. 3D meshing parameters
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.5)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 2.0)
gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT tetrahedral mesher
gmsh.option.setNumber("Mesh.Optimize", 2)  # Netgen optimization
gmsh.option.setNumber("Mesh.AngleToleranceFacetOverlap", 10.0)

# 7. Generate 3D mesh
print("\nGenerating 3D mesh...")
start = time.time()
ok = gmsh.model.mesh.generate(3)
if not ok:
    print("3D failed - generating surface mesh")
    gmsh.model.mesh.generate(2)
print(f"✓ Meshed in {(time.time() - start)/60:.1f} minutes")

# 8. Save outputs
gmsh.write(OUTPUT_MSH)
gmsh.write(OUTPUT_STL)
print(f"Saved {OUTPUT_MSH} and {OUTPUT_STL}")

# 9. Optional GUI
gmsh.fltk.run()
gmsh.finalize()
