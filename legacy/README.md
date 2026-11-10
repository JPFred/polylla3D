# Legacy material (reference only)

Original code, notebooks and results from Magdalena Álvarez's thesis
("Polyhedral Mesh Generation: Optimization and Evaluation", 2025).
Kept unchanged for reference and comparison; do not modify.

## Contents
- src/mesh.py: original mesh structure (TetrahedronMesh), by Sergio.
- src/PolyllaEdge.py: original Polylla 3D Edge (DFS version with repair).
  Replaced by src/PolyllaEdge.py in the main project.
- notebooks/faceExperiments.ipynb: Polylla Face experiments (random, poisson,
  semiuniform, uniform; 1000 to 50000 points; union flags r, a, are1, are2, tra).
  Writes one JSON per run to Results/.
- notebooks/experiments_analysis.ipynb: loads the JSON results and builds the
  LaTeX tables of the thesis.
- notebooks/Voronoi.ipynb, notebooks/voronoi.py: Voronoi meshes (pyvoro) from
  the same point sets, used as a reference polyhedral mesh.
- notebooks/generatePolylla.ipynb, notebooks/polylla-time.ipynb: comparison of
  mesh.py (Sergio) vs newMesh.py (Magdalena) structures and reading times.
- notebooks/kernel: Linux binary (not versioned) that computes kernel volumes.
- results/: the 145 JSON results of faceExperiments.ipynb.

## Kernel tool
Built from "polyhedron_kernel" (Sorgente, Biasotti, Spagnuolo, "Polyhedron
kernel computation using a geometric approach", Computers & Graphics 105, 2022),
using cinolib (commit f7d14627f353e37e8a5c6c7c2bf442cc956a4a8b), with main.cpp
modified to read the .txt written by PolyllaFace.writePolygonFile and to write
<name>_kernel_volumes.txt: one kernel volume per non-convex polyhedron, and a
last line "<polyhedra with kernel> <total>".

## Known issues (found while reviewing, not fixed here)
- The notebooks expect paths of the original repository (../data/Experiments/,
  sys.path to the old src) and will not run as they are.
- Input meshes of the experiments were not kept; random and poisson point sets
  were generated without a fixed seed.
- convex_polyhedrons() counts a polyhedron as convex when all its vertices lie
  on the convex hull, which also holds for some non-convex polyhedra.
- volume_ratio(): len([kernel_volumes != 0]) is always 1, and polyhedra with
  kernel volume 0 are left out of the average ratio.
