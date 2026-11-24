# Experiments

Quality experiments for Polylla 3D Face and Edge.

- run_experiments.py (pending): runs each algorithm over every distribution
  and size in data/experiments/ and writes one JSON per run to results/.
- analysis.ipynb (pending): builds tables and figures from results/ and
  compares them with legacy/results/ (Magdalena's original results).
- neighbor_study.py: face and edge neighbors of the single-tetrahedron
  polyhedra of PolyllaEdge (quality and random meshes, 1000-10000
  points). Writes results/neighbor_study.json.
- generate_quality_meshes.py: quality meshes of the unit cube with TetGen
  (-pq, -a); by default writes the small 43quality mesh to data/input/.
- merge_comparison.py: PolyllaEdge without and with the merge phase of
  single tetrahedra (ConvexityCriterion) on quality meshes of 1000-10000
  points. Writes results/merge_comparison.json.
- results/: JSON results, versioned.
