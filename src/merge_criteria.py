# Criteria to choose the polyhedron a single tetrahedron is merged into
# (strategy pattern): PolyllaEdge receives one criterion and only calls score().

import sys
from abc import ABC, abstractmethod
from pathlib import Path

# Allow imports to work both as a package and as a direct script
try:
    from .metrics import face_area, face_equilaterality, kernel_ratio, solidity
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from metrics import face_area, face_equilaterality, kernel_ratio, solidity


class MergeCriterion(ABC):
    name = "criterion"

    # Score of merging tetrahedron t into a polyhedron: tetras is the resulting
    # polyhedron and shared_faces the faces t shares with it. Higher is better.
    @abstractmethod
    def score(self, mesh, tetras, t, shared_faces):
        pass


# Most convex result: highest solidity (volume / convex hull volume)
class ConvexityCriterion(MergeCriterion):
    name = "convexity"

    def score(self, mesh, tetras, t, shared_faces):
        return solidity(mesh, tetras)


# Neighbor across the largest face of t
class LargestFaceCriterion(MergeCriterion):
    name = "largest_face"

    def score(self, mesh, tetras, t, shared_faces):
        return max(face_area(mesh, f) for f in shared_faces)


# Neighbor across the most equilateral face of t
class EquilateralFaceCriterion(MergeCriterion):
    name = "equilateral_face"

    def score(self, mesh, tetras, t, shared_faces):
        return max(face_equilaterality(mesh, f) for f in shared_faces)


# Result with the largest kernel relative to its volume (kernel ratio)
class KernelCriterion(MergeCriterion):
    name = "kernel"

    def score(self, mesh, tetras, t, shared_faces):
        return kernel_ratio(mesh, tetras)
