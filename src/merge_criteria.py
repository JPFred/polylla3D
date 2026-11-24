# Criteria to choose the polyhedron a single tetrahedron is merged into
# (strategy pattern): PolyllaEdge receives one criterion and only calls score().

import sys
from abc import ABC, abstractmethod
from pathlib import Path

# Allow imports to work both as a package and as a direct script
try:
    from .metrics import solidity
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from metrics import solidity


class MergeCriterion(ABC):
    name = "criterion"

    # Score of the polyhedron formed by tetras; higher is better
    @abstractmethod
    def score(self, mesh, tetras):
        pass


# Most convex result: highest solidity (volume / convex hull volume)
class ConvexityCriterion(MergeCriterion):
    name = "convexity"

    def score(self, mesh, tetras):
        return solidity(mesh, tetras)
