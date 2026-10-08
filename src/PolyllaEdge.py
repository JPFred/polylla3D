# Polylla 3D Edge (thesis of Magdalena Alvarez, section 3.3.2): tetrahedra
# are joined by their longest edge and hanging polyhedra are then separated.

import sys
from collections import Counter
from pathlib import Path

# Allow imports to work both as a package and as a direct script
try:
    from .newMesh import EdgeTetrahedronMesh, Polyhedron
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from newMesh import EdgeTetrahedronMesh, Polyhedron


class PolyllaEdge:
    def __init__(self, mesh):
        self.mesh = mesh
        # Label phase
        self.longest_edges = self.calculate_longest_edges()
        self.sorted_edges = self.sort_edges_by_length()

        # Joining phase
        self.visited_tetra = [False] * mesh.n_tetrahedrons
        self.polyhedra_with_hanging = 0
        self.polyhedral_mesh = []
        for e in self.sorted_edges:
            polyhedron_tetras = self.DepthFirstSearch(e)
            if polyhedron_tetras:
                # Repair phase: split parts joined only by edge e
                parts = self.separate_hanging_polyhedra(e, polyhedron_tetras)
                if len(parts) > 1:
                    self.polyhedra_with_hanging += 1
                for tetras in parts:
                    poly = Polyhedron()
                    poly.tetras = tetras
                    poly.faces = self.polyhedron_faces(tetras)
                    poly.was_repaired = len(parts) > 1
                    self.polyhedral_mesh.append(poly)

#############################################################################################
# LABEL PHASE
#############################################################################################

    # Edges are compared by (length, id), so ties are broken the same way everywhere
    def edge_key(self, e):
        return (self.mesh.edge_list[e].length, e)

    # Id of the longest edge of each tetrahedron
    def calculate_longest_edges(self):
        return [max(tetra.edges, key=self.edge_key) for tetra in self.mesh.tetra_list]

    # Edge ids from longest to shortest
    def sort_edges_by_length(self):
        return sorted(range(self.mesh.n_edges), key=self.edge_key, reverse=True)

#############################################################################################
# JOINING PHASE
#############################################################################################

    # Algorithm 9 (iterative): joins the unvisited tetrahedra around edge e and,
    # from each of them, the tetrahedra around its own longest edge
    def DepthFirstSearch(self, e):
        polyhedron_tetras = []
        stack = [e]
        while stack:
            edge = stack.pop()
            for tetra in self.mesh.edge_list[edge].tetrahedrons:
                if not self.visited_tetra[tetra]:
                    self.visited_tetra[tetra] = True
                    polyhedron_tetras.append(tetra)
                    e_max = self.longest_edges[tetra]
                    if e_max != edge:
                        stack.append(e_max)
        return polyhedron_tetras

    # Faces of the polyhedron: faces that belong to only one of its tetrahedra
    def polyhedron_faces(self, polyhedron_tetras):
        counts = Counter(f for t in polyhedron_tetras for f in self.mesh.tetra_list[t].faces)
        return [f for f, c in counts.items() if c == 1]

#############################################################################################
# REPAIR PHASE
#############################################################################################

    # Algorithm 10: all tetrahedra of the polyhedron surround edge e. Splits them
    # into the maximal runs of consecutive tetrahedra around e, which share faces
    # with each other, while different runs only touch at e.
    def separate_hanging_polyhedra(self, e, polyhedron_tetras):
        inside = set(polyhedron_tetras)
        ring = self.mesh.edge_list[e].tetrahedrons
        runs = []
        current = []
        for tetra in ring:
            if tetra in inside:
                current.append(tetra)
            elif current:
                runs.append(current)
                current = []
        if current:
            runs.append(current)
        # in a closed ring the last run continues into the first one
        closed = len(self.mesh.edge_list[e].faces) == len(ring)
        if closed and len(runs) > 1 and ring[0] in inside and ring[-1] in inside:
            runs[0] = runs.pop() + runs[0]
        return runs
