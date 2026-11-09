# Polylla 3D Edge (thesis of Magdalena Alvarez, section 3.3.2): tetrahedra
# are joined by their longest edge and hanging polyhedra are then separated.

import sys
from collections import Counter
from pathlib import Path

# Allow imports to work both as a package and as a direct script
try:
    from .newMesh import EdgeTetrahedronMesh, Polyhedron
    from .polylla_mesh import PolyllaMesh
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from newMesh import EdgeTetrahedronMesh, Polyhedron
    from polylla_mesh import PolyllaMesh


class PolyllaEdge(PolyllaMesh):
    # separate_hanging=False builds every polyhedron as it is (simple or not);
    # True splits the non simple ones with Algorithm 10 (later stage)
    def __init__(self, mesh, separate_hanging=False):
        super().__init__(mesh)
        # Label phase
        self.longest_edges = self.calculate_longest_edges()
        self.sorted_edges = self.sort_edges_by_length()

        # Joining phase: each edge, from longest to shortest, takes the
        # tetrahedra around it that were not used by a previous edge
        self.used_tetra = [False] * mesh.n_tetrahedrons
        self.polyhedra_with_hanging = 0
        for e in self.sorted_edges:
            polyhedron_tetras = self.join_around_edge(e)
            if not polyhedron_tetras:
                continue
            # Parts joined only by edge e (non simple polyhedron)
            parts = self.separate_hanging_polyhedra(e, polyhedron_tetras)
            has_hanging = len(parts) > 1
            if has_hanging:
                self.polyhedra_with_hanging += 1
            if not separate_hanging:
                parts = [polyhedron_tetras]
            for tetras in parts:
                poly = Polyhedron()
                poly.tetras = tetras
                poly.faces = self.polyhedron_faces(tetras)
                poly.edge = e
                poly.has_hanging = has_hanging
                poly.was_repaired = has_hanging and separate_hanging
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

    # Unused tetrahedra around edge e, marked as used
    def join_around_edge(self, e):
        polyhedron_tetras = [t for t in self.mesh.edge_list[e].tetrahedrons
                             if not self.used_tetra[t]]
        for t in polyhedron_tetras:
            self.used_tetra[t] = True
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

#############################################################################################
# EXTRA
#############################################################################################

    def get_info(self):
        print("PolyllaEdge info:")
        print("Number of polyhedrons: " + str(len(self.polyhedral_mesh)))
        print("Number of polyhedra with hanging parts: " + str(self.polyhedra_with_hanging))
        count = sum(1 for poly in self.polyhedral_mesh if len(poly.tetras) == 1)
        faces = set()
        edges = set()
        for poly in self.polyhedral_mesh:
            faces.update(poly.faces)
            for f in poly.faces:
                edges.update(self.mesh.face_list[f].edges)
        print("Number of polyhedrons that are tetrahedrons: " + str(count))
        print("Number of mesh faces: ", len(faces))
        print("Number of mesh edges:", len(edges))
        return len(self.polyhedral_mesh), self.polyhedra_with_hanging, count, len(faces), len(edges)


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.parent
    input_folder = PROJECT_ROOT / "data" / "input"
    output_folder = PROJECT_ROOT / "data" / "output"
    output_folder.mkdir(parents=True, exist_ok=True)

    # file can be "1000points.1", "1000poisson.1", "1000random.1", "1000semiuniform.1", "1000uniform.1"
    file = "1000random.1"
    separate_hanging = False
    node_file = input_folder / f"{file}.node"
    ele_file = input_folder / f"{file}.ele"
    face_file = input_folder / f"{file}.face"
    edge_file = input_folder / f"{file}.edge"

    print(f"Reading files from: {input_folder}")
    print(f"Writing files to: {output_folder}")

    mesh = EdgeTetrahedronMesh(str(node_file), str(face_file), str(ele_file), str(edge_file))
    polylla_mesh = PolyllaEdge(mesh, separate_hanging)
    polylla_mesh.validate()

    polylla_mesh.printVISF_polyhedralmesh(str(output_folder / f"{file}_edge_polyhedral_mesh.visf"))
    polylla_mesh.printVISF_camaron(str(output_folder / f"{file}_edge_polyhedral_mesh_camaron.visf"))

    polylla_mesh.get_info()
