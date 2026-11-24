# Polylla 3D Edge: tetrahedra are joined by their longest edge and 
# hanging polyhedra are then separated.

import sys
from collections import Counter
from pathlib import Path

# Allow imports to work both as a package and as a direct script
try:
    from .newMesh import EdgeTetrahedronMesh, Polyhedron
    from .polylla_mesh import PolyllaMesh
    from .metrics import face_area, polyhedron_faces
    from .merge_criteria import ConvexityCriterion
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from newMesh import EdgeTetrahedronMesh, Polyhedron
    from polylla_mesh import PolyllaMesh
    from metrics import face_area, polyhedron_faces
    from merge_criteria import ConvexityCriterion


class PolyllaEdge(PolyllaMesh):
    # separate_hanging=False builds every polyhedron as it is (simple or not);
    # True splits the non simple ones with Algorithm 10 (later stage)
    def __init__(self, mesh, separate_hanging=False, verbose=False, merge_criterion=None):
        super().__init__(mesh)
        # Label phase
        self.longest_edges = self.calculate_longest_edges()
        self.sorted_edges = self.sort_edges_by_length()
        if verbose:
            self.check_edge_order()

        # Joining phase: each edge, from longest to shortest, takes the
        # tetrahedra around it that were not used by a previous edge
        self.used_tetra = [False] * mesh.n_tetrahedrons
        self.polyhedra_with_hanging = 0
        for e in self.sorted_edges:
            polyhedron_tetras = self.join_around_edge(e)
            if not polyhedron_tetras:
                if verbose:
                    self.print_join_step(e, polyhedron_tetras, [])
                continue
            # Parts joined only by edge e (non simple polyhedron)
            parts = self.separate_hanging_polyhedra(e, polyhedron_tetras)
            has_hanging = len(parts) > 1
            if has_hanging:
                self.polyhedra_with_hanging += 1
            if not separate_hanging:
                parts = [polyhedron_tetras]
            first_poly = len(self.polyhedral_mesh)
            for tetras in parts:
                poly = Polyhedron()
                poly.tetras = tetras
                poly.faces = polyhedron_faces(self.mesh, tetras)
                poly.edge = e
                poly.has_hanging = has_hanging
                poly.was_repaired = has_hanging and separate_hanging
                self.polyhedral_mesh.append(poly)
            if verbose:
                self.print_join_step(e, polyhedron_tetras,
                                     range(first_poly, len(self.polyhedral_mesh)))

        # Merge phase: single tetrahedra are merged into a face neighbor
        # chosen by merge_criterion (None skips this phase)
        self.merged_singles = 0      # single tetrahedra merged into another polyhedron
        self.paired_singles = 0      # merges where the target was also a single tetrahedron
        self.unmerged_singles = 0    # single tetrahedra with no valid candidate
        if merge_criterion is not None:
            self.merge_single_tetrahedra(merge_criterion, verbose)

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
# MERGE PHASE
#############################################################################################

    # Each single tetrahedron, in creation order, is merged into the face neighbor
    # polyhedron with the best criterion score. If that neighbor is also a single
    # tetrahedron, both leave the pending list.
    def merge_single_tetrahedra(self, criterion, verbose=False):
        poly_of = {t: k for k, poly in enumerate(self.polyhedral_mesh) for t in poly.tetras}
        order = [poly.tetras[0] for poly in self.polyhedral_mesh if len(poly.tetras) == 1]
        pending = set(order)
        if verbose:
            print(f"Merge phase ({criterion.name}):")
        for t in order:
            if t not in pending:
                if verbose:
                    print(f"t{t} skipped (already merged)")
                continue
            pending.discard(t)
            k, evaluated = self.best_merge(t, poly_of, criterion)
            if verbose:
                self.print_merge_step(t, poly_of[t], evaluated, k)
            if k is None:
                self.unmerged_singles += 1
                continue
            target = self.polyhedral_mesh[k]
            if len(target.tetras) == 1:
                pending.discard(target.tetras[0])
                self.paired_singles += 1
            self.polyhedral_mesh[poly_of[t]] = None
            target.tetras = target.tetras + [t]
            target.faces = polyhedron_faces(self.mesh, target.tetras)
            target.was_merged = True
            poly_of[t] = k
            self.merged_singles += 1
        self.polyhedral_mesh = [poly for poly in self.polyhedral_mesh if poly is not None]
        if verbose:
            print(f"Merged {self.merged_singles}, paired {self.paired_singles}, unmerged {self.unmerged_singles}")

    # Polyhedra next to tetrahedron t, each with the faces it shares with t
    def merge_candidates(self, t, poly_of):
        shared = {}
        for f in self.mesh.tetra_list[t].faces:
            face = self.mesh.face_list[f]
            other = face.n2 if face.n1 == t else face.n1
            if other != -1:
                shared.setdefault(poly_of[other], []).append(f)
        return shared

    # Best candidate by (score, shared face area, lowest polyhedron id); candidates
    # that add non-manifold edges are discarded. Returns the chosen polyhedron (None
    # if there is no valid candidate) and the evaluated candidates as
    # (polyhedron, score, shared area), with score None when discarded.
    def best_merge(self, t, poly_of, criterion):
        best, best_key = None, None
        evaluated = []
        for k, shared in self.merge_candidates(t, poly_of).items():
            poly = self.polyhedral_mesh[k]
            tetras = poly.tetras + [t]
            area = sum(face_area(self.mesh, f) for f in shared)
            if self.non_manifold_edges(polyhedron_faces(self.mesh, tetras)) > self.non_manifold_edges(poly.faces):
                evaluated.append((k, None, area))
                continue
            score = criterion.score(self.mesh, tetras)
            evaluated.append((k, score, area))
            key = (round(score, 9), area, -k)
            if best_key is None or key > best_key:
                best, best_key = k, key
        return best, evaluated

    # Surface edges shared by more than two faces
    def non_manifold_edges(self, faces):
        counts = Counter(e for f in faces for e in self.mesh.face_list[f].edges)
        return sum(1 for c in counts.values() if c > 2)

#############################################################################################
# EXTRA
#############################################################################################

    # Debug: checks that edges go from longest to shortest
    def check_edge_order(self):
        lengths = [self.mesh.edge_list[e].length for e in self.sorted_edges]
        ok = all(a >= b for a, b in zip(lengths, lengths[1:]))
        print(f"Edges sorted from longest to shortest: {'OK' if ok else 'ERROR'}")

    # Debug: what edge e took, what was already used and the polyhedra it built
    def print_join_step(self, e, polyhedron_tetras, new_polys):
        edge = self.mesh.edge_list[e]
        ring = edge.tetrahedrons
        boundary = " (boundary)" if len(edge.faces) != len(ring) else ""
        print(f"e{e} v{edge.v1}-v{edge.v2} L={edge.length:.4f}{boundary} ring={ring}")
        if not polyhedron_tetras:
            print("    no free tetrahedra")
            return
        used_before = [t for t in ring if t not in polyhedron_tetras]
        print(f"    takes {polyhedron_tetras}, already used {used_before}")
        not_longest = [t for t in polyhedron_tetras if self.longest_edges[t] != e]
        if not_longest:
            print(f"    WARNING: e{e} is not the longest edge of {not_longest}")
        for k in new_polys:
            tetras = self.polyhedral_mesh[k].tetras
            single = "  <- single tetrahedron" if len(tetras) == 1 else ""
            print(f"    -> polyhedron {k}: {tetras}{single}")

    # Debug: candidates evaluated for single tetrahedron t and the chosen one
    def print_merge_step(self, t, own, evaluated, chosen):
        parts = []
        for k, score, area in evaluated:
            size = len(self.polyhedral_mesh[k].tetras)
            label = f"P{k} [{'single' if size == 1 else f'{size} tets'}]"
            if score is None:
                parts.append(f"{label} rejected (non-manifold)")
            else:
                parts.append(f"{label} score={score:.4f} area={area:.4f}")
        print(f"t{t} (polyhedron {own}): " + (" | ".join(parts) if parts else "no candidates"))
        if chosen is None:
            print("    -> stays single")
            return
        target = self.polyhedral_mesh[chosen].tetras
        result = target + [t]
        if len(target) == 1:
            print(f"    -> merged into polyhedron {chosen}, now tetrahedra {result} "
                  f"(single t{target[0]}: both leave the list)")
        else:
            print(f"    -> merged into polyhedron {chosen}, now tetrahedra {result}")

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
        print(f"Single tetrahedra merged: {self.merged_singles} (paired: {self.paired_singles}), "
              f"unmerged: {self.unmerged_singles}")
        print("Number of mesh faces: ", len(faces))
        print("Number of mesh edges:", len(edges))
        return len(self.polyhedral_mesh), self.polyhedra_with_hanging, count, len(faces), len(edges)


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.parent
    input_folder = PROJECT_ROOT / "data" / "input"
    output_folder = PROJECT_ROOT / "data" / "output"
    output_folder.mkdir(parents=True, exist_ok=True)

    # file can be "27uniform.1", "43quality.1", ... (quality meshes: experiments/generate_quality_meshes.py)
    file = "43quality.1"
    separate_hanging = False
    verbose = True
    merge_criterion = ConvexityCriterion()   # None to skip the merge phase
    node_file = input_folder / f"{file}.node"
    ele_file = input_folder / f"{file}.ele"
    face_file = input_folder / f"{file}.face"
    edge_file = input_folder / f"{file}.edge"

    print(f"Reading files from: {input_folder}")
    print(f"Writing files to: {output_folder}")

    mesh = EdgeTetrahedronMesh(str(node_file), str(face_file), str(ele_file), str(edge_file))
    polylla_mesh = PolyllaEdge(mesh, separate_hanging, verbose, merge_criterion)
    polylla_mesh.validate()

    polylla_mesh.printVISF_polyhedralmesh(str(output_folder / f"{file}_edge_polyhedral_mesh.visf"))
    polylla_mesh.printVISF_camaron(str(output_folder / f"{file}_edge_polyhedral_mesh_camaron.visf"))
    polylla_mesh.printPolyhedraIndex_camaron(str(output_folder / f"{file}_edge_polyhedra_camaron.txt"))

    polylla_mesh.get_info()
