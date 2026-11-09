# Common base for Polylla 3D algorithms: holds the tetrahedral mesh and the
# resulting polyhedral mesh, and provides validation and export to files.

import sys
from collections import Counter
from pathlib import Path

# Allow imports to work both as a package and as a direct script
try:
    from .utils import ccw_check
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import ccw_check


class PolyllaMesh:
    def __init__(self, mesh):
        self.mesh = mesh
        self.polyhedral_mesh = []

#############################################################################################
# VALIDATION
#############################################################################################

    # Topological checks of the polyhedral mesh. Returns the number of errors;
    # non-manifold edges are reported but not counted (algorithm limitation).
    def validate(self):
        membership = Counter(t for poly in self.polyhedral_mesh for t in poly.tetras)
        lost = [t for t in range(self.mesh.n_tetrahedrons) if membership[t] == 0]
        repeated = [t for t, n in membership.items() if n > 1]
        internal_faces = []   # both tetrahedra of a face inside the polyhedron
        open_polys = []       # some surface edge used an odd number of times
        non_manifold = []     # some surface edge shared by more than two faces
        for i, poly in enumerate(self.polyhedral_mesh):
            tetras = set(poly.tetras)
            faces = set(poly.faces)
            if any(self.mesh.face_list[f].n1 in tetras and self.mesh.face_list[f].n2 in tetras for f in faces):
                internal_faces.append(i)
            edge_count = Counter(e for f in faces for e in self.mesh.face_list[f].edges)
            if any(c % 2 for c in edge_count.values()):
                open_polys.append(i)
            elif any(c > 2 for c in edge_count.values()):
                non_manifold.append(i)
        print("Validation:")
        print("Lost tetrahedra:", len(lost), lost[:10])
        print("Tetrahedra in more than one polyhedron:", len(repeated), repeated[:10])
        print("Polyhedra with internal faces:", len(internal_faces), internal_faces[:10])
        print("Open polyhedra:", len(open_polys), open_polys[:10])
        print("Polyhedra with non-manifold edges:", len(non_manifold), non_manifold[:10])
        return len(lost) + len(repeated) + len(internal_faces) + len(open_polys)

#############################################################################################
# EXPORT
#############################################################################################

    def printOFF_faces(self, filename, faces):
            print("writing OFF file: "+ filename)
            with open(filename, 'w') as fh:
                fh.write("OFF\n")
                fh.write("%d %d 0\n" % (self.mesh.n_nodes, len(faces)))
                for v in self.mesh.node_list:
                    fh.write("%f %f %f\n" % (v.x, v.y, v.z))
                for f in faces:
                    v1 = self.mesh.face_list[f.i].v1
                    v2 = self.mesh.face_list[f.i].v2
                    v3 = self.mesh.face_list[f.i].v3
                    fh.write("3 %d %d %d\n" % (v1, v2, v3))

# Revisar vs anterior colores
    def printOFF_polyhedralmesh_colors(self, filename):
        print("writing OFF file: " + filename)
        list_face = []

        # Center the mesh around the centroid
        centroid_x = sum(v.x for v in self.mesh.node_list) / len(self.mesh.node_list)
        centroid_y = sum(v.y for v in self.mesh.node_list) / len(self.mesh.node_list)
        centroid_z = sum(v.z for v in self.mesh.node_list) / len(self.mesh.node_list)

        for polyhedron in self.polyhedral_mesh:
            for face in polyhedron.faces:
                t = self.mesh.face_list[face].n1 if (self.mesh.face_list[face].n1 in polyhedron.tetras) else self.mesh.face_list[face].n2
                if not ccw_check(self.mesh.face_list[face], self.mesh.tetra_list[t], self.mesh.node_list):
                    v2 = self.mesh.face_list[face].v2
                    v3 = self.mesh.face_list[face].v3
                    self.mesh.face_list[face].v2 = v3
                    self.mesh.face_list[face].v3 = v2
                list_face.append(face)

        list_face = list(dict.fromkeys(list_face))

        with open(filename, 'w') as fh:
            fh.write("OFF\n")
            fh.write("%d %d 0\n" % (self.mesh.n_nodes, len(list_face)))
            for v in self.mesh.node_list:
                fh.write("%f %f %f\n" % (
                    (v.x - centroid_x) * 100,
                    (v.y - centroid_y) * 100,
                    (v.z - centroid_z) * 100
                ))
            for f in list_face:
                v1 = self.mesh.face_list[f].v1
                v2 = self.mesh.face_list[f].v2
                v3 = self.mesh.face_list[f].v3
                fh.write("3 %d %d %d 3 0.8 0.8 0.8\n" % (v1, v2, v3))

    def printOFF_polyhedralmesh(self, filename):
        print("writing OFF file: "+ filename)
        list_face = []
        
        # Calculate centroid
        centroid_x = sum(v.x for v in self.mesh.node_list) / len(self.mesh.node_list)
        centroid_y = sum(v.y for v in self.mesh.node_list) / len(self.mesh.node_list)
        centroid_z = sum(v.z for v in self.mesh.node_list) / len(self.mesh.node_list)
        
        for polyhedron in self.polyhedral_mesh:
            for face in polyhedron.faces:
                t = self.mesh.face_list[face].n1 if (self.mesh.face_list[face].n1 in polyhedron.tetras) else self.mesh.face_list[face].n2
                if not ccw_check(self.mesh.face_list[face], self.mesh.tetra_list[t],self.mesh.node_list):
                    # print('check face', face)
                    v2 = self.mesh.face_list[face].v2
                    v3 = self.mesh.face_list[face].v3
                    self.mesh.face_list[face].v2 = v3
                    self.mesh.face_list[face].v3 = v2
                list_face.append(face)
        list_face =  list(dict.fromkeys(list_face))
        
        with open(filename, 'w') as fh:
            fh.write("OFF\n")
            fh.write("%d %d 0\n" % (self.mesh.n_nodes, len(list_face)))
            for v in self.mesh.node_list:
                fh.write("%f %f %f\n" % ((v.x - centroid_x)*100, (v.y - centroid_y)*100, (v.z - centroid_z)*100))
            for f in list_face:
                v1 = self.mesh.face_list[f].v1
                v2 = self.mesh.face_list[f].v2
                v3 = self.mesh.face_list[f].v3
                fh.write("3 %d %d %d\n" % (v1, v2, v3))

    def printVISF_polyhedralmesh(self, filename):
        print("writing VISF file: " + filename)
        list_face = []

        # Center the mesh around the centroid
        centroid_x = sum(v.x for v in self.mesh.node_list) / len(self.mesh.node_list)
        centroid_y = sum(v.y for v in self.mesh.node_list) / len(self.mesh.node_list)
        centroid_z = sum(v.z for v in self.mesh.node_list) / len(self.mesh.node_list)

        for polyhedron in self.polyhedral_mesh:
            for face in polyhedron.faces:
                t = self.mesh.face_list[face].n1 if (self.mesh.face_list[face].n1 in polyhedron.tetras) else self.mesh.face_list[face].n2
                if not ccw_check(self.mesh.face_list[face], self.mesh.tetra_list[t], self.mesh.node_list):
                    v2 = self.mesh.face_list[face].v2
                    v3 = self.mesh.face_list[face].v3
                    self.mesh.face_list[face].v2 = v3
                    self.mesh.face_list[face].v3 = v2
                list_face.append(face)

        list_face = list(dict.fromkeys(list_face))

        with open(filename, 'w') as fh:
            fh.write("2 2\n")
            fh.write("%d\n" % (self.mesh.n_nodes))
            for v in self.mesh.node_list:
                fh.write("%f %f %f\n" % (
                    (v.x - centroid_x) * 100,
                    (v.y - centroid_y) * 100,
                    (v.z - centroid_z) * 100
                ))

            fh.write("%d\n" % (len(list_face)))
            for f in list_face:
                v1 = self.mesh.face_list[f].v1
                v2 = self.mesh.face_list[f].v2
                v3 = self.mesh.face_list[f].v3
                fh.write("3 %d %d %d\n" % (v1, v2, v3))

            fh.write("%d\n" % (len(self.polyhedral_mesh)))
            for poly in self.polyhedral_mesh:
                fh.write("%d" % (len(poly.faces)))
                for face in poly.faces:
                    fh.write(" %d" % (list_face.index(face)))
                fh.write("\n")

    # VISF for Camaron, which assigns each polygon to a single polyhedron: faces
    # shared by two polyhedra are written once per polyhedron, oriented outwards.
    # Only for visualization; the polyhedral mesh keeps shared faces.
    def printVISF_camaron(self, filename):
        print("writing VISF file (Camaron): " + filename)
        polygons = []
        poly_faces = []

        centroid_x = sum(v.x for v in self.mesh.node_list) / len(self.mesh.node_list)
        centroid_y = sum(v.y for v in self.mesh.node_list) / len(self.mesh.node_list)
        centroid_z = sum(v.z for v in self.mesh.node_list) / len(self.mesh.node_list)

        for polyhedron in self.polyhedral_mesh:
            tetras = set(polyhedron.tetras)
            indices = []
            for f in dict.fromkeys(polyhedron.faces):
                face = self.mesh.face_list[f]
                t = face.n1 if face.n1 in tetras else face.n2
                if ccw_check(face, self.mesh.tetra_list[t], self.mesh.node_list):
                    polygons.append((face.v1, face.v2, face.v3))
                else:
                    polygons.append((face.v1, face.v3, face.v2))
                indices.append(len(polygons) - 1)
            poly_faces.append(indices)

        with open(filename, 'w') as fh:
            fh.write("2 2\n")
            fh.write("%d\n" % (self.mesh.n_nodes))
            for v in self.mesh.node_list:
                fh.write("%f %f %f\n" % (
                    (v.x - centroid_x) * 100,
                    (v.y - centroid_y) * 100,
                    (v.z - centroid_z) * 100
                ))
            fh.write("%d\n" % (len(polygons)))
            for v1, v2, v3 in polygons:
                fh.write("3 %d %d %d\n" % (v1, v2, v3))
            fh.write("%d\n" % (len(poly_faces)))
            for indices in poly_faces:
                fh.write("%d %s\n" % (len(indices), " ".join(map(str, indices))))

    def printOFF_each_poly(self,filename):
        # print("writing OFF files: "+ filename)
        i = 0
        for polyhedron in self.polyhedral_mesh:
            list_face = polyhedron.faces
            nodes = []
            with open(filename+str(i)+'.off', 'w') as fh:
                fh.write("OFF\n")
                for tetra in polyhedron.tetras:
                    vertex = [self.mesh.tetra_list[tetra].v1, self.mesh.tetra_list[tetra].v2,self.mesh.tetra_list[tetra].v3,self.mesh.tetra_list[tetra].v4]
                    for v in vertex:
                        if v not in nodes :
                            nodes.append(v) 
                fh.write("%d %d 0\n" % (len(nodes), len(list_face)))
                for node in nodes:
                    v = self.mesh.node_list[node]
                    fh.write("%f %f %f\n" % (v.x, v.y, v.z))
                for f in list_face:
                    t = self.mesh.face_list[f].n1 if (self.mesh.face_list[f].n1 in polyhedron.tetras) else self.mesh.face_list[f].n2
                    if ccw_check(self.mesh.face_list[f], self.mesh.tetra_list[t],self.mesh.node_list):
                        v1 = nodes.index(self.mesh.face_list[f].v1)
                        v2 = nodes.index(self.mesh.face_list[f].v2)
                        v3 = nodes.index(self.mesh.face_list[f].v3)
                    else:
                        v1 = nodes.index(self.mesh.face_list[f].v1)
                        v2 = nodes.index(self.mesh.face_list[f].v3)
                        v3 = nodes.index(self.mesh.face_list[f].v2)
                    fh.write("3 %d %d %d # %d\n" % (v1, v2, v3,f))
            i+=1

    def printOFF_one_poly(self,index,filename):
        # print("writing OFF files: "+ filename)
        polyhedron = self.polyhedral_mesh[index]
        list_face = polyhedron.faces
        nodes = []
        with open(filename+'.off', 'w') as fh:
            fh.write("OFF\n")
            for tetra in polyhedron.tetras:
                vertex = [self.mesh.tetra_list[tetra].v1, self.mesh.tetra_list[tetra].v2,self.mesh.tetra_list[tetra].v3,self.mesh.tetra_list[tetra].v4]
                for v in vertex:
                    if v not in nodes :
                        nodes.append(v) 
            fh.write("%d %d 0\n" % (len(nodes), len(list_face)))
            for node in nodes:
                v = self.mesh.node_list[node]
                fh.write("%f %f %f\n" % (v.x, v.y, v.z))
            for f in list_face:
                t = self.mesh.face_list[f].n1 if (self.mesh.face_list[f].n1 in polyhedron.tetras) else self.mesh.face_list[f].n2
                if ccw_check(self.mesh.face_list[f], self.mesh.tetra_list[t],self.mesh.node_list):
                    v1 = nodes.index(self.mesh.face_list[f].v1)
                    v2 = nodes.index(self.mesh.face_list[f].v2)
                    v3 = nodes.index(self.mesh.face_list[f].v3)
                else:
                    v1 = nodes.index(self.mesh.face_list[f].v1)
                    v2 = nodes.index(self.mesh.face_list[f].v3)
                    v3 = nodes.index(self.mesh.face_list[f].v2)
                fh.write("3 %d %d %d # %d\n" % (v1, v2, v3,f))
