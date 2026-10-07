##Notice: Polyhedrons are representd a list of faces

import sys
from pathlib import Path

# Allow imports to work both as a package and as a direct script
try:
    from .newMesh import FaceTetrahedronMesh, Polyhedron
    from .utils import ccw_check
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from newMesh import FaceTetrahedronMesh, Polyhedron
    from utils import ccw_check

import statistics
import numpy as np
import sys as sys2
from collections import Counter
from math import floor, sqrt
import random
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import tetgen


class PolyllaFace:
    def __init__(self, mesh, flag = 'are1'):
        self.mesh = mesh
        self.flag = flag
        self.n_barrier_faces = 0
        self.polyhedra_with_barriers = 0
        self.FLAGS = {
            'r' : self.calculate_max_incircle_faces,
            'R' : self.calculate_max_circumcircle_faces,
            'tra' : self.calculate_max_triangle_aspect_faces,
            'are1' : self.calculate_max_aspect_ratio_e1_faces,
            'are2' : self.calculate_max_aspect_ratio_e2_faces,
            # 'are3' : self.calculate_max_aspect_ratio_e3_faces,
            'a' : self.calculate_max_area_faces
        }
        #self.longest_faces = self.calculate_max_area_faces()
        if(flag!='r'):
            self.calculate_area_triangle_3d()
        self.longest_faces = self.FLAGS[flag]()
        self.seed_tetra = self.calculate_seed_tetrahedrons()
        self.bitvector_frontier_faces = self.calculate_frontier_faces()

        self.visited_tetra = [False] * mesh.n_tetrahedrons
        self.polyhedral_mesh = []
        for terminal_tetra in self.seed_tetra:
            polyhedron = []
            polyhedron_tetras = []
            self.DepthFirstSearch(polyhedron, polyhedron_tetras, terminal_tetra)
            #check if the polyhedron has barriers faces
            barrierFaces = self.count_barrierFaces(polyhedron)
            if barrierFaces > 0:
                self.repairPhase(polyhedron, polyhedron_tetras)
            else:
                poly = Polyhedron()
                poly.tetras = polyhedron_tetras.copy()
                poly.faces = polyhedron.copy()
                self.polyhedral_mesh.append(poly)

#############################################################################################
# FACE METRICS
#############################################################################################   
    def area(self,face):
        v1 = self.mesh.node_list[face.v1]
        v2 = self.mesh.node_list[face.v2]
        v3 = self.mesh.node_list[face.v3]
        
        #calculate the area of the triangle
        av1 = np.array([v1.x, v1.y, v1.z])
        av2 = np.array([v2.x, v2.y, v2.z])
        av3 = np.array([v3.x, v3.y, v3.z])
        area = np.linalg.norm(np.cross(av2-av1, av3-av1))
        face.area = area

    # For each tetrahedron, local index (0-3) of its face with the largest value.
    # Ties are broken by face id, so both tetrahedra sharing a face agree on the
    # order and no cycles without a terminal face can appear.
    def select_largest_faces(self, face_values):
        longest_faces = []
        for tetra in self.mesh.tetra_list:
            longest_faces.append(max(range(4), key=lambda k: (face_values[tetra.faces[k]], tetra.faces[k])))
        return longest_faces

    def calculate_max_triangle_aspect_faces(self):
        # self.calculate_edges_length()
        # self.calculate_area_triangle_3d()
        aspects = []
        for i in range(0, self.mesh.n_faces):
            length_edge_a = self.mesh.edge_list[self.mesh.face_list[i].edges[0]].length
            length_edge_b = self.mesh.edge_list[self.mesh.face_list[i].edges[1]].length
            length_edge_c = self.mesh.edge_list[self.mesh.face_list[i].edges[2]].length
            area = self.mesh.face_list[i].area

            L_max = max(length_edge_a,length_edge_b,length_edge_c)

            q = L_max*(length_edge_a + length_edge_b + length_edge_c) / (4 * sqrt(3) * area) # type: ignore
            aspects.append(area / q)
        return self.select_largest_faces(aspects)
    # aspect ratio for triangles from paper A Survey of Indicators for Mesh Quality Assessment
    def calculate_max_aspect_ratio_e1_faces(self):
        # self.calculate_edges_length()
        # self.calculate_area_triangle_3d()
        aspects = []
        for i in range(0, self.mesh.n_faces):
            length_edge_a = self.mesh.edge_list[self.mesh.face_list[i].edges[0]].length
            length_edge_b = self.mesh.edge_list[self.mesh.face_list[i].edges[1]].length
            length_edge_c = self.mesh.edge_list[self.mesh.face_list[i].edges[2]].length

            # radius ratio r/R (eq. 5.4 in the thesis gives r*R by mistake)
            area = self.mesh.face_list[i].area
            ar = 8 * area**2 / ((length_edge_a + length_edge_b + length_edge_c) * length_edge_a * length_edge_b * length_edge_c)
            aspects.append(self.mesh.face_list[i].area * ar)
        return self.select_largest_faces(aspects)
        
    def calculate_max_aspect_ratio_e2_faces(self):
        # self.calculate_edges_length()
        # self.calculate_area_triangle_3d()
        aspects = []
        for i in range(0, self.mesh.n_faces):
            length_edge_a = self.mesh.edge_list[self.mesh.face_list[i].edges[0]].length
            length_edge_b = self.mesh.edge_list[self.mesh.face_list[i].edges[1]].length
            length_edge_c = self.mesh.edge_list[self.mesh.face_list[i].edges[2]].length
            semiperimeter = (length_edge_a + length_edge_b + length_edge_c) / 2
            L_max = max(length_edge_a,length_edge_b,length_edge_c)

            # (s-a)(s-b)(s-c)/s is r^2
            radious = ((semiperimeter - length_edge_a) * (semiperimeter - length_edge_b) * (semiperimeter - length_edge_c) / semiperimeter) ** 0.5
            ar = radious / L_max

            aspects.append(self.mesh.face_list[i].area * ar)
        return self.select_largest_faces(aspects)
    
    def calculate_max_aspect_ratio_e3_faces(self):
        # self.calculate_edges_length()
        # self.calculate_area_triangle_3d()
        aspects = []
        for i in range(0, self.mesh.n_faces):
            length_edge_a = self.mesh.edge_list[self.mesh.face_list[i].edges[0]].length
            length_edge_b = self.mesh.edge_list[self.mesh.face_list[i].edges[1]].length
            length_edge_c = self.mesh.edge_list[self.mesh.face_list[i].edges[2]].length
            L_max = max(length_edge_a,length_edge_b,length_edge_c)

            Radious = (length_edge_a * length_edge_b * length_edge_c) / (4 * self.mesh.face_list[i].area) # type: ignore
            ar =  L_max / Radious

            aspects.append(ar * self.mesh.face_list[i].area)
        return self.select_largest_faces(aspects)
    
    def calculate_max_circumcircle_faces(self):
        # self.calculate_edges_length()
        aspects = []
        for i in range(0, self.mesh.n_faces):
            length_edge_a = self.mesh.edge_list[self.mesh.face_list[i].edges[0]].length
            length_edge_b = self.mesh.edge_list[self.mesh.face_list[i].edges[1]].length
            length_edge_c = self.mesh.edge_list[self.mesh.face_list[i].edges[2]].length

            Radious = (length_edge_a * length_edge_b * length_edge_c) / (4 * self.mesh.face_list[i].area) # type: ignore

            aspects.append(Radious)
        return self.select_largest_faces(aspects)


#############################################################################################   
# LABEL PHASE
#############################################################################################

    #Calculate length of each edge
    def calculate_edges_length(self):
        for edge in self.mesh.edge_list:
            v1 = self.mesh.node_list[edge.v1]
            v2 = self.mesh.node_list[edge.v2]
            distance = (v1.x - v2.x)**2 + (v1.y - v2.y)**2 + (v1.z - v2.z)**2 #without sqrt for performance
            if(distance < 0):
                print('-1')
            edge.length = distance

    ## Create a list with the index of the face of each treehedral that have the longest incircle radius
    def calculate_max_incircle_faces(self):
        #Caclulate the length of each edge
        # self.calculate_edges_length()

        # Calculate the incircle radius of each face
        face_radious = []
        for i in range(0, self.mesh.n_faces):
            length_edge_a = self.mesh.edge_list[self.mesh.face_list[i].edges[0]].length
            length_edge_b = self.mesh.edge_list[self.mesh.face_list[i].edges[1]].length
            length_edge_c = self.mesh.edge_list[self.mesh.face_list[i].edges[2]].length
            semiperimeter = (length_edge_a + length_edge_b + length_edge_c) / 2
            #we avoid calculate the square root to opt the code
            radious = (semiperimeter - length_edge_a) * (semiperimeter - length_edge_b) * (semiperimeter - length_edge_c) / semiperimeter
            face_radious.append(radious)

        #compare the radious of each face of all tetrahedros and return the index of the face with the longest radious
        return self.select_largest_faces(face_radious)


    def calculate_area_triangle_3d(self):
        for face in self.mesh.face_list:
            v1 = self.mesh.node_list[face.v1]
            v2 = self.mesh.node_list[face.v2]
            v3 = self.mesh.node_list[face.v3]
            
            #calculate the area of the triangle
            av1 = np.array([v1.x, v1.y, v1.z])
            av2 = np.array([v2.x, v2.y, v2.z])
            av3 = np.array([v3.x, v3.y, v3.z])
            area = np.linalg.norm(np.cross(av2-av1, av3-av1))
            # print(face.i, area)
            if(area < 0):
                print('-1')
            face.area = area*0.5

    def calculate_max_area_faces(self):
        return self.select_largest_faces([face.area for face in self.mesh.face_list])

    def calculate_seed_tetrahedrons(self):
        seed_tetra = []
        for f in range(0, self.mesh.n_faces):
            #Get two tetrahedrons adjacens to the face
            n1 = self.mesh.face_list[f].n1
            n2 = self.mesh.face_list[f].n2

            # Boundary face: seed if it is the largest face of its only tetrahedron
            if n1 == -1 or n2 == -1:
                t = n2 if n1 == -1 else n1
                if self.mesh.tetra_list[t].faces[self.longest_faces[t]] == f:
                    seed_tetra.append(t)
            # Terminal face: largest face of both tetrahedra
            else:
                longest_face_n1 = self.mesh.tetra_list[n1].faces[self.longest_faces[n1]]
                longest_face_n2 = self.mesh.tetra_list[n2].faces[self.longest_faces[n2]]
                if f == longest_face_n1 and f == longest_face_n2:
                    seed_tetra.append(n1)
        # print(seed_tetra)
        return seed_tetra


    # Retorna un bitvector de largo n_faces que indica si la cara es frontier-face o no
    def calculate_frontier_faces(self):
        frontier_faces = []
        for f in range(0, self.mesh.n_faces):
            n1 = self.mesh.face_list[f].n1
            n2 = self.mesh.face_list[f].n2

            # Si la cara es de borde, es una frontier-edge
            if n1 == -1 or n2 == -1:
                frontier_faces.append(True) # True para imprimir caras de borde
            else: 
                longest_face_n1 = self.mesh.tetra_list[n1].faces[self.longest_faces[n1]]
                longest_face_n2 = self.mesh.tetra_list[n2].faces[self.longest_faces[n2]]

                # Si no es la cara más larga de ningún tetra de n1 o n2, es una frontier-face
                frontier_faces.append(f != longest_face_n1 and f != longest_face_n2)
            
        return frontier_faces

##########################################################################################
#   TRAVEL PHASE
##########################################################################################


    # return list of faces 
    def DepthFirstSearch(self, polyhedron, polyhedron_tetras, tetra):
        self.visited_tetra[tetra] = True
        polyhedron_tetras.append(tetra)
        # not_fronteir_index = 0
        ## for each face of tetra
        for i in range(0, 4):
            face_id = self.mesh.tetra_list[tetra].faces[i]
            tetra_neighs = self.mesh.tetra_list[tetra].neighs
            if face_id != -1:
                #si la cara es un frontier-face, entonces no se sigue la recursión
                if self.bitvector_frontier_faces[face_id] == True:
                    polyhedron.append(face_id)
                else: #si es internal-face, se sigue la recursión por su tetra vecino
                    # print(i,tetra_neighs,self.mesh.tetra_list[tetra],'\n', self.mesh.face_list[face_id])
                    next_tetra = tetra_neighs[i]

                    if(self.visited_tetra[next_tetra] == False):
                        self.DepthFirstSearch(polyhedron, polyhedron_tetras, next_tetra)
        
############################################################################################################
# REPAIR PHASE
############################################################################################################


    def count_barrierFaces(self, polyhedron):
        repeated = [k for k, v in Counter(polyhedron).items() if v > 1]
        if(len(repeated) > 0):
            self.n_barrier_faces += len(repeated)
            self.polyhedra_with_barriers += 1
        return len(repeated)

    def detectBarrierFaceTips(self, terminalFace):
        # barrierFacesTips = []
        # #list of all reapeted faces
        # barrierFaces = [k for k, v in Counter(terminalFace).items() if v > 1]
        # facesNoBarrier = [k for k, v in Counter(terminalFace).items() if v == 1]
        # points_surph = []
        # for face in facesNoBarrier:
        #     points_surph.append(self.mesh.face_list[face].v1)
        #     points_surph.append(self.mesh.face_list[face].v2)
        #     points_surph.append(self.mesh.face_list[face].v3)
        
        # #List of all edges of the barrier faces
        # possibleTips = set()
        # tmpTermianlFace = list(set(terminalFace.copy()))
        # for face in barrierFaces:
        #     if((self.mesh.face_list[face].v1 in points_surph) and (self.mesh.face_list[face].v2 in points_surph) and (self.mesh.face_list[face].v3 in points_surph)):
                
        #         # print('No tips in face', face)
        #         if(self.mesh.face_list[face].n1 == -1 or self.mesh.face_list[face].n2 == -1):
        #             while terminalFace.count(face) > 1:
        #                 terminalFace.remove(face)
        #         else:
        #             while terminalFace.count(face) > 0:
        #                 terminalFace.remove(face)
        #     else:
        #         possibleTips.update(self.mesh.face_list[face].edges)
        # possibleTips = list(possibleTips)
        # for e in possibleTips:
        #     #List of all faces incident to e
        #     face_of_edge = self.mesh.edge_list[e].faces
        #     L1 = len(list(set(face_of_edge) & set(terminalFace)))
        #     L2 = len(terminalFace)
        #     if L2 - L1 == L2 - 1:
        #         barrierFacesTips.append(e)

        # return barrierFacesTips
        barrierFacesTips = []
        #list of all reapeted faces
        barrierFaces = [k for k, v in Counter(terminalFace).items() if v > 1]
        #List of all edges of the barrier faces
        possibleTips = set()
        for face in barrierFaces:
            possibleTips.update(self.mesh.face_list[face].edges)
        possibleTips = list(possibleTips)
        for e in possibleTips:
            #List of all faces incident to e
            face_of_edge = self.mesh.edge_list[e].faces
            L1 = len(list(set(face_of_edge) & set(terminalFace)))
            L2 = len(terminalFace)
            if L2 - L1 == L2 - 1:
                barrierFacesTips.append(e)
        return barrierFacesTips


    # Faces incident to edge e, in cyclic order around e, starting after start_face.
    # Returns None if the ring is open (e lies on the mesh boundary).
    def faces_around_edge(self, e, start_face):
        ring = []
        prev_face = start_face
        tetra = self.mesh.face_list[start_face].n1
        while True:
            if tetra == -1:
                return None
            # each tetrahedron has exactly two faces incident to e
            next_face = next(f for f in self.mesh.tetra_list[tetra].faces
                             if f != prev_face and e in self.mesh.face_list[f].edges)
            if next_face == start_face:
                return ring
            ring.append(next_face)
            face = self.mesh.face_list[next_face]
            tetra = face.n2 if face.n1 == tetra else face.n1
            prev_face = next_face

    # Algorithm 8 (thesis), plus: repeated repair of non-simple results (section 4.1.2)
    # and deletion of barrier faces when they can not be split (Algorithm 13).
    def repairPhase(self, polyhedron, polyhedron_tetras):
        seeds = []
        for e in self.detectBarrierFaceTips(polyhedron):
            # e is a tip, so exactly one face of the polyhedron contains it
            barrierFace = next(f for f in polyhedron if e in self.mesh.face_list[f].edges)
            internal_faces = self.faces_around_edge(e, barrierFace)
            # skip open rings and rings already split by another tip
            if not internal_faces or any(self.bitvector_frontier_faces[f] for f in internal_faces):
                continue
            middle_Face = internal_faces[(len(internal_faces) - 1) // 2]
            self.bitvector_frontier_faces[middle_Face] = True
            seeds.append(self.mesh.face_list[middle_Face].n1)
            seeds.append(self.mesh.face_list[middle_Face].n2)

        if not seeds:
            # the polyhedron can not be split: delete its barrier faces
            counts = Counter(polyhedron)
            poly = Polyhedron()
            poly.tetras = polyhedron_tetras.copy()
            poly.faces = [f for f in polyhedron if counts[f] == 1]
            poly.was_repaired = True
            self.polyhedral_mesh.append(poly)
            return

        for tetra in polyhedron_tetras:
            self.visited_tetra[tetra] = False
        # remaining tetrahedra are added as seeds so that none is left out
        for tetra in seeds + polyhedron_tetras:
            if self.visited_tetra[tetra]:
                continue
            new_polyhedron = []
            new_polyhedron_tetras = []
            self.DepthFirstSearch(new_polyhedron, new_polyhedron_tetras, tetra)
            if self.count_barrierFaces(new_polyhedron) > 0:
                self.repairPhase(new_polyhedron, new_polyhedron_tetras)
            else:
                poly = Polyhedron()
                poly.faces = new_polyhedron.copy()
                poly.tetras = new_polyhedron_tetras.copy()
                poly.was_repaired = True
                self.polyhedral_mesh.append(poly)

############################################################################################################
# EXTRA
############################################################################################################

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

    


    def writePolygonFile(self,filename):
        # print("writing OFF files: "+ filename)
        with open(filename+'.txt', 'w') as fh:
            polys = self.polyhedral_mesh
            
            c = 0
            not_convex_poly = []
            for poly in polys:
                if not poly.is_convex:
                    not_convex_poly.append(poly)
            p = len(not_convex_poly)
            fh.write(str(p)+'\n')
            for polyhedron in not_convex_poly:
                # print(str(polyhedron))
                list_face = polyhedron.faces
                nodes = []
                for tetra in polyhedron.tetras:
                    vertex = [self.mesh.tetra_list[tetra].v1, self.mesh.tetra_list[tetra].v2,self.mesh.tetra_list[tetra].v3,self.mesh.tetra_list[tetra].v4]
                    for v in vertex:
                        if v not in nodes :
                            nodes.append(v) 
                # if not polyhedron.is_convex:
                fh.write("%d %d\n" % (len(nodes), len(list_face)))
                for node in nodes:
                    v = self.mesh.node_list[node]
                    fh.write("%f %f %f\n" % (v.x, v.y, v.z))
                
                for f in list_face:
                    # print(list_face)
                    t = self.mesh.face_list[f].n1 if (self.mesh.face_list[f].n1 in polyhedron.tetras) else self.mesh.face_list[f].n2
                    if ccw_check(self.mesh.face_list[f], self.mesh.tetra_list[t],self.mesh.node_list):
                        v1 = nodes.index(self.mesh.face_list[f].v1)
                        v2 = nodes.index(self.mesh.face_list[f].v2)
                        v3 = nodes.index(self.mesh.face_list[f].v3)
                    else:
                        v1 = nodes.index(self.mesh.face_list[f].v1)
                        v2 = nodes.index(self.mesh.face_list[f].v3)
                        v3 = nodes.index(self.mesh.face_list[f].v2)
                    fh.write("3 %d %d %d \n" % (v1, v2, v3))# %d %d
                # else:
                #     print('si convex')
                c+=1


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

    def get_info(self):
        print("PolyllaFace info:")
        print("Number of polyhedrons: " + str(len(self.polyhedral_mesh)))
        print("Number of barrier faces: " + str(self.n_barrier_faces))
        print("Number of polyhedra with barrier faces: " + str(self.polyhedra_with_barriers))
        count = 0
        for polyhedron in self.polyhedral_mesh:
            if len(polyhedron.tetras) == 1:
                count += 1
        faces = set()
        edges = set()
        for poly in self.polyhedral_mesh:
            faces.update(poly.faces)
            for f in poly.faces:
                face = self.mesh.face_list[f]
                edges.update(face.edges)
        print("Number of polyhedrons that are tetrahedrons: " + str(count))
        print("Number of mesh faces: " , len(list(faces)))
        print("Numbre of mesh edges:", len(list(edges)))
        return len(self.polyhedral_mesh), self.n_barrier_faces, self.polyhedra_with_barriers, count, len(list(faces)), len(list(edges))

#############################################################################################
#POLYHEDRA METRICS
#############################################################################################
    # def edge_length(self, edge):
    #     v1 = self.mesh.node_list[edge.v1]
    #     v2 = self.mesh.node_list[edge.v2]
    #     distance = (v1.x - v2.x)**2 + (v1.y - v2.y)**2 + (v1.z - v2.z)**2 #without sqrt for performance
    #     edge.length = distance
    
    def edge_ratio(self):
        polyhedrons = self.polyhedral_mesh
        ratios = []
        # if self.mesh.edge_list[0].length < 0:
        #     self.calculate_edges_length()
        for poly in polyhedrons:
            # print(poly)
            faces = poly.faces
            # face_mins = []
            # face_maxs = []
            edges = []
            for face in faces:
                # edges = []
                for edge in self.mesh.face_list[face].edges:
                    # self.edge_length(self.mesh.edge_list[edge])
                    edges.append(self.mesh.edge_list[edge].length)
                # edge_min = min(edges)
                # edge_max = max(edges)
                # face_mins.append(edge_min)
                # face_maxs.append(edge_max)
            poly_min = min(edges)
            poly_max = max(edges)
            ratio = poly_min/poly_max
            ratios.append(ratio)
            # print(poly_min,poly_max)
        mean_ratio = statistics.mean(ratios)
        median_ratio = statistics.median(ratios)
        variance_ratio = statistics.variance(ratios)
        min_ratio = min(ratios)
        max_ratio = max(ratios)

        return [mean_ratio, min_ratio, max_ratio,median_ratio,variance_ratio]
    
    def tetra_per_poly(self):
        polyhedrons = self.polyhedral_mesh
        tetras = []
        for poly in polyhedrons:
            tetra_num = len(poly.tetras)
            tetras.append(tetra_num)
        
        mean_tetra_num = statistics.mean(tetras)
        median_tetra_num = statistics.median(tetras)
        variance_tetra_num = statistics.variance(tetras)
        min_tetra_num = min(tetras)
        max_tetra_num = max(tetras)
        i = tetras.index(max_tetra_num)
        # print(tetras)
        self.printOFF_one_poly(i,str(self.mesh.n_nodes)+self.flag)
        return [mean_tetra_num, min_tetra_num, max_tetra_num,median_tetra_num,variance_tetra_num]
    
    def faces_per_poly(self):
        polyhedrons = self.polyhedral_mesh
        faces_num = []
        for poly in polyhedrons:
            face_num = len(poly.faces)
            faces_num.append(face_num)
        
        mean_face_num = statistics.mean(faces_num)
        median_face_num = statistics.median(faces_num)
        variance_face_num = statistics.variance(faces_num)
        min_face_num = min(faces_num)
        max_face_num = max(faces_num)

        return [mean_face_num, min_face_num, max_face_num,median_face_num,variance_face_num]
    
    def convex_polyhedrons(self):
        conv_polys = 0
        for polyhedron in self.polyhedral_mesh:
            nodes = []
            for tetra in polyhedron.tetras:
                    vertex = [self.mesh.tetra_list[tetra].v1, self.mesh.tetra_list[tetra].v2,self.mesh.tetra_list[tetra].v3,self.mesh.tetra_list[tetra].v4]
                    for v in vertex:
                        if v not in nodes :
                            nodes.append(v)
            for i in range(len(nodes)):
                v = np.array([self.mesh.node_list[nodes[i]].x,self.mesh.node_list[nodes[i]].y,self.mesh.node_list[nodes[i]].z])
                nodes[i] = v
            nodes = np.array(nodes)
            cvhull = ConvexHull(nodes)
            if set(cvhull.vertices) == set(range(len(nodes))):
                conv_polys+=1
                polyhedron.is_convex = True

        return conv_polys/len(self.polyhedral_mesh)
    
    def polyhedron_area(self):
        ratios = []
        # if self.mesh.face_list[0].area < 0:
        #     self.calculate_area_triangle_3d()
        for polyhedron in self.polyhedral_mesh:
            suma_area = 0
            nodes = []
            for face in polyhedron.faces:
                # print(self.mesh.face_list[face].i, self.mesh.face_list[face].area)
                self.area(self.mesh.face_list[face])
                suma_area+= self.mesh.face_list[face].area
            for tetra in polyhedron.tetras:
                    vertex = [self.mesh.tetra_list[tetra].v1, self.mesh.tetra_list[tetra].v2,self.mesh.tetra_list[tetra].v3,self.mesh.tetra_list[tetra].v4]
                    for v in vertex:
                        if v not in nodes :
                            nodes.append(v)
            for i in range(len(nodes)):
                v = np.array([self.mesh.node_list[nodes[i]].x,self.mesh.node_list[nodes[i]].y,self.mesh.node_list[nodes[i]].z])
                nodes[i] = v
            nodes = np.array(nodes)
            cvhull = ConvexHull(nodes)
            cvHull_area = cvhull.area
            ratio = suma_area / cvHull_area
            ratios.append(ratio)
        mean_ratio_area = statistics.mean(ratios)
        median_ratio_area = statistics.median(ratios)
        variance_ratio_area = statistics.variance(ratios)
        min_ratio_area = min(ratios)
        max_ratio_area = max(ratios)

        return [mean_ratio_area, min_ratio_area, max_ratio_area,median_ratio_area,variance_ratio_area]
    
    def tetra_volume(self, tetra):
        t = self.mesh.tetra_list[tetra]
        v1 = self.mesh.node_list[t.v1]
        v2 = self.mesh.node_list[t.v2]
        v3 = self.mesh.node_list[t.v3]
        v4 = self.mesh.node_list[t.v4]
        A = np.array([v2.x - v1.x,v2.y - v1.y,v2.z - v1.z])
        B = np.array([v3.x - v1.x,v3.y - v1.y,v3.z - v1.z])
        C = np.array([v4.x - v1.x,v4.y - v1.y,v4.z - v1.z])

        # vec3d L0 = p1 - p0;
        # vec3d L2 = p0 - p2;
        # vec3d L3 = p3 - p0;

        # return (L2.cross(L0)).dot(L3) / 6.0;
        return np.dot(A,np.cross(B,C))/6.0
        

    def polyhedron_volume(self):
        volumes = []
        for poly in self.polyhedral_mesh:
            volume = 0
            for tetra in poly.tetras:
                volume += self.tetra_volume(tetra)
            volumes.append(volume)
        mean_volume = statistics.mean(volumes)
        min_volume = min(volumes)
        max_volume = max(volumes)

        return [mean_volume, min_volume, max_volume]
    
    def volume_ratio(self, kernelfile):
        kfile = open(kernelfile+'.txt','r')
        filelines = kfile.readlines()
        string_num_kernel = filelines.pop().split(' ')[0]
        kernel_volumes = list(map(float, filelines))
        print(float(string_num_kernel) < 1)
        if float(string_num_kernel) < 1:
            polys_w_kernel = len([kernel_volumes != 0])
        else :
            polys_w_kernel = int(string_num_kernel)
        
        # for line in filelines:
        #     kernel_volumes.append(float(line))
        ratios = []
        convex_poly = []
        # for poly in self.polyhedral_mesh:
        #     if poly.is_convex:
        #         convex_poly.append(poly)
        not_convexs = 0
        for i in range(len(self.polyhedral_mesh)):
            if(not_convexs >= len(kernel_volumes)):
                print("Bad kernel calculation")
                break
            poly = self.polyhedral_mesh[i]
            volume = 0
            for tetra in poly.tetras:
                volume += self.tetra_volume(tetra)
            if not poly.is_convex:
                if kernel_volumes[not_convexs] > 0:
                    ratio = kernel_volumes[not_convexs] / volume
                    ratios.append(ratio)
                not_convexs+=1
            else:
                ratio = 1
                ratios.append(ratio)
                # print(volume,kernel_volumes[i],ratio)
        mean_volume = statistics.mean(ratios)
        median_volume= statistics.median(ratios)
        variance_volume = statistics.variance(ratios)
        min_volume = min(ratios)
        max_volume = max(ratios)
        convex_num = len(self.polyhedral_mesh) - not_convexs
        kernel_rate = ((polys_w_kernel+convex_num)/len(self.polyhedral_mesh))*100
        
        return [mean_volume, min_volume, max_volume, kernel_rate,median_volume,variance_volume]
    
    def original_mesh_edge_ratio(self):
        # self.calculate_edges_length()
        ratios = []
        for tetra in self.mesh.tetra_list:
            edges = []
            for face in tetra.faces:
                for edge in self.mesh.face_list[face].edges:
                    edges.append(self.mesh.edge_list[edge].length)
            ratio = min(edges)/max(edges)
            ratios.append(ratio)
        mean_edge_ratio = statistics.mean(ratios)
        min_edge_ratio = min(ratios)
        max_edge_ratio = max(ratios)
        return [mean_edge_ratio, min_edge_ratio, max_edge_ratio]


if __name__ == "__main__":
    from pathlib import Path
    
    # Define the project root and data paths
    PROJECT_ROOT = Path(__file__).parent.parent  # Go up from src/ to project root
    input_folder = PROJECT_ROOT / "data" / "input"
    output_folder = PROJECT_ROOT / "data" / "output"
    
    # Create output folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # File configuration
    # file can be "1000points.1", "1000poisson.1", "1000random.1", "1000semiuniform.1", "1000uniform.1". 
    # Number of points can vary
    file = "1000uniform.1"
    node_file = input_folder / f"{file}.node"
    ele_file = input_folder / f"{file}.ele"
    face_file = input_folder / f"{file}.face"
    edge_file = input_folder / f"{file}.edge"
    
    print(f"Reading files from: {input_folder}")
    print(f"Writing files to: {output_folder}")
    
    mesh = FaceTetrahedronMesh(str(node_file), str(face_file), str(ele_file), str(edge_file))
    polylla_mesh = PolyllaFace(mesh)
    polylla_mesh.validate()

    # Write output files to data/output/
    polylla_mesh.printOFF_polyhedralmesh(str(output_folder / f"{file}_polyhedral_mesh.off"))
    polylla_mesh.printOFF_polyhedralmesh_colors(str(output_folder / f"{file}_polyhedral_mesh_colors.visf"))
    polylla_mesh.printVISF_polyhedralmesh(str(output_folder / f"{file}_polyhedral_mesh.visf"))
    polylla_mesh.printVISF_camaron(str(output_folder / f"{file}_polyhedral_mesh_camaron.visf"))

    polylla_mesh.get_info()
