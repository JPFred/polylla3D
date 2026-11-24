# Quality metrics for a polyhedron given as a list of tetrahedra of the mesh.
# They take tetrahedra instead of a Polyhedron object, so a candidate
# polyhedron can be evaluated before it is built (merge criteria).

from collections import Counter

import numpy as np
from scipy.spatial import ConvexHull


def vertex_coords(mesh, v):
    node = mesh.node_list[v]
    return np.array([node.x, node.y, node.z])


# Area of face f, computed once and cached in face.area (-1 means not computed)
def face_area(mesh, f):
    face = mesh.face_list[f]
    if face.area < 0:
        a, b, c = (vertex_coords(mesh, v) for v in (face.v1, face.v2, face.v3))
        face.area = 0.5 * np.linalg.norm(np.cross(b - a, c - a))
    return face.area


# Shape quality of face f: 4*sqrt(3)*area / (sum of squared edge lengths).
# It is in (0, 1] and equals 1 only for an equilateral triangle.
def face_equilaterality(mesh, f):
    face = mesh.face_list[f]
    a, b, c = (vertex_coords(mesh, v) for v in (face.v1, face.v2, face.v3))
    squared = np.dot(b - a, b - a) + np.dot(c - b, c - b) + np.dot(a - c, a - c)
    return 4.0 * np.sqrt(3.0) * face_area(mesh, f) / squared


# Faces of the polyhedron: faces that belong to only one of its tetrahedra
def polyhedron_faces(mesh, tetras):
    counts = Counter(f for t in tetras for f in mesh.tetra_list[t].faces)
    return [f for f, c in counts.items() if c == 1]


# Ids of the vertices of the polyhedron
def polyhedron_vertices(mesh, tetras):
    return sorted({v for t in tetras for v in (mesh.tetra_list[t].v1, mesh.tetra_list[t].v2,
                                               mesh.tetra_list[t].v3, mesh.tetra_list[t].v4)})


# Convexity index: surface area of the polyhedron / surface area of its convex hull.
# It is 1 for a convex polyhedron, but unlike the 2D perimeter version it is not
# always > 1 for non convex ones: in 3D a non convex polyhedron can have less area
# than its convex hull. Kept to compare with Magdalena's "Area Ratio"; use
# solidity() to measure convexity.
def convexity_index(mesh, tetras):
    area = sum(face_area(mesh, f) for f in polyhedron_faces(mesh, tetras))
    points = np.array([vertex_coords(mesh, v) for v in polyhedron_vertices(mesh, tetras)])
    return area / ConvexHull(points).area


# Volume of tetrahedron t
def tetra_volume(mesh, t):
    tet = mesh.tetra_list[t]
    a, b, c, d = (vertex_coords(mesh, v) for v in (tet.v1, tet.v2, tet.v3, tet.v4))
    return abs(np.dot(b - a, np.cross(c - a, d - a))) / 6.0


def polyhedron_volume(mesh, tetras):
    return sum(tetra_volume(mesh, t) for t in tetras)


# Solidity: volume of the polyhedron / volume of its convex hull.
# It is in (0, 1] and equals 1 only for a convex polyhedron.
def solidity(mesh, tetras):
    points = np.array([vertex_coords(mesh, v) for v in polyhedron_vertices(mesh, tetras)])
    return polyhedron_volume(mesh, tetras) / ConvexHull(points).volume
