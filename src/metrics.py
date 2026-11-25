# Quality metrics for a polyhedron given as a list of tetrahedra of the mesh.
# They take tetrahedra instead of a Polyhedron object, so a candidate
# polyhedron can be evaluated before it is built (merge criteria).

from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection, QhullError


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


# Outward unit normal of each boundary face of the polyhedron (pointing away
# from the vertex of its tetrahedron opposite the face), with a point of the face
def boundary_normals(mesh, tetras):
    boundary = set(polyhedron_faces(mesh, tetras))
    normals = {}
    for t in tetras:
        tet = mesh.tetra_list[t]
        tet_vertices = {tet.v1, tet.v2, tet.v3, tet.v4}
        for f in tet.faces:
            if f not in boundary:
                continue
            face = mesh.face_list[f]
            a, b, c = (vertex_coords(mesh, v) for v in (face.v1, face.v2, face.v3))
            (opposite,) = tet_vertices - {face.v1, face.v2, face.v3}
            n = np.cross(b - a, c - a)
            n /= np.linalg.norm(n)
            if np.dot(n, vertex_coords(mesh, opposite) - a) > 0:
                n = -n
            normals[f] = (n, a)
    return normals


# Inequalities n·x + d <= 0 of the boundary faces of the polyhedron, one row
# [nx, ny, nz, d] per face, with n the outward unit normal
def polyhedron_halfspaces(mesh, tetras):
    return np.array([np.append(n, -np.dot(n, a)) for n, a in boundary_normals(mesh, tetras).values()])


# Kernel ratio: volume of the kernel (the points that see the whole polyhedron)
# / volume of the polyhedron. It is 1 for a convex polyhedron and 0 when the
# kernel is empty or flat (zero volume), so the share of polyhedra with
# ratio > 0 is the rate of kernels with positive volume.
def kernel_ratio(mesh, tetras, eps=1e-12):
    if solidity(mesh, tetras) >= 1 - 1e-9:
        return 1.0
    halfspaces = polyhedron_halfspaces(mesh, tetras)
    # Chebyshev center: the point inside the kernel farthest from every plane
    # (maximize r subject to n·x + d + r <= 0, the normals are unit vectors)
    res = linprog(c=[0, 0, 0, -1],
                  A_ub=np.column_stack((halfspaces[:, :3], np.ones(len(halfspaces)))),
                  b_ub=-halfspaces[:, 3], bounds=[(None, None)] * 3 + [(0, None)])
    if not res.success or res.x[3] <= eps:
        return 0.0
    try:
        kernel = HalfspaceIntersection(halfspaces, res.x[:3])
        return ConvexHull(kernel.intersections).volume / polyhedron_volume(mesh, tetras)
    except QhullError:
        # Degenerate (almost flat) kernel: its volume is negligible
        return 0.0


# Edge ratio: shortest / longest edge on the boundary of the polyhedron. An edge
# between two coplanar boundary faces is skipped: it only splits a flat polygon
# (common on the domain boundary and in grid meshes) and is not a real edge.
# It is in (0, 1]; values near 0 mean very short edges next to very long ones.
def edge_ratio(mesh, tetras, tol=1e-9):
    normals = boundary_normals(mesh, tetras)
    edge_faces = defaultdict(list)
    for f in normals:
        face = mesh.face_list[f]
        v = (face.v1, face.v2, face.v3)
        for i in range(3):
            edge_faces[tuple(sorted((v[i], v[(i + 1) % 3])))].append(f)
    lengths = []
    for (a, b), faces in edge_faces.items():
        if len(faces) == 2:
            n1, n2 = normals[faces[0]][0], normals[faces[1]][0]
            if np.dot(n1, n2) > 0 and np.linalg.norm(np.cross(n1, n2)) < tol:
                continue
        lengths.append(np.linalg.norm(vertex_coords(mesh, a) - vertex_coords(mesh, b)))
    return min(lengths) / max(lengths)
