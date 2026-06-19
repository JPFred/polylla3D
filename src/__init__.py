"""
Polylla3D - Polyhedral mesh generation and analysis library
"""

from .mesh import TetrahedronMesh, Polyhedron, Vertex, Face, Tetrahedron, Edge
from .newMesh import FaceTetrahedronMesh, EdgeTetrahedronMesh
from .PolyllaEdge import PolyllaEdge
from .PolyllaFace import PolyllaFace
from .utils import ccw_check, calculate_face_normal, is_polyhedron_convex

__all__ = [
    'TetrahedronMesh',
    'Polyhedron',
    'Vertex',
    'Face',
    'Tetrahedron',
    'Edge',
    'FaceTetrahedronMesh',
    'EdgeTetrahedronMesh',
    'PolyllaEdge',
    'PolyllaFace',
    'ccw_check',
    'calculate_face_normal',
    'is_polyhedron_convex'
]
