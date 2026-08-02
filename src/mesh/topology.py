"""
Topological repair for per-label surfaces extracted from a multi-label mesh.

vtkSurfaceNets3D emits one shared, non-intersecting surface for every pair of
adjacent labels. Slicing that surface down to a single label (as run_meshing.py
does) leaves two defects that the rest of the pipeline used to inherit:

1. Inconsistent winding. Each quad is oriented by its BoundaryLabels pair, so
   the femur's quads against background wind one way and its quads against
   femoral cartilage wind the other. Roughly a fifth of the edges of a bone
   surface disagree, which made trimesh report volumes 1.4x-2.2x the true value
   even for the meshes that happened to be closed.

2. Non-manifold pinch edges. Where the label mask contains a checkerboard voxel
   configuration - two voxels of a label meeting only along an edge, with the
   two complementary voxels doing the same - the dual mesh represents the
   junction as a single edge shared by four faces. That is what stopped
   trimesh calling the bone meshes watertight; it is a handful of edges out of
   ~33k, not a hole, so fill_holes could never fix it.

Both repairs are index-only: no vertex coordinate is ever created, moved or
removed, so the surface the clinical measurements are taken from is unchanged.
A vertex duplicated to unpick a pinch is a bit-exact copy of the original.
"""

from collections import defaultdict

import numpy as np


def _edge_faces(faces):
    """Map each undirected edge to the list of faces using it."""
    edge_faces = defaultdict(list)
    for face_index, (a, b, c) in enumerate(faces):
        for u, v in ((a, b), (b, c), (c, a)):
            edge_faces[(u, v) if u < v else (v, u)].append(face_index)
    return edge_faces


def _traversal(face, u, v):
    """+1 if `face` traverses the edge u->v, -1 if v->u, 0 if it uses neither."""
    for i in range(3):
        if face[i] == u and face[(i + 1) % 3] == v:
            return 1
        if face[i] == v and face[(i + 1) % 3] == u:
            return -1
    return 0


def _pair_around_edge(vertices, faces, u, v, incident):
    """
    Pair up the faces meeting at a non-manifold edge.

    The faces are sorted by their angle about the edge axis. On a consistently
    wound solid the traversal direction alternates as you go round, and each
    v->u face opens a wedge of material that closes at the next face
    counter-clockwise. Those two faces bound one sheet, so they are the pair
    that must keep sharing the edge.

    Returns [] if the fan does not alternate, which means the local
    configuration is not a clean pinch and is safer left alone.
    """
    origin = vertices[u]
    axis = vertices[v] - origin
    length = np.linalg.norm(axis)
    if length == 0:
        return []
    axis = axis / length

    reference = np.array([1.0, 0.0, 0.0])
    if abs(float(axis @ reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    basis_x = np.cross(axis, reference)
    basis_x /= np.linalg.norm(basis_x)
    basis_y = np.cross(axis, basis_x)

    fan = []
    for face_index in incident:
        opposite = [w for w in faces[face_index] if w not in (u, v)]
        if len(opposite) != 1:
            return []
        spoke = vertices[opposite[0]] - origin
        spoke = spoke - (spoke @ axis) * axis
        if np.linalg.norm(spoke) == 0:
            return []
        fan.append((float(np.arctan2(spoke @ basis_y, spoke @ basis_x)),
                    face_index,
                    _traversal(faces[face_index], u, v)))
    fan.sort()

    pairs = []
    for i, (_, face_index, direction) in enumerate(fan):
        if direction != -1:
            continue
        _, next_index, next_direction = fan[(i + 1) % len(fan)]
        if next_direction != 1:
            return []
        pairs.append((face_index, next_index))
    return pairs if len(pairs) * 2 == len(fan) else []


def split_non_manifold_edges(vertices, faces):
    """
    Duplicate vertices until no edge is shared by more than two faces.

    Expects consistently wound faces (run `trimesh.repair.fix_winding` first),
    because the pairing of faces around a pinch is decided from their traversal
    direction.

    Returns (vertices, faces, duplicated, unresolved) where `duplicated` counts
    the copied vertices and `unresolved` counts edges whose fan could not be
    interpreted and were left untouched.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64).copy()
    edge_faces = _edge_faces(faces)

    # Which faces must go on staying joined at each vertex: a manifold edge
    # joins its two faces, a pinch joins only the radially paired ones.
    joins = defaultdict(list)
    unresolved = 0
    for (u, v), incident in edge_faces.items():
        if len(incident) == 2:
            pairs = [(incident[0], incident[1])]
        elif len(incident) > 2 and len(incident) % 2 == 0:
            pairs = _pair_around_edge(vertices, faces, u, v, incident)
            if not pairs:
                unresolved += 1
        else:
            pairs = []
        for pair in pairs:
            joins[u].append(pair)
            joins[v].append(pair)

    vertex_faces = defaultdict(list)
    for face_index, face in enumerate(faces):
        for w in face:
            vertex_faces[w].append(face_index)

    duplicates = []
    for w, pairs in joins.items():
        incident = vertex_faces[w]
        if len(incident) < 2:
            continue

        parent = {f: f for f in incident}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for f, g in pairs:
            if f in parent and g in parent:
                root_f, root_g = find(f), find(g)
                if root_f != root_g:
                    parent[root_f] = root_g

        fans = defaultdict(list)
        for f in incident:
            fans[find(f)].append(f)
        if len(fans) <= 1:
            continue

        # The first fan keeps the original vertex; the rest get exact copies.
        for fan_index, fan in enumerate(fans.values()):
            if fan_index == 0:
                continue
            new_index = len(vertices) + len(duplicates)
            duplicates.append(vertices[w])
            for f in fan:
                faces[f][faces[f] == w] = new_index

    if duplicates:
        vertices = np.vstack([vertices, np.asarray(duplicates)])
    return vertices, faces, len(duplicates), unresolved


# A patch bigger than this is not a decimation artefact, it is a real hole, and
# capping it would move the surface a measurement could be taken from.
MAX_PATCH_AREA_MM2 = 1.0


def repair_label_surface(surface, close_holes=False):
    """
    Make one label's surface consistently wound, outward facing and manifold.

    Takes and returns a pyvista.PolyData. Vertex coordinates are untouched.

    `close_holes` additionally caps leftover boundary loops, for use after
    decimation, which occasionally tears a two-edge slit in an otherwise closed
    surface. The cap only reuses existing vertices, and is kept only if it both
    achieves watertightness and stays under MAX_PATCH_AREA_MM2, so a genuinely
    open surface (thin cartilage, say) is reported open rather than papered over.
    """
    import pyvista as pv
    import trimesh

    triangulated = surface.triangulate()
    vertices = np.asarray(triangulated.points)
    faces = triangulated.faces.reshape(-1, 4)[:, 1:]

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    trimesh.repair.fix_winding(mesh)

    vertices, faces, duplicated, unresolved = split_non_manifold_edges(
        mesh.vertices, mesh.faces)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    if close_holes and not mesh.is_watertight:
        patched = mesh.copy()
        try:
            trimesh.repair.fill_holes(patched)
        except Exception:
            patched = None
        if patched is not None and patched.is_watertight:
            patch_area = float(patched.area - mesh.area)
            if patch_area <= MAX_PATCH_AREA_MM2:
                mesh = patched

    # fix_winding settles on an orientation but not on which side is outside, and
    # it can settle on inward. trimesh.repair.fix_inversion declines to act on a
    # surface it does not consider closed, so decide from the signed volume
    # directly: with the winding now consistent its sign is the reliable test.
    if mesh.volume < 0:
        mesh.invert()

    repaired = pv.PolyData(
        np.asarray(mesh.vertices),
        np.hstack([np.full((len(mesh.faces), 1), 3), mesh.faces]).astype(np.int64))
    repaired.field_data['repair_duplicated_vertices'] = np.array([duplicated])
    repaired.field_data['repair_unresolved_edges'] = np.array([unresolved])
    return repaired
