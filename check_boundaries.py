import trimesh
import collections
import glob
import os

files = glob.glob("data/ct_knee/STS_*_meshes/*_decimated.obj")
for f in files:
    mesh = trimesh.load(f, process=True)
    edge_counts = collections.Counter([tuple(sorted(e)) for e in mesh.edges])
    boundary = [e for e, c in edge_counts.items() if c == 1]
    nonmanifold = [e for e, c in edge_counts.items() if c > 2]
    print(f"{os.path.basename(f):25s} | Boundary: {len(boundary):4d} | Non-Manifold: {len(nonmanifold):4d} | Verts: {len(mesh.vertices)}")
