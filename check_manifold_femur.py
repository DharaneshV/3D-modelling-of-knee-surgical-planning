import trimesh
import collections
import sys
mesh = trimesh.load(sys.argv[1], process=True)
print(f"Is watertight: {mesh.is_watertight}")
print(f"Is volume: {mesh.is_volume}")
edge_counts = collections.Counter([tuple(sorted(e)) for e in mesh.edges])
boundary = [e for e, c in edge_counts.items() if c == 1]
nonmanifold = [e for e, c in edge_counts.items() if c > 2]
print(f"Boundary edges (count=1): {len(boundary)}")
print(f"Non-manifold edges (count>2): {len(nonmanifold)}")
