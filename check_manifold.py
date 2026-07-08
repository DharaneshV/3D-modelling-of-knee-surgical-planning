import trimesh
mesh = trimesh.load("data/ct_knee/STS_006_meshes/case01_STS_006_bone_mask_ct_bone_smoothed_pre_decimation.obj", process=True)
comps = mesh.split(only_watertight=False)
comps = sorted(comps, key=lambda c: len(c.faces), reverse=True)
comp = comps[0]
print(f"Is watertight: {comp.is_watertight}")
print(f"Is volume: {comp.is_volume}")
# Watertight = len(edges) == len(edges_unique) * 2
# i.e., every edge shared by exactly 2 faces
import collections
edge_counts = collections.Counter([tuple(sorted(e)) for e in comp.edges])
boundary = [e for e, c in edge_counts.items() if c == 1]
nonmanifold = [e for e, c in edge_counts.items() if c > 2]
print(f"Boundary edges (count=1): {len(boundary)}")
print(f"Non-manifold edges (count>2): {len(nonmanifold)}")
