import trimesh
import numpy as np

cases = ['STS_006', 'STS_035', 'STS_043', 'STS_051']
bones = ['femur', 'tibia', 'patella']

print("Mesh Topology Check (Decimated Meshes)")
print(f"{'Case':<10} | {'Bone':<8} | {'Watertight':<10} | {'Bound. Edges':<12} | {'Non-Manifold Edges':<20}")
print("-" * 75)

for case in cases:
    for bone in bones:
        path = f"data/ct_knee/{case}_meshes/{bone}_decimated.obj"
        m = trimesh.load(path)
        
        edges = m.edges_sorted
        unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
        
        boundary_edges = np.sum(counts == 1)
        non_manifold_edges = np.sum(counts > 2)
        
        print(f"{case:<10} | {bone:<8} | {str(m.is_watertight):<10} | {boundary_edges:<12} | {non_manifold_edges:<20}")
