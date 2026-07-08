import trimesh

case = 'STS_006'
for bone in ['femur', 'tibia', 'patella']:
    path = f"data/ct_knee/{case}_meshes/{bone}_decimated.obj"
    m = trimesh.load(path)
    
    edges = m.edges_sorted
    import numpy as np
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    
    boundary_edges = np.sum(counts == 1)
    non_manifold_edges = np.sum(counts > 2)
    
    print(f"\n{bone}: Watertight={m.is_watertight}, Boundary edges={boundary_edges}, Non-manifold edges={non_manifold_edges}")
