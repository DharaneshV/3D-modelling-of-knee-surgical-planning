import trimesh

case = 'STS_035'
for bone in ['femur', 'tibia', 'patella']:
    path = f"data/ct_knee/{case}_meshes/{bone}_decimated.obj"
    m = trimesh.load(path)
    
    print(f"\n{bone}:")
    print(f"  Watertight: {m.is_watertight}")
    
    # Boundary edges: edges shared by exactly 1 face
    # We can get this from m.edges_unique length vs faces
    # Trimesh provides m.is_watertight which checks if all edges are shared by exactly 2 faces.
    
    # Calculate boundary edges (degree 1)
    # m.edges_unique gives unique edges
    # m.edges gives all directed edges (3 per face)
    # A watertight mesh has exactly 2 directed edges per unique edge (opposite directions)
    edges = m.edges_sorted
    import numpy as np
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    
    boundary_edges = np.sum(counts == 1)
    non_manifold_edges = np.sum(counts > 2)
    
    print(f"  Boundary edges (open holes): {boundary_edges}")
    print(f"  Non-manifold edges: {non_manifold_edges}")
