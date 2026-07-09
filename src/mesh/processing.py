import pyvista as pv

def apply_taubin_smoothing(mesh: pv.PolyData, n_iter: int = 20, pass_band: float = 0.1) -> pv.PolyData:
    """
    Apply Taubin smoothing to a PyVista mesh.
    Unlike Laplacian smoothing, Taubin smoothing preserves the volume of the geometry
    by alternating shrinking and expanding passes, effectively removing high-frequency
    stair-step artifacts without distorting the overall shape.
    
    Args:
        mesh: PyVista PolyData mesh.
        n_iter: Number of smoothing iterations.
        pass_band: Passband spatial frequency (lower value = more smoothing).
        
    Returns:
        Smoothed PyVista PolyData mesh.
    """
    return mesh.smooth_taubin(n_iter=n_iter, pass_band=pass_band)

def decimate_mesh(mesh: pv.PolyData, target_reduction: float = 0.5) -> pv.PolyData:
    """
    Reduce the polygon count of the mesh for real-time AR/VR rendering.
    Uses decimate_pro to preserve topological structure, which is critical
    for thin, complex anatomical structures like cartilage.
    
    Args:
        mesh: PyVista PolyData mesh.
        target_reduction: Target reduction fraction (e.g., 0.5 means 50% of triangles are removed).
        
    Returns:
        Decimated PyVista PolyData mesh.
    """
    # decimate_pro uses the vtkDecimatePro algorithm, which preserves topology 
    # and gives better control over sharp edges compared to standard decimate.
    # We turn off boundary_vertex_deletion to prevent holes from forming at the capped ends
    # and use a less aggressive default reduction (50% instead of 90%).
    # Ensure mesh is fully triangulated, as decimate_pro requires it
    mesh = mesh.triangulate()
    
    return mesh.decimate_pro(
        target_reduction, 
        feature_angle=60, 
        preserve_topology=True, 
        boundary_vertex_deletion=False
    )
