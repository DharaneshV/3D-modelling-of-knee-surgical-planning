import vtk
import pyvista as pv
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

def resolve_bone_cartilage_boundary(bone_mesh: pv.PolyData, cartilage_mesh: pv.PolyData) -> Tuple[pv.PolyData, str]:
    """
    Trims the bone mesh at the cartilage interface using a simple boolean difference
    for visualization purposes.
    
    If the boolean operation fails or returns an empty mesh, it degrades gracefully
    by returning the unclipped original bone mesh, so the visualization scene 
    doesn't break or render a black hole.
    """
    bone_b = bone_mesh.bounds
    cart_b = cartilage_mesh.bounds
    
    # 0. Check for disjoint bounding boxes
    if (bone_b[0] > cart_b[1] or bone_b[1] < cart_b[0] or
        bone_b[2] > cart_b[3] or bone_b[3] < cart_b[2] or
        bone_b[4] > cart_b[5] or bone_b[5] < cart_b[4]):
        # No overlap at all, return bone_mesh as is
        return bone_mesh.copy(), "No Overlap"
        
    try:
        # Perform difference (trim bone using cartilage)
        bool_filter = vtk.vtkBooleanOperationPolyDataFilter()
        bool_filter.SetOperationToDifference()
        bool_filter.SetInputData(0, bone_mesh)
        bool_filter.SetInputData(1, cartilage_mesh)
        bool_filter.Update()
        
        result_mesh = pv.wrap(bool_filter.GetOutput())
        
        # Basic Sanity Floor: if empty, fall back
        if result_mesh.n_points == 0 or result_mesh.n_cells == 0:
            logger.warning("Visual boolean clipping produced an empty mesh. Falling back to unclipped bone.")
            return bone_mesh.copy(), "Fallback (Empty Clip)"
            
        # Cheap .clean() pass to merge duplicate points and remove degenerate cells
        result_mesh.clean(inplace=True)
        
        return result_mesh, "Boolean Difference Success"
        
    except Exception as e:
        logger.warning(f"Visual boolean clipping raised an exception: {e}. Falling back to unclipped bone.")
        return bone_mesh.copy(), f"Fallback (Exception: {e})"
