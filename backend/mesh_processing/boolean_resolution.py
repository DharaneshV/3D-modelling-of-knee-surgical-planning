# =============================================================================
# ARCHIVED — not called by the live pipeline as of PIPELINE_VERSION v3.2.
#
# This module boolean-clipped an independently synthesised bone mesh against the
# patient's cartilage so the two would not interpenetrate in the viewer. It only
# ever made sense while bone and cartilage came from *separate* sources: bone was
# a generic reference proxy fitted to cartilage by scale + rigid ICP, so nothing
# guaranteed a consistent boundary between them.
#
# It is no longer needed because:
#   - MRI bone is now meshed directly from CartiMorph's native labels 1 (femur)
#     and 3 (tibia), in the same unified vtkSurfaceNets3D pass as the cartilage.
#   - A single SurfaceNets pass over one label volume yields shared, topologically
#     consistent bone/cartilage boundaries by construction — there is no overlap
#     left to resolve, so there is nothing for a boolean difference to do.
#
# Its only remaining caller is src/synthesis/bone_from_mri.py, which is itself
# ARCHIVED under the same rationale. scripts/run_meshing.py explicitly dropped
# its import of this module (see the comment there).
#
# Its unit test, tests/test_boolean_resolution.py, was retired alongside this
# banner. That test targeted the pre-v3.2 contract, which differed on every
# point that mattered: it expected a bare PolyData return (this returns a
# (mesh, status) tuple), expected RuntimeError("Boundary resolution failed") on
# deep overlap (this never raises — it logs and degrades to the unclipped bone),
# and exercised a check_self_intersection() QA gate that no longer exists
# anywhere in the tree. The module was deliberately rewritten from a hard QA
# gate into a best-effort visual clip, so those assertions describe behaviour
# the project intentionally abandoned rather than behaviour that regressed.
#
# Do NOT re-import without re-evaluating the full rationale. If independent-mesh
# bone ever returns, restore the test to match whatever contract is chosen then —
# do not resurrect the old assertions verbatim.
# =============================================================================
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
