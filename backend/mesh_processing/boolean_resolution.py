import vtk
import pyvista as pv
import numpy as np
import trimesh

def check_self_intersection(mesh: pv.DataSet) -> bool:
    """
    Detects true geometric self-intersections (face penetrations)
    using pymeshlab. Correctly ignores topological adjacencies (shared edges/vertices).
    """
    import pymeshlab
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
    if mesh.n_cells == 0:
        return False
        
    mesh = mesh.triangulate()
    
    # Extract vertices and pure triangle indices, cast to correct types for pybind11
    v = mesh.points.astype(np.float64)
    f = mesh.faces.reshape(-1, 4)[:, 1:].astype(np.int32)
    
    # Load into pymeshlab
    ms = pymeshlab.MeshSet()
    m = pymeshlab.Mesh(vertex_matrix=v, face_matrix=f)
    ms.add_mesh(m)
    
    try:
        ms.compute_selection_by_self_intersections_per_face()
        selected_faces = ms.current_mesh().face_selection_array()
        return np.any(selected_faces)
    except Exception as e:
        print(f"Warning: pymeshlab self-intersection check failed: {e}")
        # Fail closed on a clinical gate error
        return True

from typing import Tuple

def resolve_bone_cartilage_boundary(bone_mesh: pv.PolyData, cartilage_mesh: pv.PolyData) -> Tuple[pv.PolyData, str]:
    """
    Trims the bone mesh at the cartilage interface using boolean difference.
    If the operation produces non-manifold geometry, self-intersections, 
    or violates volume sanity bounds, it falls back to a vertex-projection approach.
    """
    bone_b = bone_mesh.bounds
    cart_b = cartilage_mesh.bounds
    
    # 0. Check for disjoint bounding boxes
    if (bone_b[0] > cart_b[1] or bone_b[1] < cart_b[0] or
        bone_b[2] > cart_b[3] or bone_b[3] < cart_b[2] or
        bone_b[4] > cart_b[5] or bone_b[5] < cart_b[4]):
        # No overlap at all, return bone_mesh as is
        return bone_mesh.copy(), "No Overlap"
        
    overlap_bounds = (
        max(bone_b[0], cart_b[0]), min(bone_b[1], cart_b[1]),
        max(bone_b[2], cart_b[2]), min(bone_b[3], cart_b[3]),
        max(bone_b[4], cart_b[4]), min(bone_b[5], cart_b[5])
    )
    
    pad = 5.0
    padded_bounds = (
        overlap_bounds[0]-pad, overlap_bounds[1]+pad, 
        overlap_bounds[2]-pad, overlap_bounds[3]+pad, 
        overlap_bounds[4]-pad, overlap_bounds[5]+pad
    )

    # Calculate expected overlap volume via VTK_INTERSECTION
    overlap_filter = vtk.vtkBooleanOperationPolyDataFilter()
    overlap_filter.SetOperationToIntersection()
    overlap_filter.SetInputData(0, bone_mesh)
    overlap_filter.SetInputData(1, cartilage_mesh)
    overlap_filter.Update()
    overlap_mesh = pv.wrap(overlap_filter.GetOutput())
    
    v_overlap = overlap_mesh.volume if overlap_mesh.n_cells > 0 else 0.0
    v_bone = bone_mesh.volume
    
    # Perform difference (trim bone using cartilage)
    bool_filter = vtk.vtkBooleanOperationPolyDataFilter()
    bool_filter.SetOperationToDifference()
    bool_filter.SetInputData(0, bone_mesh)
    bool_filter.SetInputData(1, cartilage_mesh)
    bool_filter.Update()
    
    result_mesh = pv.wrap(bool_filter.GetOutput())
    
    # Validation helper
    def validate_mesh(mesh_to_check: pv.PolyData, context: str = "", max_bad_edges: int = 0) -> bool:
        if mesh_to_check.n_points == 0:
            print(f"[{context}] Validation failed: mesh is empty.")
            return False
        # 1. Manifold Check
        edges = mesh_to_check.extract_feature_edges(boundary_edges=True, non_manifold_edges=True, feature_edges=False, manifold_edges=False)
        if edges.n_cells > max_bad_edges:
            print(f"[{context}] Validation failed: mesh has {edges.n_cells} non-manifold or boundary edges (max allowed: {max_bad_edges}).")
            return False
        elif edges.n_cells > 0:
            print(f"[{context}] WARNING: mesh has {edges.n_cells} non-manifold or boundary edges, but keeping as it is below threshold ({max_bad_edges}).")
            
        # 2. Localized Self-Intersection Check
        try:
            clipped_result = mesh_to_check.clip_box(padded_bounds, invert=False)
        except Exception as e:
            print(f"[{context}] Validation failed: clip_box raised {e}")
            return False
            
        if clipped_result.n_cells > 0:
            if check_self_intersection(clipped_result):
                print(f"[{context}] Validation failed: self-intersection detected.")
                return False
                
        # 3. Volume Sanity Bound
        v_result = mesh_to_check.volume
        expected_v = v_bone - v_overlap
        tolerance = max(1.0, 0.05 * v_overlap)
        if abs(v_result - expected_v) > tolerance:
            print(f"[{context}] Validation failed: volume sanity bound exceeded (diff={abs(v_result - expected_v):.3f}, tol={tolerance:.3f}).")
            return False
            
        # 4. Gap Check
        # Isolate the interface points using a truly tight bound (0.1mm) to prevent 
        # testing far-side bone geometry that falls into the 5.0mm padded_bounds.
        tight_pad = 0.1
        tight_bounds = (
            overlap_bounds[0]-tight_pad, overlap_bounds[1]+tight_pad,
            overlap_bounds[2]-tight_pad, overlap_bounds[3]+tight_pad,
            overlap_bounds[4]-tight_pad, overlap_bounds[5]+tight_pad
        )
        interface_region = mesh_to_check.clip_box(tight_bounds, invert=False)
        if interface_region.n_points > 0:
            dist_filter = vtk.vtkImplicitPolyDataDistance()
            dist_filter.SetInput(cartilage_mesh)
            
            pts = np.array(interface_region.points)
            distances = np.array([dist_filter.EvaluateFunction(p) for p in pts])
            actual_gap = np.max(distances)
            
            print(f"[{context}] Max interface gap: {actual_gap:.3f} mm")
            
            GAP_TOLERANCE = 2.0 # mm
            if actual_gap > GAP_TOLERANCE:
                print(f"[{context}] Gap check failed: actual_gap ({actual_gap:.3f} mm) exceeds tolerance.")
                return False
                
        return True

    # Check the boolean result
    if validate_mesh(result_mesh, "Boolean"):
        return result_mesh, "Boolean"
        
    # If boolean fails, trigger vertex-projection fallback
    fallback_mesh = vertex_projection_fallback(bone_mesh, cartilage_mesh, padded_bounds)
    
    # Final check on the projected mesh, allowing up to 50 bad edges
    if validate_mesh(fallback_mesh, "Fallback", max_bad_edges=50):
        print("Vertex projection fallback succeeded!")
        return fallback_mesh, "Fallback"
    else:
        raise RuntimeError("Boundary resolution failed: both boolean and vertex-projection fallbacks produced invalid meshes.")

def vertex_projection_fallback(bone_mesh: pv.PolyData, cartilage_mesh: pv.PolyData, overlap_bounds: tuple) -> pv.PolyData:
    """
    Fallback to purely localized vertex projection (shrink-wrap) to resolve overlap.
    Moves bone vertices that are inside the cartilage out to the surface along the SDF gradient.
    Capped by average edge length to prevent inverted triangles.
    """
    result = bone_mesh.copy()
    
    # Calculate per-vertex 1-ring local average edge length for the displacement cap
    edges = result.extract_all_edges()
    lines = edges.lines.reshape(-1, 3)[:, 1:]
    pts = edges.points
    edge_lens = np.linalg.norm(pts[lines[:, 0]] - pts[lines[:, 1]], axis=1)
    
    vertex_edge_sum = np.zeros(len(result.points))
    vertex_edge_count = np.zeros(len(result.points))
    np.add.at(vertex_edge_sum, lines[:, 0], edge_lens)
    np.add.at(vertex_edge_sum, lines[:, 1], edge_lens)
    np.add.at(vertex_edge_count, lines[:, 0], 1)
    np.add.at(vertex_edge_count, lines[:, 1], 1)
    
    # Per-vertex displacement cap (fallback to 2.0mm if isolated vertex)
    local_edge_caps = np.where(vertex_edge_count > 0, vertex_edge_sum / np.maximum(1, vertex_edge_count) * 2.5, 2.0)
    
    # Create distance filter
    distance_filter = vtk.vtkImplicitPolyDataDistance()
    distance_filter.SetInput(cartilage_mesh)
    
    points = np.array(result.points)
    
    print(f"Vertex projection started for {len(points)} points...")
    inside_count = 0
    fallback_voxelization_triggered = False
    
    # We only process points within the bounding box
    for i in range(len(points)):
        p = points[i]
        
        # Check if point is inside overlap bounding box
        if (p[0] < overlap_bounds[0] or p[0] > overlap_bounds[1] or
            p[1] < overlap_bounds[2] or p[1] > overlap_bounds[3] or
            p[2] < overlap_bounds[4] or p[2] > overlap_bounds[5]):
            continue
            
        dist = distance_filter.EvaluateFunction(p)
        if dist < 0:
            inside_count += 1
            displacement = abs(dist)
            displacement_cap = local_edge_caps[i]
            if displacement > displacement_cap:
                # Clamp displacement to the local cap to prevent inverted triangles, 
                # instead of aborting the entire projection.
                displacement = displacement_cap
                
            grad = [0.0, 0.0, 0.0]
            distance_filter.EvaluateGradient(p, grad)
            
            # Project vertex exactly to surface
            points[i] = p + np.array(grad) * displacement
            
    print(f"Vertex projection finished. Projected {inside_count} vertices.")
    result.points = points
    
    # Clean and fill any micro-holes created by vertex shifting
    result = result.clean()
    if result.n_open_edges > 0:
        result = result.fill_holes(100).clean()
        
    # Apply a light, non-shrinking smoothing pass to resolve micro self-intersections 
    # created by varying per-vertex displacements.
    result = result.smooth_taubin(n_iter=10, pass_band=0.1)
        
    return result


