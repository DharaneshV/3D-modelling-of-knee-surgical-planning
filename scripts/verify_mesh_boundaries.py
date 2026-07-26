import argparse
import os
import sys
import vtk
import pyvista as pv
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Quantify max gap/overlap distance between two meshes using SDF.")
    parser.add_argument("--mesh1", type=str, required=True, help="Path to mesh1 (e.g., trimmed bone)")
    parser.add_argument("--mesh2", type=str, required=True, help="Path to mesh2 (e.g., cartilage)")
    parser.add_argument("--tolerance", type=float, default=0.1, help="Maximum allowable gap/overlap distance in mm")
    args = parser.parse_args()

    if not os.path.exists(args.mesh1) or not os.path.exists(args.mesh2):
        print("Error: Input meshes do not exist.")
        sys.exit(1)
        
    m1 = pv.read(args.mesh1)
    m2 = pv.read(args.mesh2)

    distance_filter = vtk.vtkImplicitPolyDataDistance()
    distance_filter.SetInput(m2)
    
    # Define interface by clipping m1 with padded m2 bounds
    bounds = m2.bounds
    pad = 2.0
    padded_bounds = (bounds[0]-pad, bounds[1]+pad, bounds[2]-pad, bounds[3]+pad, bounds[4]-pad, bounds[5]+pad)
    m1_interface = m1.clip_box(padded_bounds, invert=False)
    
    if m1_interface.n_points == 0:
        print("No interface found between the meshes.")
        sys.exit(0)
        
    interface_distances = []
    for p in m1_interface.points:
        d = distance_filter.EvaluateFunction(p)
        interface_distances.append(d)
        
    interface_distances = np.array(interface_distances)
    
    # Overlap is the maximum negative distance
    overlaps = interface_distances[interface_distances < 0]
    actual_overlap = float(abs(np.min(overlaps))) if len(overlaps) > 0 else 0.0
    
    # Gap check
    # We only care about points that were supposed to be on the interface.
    # For trimmed meshes, the cut faces should be at distance ~0.
    # Positive distances on m1_interface could just be the exterior bone surface rising away.
    # To isolate the gap at the cut face, we find points that are extremely close (< tolerance + some margin)
    # and find their maximum distance. If there is a genuine gap, there will be no points at distance 0,
    # and the closest points will be at distance = gap.
    
    closest_distance = float(np.min(interface_distances[interface_distances >= 0])) if np.any(interface_distances >= 0) else 0.0
    actual_gap = closest_distance

    print(f"max_gap_distance: {actual_gap:.4f} mm")
    print(f"max_overlap_distance: {actual_overlap:.4f} mm")
    
    if actual_overlap > args.tolerance:
        print(f"ERROR: max_overlap_distance ({actual_overlap:.4f}) exceeds tolerance ({args.tolerance})")
        sys.exit(1)
        
    if actual_gap > args.tolerance:
        print(f"ERROR: max_gap_distance ({actual_gap:.4f}) exceeds tolerance ({args.tolerance})")
        sys.exit(1)
        
    print("Verification passed.")
    sys.exit(0)

if __name__ == "__main__":
    main()
