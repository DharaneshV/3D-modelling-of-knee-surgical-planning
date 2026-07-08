import trimesh
import numpy as np
from scipy.spatial import cKDTree

cases = ['STS_006', 'STS_035', 'STS_043', 'STS_051']
for case in cases:
    femur = trimesh.load(f"data/ct_knee/{case}_meshes/femur_decimated.obj")
    tibia = trimesh.load(f"data/ct_knee/{case}_meshes/tibia_decimated.obj")

    # Joint line is approximately at the top of the Tibia
    joint_z = np.percentile(tibia.vertices[:, 2], 99) # 99th percentile to avoid stray top voxels
    
    # Constrain search region:
    # Femur distal: from joint_z - 20 to joint_z + 40
    # Tibia proximal: from joint_z - 40 to joint_z + 20
    f_mask = (femur.vertices[:, 2] >= joint_z - 20) & (femur.vertices[:, 2] <= joint_z + 40)
    t_mask = (tibia.vertices[:, 2] >= joint_z - 40) & (tibia.vertices[:, 2] <= joint_z + 20)
    
    f_pts = femur.vertices[f_mask]
    t_pts = tibia.vertices[t_mask]
    
    if len(f_pts) > 0 and len(t_pts) > 0:
        tree = cKDTree(t_pts)
        dists, indices = tree.query(f_pts)
        min_idx = np.argmin(dists)
        jsw = dists[min_idx]
        print(f"{case} JSW: {jsw:.2f} mm")
    else:
        print(f"{case} Error: No points in constrained region!")

