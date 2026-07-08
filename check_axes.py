import trimesh
import numpy as np

cases = ['STS_006', 'STS_035', 'STS_043', 'STS_051']

for case in cases:
    print(f"\n--- {case} ---")
    femur = trimesh.load(f"data/ct_knee/{case}_meshes/femur_decimated.obj")
    tibia = trimesh.load(f"data/ct_knee/{case}_meshes/tibia_decimated.obj")
    
    fx = np.ptp(femur.vertices[:, 0])
    fy = np.ptp(femur.vertices[:, 1])
    fz = np.ptp(femur.vertices[:, 2])
    print(f"Femur Extents: X={fx:.1f}, Y={fy:.1f}, Z={fz:.1f}")
    
    tx = np.ptp(tibia.vertices[:, 0])
    ty = np.ptp(tibia.vertices[:, 1])
    tz = np.ptp(tibia.vertices[:, 2])
    print(f"Tibia Extents: X={tx:.1f}, Y={ty:.1f}, Z={tz:.1f}")

