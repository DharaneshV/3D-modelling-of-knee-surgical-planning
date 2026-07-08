import trimesh
import os

cases = ['STS_006', 'STS_035']
for case in cases:
    print(f"\n{case}:")
    for bone in ['femur', 'tibia']:
        path = f"data/ct_knee/{case}_meshes/{bone}_decimated.obj"
        m = trimesh.load(path)
        print(f"  {bone} decimated: Watertight={m.is_watertight}")
