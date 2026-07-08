import trimesh
import os

cases = ['STS_006']
for case in cases:
    print(f"\n{case}:")
    for bone in ['femur', 'tibia']:
        print(f"  {bone}:")
        
        path_raw = f"data/ct_knee/{case}_meshes/{case}_ct_bone_raw.obj" # This is multi-label
        # Instead, check the smoothed ones if they exist, or extract again
        pass
        # I didn't save the _smoothed.obj in run_meshing.py!
