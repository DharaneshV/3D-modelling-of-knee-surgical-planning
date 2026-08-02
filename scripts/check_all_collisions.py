import os
import pyvista as pv
from pathlib import Path

mesh_dir = Path("meshes")
if not mesh_dir.exists():
    print("No meshes directory found.")
    exit(1)

for task_dir in mesh_dir.iterdir():
    if not task_dir.is_dir():
        continue
    
    print(f"\nTask: {task_dir.name}")
    for side in ["left", "right"]:
        femur_path = task_dir / f"femur_{side}.obj"
        tibia_path = task_dir / f"tibia_{side}.obj"
        
        if femur_path.exists() and tibia_path.exists():
            try:
                f = pv.read(femur_path)
                t = pv.read(tibia_path)
                col, n = f.collision(t)
                print(f"  [{side.upper()}] Intersecting faces: {n}")
            except Exception as e:
                print(f"  [{side.upper()}] Error calculating collision: {e}")
        else:
            print(f"  [{side.upper()}] Missing femur or tibia mesh.")
