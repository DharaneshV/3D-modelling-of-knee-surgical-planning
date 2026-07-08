import trimesh
import sys

cases = ['STS_006', 'STS_035', 'STS_043', 'STS_051']
for case in cases:
    obj_path = f"data/ct_knee/{case}_meshes/case01_{case}_bone_mask_ct_bone_final.obj"
    mesh = trimesh.load(obj_path, process=False)
    # the multi-label mesh might be loaded as a Scene if it has multiple objects,
    # or a single Trimesh if it's merged.
    if isinstance(mesh, trimesh.Scene):
        meshes = list(mesh.geometry.values())
        print(f"[{case}] Loaded as Scene with {len(meshes)} geometries.")
        for name, geom in mesh.geometry.items():
            print(f"  {name}: Watertight = {geom.is_watertight}")
    else:
        # Check submeshes
        components = mesh.split(only_watertight=False)
        print(f"[{case}] Single mesh containing {len(components)} disconnected components.")
        for i, comp in enumerate(components):
            print(f"  Component {i+1}: Watertight = {comp.is_watertight}, Faces={len(comp.faces)}")
