import trimesh
mesh = trimesh.load("data/ct_knee/STS_006_meshes/case01_STS_006_bone_mask_ct_bone_smoothed_pre_decimation.obj", process=False)
comps = mesh.split(only_watertight=False)
comps = sorted(comps, key=lambda c: len(c.faces), reverse=True)
comp = comps[0]
print(f"Watertight: {comp.is_watertight}")
print(f"Is volume: {comp.is_volume}")
print(f"Euler number: {comp.euler_number}")
print(f"Edges without 2 faces: {len(comp.edges_unique) - len(comp.faces)*3//2}") 
