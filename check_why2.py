import trimesh
mesh = trimesh.load("data/ct_knee/STS_006_meshes/case01_STS_006_bone_mask_ct_bone_smoothed_pre_decimation.obj", process=True)
comps = mesh.split(only_watertight=False)
comps = sorted(comps, key=lambda c: len(c.faces), reverse=True)
comp = comps[0]
print(f"Watertight: {comp.is_watertight}")
print(f"Euler number: {comp.euler_number}")
