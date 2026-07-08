import pyvista as pv
mesh = pv.read("data/ct_knee/STS_006_meshes/case01_STS_006_ct_bone_raw.obj")
print(f"Array names: {mesh.array_names}")
