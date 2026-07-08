import pyvista as pv
mesh = pv.read("data/ct_knee/STS_006_meshes/case01_STS_006_bone_mask_ct_bone_smoothed_pre_decimation.obj")
print(f"Number of points: {mesh.n_points}")
print(f"Number of cells: {mesh.n_cells}")
print(f"Is manifold: {mesh.is_manifold}")
print(f"Array names: {mesh.array_names}")
