import pyvista as pv
mesh = pv.read("data/ct_knee/STS_006_meshes/case01_STS_006_bone_mask_ct_bone_final.obj")
print(f"Overall manifold: {mesh.is_manifold}")

for label_id in [1, 2, 3]:
    sub_mesh = mesh.extract_cells(mesh.cell_data['GroupIds'] == label_id)
    # extract_surface to clean up
    surf = sub_mesh.extract_surface()
    is_watertight = (surf.n_open_edges == 0)
    print(f"Label {label_id}: Watertight={is_watertight}, Open Edges={surf.n_open_edges}, Points={surf.n_points}, Cells={surf.n_cells}")
