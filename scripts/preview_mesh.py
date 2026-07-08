import pyvista as pv
import os

mesh_path = "data/ct_knee/case01_STS_006_meshes/case01_STS_006_bone_mask_ct_bone_final.obj"
output_image = "case01_STS_006_mesh_preview.png"

# Read mesh
mesh = pv.read(mesh_path)

# Set up plotter
plotter = pv.Plotter(off_screen=True)
plotter.add_mesh(mesh, color='lightblue', specular=0.5, ambient=0.2)
plotter.camera_position = 'xy'
plotter.show(screenshot=output_image)
print(f"Saved preview to {output_image}")
