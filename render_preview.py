import pyvista as pv
import sys

mesh_path = "data/ct_knee/STS_051_meshes/femur_decimated.obj"
mesh = pv.read(mesh_path)

plotter = pv.Plotter(off_screen=True)
plotter.add_mesh(mesh, color='white', smooth_shading=True, specular=0.5)
plotter.camera_position = 'xy'
plotter.screenshot('STS_051_femur_0.5_decimated.png')
