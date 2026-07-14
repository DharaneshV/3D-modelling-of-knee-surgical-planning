import pyvista as pv

intersections = pv.read("scratch/sts017_intersections.vtk")
f = pv.read("meshes/a0810f6b-f014-44df-91ff-0428436c59e5/femur_right.obj")
t = pv.read("meshes/a0810f6b-f014-44df-91ff-0428436c59e5/tibia_right.obj")

p = pv.Plotter(off_screen=True)
p.add_mesh(f, color='tan', opacity=0.3)
p.add_mesh(t, color='green', opacity=0.3)
p.add_mesh(intersections, color='red')

p.camera_position = 'yz' # look from the side
p.screenshot("scratch/sts017_intersection_view.png")

p2 = pv.Plotter(off_screen=True)
p2.add_mesh(intersections, color='red')
p2.camera_position = 'yz'
p2.screenshot("scratch/sts017_intersection_only.png")

print("Saved scratch/sts017_intersection_view.png")
