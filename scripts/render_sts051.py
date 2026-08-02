import pyvista as pv

f = pv.read('data/ct_knee/STS_051_meshes/femur_left.obj')
t = pv.read('data/ct_knee/STS_051_meshes/tibia_left.obj')

col, n = f.collision(t)

# Create a plotter
p = pv.Plotter(off_screen=True)
p.add_mesh(f, color='tan', opacity=0.5, label='Femur')
p.add_mesh(t, color='lightblue', opacity=0.5, label='Tibia')

# Extract contact cells
if n > 0:
    contact_cells = col.field_data["ContactCells"]
    f_contact = col.extract_cells(contact_cells)
    p.add_mesh(f_contact, color='red', line_width=2, label='Intersections')

p.view_vector([1, -1, 0.5])
p.add_legend()
p.screenshot('scratch/sts051_intersection_view.png')
