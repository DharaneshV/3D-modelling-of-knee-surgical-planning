import pyvista as pv

task_id = "a0810f6b-f014-44df-91ff-0428436c59e5"
base_dir = f"meshes/{task_id}"

f = pv.read(f"{base_dir}/femur_right.obj")
t = pv.read(f"{base_dir}/tibia_right.obj")

col, n = f.collision(t)

# ContactCells is a field array containing indices of intersecting cells
contact_indices = col.field_data["ContactCells"]
contact_cells = col.extract_cells(contact_indices)

# Save the intersecting part to view it
contact_cells.save("scratch/sts017_intersections.vtk")
print(f"Saved {n} intersecting faces to scratch/sts017_intersections.vtk")
