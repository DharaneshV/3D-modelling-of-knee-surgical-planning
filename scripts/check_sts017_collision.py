import pyvista as pv

task_id = "a0810f6b-f014-44df-91ff-0428436c59e5"
base_dir = f"meshes/{task_id}"

for side in ["left", "right"]:
    femur_path = f"{base_dir}/femur_{side}.obj"
    tibia_path = f"{base_dir}/tibia_{side}.obj"
    
    try:
        f = pv.read(femur_path)
        t = pv.read(tibia_path)
        
        # Compute collision
        col, n = f.collision(t)
        
        print(f"[{side.upper()} KNEE]")
        print(f"  Contacts: {n} intersecting faces")
        print(f"  Array names: {col.array_names}")
        
        if "ContactCells" in col.array_names:
            contact_mask = col["ContactCells"] > 0
            contact_cells = col.extract_cells(contact_mask)
            if contact_cells.n_points > 0:
                bounds = contact_cells.bounds
                extent = f"X: {bounds[1]-bounds[0]:.1f}mm, Y: {bounds[3]-bounds[2]:.1f}mm, Z: {bounds[5]-bounds[4]:.1f}mm"
                print(f"  Extent: {extent}")
                
    except Exception as e:
        print(f"[{side.upper()} KNEE] Error: {e}")
