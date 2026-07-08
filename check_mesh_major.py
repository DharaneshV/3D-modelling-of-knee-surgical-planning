import trimesh

cases = ['STS_006', 'STS_035', 'STS_043', 'STS_051']
for case in cases:
    obj_path = f"data/ct_knee/{case}_meshes/case01_{case}_bone_mask_ct_bone_final.obj"
    mesh = trimesh.load(obj_path, process=False)
    components = mesh.split(only_watertight=False)
    
    print(f"\n[{case}] Single mesh containing {len(components)} disconnected components.")
    # Sort by number of faces descending
    components = sorted(components, key=lambda c: len(c.faces), reverse=True)
    
    # Print the top 3 components (femur, tibia, patella)
    for i in range(min(3, len(components))):
        comp = components[i]
        print(f"  Major Component {i+1}: Watertight = {comp.is_watertight}, Faces={len(comp.faces)}")
