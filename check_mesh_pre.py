import trimesh

cases = ['STS_006']
for case in cases:
    obj_path = f"data/ct_knee/{case}_meshes/case01_{case}_bone_mask_ct_bone_smoothed_pre_decimation.obj"
    mesh = trimesh.load(obj_path, process=False)
    components = mesh.split(only_watertight=False)
    components = sorted(components, key=lambda c: len(c.faces), reverse=True)
    
    print(f"\n[{case}] Pre-decimation mesh:")
    for i in range(min(3, len(components))):
        comp = components[i]
        print(f"  Major Component {i+1}: Watertight = {comp.is_watertight}, Faces={len(comp.faces)}")
