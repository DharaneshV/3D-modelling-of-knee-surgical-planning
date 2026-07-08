import trimesh
mesh = trimesh.load("femur_smoothed.obj", process=True)
comps = mesh.split(only_watertight=False)
comps = sorted(comps, key=lambda c: len(c.faces), reverse=True)
print(f"Number of connected components: {len(comps)}")
comp = comps[0]
print(f"Largest component watertight: {comp.is_watertight}")
