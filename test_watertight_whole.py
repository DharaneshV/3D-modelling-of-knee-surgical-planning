import os
import trimesh
import pyvista as pv
import vtk

case = 'STS_035'

from src.mesh.surface_nets import generate_multilabel_mesh

label_map = {"femur": 1, "tibia": 2, "patella": 3}
mask_path = f"data/ct_knee/case01_{case}_bone_mask.nii.gz"
raw = generate_multilabel_mesh(mask_path, label_map)

# Let's save the whole multi-label mesh and see if it's watertight
import tempfile
with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f:
    raw.save(f.name)
    tm = trimesh.load(f.name)
    print(f"Whole mesh watertight: {tm.is_watertight}")
os.unlink(f.name)

# Now check femur by itself using PyVista extract_cells vs trimesh
labels = raw.cell_data['BoundaryLabels']
mask = (labels[:, 0] == 1) | (labels[:, 1] == 1)
sub = raw.extract_cells(mask)
surf = sub.extract_surface(algorithm='dataset_surface').clean()

with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f:
    surf.save(f.name)
    tm2 = trimesh.load(f.name)
    print(f"Femur extracted watertight: {tm2.is_watertight}")
os.unlink(f.name)
