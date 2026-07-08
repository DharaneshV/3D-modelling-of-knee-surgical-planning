import os
import trimesh
import pyvista as pv
import numpy as np
import vtk

case = 'STS_035'

from src.mesh.surface_nets import generate_multilabel_mesh

label_map = {"femur": 1, "tibia": 2, "patella": 3}
mask_path = f"data/ct_knee/case01_{case}_bone_mask.nii.gz"
raw = generate_multilabel_mesh(mask_path, label_map)

for label_name, label_id in label_map.items():
    if label_name != 'femur': continue
    labels = raw.cell_data['BoundaryLabels']
    mask = (labels[:, 0] == label_id) | (labels[:, 1] == label_id)
    sub = raw.extract_cells(mask)
    surf = sub.extract_surface(algorithm='dataset_surface').clean()
    
    import tempfile
    
    def check_pv_watertight(m):
        with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f:
            m.save(f.name)
            tm = trimesh.load(f.name)
            res = tm.is_watertight
        os.unlink(f.name)
        return res
        
    print(f"{label_name} raw (WITH pad): {check_pv_watertight(surf)}")
