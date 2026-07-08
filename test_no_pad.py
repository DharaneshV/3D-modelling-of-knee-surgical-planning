import os
import trimesh
import pyvista as pv
import numpy as np
import vtk

case = 'STS_006'

# generate_multilabel_mesh without padding
def generate_multilabel_mesh_no_pad(mask_path: str, label_map: dict) -> pv.PolyData:
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(mask_path)
    reader.Update()
    
    surfacenets = vtk.vtkSurfaceNets3D()
    surfacenets.SetInputConnection(reader.GetOutputPort())
    
    surfacenets.SetNumberOfLabels(len(label_map))
    for i, (label_name, label_value) in enumerate(label_map.items()):
        surfacenets.SetValue(i, label_value)
    
    surfacenets.Update()
    return pv.wrap(surfacenets.GetOutput())

label_map = {"femur": 1, "tibia": 2, "patella": 3}
mask_path = f"data/ct_knee/case01_{case}_bone_mask.nii.gz"
raw = generate_multilabel_mesh_no_pad(mask_path, label_map)

for label_name, label_id in label_map.items():
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
        
    print(f"{label_name} raw: {check_pv_watertight(surf)}")
