import sys
import os
import pyvista as pv
sys.path.insert(0, os.path.abspath('.'))
from src.mesh.surface_nets import extract_multilabel_surface

mask_path = 'meshes/a0810f6b-f014-44df-91ff-0428436c59e5/a0810f6b-f014-44df-91ff-0428436c59e5_mask.nii.gz'
label_map = {
    "femur_left": 1, "femur_right": 2, 
    "tibia_left": 3, "tibia_right": 4, 
    "patella_left": 5, "patella_right": 6
}

raw_mesh = extract_multilabel_surface(mask_path, label_map)
labels = raw_mesh.cell_data['BoundaryLabels']

import numpy as np
mask = ((labels[:,0]==2) & (labels[:,1]==4)) | ((labels[:,0]==4) & (labels[:,1]==2))
print("Shared faces between 2 and 4 (Right Femur/Tibia):", np.sum(mask))

mask_l = ((labels[:,0]==1) & (labels[:,1]==3)) | ((labels[:,0]==3) & (labels[:,1]==1))
print("Shared faces between 1 and 3 (Left Femur/Tibia):", np.sum(mask_l))
