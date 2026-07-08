import sys
sys.path.append(".")
from src.mesh.surface_nets import generate_multilabel_mesh
from src.mesh.processing import apply_taubin_smoothing, decimate_mesh
import numpy as np

raw_mesh = generate_multilabel_mesh("data/ct_knee/case01_STS_006_bone_mask.nii.gz", {'femur': 1})
labels = raw_mesh.cell_data['BoundaryLabels']
mask = (labels[:, 0] == 1) | (labels[:, 1] == 1)
sub_mesh = raw_mesh.extract_cells(mask).extract_surface(algorithm='dataset_surface').clean()

print("Saving raw femur to femur_raw.obj")
sub_mesh.save("femur_raw.obj")

smoothed = apply_taubin_smoothing(sub_mesh)
print("Saving smoothed femur to femur_smoothed.obj")
smoothed.save("femur_smoothed.obj")
