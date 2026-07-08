import sys
import numpy as np
sys.path.append(".")
from src.mesh.surface_nets import generate_multilabel_mesh
raw_mesh = generate_multilabel_mesh("data/ct_knee/case01_STS_006_bone_mask.nii.gz", {'femur': 1, 'tibia': 2, 'patella': 3})
labels = raw_mesh.cell_data['BoundaryLabels']
print(f"Labels shape: {labels.shape}")
print(f"Sample labels: {labels[:5]}")
