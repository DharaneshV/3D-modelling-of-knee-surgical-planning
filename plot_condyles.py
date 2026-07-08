import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

case = 'STS_006'
our_mask_path = f"data/ct_knee/case01_{case}_bone_mask.nii.gz"
arr = sitk.GetArrayFromImage(sitk.ReadImage(our_mask_path))

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

for i, z in enumerate([130, 150]):
    axes[i].imshow(arr[z] == 1, cmap='gray')
    axes[i].set_title(f"Z={z} Mask (Condyles)")

plt.tight_layout()
plt.savefig('condyle_spot_check.png')
