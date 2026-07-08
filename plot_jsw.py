import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

case = 'STS_006'
ct_path = f"data/ct_knee/case01_{case}_cropped.nii.gz"
our_mask_path = f"data/ct_knee/case01_{case}_bone_mask.nii.gz"

ct = sitk.GetArrayFromImage(sitk.ReadImage(ct_path))
arr = sitk.GetArrayFromImage(sitk.ReadImage(our_mask_path))

f_mask = arr == 1
t_mask = arr == 2

# Find Z where both exist
f_z = np.where(np.any(f_mask, axis=(1,2)))[0]
t_z = np.where(np.any(t_mask, axis=(1,2)))[0]
overlap_z = np.intersect1d(f_z, t_z)

if len(overlap_z) > 0:
    print(f"Overlap Z slices: {overlap_z}")
    plot_z = overlap_z[len(overlap_z)//2] # middle of overlap
else:
    # They don't overlap, they are adjacent
    print(f"Femur Z min: {f_z[0]}, Tibia Z max: {t_z[-1]}")
    plot_z = f_z[0] # just above tibia

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
zs_to_plot = [plot_z - 2, plot_z, plot_z + 2]

for i, z in enumerate(zs_to_plot):
    if 0 <= z < ct.shape[0]:
        axes[i].imshow(ct[z], cmap='gray', vmin=-200, vmax=1000)
        axes[i].contour(f_mask[z], colors='red', linewidths=1)
        axes[i].contour(t_mask[z], colors='green', linewidths=1)
        axes[i].set_title(f"Z={z}")

plt.tight_layout()
plt.savefig('jsw_verification.png')
