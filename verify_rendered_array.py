import numpy as np
import SimpleITK as sitk

# Load CT and mask
ct_path = "data/ct_knee/case01_STS_006_cropped_leg.nii.gz"
mask_path = "data/ct_knee/case01_STS_006_bone_mask.nii.gz"
ct = sitk.ReadImage(ct_path)
mask = sitk.ReadImage(mask_path)
ct_arr = sitk.GetArrayFromImage(ct)
mask_arr = sitk.GetArrayFromImage(mask)

# Find centroids of femur (1) and tibia (2) in Z-index space
femur_indices = np.where(mask_arr == 1)
tibia_indices = np.where(mask_arr == 2)

femur_z_centroid = np.mean(femur_indices[0])
tibia_z_centroid = np.mean(tibia_indices[0])

print(f"Femur Z-index centroid: {femur_z_centroid:.1f}")
print(f"Tibia Z-index centroid: {tibia_z_centroid:.1f}")

# The "rendered array" for coronal view in matplotlib is ct_arr[:, y_best, :]
# We are plotting it with origin='lower'.
# In matplotlib, when origin='lower', the row index 0 is at the visual BOTTOM.
# The row index max is at the visual TOP.

# Let's map this to a "visual Y" coordinate where 0 is TOP and max is BOTTOM (like a PNG image).
# visual_Y = (max_Z_index) - Z_index
max_z_index = ct_arr.shape[0] - 1
femur_visual_y = max_z_index - femur_z_centroid
tibia_visual_y = max_z_index - tibia_z_centroid

print(f"Visual Y (0=TOP, {max_z_index}=BOTTOM) for Femur: {femur_visual_y:.1f}")
print(f"Visual Y (0=TOP, {max_z_index}=BOTTOM) for Tibia: {tibia_visual_y:.1f}")

# Assertion: Femur should be in the top half (visual Y < mid), Tibia in the bottom half (visual Y > mid)
mid_y = max_z_index / 2
print(f"Midpoint visual Y: {mid_y:.1f}")

if femur_visual_y < mid_y and tibia_visual_y > mid_y:
    print("\nASSERTION PASSED: Femur is in the TOP half, Tibia is in the BOTTOM half of the rendered image.")
else:
    print("\nASSERTION FAILED: Anatomical orientation in rendered image is incorrect.")
