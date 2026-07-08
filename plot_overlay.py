import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os

case = 'STS_006'
our_mask_path = f"data/ct_knee/case01_{case}_bone_mask.nii.gz"
ts_total_dir = f"data/ct_knee/{case}_totalseg_reference"

pred_img = sitk.ReadImage(our_mask_path)
pred_arr = sitk.GetArrayFromImage(pred_img)

# We want the CT image to show as background, but our mask is on the resampled 1x1x1 grid
# Let's load the raw CT and resample it to pred_img grid
ct_raw = sitk.ReadImage(f"data/ct_knee/case01_{case}_cropped.nii.gz")
resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(pred_img)
resampler.SetInterpolator(sitk.sitkLinear)
resampler.SetDefaultPixelValue(-1000)
ct_resampled = resampler.Execute(ct_raw)
ct_arr = sitk.GetArrayFromImage(ct_resampled)

# Get TS femur
template_img = sitk.ReadImage(os.path.join(ts_total_dir, "femur_right.nii.gz"))
template_arr = sitk.GetArrayFromImage(template_img)
combined = np.zeros_like(template_arr, dtype=np.uint8)

for side in ['left', 'right']:
    p = os.path.join(ts_total_dir, f"femur_{side}.nii.gz")
    if os.path.exists(p):
        arr = sitk.GetArrayFromImage(sitk.ReadImage(p))
        combined[arr > 0] = 1

ts_img_raw = sitk.GetImageFromArray(combined)
ts_img_raw.CopyInformation(template_img)

resampler.SetInterpolator(sitk.sitkNearestNeighbor)
resampler.SetDefaultPixelValue(0)
ts_img = resampler.Execute(ts_img_raw)
ts_arr = sitk.GetArrayFromImage(ts_img)

# We only care about Femur (label 1)
our_femur = (pred_arr == 1)
ts_femur = (ts_arr == 1)

# Pick a coronal slice that intersects the femur well
# The shape is (296, 502, 252) -> Z, Y, X
# Coronal is Y. Let's find a Y slice with a lot of femur
y_sums = our_femur.sum(axis=(0, 2))
best_y = np.argmax(y_sums)

# Let's also do a sagittal slice (X)
x_sums = our_femur.sum(axis=(0, 1))
best_x = np.argmax(x_sums)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

def plot_overlay(ax, ct_slice, our_slice, ts_slice, title):
    ax.imshow(ct_slice, cmap='gray', vmin=-200, vmax=1000)
    
    # Create RGBA overlays
    # Our mask (red)
    red = np.zeros(our_slice.shape + (4,))
    red[our_slice, 0] = 1.0 # R
    red[our_slice, 3] = 0.5 # Alpha
    
    # TS mask (green)
    green = np.zeros(ts_slice.shape + (4,))
    green[ts_slice, 1] = 1.0 # G
    green[ts_slice, 3] = 0.5 # Alpha
    
    ax.imshow(red)
    ax.imshow(green)
    ax.set_title(title)
    ax.axis('off')

# Sagittal (x)
plot_overlay(axes[0], np.flipud(ct_arr[:, :, best_x]), np.flipud(our_femur[:, :, best_x]), np.flipud(ts_femur[:, :, best_x]), f"Sagittal Slice {best_x} (Red=Ours, Green=TS)")
# Coronal (y)
plot_overlay(axes[1], np.flipud(ct_arr[:, best_y, :]), np.flipud(our_femur[:, best_y, :]), np.flipud(ts_femur[:, best_y, :]), f"Coronal Slice {best_y} (Red=Ours, Green=TS)")

plt.tight_layout()
plt.savefig(f"STS_006_femur_overlay.png", dpi=300)
print(f"Saved overlay to STS_006_femur_overlay.png")
