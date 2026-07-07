"""
Quick visualization of CT + bone mask overlay using matplotlib.
Generates a PNG grid showing axial, coronal, sagittal slices.

Usage:
    python scripts/visualize_segmentation.py
"""

import os
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

CT_PATH = "data/ct_knee/temp_EAY131-5310722.nii.gz"
MASK_PATH = "data/ct_knee/case01_bone_mask.nii.gz"
OUTPUT_PATH = "results/bone_segmentation_preview.png"


def main():
    print("Loading CT volume...")
    ct = sitk.ReadImage(CT_PATH, sitk.sitkFloat32)
    ct_arr = sitk.GetArrayFromImage(ct)  # shape: (Z, Y, X)

    print("Loading bone mask...")
    mask = sitk.ReadImage(MASK_PATH, sitk.sitkUInt16)
    mask_arr = sitk.GetArrayFromImage(mask)  # shape: (Z, Y, X)

    print(f"CT shape: {ct_arr.shape}, Mask shape: {mask_arr.shape}")
    print(f"CT range: [{ct_arr.min():.0f}, {ct_arr.max():.0f}] HU")
    print(f"Mask labels present: {np.unique(mask_arr)}")

    # If shapes don't match (CT wasn't resampled), resample CT to match mask
    if ct_arr.shape != mask_arr.shape:
        print("Resampling CT to match mask dimensions...")
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(mask)
        resampler.SetInterpolator(sitk.sitkLinear)
        ct = resampler.Execute(ct)
        ct_arr = sitk.GetArrayFromImage(ct)

    # Windowing for bone CT display
    ct_display = np.clip(ct_arr, -200, 1500)
    ct_display = (ct_display - ct_display.min()) / (ct_display.max() - ct_display.min())

    # Custom colormap: 0=transparent, 1=red(femur), 2=green(tibia), 3=blue(patella)
    colors = [
        (0, 0, 0, 0),        # 0: transparent
        (0.9, 0.2, 0.2, 0.6),  # 1: femur - red
        (0.2, 0.8, 0.2, 0.6),  # 2: tibia - green
        (0.2, 0.4, 0.9, 0.6),  # 3: patella - blue
    ]
    cmap_mask = ListedColormap(colors)

    # Pick center slices for each view
    z_mid = mask_arr.shape[0] // 2
    y_mid = mask_arr.shape[1] // 2
    x_mid = mask_arr.shape[2] // 2

    # Find slices with the most mask content for better visualization
    z_counts = np.array([np.count_nonzero(mask_arr[z, :, :]) for z in range(mask_arr.shape[0])])
    y_counts = np.array([np.count_nonzero(mask_arr[:, y, :]) for y in range(mask_arr.shape[1])])
    x_counts = np.array([np.count_nonzero(mask_arr[:, :, x]) for x in range(mask_arr.shape[2])])

    z_best = int(np.argmax(z_counts)) if z_counts.max() > 0 else z_mid
    y_best = int(np.argmax(y_counts)) if y_counts.max() > 0 else y_mid
    x_best = int(np.argmax(x_counts)) if x_counts.max() > 0 else x_mid

    print(f"Best axial slice: z={z_best} ({z_counts[z_best]} mask voxels)")
    print(f"Best coronal slice: y={y_best} ({y_counts[y_best]} mask voxels)")
    print(f"Best sagittal slice: x={x_best} ({x_counts[x_best]} mask voxels)")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor='#0a0e1a')
    fig.suptitle('Bone Segmentation Results — HU Thresholding',
                 fontsize=16, color='white', fontweight='bold', y=0.98)

    views = [
        ("Axial (best)", ct_display[z_best, :, :], mask_arr[z_best, :, :]),
        ("Coronal (best)", ct_display[:, y_best, :], mask_arr[:, y_best, :]),
        ("Sagittal (best)", ct_display[:, :, x_best], mask_arr[:, :, x_best]),
        ("Axial (center)", ct_display[z_mid, :, :], mask_arr[z_mid, :, :]),
        ("Coronal (center)", ct_display[:, y_mid, :], mask_arr[:, y_mid, :]),
        ("Sagittal (center)", ct_display[:, :, x_mid], mask_arr[:, :, x_mid]),
    ]

    for ax, (title, ct_slice, mask_slice) in zip(axes.flat, views):
        ax.imshow(ct_slice, cmap='gray', aspect='auto')
        # Only overlay where mask > 0
        masked = np.ma.masked_where(mask_slice == 0, mask_slice)
        ax.imshow(masked, cmap=cmap_mask, vmin=0, vmax=3, aspect='auto')
        ax.set_title(title, color='white', fontsize=12, pad=8)
        ax.axis('off')
        ax.set_facecolor('#0a0e1a')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=(0.9, 0.2, 0.2), label='Femur (1)'),
        Patch(facecolor=(0.2, 0.8, 0.2), label='Tibia (2)'),
        Patch(facecolor=(0.2, 0.4, 0.9), label='Patella (3)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               fontsize=12, frameon=False, labelcolor='white',
               bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight', facecolor='#0a0e1a')
    plt.close()
    print(f"\nSaved preview to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
