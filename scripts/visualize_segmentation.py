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

import argparse

def main():
    parser = argparse.ArgumentParser(description="Visualize CT + bone mask overlay")
    parser.add_argument("--ct", required=True, help="Path to CT volume")
    parser.add_argument("--mask", required=True, help="Path to bone mask")
    parser.add_argument("--output", required=True, help="Path to save preview image")
    args = parser.parse_args()

    print(f"Loading CT volume from {args.ct}...")
    ct = sitk.ReadImage(args.ct, sitk.sitkFloat32)
    ct_arr = sitk.GetArrayFromImage(ct)  # shape: (Z, Y, X)

    print(f"Loading bone mask from {args.mask}...")
    mask = sitk.ReadImage(args.mask, sitk.sitkUInt16)
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

    # Create Maximum Intensity Projections (MIP) for the CT volume
    ct_mip_z = np.max(ct_display, axis=0)
    ct_mip_y = np.max(ct_display, axis=1)
    ct_mip_x = np.max(ct_display, axis=2)

    # For the mask, we want to show the labels. 
    # Taking a max over the labels can arbitrarily favor higher labels.
    # A better approach is to take the MIP of each label independently, 
    # then combine them, but for quick 2D visualization, we can just take the max
    # since labels are 1, 2, 3 and we don't have overlapping bone structures along the projection
    # unless there's severe deformity. However, to ensure tibia (2) doesn't just overwrite femur (1),
    # we can create independent MIPs per label and combine them.
    def get_label_mip(mask, label, axis):
        return np.max((mask == label).astype(np.uint8), axis=axis)
        
    mask_mip_z = np.zeros_like(ct_mip_z, dtype=np.uint16)
    mask_mip_y = np.zeros_like(ct_mip_y, dtype=np.uint16)
    mask_mip_x = np.zeros_like(ct_mip_x, dtype=np.uint16)
    
    for l in [1, 2, 3]:
        mask_mip_z[get_label_mip(mask_arr, l, 0) > 0] = l
        mask_mip_y[get_label_mip(mask_arr, l, 1) > 0] = l
        mask_mip_x[get_label_mip(mask_arr, l, 2) > 0] = l

    z_mid = mask_arr.shape[0] // 2
    y_mid = mask_arr.shape[1] // 2
    x_mid = mask_arr.shape[2] // 2

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='#0a0e1a')
    fig.suptitle('Bone Segmentation Results — MIP (Maximum Intensity Projection)',
                 fontsize=16, color='white', fontweight='bold', y=1.05)

    views = [
        ("Axial MIP", ct_mip_z, mask_mip_z),
        ("Coronal MIP", ct_mip_y, mask_mip_y),
        ("Sagittal MIP", ct_mip_x, mask_mip_x),
    ]

    for ax, (title, ct_slice, mask_slice) in zip(axes, views):
        origin = 'lower' if ('Coronal' in title or 'Sagittal' in title) else 'upper'
        ax.imshow(ct_slice, cmap='gray', aspect='auto', origin=origin)
        # Only overlay where mask > 0
        masked = np.ma.masked_where(mask_slice == 0, mask_slice)
        ax.imshow(masked, cmap=cmap_mask, vmin=0, vmax=3, aspect='auto', origin=origin)
        ax.set_title(title, color='white', fontsize=12, pad=8)
        
        if 'Coronal' in title or 'Sagittal' in title:
            ax.set_ylabel("Z-index", color='white')
            ax.tick_params(axis='y', colors='white')
            ax.tick_params(axis='x', bottom=False, labelbottom=False)
            
            # Annotate top and bottom of the plot visually
            height = ct_slice.shape[0]
            if origin == 'lower':
                top_z = height - 1
                bottom_z = 0
            else:
                top_z = 0
                bottom_z = height - 1
                
            ax.text(0.5, 0.98, f"Z-index={top_z}", color='yellow', fontsize=12, ha='center', va='top', transform=ax.transAxes, fontweight='bold')
            ax.text(0.5, 0.02, f"Z-index={bottom_z}", color='yellow', fontsize=12, ha='center', va='bottom', transform=ax.transAxes, fontweight='bold')
        else:
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

    if os.path.dirname(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches='tight', facecolor='#0a0e1a')
    plt.close()
    print(f"\nSaved preview to {args.output}")


if __name__ == "__main__":
    main()
