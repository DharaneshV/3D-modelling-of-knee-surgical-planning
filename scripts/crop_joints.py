import os
import glob
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

def locate_joint_center(binary_bone_mask):
    """Find z-slice with locally maximal bone cross-sectional area."""
    arr = sitk.GetArrayFromImage(binary_bone_mask)  # z, y, x
    area_profile = arr.sum(axis=(1, 2))  # bone voxel count per z-slice

    # Smooth to avoid noise picking a spurious single-slice spike
    smoothed = uniform_filter1d(area_profile.astype(float), size=15)

    joint_z = int(smoothed.argmax())
    return joint_z, smoothed

def crop_around_joint(ct_image, joint_z, half_window_mm=150):
    spacing_z = ct_image.GetSpacing()[2]
    half_window_voxels = int(half_window_mm / spacing_z)
    
    z_start = max(0, joint_z - half_window_voxels)
    z_end = min(ct_image.GetSize()[2], joint_z + half_window_voxels)
    
    return sitk.RegionOfInterest(
        ct_image, 
        size=[ct_image.GetSize()[0], ct_image.GetSize()[1], z_end - z_start],
        index=[0, 0, z_start]
    )

def generate_preview(ct_cropped, joint_z_in_crop, out_path, title="Knee Crop"):
    arr = sitk.GetArrayFromImage(ct_cropped)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Axial slice at joint
    axial = np.flipud(arr[joint_z_in_crop, :, :])
    axes[0].imshow(axial, cmap='gray', vmin=-200, vmax=1000)
    axes[0].set_title(f"Axial @ Joint Center")
    axes[0].axis('off')
    
    # 2. Coronal MIP
    coronal = np.flipud(np.max(arr, axis=1))
    axes[1].imshow(coronal, cmap='gray', vmin=-200, vmax=1000)
    axes[1].set_title("Coronal MIP (300mm crop)")
    axes[1].axis('off')
    
    # 3. Sagittal MIP
    sagittal = np.flipud(np.max(arr, axis=2))
    axes[2].imshow(sagittal, cmap='gray', vmin=-200, vmax=1000)
    axes[2].set_title("Sagittal MIP (300mm crop)")
    axes[2].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

def main():
    cases = glob.glob("data/ct_knee/case01_STS_*.nii.gz")
    cases = [c for c in cases if not "bone_mask" in c and not "cropped" in c]
    
    for case_path in cases:
        name = os.path.basename(case_path).replace(".nii.gz", "")
        print(f"\nProcessing {name}...")
        
        ct = sitk.ReadImage(case_path)
        
        # Simple threshold to get bone for joint locator
        # We use a lower threshold to capture all bone including cancellous
        binary = ct > 200
        
        joint_z, profile = locate_joint_center(binary)
        print(f"  -> Found joint peak at Z-slice: {joint_z} out of {ct.GetSize()[2]}")
        
        ct_cropped = crop_around_joint(ct, joint_z, half_window_mm=150)
        
        # Calculate where the joint is in the cropped image
        spacing_z = ct.GetSpacing()[2]
        half_window_voxels = int(150 / spacing_z)
        z_start = max(0, joint_z - half_window_voxels)
        joint_z_in_crop = joint_z - z_start
        
        # Save cropped NIfTI
        out_nifti = case_path.replace(".nii.gz", "_cropped.nii.gz")
        sitk.WriteImage(ct_cropped, out_nifti)
        print(f"  -> Saved {out_nifti}")
        
        # Save preview
        out_png = f"{name}_crop_preview.png"
        generate_preview(ct_cropped, joint_z_in_crop, out_png, title=f"{name} Knee Crop")
        print(f"  -> Saved preview: {out_png}")

if __name__ == "__main__":
    main()
