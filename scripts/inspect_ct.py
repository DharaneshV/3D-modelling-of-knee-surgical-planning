import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import sys

def main():
    path = "data/ct_knee/temp_EAY131-5310722.nii.gz"
    print(f"Loading {path}...")
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    print(f"Shape: {arr.shape} (Z, Y, X)")
    
    z, y, x = arr.shape
    
    # Get middle slices
    axial = arr[z//2, :, :]
    coronal = arr[:, y//2, :]
    sagittal = arr[:, :, x//2]
    
    # Also get MIP (Maximum Intensity Projection) which is great for bones
    mip_coronal = np.max(arr, axis=1)
    mip_sagittal = np.max(arr, axis=2)
    mip_axial = np.max(arr, axis=0)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Slices
    axes[0, 0].imshow(axial, cmap='gray', vmin=-1000, vmax=1000)
    axes[0, 0].set_title(f"Axial Slice {z//2}")
    axes[0, 1].imshow(coronal, cmap='gray', vmin=-1000, vmax=1000)
    axes[0, 1].set_title(f"Coronal Slice {y//2}")
    axes[0, 2].imshow(sagittal, cmap='gray', vmin=-1000, vmax=1000)
    axes[0, 2].set_title(f"Sagittal Slice {x//2}")
    
    # MIPs
    axes[1, 0].imshow(mip_axial, cmap='gray', vmin=0, vmax=1500)
    axes[1, 0].set_title("Axial MIP")
    axes[1, 1].imshow(mip_coronal, cmap='gray', vmin=0, vmax=1500)
    axes[1, 1].set_title("Coronal MIP")
    axes[1, 2].imshow(mip_sagittal, cmap='gray', vmin=0, vmax=1500)
    axes[1, 2].set_title("Sagittal MIP")
    
    plt.tight_layout()
    plt.savefig("ct_inspection.png")
    print("Saved ct_inspection.png")

if __name__ == "__main__":
    main()
