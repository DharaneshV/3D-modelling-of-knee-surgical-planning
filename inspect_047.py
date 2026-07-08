import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np

def create_mip(img_path, output_path):
    img = sitk.ReadImage(img_path)
    arr = sitk.GetArrayFromImage(img)
    
    mip_coronal = np.max(arr, axis=1) # Y-axis projection
    mip_sagittal = np.max(arr, axis=2) # X-axis projection
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 8))
    axes[0].imshow(mip_coronal, cmap='gray', aspect='auto', origin='lower')
    axes[0].set_title('Coronal MIP')
    
    axes[1].imshow(mip_sagittal, cmap='gray', aspect='auto', origin='lower')
    axes[1].set_title('Sagittal MIP')
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved {output_path}")

create_mip("data/ct_knee/case01_STS_047.nii.gz", "inspect_047_raw.png")
create_mip("data/ct_knee/case01_STS_047_cropped.nii.gz", "inspect_047_cropped.png")
