import SimpleITK as sitk
import numpy as np

for name in ["cropped", "bone_mask"]:
    path = f"data/ct_knee/case01_STS_006_{name}.nii.gz"
    img = sitk.ReadImage(path)
    print(f"\n{name}:")
    print(f"  Size: {img.GetSize()}")
    print(f"  Spacing: {img.GetSpacing()}")
    print(f"  Origin: {img.GetOrigin()}")
    print(f"  Direction: {img.GetDirection()}")
