import SimpleITK as sitk

for name in ["cropped", "bone_mask"]:
    path = f"data/ct_knee/case01_STS_006_{name}.nii.gz"
    try:
        img = sitk.ReadImage(path)
        print(f"{name} size: {img.GetSize()}")
    except Exception as e:
        print(f"Error reading {name}: {e}")
