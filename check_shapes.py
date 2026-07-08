import SimpleITK as sitk
try:
    img1 = sitk.ReadImage("data/ct_knee/case01_STS_006_bone_labels.nii.gz")
    print(f"bone_labels shape: {img1.GetSize()}")
    img2 = sitk.ReadImage("data/ct_knee/case01_STS_006_bone_mask.nii.gz")
    print(f"bone_mask shape: {img2.GetSize()}")
except Exception as e:
    print(e)
