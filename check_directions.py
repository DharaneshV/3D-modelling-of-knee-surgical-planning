import SimpleITK as sitk

cases = ['STS_006', 'STS_035', 'STS_043', 'STS_051']
for case in cases:
    img = sitk.ReadImage(f"data/ct_knee/case01_{case}_bone_mask.nii.gz")
    print(f"{case} Direction: {img.GetDirection()}")
