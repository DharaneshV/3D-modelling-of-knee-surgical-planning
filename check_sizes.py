import SimpleITK as sitk
try:
    pred = sitk.ReadImage("data/ct_knee/STS_006/soft_tissue_sarcoma/STS_006/case01_STS_006.nii.gz")
    print("Original size:", pred.GetSize())
except Exception as e:
    print(e)
