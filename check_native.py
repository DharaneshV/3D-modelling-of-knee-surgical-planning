import SimpleITK as sitk
img = sitk.ReadImage('data/oaizib/imagesTr/oaizib_001_0000.nii.gz')
print(f'Native shape: {img.GetSize()}')
print(f'Native spacing: {img.GetSpacing()}')
