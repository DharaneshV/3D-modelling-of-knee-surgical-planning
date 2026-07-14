import SimpleITK as sitk
import numpy as np

img = sitk.ReadImage('meshes/a0810f6b-f014-44df-91ff-0428436c59e5/a0810f6b-f014-44df-91ff-0428436c59e5_mask.nii.gz')
arr = sitk.GetArrayFromImage(img)
print('Labels:', np.unique(arr))
for i in range(1,7):
    print(f'Label {i}: {np.sum(arr==i)}')
