import os
import glob
import SimpleITK as sitk
from src.preprocessing.data_utils import preprocess_mri

cases = [f'oaizib_{i}' for i in range(405, 420)]
gt_out_dir = 'temp_cartimorph_io/gt_test_15'
os.makedirs(gt_out_dir, exist_ok=True)

for case in cases:
    label_path = f'data/oaizib/labelsTs/{case}.nii.gz'
    if os.path.exists(label_path):
        label = sitk.ReadImage(label_path)
        # resample to 0.5mm isotropic using nearest neighbor
        from src.preprocessing.data_utils import resample_label_to_isotropic
        label_resampled = resample_label_to_isotropic(label, target_spacing=0.5)
        out_path = f'{gt_out_dir}/{case}_label.nii.gz'
        sitk.WriteImage(label_resampled, out_path)
        print(f'Processed GT {out_path}')
    else:
        print(f'Warning: no label for {case}')
