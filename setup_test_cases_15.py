import os
import shutil

cases = [f'oaizib_{i}' for i in range(405, 420)]
input_dir = 'temp_cartimorph_io/input_test_15'
output_dir = 'temp_cartimorph_io/output_test_15'

os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

for case in cases:
    src = f'data/oaizib/imagesTs/{case}_0000.nii.gz'
    dst = f'{input_dir}/{case}_0000.nii.gz'
    if os.path.exists(src):
        shutil.copy(src, dst)
print(f'Copied {len(cases)} test cases.')
