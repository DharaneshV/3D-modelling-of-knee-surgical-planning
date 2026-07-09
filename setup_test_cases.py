import os
import shutil

cases = ['oaizib_405', 'oaizib_406', 'oaizib_407']
input_dir = 'temp_cartimorph_io/input_test'
output_dir = 'temp_cartimorph_io/output_test'

os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

for case in cases:
    src = f'data/oaizib/imagesTs/{case}_0000.nii.gz'
    dst = f'{input_dir}/{case}_0000.nii.gz'
    shutil.copy(src, dst)
print('Copied test cases.')
