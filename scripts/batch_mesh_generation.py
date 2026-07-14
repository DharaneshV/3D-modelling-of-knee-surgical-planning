import os
import sys
import subprocess
import json

log_path = 'data/build_log.json'
with open(log_path, 'r') as f:
    log = json.load(f)

for case_id, info in log.items():
    if info.get('status') == 'pass':
        mask_path = f'outputs/{case_id}/masks/bone_mask.nii.gz'
        out_dir = f'outputs/{case_id}/meshes'
        if os.path.exists(mask_path):
            if not os.path.exists(os.path.join(out_dir, 'femur.obj')) and not os.path.exists(os.path.join(out_dir, 'femur_unknown.obj')) and not os.path.exists(os.path.join(out_dir, 'femur_left.obj')):
                print(f'Meshing {case_id}...')
                subprocess.run(['.\\venv\\Scripts\\python.exe', 'scripts/run_meshing.py', '--input', mask_path, '--output_dir', out_dir, '--track', 'ct_bone'], check=True)
            else:
                print(f'{case_id} meshes already exist.')

print('All meshing complete. Running QA report...')
env = os.environ.copy()
env['PYTHONIOENCODING'] = 'utf8'
subprocess.run(['.\\venv\\Scripts\\python.exe', 'scripts/batch_qa_report.py'], env=env, check=True)
