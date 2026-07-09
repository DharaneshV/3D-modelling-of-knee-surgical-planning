import os
import sys
import subprocess
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

sys.path.insert(0, str(os.path.abspath('.')))
from scripts.postprocess_mesh import apply_morphological_closing

log_path = 'data/build_log.json'
with open(log_path, 'r') as f:
    log = json.load(f)

for case_id, info in log.items():
    if info.get('status') == 'pass':
        orig_mask_path = f'outputs/{case_id}/masks/bone_mask.nii.gz'
        closed_mask_path = f'outputs/{case_id}/masks/bone_mask_closed.nii.gz'
        out_dir = f'outputs/{case_id}/meshes'
        
        if os.path.exists(orig_mask_path):
            logging.info(f'--- Processing {case_id} ---')
            # 1. Apply closing to the mask (save to new file for safety)
            apply_morphological_closing(orig_mask_path, closed_mask_path, radius=1)
            
            # 2. Re-run meshing from the safely closed mask
            subprocess.run(['.\\venv\\Scripts\\python.exe', 'scripts/run_meshing.py', '--input', closed_mask_path, '--output_dir', out_dir, '--track', 'ct_bone'], check=True)
