import os
import sys
import argparse
import subprocess
from pathlib import Path

# Ensure the root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import SimpleITK as sitk
from src.preprocessing.data_utils import resample_label_to_isotropic

def run_mri_segmentation(input_path: str, output_path: str):
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    
    # 1. Setup temp IO directories
    temp_dir = Path(f"temp_mri_inference_{input_path.stem.replace('.nii', '')}")
    input_dir = temp_dir / "input"
    output_dir = temp_dir / "output"
    
    import shutil
    if input_dir.exists():
        shutil.rmtree(input_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
        
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy the input file to temp input directory with correct _0000 suffix
    case_id = input_path.name.split('.')[0].replace('_0000', '')
    temp_in_file = input_dir / f"{case_id}_0000.nii.gz"
    
    # CartiMorph expects RAI orientation. Standardize it here.
    img = sitk.ReadImage(str(input_path))
    orient_filter = sitk.DICOMOrientImageFilter()
    orient_filter.SetDesiredCoordinateOrientation("RAI")
    rai_img = orient_filter.Execute(img)
    sitk.WriteImage(rai_img, str(temp_in_file))
    
    # 2. Write WSL script
    wsl_script = temp_dir / "run_inference.py"
    script_content = f"""
import torch
original_load = torch.load
def new_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = new_load

from CartiMorph_nnUNet.inference.predict import predict_from_folder

model_dir = "temp_cartimorph/Models/segModel/segModel-OAIZIB-19Mar2024"
predict_from_folder(
    model=model_dir,
    input_folder="{input_dir.as_posix()}",
    output_folder="{output_dir.as_posix()}",
    folds=(0,),
    save_npz=False,
    num_threads_preprocessing=2,
    num_threads_nifti_save=2,
    lowres_segmentations=None,
    part_id=0,
    num_parts=1,
    tta=False, 
    overwrite_existing=True,
    mode='normal',
    overwrite_all_in_gpu=None,
    mixed_precision=True,
    step_size=0.5,
    checkpoint_name="segModel" 
)
"""
    wsl_script.write_text(script_content)
    
    # Run the script in WSL
    print("Running CartiMorph inference in WSL...")
    # use relative path for WSL execution
    rel_wsl_script = f"{temp_dir.name}/run_inference.py"
    cmd = ["wsl", "-e", "bash", "-c", f"source ~/cartimorph_venv/bin/activate && python3 {rel_wsl_script}"]
    # Generous, not tight: WSL not started or a GPU hang inside it used to
    # block forever with no recovery. This exists to fail loudly after a long
    # wait, not to catch merely-slow-but-healthy runs.
    CARTIMORPH_TIMEOUT_S = 900
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=CARTIMORPH_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise Exception(f"CartiMorph WSL inference timed out after {CARTIMORPH_TIMEOUT_S}s "
                       f"(possible WSL/GPU hang)")
    if result.returncode != 0:
        raise Exception(f"Inference Failed! \nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
        
    print("Inference completed. Resampling to 0.5mm isotropic grid...")
    
    # 3. Read the native prediction and resample
    native_pred_path = output_dir / f"{case_id}.nii.gz"
    if not native_pred_path.exists():
        raise Exception(f"Error: expected output {native_pred_path} not found.\nSTDOUT: {result.stdout}")
        
    pred_native = sitk.ReadImage(str(native_pred_path))
    pred_iso = resample_label_to_isotropic(pred_native, target_spacing=0.5)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(pred_iso, str(output_path))
    print(f"Saved resampled segmentation to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MRI Segmentation via CartiMorph")
    parser.add_argument("-i", "--input", required=True, help="Input raw NIfTI scan")
    parser.add_argument("-o", "--output", required=True, help="Output NIfTI label path")
    args = parser.parse_args()
    
    run_mri_segmentation(args.input, args.output)
