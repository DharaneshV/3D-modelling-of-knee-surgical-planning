import os
import sys
import subprocess
import glob
from pathlib import Path
import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.synthesis.extract_cart_features import extract_cart_features

CASES = ['DU02', 'DU03', 'DU06']
DATA_DIR = Path('d:/knee surgery model/data/mri&ct')
SCRATCH_DIR = Path('d:/knee surgery model/scratch')

def dicom_to_nifti(dicom_dir: Path, out_path: Path):
    if out_path.exists():
        print(f"Already exists: {out_path}")
        return
    print(f"Converting {dicom_dir} to {out_path}...")
    reader = sitk.ImageSeriesReader()
    series_IDs = reader.GetGDCMSeriesIDs(str(dicom_dir))
    if not series_IDs:
        raise ValueError(f"No DICOM series found in {dicom_dir}")
    
    dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_dir), series_IDs[0])
    reader.SetFileNames(dicom_names)
    img = reader.Execute()
    sitk.WriteImage(img, str(out_path))

def reorient_to_rai(input_path: Path, output_path: Path):
    if output_path.exists():
        print(f"Already exists: {output_path}")
        return
    print(f"Reorienting {input_path} to RAI...")
    img = sitk.ReadImage(str(input_path))
    orient_filter = sitk.DICOMOrientImageFilter()
    orient_filter.SetDesiredCoordinateOrientation("RAI")
    rai_img = orient_filter.Execute(img)
    sitk.WriteImage(rai_img, str(output_path))

def run_cmd(cmd):
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    for case in CASES:
        print(f"\n{'='*40}\nProcessing {case}\n{'='*40}")
        case_dir = DATA_DIR / case
        out_dir = SCRATCH_DIR / case
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Find CT and MRI dirs
        ct_dir = None
        mri_dir = None
        for d in case_dir.iterdir():
            if not d.is_dir(): continue
            name = d.name.lower()
            if 'knee_' in name:
                ct_dir = d
            elif 'sag' in name or 'fs_sag' in name or 'mri' in name:
                mri_dir = d
                
        if not ct_dir or not mri_dir:
            print(f"Warning: Could not find CT or MRI dir for {case}")
            continue
            
        ct_nii = out_dir / "CT.nii.gz"
        mri_nii = out_dir / "MRI.nii.gz"
        
        # 1. Convert DICOM to NIfTI
        dicom_to_nifti(ct_dir, ct_nii)
        dicom_to_nifti(mri_dir, mri_nii)
        
        # 2. CT Segmentation
        ct_mask = out_dir / "bone_mask.nii.gz"
        if not ct_mask.exists():
            run_cmd([sys.executable, "src/segmentation/run_ct_segmentation.py", "--input", str(ct_nii), "--output", str(ct_mask)])
            
        # 3. CT Meshing
        mesh_dir = out_dir / "meshes"
        if not mesh_dir.exists():
            mesh_dir.mkdir()
            run_cmd([sys.executable, "scripts/run_meshing.py", "--input", str(ct_mask), "--output_dir", str(mesh_dir), "--track", "ct_bone"])
            
        # 4. MRI Reorientation
        mri_rai = out_dir / "MRI_RAI.nii.gz"
        reorient_to_rai(mri_nii, mri_rai)
        
        # 5. MRI Segmentation
        mri_mask = out_dir / "cartilage_mask_rai.nii.gz"
        if not mri_mask.exists():
            run_cmd([sys.executable, "src/segmentation/run_mri_segmentation.py", "--input", str(mri_rai), "--output", str(mri_mask)])
            
        # 6. Extract Features
        features_path = out_dir / "features.npy"
        if not features_path.exists():
            print(f"Extracting features for {case}...")
            feats = extract_cart_features(str(mri_mask))
            np.save(str(features_path), feats)
            print(f"Saved {features_path}")
            
    print("\nData preparation complete!")

if __name__ == "__main__":
    main()
