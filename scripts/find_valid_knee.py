import os
import subprocess
import SimpleITK as sitk
from idc_index import IDCClient

CANDIDATES = {
    "STS_006": "1.3.6.1.4.1.14519.5.2.1.5168.1900.849567849855022085102637138725",
    "STS_035": "1.3.6.1.4.1.14519.5.2.1.5168.1900.272272069210338258600600991202",
    "STS_043": "1.3.6.1.4.1.14519.5.2.1.5168.1900.187333476409221178353189652627",
    "STS_047": "1.3.6.1.4.1.14519.5.2.1.5168.1900.294113156667293063949985551175",
    "STS_051": "1.3.6.1.4.1.14519.5.2.1.5168.1900.186653801311189416337649712538"
}

def convert_dicom_to_nifti(dicom_dir, out_nifti):
    # Find the deepest dir with .dcm files
    dcm_dirs = []
    for root, dirs, files in os.walk(dicom_dir):
        if any(f.endswith('.dcm') for f in files):
            dcm_dirs.append(root)
    
    if not dcm_dirs:
        print(f"No DICOMs found in {dicom_dir}")
        return False
        
    series_dir = dcm_dirs[0]
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(series_dir)
    reader.SetFileNames(dicom_names)
    img = reader.Execute()
    
    sitk.WriteImage(img, out_nifti)
    print(f"Saved {out_nifti}")
    return True

def main():
    client = IDCClient.client()
    
    for name, uid in CANDIDATES.items():
        print(f"\n{'='*50}\nTesting candidate {name}\n{'='*50}")
        
        out_dir = f"data/ct_knee/{name}"
        nifti_path = f"data/ct_knee/case01_{name}.nii.gz"
        mask_path = f"data/ct_knee/case01_{name}_bone_mask.nii.gz"
        
        if not os.path.exists(nifti_path):
            print(f"Downloading {name}...")
            os.makedirs(out_dir, exist_ok=True)
            client.download_dicom_series(seriesInstanceUID=uid, downloadDir=out_dir)
            if not convert_dicom_to_nifti(out_dir, nifti_path):
                continue
        
        print(f"Running bone segmentation with hard gates on {name}...")
        
        # Run segmentation
        cmd = [
            "python", "scripts/bone_segmentation.py",
            "--input", nifti_path,
            "--output", mask_path
        ]
        
        # Run process and capture output
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"SUCCESS! {name} passed all anatomical hard gates!")
            print(f"Ready for meshing. The mask is at {mask_path}")
            return
        else:
            print(f"{name} failed validation. Reason:")
            
            # Extract just the validation error to print cleanly
            lines = result.stderr.split('\n')
            for line in lines:
                if "Anatomical validation failed:" in line or "ValueError:" in line:
                    print(f"  -> {line.strip()}")

if __name__ == "__main__":
    main()
