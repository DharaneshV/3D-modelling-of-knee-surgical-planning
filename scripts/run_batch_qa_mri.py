import os
import sys
import glob
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.pipeline_runner import _execute_pipeline

def main():
    oaizib_files = sorted(glob.glob(r"d:\knee surgery model\data\oaizib\imagesTs\*.nii.gz"))
    
    results = []
    
    for file in oaizib_files:
        case_id = Path(file).name.split("_0000.nii.gz")[0]
        if case_id == Path(file).name:
            case_id = Path(file).name.replace(".nii.gz", "")
            
        print(f"==================================================")
        print(f"Running pipeline on {case_id}...")
        
        status = "Pass"
        error_msg = ""
        try:
            _execute_pipeline(case_id, file, "MRI")
        except Exception as e:
            status = "Fail"
            error_msg = str(e)
            
        # Parse metrics
        mesh_dir = Path(r"d:\knee surgery model\meshes") / case_id
        
        sigma_femur = "N/A"
        sigma_tibia = "N/A"
        route_femur = "N/A"
        route_tibia_med = "N/A"
        route_tibia_lat = "N/A"
        
        if (mesh_dir / "femur_sigma.json").exists():
            with open(mesh_dir / "femur_sigma.json") as f:
                sigma_femur = json.load(f).get("sigma", "N/A")
                if isinstance(sigma_femur, float):
                    sigma_femur = round(sigma_femur, 2)
                    
        if (mesh_dir / "tibia_sigma.json").exists():
            with open(mesh_dir / "tibia_sigma.json") as f:
                sigma_tibia = json.load(f).get("sigma", "N/A")
                if isinstance(sigma_tibia, float):
                    sigma_tibia = round(sigma_tibia, 2)
                    
        if (mesh_dir / "femur_unknown_femoral_cartilage_resolution.json").exists():
            with open(mesh_dir / "femur_unknown_femoral_cartilage_resolution.json") as f:
                route_femur = json.load(f).get("route", "N/A")
                
        # Since we use tibia_unknown for both medial and lateral, run_meshing.py overwrites it.
        # But we can capture whichever was used (or maybe it failed earlier).
        if (mesh_dir / "tibia_unknown_medial_tibial_cartilage_resolution.json").exists():
            with open(mesh_dir / "tibia_unknown_medial_tibial_cartilage_resolution.json") as f:
                route_tibia_med = json.load(f).get("route", "N/A")
        
        results.append({
            "case_id": case_id,
            "status": status,
            "error": error_msg,
            "sigma_femur": sigma_femur,
            "sigma_tibia": sigma_tibia,
            "route_femur": route_femur,
            "route_tibia_med": route_tibia_med
        })
        
        print(f"Result for {case_id}: {status} (Femur: {sigma_femur} sigma, Route: {route_femur})")
        
    print("\n\n" + "="*80)
    print("BATCH QA REPORT")
    print("="*80)
    print(f"{'Case ID':<15} | {'Status':<15} | {'Femur sigma':<12} | {'Tibia sigma':<12} | {'Femur Route':<15} | {'Tibia Route':<15}")
    print("-" * 80)
    for res in results:
        print(f"{res['case_id']:<15} | {res['status']:<15} | {res['sigma_femur']:<8} | {res['sigma_tibia']:<8} | {res['route_femur']:<15} | {res['route_tibia_med']:<15}")
        
    with open("batch_qa_mri_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
if __name__ == "__main__":
    main()
