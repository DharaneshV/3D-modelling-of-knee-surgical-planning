import os
import sys
import json
import subprocess
import threading
import time
import hashlib
import shutil
from pathlib import Path
from backend.modality_detector import detect_modality
from backend.config import PIPELINE_VERSION

TASKS_DIR = Path("tasks")
TASKS_DIR.mkdir(exist_ok=True)
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)
MESHES_DIR = Path("meshes")
MESHES_DIR.mkdir(exist_ok=True)
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# Global lock for single-job pipeline processing
PIPELINE_LOCK = threading.Lock()

def get_status_path(task_id: str) -> Path:
    task_dir = TASKS_DIR / task_id
    task_dir.mkdir(exist_ok=True)
    return task_dir / "status.json"

def update_status(task_id: str, state: str, reason: str = "", modality: str = "", manifest: dict = None):
    status_path = get_status_path(task_id)
    
    # Read existing
    data = {}
    if status_path.exists():
        try:
            with open(status_path, "r") as f:
                data = json.load(f)
        except:
            pass
            
    data["task_id"] = task_id
    data["state"] = state
    if reason:
        data["reason"] = reason
    if modality:
        data["modality"] = modality
    if manifest:
        data["manifest"] = manifest
        
    with open(status_path, "w") as f:
        json.dump(data, f, indent=4)

def run_pipeline_async(task_id: str, file_path: str, file_hash: str = None):
    """
    Runs the pipeline in a background thread, using a lock to prevent concurrent runs.
    """
    def _run():
        update_status(task_id, "detecting_modality")
        
        # 1. Modality Detection
        mod_result = detect_modality(file_path)
        modality = mod_result["modality"]
        reason = mod_result["reason"]
        
        if modality == "UNKNOWN":
            update_status(task_id, "failed", reason=reason, modality="UNKNOWN")
            return
            
        update_status(task_id, "pending", reason=reason, modality=modality)
        
        # Wait for lock
        with PIPELINE_LOCK:
            update_status(task_id, "segmenting", modality=modality)
            try:
                _execute_pipeline(task_id, file_path, modality)
                
                # Write to cache on success
                if file_hash:
                    write_cache(file_hash, task_id)
            except Exception as e:
                update_status(task_id, "failed", reason=str(e), modality=modality)
                
    threading.Thread(target=_run, daemon=True).start()

def _execute_pipeline(task_id: str, file_path: str, modality: str):
    # Ensure file_path is absolute
    file_path = str(Path(file_path).absolute())
    
    task_mesh_dir = MESHES_DIR / task_id
    task_mesh_dir.mkdir(exist_ok=True)
    
    mask_output = str((task_mesh_dir / f"{task_id}_mask.nii.gz").absolute())
    
    # Use the active environment's python executable
    python_exe = sys.executable
    
    # 2. Segmenting
    if modality == "CT":
        track = "ct_bone"
        expected_parts = [
            {"file": "femur_left.obj",   "label": "Femur (Left)",   "color": "#c4956a", "side": "left"},
            {"file": "femur_right.obj",  "label": "Femur (Right)",  "color": "#c4956a", "side": "right"},
            {"file": "tibia_left.obj",   "label": "Tibia (Left)",   "color": "#4a9e7a", "side": "left"},
            {"file": "tibia_right.obj",  "label": "Tibia (Right)",  "color": "#4a9e7a", "side": "right"},
            {"file": "patella_left.obj", "label": "Patella (Left)", "color": "#e8922d", "side": "left"},
            {"file": "patella_right.obj","label": "Patella (Right)","color": "#e8922d", "side": "right"},
        ]
        
        update_status(task_id, "segmenting", modality=modality, reason="Running TotalSegmentator (femur, tibia, patella)...")
        seg_cmd = [python_exe, "src/segmentation/run_ct_segmentation.py", "--input", file_path, "--output", mask_output]
        result = subprocess.run(seg_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"TotalSegmentator script failed. Error: {result.stderr.strip()}")
            
    else:
        # Run MRI pipeline
        seg_cmd = [python_exe, "src/segmentation/run_mri_segmentation.py", "--input", file_path, "--output", mask_output]
        track = "mri_cartilage"
        expected_parts = [
            {"file": "femur_unknown.obj", "label": "Femur Bone", "color": "#e74c3c"},
            {"file": "tibia_unknown.obj", "label": "Tibia Bone", "color": "#2ecc71"},
            {"file": "femoral_cartilage.obj", "label": "Femoral Cartilage", "color": "#ff9f43"},
            {"file": "medial_tibial_cartilage.obj", "label": "Medial Tibial Cartilage", "color": "#00d2d3"},
            {"file": "lateral_tibial_cartilage.obj", "label": "Lateral Tibial Cartilage", "color": "#54a0ff"}
        ]
        
        result = subprocess.run(seg_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Segmentation script failed: {result.stderr.strip()}")

    if not os.path.exists(mask_output):
        raise Exception(f"Segmentation returned 0 but mask file {mask_output} is missing. STDOUT: {result.stdout} STDERR: {result.stderr}")

    # 3. Meshing
    update_status(task_id, "meshing", modality=modality)
    mesh_cmd = [python_exe, "scripts/run_meshing.py", "--input", mask_output, "--output_dir", str(task_mesh_dir), "--track", track]
    result_mesh = subprocess.run(mesh_cmd, capture_output=True, text=True)
    if result_mesh.returncode != 0:
        raise Exception(f"Meshing script failed:\nSTDOUT:\n{result_mesh.stdout.strip()}\nSTDERR:\n{result_mesh.stderr.strip()}")

    # Bone labels (1=femur, 3=tibia) are now meshed in the same unified SurfaceNets
    # pass as cartilage — no synthesis step needed.
    
    # Read laterality summary if it exists (CT only)
    laterality_summary = {}
    laterality_summary_path = task_mesh_dir / "laterality_summary.json"
    if laterality_summary_path.exists():
        with open(laterality_summary_path, "r") as f:
            laterality_summary = json.load(f)
            
    laterality = laterality_summary.get("laterality", "unknown")
    sides_present = laterality_summary.get("sides_present", ["left", "right"])

    manifest = {
        "task_id": task_id,
        "modality": modality,
        "laterality": laterality,
        "parts": []
    }
    
    missing_files = []
    
    for part in expected_parts:
        # If this part belongs to a side that isn't present in the scan, skip it
        if "side" in part and part["side"] not in sides_present:
            continue
            
        part_path = task_mesh_dir / part["file"]
        if not part_path.exists() or part_path.stat().st_size == 0:
            missing_files.append(part["file"])
            continue
                
        # Validate vertex count (naive check for non-empty OBJ)
        with open(part_path, "r") as f:
            content = f.read()
            if "v " not in content:
                missing_files.append(part["file"] + " (invalid format, missing vertices)")
                continue
                
        manifest["parts"].append(part)
        
    if missing_files:
        raise Exception(f"Validation failed. Missing or invalid meshes (possibly deleted due to QA failure): {', '.join(missing_files)}")
        
    # Write manifest
    with open(task_mesh_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=4)
        
    # 4. Generate Report
    update_status(task_id, "generating_report", modality=modality)
    try:
        report_cmd = [
            python_exe, "backend/report_generator.py",
            task_id, modality, str(file_path), str(task_mesh_dir), str(mask_output)
        ]
        result_report = subprocess.run(report_cmd, capture_output=True, text=True)
        if result_report.returncode != 0:
            raise Exception(f"Report generation script failed: {result_report.stderr.strip()}")
            
        report_pdf = task_mesh_dir / "report.pdf"
        if not report_pdf.exists() or report_pdf.stat().st_size == 0:
            raise Exception("Report generation failed: report.pdf is missing or empty.")
            
        update_status(task_id, "complete", modality=modality, manifest=manifest)
        
    except Exception as e:
        update_status(task_id, "failed", reason=f"report_generation_error: {str(e)}", modality=modality)
        return


def get_file_hash(file_path: str) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def check_cache(file_path: str) -> dict:
    file_hash = get_file_hash(file_path)
    cache_file = CACHE_DIR / f"{file_hash}.json"
    if cache_file.exists():
        with open(cache_file, "r") as f:
            data = json.load(f)
        if data.get("pipeline_version") == PIPELINE_VERSION:
            return {"hit": True, "old_task_id": data["task_id"], "hash": file_hash}
    return {"hit": False, "hash": file_hash}

def write_cache(file_hash: str, task_id: str):
    # Note: cache/*.json entries never expire or get pruned. Fine for POC, 
    # but needs a cleanup policy for long-running production usage.
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{file_hash}.json"
    with open(cache_file, "w") as f:
        json.dump({
            "task_id": task_id,
            "pipeline_version": PIPELINE_VERSION,
            "timestamp": time.time()
        }, f, indent=4)

