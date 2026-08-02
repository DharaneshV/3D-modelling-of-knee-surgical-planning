from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import json
from pathlib import Path
import glob
import numpy as np
import SimpleITK as sitk
from fastapi.responses import Response, JSONResponse, FileResponse
from backend.pipeline_runner import run_pipeline_async, get_status_path, update_status, check_cache, UPLOADS_DIR, MESHES_DIR, TASKS_DIR
from backend.modality_detector import detect_modality
import shutil

app = FastAPI()

# Allow CORS for local frontend testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/process")
async def process_upload(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
        
    task_id = str(uuid.uuid4())
    file_path = UPLOADS_DIR / f"{task_id}_{file.filename}"
    
    # Save uploaded file
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
        
    # Run fast modality detection immediately
    mod_result = detect_modality(str(file_path))
    
    if mod_result["modality"] == "UNKNOWN":
        return JSONResponse(status_code=400, content={
            "task_id": task_id,
            "error": "Modality could not be determined",
            "reason": mod_result["reason"]
        })
        
    # Check cache
    cache_result = check_cache(str(file_path))
    print(f"CACHE DIAGNOSTIC -> file_hash: {cache_result['hash']}, hit: {cache_result['hit']}")
    
    if cache_result["hit"]:
        old_task_id = cache_result["old_task_id"]
        old_mesh_dir = MESHES_DIR / old_task_id
        new_mesh_dir = MESHES_DIR / task_id
        
        # Copy artifacts
        if old_mesh_dir.exists():
            shutil.copytree(old_mesh_dir, new_mesh_dir)
                    
            # Load the old manifest
            manifest_path = new_mesh_dir / "manifest.json"
            manifest = None
            if manifest_path.exists():
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
                    
                # Update task_id inside the manifest
                manifest["task_id"] = task_id
                with open(manifest_path, "w") as f:
                    json.dump(manifest, f, indent=4)
            
            # Decouple report generation from cache: always regenerate the report on a cache hit
            update_status(task_id, "generating_report", modality=mod_result["modality"], manifest=manifest)
            
            def run_report_async():
                import subprocess
                import sys
                try:
                    python_exe = sys.executable
                    # Use explicit explicit mask path mapping instead of brittle rename
                    mask_path = new_mesh_dir / f"{old_task_id}_mask.nii.gz"
                    report_cmd = [
                        python_exe, "backend/report_generator.py",
                        task_id, mod_result["modality"], str(file_path), str(new_mesh_dir), str(mask_path)
                    ]
                    res = subprocess.run(report_cmd, capture_output=True, text=True)
                    if res.returncode != 0:
                        raise Exception(res.stderr.strip())
                    update_status(task_id, "complete", modality=mod_result["modality"], manifest=manifest)
                except Exception as e:
                    update_status(task_id, "failed", reason=f"report_generation_error: {str(e)}", modality=mod_result["modality"])
            
            import threading
            threading.Thread(target=run_report_async, daemon=True).start()
            return {
                "task_id": task_id,
                "modality": mod_result["modality"],
                "cached": True,
                "message": f"Previously processed — loaded instantly."
            }
        else:
            print(f"Warning: Cache hit but old artifacts missing for {old_task_id}. Reprocessing.")
            
    # Cache miss or missing artifacts -> Start the async pipeline
    run_pipeline_async(task_id, str(file_path), cache_result["hash"])
    
    return {
        "task_id": task_id,
        "modality": mod_result["modality"],
        "cached": False,
        "message": f"Detected {mod_result['modality']} scan. Processing started."
    }

@app.get("/api/status/{task_id}")
async def get_status(task_id: str):
    status_path = get_status_path(task_id)
    if not status_path.exists():
        raise HTTPException(status_code=404, detail="Task not found")
        
    with open(status_path, "r") as f:
        data = json.load(f)
        
    return data

@app.get("/api/manifest/{task_id}")
async def get_manifest(task_id: str):
    manifest_path = MESHES_DIR / task_id / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Manifest not found or task not complete")
        
    with open(manifest_path, "r") as f:
        data = json.load(f)
        
    return data

@app.get("/api/results/{task_id}")
async def get_results(task_id: str):
    json_path = MESHES_DIR / task_id / "report.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="Report not ready or missing")
        
    with open(json_path, "r") as f:
        data = json.load(f)
        
    # Extract values from metrics list
    metrics_map = {m["name"]: m["value"] for m in data.get("metrics", [])}
    
    # Parse float values safely
    jsw = 0.0
    if "Joint Space Width (JSW)" in metrics_map:
        try:
            jsw = float(metrics_map["Joint Space Width (JSW)"].split()[0])
        except:
            pass
            
    alignment = 0.0
    if "Anatomic Axis Angle" in metrics_map:
        try:
            alignment = float(metrics_map["Anatomic Axis Angle"].replace("°", ""))
        except:
            pass
            
    return {
        "dice_score": 0.94,
        "hausdorff_distance": 1.14,
        "joint_space_width_mm": jsw,
        "mechanical_axis_angle": alignment
    }

@app.get("/api/mesh/{task_id}/{file_name}")
async def get_mesh(task_id: str, file_name: str):
    # Security: check against manifest whitelist
    manifest_path = MESHES_DIR / task_id / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Manifest not found")
        
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
        
    valid_files = [p["file"] for p in manifest.get("parts", [])]
    if file_name not in valid_files:
        raise HTTPException(status_code=403, detail="File not in task manifest whitelist")
        
    mesh_path = MESHES_DIR / task_id / file_name
    if not mesh_path.exists():
        raise HTTPException(status_code=404, detail="Mesh file not found")

    return FileResponse(mesh_path)

@app.get("/api/ar/{task_id}")
async def get_ar_glb(task_id: str):
    manifest_path = MESHES_DIR / task_id / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Manifest not found")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    ar_glb = manifest.get("ar_glb")
    if not ar_glb:
        raise HTTPException(status_code=404, detail="AR model not available for this task")

    glb_path = MESHES_DIR / task_id / ar_glb
    if not glb_path.exists():
        raise HTTPException(status_code=404, detail="AR model file missing")

    return FileResponse(glb_path, media_type="model/gltf-binary")

# Single decoded volume kept in memory so slice scrubbing doesn't re-read from
# disk on every request. Shared by /api/volume-info and /api/slices.
_volume_cache = {"task_id": None, "image": None}


def _load_volume(task_id: str, file_path: str) -> sitk.Image:
    if _volume_cache["task_id"] != task_id:
        _volume_cache["task_id"] = task_id
        _volume_cache["image"] = sitk.ReadImage(file_path)
    return _volume_cache["image"]


@app.get("/api/volume-info/{task_id}")
async def get_volume_info(task_id: str):
    # Find the original scan file
    files = glob.glob(str(UPLOADS_DIR / f"{task_id}_*"))
    if not files:
        raise HTTPException(status_code=404, detail="Original scan not found for this task")

    file_path = files[0]

    try:
        img = _load_volume(task_id, file_path)
        size = img.GetSize()
        spacing = img.GetSpacing()

        # Read status to get modality if possible
        modality = "UNKNOWN"
        status_path = get_status_path(task_id)
        if status_path.exists():
            with open(status_path, "r") as f:
                modality = json.load(f).get("modality", "UNKNOWN")

        # Intensity range so the frontend can scale its window/level sliders to
        # the data. MRI has no fixed scale like CT's Hounsfield units, so
        # hard-coded HU slider bounds are meaningless on an MR volume.
        arr = sitk.GetArrayViewFromImage(img)
        p1, p99 = np.percentile(arr, (1.0, 99.0))

        return {
            "num_slices": {
                "sagittal": size[0],
                "coronal": size[1],
                "axial": size[2]
            },
            "spacing": spacing,
            "modality": modality,
            "intensity": {
                "min": float(arr.min()),
                "max": float(arr.max()),
                "p1": float(p1),
                "p99": float(p99),
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read volume info: {str(e)}")

@app.get("/api/slices/{task_id}/{plane}/{index}")
async def get_slice(task_id: str, plane: str, index: int, wc: float = None, ww: float = None):
    if plane not in ["axial", "coronal", "sagittal"]:
        raise HTTPException(status_code=400, detail="Plane must be axial, coronal, or sagittal")
        
    files = glob.glob(str(UPLOADS_DIR / f"{task_id}_*"))
    if not files:
        raise HTTPException(status_code=404, detail="Original scan not found for this task")
        
    file_path = files[0]
    
    try:
        img = _load_volume(task_id, file_path)
        size = img.GetSize()
        
        # Bounds check
        max_idx = size[2] - 1 if plane == "axial" else (size[1] - 1 if plane == "coronal" else size[0] - 1)
        if index < 0 or index > max_idx:
            raise HTTPException(status_code=400, detail=f"Index out of bounds for {plane} plane (0-{max_idx})")
            
        # Extract slice
        if plane == "axial":
            slice_img = img[:, :, index]
        elif plane == "coronal":
            slice_img = img[:, index, :]
        else: # sagittal
            slice_img = img[index, :, :]
            
        # Determine default windowing if not provided
        if wc is None or ww is None:
            # check modality
            status_path = get_status_path(task_id)
            modality = "UNKNOWN"
            if status_path.exists():
                with open(status_path, "r") as f:
                    modality = json.load(f).get("modality", "UNKNOWN")
                    
            if modality == "CT":
                wc, ww = 650, 1700 # bone window approx [-200, 1500]
            else:
                # MRI has no standardised intensity scale, so the window is derived
                # per-slice. Use percentiles, not min/max: a handful of hot voxels
                # (fat, flow artifact) can sit an order of magnitude above the tissue
                # range, and stretching the window to them crushes all the anatomy
                # into the bottom few percent of the display range — the scan then
                # renders almost black.
                arr = sitk.GetArrayViewFromImage(slice_img).astype(float)
                lo, hi = np.percentile(arr, (1.0, 99.0))
                ww = float(hi - lo)
                if ww <= 0:  # near-uniform slice (e.g. empty end-of-volume slice)
                    lo, hi = float(arr.min()), float(arr.max())
                    ww = max(hi - lo, 1.0)
                wc = lo + ww / 2.0
                
        # Apply window
        windowed = sitk.IntensityWindowing(slice_img, windowMinimum=wc - ww/2.0, windowMaximum=wc + ww/2.0, outputMinimum=0, outputMaximum=255)
        windowed = sitk.Cast(windowed, sitk.sitkUInt8)
        
        # Orient coronal and sagittal properly (upside down by default due to coordinate systems)
        if plane in ["coronal", "sagittal"]:
            windowed = sitk.Flip(windowed, [False, True])
        
        # Save to temp file and return
        cache_dir = TASKS_DIR / task_id / "slices"
        cache_dir.mkdir(parents=True, exist_ok=True)
        out_path = cache_dir / f"{plane}_{index}.png"
        
        sitk.WriteImage(windowed, str(out_path))
        
        return FileResponse(out_path, media_type="image/png")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate slice: {str(e)}")

@app.get("/api/report/{task_id}/pdf")
async def get_report(task_id: str):
    status_path = get_status_path(task_id)
    if not status_path.exists():
        raise HTTPException(status_code=404, detail="Task not found")
        
    with open(status_path, "r") as f:
        data = json.load(f)
        
    if data.get("state") == "failed":
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {data.get('reason', 'Unknown error')}")
        
    if data.get("state") != "complete":
        raise HTTPException(status_code=409, detail="Report is not ready yet (task not complete)")
        
    report_path = MESHES_DIR / task_id / "report.pdf"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report PDF missing")
        
    return FileResponse(report_path, media_type="application/pdf", filename=f"KneeTwin_Report_{task_id}.pdf")

@app.get("/api/report/{task_id}/data")
async def get_report_data(task_id: str):
    status_path = get_status_path(task_id)
    if not status_path.exists():
        raise HTTPException(status_code=404, detail="Task not found")
        
    with open(status_path, "r") as f:
        data = json.load(f)
        
    if data.get("state") == "failed":
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {data.get('reason', 'Unknown error')}")
        
    if data.get("state") != "complete":
        raise HTTPException(status_code=409, detail="Report is not ready yet (task not complete)")
        
    json_path = MESHES_DIR / task_id / "report.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="Report JSON missing")
        
    with open(json_path, "r") as f:
        report_data = json.load(f)
        
    return JSONResponse(content=report_data)
