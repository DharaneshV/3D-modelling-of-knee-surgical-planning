from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import json
from pathlib import Path
from backend.pipeline_runner import run_pipeline_async, get_status_path, UPLOADS_DIR, MESHES_DIR
from backend.modality_detector import detect_modality

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
        
    # Start the async pipeline
    run_pipeline_async(task_id, str(file_path))
    
    return {
        "task_id": task_id,
        "modality": mod_result["modality"],
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
    # Dummy scores for POC (would normally be parsed from calculate_clinical_measurements.py output)
    return {
        "dice_score": 0.92,
        "hausdorff_distance": 1.14,
        "joint_space_width_mm": 4.5,
        "mechanical_axis_angle": 1.2
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
