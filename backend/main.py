from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import json
import threading
from pathlib import Path
import glob
import numpy as np
import SimpleITK as sitk
from backend.pipeline_runner import (
    run_pipeline_async, get_status_path, update_status, check_cache,
    UPLOADS_DIR, MESHES_DIR, TASKS_DIR, REPORT_TIMEOUT_S, atomic_write_json,
)
from backend.modality_detector import detect_modality
import shutil

app = FastAPI()

# Same-origin only. The frontend is served by this app (see the StaticFiles
# mount at the bottom of this file), so no page legitimately needs to call this
# API cross-origin. allow_origins=["*"] with allow_credentials=True was also an
# invalid combination per the Fetch spec (browsers reject a wildcard origin
# once credentials are involved) — a leftover from an earlier two-port dev
# setup (frontend on :8090, backend on :8000) that no longer exists. Same-origin
# requests are never subject to CORS regardless of this config, so the normal
# app flow (including via the LAN IP or a tunnel domain) is unaffected.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _validate_task_id(task_id: str) -> None:
    """
    Task IDs are our own uuid4() tokens (backend/main.py mints them with
    str(uuid.uuid4())). Reject anything else before it reaches a path join,
    rather than trusting every downstream MESHES_DIR/UPLOADS_DIR lookup to fail
    safely on a malformed value.
    """
    try:
        uuid.UUID(task_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid task_id")


# Per-task locks for /api/resect. Two concurrent calls for the SAME task_id
# both write tibia_resected.obj, femur_resected.obj, tibial_tray.obj,
# femoral_component.obj and the same {task_id}_resected_ar.glb path — trimesh's
# .export() writes straight to the destination with no temp-file-then-rename,
# so a concurrent reader could observe a torn file, and the two responses'
# JSON could disagree with whatever actually landed on disk.
_resect_locks: dict[str, threading.Lock] = {}
_resect_locks_guard = threading.Lock()


def _acquire_resect_lock(task_id: str) -> bool:
    """
    True if this call now owns the lock for task_id (must release it when
    done); False if another resection for this task is already in flight.

    Deliberately non-blocking: queueing a second caller behind an already
    slow (~24s) resection would just reproduce a milder version of the
    problem this exists to fix — a client left waiting with no feedback — so
    a concurrent call is rejected outright with 409 rather than made to wait.
    """
    with _resect_locks_guard:
        lock = _resect_locks.setdefault(task_id, threading.Lock())
    return lock.acquire(blocking=False)


def _release_resect_lock(task_id: str) -> None:
    _resect_locks[task_id].release()


def _require_complete_status(task_id: str) -> dict:
    """
    Shared by every route that only makes sense once a task's pipeline has
    finished (currently the two /api/report/* routes, which duplicated this
    exact "load status.json, 404/500/409" block near-verbatim). Also the one
    place that needs to handle a torn read on status.json — get_status_path's
    file is now written atomically (see atomic_write_json in
    pipeline_runner.py), so a reader can no longer observe a truncated write,
    but a genuinely corrupt or unreadable file is still handled explicitly
    rather than raising an unhandled 500 from json.load.
    """
    status_path = get_status_path(task_id)
    if not status_path.exists():
        raise HTTPException(status_code=404, detail="Task not found")

    try:
        with open(status_path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        raise HTTPException(status_code=500, detail="Task status is unreadable") from e

    if data.get("state") == "failed":
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {data.get('reason', 'Unknown error')}")

    if data.get("state") != "complete":
        raise HTTPException(status_code=409, detail="Report is not ready yet (task not complete)")

    return data


def _atomic_export(mesh, path: Path) -> None:
    """
    trimesh's mesh.export() writes straight to the destination path with no
    temp-file-then-rename, so a GET on /api/mesh/{task_id}/{file} concurrent
    with a resect call (the per-task lock only serializes writers against each
    other, not readers) could observe a truncated .obj mid-write. Export to a
    sibling temp file and os.replace() into place instead, same pattern as
    atomic_write_json in pipeline_runner.py.

    The uniqueness token goes before the extension, not after: trimesh infers
    the export format from the file suffix, so a temp name like
    "tibia_resected.obj.tmp1234" (no trailing .obj) makes export() fail with
    "exporter not available" — confirmed live before switching to this order.
    """
    tmp_path = path.with_name(f"{path.stem}.tmp{os.getpid()}{path.suffix}")
    mesh.export(str(tmp_path))
    os.replace(tmp_path, path)


# Generous enough for a full CT/MRI series (typically tens to low hundreds of
# MB) while stopping an unbounded upload from exhausting disk or, since the
# old code read the whole body into memory before writing, RAM.
MAX_UPLOAD_BYTES = 500 * 1024 * 1024


@app.post("/api/process")
async def process_upload(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    # .name strips every directory component (forward slash, backslash, ..,
    # drive letters, UNC prefixes — verified on all of those on this Windows
    # deployment) leaving only the final path segment. file.filename is fully
    # attacker-controlled by the multipart client; unsanitized, a filename
    # like "../../../backend/main.py" resolved to a write straight into the
    # live source tree, and StaticFiles serves frontend/ from this same app.
    safe_filename = Path(file.filename).name
    if not safe_filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    task_id = str(uuid.uuid4())
    file_path = UPLOADS_DIR / f"{task_id}_{safe_filename}"

    # Stream in bounded chunks rather than buffering the whole upload into
    # memory with one read(), and enforce the size cap while writing so a
    # too-large upload is rejected (and its partial file removed) instead of
    # silently consuming unbounded disk.
    try:
        size = 0
        with open(file_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit")
                buffer.write(chunk)
    except HTTPException:
        file_path.unlink(missing_ok=True)
        raise
    except OSError as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Failed to save upload") from e

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
                atomic_write_json(manifest_path, manifest)
            
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
                    res = subprocess.run(report_cmd, capture_output=True, text=True,
                                         timeout=REPORT_TIMEOUT_S)
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
    _validate_task_id(task_id)
    status_path = get_status_path(task_id)
    if not status_path.exists():
        raise HTTPException(status_code=404, detail="Task not found")
        
    with open(status_path, "r") as f:
        data = json.load(f)
        
    return data

@app.get("/api/manifest/{task_id}")
async def get_manifest(task_id: str):
    _validate_task_id(task_id)
    manifest_path = MESHES_DIR / task_id / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Manifest not found or task not complete")
        
    with open(manifest_path, "r") as f:
        data = json.load(f)
        
    return data

@app.get("/api/results/{task_id}")
async def get_results(task_id: str):
    _validate_task_id(task_id)
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
        # Segmentation accuracy needs ground truth to compare against, which a
        # live upload does not have — these were previously hardcoded to
        # 0.94/1.14 regardless of the actual case, a fabricated-looking number
        # for a tool that otherwise refuses to publish clinical figures it
        # hasn't earned. Accuracy is only knowable in aggregate, from the
        # offline OAI-ZIB validation (results/dice_scores_summary.csv,
        # summarized in MODEL_CARD.md) — not as a per-case number.
        "dice_score": None,
        "hausdorff_distance": None,
        "joint_space_width_mm": jsw,
        "mechanical_axis_angle": alignment
    }

@app.get("/api/mesh/{task_id}/{file_name}")
async def get_mesh(task_id: str, file_name: str):
    _validate_task_id(task_id)
    # Security: check against manifest whitelist
    manifest_path = MESHES_DIR / task_id / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Manifest not found")
        
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
        
    valid_files = [p["file"] for p in manifest.get("parts", [])]
    # Resection output is generated on demand by /api/resect, so it is not in the
    # manifest; allow the fixed set of names that endpoint writes.
    valid_files += ["femur_resected.obj", "tibia_resected.obj",
                    "tibial_tray.obj", "femoral_component.obj"]
    if file_name not in valid_files:
        raise HTTPException(status_code=403, detail="File not in task manifest whitelist")
        
    mesh_path = MESHES_DIR / task_id / file_name
    if not mesh_path.exists():
        raise HTTPException(status_code=404, detail="Mesh file not found")

    return FileResponse(mesh_path)

@app.get("/api/ar/{task_id}")
async def get_ar_glb(task_id: str):
    _validate_task_id(task_id)
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


@app.get("/api/resect/{task_id}")
def resect_bones(task_id: str,
                 femur_depth_mm: float = 9.0,
                 tibia_depth_mm: float = 10.0,
                 varus_deg: float = 0.0,
                 slope_deg: float = 0.0):
    """
    Plan the distal-femoral and proximal-tibial resections for a task.

    Writes the cut meshes alongside the originals and returns the measurements a
    surgeon would size a component from. Only the MRI track is supported: the
    label names below are CartiMorph's.

    Deliberately a plain `def`, not `async def`: the body below is pure
    synchronous CPU work (trimesh/pyvista, no I/O to await), and an `async def`
    with no `await` inside runs directly on the single event loop, blocking
    EVERY other request on the server — not just other resect calls — for the
    full ~24s duration. Verified directly: with this handler still `async def`,
    a concurrent /api/manifest call for a different, unrelated task queued
    behind it and timed out entirely. A plain `def` tells Starlette to run it
    in its threadpool instead, so other requests are served concurrently.
    """
    _validate_task_id(task_id)

    import trimesh
    from src.mesh.resection import limb_axis, plan_resection
    from src.mesh.implant import fit_femoral_component, fit_tibial_tray
    from src.mesh.export_ar_glb import export_ar_glb_from_meshes

    task_dir = MESHES_DIR / task_id
    femur_path = task_dir / "femur_unknown.obj"
    tibia_path = task_dir / "tibia_unknown.obj"

    if not femur_path.exists() or not tibia_path.exists():
        raise HTTPException(status_code=404,
                            detail="Femur and tibia meshes not found for this task")

    if not (0 < femur_depth_mm < 40 and 0 < tibia_depth_mm < 40):
        raise HTTPException(status_code=400,
                            detail="Resection depth must be between 0 and 40 mm")
    if abs(varus_deg) > 15 or abs(slope_deg) > 15:
        raise HTTPException(status_code=400,
                            detail="Angulation must be within +/-15 degrees")

    # Now in the threadpool, a second call for this same task_id could
    # genuinely run concurrently with this one in a different worker thread —
    # guard the file-writing section against that.
    if not _acquire_resect_lock(task_id):
        raise HTTPException(status_code=409,
                            detail="A resection is already being planned for this task")

    try:
        femur = trimesh.load_mesh(str(femur_path), process=False)
        tibia = trimesh.load_mesh(str(tibia_path), process=False)
        axis = limb_axis(femur.vertices, tibia.vertices)

        results = {}
        implants = {}
        ar_parts = {}
        ar_colors = {"femur_unknown": "#e74c3c", "tibia_unknown": "#2ecc71",
                     "femoral_component": "#c8d0dc", "tibial_tray": "#c8d0dc"}

        # Tibia: a single proximal cut, then a tray sized to it.
        tibia_plan = plan_resection(tibia, axis, "tibia", depth_mm=tibia_depth_mm,
                                    varus_deg=varus_deg, slope_deg=slope_deg)
        tibia_cut = tibia_plan.pop("mesh")
        _atomic_export(tibia_cut, task_dir / "tibia_resected.obj")
        results["tibia"] = tibia_plan
        ar_parts["tibia_unknown"] = tibia_cut

        try:
            fit = fit_tibial_tray(tibia_plan, tibia_cut)
            tray = fit.pop("mesh")
            _atomic_export(tray, task_dir / "tibial_tray.obj")
            ar_parts["tibial_tray"] = tray
            implants["tibial"] = fit
        except Exception as e:
            print(f"Tibial tray fitting failed for {task_id}: {e}")

        # Femur: the component dictates the preparation, so sizing comes first
        # and the five box cuts follow from it. A single distal plane would not
        # give the component anything to seat against.
        try:
            fem = fit_femoral_component(femur, axis, distal_depth_mm=femur_depth_mm)
            femur_cut = fem.pop("prepared_femur")
            component = fem.pop("mesh")
            _atomic_export(femur_cut, task_dir / "femur_resected.obj")
            _atomic_export(component, task_dir / "femoral_component.obj")
            ar_parts["femur_unknown"] = femur_cut
            ar_parts["femoral_component"] = component
            implants["femoral"] = fem
            results["femur"] = {
                "preparation": "five-cut box",
                "distal_depth_mm": femur_depth_mm,
                "component_size": fem["size"],
            }
        except Exception as e:
            # Fall back to the plain distal cut so the resection is still usable.
            print(f"Femoral box preparation failed for {task_id}: {e}")
            femur_plan = plan_resection(femur, axis, "femur", depth_mm=femur_depth_mm,
                                        varus_deg=varus_deg, slope_deg=slope_deg)
            femur_cut = femur_plan.pop("mesh")
            _atomic_export(femur_cut, task_dir / "femur_resected.obj")
            ar_parts["femur_unknown"] = femur_cut
            results["femur"] = femur_plan

        # AR-ready GLB of the resected state, same transform as the intact export.
        glb_name = f"{task_id}_resected_ar.glb"
        export_ar_glb_from_meshes(ar_parts, str(task_dir / glb_name),
                                  {k: ar_colors[k] for k in ar_parts})

        return {
            "task_id": task_id,
            "limb_axis": [round(float(x), 4) for x in axis],
            "axis_note": ("Limb-axis proxy from bone centroids. A knee-only field "
                          "of view contains no hip or ankle centre, so this is not "
                          "a mechanical axis."),
            "resections": results,
            "implants": implants,
            "implant_note": ("Generic parametric components, not specific commercial "
                             "implants. Indicates that components of these sizes fit "
                             "this anatomy; it is not a product selection."),
            "ar_glb": glb_name,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Resection failed: {str(e)}")
    finally:
        _release_resect_lock(task_id)


@app.get("/api/volume-info/{task_id}")
async def get_volume_info(task_id: str):
    _validate_task_id(task_id)
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
    _validate_task_id(task_id)
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
    _validate_task_id(task_id)
    _require_complete_status(task_id)

    report_path = MESHES_DIR / task_id / "report.pdf"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report PDF missing")

    return FileResponse(report_path, media_type="application/pdf", filename=f"KneeTwin_Report_{task_id}.pdf")

@app.get("/api/report/{task_id}/data")
async def get_report_data(task_id: str):
    _validate_task_id(task_id)
    _require_complete_status(task_id)

    json_path = MESHES_DIR / task_id / "report.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="Report JSON missing")
        
    with open(json_path, "r") as f:
        report_data = json.load(f)
        
    return JSONResponse(content=report_data)


# Serve the frontend from this same app. Mounted last so it does not shadow the
# /api routes above — Starlette matches in definition order.
#
# One origin matters for phone access: the frontend used to hardcode
# localhost:8000, which on a phone means the phone itself, so every API call
# failed. Same-origin also means a single port to expose, which is what makes
# an HTTPS tunnel practical — and WebXR will not start without HTTPS.
from fastapi.staticfiles import StaticFiles  # noqa: E402

_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if _FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")
