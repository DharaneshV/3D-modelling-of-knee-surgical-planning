# KneeTwin Implementation Plan v2 — MRI-Only Pipeline

**Supersedes:** the 9-week CT+MRI dual-track plan.
**Core change:** per-patient CT is dropped. Bone is synthesized from MRI cartilage geometry, using a shape model trained once, offline, on your existing paired CT+MRI cases.

Status tags used throughout:
- ✅ **Done** — already built and working, per current pipeline state. Reuse as-is.
- ⚠️ **Done but needs rewiring** — code exists but assumes the old CT-per-case flow; needs to be repointed.
- 🔲 **Not built** — net-new work for this plan.
- ❌ **Cut** — remove from the pipeline entirely; do not port forward.

---

## 0. Decisions Locked

| Decision | Detail |
|---|---|
| Per-case modality | MRI only. No CT ingestion, no CT-MRI registration, at inference time. |
| Bone generation | Synthetic — Statistical Shape Model (PCA), conditioned on MRI-measured cartilage landmarks. Not a learned network (insufficient paired data). |
| Role of CT | Training data only, consumed once, offline, to build the shape model. Never touched again per-case. |
| Meshing | Single unified multi-label `vtkSurfaceNets3D` pass across cartilage + synthesized bone labels, Taubin-smoothed. |
| Pitch framing | "MRI-only surgical planning — cartilage measured directly, bone synthesized and scaled to it, no radiation, single scan, single visit." |

**⚠️ Offline training dataset:** of the 7 paired CT-MRI cases, only DU02, DU03, and DU06 have confirmed valid registration. DU01 is a post-surgical TKA case (excluded — non-representative anatomy), DU04/DU05 are missing CT (excluded — no pair to train on), and DU07 failed due to input segmentation failure (TotalSegmentator missed the femur due to cropped FOV). **Effective usable training set is 3 cases**. Section A.7 below reflects this.

---

## Track 1 — Per-Case Pipeline (production, runs on every new patient, MRI only)

### Step 1 — MRI Ingestion & Cartilage/Bone-Region Segmentation
- ✅ CartiMorph segmentation — label semantics confirmed (`Background=0, Femur=1, Femoral Cartilage=2, Tibia=3, MTiC=4, LTiC=5`)
- ✅ RAI reorientation fix — required for CartiMorph to segment correctly, already applied
- ✅ Segmentation collapse gate — QA safeguard against CartiMorph failure modes
- 🔲 ACL/PCL/meniscus segmentation — CartiMorph doesn't cover these; still needs the pre-trained U-Net / Swin UNETR track from the original spec (Section 3). Not started.
- ❌ Any CT ingestion, N4-skip / HU-threshold preprocessing for CT — remove Track B entirely from the per-case path.

### Step 2 — Bone Synthesis (replaces CT bone segmentation)
- 🔲 `synthesize_bone()` call into the per-case pipeline — **blocked on Track 2 (offline training) completing first.** This is the single largest gap between the current pipeline and this plan.
- Inference QA gate (Mahalanobis distance vs. training distribution, reject + flag if >3σ) — 🔲 not built, spec'd in A.6.

### Step 3 — Unified Meshing
- ✅ Migrated to unified multi-label `vtkSurfaceNets3D` meshing (already replaces old per-label marching-cubes approach that caused joint-boundary overlaps)
- ✅ 1-voxel-gap smoothing bulge bug fixed
- ⚠️ Needs rewiring: currently meshes cartilage + CT-derived bone labels together; repoint to mesh cartilage + **synthesized** bone labels together once Step 2 exists. No change to the meshing code itself, only its inputs.
- 🔲 Confirm Taubin smoothing (vs. default Laplacian) is applied post-mesh — carry forward from original spec if not already in the current pipeline; Laplacian will corrupt implant-sizing measurements.

### Step 4 — Measurements
- ✅ JSW metric (ROI-cropped boundary erosion + cKDTree, anisotropic surface projection, ~9.6s/side) — valid on the MRI track, keep as-is
- ❌ CT-track JSW — already disabled (correctly; CT can't image cartilage), no action needed
- ✅ Fabricated cartilage metric removed
- ✅ Tibia AP depth fix (10%/30mm ROI + 2D PCA axis recovery, replacing full-shaft measurement)
- ✅ Bilateral knee support (laterality-aware label maps, centroid-distance splitting) — carries forward unchanged, MRI-only doesn't affect this

### Step 5 — Reporting & AR
- ✅ FastAPI backend with modality auto-detection and background task processing — ⚠️ modality auto-detection can be simplified/removed since only MRI is accepted per-case now; decide whether to keep as a validation guard (reject CT uploads with a clear message) or strip it
- ✅ Three.js 3D viewer (three-column layout: original scan, left knee, right knee)
- ✅ Batch QA system (`batch_qa_report.py`) with quarantine logging and clinical PDF warnings
- ✅ Caching architecture (SHA-256 hash + `PIPELINE_VERSION` as cache key)
- ✅ PDF clinical report pipeline via ReportLab

### Cut from the per-case path entirely
- ❌ CT ingestion (Track B from original Week 1)
- ❌ CT bone segmentation via TotalSegmentator, per-case
- ❌ CT-MRI rigid/affine registration (original Week 5)
- ❌ Metal streak artifact check, HU-threshold QA gate — these were CT-specific

---

## Track 2 — Offline Bone-Shape-Model Training (one-time, run before Track 1 Step 2 works)

This is new work. Nothing here has been built yet, but two pieces of existing infrastructure carry over directly:
- ✅ GPU-accelerated TotalSegmentator inference (RTX 4050, CUDA 12.1, ~4 min/case) — use this to produce the bone meshes for the 3 training cases.
- ✅ `vtkSurfaceNets3D` unified meshing — reuse for producing clean bone mesh surfaces from the training CT labels before correspondence/PCA.

### A.1 Component Summary
Produces a PCA shape space over bone meshes plus a ridge regressor mapping MRI-measured cartilage landmarks → shape coefficients. At inference, `synthesize_bone()` predicts coefficients from a new patient's cartilage mask and deforms the mean bone shape accordingly.

### A.2 Inputs
| Input | Source | Notes |
|---|---|---|
| CT bone meshes (training) | DU02, DU03, DU06 | Generate via existing TotalSegmentator + vtkSurfaceNets3D pipeline |
| MRI cartilage segmentation (training + inference) | CartiMorph output, resampled to 0.5mm isotropic | Labels: femoral cartilage, MTiC, LTiC |
| Cartilage landmark vector (training + inference) | Extracted from cartilage mask, see A.4 | 9-dim per-case feature vector |

### A.3 Training Procedure
**Step 1 — Mesh correspondence.** Register all training bone meshes into point-to-point correspondence via non-rigid ICP (e.g. `probreg` CPD) against a single reference case. 🔲 not built.

**Step 2 — PCA shape space.** Stack aligned vertex matrices, fit PCA. With only 3–4 cases, expect at most 2–3 usable components (not the 6 assumed when 7 cases were in scope) — retain components explaining ≥95% variance and check that number honestly once correspondence is done. 🔲 not built.

**Step 3 — Landmark regressor.** Ridge regression from the 9-dim cartilage feature vector to PCA coefficients, fit on the same 3–4 cases. 🔲 not built.

**QA gate (training):** LOO-CV mean-surface-distance ≤3.0mm (femur) / ≤2.5mm (tibia). With N=3, LOO-CV is a weak signal — flag this explicitly in the model card rather than presenting it as a validated error bound.

### A.4 Cartilage Landmark Feature Vector (9-dim)
| # | Feature | Computation |
|---|---|---|
| 1 | Femoral cartilage volume (cm³) | `sum(fc_mask) * vox_vol` |
| 2 | Femoral cartilage mean thickness (mm) | `volume / (surface_area / 2)` |
| 3 | Femoral contact area ML-extent (mm) | bbox X-width of `fc_mask>0`, physical coords |
| 4 | Medial tibial cartilage volume (cm³) | same method |
| 5 | Medial tibial cartilage mean thickness (mm) | same method |
| 6 | Lateral tibial cartilage volume (cm³) | same method |
| 7 | Lateral tibial cartilage mean thickness (mm) | same method |
| 8 | JSW — medial (mm) | existing `get_jsw.py`, medial compartment |
| 9 | JSW — lateral (mm) | existing `get_jsw.py`, lateral compartment |

All z-scored before regression; save the scaler.

### A.5 Inference Interface
```python
# src/synthesis/bone_from_mri.py
def synthesize_bone(cart_mask_path: str, bone: str = 'femur') -> pv.PolyData:
    """
    Args:
        cart_mask_path: CartiMorph output .nii.gz (0.5mm isotropic)
        bone: 'femur' or 'tibia'
    Returns:
        pv.PolyData synthetic bone mesh, physical (LPS) coords.
        Drop-in input to the existing vtkSurfaceNets3D meshing step.
    Raises:
        ReconstructionQAError: if Mahalanobis distance > 3.0σ (A.6)
    """
    features = extract_cart_features(cart_mask_path)
    features_scaled = scaler.transform(features.reshape(1, -1))
    coeffs = regressor.predict(features_scaled)
    flat_verts = pca_model.mean_ + coeffs @ pca_model.components_
    return flat_to_mesh(flat_verts.squeeze(), template_mesh)
```

### A.6 Inference QA Gate
Reject and flag (don't silently extrapolate) if Mahalanobis distance of predicted coefficients exceeds 3σ from the training distribution:
```
Synthetic bone rejected: Mahalanobis distance = {d:.2f} (threshold 3.0σ).
Patient anatomy is outside the training cohort range — flag for manual review.
```

### A.7 Minimum Viable Case Count — revised
| Condition | Cases needed | Status |
|---|---|---|
| SSM/PCA training | 3 minimum, more is better | ⚠️ Have 3 confirmed (DU02, DU03, DU06) |
| Reliable LOO-CV | ~5+ | ❌ Not yet met — treat current LOO-CV numbers as directional, not a validated bound |
| Learned MRI→bone network | ~30–50 paired cases | ❌ Not in scope |

**Action before Track 2 starts:** DU07 classification is complete (excluded due to CT segmentation failure). Training set is 3. Document the LOO-CV caveat in the model card rather than presenting it as equivalent to the 7-case estimate from the earlier draft.

### A.8 Artifact Locations
```
src/synthesis/
    bone_from_mri.py
    train_ssm.py
    extract_cart_features.py
    MODEL_CARD.md          # method rationale, QA thresholds, N=3/4 caveat
models/ssm/
    pca_model_femur.pkl
    pca_model_tibia.pkl
    shape_regressor_femur.pkl
    shape_regressor_tibia.pkl
    cart_feature_scaler.pkl
    template_femur.obj
    template_tibia.obj
# KneeTwin Implementation Plan v2 — MRI-Only Pipeline

**Supersedes:** the 9-week CT+MRI dual-track plan.
**Core change:** per-patient CT is dropped. Bone is synthesized from MRI cartilage geometry, using a shape model trained once, offline, on your existing paired CT+MRI cases.

Status tags used throughout:
- ✅ **Done** — already built and working, per current pipeline state. Reuse as-is.
- ⚠️ **Done but needs rewiring** — code exists but assumes the old CT-per-case flow; needs to be repointed.
- 🔲 **Not built** — net-new work for this plan.
- ❌ **Cut** — remove from the pipeline entirely; do not port forward.

---

## 0. Decisions Locked

| Decision | Detail |
|---|---|
| Per-case modality | MRI only. No CT ingestion, no CT-MRI registration, at inference time. |
| Bone generation | Synthetic — Statistical Shape Model (PCA), conditioned on MRI-measured cartilage landmarks. Not a learned network (insufficient paired data). |
| Role of CT | Training data only, consumed once, offline, to build the shape model. Never touched again per-case. |
| Meshing | Single unified multi-label `vtkSurfaceNets3D` pass across cartilage + synthesized bone labels, Taubin-smoothed. |
| Pitch framing | "MRI-only surgical planning — cartilage measured directly, bone synthesized and scaled to it, no radiation, single scan, single visit." |

**⚠️ Offline training dataset:** of the 7 paired CT-MRI cases, only DU02, DU03, and DU06 have confirmed valid registration. DU01 is a post-surgical TKA case (excluded — non-representative anatomy), DU04/DU05 are missing CT (excluded — no pair to train on), and DU07 failed due to input segmentation failure (TotalSegmentator missed the femur due to cropped FOV). **Effective usable training set is 3 cases**. Section A.7 below reflects this.

---

## Track 1 — Per-Case Pipeline (production, runs on every new patient, MRI only)

### Step 1 — MRI Ingestion & Cartilage/Bone-Region Segmentation
- ✅ CartiMorph segmentation — label semantics confirmed (`Background=0, Femur=1, Femoral Cartilage=2, Tibia=3, MTiC=4, LTiC=5`)
- ✅ RAI reorientation fix — required for CartiMorph to segment correctly, already applied
- ✅ Segmentation collapse gate — QA safeguard against CartiMorph failure modes
- 🔲 ACL/PCL/meniscus segmentation — CartiMorph doesn't cover these; still needs the pre-trained U-Net / Swin UNETR track from the original spec (Section 3). Not started.
- ❌ Any CT ingestion, N4-skip / HU-threshold preprocessing for CT — remove Track B entirely from the per-case path.

### Step 2 — Bone Synthesis (replaces CT bone segmentation)
- 🔲 `synthesize_bone()` call into the per-case pipeline — **blocked on Track 2 (offline training) completing first.** This is the single largest gap between the current pipeline and this plan.
- Inference QA gate (Mahalanobis distance vs. training distribution, reject + flag if >3σ) — 🔲 not built, spec'd in A.6.

### Step 3 — Unified Meshing
- ✅ Migrated to unified multi-label `vtkSurfaceNets3D` meshing (already replaces old per-label marching-cubes approach that caused joint-boundary overlaps)
- ✅ 1-voxel-gap smoothing bulge bug fixed
- ⚠️ Needs rewiring: currently meshes cartilage + CT-derived bone labels together; repoint to mesh cartilage + **synthesized** bone labels together once Step 2 exists. No change to the meshing code itself, only its inputs.
- 🔲 Confirm Taubin smoothing (vs. default Laplacian) is applied post-mesh — carry forward from original spec if not already in the current pipeline; Laplacian will corrupt implant-sizing measurements.

### Step 4 — Measurements
- ✅ JSW metric (ROI-cropped boundary erosion + cKDTree, anisotropic surface projection, ~9.6s/side) — valid on the MRI track, keep as-is
- ❌ CT-track JSW — already disabled (correctly; CT can't image cartilage), no action needed
- ✅ Fabricated cartilage metric removed
- ✅ Tibia AP depth fix (10%/30mm ROI + 2D PCA axis recovery, replacing full-shaft measurement)
- ✅ Bilateral knee support (laterality-aware label maps, centroid-distance splitting) — carries forward unchanged, MRI-only doesn't affect this

### Step 5 — Reporting & AR
- ✅ FastAPI backend with modality auto-detection and background task processing — ⚠️ modality auto-detection can be simplified/removed since only MRI is accepted per-case now; decide whether to keep as a validation guard (reject CT uploads with a clear message) or strip it
- ✅ Three.js 3D viewer (three-column layout: original scan, left knee, right knee)
- ✅ Batch QA system (`batch_qa_report.py`) with quarantine logging and clinical PDF warnings
- ✅ Caching architecture (SHA-256 hash + `PIPELINE_VERSION` as cache key)
- ✅ PDF clinical report pipeline via ReportLab

### Cut from the per-case path entirely
- ❌ CT ingestion (Track B from original Week 1)
- ❌ CT bone segmentation via TotalSegmentator, per-case
- ❌ CT-MRI rigid/affine registration (original Week 5)
- ❌ Metal streak artifact check, HU-threshold QA gate — these were CT-specific

---

## Track 2 — Offline Bone-Shape-Model Training (one-time, run before Track 1 Step 2 works)

This is new work. Nothing here has been built yet, but two pieces of existing infrastructure carry over directly:
- ✅ GPU-accelerated TotalSegmentator inference (RTX 4050, CUDA 12.1, ~4 min/case) — use this to produce the bone meshes for the 3 training cases.
- ✅ `vtkSurfaceNets3D` unified meshing — reuse for producing clean bone mesh surfaces from the training CT labels before correspondence/PCA.

### A.1 Component Summary
Produces a PCA shape space over bone meshes plus a ridge regressor mapping MRI-measured cartilage landmarks → shape coefficients. At inference, `synthesize_bone()` predicts coefficients from a new patient's cartilage mask and deforms the mean bone shape accordingly.

### A.2 Inputs
| Input | Source | Notes |
|---|---|---|
| CT bone meshes (training) | DU02, DU03, DU06 | Generate via existing TotalSegmentator + vtkSurfaceNets3D pipeline |
| MRI cartilage segmentation (training + inference) | CartiMorph output, resampled to 0.5mm isotropic | Labels: femoral cartilage, MTiC, LTiC |
| Cartilage landmark vector (training + inference) | Extracted from cartilage mask, see A.4 | 9-dim per-case feature vector |

### A.3 Training Procedure
**Step 1 — Mesh correspondence.** Register all training bone meshes into point-to-point correspondence via non-rigid ICP (e.g. `probreg` CPD) against a single reference case. 🔲 not built.

**Step 2 — PCA shape space.** Stack aligned vertex matrices, fit PCA. With only 3 cases, expect at most 2 usable components — retain components explaining ≥95% variance and check that number honestly once correspondence is done. 🔲 not built.

**Step 3 — Landmark regressor.** Ridge regression from the 9-dim cartilage feature vector to PCA coefficients, fit on the same 3 cases. 🔲 not built.

**QA gate (training):** LOO-CV mean-surface-distance ≤3.0mm (femur) / ≤2.5mm (tibia). With N=3, LOO-CV is a weak signal — flag this explicitly in the model card rather than presenting it as a validated error bound.

### A.4 Cartilage Landmark Feature Vector (9-dim)
| # | Feature | Computation |
|---|---|---|
| 1 | Femoral cartilage volume (cm³) | `sum(fc_mask) * vox_vol` |
| 2 | Femoral cartilage mean thickness (mm) | `volume / (surface_area / 2)` |
| 3 | Femoral contact area ML-extent (mm) | bbox X-width of `fc_mask>0`, physical coords |
| 4 | Medial tibial cartilage volume (cm³) | same method |
| 5 | Medial tibial cartilage mean thickness (mm) | same method |
| 6 | Lateral tibial cartilage volume (cm³) | same method |
| 7 | Lateral tibial cartilage mean thickness (mm) | same method |
| 8 | JSW — medial (mm) | existing `get_jsw.py`, medial compartment |
| 9 | JSW — lateral (mm) | existing `get_jsw.py`, lateral compartment |

All z-scored before regression; save the scaler.

### A.5 Inference Interface
```python
# src/synthesis/bone_from_mri.py
def synthesize_bone(cart_mask_path: str, bone: str = 'femur') -> pv.PolyData:
    """
    Args:
        cart_mask_path: CartiMorph output .nii.gz (0.5mm isotropic)
        bone: 'femur' or 'tibia'
    Returns:
        pv.PolyData synthetic bone mesh, physical (LPS) coords.
        Drop-in input to the existing vtkSurfaceNets3D meshing step.
    Raises:
        ReconstructionQAError: if Mahalanobis distance > 3.0σ (A.6)
    """
    features = extract_cart_features(cart_mask_path)
    features_scaled = scaler.transform(features.reshape(1, -1))
    coeffs = regressor.predict(features_scaled)
    flat_verts = pca_model.mean_ + coeffs @ pca_model.components_
    return flat_to_mesh(flat_verts.squeeze(), template_mesh)
```

### A.6 Inference QA Gate
Reject and flag (don't silently extrapolate) if Mahalanobis distance of predicted coefficients exceeds 3σ from the training distribution:
```
Synthetic bone rejected: Mahalanobis distance = {d:.2f} (threshold 3.0σ).
Patient anatomy is outside the training cohort range — flag for manual review.
```

### A.7 Minimum Viable Case Count — revised
| Condition | Cases needed | Status |
|---|---|---|
| SSM/PCA training | 3 minimum, more is better | ✅ Have 3 confirmed (DU02, DU03, DU06) |
| Reliable LOO-CV | ~5+ | ❌ Not yet met — treat current LOO-CV numbers as directional, not a validated bound |
| Learned MRI→bone network | ~30–50 paired cases | ❌ Not in scope |

**Action before Track 2 starts:** DU07 classification is complete (excluded due to CT segmentation failure). Training set is 3. Document the LOO-CV caveat in the model card rather than presenting it as equivalent to the 7-case estimate from the earlier draft.

### A.8 Artifact Locations
```
src/synthesis/
    bone_from_mri.py
    train_ssm.py
    extract_cart_features.py
    MODEL_CARD.md          # method rationale, QA thresholds, N=3 caveat
models/ssm/
    pca_model_femur.pkl
    pca_model_tibia.pkl
    shape_regressor_femur.pkl
    shape_regressor_tibia.pkl
    cart_feature_scaler.pkl
    template_femur.obj
    template_tibia.obj
```

---

## What Antigravity Should Actually Build Next, In Order

1. Implement `src/synthesis/train_ssm.py` to handle mesh correspondence (via `probreg`) and PCA fitting.
2. Implement `src/synthesis/bone_from_mri.py`.
3. `synthesize_bone()` + Mahalanobis QA gate (A.5, A.6).
4. Rewire Track 1 Step 3 (unified meshing) to consume synthesized bone instead of CT-derived bone.
5. Strip/guard the CT path out of the FastAPI ingestion and modality auto-detection.
6. Re-run the batch QA system end-to-end on a held-out MRI-only case to confirm the full chain works without any CT touching the per-case path.
7. Only after that: revisit ACL/PCL/meniscus segmentation (Step 1, still not started).
