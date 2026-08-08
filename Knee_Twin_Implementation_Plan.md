# KneeTwin Implementation Plan v3 — MRI-Only Pipeline

**Current as of `PIPELINE_VERSION` v3.2** (`backend/config.py`).

**Supersedes:**
- the original 9-week CT+MRI dual-track plan, and
- plan v2, which was built around synthesising bone from cartilage with a Statistical Shape Model. **That approach is retired** — see §1. Bone is now meshed directly from the MRI segmentation.

Status tags used throughout:
- ✅ **Done** — built, running in the live pipeline, verified in the repo.
- ⚠️ **Done but needs rewiring** — code exists and works, but the pipeline does not yet consume it (or consumes an older path).
- 🔲 **Not built** — net-new work.
- ❌ **Cut** — removed from the pipeline; do not port forward.

---

## 0. Decisions Locked

| Decision | Detail |
|---|---|
| Per-case modality | MRI is the primary track. CT is retained as a **separate, parallel track**, not deleted — see the note below. |
| Bone generation | **Native segmentation.** CartiMorph labels 1 (femur) and 3 (tibia) are meshed directly. No synthesis, no shape model, no template fitting. |
| Meshing | Single unified multi-label `vtkSurfaceNets3D` pass over bone + cartilage labels together, so bone/cartilage boundaries are shared by construction. |
| Role of paired CT | Historical only. It was training data for the retired shape model. Nothing in the live per-case path reads it. |
| Clinical framing | Anatomy is patient-specific and segmentation-validated. Bone **dimensional** accuracy has not been validated against CT, so sizing outputs remain investigational. |

**⚠️ Correction to plan v2:** v2 tagged the entire CT track ❌ Cut. That was never executed and is no longer the intent. `backend/pipeline_runner.py` still auto-detects modality and runs the TotalSegmentator CT path; `MODEL_CARD.md` describes CT as the pathway for tight-tolerance sizing. What *is* genuinely cut is **per-case CT–MRI registration** — the two tracks run independently and are never spatially fused at inference.

---

## 1. Retired: Bone Synthesis from Cartilage (SSM → ICP → native labels)

This section exists so the current design is legible. **None of it is live. Do not reintroduce any of it without re-reading the reasons below.**

### 1.1 Attempt 1 — Statistical Shape Model (PCA + ridge regression). ❌ Retired.
Built, not merely planned: `src/synthesis/train_ssm.py`, `src/synthesis/extract_cart_features.py`, `src/synthesis/prep_training_data.py`. It fitted a PCA shape space over CPD-corresponded CT bone meshes and a ridge regressor mapping a 9-dim MRI cartilage feature vector (volumes, thicknesses, ML extent, medial/lateral JSW) to shape coefficients, gated at inference by a 3σ Mahalanobis distance check.

Why it was abandoned:

1. **N=3.** Of 7 paired CT-MRI cases, only DU02, DU03 and DU06 survived: DU01 is post-TKA (non-representative anatomy), DU04/DU05 have no CT, and DU07's CT segmentation failed — TotalSegmentator's `total` model could not find the femur on a tight knee FOV, and `appendicular_bones` does not emit femur at all (`MODEL_CARD.md`). `train_ssm.py` therefore hard-codes `n_components=2`, and each LOO-CV fold degenerates to a 1-component PCA fitted on **two** samples. That is not a validation; it is a smoke test, and the script's own log message says so.
2. **The measured error was disqualifying.** The archived QA-gate comment in `src/synthesis/bone_from_mri.py` recorded a true held-out error of **~7 mm** against gates of ≤3.0 mm (femur) / ≤2.5 mm (tibia). The Mahalanobis gate caught egregious inputs only and guaranteed nothing geometric.
3. **The CT and MRI sides were never brought into a common frame.** `prep_training_data.py` segments CT bone and MRI cartilage per case and stops — there is no CT↔MRI co-registration step anywhere in the training path. Bone shape was learned in CT space while the predictors were scalar cartilage features from MRI space, so the model could never learn a *spatial* cartilage→bone relationship, only a size correlation across three patients.

**Orphaned artifacts:** `models/ssm/*.pkl` (2 PCA models, 2 ridge regressors, 1 scaler) are tracked in git, deliberately, as historical reference (commit `42437d3`). They are **not loaded by anything**. The `template_femur.obj` / `template_tibia.obj` meshes they would need are present on disk but *untracked* — `.gitignore`'s blanket `*.obj` rule excludes them — so a fresh clone gets the pickles without the templates they depend on.

### 1.2 Attempt 2 — generic reference bone fitted by scale + rigid ICP. ❌ Retired at v3.2.
The interim replacement (commit `4956223`) dropped the shape model for a DU02-derived generic bone, uniformly scaled to the patient's cartilage bounding box and seated by rigid ICP. It was explicitly labelled illustrative-only: no patient-specific morphology, no osteophytes, no varus/valgus, and visible clipping against severe-OA cartilage.

### 1.3 Current approach — native CartiMorph bone labels. ✅ Live since v3.2 (commit `76e1f11`).
OAI-ZIB is an MRI-only dataset, and CartiMorph already emits femur (1) and tibia (3) as real per-patient labels from the same inference run as cartilage. Meshing those labels in the same `vtkSurfaceNets3D` pass gives patient-specific bone with topologically consistent bone/cartilage boundaries, and removes the entire synthesis stage. Validated across all 103 OAI-ZIB test cases for mesh sanity (`bone_qa_oaizib_v32.json`, 0 failures) and for segmentation accuracy (§3).

`src/synthesis/bone_from_mri.py` is retained with an `ARCHIVED` banner and is called only by `scripts/run_batch_visual.py`, itself marked archived. Neither is on the live path.

### 1.4 Downstream casualty — boolean bone/cartilage clipping. ❌ Archived (commit `a7315a1`).
`backend/mesh_processing/boolean_resolution.py` boolean-clipped a synthesised bone mesh against cartilage so the two would not interpenetrate in the viewer. That was only ever necessary because bone and cartilage came from *separate* sources. A single SurfaceNets pass over one label volume produces a shared boundary by construction, so there is nothing left to resolve. Its only remaining caller is the archived `bone_from_mri.py`; `scripts/run_meshing.py` explicitly dropped its import. Archived in place with a banner, matching project convention. See §8 for why its test outlived it.

---

## 2. Track 1 — Per-Case MRI Pipeline

### Step 1 — Ingestion & Segmentation
- ✅ CartiMorph (nnU-Net) segmentation — labels confirmed: `Background=0, Femur=1, Femoral Cartilage=2, Tibia=3, MTiC=4, LTiC=5`.
- ✅ RAI reorientation before inference (`src/segmentation/run_mri_segmentation.py`). This fixed the sagittal-orientation collapse bug outright; there is **no separate runtime collapse gate**, contrary to plan v2's claim. The nearest equivalents are the mesh-level checks in Step 3.
- ✅ Resample prediction to 0.5 mm isotropic.
- 🔲 ACL / PCL / meniscus segmentation — **not started, and blocked on data, not on effort.** See §5.
- ⚠️ CartiMorph runs as a `wsl -e bash` subprocess against a venv at `~/cartimorph_venv`. It works, but it is a hard host dependency and is not containerised.

### Step 2 — Bone
- ✅ Bone is a segmentation output, not a synthesis step. There is no Step 2 work item any more.
- ❌ `synthesize_bone()`, the SSM training pipeline, and the Mahalanobis inference gate — cut (§1).

### Step 3 — Unified Meshing
- ✅ Unified multi-label `vtkSurfaceNets3D` pass over labels 1–5 (`scripts/run_meshing.py`), replacing per-label marching cubes.
- ✅ `fill_label_gaps` pre-meshing tie-break (Signed Maurer distance) to resolve contested voxels.
- ✅ 1-voxel-gap smoothing bulge bug fixed.
- ✅ Per-part extraction: largest connected component, `decimate(0.9)`.
- ✅ Femur/tibia collision QA with quarantine: >13 000 intersecting faces deletes the pair and fails the task loudly; >6 000 logs a warning to `quarantine_log.json`.
- ❌ **Taubin smoothing is deliberately not applied in the live path.** Plan v2 carried this as an open TODO; it has since been decided against. SurfaceNets output is volume-preserving natively, so an extra smoothing pass adds vertex displacement without benefit. `scripts/postprocess_mesh.py` retains a working Taubin implementation for offline experiments only.

### Step 4 — Measurements
- ✅ JSW (ROI-cropped boundary erosion + cKDTree, anisotropic surface projection, ~9.6 s/side), MRI track only.
- ✅ Fabricated cartilage metric removed; missing geometry reports `N/A - Mesh data missing` rather than a false zero.
- ✅ **Anatomic axis fixed** (commit `aad6b1f`). PCA of the bone vertex cloud cannot recover shaft direction on a knee-only FOV — the tibia is ~60 mm of imaged length against a ~73 mm-wide plateau, so the principal component resolves *mediolaterally*. Reported "Anatomic Axis Angle" previously ranged 13°–133° where the true figure is ~5–7°. `calculate_anatomic_axis()` now takes the femur→tibia centroid vector as a reference and rejects any PCA axis disagreeing with it by more than ~45°; where the shaft cannot be recovered for both bones the metric reports `N/A - Requires full-limb imaging` rather than the fabricated 0° a shared fallback axis would produce.
- ❌ Mechanical axis — **not achievable from this data at all.** It is defined by the hip and ankle centres, neither of which is imaged. Full-limb (hip-to-ankle) imaging would be required. Everything the pipeline reports is a limb-axis *proxy* relative to the scan frame.
- ✅ **Implant sizing is measured on the resection surface, and the PDF now uses it** (commit `08558fc`). Previously the report measured a top-10%/30 mm ROI of the intact bone, which on the tibia reports the intercondylar eminence rather than the plateau — **45.6 mm ML against a true 83.7 mm**, a ~46% underestimate, in a field labelled "implant sizing reference dimensions". `backend/report_generator.calculate_side_metrics()` now calls `plan_resection()` at the default 9 mm femoral / 10 mm tibial cuts using the same limb axis as `/api/resect`, so the two agree exactly (verified: femur 71.3/44.2 mm, tibia 71.5/46.3 mm from both paths). The old joint-surface ROI survives **only as a fallback**: if resection fails it logs a warning and the report's caveat column labels itself `joint-surface ROI (fallback; under-reports tibial width)` rather than passing silently as equivalent. The report's sizing caveat now always states its basis.
- ✅ Bilateral knee support (laterality-aware label maps, centroid-distance splitting) — CT track; the MRI track produces a single `unknown` side.

### Step 5 — Resection Planning ✅ Done
`src/mesh/resection.py` + `GET /api/resect/{task_id}` + viewer panel (commits `f1206ba`, `78e9dbf`).
- Distal-femoral and proximal-tibial cuts, parameterised by depth (default 9 mm / 10 mm), varus/valgus, and posterior slope. API clamps depth to 0–40 mm and angulation to ±15°.
- Capped plane slice, so the retained fragment stays a closed solid.
- Cut-surface ML/AP/area measured from faces within 10° of the cut normal.
- `ensure_watertight()` repairs on a copy and keeps the repair **only if it actually achieves watertightness**, so a broken mesh is left visibly broken rather than silently patched.
- Response carries an explicit `axis_note` that the axis is a proxy, not mechanical.
- Viewer swaps intact bone for resected bone and hides cartilage while a cut is shown.

### Step 6 — Reporting, Viewer & AR
- ✅ FastAPI backend, modality auto-detection, background task processing, single-job lock.
- ✅ Three.js viewer (three-column: original scan, left knee, right knee), per-part visibility toggles, slice viewer with data-scaled window/level.
- ✅ SHA-256 + `PIPELINE_VERSION` cache key (v3.2 bump correctly invalidated stale synthesis caches). Note: cache entries never expire or get pruned — fine for POC, needs a policy for production.
- ✅ ReportLab PDF clinical report. Sizing basis is stated in the caveat column (§2 Step 4).
- ✅ Dashboard modality badge fixed — it was hard-coded to "CT Scan" in the markup and only corrected during upload, so any cached or revisited task displayed the wrong modality.
- ✅ Batch QA (`scripts/batch_qa_report.py`, `scripts/run_batch_qa_mri.py`) with quarantine logging and PDF warnings.
- ✅ **AR export** (commit `f275ec9`): `src/mesh/export_ar_glb.py` emits one colored GLB at true 1:1 scale, mm→m with an LPS→glTF Y-up axis remap. `GET /api/ar/{task_id}` serves it; the viewer exposes "View in AR" via `<model-viewer>` with `ar-modes="webxr scene-viewer"`, gated on the manifest actually containing a GLB. Export runs *after* the collision QA so quarantined parts are excluded.
- ⚠️ **AR is untested on a device.** The pipeline, the endpoint and the button all work; nobody has put it on an Android handset. Scale correctness is verified geometrically in the export transform, not empirically in a room.
- 🔲 **iOS AR is not built.** Quick Look needs USDZ; there is no USDZ export anywhere in the repo and `ar-modes` does not list `quick-look`.
- 🔲 AR measurement and annotation tools from the original spec — not built.
- ⚠️ `<model-viewer>` is loaded from `ajax.googleapis.com`. The viewer will not do AR on an air-gapped deployment.

---

## 3. Accuracy Validation ✅ Done — was the outstanding hard gate, now closed

Run on **all 103 OAI-ZIB test cases** via `scripts/run_oaizib_validation.py`; per-case results in `results/dice_scores_summary.csv` (515 rows). Predictions are resampled onto the ground truth's native grid with nearest-neighbour before comparison, so metrics are computed in the reference standard's own space.

| Structure | Dice (mean ± sd) | Max HD (mm) | ASSD (mm) | Gate | Verdict |
|---|---|---|---|---|---|
| Femur | 0.9825 ± 0.0025 | 3.04 | 0.560 | ≥0.90 | PASS |
| Tibia | 0.9851 ± 0.0026 | 2.61 | 0.404 | ≥0.90 | PASS |
| Femoral cartilage | 0.8741 ± 0.0247 | 5.10 | 0.516 | ≥0.75 | PASS |
| Medial tibial cartilage | 0.8256 ± 0.0450 | 3.65 | 0.526 | ≥0.75 | PASS |
| Lateral tibial cartilage | 0.8457 ± 0.0425 | 4.00 | 0.570 | ≥0.75 | PASS |

**Per-case gate: 8 of 515 structure evaluations fall below threshold** — 6 medial tibial cartilage, 2 lateral tibial cartilage, across 8 distinct cases (worst: `oaizib_497` MTiC at 0.6812). **No bone or femoral-cartilage evaluation fails on any case.** Tibial cartilage is the thinnest structure in the label set and the one most degraded by OA, so this is the expected failure mode rather than a surprise — but it means per-case tibial cartilage output should not be trusted unreviewed.

**What this does and does not establish.** It establishes that the segmentation agrees with expert OAI-ZIB masks. It does **not** establish that bone geometry is dimensionally accurate to CT grade — no CT ground truth exists for these cases — nor that any downstream measurement (JSW, sizing, resection depth) is clinically correct. The separate mesh QA (`bone_qa_oaizib_v32.json`, 103/103 pass) checks vertex count and bounding-box size only; it is a sanity floor, not anatomical verification.

---

## 4. Not Built

| Item | Status | Blocker |
|---|---|---|
| Implant geometry | ✅ Built | `src/mesh/implant.py` — generic parametric tibial tray and femoral component, sized from measured anatomy and fitted in `/api/resect`. Deliberately generic and labelled as such (the API returns an `implant_note` saying so): no public vendor CAD exists, so these match no real implant SKU. Tray sizing tests true 2D containment against the resection outline rather than bounding boxes, and searches seating position, because a centred symmetric tray overhangs an asymmetric plateau. Femoral sizing never rounds AP up, since oversizing notches the anterior cortex. The femoral component's articular surface is a J-curve — its radius tightens ~4x from distal to posterior, as a real condyle's does through flexion — and the condyles are divided by an intercondylar notch, joined anteriorly by the trochlear flange. |
| ACL / PCL / meniscus segmentation | 🔲 Not built | Data (§5), not effort. |
| Tear detection / grading | ❌ Cut | Data (§5). |
| USDZ / iOS AR | 🔲 Not built | — |
| AR measurement + annotation tools | 🔲 Not built | — |
| Patella (MRI track) | 🔲 Not built | CartiMorph does not label it. Reported as `N/A - Not segmented`. |
| Containerised CartiMorph | 🔲 Not built | Currently a `wsl -e bash` host dependency. |

---

## 5. Data Constraints — record these before promising anything

**OAI-ZIB** is a degenerative-osteoarthritis cohort carrying exactly five labels: femur, femoral cartilage, tibia, medial tibial cartilage, lateral tibial cartilage. It contains **no ligaments, no meniscus, and no tears.** Every accuracy figure in §3 is therefore an OA-population figure over those five structures and nothing else.

**Tear datasets do not close the gap.** MRNet and KneeMRI ship **classification labels only** (abnormal / ACL tear / meniscal tear at the exam level) — no voxel-wise masks. Nothing can be meshed from a classification label. Adding ACL/PCL/meniscus geometry therefore requires either a segmentation dataset that includes those structures, or manual annotation; it is not a matter of pointing an existing model at more data.

**Paired CT+MRI** stands at 3 usable cases (DU02, DU03, DU06). That is why the shape-model route died (§1.1) and why nothing in this plan depends on paired data any more.

**Domain shift is untested.** Everything is validated on 3D DESS. Behaviour on anisotropic clinical FSE is unknown; `scripts/run_ood_mri.py` exists for this and has no recorded results in the repo.

---

## 6. Honest Caveats — carry these into every deliverable

1. **Not for implant sizing — the measurement is now correct, the geometry is still unvalidated.** Sizing is taken from the right surface (§2 Step 4), but the bone it is taken *from* has never been checked against CT-grade dimensional accuracy. A correctly-measured dimension on an unvalidated surface is still not a basis for an implant decision, a JSW call, or a surgical plan.
2. **Segmentation-bound fidelity.** Whether osteophytes or malalignment appear at all is bounded entirely by CartiMorph's MRI segmentation, which has not been checked against CT for bone-boundary accuracy.
3. **No mechanical axis.** Every angle is relative to a knee-FOV limb-axis proxy. Full-limb imaging is required for a real one, and no post-hoc correction substitutes for it.
4. **Sizing is a resection-surface measurement at a default cut depth**, not a validated implant recommendation. It is now internally consistent between the PDF and `/api/resect` (§2 Step 4), which is a correctness fix, not a clinical validation. If the fallback path ever fires, the report says so — read the caveat column.
5. **AR has never run on a device.**
6. **Tibial cartilage fails the per-case gate on 8 evaluations.** Batch means pass; individual cases do not always.
7. **Investigational software.** All outputs require correlation with the original imaging and independent radiologist review.

---

## 7. Build Order From Here

1. **Test AR on an Android handset.** Verify 1:1 scale against a ruler, not against the transform matrix. Cheapest way to convert a known unknown into a known.
2. **Generic parametric implant component**, sized from the resection-surface ML/AP, seated on the cut plane. Label it generic in the UI and the PDF.
3. **Decide the ligament/meniscus question on data grounds** — source a segmentation dataset that includes them, or drop the capability from the roadmap. Do not carry it as an open engineering task.
4. **Run the OOD check** (`scripts/run_ood_mri.py`) on anisotropic FSE and record the result, so the domain-shift caveat becomes a measurement instead of an assumption.
5. **Add regression coverage for the measurement path** (§8) — sizing, axis, JSW. The suite currently covers gap-filling, the report builder, and the slice endpoint; the numbers most likely to be wrong in a clinical document have no test.
6. Containerise the CartiMorph/WSL dependency before any deployment discussion.
7. Cache-pruning policy before any multi-user deployment.

---

## 8. Test Suite & A Version-Control Gap Worth Remembering

**Current state:** ✅ `pytest tests/ -q` → **125 passed**, no ignore flags, no GPU or external models required (everything runs against synthetic fixtures). Files: `test_fill_label_gaps.py`, `test_report_generator.py`, `test_slice_endpoint.py`, `test_implant.py`, `test_mesh_topology.py`, `test_cut_surface.py`, `test_upload_security.py`, `test_anatomic_axis.py`, `test_ar_export.py`, `test_resect_endpoint.py`, `test_auth.py`.

The last four close the gaps this section previously flagged as thin coverage:
`test_upload_security.py` pins the upload path-traversal fix (parametrised over
separator styles, since `\` is also a separator on Windows);
`test_anatomic_axis.py` guards the PCA-rejection fix — verified genuinely
protective by running its assertions against a reimplementation of the pre-fix
version, which returns a mediolateral axis and reproduces a 91° "anatomic axis
angle"; `test_ar_export.py` covers the AR triangle budget, above all that
decimation preserves real-world scale (AR renders at 1:1, so a shifted bounding
box means differently-sized anatomy); and `test_resect_endpoint.py` covers the
`/api/resect` route itself, including the two divergent shapes under
`resections` — a divergence that had already crashed the frontend once.

**What went wrong.** `.gitignore` carried an unanchored `test_*.py` rule, aimed at ~35 ad-hoc scratch scripts sitting at the repo root. Git applies unanchored patterns at *every* directory level, so the rule also matched inside `tests/`. Two of the four suite files — `tests/test_boolean_resolution.py` and `tests/test_fill_label_gaps.py` — were consequently never tracked. (The other two, `test_report_generator.py` and `test_slice_endpoint.py`, *were* tracked throughout; the suite as a whole was in version control, the gap was file-specific.)

**Why it mattered.** An untracked test never appears in a diff. `boolean_resolution.py` was deliberately rewritten from a hard QA gate into a best-effort visual clip in a tracked commit, and its untracked test kept asserting the pre-v3.2 contract with nobody seeing the mismatch. It eventually broke *collection* — `from ... import check_self_intersection`, a symbol with zero references anywhere in the tree — which took the entire suite down, not just that file. Three further assertions were also stale (tuple-vs-PolyData return, an expected `RuntimeError` the function no longer raises), so restoring the missing import would have left 3 of 4 failing. The test described behaviour the project abandoned on purpose. It was removed with the module it covered (§1.4).

**Fix:** a `!tests/test_*.py` negation re-includes the real suite while root and `scripts/` clutter stay ignored. `.claude/worktrees/` is now ignored too.

**Standing lesson for this repo:** ignore rules for scratch files must be anchored (`/test_*.py`), and an archived module should take its test with it in the same commit. Coverage remains thin — see build-order item 5.

---

## Appendix A — Archived SSM Specification

Kept only so the retired work is traceable. **Superseded; do not implement.** Full detail lives in git history (`f901014`, `78969b6`, `4956223`, `76e1f11`) and in the archived source under `src/synthesis/`.

- **Method:** CPD rigid + deformable correspondence of CT bone meshes to a DU02 reference → PCA shape space (`n_components=2`, hard-coded for N=3) → `StandardScaler` + `Ridge(alpha=1.0)` from a 9-dim cartilage feature vector to PCA scores.
- **Feature vector (9-dim):** femoral cartilage volume / mean thickness / ML extent; medial and lateral tibial cartilage volume and mean thickness; medial and lateral JSW. All z-scored.
- **Inference gate:** Mahalanobis distance of predicted coefficients > 3.0σ → reject and flag.
- **Training gate:** LOO-CV mean surface distance ≤3.0 mm (femur) / ≤2.5 mm (tibia).
- **Outcome:** ~7 mm held-out error against those gates, on N=3, with no CT↔MRI co-registration in the training path. Retired.
- **Surviving artifacts:** `models/ssm/{pca_model_femur,pca_model_tibia,shape_regressor_femur,shape_regressor_tibia,cart_feature_scaler}.pkl` — orphaned and unloaded. The `template_*.obj` meshes they would require exist on disk but are untracked (`.gitignore`'s `*.obj`), so they are absent from a fresh clone.
