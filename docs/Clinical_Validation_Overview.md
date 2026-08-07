# KneeTwin: Clinical Validation & Pipeline Overview

**Canonical validation document.** The authoritative accuracy figures, failure
modes and usage caveats live in [`../MODEL_CARD.md`](../MODEL_CARD.md) — read
that before interpreting any output. This document describes the pipeline and
how it is validated; where the two disagree, the model card wins.

> [!WARNING]
> Investigational research software. Not a medical device. Every 3D
> visualisation and quantitative measurement must be correlated against the
> original radiological imaging, and no surgical sizing or diagnostic
> determination is finalised by this tool without independent radiologist
> review.

## 1. Aim & scope

An automated pipeline turning volumetric CT/MRI into patient-specific 3D knee
geometry (femur, tibia, patella, cartilage compartments), intended to support
pre-operative planning for Total Knee Arthroplasty: reference dimensions for
implant sizing, joint space width, and a resection plan with generic implant
components fitted to it.

Runs locally as a proof of concept — FastAPI backend serving both the API and
an interactive web frontend from one origin.

## 2. Pipeline

- **Ingestion & modality detection** — NIfTI volumes, routed to the CT
  (TotalSegmentator) or MRI (CartiMorph/nnU-Net) track by intensity heuristics.
- **Meshing** — `vtkSurfaceNets3D` multi-label extraction, so bone and
  cartilage boundaries are consistent by construction rather than reconciled
  afterwards. SurfaceNets is volume-preserving natively, so no separate
  smoothing pass is applied.
- **Topological repair** — per-label surfaces are re-wound and un-pinched (see
  §3.A), then decimated for web rendering.
- **Resection planning** — distal-femoral five-cut box and proximal-tibial
  cut, with generic parametric components fitted to the result.
- **Visualisation** — 2D slice scrubbing tied to the 3D model, plus a
  1:1-scale AR export for ARCore devices.

## 3. Validated behaviour

### A. Mesh topology

Slicing one label out of the shared multi-label surface leaves two defects,
both repaired in `src/mesh/topology.py`:

- **Inconsistent winding.** Each quad is oriented by its `BoundaryLabels` pair,
  so a bone's faces against background wind opposite to its faces against
  cartilage. This did not only affect watertightness — it made reported volumes
  1.4×–2.1× too large (a femur measured at 325 cm³ against a true 156 cm³).
- **Non-manifold pinch edges** where the label mask contains a checkerboard
  voxel configuration — a handful of edges out of ~33k, which is why hole
  filling never fixed it.

Both repairs are index-only: no vertex coordinate is created, moved or removed,
so the surfaces measurements are taken from do not shift.

### B. Segmentation accuracy

Validated against all 103 OAI-ZIB test cases
(`results/dice_scores_summary.csv`, summarised in the model card). Bone Dice
0.98+; cartilage 0.83–0.87 — which must be read against a **ceiling of
~0.93–0.96**, not 1.0, because the pipeline's own 0.5 mm isotropic resampling
costs that much before the model does anything.

> [!WARNING]
> **Cartilage loss is systematically under-reported.** In full-thickness
> defects the model labels residual tissue or joint fluid in the denuded
> region as cartilage. Any cartilage-derived output understates the loss.

### C. Implant sizing is measured at the resection surface

Sizing is taken from the flat face created by the cut, matching how a component
is sized surgically.

> [!NOTE]
> This **replaces** an earlier approach that measured a proportional ROI at the
> joint surface (distal 10% of the femur / proximal 10% of the tibia). That
> method was wrong on the tibia: the intercondylar eminence sits above the
> plateau, so the ROI measured the spines rather than the plateau and reported
> 45.6 mm ML against a true 83.7 mm — a 46% underestimate, in a field labelled
> as an implant sizing reference. Earlier revisions of this document described
> that ROI method as a feature; it is not.

### D. Longitudinal axis, and what cannot be measured

The axis used for resection planning is a **limb-axis proxy** — the
femur→tibia centroid vector.

> [!WARNING]
> **A knee-only field of view cannot yield a mechanical axis.** That is defined
> by the hip and ankle centres, neither of which is imaged. Any alignment or
> resection angle here is relative to the scan frame.

PCA of the bone vertex cloud is **not** used to recover the shaft direction and
must not be reintroduced. It only works when the bone is imaged long enough to
be its own longest dimension, which a knee FOV is not — the tibia is ~60 mm of
imaged length against a ~73 mm-wide plateau, so the principal component
resolves mediolaterally. Before this was corrected the reported "Anatomic Axis
Angle" ranged from 13° to 133° across cases where the true figure is ~5–7°.
Where the shaft direction cannot be recovered for both bones, that metric now
reports N/A rather than a fabricated number. Regression-tested in
`tests/test_anatomic_axis.py`.

### E. Joint space width & collision guardrails

Smoothing can inflate femoral and tibial boundaries into interpenetration when
the joint space is severely narrowed, falsely reporting 0.0 mm. Direct 3D
collision detection (`pyvista.collision`) flags mesh overlap explicitly
(*"Measurement uncertain — mesh overlap detected"*) rather than letting a
smoothing artifact masquerade as bone-on-bone contact.

### F. Loud failures

Missing geometry surfaces as unambiguous `N/A` flags rather than silent
zero-defaults. Consistent with this, `/api/results` returns `null` for
segmentation accuracy on a live upload — there is no ground truth to compare a
new scan against, and accuracy is only knowable in aggregate from the offline
OAI-ZIB validation.

## 4. Known limitations

- **Single shared key, not per-user access control.** Setting
  `KNEETWIN_API_KEY` gates every `/api/` route, but there are no accounts and
  no per-task ownership — anyone holding the key sees every case. If the
  variable is unset the API is open, with a startup warning. Adequate for a
  local or demo instance; a hospital deployment needs real identity.
- **Generic implants.** No manufacturer publishes component CAD, so the
  geometry is a parametric approximation. Sizes indicate that a component of
  those dimensions fits the anatomy — not a product selection.
- **Resection planning is MRI-track only** (it targets the CartiMorph labels).
- **Single-node processing.** Heavy geometry runs in the server's threadpool.
  Concurrent requests are served, but throughput is bounded by one machine; a
  production deployment would want a task queue.
- **No native DICOM ingestion** — pre-converted `.nii.gz` only.
- **AR is Android/ARCore only.** iOS would need USDZ, which is not implemented.

## 5. Priorities for production

1. Real identity — per-user accounts and per-task access control. The shared
   key covers "keep strangers out"; it does not answer who accessed which
   patient's data, which an audit trail requires.
2. Native DICOM ingestion with PACS integration and anonymisation.
3. Distributed task queue (Celery/Redis) to decouple processing from the API.
4. Expanded test coverage for edge-case anatomy — dysplasia, existing hardware.
5. Soft-tissue segmentation (meniscus, ligaments) for balancing, which
   CartiMorph does not cover.
