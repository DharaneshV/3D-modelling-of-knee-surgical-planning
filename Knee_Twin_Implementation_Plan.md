# Knee Surgical Planning Twin — Implementation Plan
### Merging the Dev Spec (9-week POC) with Research-Backed Upgrades

This plan takes the baseline pipeline from the dev spec (threshold segmentation → marching cubes → AR) and layers in specific upgrades from the research report wherever they meaningfully raise accuracy or de-risk a milestone, without blowing up the 9-week timeline. Each week below states: **what the spec asks for**, **what to actually build**, and **why** (citing the research where it changes the approach).

---

## 0. Before Week 1 — Environment & Decisions

| Decision | Recommendation | Why |
|---|---|---|
| Language | Python-only | MATLAB parallel path adds no accuracy benefit; skip unless a sponsor explicitly needs MathWorks credit |
| Core framework | **MONAI** (PyTorch) instead of raw U-Net scripts | MONAI Core ships NIfTI/DICOM-native transforms, pre-built Swin UNETR, and class-imbalance loss functions out of the box — saves you writing preprocessing boilerplate |
| MRI Dataset | **OAI-ZIB** (3D DESS MRI, expert bone+cartilage masks) | Standard benchmark set with expert annotations — avoids manual tracing for Track A |
| CT Dataset | **TotalSegmentator** (CC-BY whole-body CT + bone masks) | Ships femur/tibia ground-truth masks for free; crop to knee region. Skip TCIA — cancer-focused, thin knee OA presence |
| CT–MRI registration scoping | **Option (b): independent tracks** — CT (bone) and MRI (soft tissue) validated independently; Week 5 registration is a capability demo on matched cases if found, not a full-population requirement | Honest scoping for a 9-week timeline; avoids silently faking alignment between different patient anatomies |
| Repo structure | Use the spec's structure as-is | It's already sound; add a `src/preprocessing/` folder (see Week 1) |

Set up:
```bash
pip install monai simpleitk pydicom scikit-image pyvista vtk pandas
```

---

## Week 1 — Data Acquisition + QA

**Spec deliverable:** 5–8 validated DICOM cases, 1–2 held out.

> **Revised:** Two separate data tracks — preprocessing differs by modality.

### Track A — MRI (Soft Tissue) → OAI-ZIB
1. Pull cases from OAI-ZIB via HuggingFace (`YongchengYAO/OAIZIB-CM`) — already done.
2. Preprocessing:
   - **N4 Bias Field Correction** (MRI-only — corrects RF coil sensitivity variation, does not apply to CT).
   - **Resampling to isotropic spacing** (B-spline for images, nearest-neighbor for masks).
3. QA gate: flag slice spacing > 1.5mm or motion artifact — run **after** N4, since correction sometimes reveals hidden artifacts.

### Track B — CT (Bone) → TotalSegmentator
1. Pull 5–8 whole-body CT scans from the **TotalSegmentator** dataset (CC-BY licensed, ships with femur/tibia bone masks).
2. **Crop to knee region** — TotalSegmentator volumes are whole-body; extract the knee bounding box using the femur/tibia label extent.
3. Check patella label coverage — TotalSegmentator's label list has expanded across versions; if patella is absent, manually trace on the handful of cropped cases only.
4. Preprocessing:
   - **Resampling to isotropic spacing only** — no N4 (CT has no RF bias field).
   - **Metal streak artifact check** — flag any post-op cases with implants; they corrupt HU thresholding.
5. QA gate: same slice-spacing check; additionally flag any case where max HU > 5000 (probable metal artifact).

### Patient-Mismatch Decision (flag now, not at Week 5)
CT bone cases and MRI soft-tissue cases are from **different patients**. The Week 5 CT–MRI registration step assumes same-patient data. Resolution: **Option (b)** — validate tracks independently through Weeks 2–4; treat Week 5 registration as a capability demo on any matched cases found. Document explicitly in deliverables that full same-patient multi-modal validation is future work.

**Why it matters:** the research report specifically flags the domain-shift problem — models trained on clean 3D DESS data fail on anisotropic FSE scans. Preprocessing differences between modalities must be applied correctly per-track from day one.

---

## Week 2–3 — Bone Segmentation (CT)

**Spec deliverable:** threshold-based femur/tibia/patella masks.

**Build exactly as specified** — Hounsfield-unit thresholding is correct here. Bone-CT contrast is high enough that deep learning is unnecessary overhead for the POC.

**Data source update:** Run against **TotalSegmentator-derived CT cases** (cropped to knee region), not OAI-ZIB.

**New: visual sanity pass first.** TotalSegmentator uses whole-body CT protocols which vary more in slice thickness and reconstruction kernel than a dedicated knee-protocol CT. Do a quick visual check on your 5–8 cropped cases to confirm HU values behave consistently before locking in threshold values — don't assume the standard [200, 3000] range is optimal across all cases without verifying.

**One addition:** run your Dice/Hausdorff metrics (Week 4 code) on this bone output as soon as it exists, even informally — don't wait for the formal Week 4 milestone to discover threshold values need tuning per-scanner.

---

## Week 3–4 — Soft Tissue Segmentation (MRI)

**Spec deliverable:** cartilage/ACL/PCL/meniscus masks via a pre-trained U-Net/nnU-Net checkpoint.

**Upgrade recommendation:** if a pre-trained nnU-Net checkpoint isn't readily available or its Dice on your cases falls in the 0.70s, swap in **MONAI's Swin UNETR** reference implementation rather than training a U-Net from scratch. Reasoning from the research:
- Standard CNNs struggle with cartilage specifically because they can't model long-range dependencies between the femoral and tibial articular surfaces — this is exactly why cartilage Dice lags bone Dice industry-wide.
- Swin UNETR on knee MRI is reported at ~89% Dice for femoral cartilage and ~85% for tibial cartilage, which comfortably clears the spec's 0.75–0.85 soft-tissue target in Section 8.5, whereas a plain U-Net often sits right at the bottom edge of that range or below it.
- MONAI ships this architecture pre-built — using it isn't "training a research model from scratch," it's swapping one config in an existing pipeline.

**Fallback if timeline is tight:** stick with the pre-trained U-Net as the spec says; just budget an extra buffer day for Week 4 because soft-tissue Dice is the metric most likely to trigger the manual-review gate.

---

## Week 4 — Accuracy Validation (Hard Gate)

**Spec deliverable:** Dice + Hausdorff report per structure per case, using the exact `SimpleITK.LabelOverlapMeasuresImageFilter` / `HausdorffDistanceImageFilter` code already provided in Section 8.2–8.3.

**Build as specified, no changes needed** — this code is already correct and matches standard practice (DSC, ASSD/Hausdorff are the field's standard metrics). Two small additions:
1. Also log **Average Symmetric Surface Distance** if your SimpleITK version supports it — it's more sensitive to boundary jaggedness than Dice alone and gives you an early signal on mesh quality before Week 5.
2. Enforce the gate literally in code (not just process): have the batch-eval script raise/flag any row below 0.90 (bone) or 0.75 (soft tissue) rather than relying on someone reading the CSV.

---

## Week 5 — Mesh Generation + Registration

**Spec deliverable:** marching cubes → aligned CT+MRI mesh, overlay-verified.

**Upgrade recommendation — this is the highest-value change in the whole plan:**

Replace plain `skimage.measure.marching_cubes` (run per-structure) with **`vtkSurfaceNets3D`**, run once across the full multi-label mask (bone + cartilage labels together).

Why this matters concretely for your project: marching cubes evaluates each label independently, which means the femur mesh and the cartilage mesh will have **non-coincident, sometimes intersecting triangles** exactly where they touch — the articular surface, which is the single most clinically important region for implant sizing in Section 10. Surface Nets is built for labeled volumes specifically to guarantee shared, non-intersecting boundaries between adjacent tissues. This directly serves the spec's own "overlay check" requirement — a Surface Nets mesh will pass that visual alignment check more reliably than stitched-together marching-cubes outputs.

```python
# vtk pipeline sketch
import vtk
reader = vtk.vtkNIFTIImageReader()
reader.SetFileName("multilabel_mask.nii.gz")
reader.Update()

surfacenets = vtk.vtkSurfaceNets3D()
surfacenets.SetInputConnection(reader.GetOutputPort())
surfacenets.SetLabels(0, 1)  # femur
surfacenets.SetLabels(1, 2)  # tibia
surfacenets.SetLabels(2, 3)  # cartilage
surfacenets.Update()
```

**Smoothing:** the spec doesn't mention smoothing explicitly, but raw voxel meshes (from either algorithm) are stair-stepped. Do **not** use default Laplacian smoothing — it shrinks volume with every iteration, which will silently corrupt the implant-sizing measurements in Week 6. Use **Taubin smoothing** instead (`pyvista`'s `smooth_taubin()`, tunable `pass_band`), which alternates a shrink and an expand pass per iteration to remove high-frequency jaggedness while preserving the low-frequency volume/shape. This is a two-line swap:
```python
mesh_smoothed = mesh.smooth_taubin(n_iter=20, pass_band=0.1)
```

**Registration:** run CT–MRI rigid/affine registration exactly as the spec requires (Section 9) — the research doesn't suggest a change here; SimpleITK's registration module is standard and appropriate. The spec is right that this step is commonly skipped and is a real source of silent error; don't cut it even if the timeline is tight.

**Decimation:** the spec's instruction to validate measurements pre/post-decimation still holds. Do this check on the Taubin-smoothed mesh, since smoothing changes vertex positions slightly and you want the final validated numbers to reflect the mesh that actually ships to AR.

---

## Week 6 — Clinical Measurements

**Build as specified** — mechanical axis angle, joint space width, implant sizing (AP/ML), with error margin reported vs. ground truth in `measurement_validation.csv`.

**No architectural change needed here**, but note for the pitch deck: research-grade radiograph-to-3D systems (e.g., BoneVision) report sub-millimeter RMSE (~0.9–1.0mm) against CT ground truth for femur/tibia — useful as an external benchmark to frame your own error-margin numbers against when presenting results, even though your pipeline is CT/MRI-based rather than X-ray-based.

---

## Week 7–8 — AR Visualization

**Build exactly as specified** — Unity + ARFoundation (or WebXR fallback), semi-transparent soft tissue / opaque bone materials, marker-based scale calibration, slice plane, measurement tool, annotation, color-coded risk zones.

No research-driven changes apply here — this is an engineering/integration week, not a modeling-accuracy week. One practical note: import the Taubin-smoothed, decimated mesh (not the raw Surface Nets output) — Unity/WebXR performance depends on the ~100K polygon budget the spec already specifies.

---

## Week 9 — Demo Packaging

**Build as specified.** For the pitch deck's accuracy table (Structure | Dice | Hausdorff | N cases), consider adding a short callout row noting *which* segmentation approach was used per structure (threshold vs. pre-trained U-Net vs. Swin UNETR) — it's a natural way to show engineering judgment (right tool per structure) rather than one-size-fits-all deep learning.

---

## Summary of Deviations from the Spec

| Spec item | Plan | Reason |
|---|---|---|
| Section 3, MRI segmentation | Consider Swin UNETR via MONAI if pre-trained U-Net underperforms the 0.75–0.85 cartilage target | Cartilage needs long-range context CNNs don't capture well |
| Section 9, mesh generation | Use `vtkSurfaceNets3D` on the combined multi-label mask instead of per-structure marching cubes | Prevents intersecting/non-coincident triangles at bone–cartilage boundaries |
| Section 9, mesh cleanup | Add Taubin smoothing (`pyvista.smooth_taubin`) before decimation; avoid plain Laplacian | Laplacian shrinks volume every iteration, corrupting implant-sizing measurements |
| Section 6, ground truth | Prefer OAI-ZIB/SKI10 pre-annotated masks over manual tracing where cases overlap those datasets | Saves manual annotation time within the 9-week window |

Everything else in the spec (architecture, repo layout, milestones, thresholds, AR requirements, deliverables checklist) holds as written — the changes above are targeted swaps at exactly the three points where the research report shows the "obvious" approach has a known failure mode.

---

## Open Questions Carried Forward (from spec Section 13)
- Demo narrative pathology (e.g., TKA candidate) — needed before Week 6 measurement framing.
- MATLAB licensing — plan above assumes Python-only; confirm before Week 3.
- Target AR hardware (phone/Quest/HoloLens) — needed before Week 7 to pick ARFoundation vs. WebXR.
- Ground-truth availability per case — determines how much Week 1 time goes to manual tracing vs. using OAI-ZIB masks directly.
