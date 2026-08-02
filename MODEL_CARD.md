# KneeTwin Model Card

## Training Data Exclusions
The shape model is trained on paired CT-MRI cases from the DU dataset. Several cases were excluded during the data preparation phase for the following reasons:

- **Missing CT Data:** DU04, DU05
- **Post-Surgical/Implant (TKA):** DU01
- **Input Segmentation Failure:** DU07. The CT segmentation failed to detect the right femur due to a small, localized field of view (FOV). TotalSegmentator's `total` model requires a larger FOV to robustly detect the femur, and the `appendicular_bones` model does not output the femur. While an MRI segmentation bug (collapse due to sagittal orientation) was successfully fixed by enforcing RAI orientation prior to inference, the CT failure is a fundamental limitation of the current CT segmentation pipeline on tight knee FOVs.

## Clinical Scope & Illustrative Rendering

### MRI-Only Pipeline: Native Bone Geometry (v3.2+)
As of `PIPELINE_VERSION` v3.2, the MRI-only pipeline meshes bone directly from CartiMorph's native segmentation labels (1 = femur, 3 = tibia) in the same unified SurfaceNets pass used for cartilage. The earlier approach — fitting a generic reference bone to the patient's cartilage via scale + rigid ICP (`src/synthesis/bone_from_mri.py`) — has been retired. That module is kept in the codebase, clearly marked ARCHIVED, for reference only.

> [!WARNING]
> **CRITICAL USAGE CAVEAT:**
> 1. **Not for Implant Sizing:** Although bone geometry is now patient-specific (derived from the same MRI segmentation as cartilage, not a generic proxy), it has not been validated against CT-grade dimensional accuracy. Outputs must **not** be used as the basis for implant dimension decisions, joint space width calculations, or surgical planning.
> 2. **Segmentation-Bound Fidelity:** Bone shape fidelity — including whether osteophytes or malalignment are captured — is bounded entirely by CartiMorph/nnU-Net segmentation quality on MRI, which has not been independently validated against CT for bone-boundary accuracy.
> 3. **Mesh Sanity, Not Clinical Validation:** Batch QA (`scripts/run_batch_qa_mri.py`) checks mesh sanity only — vertex count and bounding-box size — across all 103 OAIZIB test cases (`bone_qa_oaizib_v32.json`, 0 failures). It does not verify anatomical or dimensional correctness against ground truth.

### Cartilage Accuracy: Ceiling and Failure Mode

**Reported cartilage Dice should be read against ~0.93–0.96, not 1.0.** The pipeline
resamples segmentations to a 0.5 mm isotropic grid
(`src/segmentation/run_mri_segmentation.py`). Passing the *ground truth itself*
through that resampling and back — with no model involved — already costs:

| | femur | tibia | femoral cart. | medial tib. cart. | lateral tib. cart. |
|---|---|---|---|---|---|
| Round-trip Dice | 0.995 | 0.995 | 0.956–0.967 | 0.934–0.958 | 0.941–0.964 |

Bone is effectively unaffected; thin cartilage is not. So the 0.75 soft-tissue
gate has materially less headroom above it than the raw figure suggests.

> [!WARNING]
> **Cartilage loss is systematically under-reported.** In full-thickness defects
> the model labels bright residual tissue or joint fluid in the denuded region as
> cartilage (cohort mean precision 0.835 medial / 0.841 lateral). Any
> cartilage-derived output — denudation area, defect extent — will therefore
> understate the loss. This does **not** affect implant sizing, which is measured
> on bone at the resection surface.

Maximum Hausdorff is a weak QA signal on these structures: across all 515
evaluations, 90 exceed 5 mm and **87 of those still pass their Dice gate**
(correlation with Dice: −0.40). `HD95` is reported alongside it for that reason,
and is far tighter — cohort means 1.34–2.09 mm against 2.61–5.10 mm for the
maximum. The extreme case, `oaizib_491`
medial tibial cartilage at 12.18 mm, is **not a mislocalisation** — its ground
truth is severed into two components (19.5% of volume detached) by a
full-thickness defect, and the prediction bridges the gap, so distant voxels have
no nearby ground truth to match. Its ASSD is 1.44 mm and no predicted voxel lies
more than 3.57 mm from the tibial bone surface — less than a *passing* case
(`oaizib_408`, 4.79 mm).

Bone labels do not share this failure mode: 99th-percentile excursion outside
ground truth is 0.35 mm (femur) and 0.36 mm (tibia), single connected component
in 205 of 206 evaluations, and zero medial/lateral cartilage label confusion
across all 103 cases.

### Longitudinal Axis and Resection Planning

Resection planes, and the sizing measured on them, are referenced to a **limb-axis
proxy** taken from the femur→tibia centroid vector — not to a mechanical axis.

> [!WARNING]
> **A knee-only field of view cannot yield a mechanical axis.** The mechanical
> axis is defined by the hip and ankle centres, neither of which is imaged. Any
> alignment or resection angle produced here is relative to the scan frame and
> must not be treated as a mechanical-axis measurement. Full-limb (hip-to-ankle)
> imaging is required for that.

PCA of the bone vertex cloud is **not** used to recover the shaft direction, and
should not be reintroduced. It only works when the bone is imaged long enough to
be its own longest dimension, which is not the case here: the tibia is roughly
60 mm of imaged length against a ~73 mm-wide plateau, so the principal component
resolves mediolaterally. Before this was corrected the reported "Anatomic Axis
Angle" ranged from 13° to 133° across cases where the true figure is ~5–7°.
Where the shaft direction cannot be recovered for both bones, that metric now
reports N/A rather than the 0° that a shared fallback axis would produce.

Component sizing is measured **on the resection surface**, not at the joint
surface, matching surgical practice. Measuring at the joint surface reports the
tibial intercondylar eminence rather than the plateau — 45.6 mm ML against a
true 83.7 mm on one verified case.

### CT Pipeline: Clinical Fallback
The CT pipeline (`run_ct_segmentation.py`) remains the default, clinically viable pathway for tight-tolerance sizing and morphological analysis.
