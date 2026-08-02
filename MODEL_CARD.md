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
