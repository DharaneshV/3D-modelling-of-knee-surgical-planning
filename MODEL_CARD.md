# KneeTwin Model Card

## Training Data Exclusions
The shape model is trained on paired CT-MRI cases from the DU dataset. Several cases were excluded during the data preparation phase for the following reasons:

- **Missing CT Data:** DU04, DU05
- **Post-Surgical/Implant (TKA):** DU01
- **Input Segmentation Failure:** DU07. The CT segmentation failed to detect the right femur due to a small, localized field of view (FOV). TotalSegmentator's `total` model requires a larger FOV to robustly detect the femur, and the `appendicular_bones` model does not output the femur. While an MRI segmentation bug (collapse due to sagittal orientation) was successfully fixed by enforcing RAI orientation prior to inference, the CT failure is a fundamental limitation of the current CT segmentation pipeline on tight knee FOVs.

## Clinical Scope & Illustrative Rendering

### MRI-Only Pipeline: Illustrative Rendering Only
The MRI-only pipeline (`bone_from_mri.py`) generates bone meshes purely for **illustrative visualization**. It scales a generic, healthy reference bone to match the patient's native cartilage bounding box and aligns it using a rigid Iterative Closest Point (ICP) transform. 

> [!WARNING]
> **CRITICAL USAGE CAVEAT (ILLUSTRATIVE ONLY):**
> 1. **No Quantitative Accuracy:** This pipeline does **NOT** synthesize patient-specific bone morphology. It provides a visual proxy only.
> 2. **Not for Implant Sizing:** Outputs from this module must **not** be used as the basis for implant dimension decisions, joint space width calculations, or surgical planning.
> 3. **Osteophyte Omission:** The generic bone model does not contain osteophytes or patient-specific deformities (e.g., varus/valgus). When visualized next to diseased cartilage (e.g., severe OA cases from the OAI dataset), the generic bone shape will not reflect these abnormalities, which may result in visual misalignment or clipping fallback.

### CT Pipeline: Clinical Fallback
The CT pipeline (`run_ct_segmentation.py`) remains the default, clinically viable pathway for tight-tolerance sizing and morphological analysis.
