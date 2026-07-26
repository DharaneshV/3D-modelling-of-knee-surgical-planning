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
> 3. **Osteophyte & Deformity Omission:** The generic reference bone does not contain osteophytes or patient-specific deformities (e.g., varus/valgus malalignment). When visualised next to diseased cartilage, the smooth generic bone shape will not reflect these abnormalities.
> 4. **Tibial Clipping Behaviour in Severe OA:** Bone-cartilage interface clipping uses VTK's boolean difference filter. In severe OA cases where tibial cartilage is heavily eroded or produces non-manifold mesh geometry, this filter commonly fails and the pipeline falls back to an unclipped generic tibia. The unclipped tibia extends beyond the joint space and can appear oversized relative to the cartilage remnants. This is an expected graceful degradation — the scene remains renderable and communicates the degree of cartilage loss — but viewers should be aware that the tibia geometry in advanced OA cases does not represent the true joint boundary.

### CT Pipeline: Clinical Fallback
The CT pipeline (`run_ct_segmentation.py`) remains the default, clinically viable pathway for tight-tolerance sizing and morphological analysis.
