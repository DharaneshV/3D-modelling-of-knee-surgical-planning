# KneeTwin: Clinical Validation Overview

## Project Scope
KneeTwin is a fully automated, cloud-ready software pipeline designed to extract patient-specific 3D knee geometry (bones and cartilage) from clinical CT and MRI scans. The system generates high-fidelity, watertight 3D models and automatically computes critical clinical measurements used in surgical planning and implant sizing.

## Key Capabilities
- **Multi-Modality Segmentation**: Leverages deep learning models (TotalSegmentator for CT, nnU-Net for MRI) to segment the femur, tibia, patella, and associated cartilage layers.
- **Watertight 3D Meshing**: Utilizes advanced Surface Nets extraction combined with morphological gap-filling to ensure meshes are contiguous, watertight, and free of artificial overlaps.
- **Automated Geometry Analysis**: Computes the anatomic axis alignment, Joint Space Width (JSW), and bounding-box dimensions (ML/AP widths) for implant sizing.
- **High-Performance Caching**: Features an intelligent caching mechanism to prevent redundant processing, enabling near-instantaneous retrieval of previously processed scans.

## Production Readiness and Validation
The pipeline has undergone significant architectural improvements to ensure clinical reliability:

1. **Topological Consistency**
   - **Pre-meshing Gap Filling**: A `fill_label_gaps` algorithm resolves "contested" voxels using a Signed Maurer Distance Map tie-breaker. This prevents non-manifold edges and artificial overlapping of adjacent structures (e.g., femur and tibia bone).
   - **Watertight Meshes**: The pipeline employs `vtkSurfaceNets3D` with volumetric padding to extract perfectly sealed boundaries without internal noise.

2. **Measurement Stability & QA**
   - **Collision Checks**: An automated PyVista-based Boolean intersection check runs at the end of the meshing pipeline to verify that no overlapping bone surfaces exceed a strictly defined clinical tolerance (1.0 mm³).
   - **Drift Validation**: Clinical measurements (JSW, bone volumes, and sizing) have been mathematically verified to remain stable across architectural upgrades, ensuring that topological fixes do not alter the quantitative data relied upon by clinicians.
   - **Loud Failures**: The reporting system correctly falls back to unambiguous "N/A - Mesh data missing" flags if an expected geometry fails to generate, preventing silent false-zero defaults that could mislead surgical planning.

## Clinical Disclaimer
KneeTwin is an investigational software tool. While rigorous automated QA checks are embedded within the pipeline, all 3D visualizations and quantitative measurements must be correlated with original radiological imaging. No surgical sizing or diagnostic determination is finalized by this software tool without independent radiologist review.
