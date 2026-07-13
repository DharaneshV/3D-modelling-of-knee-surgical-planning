# Knee Twin: Clinical Validation & Pipeline Overview

## 1. Project Aim & Scope
**Knee Twin** is an automated, end-to-end processing pipeline designed to transform raw medical imaging (CT/MRI) into high-fidelity, interactive 3D anatomical models of the knee joint (femur, tibia, patella). 

**Primary Clinical Objective:** To provide orthopedic surgeons with a highly accurate, patient-specific digital twin of the knee for pre-surgical planning—specifically targeting procedures like Total Knee Arthroplasty (TKA). The system aims to automatically extract reference dimensions (Medial-Lateral width, Anterior-Posterior depth), anatomical axis alignment, and Joint Space Width (JSW) to aid in precise implant sizing and alignment evaluation.

## 2. Current Capabilities & Pipeline
The software currently operates locally as a Proof of Concept (POC) featuring a Fast API backend and a responsive, interactive web frontend.

- **Imaging Ingestion & Modality Detection:** Automatically processes volumetric data (NIfTI format), distinguishing between modalities (e.g., CT bone scans vs. MRI cartilage) to determine the appropriate processing track.
- **Mesh Generation:** Utilizes `vtkSurfaceNets3D` for robust multi-label mesh extraction, translating voxel data into initial 3D geometries.
- **Topological Refinement:** Applies *Taubin Smoothing* (a volume-preserving algorithm that removes stair-step scanning artifacts without artificially shrinking the bone) and intelligent mesh decimation to ensure the geometries are lightweight enough for real-time 3D web rendering while preserving anatomical boundaries.
- **Interactive Visualization:** The web interface seamlessly ties a 2D slice scrubbing viewer directly to the generated 3D spatial models, allowing clinicians to cross-reference the 3D surface with the native radiological cross-sections with zero latency.

## 3. Advancements in Clinical Accuracy
A major focus of the recent development cycle has been moving away from naive bounding-box mathematics to clinically sound anatomical referencing. 

### A. Proportional Regions of Interest (ROI) for Sizing
* **The Problem:** Standard bounding boxes mistakenly measure the entire shaft of the bone, severely inflating the Anterior-Posterior (AP) depth.
* **Our Solution:** The pipeline dynamically identifies the longitudinal axis of the bone and strictly isolates the **distal femur condyles** (bottom 10%) and the **proximal tibia plateau** (top 10%, capped at 30mm). Measurements are exclusively derived from these anatomically relevant resections, closely mirroring the reference zones used for actual implant sizing.

### B. Rotational Invariance via Principal Component Analysis (PCA)
* **The Problem:** If a patient's leg is externally or internally rotated in the scanner, static X/Y measurements will skew, leading to swapped or distorted ML and AP values.
* **Our Solution:** We project the localized ROI vertices into a 2D cross-sectional plane and perform **Principal Component Analysis (PCA)**. This mathematically isolates the true Medial-Lateral (ML) and Anterior-Posterior (AP) anatomical axes independent of the scanner's coordinate system, ensuring symmetrical and rotationally-invariant measurements across both left and right knees.

### C. True Anatomical Longitudinal Alignment
* **The Problem:** Assuming the scanner's Z-axis is the true Superior-Inferior axis fails on oblique scans or differing machine conventions.
* **Our Solution:** The pipeline calculates the global Superior-Inferior axis dynamically by computing the vector between the center of mass of the femur and the tibia. The individual PCA vectors of the bones are oriented to this true anatomical baseline to calculate the overall varus/valgus Anatomic Axis Angle reliably.

### D. Joint Space Width (JSW) & Artifact Guardrails
* **The Problem:** Surface smoothing algorithms can occasionally cause the boundaries of the femur and tibia to inflate slightly and interpenetrate when the joint space is severely narrowed (e.g., severe osteoarthritis), falsely reporting a gap of 0.0mm due to algorithmic overlap rather than true bone-on-bone contact.
* **Our Solution:** We introduced direct 3D collision detection algorithms (`pyvista.collision`). Instead of using statistical band-aids to guess the distance, the system explicitly flags mesh interpenetration in the report (`Measurement uncertain — mesh overlap detected`). This ensures the clinician is never misled by a smoothing artifact masquerading as a clinical finding.

---

## 4. Path to Production Scale (Future Improvements)

To transition this POC into a resilient, production-ready hospital tool, the following architectural and pipeline improvements are prioritized:

**1. Upstream Mesh Boolean Clipping**
The current interpenetration flag is a vital safety guardrail, but the root cause must be resolved. The meshing pipeline requires a topological boolean subtraction (clipping the femur against the tibia) or selective localized smoothing near the joint gap to entirely prevent algorithmic collisions. 

**2. Native DICOM Ingestion & PACS Integration**
The pipeline currently relies on pre-converted `.nii.gz` files. A production deployment requires a robust DICOM parsing layer (e.g., `pydicom` or `Orthanc`) to interface directly with hospital PACS systems, handle Series/Study hierarchies natively, and guarantee secure patient anonymization.

**3. Distributed Asynchronous Architecture**
The current FastAPI backend manages heavy 3D mesh processing on the primary server thread. For production, the architecture must decouple the web API from the processing workload using distributed task queues (e.g., **Celery / Redis / RabbitMQ**). This will allow the system to elastically scale worker nodes to handle dozens of concurrent scan processing requests without bottlenecking the UI.

**4. Robust Automated Testing Suite**
We have begun implementing synthetic mesh collision and PCA rotation tests. This CI/CD pipeline must be expanded to cover edge-case anatomies (e.g., severe dysplasias, hardware artifacts) to guarantee measurement stability before any algorithmic updates are deployed to a clinical environment.

**5. Machine Learning Enhancements for Soft Tissue**
While the bone pipeline (CT) is highly refined, expanding the pipeline to accurately segment cartilage, meniscus, and ligaments from MRI (e.g., 3D DESS sequences) will provide the complete picture required for soft-tissue balancing in advanced surgical planning.
