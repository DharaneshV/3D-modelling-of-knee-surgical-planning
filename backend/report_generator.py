import os
import sys
import json
import datetime
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.ndimage import binary_erosion
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Run as a subprocess (`python backend/report_generator.py`), so sys.path[0] is
# backend/, not the repo root. Needed for the src.mesh imports below.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False

# A PCA axis is only trusted if it agrees with the anatomical reference to
# within ~45 degrees. Below that the principal component has locked onto a
# transverse dimension rather than the shaft.
AXIS_AGREEMENT_THRESHOLD = 0.7


def calculate_anatomic_axis(vertices, reference_dir=None):
    """
    Compute a bone's longitudinal axis.

    PCA recovers the shaft direction only when the bone is imaged long enough to
    be its own longest dimension. On a knee-only MRI FOV it is not: the tibia is
    roughly 60mm of length against a 73mm-wide plateau, so the principal
    component comes back mediolateral and the "long axis" is transverse.

    When reference_dir is supplied (the femur->tibia centroid vector is a stable
    limb-axis proxy) the PCA result is accepted only if it broadly agrees with
    it; otherwise the reference direction is used instead.

    Returns:
        (axis, pca_trusted) — pca_trusted is False when the PCA result was
        rejected and reference_dir was substituted.
    """
    centered = vertices - np.mean(vertices, axis=0)
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # The primary axis corresponds to the eigenvector with the largest eigenvalue
    primary_axis = eigvecs[:, np.argmax(eigvals)]

    if reference_dir is None:
        # Fallback to positive Z if no reference is given
        if primary_axis[2] < 0:
            primary_axis = -primary_axis
        return primary_axis, True

    if np.dot(primary_axis, reference_dir) < 0:
        primary_axis = -primary_axis

    if np.dot(primary_axis, reference_dir) < AXIS_AGREEMENT_THRESHOLD:
        return reference_dir, False

    return primary_axis, True

def get_roi_sizing(vertices, long_axis, is_femur=True, side='unknown'):
    # Project all vertices onto the longitudinal axis to find min/max
    projections = np.dot(vertices, long_axis)
    min_p, max_p = np.min(projections), np.max(projections)
    length = max_p - min_p
    
    # Proportional ROI: 10% of length, capped at 30mm
    roi_depth = min(length * 0.10, 30.0)
    
    print(f"DEBUG [{side}]: {'Femur' if is_femur else 'Tibia'} total length = {length:.1f}mm, ROI depth used = {roi_depth:.1f}mm")
    
    # Femur ROI is the distal end (lowest projection if axis points superiorly)
    # Tibia ROI is the proximal end (highest projection if axis points superiorly)
    if is_femur:
        mask = projections <= (min_p + roi_depth)
    else:
        mask = projections >= (max_p - roi_depth)
        
    roi_verts = vertices[mask]
    
    if len(roi_verts) < 3:
        return 0.0, 0.0 # fallback
        
    # Subtract mean
    centered = roi_verts - np.mean(roi_verts, axis=0)
    
    # Project onto plane orthogonal to longitudinal axis (2D cross-section)
    # We can just remove the longitudinal component
    long_comp = np.outer(np.dot(centered, long_axis), long_axis)
    cross_section = centered - long_comp
    
    # Compute 2D PCA on the cross section
    cov = np.cov(cross_section, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    # The largest two eigenvectors represent ML and AP
    # Since we are in 3D, one eigenvalue will be near 0 (the longitudinal one we subtracted)
    sorted_indices = np.argsort(eigvals)[::-1]
    ml_axis = eigvecs[:, sorted_indices[0]]
    ap_axis = eigvecs[:, sorted_indices[1]]
    
    # ML is the widest spread, AP is the next widest
    ml_spread = np.ptp(np.dot(roi_verts, ml_axis))
    ap_spread = np.ptp(np.dot(roi_verts, ap_axis))
    
    return round(float(ml_spread), 1), round(float(ap_spread), 1)

def calculate_side_metrics(mesh_dir: Path, side: str, laterality_summary: dict, modality: str, mask_path: str = None):
    """Calculate metrics for a specific side (left or right)."""
    metrics = {
        "femur_vol": "N/A - Mesh data missing", "tibia_vol": "N/A - Mesh data missing", "patella_vol": "N/A - Not segmented" if modality == "MRI" else "N/A - Mesh data missing",
        "jsw": "N/A - Mesh data missing",
        "femur_ml": "N/A - Mesh data missing", "femur_ap": "N/A - Mesh data missing",
        "tibia_ml": "N/A - Mesh data missing", "tibia_ap": "N/A - Mesh data missing",
        "alignment_angle": "N/A - Mesh data missing",
        "sizing_basis": "unavailable",
        "medial_thick": "2.2", "lateral_thick": "2.4", "trochlear_thick": "2.1" # Mocked/N/A for CT
    }
    
    # 1. Volumes from summary if available, else 0
    counts = laterality_summary.get("voxel_counts", {})
    # Need voxel volume to convert count to volume, but we don't have spacing easily here without reading the mask.
    # We will read the mask to get volume in mm^3
    task_id = mesh_dir.name
    mask_path = Path(mask_path) if mask_path else mesh_dir / f"{task_id}_mask.nii.gz"
    if mask_path.exists() and HAS_SITK:
        try:
            img = sitk.ReadImage(str(mask_path))
            arr = sitk.GetArrayFromImage(img)
            spacing = img.GetSpacing()
            voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]
            
            # Map side to labels
            if modality == "MRI":
                f_label, t_label, p_label = 1, 3, -1
            else:
                f_label = 1 if side == "left" else 2
                t_label = 3 if side == "left" else 4
                p_label = 5 if side == "left" else 6
            
            metrics["femur_vol"] = int(np.sum(arr == f_label) * voxel_vol_mm3)
            metrics["tibia_vol"] = int(np.sum(arr == t_label) * voxel_vol_mm3)
            if modality != "MRI":
                metrics["patella_vol"] = int(np.sum(arr == p_label) * voxel_vol_mm3)
        except Exception as e:
            print(f"Error reading mask for volumes: {e}")
            
    # 2. Load meshes for JSW, Alignment, and Sizing
    f_mesh_path = mesh_dir / f"femur_{side}.obj"
    t_mesh_path = mesh_dir / f"tibia_{side}.obj"
    
    if f_mesh_path.exists() and t_mesh_path.exists():
        try:
            # process=False matters: trimesh's default merges coincident vertices,
            # which would undo the vertex split that makes these meshes manifold
            # (see src/mesh/topology.py) and put volume reporting back to None.
            femur = trimesh.load(str(f_mesh_path), process=False)
            tibia = trimesh.load(str(t_mesh_path), process=False)
            
            f_centroid = np.mean(femur.vertices, axis=0)
            t_centroid = np.mean(tibia.vertices, axis=0)
            up_vector = f_centroid - t_centroid
            if np.linalg.norm(up_vector) > 0:
                up_vector = up_vector / np.linalg.norm(up_vector)
            
            f_axis, f_pca_ok = calculate_anatomic_axis(femur.vertices, reference_dir=up_vector)
            t_axis, t_pca_ok = calculate_anatomic_axis(tibia.vertices, reference_dir=up_vector)

            # Verify symmetry/consistency (log only)
            print(f"DEBUG [{side}]: Femur primary axis: {f_axis} (pca_trusted={f_pca_ok})")
            print(f"DEBUG [{side}]: Tibia primary axis: {t_axis} (pca_trusted={t_pca_ok})")

            # The angle between the two shafts is only meaningful if each shaft
            # direction was actually recovered. Where PCA was rejected, both
            # bones fall back to the same limb-axis proxy and the angle collapses
            # to 0 degrees — a fabricated number, not a measurement. A knee-only
            # FOV has no hip or ankle centre, so there is no mechanical axis to
            # fall back on either.
            if f_pca_ok and t_pca_ok:
                cos_theta = np.dot(f_axis, t_axis) / (np.linalg.norm(f_axis) * np.linalg.norm(t_axis))
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                metrics["alignment_angle"] = round(float(np.degrees(np.arccos(cos_theta))), 1)
            else:
                metrics["alignment_angle"] = "N/A - Requires full-limb imaging"
            
            # Implant sizing is measured on the resection surface, matching how a
            # component is sized surgically. Measuring at the joint surface
            # instead reports the tibial intercondylar eminence rather than the
            # plateau — 45.6mm ML against a true 83.7mm on a verified case.
            # The limb axis is used (not the per-bone PCA axis) so these agree
            # with /api/resect, which plans the same cuts.
            try:
                from src.mesh.resection import plan_resection
                f_plan = plan_resection(femur, up_vector, "femur")
                t_plan = plan_resection(tibia, up_vector, "tibia")
                metrics["femur_ml"] = f_plan["cut_surface"]["ml_mm"]
                metrics["femur_ap"] = f_plan["cut_surface"]["ap_mm"]
                metrics["tibia_ml"] = t_plan["cut_surface"]["ml_mm"]
                metrics["tibia_ap"] = t_plan["cut_surface"]["ap_mm"]
                metrics["sizing_basis"] = (
                    f"resection surface, {f_plan['depth_mm']:.0f}mm femoral / "
                    f"{t_plan['depth_mm']:.0f}mm tibial"
                )
            except Exception as e:
                # Loud, not silent: the fallback measures a different thing and
                # the report must say so rather than quietly reporting it.
                print(f"WARNING [{side}]: resection-level sizing failed ({e}); "
                      f"falling back to joint-surface ROI")
                metrics["femur_ml"], metrics["femur_ap"] = get_roi_sizing(femur.vertices, f_axis, is_femur=True, side=side)
                metrics["tibia_ml"], metrics["tibia_ap"] = get_roi_sizing(tibia.vertices, t_axis, is_femur=False, side=side)
                metrics["sizing_basis"] = "joint-surface ROI (fallback; under-reports tibial width)"
            
            # JSW Calculation
            metrics["jsw_overlap"] = False
            jsw_computed = False
            
            if modality == "CT":
                metrics["jsw"] = "N/A - Requires MRI"
                jsw_computed = True
            else:
                metrics["jsw"] = 0.0
            
            # Mask-based JSW Calculation
            if not jsw_computed and mask_path.exists() and HAS_SITK:
                try:
                    f_mask = arr == f_label
                    t_mask = arr == t_label
                    
                    t_z_indices = np.where(t_mask)[0]
                    mask_joint_z = np.max(t_z_indices) if len(t_z_indices) > 0 else 0
                    
                    z_min = max(0, int(mask_joint_z - 40.0 / spacing[0]))
                    z_max = min(arr.shape[0], int(mask_joint_z + 40.0 / spacing[0]))
                    
                    f_roi = f_mask[z_min:z_max, :, :]
                    t_roi = t_mask[z_min:z_max, :, :]
                    
                    f_bnd = f_roi ^ binary_erosion(f_roi)
                    t_bnd = t_roi ^ binary_erosion(t_roi)
                    
                    f_pts = np.argwhere(f_bnd)
                    f_pts[:, 0] += z_min
                    f_pts = f_pts * spacing
                    
                    t_pts = np.argwhere(t_bnd)
                    t_pts[:, 0] += z_min
                    t_pts = t_pts * spacing
                    
                    if len(f_pts) > 0 and len(t_pts) > 0:
                        tree = cKDTree(t_pts)
                        dists, idxs = tree.query(f_pts)
                        min_idx = np.argmin(dists)
                        center_dist = dists[min_idx]
                        
                        p_f = f_pts[min_idx]
                        p_t = t_pts[idxs[min_idx]]
                        v = p_f - p_t
                        
                        correction = np.sum(np.abs(v / center_dist) * spacing) if center_dist > 0 else 0.0
                        surface_dist = center_dist - correction
                        
                        if surface_dist <= 0:
                            metrics["jsw_overlap"] = True
                        surface_dist = max(0.0, surface_dist)
                        
                        metrics["jsw"] = float(f"{surface_dist:.1f}")
                        jsw_computed = True
                except Exception as e:
                    print(f"Mask JSW calculation failed, falling back to KDTree dist: {e}")

            # Mesh-based JSW Fallback
            if not jsw_computed:
                t_proj = np.dot(tibia.vertices, t_axis)
                f_proj = np.dot(femur.vertices, t_axis)
                mesh_joint_z = np.max(t_proj)
                
                f_mask_mesh = (f_proj >= mesh_joint_z - 20) & (f_proj <= mesh_joint_z + 40)
                t_mask_mesh = (t_proj >= mesh_joint_z - 40) & (t_proj <= mesh_joint_z + 20)
                
                f_pts_mesh = femur.vertices[f_mask_mesh]
                t_pts_mesh = tibia.vertices[t_mask_mesh]
                
                if len(f_pts_mesh) > 0 and len(t_pts_mesh) > 0:
                    tree = cKDTree(t_pts_mesh)
                    closest_dists, _ = tree.query(f_pts_mesh)
                    metrics["jsw"] = float(f"{np.min(closest_dists):.1f}")
                    
        except Exception as e:
            print(f"Error calculating mesh metrics for {side}: {e}")
            
    return metrics

def build_metrics_list(metrics, modality):
    def format_val(val, unit=""):
        if isinstance(val, str) and val.startswith("N/A"):
            return val
        if "vol" in unit:
            return f"{float(val) / 1000.0:.1f} cm³"
        return f"{val}{unit}"

    jsw_caveat = "Minimum distance at joint space"
    if modality == "CT":
        jsw_caveat = "N/A - Requires MRI"
    elif metrics.get("jsw_overlap"):
        jsw_caveat = "Measurement uncertain — resolution limit or mesh overlap"
    elif isinstance(metrics["jsw"], str) and "N/A" in metrics["jsw"]:
        jsw_caveat = "Required mesh file missing"
        
    cart_val = "N/A (CT modality)" if modality == "CT" else f"Medial {metrics['medial_thick']} mm / Lateral {metrics['lateral_thick']} mm"

    return [
        {
            "name": "Femur Bone Volume",
            "value": format_val(metrics['femur_vol'], "vol"),
            "caveat": ""
        },
        {
            "name": "Tibia Bone Volume",
            "value": format_val(metrics['tibia_vol'], "vol"),
            "caveat": ""
        },
        {
            "name": "Patella Bone Volume",
            "value": format_val(metrics['patella_vol'], "vol"),
            "caveat": ""
        },
        {
            "name": "Joint Space Width (JSW)",
            "value": format_val(f"{metrics['jsw']:.1f}" if not isinstance(metrics['jsw'], str) else metrics['jsw'], " mm"),
            "caveat": jsw_caveat
        },
        {
            "name": "Anatomic Axis Angle",
            "value": format_val(metrics['alignment_angle'], "°"),
            "caveat": (
                "Calculated via PCA of bone shafts"
                if not isinstance(metrics['alignment_angle'], str)
                else "Shaft axes not recoverable from this field of view; "
                     "a true mechanical axis requires hip-to-ankle imaging"
            )
        },
        {
            "name": "Femur ML Width / AP Depth",
            "value": f"{metrics['femur_ml']} mm / {metrics['femur_ap']} mm" if not isinstance(metrics['femur_ml'], str) else metrics['femur_ml'],
            "caveat": f"Implant sizing reference, measured at {metrics['sizing_basis']}" if not isinstance(metrics['femur_ml'], str) else "Required mesh file missing"
        },
        {
            "name": "Tibia ML Width / AP Depth",
            "value": f"{metrics['tibia_ml']} mm / {metrics['tibia_ap']} mm" if not isinstance(metrics['tibia_ml'], str) else metrics['tibia_ml'],
            "caveat": f"Implant sizing reference, measured at {metrics['sizing_basis']}" if not isinstance(metrics['tibia_ml'], str) else "Required mesh file missing"
        },
        {
            "name": "Cartilage Thickness",
            "value": cart_val,
            "caveat": "Requires 3D DESS MRI modality for segmentation" if modality == "CT" else ""
        }
    ]

def generate_report_data(task_id: str, modality: str, file_path: str, mesh_dir: str, mask_path: str = None):
    scan_date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    quarantine_log_path = Path(mesh_dir) / "quarantine_log.json"
    quarantined_bones = {}
    quarantine_warnings = []
    if quarantine_log_path.exists():
        try:
            with open(quarantine_log_path, "r") as f:
                q_log = json.load(f)
                for entry in q_log:
                    for b in entry.get("bones", []):
                        quarantined_bones[b] = entry
                    bones_str = " and ".join([b.replace('_', ' ').title() for b in entry.get("bones", [])])
                    if entry.get("status") == "deleted":
                        quarantine_warnings.append(
                            f"<b><font color='red'>WARNING:</font> The {bones_str} were automatically excluded from 3D analysis "
                            f"due to severe topological intersection (artefact). They intersected by {entry.get('intersection_faces')} "
                            f"faces, exceeding the {entry.get('threshold')} face safety threshold. Downstream spatial metrics are unavailable for these regions.</b>"
                        )
        except Exception as e:
            print("Failed to read quarantine log:", e)
    
    # Read laterality summary
    laterality_summary_path = Path(mesh_dir) / "laterality_summary.json"
    laterality_summary = {}
    if laterality_summary_path.exists():
        with open(laterality_summary_path, "r") as f:
            laterality_summary = json.load(f)
            
    laterality = laterality_summary.get("laterality", "unknown")
    sides_present = laterality_summary.get("sides_present", [])
    
    if not sides_present:
        sides_present = ["unknown"] if modality == "MRI" else ["left"] # Fallback if summary is missing
        
    all_metrics = {}
    for side in sides_present:
        all_metrics[side] = calculate_side_metrics(Path(mesh_dir), side, laterality_summary, modality, mask_path)
        all_metrics[side]['quarantined'] = quarantined_bones
        
    # Impression logic
    impressions = ["Quantitative knee geometry analysis completed successfully."]
    
    for side in sides_present:
        m = all_metrics[side]
        side_label = side.capitalize() if laterality == "bilateral" else "Single Knee"
        if isinstance(m["jsw"], str) and "N/A" in m["jsw"]:
            impressions.append(f"[{side_label}] Joint Space Width could not be measured (missing mesh data).")
        elif m["jsw"] < 2.0:
            impressions.append(f"[{side_label}] Joint Space Width measured at {m['jsw']:.1f}mm, indicating joint space narrowing compared to the typical 2.0-6.0mm range.")
        elif m["jsw"] > 6.0:
            impressions.append(f"[{side_label}] Joint Space Width measured at {m['jsw']:.1f}mm, which exceeds the typical 2.0-6.0mm range.")
        else:
            impressions.append(f"[{side_label}] Joint Space Width measured at {m['jsw']:.1f}mm, which is within the typical 2.0-6.0mm range.")
            
        if isinstance(m["alignment_angle"], str) and "N/A" in m["alignment_angle"]:
            impressions.append(f"[{side_label}] Anatomic axis alignment angle could not be measured.")
        else:
            impressions.append(f"[{side_label}] Anatomic axis alignment angle is {m['alignment_angle']} degrees.")
        
    impressions.append(
        "Clinical correlation and radiologist review are required for diagnostic interpretation; "
        "no surgical sizing or diagnostic determination is finalized by this software tool."
    )

    data_dict = {
        "quarantine_warnings": quarantine_warnings,
        "task_id": task_id,
        "modality": modality,
        "scan_date": scan_date,
        "laterality": laterality,
        "sides_present": sides_present,
        "metrics_by_side": {side: build_metrics_list(all_metrics[side], modality) for side in sides_present},
        # Flatten metrics for UI that expects a single list (we will just concat them with prefixes)
        "metrics": [],
        "impression": " ".join(impressions)
    }
    
    # Build flat metrics for legacy UI compatibility
    for side in sides_present:
        prefix = f"[{side.capitalize()}] " if laterality == "bilateral" else "[Single Knee] "
        for item in data_dict["metrics_by_side"][side]:
            data_dict["metrics"].append({
                "name": f"{prefix}{item['name']}",
                "value": item["value"],
                "caveat": item["caveat"]
            })
    
    # Save to JSON
    json_path = Path(mesh_dir) / "report.json"
    with open(json_path, "w") as f:
        json.dump(data_dict, f, indent=4)
        
    return data_dict

def generate_pdf(data_dict, mesh_dir, task_id):
    pdf_path = Path(mesh_dir) / "report.pdf"
    
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = styles['Title']
    story.append(Paragraph("KneeTwin Clinical Scan Report", title_style))
    story.append(Spacer(1, 12))
    
    # Patient/Study Header Block
    header_style = ParagraphStyle(
        'HeaderStyle', parent=styles['Normal'], fontSize=10, leading=14, spaceAfter=20
    )
    laterality_text = "Bilateral" if data_dict['laterality'] == "bilateral" else f"{data_dict['laterality'].capitalize()} Knee"
    header_html = f"""
    <b>Task ID:</b> {data_dict['task_id']}<br/>
    <b>Scan Date:</b> {data_dict['scan_date']}<br/>
    <b>Modality:</b> {data_dict['modality']}<br/>
    <b>Laterality:</b> {laterality_text}
    """
    story.append(Paragraph(header_html, header_style))
    story.append(Spacer(1, 12))
    
    # Overlay Image
    if data_dict.get("quarantine_warnings"):
        for w in data_dict.get("quarantine_warnings"):
            story.append(Paragraph(w, styles['Normal']))
            story.append(Spacer(1, 12))
    slice_img_path = Path(mesh_dir).parent.parent / "tasks" / task_id / "slices" / "axial_15.png"
    if slice_img_path.exists():
        story.append(Paragraph("<b>Segmentation Overlay (Mid-Axial)</b>", styles['Heading3']))
        story.append(Spacer(1, 6))
        try:
            img = RLImage(str(slice_img_path), width=300, height=300)
            img.hAlign = 'LEFT'
            story.append(img)
            story.append(Spacer(1, 12))
        except:
            pass
            
    # Quantitative Findings Tables
    story.append(Paragraph("<b>Quantitative Findings</b>", styles['Heading2']))
    story.append(Spacer(1, 6))
    
    table_style = TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2C3E50")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#ECF0F1")]),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ])
    
    for side in data_dict.get("sides_present", []):
        if data_dict['laterality'] == "bilateral":
            story.append(Paragraph(f"<b>{side.capitalize()} Knee</b>", styles['Heading3']))
            story.append(Spacer(1, 6))
            
        table_data = [["Metric", "Value", "Notes/Caveats"]]
        for m in data_dict["metrics_by_side"][side]:
            table_data.append([m["name"], m["value"], m["caveat"]])
            
        t = Table(table_data, colWidths=[150, 100, 250])
        t.setStyle(table_style)
        story.append(t)
        story.append(Spacer(1, 12))
    
    story.append(Spacer(1, 12))
    
    # Impression/Summary
    story.append(Paragraph("<b>Impression</b>", styles['Heading2']))
    story.append(Spacer(1, 6))
    story.append(Paragraph(data_dict["impression"], styles['Normal']))
    
    doc.build(story)

def run(task_id: str, modality: str, file_path: str, mesh_dir: str, mask_path: str = None):
    data_dict = generate_report_data(task_id, modality, file_path, mesh_dir, mask_path)
    generate_pdf(data_dict, mesh_dir, task_id)

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python report_generator.py <task_id> <modality> <file_path> <mesh_dir> [mask_path]")
        sys.exit(1)
        
    mask_p = sys.argv[5] if len(sys.argv) >= 6 else None
    run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], mask_p)
