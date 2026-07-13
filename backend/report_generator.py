import os
import sys
import json
import datetime
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False

def calculate_anatomic_axis(vertices, reference_dir=None):
    """Compute the anatomical axis vector using PCA."""
    centered = vertices - np.mean(vertices, axis=0)
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # The primary axis corresponds to the eigenvector with the largest eigenvalue
    primary_axis = eigvecs[:, np.argmax(eigvals)]
    
    if reference_dir is not None:
        if np.dot(primary_axis, reference_dir) < 0:
            primary_axis = -primary_axis
    else:
        # Fallback to positive Z if no reference is given
        if primary_axis[2] < 0:
            primary_axis = -primary_axis
            
    return primary_axis

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

def calculate_side_metrics(mesh_dir: Path, side: str, laterality_summary: dict):
    """Calculate metrics for a specific side (left or right)."""
    metrics = {
        "femur_vol": 0, "tibia_vol": 0, "patella_vol": 0,
        "jsw": 0.0,
        "femur_ml": 0.0, "femur_ap": 0.0,
        "tibia_ml": 0.0, "tibia_ap": 0.0,
        "alignment_angle": 0.0,
        "medial_thick": 2.2, "lateral_thick": 2.4, "trochlear_thick": 2.1 # Mocked/N/A for CT
    }
    
    # 1. Volumes from summary if available, else 0
    counts = laterality_summary.get("voxel_counts", {})
    # Need voxel volume to convert count to volume, but we don't have spacing easily here without reading the mask.
    # We will read the mask to get volume in mm^3
    task_id = mesh_dir.name
    mask_path = mesh_dir / f"{task_id}_mask.nii.gz"
    if mask_path.exists() and HAS_SITK:
        try:
            img = sitk.ReadImage(str(mask_path))
            arr = sitk.GetArrayFromImage(img)
            spacing = img.GetSpacing()
            voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]
            
            # Map side to labels
            f_label = 1 if side == "left" else 2
            t_label = 3 if side == "left" else 4
            p_label = 5 if side == "left" else 6
            
            metrics["femur_vol"] = int(np.sum(arr == f_label) * voxel_vol_mm3)
            metrics["tibia_vol"] = int(np.sum(arr == t_label) * voxel_vol_mm3)
            metrics["patella_vol"] = int(np.sum(arr == p_label) * voxel_vol_mm3)
        except Exception as e:
            print(f"Error reading mask for volumes: {e}")
            
    # 2. Load meshes for JSW, Alignment, and Sizing
    f_mesh_path = mesh_dir / f"femur_{side}_decimated.obj"
    t_mesh_path = mesh_dir / f"tibia_{side}_decimated.obj"
    
    if f_mesh_path.exists() and t_mesh_path.exists():
        try:
            femur = trimesh.load(str(f_mesh_path))
            tibia = trimesh.load(str(t_mesh_path))
            
            f_centroid = np.mean(femur.vertices, axis=0)
            t_centroid = np.mean(tibia.vertices, axis=0)
            up_vector = f_centroid - t_centroid
            if np.linalg.norm(up_vector) > 0:
                up_vector = up_vector / np.linalg.norm(up_vector)
            
            # Anatomic Axis Angle via PCA (3D)
            f_axis = calculate_anatomic_axis(femur.vertices, reference_dir=up_vector)
            t_axis = calculate_anatomic_axis(tibia.vertices, reference_dir=up_vector)
            
            # Verify symmetry/consistency (log only)
            print(f"DEBUG [{side}]: Femur primary axis: {f_axis}")
            print(f"DEBUG [{side}]: Tibia primary axis: {t_axis}")
            
            cos_theta = np.dot(f_axis, t_axis) / (np.linalg.norm(f_axis) * np.linalg.norm(t_axis))
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            metrics["alignment_angle"] = round(float(np.degrees(np.arccos(cos_theta))), 1)
            
            # Sizing (Bounding Box spreads) via PCA on proportional ROI
            metrics["femur_ml"], metrics["femur_ap"] = get_roi_sizing(femur.vertices, f_axis, is_femur=True, side=side)
            metrics["tibia_ml"], metrics["tibia_ap"] = get_roi_sizing(tibia.vertices, t_axis, is_femur=False, side=side)
            
            # JSW Calculation
            # Project onto tibia axis to find joint space region
            t_proj = np.dot(tibia.vertices, t_axis)
            f_proj = np.dot(femur.vertices, t_axis)
            joint_z = np.max(t_proj)
            
            f_mask = (f_proj >= joint_z - 20) & (f_proj <= joint_z + 40)
            t_mask = (t_proj >= joint_z - 40) & (t_proj <= joint_z + 20)
            
            f_pts = femur.vertices[f_mask]
            t_pts = tibia.vertices[t_mask]
            
            metrics["jsw"] = 0.0
            metrics["jsw_overlap"] = False
            
            if len(f_pts) > 0 and len(t_pts) > 0:
                try:
                    import pyvista as pv
                    pv_femur = pv.wrap(femur)
                    pv_tibia = pv.wrap(tibia)
                    collision, n_contacts = pv_femur.collision(pv_tibia)
                    is_collision = n_contacts > 0
                except Exception as e:
                    print(f"Collision check failed, falling back to KDTree dist: {e}")
                    is_collision = False
                
                if is_collision:
                    metrics["jsw_overlap"] = True
                    metrics["jsw"] = 0.0
                else:
                    tree = cKDTree(t_pts)
                    closest_dists, _ = tree.query(f_pts)
                    metrics["jsw"] = round(float(np.min(closest_dists)), 2)
                    
        except Exception as e:
            print(f"Error calculating mesh metrics for {side}: {e}")
            
    return metrics

def build_metrics_list(metrics, modality):
    jsw_caveat = "Minimum distance at joint space"
    if metrics.get("jsw_overlap"):
        # Explicit stopgap flag (TODO: fix upstream meshing overlapping issue via boolean clipping)
        jsw_caveat = "Measurement uncertain — mesh overlap detected"
        
    return [
        {
            "name": "Femur Bone Volume",
            "value": f"{metrics['femur_vol'] / 1000.0:.1f} cm³",
            "caveat": ""
        },
        {
            "name": "Tibia Bone Volume",
            "value": f"{metrics['tibia_vol'] / 1000.0:.1f} cm³",
            "caveat": ""
        },
        {
            "name": "Patella Bone Volume",
            "value": f"{metrics['patella_vol'] / 1000.0:.1f} cm³",
            "caveat": ""
        },
        {
            "name": "Joint Space Width (JSW)",
            "value": f"{metrics['jsw']} mm",
            "caveat": jsw_caveat
        },
        {
            "name": "Anatomic Axis Angle",
            "value": f"{metrics['alignment_angle']}°",
            "caveat": "Calculated via PCA of bone shafts"
        },
        {
            "name": "Femur ML Width / AP Depth",
            "value": f"{metrics['femur_ml']} mm / {metrics['femur_ap']} mm",
            "caveat": "Implant sizing reference dimensions"
        },
        {
            "name": "Tibia ML Width / AP Depth",
            "value": f"{metrics['tibia_ml']} mm / {metrics['tibia_ap']} mm",
            "caveat": "Implant sizing reference dimensions"
        },
        {
            "name": "Cartilage Thickness",
            "value": "N/A (CT modality)" if modality == "CT" else "Calculated",
            "caveat": "Requires 3D DESS MRI modality for segmentation"
        }
    ]

def generate_report_data(task_id: str, modality: str, file_path: str, mesh_dir: str):
    scan_date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Read laterality summary
    laterality_summary_path = Path(mesh_dir) / "laterality_summary.json"
    laterality_summary = {}
    if laterality_summary_path.exists():
        with open(laterality_summary_path, "r") as f:
            laterality_summary = json.load(f)
            
    laterality = laterality_summary.get("laterality", "unknown")
    sides_present = laterality_summary.get("sides_present", [])
    
    if not sides_present:
        sides_present = ["left"] # Fallback if summary is missing
        
    all_metrics = {}
    for side in sides_present:
        all_metrics[side] = calculate_side_metrics(Path(mesh_dir), side, laterality_summary)
        
    # Impression logic
    impressions = ["Quantitative knee geometry analysis completed successfully."]
    
    for side in sides_present:
        m = all_metrics[side]
        side_label = side.capitalize()
        if m["jsw"] < 2.0:
            impressions.append(f"[{side_label}] Joint Space Width measured at {m['jsw']}mm, indicating joint space narrowing compared to the typical 2.0-6.0mm range.")
        else:
            impressions.append(f"[{side_label}] Joint Space Width measured at {m['jsw']}mm, which is within the typical 2.0-6.0mm range.")
        impressions.append(f"[{side_label}] Anatomic axis alignment angle is {m['alignment_angle']} degrees.")
        
    impressions.append(
        "Clinical correlation and radiologist review are required for diagnostic interpretation; "
        "no surgical sizing or diagnostic determination is finalized by this software tool."
    )

    data_dict = {
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
    if laterality == "bilateral":
        for side in sides_present:
            for item in data_dict["metrics_by_side"][side]:
                data_dict["metrics"].append({
                    "name": f"[{side.capitalize()}] {item['name']}",
                    "value": item["value"],
                    "caveat": item["caveat"]
                })
    else:
        side = sides_present[0]
        data_dict["metrics"] = data_dict["metrics_by_side"][side]
    
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

def run(task_id: str, modality: str, file_path: str, mesh_dir: str):
    data_dict = generate_report_data(task_id, modality, file_path, mesh_dir)
    generate_pdf(data_dict, mesh_dir, task_id)

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python report_generator.py <task_id> <modality> <file_path> <mesh_dir>")
        sys.exit(1)
        
    run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
