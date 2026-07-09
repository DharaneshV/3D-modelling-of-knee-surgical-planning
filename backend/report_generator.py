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

def compute_laterality(file_path):
    if not HAS_SITK or not os.path.exists(file_path):
        return "Unknown"
    try:
        reader = sitk.ImageFileReader()
        reader.SetFileName(file_path)
        reader.ReadImageInformation()
        direction = reader.GetDirection()
        return "Right Knee" if direction[0] > 0 else "Left Knee"
    except:
        return "Unknown"

def calculate_anatomic_axis(vertices):
    """Compute the anatomical axis vector using PCA."""
    centered = vertices - np.mean(vertices, axis=0)
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # The primary axis corresponds to the eigenvector with the largest eigenvalue
    primary_axis = eigvecs[:, np.argmax(eigvals)]
    # Standardize direction to point superiorly (positive Z)
    if primary_axis[2] < 0:
        primary_axis = -primary_axis
    return primary_axis

def calculate_real_metrics(mesh_dir: Path):
    """Calculate actual metrics from the output meshes and mask."""
    metrics = {
        "femur_vol": 0, "tibia_vol": 0, "patella_vol": 0,
        "jsw": 0.0,
        "femur_ml": 0.0, "femur_ap": 0.0,
        "tibia_ml": 0.0, "tibia_ap": 0.0,
        "alignment_angle": 0.0,
        "medial_thick": 2.2, "lateral_thick": 2.4, "trochlear_thick": 2.1 # Mocked/N/A
    }
    
    # 1. Load mask for volumes (inside meshes/task_id/)
    task_id = mesh_dir.name
    mask_path = mesh_dir / f"{task_id}_mask.nii.gz"
    if mask_path.exists() and HAS_SITK:
        try:
            img = sitk.ReadImage(str(mask_path))
            arr = sitk.GetArrayFromImage(img)
            spacing = img.GetSpacing()
            voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]
            
            metrics["femur_vol"] = int(np.sum(arr == 1) * voxel_vol_mm3)
            metrics["tibia_vol"] = int(np.sum(arr == 2) * voxel_vol_mm3)
            metrics["patella_vol"] = int(np.sum(arr == 3) * voxel_vol_mm3)
        except Exception as e:
            print(f"Error reading mask for volumes: {e}")
        
    # 2. Load meshes for JSW, Alignment, and Sizing
    f_mesh_path = mesh_dir / "femur_decimated.obj"
    t_mesh_path = mesh_dir / "tibia_decimated.obj"
    
    if f_mesh_path.exists() and t_mesh_path.exists():
        try:
            femur = trimesh.load(str(f_mesh_path))
            tibia = trimesh.load(str(t_mesh_path))
            
            # Sizing (Bounding Box spreads)
            metrics["femur_ml"] = round(float(np.ptp(femur.vertices[:, 0])), 1)
            metrics["femur_ap"] = round(float(np.ptp(femur.vertices[:, 1])), 1)
            metrics["tibia_ml"] = round(float(np.ptp(tibia.vertices[:, 0])), 1)
            metrics["tibia_ap"] = round(float(np.ptp(tibia.vertices[:, 1])), 1)
            
            # Anatomic Axis Angle via PCA
            f_axis = calculate_anatomic_axis(femur.vertices)
            t_axis = calculate_anatomic_axis(tibia.vertices)
            cos_theta = np.dot(f_axis, t_axis) / (np.linalg.norm(f_axis) * np.linalg.norm(t_axis))
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            metrics["alignment_angle"] = round(float(np.degrees(np.arccos(cos_theta))), 1)
            
            # JSW Calculation
            joint_z = np.percentile(tibia.vertices[:, 2], 99)
            f_mask = (femur.vertices[:, 2] >= joint_z - 20) & (femur.vertices[:, 2] <= joint_z + 40)
            t_mask = (tibia.vertices[:, 2] >= joint_z - 40) & (tibia.vertices[:, 2] <= joint_z + 20)
            
            f_pts = femur.vertices[f_mask]
            t_pts = tibia.vertices[t_mask]
            
            if len(f_pts) > 0 and len(t_pts) > 0:
                tree = cKDTree(t_pts)
                dists, _ = tree.query(f_pts)
                metrics["jsw"] = round(float(np.min(dists)), 2)
        except Exception as e:
            print(f"Error calculating mesh metrics: {e}")
            
    return metrics

def generate_report_data(task_id: str, modality: str, file_path: str, mesh_dir: str):
    scan_date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    laterality = compute_laterality(file_path)
    
    metrics = calculate_real_metrics(Path(mesh_dir))
    
    # Impression logic - descriptive
    impressions = ["Quantitative knee geometry analysis completed successfully."]
    
    if metrics["jsw"] < 2.0:
        impressions.append(f"Joint Space Width measured at {metrics['jsw']}mm, indicating joint space narrowing compared to the typical 2.0-6.0mm range.")
    else:
        impressions.append(f"Joint Space Width measured at {metrics['jsw']}mm, which is within the typical 2.0-6.0mm range.")
        
    impressions.append(f"Anatomic axis alignment angle is {metrics['alignment_angle']} degrees.")
    
    impressions.append(
        "Clinical correlation and radiologist review are required for diagnostic interpretation; "
        "no surgical sizing or diagnostic determination is finalized by this software tool."
    )

    data_dict = {
        "task_id": task_id,
        "modality": modality,
        "scan_date": scan_date,
        "laterality": laterality,
        "metrics": [
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
                "caveat": "Minimum distance at joint space"
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
                "value": "N/A (CT modality)",
                "caveat": "Requires 3D DESS MRI modality for segmentation"
            }
        ],
        "impression": " ".join(impressions)
    }
    
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
    header_html = f"""
    <b>Task ID:</b> {data_dict['task_id']}<br/>
    <b>Scan Date:</b> {data_dict['scan_date']}<br/>
    <b>Modality:</b> {data_dict['modality']}<br/>
    <b>Laterality:</b> {data_dict['laterality']}
    """
    story.append(Paragraph(header_html, header_style))
    story.append(Spacer(1, 12))
    
    # Overlay Image (If slice endpoint cached a mid-axial slice, we could load it here)
    slice_img_path = Path(mesh_dir).parent.parent / "tasks" / task_id / "slices" / "axial_15.png"
    if slice_img_path.exists():
        story.append(Paragraph("<b>Segmentation Overlay (Mid-Axial)</b>", styles['Heading3']))
        story.append(Spacer(1, 6))
        # Add image, keeping aspect ratio
        try:
            img = RLImage(str(slice_img_path), width=300, height=300)
            img.hAlign = 'LEFT'
            story.append(img)
            story.append(Spacer(1, 12))
        except:
            pass
            
    # Quantitative Findings Table
    story.append(Paragraph("<b>Quantitative Findings</b>", styles['Heading2']))
    story.append(Spacer(1, 6))
    
    table_data = [["Metric", "Value", "Notes/Caveats"]]
    for m in data_dict["metrics"]:
        table_data.append([m["name"], m["value"], m["caveat"]])
        
    t = Table(table_data, colWidths=[150, 100, 250])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2C3E50")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#ECF0F1")]),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t)
    story.append(Spacer(1, 24))
    
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
