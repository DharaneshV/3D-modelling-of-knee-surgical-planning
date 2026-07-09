import os
import sys
import json
import datetime
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False

def generate_report(task_id: str, modality: str, file_path: str, mesh_dir: str):
    pdf_path = Path(mesh_dir) / "report.pdf"
    
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = styles['Title']
    story.append(Paragraph("KneeTwin Clinical Scan Report", title_style))
    story.append(Spacer(1, 12))
    
    # Metadata
    metadata_style = styles['Normal']
    story.append(Paragraph(f"<b>Task ID:</b> {task_id}", metadata_style))
    story.append(Paragraph(f"<b>Date:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", metadata_style))
    story.append(Paragraph(f"<b>Detected Modality:</b> {modality}", metadata_style))
    story.append(Spacer(1, 24))
    
    # Original Scan Section
    story.append(Paragraph("Original Scan Details", styles['Heading2']))
    
    if HAS_SITK and os.path.exists(file_path):
        try:
            reader = sitk.ImageFileReader()
            reader.SetFileName(file_path)
            reader.ReadImageInformation()
            size = reader.GetSize()
            spacing = reader.GetSpacing()
            
            scan_info = [
                ["Dimensions", f"{size[0]} x {size[1]} x {size[2]}"],
                ["Spacing (mm)", f"{spacing[0]:.2f} x {spacing[1]:.2f} x {spacing[2]:.2f}"]
            ]
            
            t = Table(scan_info, colWidths=[150, 300])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (0,-1), colors.lightgrey),
                ('TEXTCOLOR', (0,0), (-1,-1), colors.black),
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                ('GRID', (0,0), (-1,-1), 1, colors.black)
            ]))
            story.append(t)
        except Exception as e:
            story.append(Paragraph(f"Failed to read image metadata: {e}", metadata_style))
    else:
        story.append(Paragraph("Scan metadata not available.", metadata_style))
        
    story.append(Spacer(1, 24))
    
    # Segmentation Overview
    story.append(Paragraph("Segmentation Overview", styles['Heading2']))
    story.append(Paragraph("The volumetric scan was successfully processed and the corresponding 3D meshes have been generated.", metadata_style))
    story.append(Spacer(1, 12))
    
    # Extract mesh info from manifest
    manifest_path = Path(mesh_dir) / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
            
        mesh_data = [["Anatomical Part", "Mesh File"]]
        for part in manifest.get("parts", []):
            mesh_data.append([part.get("label", ""), part.get("file", "")])
            
        t = Table(mesh_data, colWidths=[200, 250])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2C3E50")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
            ('GRID', (0,0), (-1,-1), 1, colors.black),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#ECF0F1")])
        ]))
        story.append(t)
    else:
        story.append(Paragraph("Manifest not found.", metadata_style))
        
    story.append(Spacer(1, 24))
    
    # Limitations Section
    story.append(Paragraph("Limitations & Caveats", styles['Heading2']))
    limitations = (
        "<b>Disclaimer:</b> Measurements and segmentations provided in this report are fully automated approximations "
        "and should not be used as the sole basis for clinical decisions. Volumes, surface areas, and cartilage thicknesses "
        "are algorithmically inferred. Any Dice scores are computed against a reference model and do not constitute manual ground truth."
    )
    story.append(Paragraph(limitations, metadata_style))
    
    # Render PDF
    doc.build(story)
    
if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python report_generator.py <task_id> <modality> <file_path> <mesh_dir>")
        sys.exit(1)
        
    generate_report(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
