import sys
from pathlib import Path
sys.path.append(".")
from backend.report_generator import calculate_side_metrics

mask_path = "meshes/5b567137-4fe5-4e1f-acff-72a631d70d6c/5b567137-4fe5-4e1f-acff-72a631d70d6c_mask.nii.gz"
mesh_dir = Path("meshes/5b567137-4fe5-4e1f-acff-72a631d70d6c")
laterality_summary = {}

left_metrics = calculate_side_metrics(mesh_dir, "left", laterality_summary, "CT", mask_path)
right_metrics = calculate_side_metrics(mesh_dir, "right", laterality_summary, "CT", mask_path)

print(f"Old Left JSW (Mask-based logic): {left_metrics.get('jsw')}")
print(f"Old Right JSW (Mask-based logic): {right_metrics.get('jsw')}")
