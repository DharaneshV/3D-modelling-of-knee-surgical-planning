import SimpleITK as sitk
import numpy as np

cases = ['STS_006', 'STS_035', 'STS_043', 'STS_051']

for case in cases:
    print(f"\n--- {case} ---")
    our_mask_path = f"data/ct_knee/case01_{case}_bone_mask.nii.gz"
    img = sitk.ReadImage(our_mask_path)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(img)
    
    for l, name in [(1, 'Femur'), (2, 'Tibia')]:
        if stats.HasLabel(l):
            bbox = stats.GetBoundingBox(l)
            # bbox is (x_start, y_start, z_start, x_len, y_len, z_len)
            x_len = bbox[3] * img.GetSpacing()[0]
            y_len = bbox[4] * img.GetSpacing()[1]
            z_len = bbox[5] * img.GetSpacing()[2]
            aspect_ratio = z_len / max(x_len, y_len)
            print(f"{name} Aspect Ratio: {aspect_ratio:.2f} (z_len={z_len:.1f}, max_xy={max(x_len, y_len):.1f})")
