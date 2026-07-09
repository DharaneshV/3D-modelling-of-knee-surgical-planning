import numpy as np
import SimpleITK as sitk
from skimage.measure import marching_cubes, mesh_surface_area
from scipy.ndimage import distance_transform_edt

cases = ['oaizib_002', 'oaizib_405', 'oaizib_406']
# For 002 it's in data/segmentations/oaizib_002_label.nii.gz
# For 405, 406 it's in temp_cartimorph_io/gt_test_15/oaizib_405_label.nii.gz (ground truth)
# Wait, for 405/406 I have the resampled *predictions* in output_test_15 ?
# No, earlier I wrote calculate_test_dice_15.py which did resampling in memory.
# Let's just use the ground truth masks of 405 and 406 for this geometric test.

paths = [
    'data/segmentations/oaizib_002_label.nii.gz',
    'temp_cartimorph_io/gt_test_15/oaizib_405_label.nii.gz',
    'temp_cartimorph_io/gt_test_15/oaizib_406_label.nii.gz'
]

cartilage_labels = {
    2: 'Femoral Cartilage',
    4: 'Medial Tibial Cartilage',
    5: 'Lateral Tibial Cartilage'
}

for case_name, path in zip(cases, paths):
    print(f"\n=== Case: {case_name} ===")
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing() # x, y, z
    voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]
    sampling = [spacing[2], spacing[1], spacing[0]] # z, y, x
    
    # Bone masks for EDT
    femur_mask = (arr == 1)
    tibia_mask = (arr == 3)
    
    if np.any(femur_mask):
        edt_femur = distance_transform_edt(~femur_mask, sampling=sampling)
    else:
        edt_femur = None
        
    if np.any(tibia_mask):
        edt_tibia = distance_transform_edt(~tibia_mask, sampling=sampling)
    else:
        edt_tibia = None
    
    for label_val, name in cartilage_labels.items():
        mask = (arr == label_val)
        if not np.any(mask):
            print(f"{name}: Missing")
            continue
            
        # Volume
        vol_mm3 = np.sum(mask) * voxel_vol_mm3
        
        # EDT Thickness
        edt = edt_femur if label_val == 2 else edt_tibia
        if edt is not None:
            edt_thickness = np.mean(edt[mask]) * 2.0
        else:
            edt_thickness = 0.0
            
        # Surface Area Thickness
        # Padding mask to avoid boundary issues during marching cubes
        padded = np.pad(mask, pad_width=1, mode='constant', constant_values=0)
        verts, faces, normals, values = marching_cubes(padded, level=0.5, spacing=sampling)
        sa_total = mesh_surface_area(verts, faces)
        # Approximate interface area is half the total surface area
        sa_interface = sa_total / 2.0
        
        sa_thickness = vol_mm3 / sa_interface
        
        print(f"{name:25s} | EDTx2: {edt_thickness:.2f} mm | Vol/(SA/2): {sa_thickness:.2f} mm | Diff: {abs(edt_thickness - sa_thickness):.2f} mm")

