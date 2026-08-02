import sys
from pathlib import Path

# Add root directory to sys.path so we can import get_jsw
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import SimpleITK as sitk
from skimage.measure import marching_cubes, mesh_surface_area
from get_jsw import compute_jsw

def extract_cart_features(mask_path: str) -> np.ndarray:
    """
    Extracts a 9-dimensional feature vector from a CartiMorph MRI mask.
    
    Args:
        mask_path: Path to the CartiMorph output .nii.gz file (0.5mm isotropic expected).
        
    Returns:
        np.ndarray of shape (9,) containing the extracted features.
    """
    img = sitk.ReadImage(mask_path)
    arr = sitk.GetArrayFromImage(img)
    
    # spacing is (x, y, z) in SimpleITK, convert to (z, y, x) for array operations
    sitk_spacing = img.GetSpacing()
    spacing = np.array([sitk_spacing[2], sitk_spacing[1], sitk_spacing[0]])
    
    voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]
    voxel_vol_cm3 = voxel_vol_mm3 / 1000.0
    
    labels = {
        'FC': 2,
        'MTiC': 4,
        'LTiC': 5
    }
    
    features = np.zeros(9, dtype=np.float32)
    
    # 1. FC Vol
    features[0] = np.sum(arr == labels['FC']) * voxel_vol_cm3
    
    def calc_thickness(label):
        mask = arr == label
        if not np.any(mask):
            return 0.0
        padded = np.pad(mask, pad_width=1, mode='constant', constant_values=0)
        verts, faces, _, _ = marching_cubes(padded, level=0.5, spacing=spacing)
        sa_total = mesh_surface_area(verts, faces)
        sa_interface = sa_total / 2.0
        vol_mm3 = np.sum(mask) * voxel_vol_mm3
        return vol_mm3 / sa_interface if sa_interface > 0 else 0.0

    # 2. FC Mean Thickness
    features[1] = calc_thickness(labels['FC'])
    
    # 3. FC ML-extent (mm)
    fc_coords = np.argwhere(arr == labels['FC'])
    if len(fc_coords) > 0:
        # X is the last axis in (Z, Y, X) array
        x_min, x_max = np.min(fc_coords[:, 2]), np.max(fc_coords[:, 2])
        features[2] = (x_max - x_min + 1) * spacing[2]
    else:
        features[2] = 0.0
        
    # 4. MTiC Vol
    features[3] = np.sum(arr == labels['MTiC']) * voxel_vol_cm3
    
    # 5. MTiC Mean Thickness
    features[4] = calc_thickness(labels['MTiC'])
    
    # 6. LTiC Vol
    features[5] = np.sum(arr == labels['LTiC']) * voxel_vol_cm3
    
    # 7. LTiC Mean Thickness
    features[6] = calc_thickness(labels['LTiC'])
    
    # 8. JSW Medial (FC vs MTiC)
    jsw_medial = compute_jsw(arr, labels['FC'], labels['MTiC'], spacing)
    features[7] = 0.0 if np.isinf(jsw_medial) else jsw_medial
    
    # 9. JSW Lateral (FC vs LTiC)
    jsw_lateral = compute_jsw(arr, labels['FC'], labels['LTiC'], spacing)
    features[8] = 0.0 if np.isinf(jsw_lateral) else jsw_lateral
    
    return features
