import os
import SimpleITK as sitk

def detect_modality(file_path: str) -> dict:
    """
    Detects whether a scan is CT or MRI based on intensity values and filename.
    Returns a dict with 'modality' ('CT', 'MRI', or 'UNKNOWN') and 'reason'.
    """
    filename = os.path.basename(file_path).lower()
    
    try:
        # Load the image using SimpleITK
        image = sitk.ReadImage(file_path)
        
        # Get min and max intensities
        # sitk.MinimumMaximum() filter computes the min/max efficiently
        min_max_filter = sitk.MinimumMaximumImageFilter()
        min_max_filter.Execute(image)
        
        min_intensity = min_max_filter.GetMinimum()
        max_intensity = min_max_filter.GetMaximum()
        
        # Heuristic: CT has air at -1000 and bone up to +3000
        # MRI doesn't typically have -1000 floor.
        if min_intensity <= -900 and max_intensity >= 700:
            return {
                "modality": "CT",
                "reason": f"Intensity range [{min_intensity:.1f}, {max_intensity:.1f}] matches CT HU scaling."
            }
        else:
            return {
                "modality": "MRI",
                "reason": f"Intensity range [{min_intensity:.1f}, {max_intensity:.1f}] does not match CT HU scaling (likely MRI)."
            }
            
    except Exception as e:
        # Fallback to filename if loading fails or something goes wrong
        print(f"Error reading intensity for modality detection: {e}")
        if "ct" in filename or "sts" in filename:
            return {
                "modality": "CT",
                "reason": f"Fallback: filename '{filename}' suggests CT."
            }
        elif "mri" in filename or "oaizib" in filename:
            return {
                "modality": "MRI",
                "reason": f"Fallback: filename '{filename}' suggests MRI."
            }
        else:
            return {
                "modality": "UNKNOWN",
                "reason": f"Could not determine modality. Intensity reading failed and filename '{filename}' is ambiguous."
            }
