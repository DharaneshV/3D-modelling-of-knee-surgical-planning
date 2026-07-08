import SimpleITK as sitk
import numpy as np

def combine_registered_labels(ct_bone_mask: sitk.Image, registered_mri_cartilage_mask: sitk.Image, offset: int = 10) -> sitk.Image:
    """
    Merge CT bone labels (e.g., 1-3) with MRI cartilage labels into one multilabel volume.
    This prepares the combined volume for a single Surface Nets pass, ensuring 
    non-intersecting bone-cartilage boundaries.
    
    Context:
    Surface Nets guarantees non-intersecting boundaries only for structures extracted 
    from the SAME multi-label volume. When running CT and MRI tracks independently, 
    the boundaries at the bone-cartilage interface are not guaranteed to be non-intersecting.
    
    For matched datasets (same patient), the proper pipeline is:
    1. Register MRI to CT.
    2. Resample the MRI cartilage label map onto the CT voxel grid.
    3. Use this function to combine the labels.
    4. Run `generate_multilabel_mesh` on the combined output.
    
    Args:
        ct_bone_mask: sitk.Image of the CT bone labels.
        registered_mri_cartilage_mask: sitk.Image of the MRI cartilage labels, 
                                       already resampled to the CT spatial domain.
        offset: Integer offset added to cartilage labels to avoid collision with bone labels.
        
    Returns:
        sitk.Image combining both masks into a single volume.
    """
    # Convert SimpleITK images to numpy arrays
    bone_arr = sitk.GetArrayFromImage(ct_bone_mask)
    cartilage_arr = sitk.GetArrayFromImage(registered_mri_cartilage_mask)
    
    # Ensure they have the same shape
    if bone_arr.shape != cartilage_arr.shape:
        raise ValueError(f"Shape mismatch: CT mask is {bone_arr.shape}, but MRI mask is {cartilage_arr.shape}. "
                         f"Did you forget to resample the MRI mask onto the CT grid?")
        
    # Offset the cartilage labels (ignoring background 0)
    cartilage_offset_arr = np.where(cartilage_arr > 0, cartilage_arr + offset, 0)
    
    # Merge them. In case of overlap, we can prioritize bone or cartilage. 
    # Here, we prioritize cartilage if there's an overlap.
    combined_arr = np.where(cartilage_offset_arr > 0, cartilage_offset_arr, bone_arr)
    
    # Convert back to SimpleITK image
    combined_img = sitk.GetImageFromArray(combined_arr)
    combined_img.CopyInformation(ct_bone_mask)
    
    return combined_img
