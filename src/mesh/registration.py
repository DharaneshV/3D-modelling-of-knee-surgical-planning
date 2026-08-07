# =============================================================================
# ARCHIVED — not called by the live pipeline.
#
# This module supported per-case CT–MRI registration: aligning an MRI volume to
# a CT volume so bone (from CT) and cartilage (from MRI) could be fused into one
# coordinate space.
#
# It is no longer used because the per-case pipeline never fuses the two
# modalities. A case is processed as CT *or* MRI, and the MRI track gets its
# bone from CartiMorph's native labels 1/3 in the same segmentation pass as
# cartilage — so there are no two volumes to bring into alignment.
#
# Kept for reference rather than deleted, matching the convention used for
# src/synthesis/bone_from_mri.py and backend/mesh_processing/boolean_resolution.py.
# Do NOT re-import without re-establishing why two modalities need fusing.
# =============================================================================
import SimpleITK as sitk

def register_mri_to_ct(fixed_ct_image: sitk.Image, moving_mri_image: sitk.Image) -> sitk.Transform:
    """
    Perform rigid/affine registration to align an MRI volume (moving) to a CT volume (fixed).
    Uses Mattes Mutual Information which is standard for multi-modal registration.
    
    Args:
        fixed_ct_image: sitk.Image of the CT scan.
        moving_mri_image: sitk.Image of the MRI scan.
        
    Returns:
        sitk.Transform representing the registration mapping.
    """
    # 1. Setup the registration method
    registration_method = sitk.ImageRegistrationMethod()

    # Metric: Mattes Mutual Information is appropriate for multi-modal (CT/MRI)
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)

    # Interpolator
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer: Gradient Descent
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0, 
        numberOfIterations=100, 
        convergenceMinimumValue=1e-6, 
        convergenceWindowSize=10
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup initial transform (Rigid 3D) aligned by geometry center
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_ct_image, 
        moving_mri_image, 
        sitk.Euler3DTransform(), 
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # 2. Execute registration
    final_transform = registration_method.Execute(fixed_ct_image, moving_mri_image)
    
    return final_transform

def apply_registration(moving_image: sitk.Image, fixed_image: sitk.Image, transform: sitk.Transform, is_label: bool = False) -> sitk.Image:
    """
    Apply a transform to a moving image, resampling it onto the fixed image's grid.
    
    Args:
        moving_image: The image to resample.
        fixed_image: The reference grid.
        transform: The transformation to apply.
        is_label: True if the moving image is a segmentation mask (uses Nearest Neighbor).
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetTransform(transform)
    
    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(moving_image.GetPixelIDValue())
        
    return resampler.Execute(moving_image)
