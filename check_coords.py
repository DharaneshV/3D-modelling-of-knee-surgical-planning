import SimpleITK as sitk
import pyvista as pv

case = 'STS_006'
img = sitk.ReadImage(f"data/ct_knee/case01_{case}_bone_mask.nii.gz")
print(f"Mask Origin: {img.GetOrigin()}")
print(f"Mask Spacing: {img.GetSpacing()}")
print(f"Mask Direction: {img.GetDirection()}")

mesh = pv.read(f"data/ct_knee/{case}_meshes/femur_decimated.obj")
print(f"Mesh bounds: {mesh.bounds}")
