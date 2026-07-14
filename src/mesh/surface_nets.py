import vtk
import pyvista as pv

def extract_multilabel_surface(mask_path: str, label_map: dict) -> pv.PolyData:
    """
    Generate a non-intersecting multi-label mesh from a volumetric mask using Surface Nets.
    
    Args:
        mask_path: Path to the NIfTI segmentation mask.
        label_map: Dictionary mapping structure names to their integer label values.
                   e.g., CT Bone: {"femur": 1, "tibia": 2, "patella": 3}
                         MRI Cartilage: {"femoral_cartilage": 2, "medial_tibial_cartilage": 4, "lateral_tibial_cartilage": 5}
    
    Returns:
        PyVista PolyData mesh containing all specified structures with shared, non-intersecting boundaries.
    """
    # 1. Read the NIfTI volume
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(mask_path)
    reader.Update()
    
    # Pad the image with 1 voxel of 0s on all sides to ensure closed meshes at volume boundaries
    pad = vtk.vtkImageConstantPad()
    pad.SetInputConnection(reader.GetOutputPort())
    extent = reader.GetOutput().GetExtent()
    pad.SetOutputWholeExtent(extent[0]-1, extent[1]+1, extent[2]-1, extent[3]+1, extent[4]-1, extent[5]+1)
    pad.SetConstant(0)
    pad.Update()
    
    # 2. Extract meshes using SurfaceNets3D
    surfacenets = vtk.vtkSurfaceNets3D()
    surfacenets.SetInputConnection(pad.GetOutputPort())
    
    # Dynamically configure labels based on the provided label_map
    surfacenets.SetNumberOfLabels(len(label_map))
    for i, (label_name, label_value) in enumerate(label_map.items()):
        surfacenets.SetValue(i, label_value)
    
    surfacenets.SmoothingOn()
    surfacenets.GetSmoother().SetNumberOfIterations(15)
    
    surfacenets.Update()
    
    # 3. Convert VTK polydata output to PyVista for easier manipulation
    vtk_polydata = surfacenets.GetOutput()
    pv_mesh = pv.wrap(vtk_polydata)
    
    return pv_mesh
