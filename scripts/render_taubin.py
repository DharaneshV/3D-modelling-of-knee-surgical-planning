import pyvista as pv
import vtk
import SimpleITK as sitk

def extract_unsmoothed(mask_path: str, label_map: dict) -> pv.PolyData:
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(mask_path)
    reader.Update()
    
    pad = vtk.vtkImageConstantPad()
    pad.SetInputConnection(reader.GetOutputPort())
    extent = reader.GetOutput().GetExtent()
    pad.SetOutputWholeExtent(extent[0]-1, extent[1]+1, extent[2]-1, extent[3]+1, extent[4]-1, extent[5]+1)
    pad.SetConstant(0)
    pad.Update()
    
    surfacenets = vtk.vtkSurfaceNets3D()
    surfacenets.SetInputConnection(pad.GetOutputPort())
    surfacenets.SetNumberOfLabels(len(label_map))
    for i, (label_name, label_value) in enumerate(label_map.items()):
        surfacenets.SetValue(i, label_value)
    
    surfacenets.SmoothingOff()
    surfacenets.Update()
    
    return pv.wrap(surfacenets.GetOutput())

mask_path = 'meshes/a0810f6b-f014-44df-91ff-0428436c59e5/a0810f6b-f014-44df-91ff-0428436c59e5_mask.nii.gz'
label_map = {"femur_right": 2, "tibia_right": 4}

raw = extract_unsmoothed(mask_path, label_map)
labels = raw.cell_data['BoundaryLabels']

f_mask = (labels[:,0]==2) | (labels[:,1]==2)
t_mask = (labels[:,0]==4) | (labels[:,1]==4)

f = raw.extract_cells(f_mask).extract_surface(algorithm='dataset_surface').clean()
t = raw.extract_cells(t_mask).extract_surface(algorithm='dataset_surface').clean()

f_smooth = f.smooth_taubin(n_iter=15, pass_band=0.1)
t_smooth = t.smooth_taubin(n_iter=15, pass_band=0.1)

col, n = f_smooth.collision(t_smooth)
contact_indices = col.field_data["ContactCells"]
contact_cells = col.extract_cells(contact_indices)

p = pv.Plotter(off_screen=True)
p.add_mesh(f_smooth, color='tan', opacity=0.3)
p.add_mesh(t_smooth, color='green', opacity=0.3)
p.add_mesh(contact_cells, color='red')
p.camera_position = 'yz'
p.screenshot("scratch/sts017_taubin_view.png")

print(f"Taubin collision faces: {n}")
