import SimpleITK as sitk
import numpy as np

def test():
    # Create a simple 3D hollow sphere
    arr = np.zeros((20, 20, 20), dtype=np.uint8)
    for z in range(20):
        for y in range(20):
            for x in range(20):
                d = (z-10)**2 + (y-10)**2 + (x-10)**2
                if 16 <= d <= 36:
                    arr[z,y,x] = 1
                    
    img = sitk.GetImageFromArray(arr)
    
    # Fill holes
    fill_filter = sitk.BinaryFillholeImageFilter()
    fill_filter.SetForegroundValue(1)
    # The filter fills completely enclosed background components.
    filled = fill_filter.Execute(img)
    
    filled_arr = sitk.GetArrayFromImage(filled)
    
    print(f"Original volume: {np.sum(arr)}")
    print(f"Filled volume: {np.sum(filled_arr)}")
    
test()
