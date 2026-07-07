import os
import glob
import torch
import numpy as np
import SimpleITK as sitk
from monai.inferers import sliding_window_inference
from monai.data import Dataset, DataLoader
from monai.transforms import AsDiscrete
import gc

from src.segmentation.dataset import get_inference_transforms
from src.segmentation.train_soft_tissue import get_model, NUM_CLASSES

def predict_case(model, device, image_path, output_path, use_amp=True):
    """
    Run inference on a single MRI volume.
    """
    transforms = get_inference_transforms()
    
    # Process single item as a dataset
    ds = Dataset(data=[{"image": image_path}], transform=transforms)
    loader = DataLoader(ds, batch_size=1, num_workers=0)
    
    model.eval()
    
    # The output size from sliding window inference
    roi_size = (96, 96, 96)
    sw_batch_size = 4
    
    with torch.no_grad():
        for batch_data in loader:
            inputs = batch_data["image"].to(device)
            
            with torch.amp.autocast(device_type="cuda" if "cuda" in str(device) else "cpu", enabled=use_amp):
                outputs = sliding_window_inference(inputs, roi_size, sw_batch_size, model)
            
            # Post-processing: argmax across classes
            post_pred = AsDiscrete(argmax=True)
            output_mask = post_pred(outputs[0]).cpu().numpy() # Shape: (1, Z, Y, X)
            output_mask = output_mask.squeeze(0).astype(np.uint16)
            
            # Load original image to copy metadata (spacing, origin, direction)
            # Wait, the transforms might have changed spacing!
            # For validation and Week 5, we want the prediction to match the resampled (0.5mm isotropic) geometry.
            # MONAI's LoadImaged reads it and Spacingd resamples it.
            # We can use the meta_dict to get the affine, but SimpleITK is easier for writing NIfTI.
            
            # Since we evaluate on the preprocessed images, let's just write the mask directly
            # using the affine from MONAI, or simply sitk.GetImageFromArray.
            
            original_sitk = sitk.ReadImage(image_path)
            
            # Because our input might have been resampled by MONAI Spacingd,
            # we should restore the original geometry or save it as is.
            # For simplicity in this pipeline, we will assume test images are ALREADY preprocessed 
            # to 0.5mm isotropic using `run_preprocessing.py`, so their shape matches exactly.
            if output_mask.shape != sitk.GetArrayViewFromImage(original_sitk).shape:
                print(f"Warning: Mask shape {output_mask.shape} does not match original {sitk.GetArrayViewFromImage(original_sitk).shape}")
                
            out_img = sitk.GetImageFromArray(output_mask)
            out_img.CopyInformation(original_sitk)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sitk.WriteImage(out_img, output_path)
            print(f"Saved prediction to {output_path}")
            
            # Cleanup
            del outputs
            del inputs
            gc.collect()


def run_inference_batch(data_dir, output_dir, model_path, model_type="swin_unetr", use_amp=True):
    """
    Run inference on all test cases in data_dir/imagesTs.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model {model_type} onto {device}...")
    
    model = get_model(model_type, device)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded checkpoint from {model_path}")
    
    test_images_dir = os.path.join(data_dir, "imagesTs")
    if not os.path.exists(test_images_dir):
        raise FileNotFoundError(f"Test images directory not found: {test_images_dir}")
        
    image_files = sorted(glob.glob(os.path.join(test_images_dir, "*.nii.gz")))
    print(f"Found {len(image_files)} test images.")
    
    for img_path in image_files:
        basename = os.path.basename(img_path).replace("_0000.nii.gz", ".nii.gz")
        out_path = os.path.join(output_dir, basename)
        
        print(f"Processing {basename}...")
        predict_case(model, device, img_path, out_path, use_amp=use_amp)
        
    print("Inference completed.")
