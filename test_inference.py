import os
import torch
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from monai.networks.nets import SwinUNETR
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete, Compose, LoadImaged, EnsureChannelFirstd, 
    Orientationd, Spacingd, ScaleIntensityRangePercentilesd, ToTensord
)
from monai.data import Dataset

print("1. Setup Model", flush=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SwinUNETR(
    spatial_dims=3,
    in_channels=1,
    out_channels=6,
    feature_size=48,
    use_checkpoint=True,
).to(device)

model_path = "results/models/best_model.pth"
if os.path.exists(model_path):
    print(f"Loading weights from {model_path}", flush=True)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
else:
    print(f"ERROR: No model found at {model_path}", flush=True)
    exit(1)

model.eval()

eval_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(0.5, 0.5, 0.5), mode=("bilinear", "nearest")),
    ScaleIntensityRangePercentilesd(keys="image", lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True),
    ToTensord(keys=["image", "label"]),
])

def predict_and_evaluate(case_id):
    print(f"\n--- Processing {case_id} ---", flush=True)
    img_path = f"data/preprocessed/{case_id}_0000_preprocessed.nii.gz"
    lbl_path = f"data/preprocessed/{case_id}_0000_label.nii.gz"
    
    ds = Dataset(data=[{"image": img_path, "label": lbl_path}], transform=eval_transforms)
    
    print("Running sliding window inference...", flush=True)
    with torch.no_grad():
        inputs = ds[0]["image"].unsqueeze(0).to(device)
        with torch.amp.autocast(device_type="cuda" if "cuda" in str(device) else "cpu", enabled=True):
            outputs = sliding_window_inference(inputs, (96, 96, 96), 4, model)
        post_pred = AsDiscrete(argmax=True)
        pred_arr = post_pred(outputs[0]).cpu().numpy().squeeze(0).astype(np.uint16)
        
    print(f"Prediction shape: {pred_arr.shape}", flush=True)
    
    lbl_tensor = ds[0]["label"].numpy().squeeze(0)
    print(f"Label tensor shape: {lbl_tensor.shape}", flush=True)
    
    labels = {
        1: "Femur",
        2: "Femoral Cartilage",
        3: "Tibia",
        4: "Medial Tibial Cartilage",
        5: "Lateral Tibial Cartilage"
    }
    
    for i, name in labels.items():
        pred_count = np.sum(pred_arr == i)
        gt_count = np.sum(lbl_tensor == i)
        
        intersection = np.sum((pred_arr == i) & (lbl_tensor == i))
        dice = (2.0 * intersection) / (pred_count + gt_count) if (pred_count + gt_count) > 0 else 0.0
        
        print(f"{name:25s} | GT: {gt_count:7d} | Pred: {pred_count:7d} | Dice: {dice:.4f}", flush=True)
        
    cart_mask = (lbl_tensor == 2) | (lbl_tensor == 4) | (lbl_tensor == 5)
    if np.sum(cart_mask) > 0:
        z_slice = np.argmax(np.sum(cart_mask, axis=(1,2)))
    else:
        z_slice = pred_arr.shape[0] // 2
        
    img_tensor = ds[0]["image"].numpy().squeeze(0)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img_tensor[z_slice], cmap='gray')
    axes[0].contour(lbl_tensor[z_slice] > 0, colors='lime', linewidths=1)
    axes[0].set_title(f"Ground Truth (Z={z_slice})")
    
    axes[1].imshow(img_tensor[z_slice], cmap='gray')
    axes[1].contour(pred_arr[z_slice] > 0, colors='red', linewidths=1)
    axes[1].set_title(f"Prediction (Z={z_slice})")
    
    plt.tight_layout()
    plt.savefig(f"{case_id}_overlay.png")
    print(f"Saved overlay to {case_id}_overlay.png", flush=True)

predict_and_evaluate("oaizib_001")
