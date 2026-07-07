import os
import time
import torch
from monai.networks.nets import SwinUNETR, SegResNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.transforms import AsDiscrete
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from src.segmentation.dataset import get_dataloaders

# Labels: Background=0, Femur=1, Femoral Cartilage=2, Tibia=3, Medial Tibial Cart=4, Lateral Tibial Cart=5
NUM_CLASSES = 6

def get_model(model_type="swin_unetr", device=None):
    """
    Instantiates the model architecture.
    SwinUNETR provides better accuracy for cartilage by capturing long-range context.
    SegResNet is a lighter, faster alternative if VRAM is an issue.
    """
    if model_type == "swin_unetr":
        model = SwinUNETR(
            spatial_dims=3,
            in_channels=1,
            out_channels=NUM_CLASSES,
            feature_size=48,
            use_checkpoint=True,  # Saves VRAM at the cost of slight compute overhead
        )
    elif model_type == "segresnet":
        model = SegResNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=NUM_CLASSES,
            init_filters=16,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    if device:
        model = model.to(device)
    return model


def train(
    data_dir,
    output_dir,
    model_type="swin_unetr",
    epochs=100,
    batch_size=1,
    learning_rate=1e-4,
    patience=15,
    max_cases=None,
    use_amp=True,
    resume_checkpoint=None
):
    """
    Main training loop for MRI soft tissue segmentation.
    Includes AMP (mixed precision) for saving VRAM and TensorBoard logging.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=log_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Dataloaders
    train_loader, val_loader = get_dataloaders(
        data_dir, 
        batch_size=batch_size, 
        max_cases=max_cases,
        cache=False # To save system RAM
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # 2. Model, Loss, Optimizer
    try:
        model = get_model(model_type, device)
    except RuntimeError as e:
        if "Out of memory" in str(e) and model_type == "swin_unetr":
            print("\n[Warning] SwinUNETR OOM'd during initialization. Falling back to SegResNet...")
            model = get_model("segresnet", device)
        else:
            raise e

    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Resuming training from checkpoint: {resume_checkpoint}")
        model.load_state_dict(torch.load(resume_checkpoint, map_location=device))

    loss_function = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # AMP scaler for mixed precision
    scaler = torch.amp.GradScaler(enabled=use_amp)
    
    # Metrics
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred = AsDiscrete(argmax=True, to_onehot=NUM_CLASSES)
    post_label = AsDiscrete(to_onehot=NUM_CLASSES)

    best_metric = -1
    best_metric_epoch = -1
    epochs_no_improve = 0

    print("Starting training...")
    for epoch in range(epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        
        start_time = time.time()
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            
            # Forward pass with AMP
            with torch.amp.autocast(device_type="cuda" if "cuda" in str(device) else "cpu", enabled=use_amp):
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
            
            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            
        epoch_loss /= step
        writer.add_scalar("train_loss", epoch_loss, epoch + 1)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f} (Time: {time.time() - start_time:.1f}s)")

        # Validation every 2 epochs
        if (epoch + 1) % 2 == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                    # Use sliding window inference since full volumes might OOM
                    roi_size = (96, 96, 96)
                    sw_batch_size = 4
                    with torch.amp.autocast(device_type="cuda" if "cuda" in str(device) else "cpu", enabled=use_amp):
                        val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                    
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    dice_metric(y_pred=val_outputs, y=val_labels)

                metric = dice_metric.aggregate().item()
                dice_metric.reset()
                
                writer.add_scalar("val_mean_dice", metric, epoch + 1)
                print(f"current epoch: {epoch + 1} current mean dice: {metric:.4f}")

                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    epochs_no_improve = 0
                    
                    torch.save(
                        model.state_dict(),
                        os.path.join(output_dir, "best_model.pth"),
                    )
                    print("saved new best metric model")
                else:
                    epochs_no_improve += 2
                    
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered after {epochs_no_improve} epochs without improvement.")
                    break

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()
    return best_metric
