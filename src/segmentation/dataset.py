import os
import glob
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRangePercentilesd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandGaussianNoised,
    ToTensord,
)
from monai.data import Dataset, CacheDataset, DataLoader
from sklearn.model_selection import train_test_split


def get_train_val_transforms():
    """
    Returns the training and validation transforms.
    The training pipeline includes spatial augmentations and intensity normalization.
    """
    # 96x96x96 patch size fits nicely in a 6GB VRAM GPU with batch_size=1 and FP16
    patch_size = (96, 96, 96)
    
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        # Resample to 0.5mm isotropic spacing (standard for this dataset)
        Spacingd(
            keys=["image", "label"],
            pixdim=(0.5, 0.5, 0.5),
            mode=("bilinear", "nearest")
        ),
        # Normalize MRI intensities
        ScaleIntensityRangePercentilesd(
            keys="image",
            lower=0.5,
            upper=99.5,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        # Crop patches balancing foreground (labels > 0) and background
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            pos=1,
            neg=1,
            num_samples=2, # Extracts 2 patches per volume
            image_key="image",
            image_threshold=0,
        ),
        # Spatial Augmentations
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.1),
        ToTensord(keys=["image", "label"]),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(0.5, 0.5, 0.5),
            mode=("bilinear", "nearest")
        ),
        ScaleIntensityRangePercentilesd(
            keys="image",
            lower=0.5,
            upper=99.5,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        ToTensord(keys=["image", "label"]),
    ])

    return train_transforms, val_transforms


def get_inference_transforms():
    """Transforms for inference on new cases (no labels required)."""
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(0.5, 0.5, 0.5), mode=("bilinear")),
        ScaleIntensityRangePercentilesd(
            keys="image",
            lower=0.5,
            upper=99.5,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        ToTensord(keys=["image"]),
    ])


def get_dataloaders(data_dir, batch_size=1, max_cases=None, cache=False):
    """
    Creates train and validation DataLoaders.
    
    Args:
        data_dir: Path to the root directory containing imagesTr and labelsTr
        batch_size: Batch size for training
        max_cases: Limit number of cases for quick sanity checks
        cache: If True, uses MONAI CacheDataset (requires a lot of RAM)
    """
    images_dir = os.path.join(data_dir, "imagesTr")
    labels_dir = os.path.join(data_dir, "labelsTr")

    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        raise FileNotFoundError(f"Missing imagesTr or labelsTr in {data_dir}. Ensure OAI-ZIB is unzipped.")

    # Match image and label files (assuming they have the same base name, e.g. oaizib_XXX_0000.nii.gz and oaizib_XXX.nii.gz)
    image_files = sorted(glob.glob(os.path.join(images_dir, "*.nii.gz")))
    
    data_dicts = []
    for img_path in image_files:
        basename = os.path.basename(img_path).replace("_0000.nii.gz", ".nii.gz")
        lbl_path = os.path.join(labels_dir, basename)
        if os.path.exists(lbl_path):
            data_dicts.append({"image": img_path, "label": lbl_path})
        else:
            print(f"Warning: No matching label found for {img_path}")

    if not data_dicts:
        raise ValueError("No matching image/label pairs found!")

    if max_cases is not None:
        data_dicts = data_dicts[:max_cases]

    # 80/20 train/val split
    train_files, val_files = train_test_split(data_dicts, test_size=0.2, random_state=42)

    train_transforms, val_transforms = get_train_val_transforms()

    dataset_cls = CacheDataset if cache else Dataset
    
    train_ds = dataset_cls(data=train_files, transform=train_transforms)
    val_ds = dataset_cls(data=val_files, transform=val_transforms)

    # Note: RandCropByPosNegLabeld returns `num_samples` patches per volume, so the effective batch size is batch_size * num_samples
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)

    return train_loader, val_loader
