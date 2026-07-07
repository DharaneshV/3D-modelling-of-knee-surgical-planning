"""
bone_segmentation.py

Week 1-2 combined: takes a raw downloaded CT volume, resamples it to a
standard isotropic spacing, then applies HU thresholding + morphological
cleanup + connected-component separation to isolate femur, tibia, patella.

Usage:
    python scripts/bone_segmentation.py --input data/ct_knee/temp_EAY131-5310722.nii.gz --output data/ct_knee/case01_bone_mask.nii.gz
"""

import argparse
import SimpleITK as sitk

DEFAULT_HU_THRESHOLD = 200
DEFAULT_SPACING = (1.0, 1.0, 1.0)  # mm, isotropic
LABELS = {"femur": 1, "tibia": 2, "patella": 3}


# ---------- Week 1: preprocessing ----------

def resample_to_isotropic(image, target_spacing=DEFAULT_SPACING, is_label=False):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [
        int(round(osz * ospc / tspc))
        for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(image.GetPixelIDValue())

    # Nearest-neighbor for label maps, linear for continuous CT intensities
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)

    return resampler.Execute(image)


# ---------- Week 2: HU thresholding + connected components ----------

def threshold_bone(ct_image, hu_threshold=DEFAULT_HU_THRESHOLD):
    return sitk.BinaryThreshold(
        ct_image,
        lowerThreshold=hu_threshold,
        upperThreshold=4000,
        insideValue=1,
        outsideValue=0,
    )


def clean_mask(binary_mask):
    closed = sitk.BinaryMorphologicalClosing(binary_mask, [2, 2, 2])
    opened = sitk.BinaryMorphologicalOpening(closed, [1, 1, 1])
    return opened


def separate_bones(binary_mask, min_component_voxels=500):
    components = sitk.ConnectedComponent(binary_mask)
    relabeled = sitk.RelabelComponent(components, minimumObjectSize=min_component_voxels)

    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(relabeled)
    present_labels = stats.GetLabels()

    if len(present_labels) < 3:
        print(f"Warning: only found {len(present_labels)} components above "
              f"min_component_voxels={min_component_voxels}. "
              "Lower this value or check the threshold.")

    top_labels = sorted(present_labels)[:3]
    kept = sitk.Mask(relabeled, sitk.Cast(
        sum(sitk.Equal(relabeled, l) for l in top_labels), sitk.sitkUInt8
    ))
    return kept, stats, top_labels


def assign_anatomical_labels(component_mask, stats, top_labels):
    """
    NOTE: assumes standard orientation (superior = higher z). Verify on
    your first case in a viewer before trusting this across the batch.
    """
    centroids = {l: stats.GetCentroid(l) for l in top_labels}
    sizes = {l: stats.GetNumberOfPixels(l) for l in top_labels}

    patella_label = min(sizes, key=sizes.get)
    remaining = [l for l in top_labels if l != patella_label]
    remaining.sort(key=lambda l: centroids[l][2], reverse=True)
    femur_label, tibia_label = remaining[0], remaining[1]

    label_map = {
        femur_label: LABELS["femur"],
        tibia_label: LABELS["tibia"],
        patella_label: LABELS["patella"],
    }

    change_filter = sitk.ChangeLabelImageFilter()
    change_filter.SetChangeMap({int(k): int(v) for k, v in label_map.items()})
    return change_filter.Execute(sitk.Cast(component_mask, sitk.sitkUInt16))


# ---------- Entry point ----------

def run(input_path, output_path, hu_threshold=DEFAULT_HU_THRESHOLD, spacing=DEFAULT_SPACING):
    raw_ct = sitk.ReadImage(input_path, sitk.sitkFloat32)

    print(f"Original spacing: {raw_ct.GetSpacing()}, size: {raw_ct.GetSize()}")
    ct = resample_to_isotropic(raw_ct, spacing, is_label=False)
    print(f"Resampled to spacing: {ct.GetSpacing()}, size: {ct.GetSize()}")

    binary = threshold_bone(ct, hu_threshold)
    binary = clean_mask(binary)
    components, stats, top_labels = separate_bones(binary)
    final_mask = assign_anatomical_labels(components, stats, top_labels)

    final_mask.CopyInformation(ct)
    sitk.WriteImage(final_mask, output_path)
    print(f"Saved multi-label bone mask to {output_path} "
          f"(labels: femur=1, tibia=2, patella=3)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resample + HU threshold + connected components bone segmentation")
    parser.add_argument("--input", required=True, help="Path to raw downloaded CT volume (NIfTI/DICOM)")
    parser.add_argument("--output", required=True, help="Path to write multi-label bone mask")
    parser.add_argument("--hu-threshold", type=int, default=DEFAULT_HU_THRESHOLD,
                         help=f"HU cutoff for bone (default: {DEFAULT_HU_THRESHOLD})")
    parser.add_argument("--spacing", type=float, nargs=3, default=DEFAULT_SPACING,
                         help="Target isotropic spacing in mm, e.g. 1.0 1.0 1.0")
    args = parser.parse_args()

    run(args.input, args.output, args.hu_threshold, tuple(args.spacing))
