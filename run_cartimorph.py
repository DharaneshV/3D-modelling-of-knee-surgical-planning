import os
import torch
original_load = torch.load
def new_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = new_load

from CartiMorph_nnUNet.inference.predict import predict_from_folder

model_dir = "temp_cartimorph/Models/segModel/segModel-OAIZIB-19Mar2024"
input_dir = "temp_cartimorph_io/input"
output_dir = "temp_cartimorph_io/output"

predict_from_folder(
    model=model_dir,
    input_folder=input_dir,
    output_folder=output_dir,
    folds=(0,),
    save_npz=False,
    num_threads_preprocessing=2,
    num_threads_nifti_save=2,
    lowres_segmentations=None,
    part_id=0,
    num_parts=1,
    tta=False, 
    overwrite_existing=True,
    mode='normal',
    overwrite_all_in_gpu=None,
    mixed_precision=True,
    step_size=0.5,
    checkpoint_name="segModel" 
)
print("Inference completed.")
