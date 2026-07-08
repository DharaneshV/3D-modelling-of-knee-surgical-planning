import os
import sys
from totalsegmentator.python_api import totalsegmentator

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

try:
    totalsegmentator("data/ct_knee/case01_STS_006_cropped.nii.gz", "temp_out", fast=True, ml=True)
    print("Success")
except Exception as e:
    print(f"Error: {e}")
