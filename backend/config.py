"""
Global configuration for the Knee Twin backend.
"""

# Bump this manually whenever segmentation, postprocessing, or meshing logic changes meaningfully.
PIPELINE_VERSION = "v3.2"  # Bumped: MRI bone now uses native CartiMorph labels 1/3 (invalidates stale synthesis cache)
