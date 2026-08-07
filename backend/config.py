"""
Global configuration for the Knee Twin backend.
"""

# Bump this manually whenever segmentation, postprocessing, or meshing logic changes meaningfully.
PIPELINE_VERSION = "v3.2"  # Bumped: MRI bone now uses native CartiMorph labels 1/3 (invalidates stale synthesis cache)

# Version of the resection/implant logic (src/mesh/resection.py, implant.py).
# Tracked separately from PIPELINE_VERSION because these outputs are generated
# on demand by /api/resect rather than by the ingest pipeline, and change
# independently of segmentation or meshing.
#
# Bump whenever cut placement, sizing tables, or fit selection change: the
# resected meshes and implant .obj files on disk are otherwise
# indistinguishable between versions, which is exactly the silent-staleness
# problem PIPELINE_VERSION already exists to prevent for the ingest cache.
RESECTION_VERSION = "r1.0"
