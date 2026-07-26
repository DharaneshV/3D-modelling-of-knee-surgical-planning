import os
import sys
import pickle
import logging
import argparse
from pathlib import Path
import numpy as np
import pyvista as pv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SCRATCH_DIR = Path('d:/knee surgery model/scratch')
MODEL_DIR = Path('d:/knee surgery model/models/ssm')

# QA Gate Threshold
# WARNING: This checks for statistical outliers in the N=3 feature space. 
# It catches egregious inputs only and DOES NOT guarantee geometric accuracy (<2mm).
# With N=3, the true held-out error is ~7mm, meaning the fallback to CT is the DEFAULT 
# clinical pathway for tight-tolerance sizing until more paired data is collected.
MAHALANOBIS_THRESHOLD = 3.0

class MahalanobisGateFailure(Exception):
    """Exception raised when MRI features are out of distribution and CT fallback is required."""
    pass

def load_models(bone_type: str):
    """Load the trained SSM models and template mesh."""
    logger.info(f"Loading SSM models for {bone_type}...")
    with open(MODEL_DIR / f"pca_model_{bone_type}.pkl", "rb") as f:
        pca = pickle.load(f)
    with open(MODEL_DIR / f"shape_regressor_{bone_type}.pkl", "rb") as f:
        regressor = pickle.load(f)
    with open(MODEL_DIR / "cart_feature_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
        
    template_mesh = pv.read(str(MODEL_DIR / f"template_{bone_type}.obj"))
    
    return pca, regressor, scaler, template_mesh

def synthesize_bone(case_dir: Path, bone_type: str, custom_features: np.ndarray = None) -> Path:
    """Synthesize a bone mesh from MRI features. Returns the path to the synthesized mesh."""
    logger.info(f"\n--- Synthesizing {bone_type} for {case_dir.name} ---")
    
    if custom_features is not None:
        features = custom_features
        logger.info("Using provided custom feature vector for testing.")
    else:
        feat_path = case_dir / "features.npy"
        if not feat_path.exists():
            raise FileNotFoundError(f"Features file not found: {feat_path}")
        # Load features
        features = np.load(str(feat_path))
        
    if features.ndim == 1:
        features = features.reshape(1, -1)
        
    # Load models
    pca, regressor, scaler, template_mesh = load_models(bone_type)
    
    # 1. Scale features
    scaled_feats = scaler.transform(features)
    
    # 2. Predict PCA scores
    pred_scores = regressor.predict(scaled_feats)
    
    # 3. QA Gate: Mahalanobis Distance
    # Calculate distance in the uncorrelated PCA space
    std_devs = np.sqrt(pca.explained_variance_)
    std_devs = np.maximum(std_devs, 1e-6) # avoid div by zero
    
    m_dist = np.sqrt(np.sum((pred_scores[0] / std_devs) ** 2))
    logger.info(f"Predicted PCA scores: {pred_scores[0]}")
    logger.info(f"Mahalanobis Distance: {m_dist:.2f} sigma")
    
    if m_dist > MAHALANOBIS_THRESHOLD:
        logger.error(f"QA GATE FAILED: Mahalanobis distance ({m_dist:.2f}) exceeds threshold ({MAHALANOBIS_THRESHOLD} sigma).")
        logger.error("The input MRI features are statistically out-of-distribution.")
        logger.error("ACTION REQUIRED: Aborting MRI-only synthesis. Fallback to patient CT scan required.")
        import json
        sigma_path = case_dir / f"{bone_type}_sigma.json"
        with open(sigma_path, "w") as f:
            json.dump({"sigma": float(m_dist)}, f)
        raise MahalanobisGateFailure(f"Mahalanobis distance ({m_dist:.2f}) exceeds threshold ({MAHALANOBIS_THRESHOLD} sigma).")
        
    logger.info("QA Gate Passed (Note: catches gross outliers only; not a substitute for the <2mm clinical accuracy bound).")
    
    import json
    sigma_path = case_dir / f"{bone_type}_sigma.json"
    with open(sigma_path, "w") as f:
        json.dump({"sigma": float(m_dist)}, f)
    
    # 4. Reconstruct mesh
    pred_flat = pca.inverse_transform(pred_scores)
    pred_verts = pred_flat.reshape(-1, 3)
    
    # Update template mesh
    synth_mesh = template_mesh.copy()
    synth_mesh.points = pred_verts
    
    # Save output
    out_path = case_dir / "meshes" / f"synthetic_{bone_type}.obj"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    synth_mesh.save(str(out_path))
    
    logger.info(f"Successfully saved synthetic mesh to {out_path.name}")
    
    # Explicit clinical warning
    logger.warning(f"CLINICAL USAGE CONSTRAINT: Outputs from this module must not be used as the sole basis for implant dimension decisions.")
    
    return out_path

def main():
    parser = argparse.ArgumentParser(description="Synthesize bone meshes from MRI features.")
    parser.add_argument("case", help="Case ID (e.g., DU03)")
    parser.add_argument("--test-ood", action="store_true", help="Test QA gate with an out-of-distribution vector")
    args = parser.parse_args()
    
    case_dir = SCRATCH_DIR / args.case
    if not case_dir.exists():
        logger.error(f"Case directory not found: {case_dir}")
        sys.exit(1)
        
    custom_features = None
    if args.test_ood:
        # Create a dummy out-of-distribution feature vector (e.g., extreme sizes)
        # Using [10000.0, ...] to ensure it trips the gate
        custom_features = np.array([[10000.0] * 9])
        logger.info("=== RUNNING IN OOD TEST MODE ===")
        
    try:
        synthesize_bone(case_dir, "femur", custom_features)
        synthesize_bone(case_dir, "tibia", custom_features)
        logger.info("\nSynthesis complete for both bones.")
    except MahalanobisGateFailure:
        logger.error("\nSynthesis aborted due to QA gate failure.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nSynthesis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

