import os
import sys
import pickle
import logging
from pathlib import Path
import numpy as np
import pyvista as pv
from pycpd import DeformableRegistration, RigidRegistration
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.spatial import cKDTree

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SCRATCH_DIR = Path('d:/knee surgery model/scratch')
MODEL_DIR = Path('d:/knee surgery model/models/ssm')
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CASES = ['DU02', 'DU03', 'DU06']
REF_CASE = 'DU02'

def load_bone_mesh(case_dir: Path, bone_type: str) -> pv.PolyData:
    """Find and load the validated bone mesh, normalizing left knees to right-knee space."""
    mesh_dir = case_dir / "meshes"
    summary_path = case_dir / "laterality_summary.json"
    
    if not mesh_dir.exists():
        raise FileNotFoundError(f"Mesh directory not found: {mesh_dir}")
        
    side = "right" # default
    if summary_path.exists():
        import json
        with open(summary_path) as f:
            summary = json.load(f)
            side = summary.get("laterality", "right")
            
    mesh_file = mesh_dir / f"{bone_type}_{side}.obj"
    if not mesh_file.exists():
        # Fallback to any matching bone file if exact side file isn't named that way
        candidates = [f for f in mesh_dir.glob(f"*{bone_type}*.obj") if "patella" not in f.name]
        if not candidates:
            raise FileNotFoundError(f"No {bone_type} mesh found in {mesh_dir}")
        mesh_file = max(candidates, key=lambda p: p.stat().st_size)
        
    logger.info(f"  Selected mesh for {case_dir.name} ({bone_type}): {mesh_file.name}")
    
    mesh = pv.read(str(mesh_file))
    
    # Normalize left knee to right knee coordinate space via X-reflection
    if "_left" in mesh_file.name:
        logger.info(f"  Reflecting left knee mesh ({mesh_file.name}) to right-knee coordinate space.")
        mesh.points[:, 0] = -mesh.points[:, 0]
        
    return mesh

def compute_msd(verts1: np.ndarray, verts2: np.ndarray) -> float:
    """Compute Mean Surface Distance (symmetric average nearest-neighbor distance)."""
    tree1 = cKDTree(verts1)
    tree2 = cKDTree(verts2)
    d1, _ = tree1.query(verts2)
    d2, _ = tree2.query(verts1)
    return float((np.mean(d1) + np.mean(d2)) / 2.0)

def register_and_align(template_mesh: pv.PolyData, target_mesh: pv.PolyData, max_pts: int = 2000) -> np.ndarray:
    """Fast CPD Rigid + Deformable registration returning canonical pose shape deformation."""
    Y_full = np.array(template_mesh.points, dtype=float)
    X_full = np.array(target_mesh.points, dtype=float)
    
    # Center both meshes to origin (0, 0, 0)
    Y_full = Y_full - np.mean(Y_full, axis=0)
    X_full = X_full - np.mean(X_full, axis=0)
    
    np.random.seed(42) # Reproducible subsampling
    
    if Y_full.shape[0] > max_pts:
        idx_y = np.random.choice(Y_full.shape[0], max_pts, replace=False)
        Y_sub = Y_full[idx_y]
    else:
        Y_sub = Y_full
        
    if X_full.shape[0] > max_pts:
        idx_x = np.random.choice(X_full.shape[0], max_pts, replace=False)
        X_sub = X_full[idx_x]
    else:
        X_sub = X_full
        
    # 1. Rigid alignment
    logger.info(f"  Running CPD Rigid Registration ({Y_sub.shape[0]} sub-points)...")
    reg_rigid = RigidRegistration(**{'X': X_sub, 'Y': Y_sub, 'max_iterations': 30})
    TY_rigid_sub, (s, R, t) = reg_rigid.register()
    
    # 2. Deformable alignment
    logger.info(f"  Running CPD Deformable Registration ({Y_sub.shape[0]} sub-points)...")
    reg_def = DeformableRegistration(**{'X': X_sub, 'Y': TY_rigid_sub, 'max_iterations': 30})
    TY_def_sub, (G, W) = reg_def.register()
    
    # 3. Compute non-rigid deformation field in canonical template space
    beta = reg_def.beta
    diff_full = Y_full[:, np.newaxis, :] - Y_sub[np.newaxis, :, :] # (N_full, N_sub, 3)
    dist_sq_full = np.sum(diff_full ** 2, axis=2)
    G_full = np.exp(-dist_sq_full / (2 * (beta ** 2)))
    
    # Apply non-rigid displacement to canonical template
    TY_canonical = Y_full + np.dot(G_full, W)
    return TY_canonical

def train_ssm_for_bone(bone_type: str):
    logger.info(f"\n========================================")
    logger.info(f"Training SSM for {bone_type.upper()}")
    logger.info(f"========================================")
    
    # 1. Load Reference Template & Center it
    ref_mesh = load_bone_mesh(SCRATCH_DIR / REF_CASE, bone_type)
    template_verts_raw = np.array(ref_mesh.points, dtype=float)
    template_verts = template_verts_raw - np.mean(template_verts_raw, axis=0) # Centroid centered
    ref_mesh.points = template_verts
    
    num_verts = template_verts.shape[0]
    logger.info(f"Reference case {REF_CASE} loaded ({num_verts} vertices, centered at origin)")
    
    # Save template mesh
    template_mesh_out = MODEL_DIR / f"template_{bone_type}.obj"
    ref_mesh.save(str(template_mesh_out))
    logger.info(f"Saved template mesh to {template_mesh_out}")
    
    # 2. Correspond all cases to template
    aligned_shapes = []
    features_list = []
    
    for case in CASES:
        case_dir = SCRATCH_DIR / case
        feat_path = case_dir / "features.npy"
        if not feat_path.exists():
            raise FileNotFoundError(f"Feature vector missing for {case} at {feat_path}")
        feat = np.load(str(feat_path))
        features_list.append(feat)
        
        if case == REF_CASE:
            aligned_shapes.append(template_verts.flatten())
        else:
            logger.info(f"Aligning {case} to {REF_CASE}...")
            tgt_mesh = load_bone_mesh(case_dir, bone_type)
            aligned_verts = register_and_align(ref_mesh, tgt_mesh)
            aligned_shapes.append(aligned_verts.flatten())
            
    shape_matrix = np.array(aligned_shapes) # (3, num_verts * 3)
    feature_matrix = np.array(features_list) # (3, 9)
    
    # 3. Fit PCA with hardcoded n_components=2 (N=3 data rank constraint)
    n_components = 2
    logger.info(f"Fitting PCA (hardcoded n_components={n_components} for N=3 samples)...")
    pca = PCA(n_components=n_components)
    pca_scores = pca.fit_transform(shape_matrix)
    logger.info(f"PCA explained variance ratio: {pca.explained_variance_ratio_} (Total: {np.sum(pca.explained_variance_ratio_):.4f})")
    
    # 4. Fit Scaler and Ridge Regressor
    scaler = StandardScaler()
    scaled_feats = scaler.fit_transform(feature_matrix)
    
    regressor = Ridge(alpha=1.0)
    regressor.fit(scaled_feats, pca_scores)
    logger.info(f"Fitted Ridge Regressor (R2 score on training: {regressor.score(scaled_feats, pca_scores):.4f})")
    
    # 5. LOO-CV Sanity Check
    logger.info("\n--- Running LOO-CV Sanity Check ---")
    logger.info("[NOTE] N=3 LOO-CV evaluates 1D interpolation behavior between training samples against a held-out case.")
    logger.info("       This serves as a sanity check for scaling/degradation, NOT a validated bound for out-of-distribution accuracy.")
    
    loo_msd_errors = []
    for i in range(len(CASES)):
        train_idx = [j for j in range(len(CASES)) if j != i]
        val_idx = i
        
        # Train fold PCA (1 component for 2 samples)
        fold_pca = PCA(n_components=1)
        fold_scores = fold_pca.fit_transform(shape_matrix[train_idx])
        
        fold_scaler = StandardScaler()
        fold_scaled_feats = fold_scaler.fit_transform(feature_matrix[train_idx])
        
        fold_reg = Ridge(alpha=1.0)
        fold_reg.fit(fold_scaled_feats, fold_scores)
        
        # Predict held-out
        val_scaled = fold_scaler.transform(feature_matrix[val_idx:val_idx+1])
        pred_score = fold_reg.predict(val_scaled)
        pred_flat = fold_pca.inverse_transform(pred_score.reshape(1, -1))
        pred_verts = pred_flat.reshape(-1, 3)
        
        gt_verts = shape_matrix[val_idx].reshape(-1, 3)
        msd = compute_msd(pred_verts, gt_verts)
        loo_msd_errors.append(msd)
        logger.info(f"  Fold {i+1} (Held out {CASES[val_idx]}): Mean Surface Distance = {msd:.3f} mm")
        
    mean_loo_msd = np.mean(loo_msd_errors)
    logger.info(f"Mean LOO-CV MSD across folds for {bone_type}: {mean_loo_msd:.3f} mm\n")
    
    # 6. Save Artifacts
    with open(MODEL_DIR / f"pca_model_{bone_type}.pkl", "wb") as f:
        pickle.dump(pca, f)
    with open(MODEL_DIR / f"shape_regressor_{bone_type}.pkl", "wb") as f:
        pickle.dump(regressor, f)
    with open(MODEL_DIR / "cart_feature_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
        
    logger.info(f"Successfully saved {bone_type} SSM models to {MODEL_DIR}")

def main():
    train_ssm_for_bone("femur")
    train_ssm_for_bone("tibia")
    logger.info("\nSSM Training complete for all bones!")

if __name__ == "__main__":
    main()
