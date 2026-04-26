# manifold_engine.py (Pipeline Phase 1 & 2 Execution)
import numpy as np
import os
import pickle
from tqdm import tqdm
from ModelBuilder import ModelBuilder
from PatchDatasetPreserve import PatchDatasetPreserve
import tensorflow as tf
from riemannian_utils import RiemannianMetrics

def extract_features_and_labels(model, generator, steps=None):
    """Helper to extract raw features (z) from the generator using the model."""
    print("[ManifoldEngine] Extracting features...")
    features_list = []
    labels_list = []
    
    if steps is None:
        steps = len(generator)
        
    for i in tqdm(range(steps)):
        batch_x, batch_y = generator[i]
        
        # If generator returns (img, tda), take img
        if isinstance(batch_x, list):
            batch_x = batch_x[0]
            
        # Get features (Penultimate Layer)
        # Assuming model outputs features directly or we use an intermediate model
        batch_feats = model.predict_on_batch(batch_x)
        
        features_list.append(batch_feats)
        labels_list.append(np.argmax(batch_y, axis=1)) # Convert one-hot to index
        
    return np.vstack(features_list), np.concatenate(labels_list)

#%%
def run_phase_1_pretraining(config, model, train_gen, fold=0, percentile=90):
    """
    Executes Phase 1: Feature Extraction & Manifold Construction.
    """
    print(f"\n{'='*40}\n[Phase 1] Pre-training & Manifold Construction (Fold {fold})\n{'='*40}")
    
    # 1. Load Pre-trained Backbone (ConvNeXtLarge)
    print("[Step 1] Loading Feature Extractor...")
    full_model = model
    
    # Create Feature Extractor (Cut off classification head)
    # Assuming 'global_average_pooling2d' or specific dense layer is the target
    feature_extractor = tf.keras.Model(
        inputs=full_model.input,
        outputs=full_model.layers[-2].output # Penultimate layer
    )
    print(f"  Feature Extractor Output Shape: {feature_extractor.output_shape}")
  
    # 3. Extract Features (Z)
    X_train_z, y_train_z = extract_features_and_labels(feature_extractor, train_gen)
    print(f"{X_train_z.shape=}, {y_train_z.shape=}")
    
    # 4. Construct Manifold (Fit Metrics)
    print("[Step 2] Constructing Riemannian Manifold...")
    riemann_tool = RiemannianMetrics(n_classes=config.MODEL.NUM_CLASSES, feature_dim=X_train_z.shape[1])
    riemann_tool.fit_manifold(X_train_z, y_train_z)
    
    # 5. Save Manifold Parameters
    save_path = os.path.join(config.SAVE_PATH, f"fold_{fold}_manifold.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(riemann_tool, f)
    print(f"  Manifold parameters saved to {save_path}")
    
    return riemann_tool, feature_extractor, X_train_z

def run_phase_2_validation(riemann_tool, feature_extractor, val_gen, config):
    """
    Executes Phase 2: Curvature Calculation & Threshold Calibration.
    """
    print(f"\n{'='*40}\n[Phase 2] Curvature Calculation & Calibration\n{'='*40}")
    
    # 1. Setup Val Generator
    preprocessing_fn = ModelBuilder(config.MODEL.NAMES[0], config).get_preprocessing_function()
    # val_gen = PatchDatasetPreserve(val_df, batch_size=32, preprocessing_function=preprocessing_fn, shuffle=False)
    
    # 2. Extract Val Features
    X_val, _ = extract_features_and_labels(feature_extractor, val_gen)
    print(f"{X_val.shape=}")
    
    # 3. Compute Curvature Energy
    print("[Step 3] Computing Curvature Energies...")
    energies = riemann_tool.compute_batch_curvature(X_val)
    print(f"  Energy Stats: Min={energies.min():.4f}, Max={energies.max():.4f}, Mean={energies.mean():.4f}")
    
    # 4. Calibrate Threshold
    # We hypothesize that informative patches are in the top percentile
    threshold = riemann_tool.calibrate_threshold(X_val, percentile=percentile) # Example: Median split
    
    return energies, threshold