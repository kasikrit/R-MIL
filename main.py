import os
import logging
import gc
import json
import platform
import pickle
import csv
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import tensorflow_addons as tfa
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.manifold import Isomap
from scipy.stats import multivariate_normal

import mygears
import configsCell_mil_stage2
from PatchDatasetPreserve import PatchDatasetPreserve
from ModelBuilder import ModelBuilder
from FullBagDataset import FullBagDataset
from riemannian_utils import CurvatureAttention, RiemannianMetrics, PoincareMath, HyperbolicDense, FrechetMean


# ==============================================================================
# Pipeline Definitions & Helper Functions
# ==============================================================================

def p1_step1_extract_features(model_name, model, data_gen, layer_index=-2, verbose=True):
    feature_extractor = tf.keras.Model(
        inputs=model.input,
        outputs=model.layers[layer_index].output
    ) 
    feature_extractor.trainable = False
    
    features = feature_extractor.predict(
        data_gen, 
        verbose=1,
        workers=4,
        use_multiprocessing=True
    )
    
    if hasattr(data_gen, 'classes'):
        labels = data_gen.classes
    else:
        labels = np.concatenate([y for x, y in data_gen], axis=0)
        if len(labels.shape) > 1: labels = np.argmax(labels, axis=1)

    fe_path = os.path.join(config['MODEL_PATH'], f"fold_{fold}_{model_name}_feature_extractor.h5")
    feature_extractor.save(fe_path)
    
    features_save_path = os.path.join(config['MODEL_PATH'], f"fold_{fold}_{model_name}_features.npy")
    np.save(features_save_path, features)
    
    return feature_extractor, features, labels

def p1_step2_fit_manifold(features, labels, n_classes, feature_dim, verbose=True):
    riemann_tool = RiemannianMetrics(n_classes=n_classes, feature_dim=feature_dim)
    riemann_tool.fit_manifold(features, labels)
    return riemann_tool

def p1_step3_compute_train_energy(riemann_tool, features, save_path=None, verbose=True):
    E_train = riemann_tool.compute_batch_curvature(features)
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(riemann_tool, f)
    return E_train

def p2_step1_extract_val_features(model_name, feature_extractor, val_gen, verbose=True):
    X_val_z = feature_extractor.predict(val_gen, verbose=1)
    if hasattr(val_gen, 'classes'):
        y_val = val_gen.classes
    else:
        y_val = np.concatenate([y for x, y in val_gen], axis=0)
        if len(y_val.shape) > 1: y_val = np.argmax(y_val, axis=1)
    return X_val_z, y_val

def p2_step2_compute_val_energy(riemann_tool, X_val_z, save_path=None, verbose=True):
    E_val = riemann_tool.compute_batch_curvature(X_val_z)
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(E_val, f)
    return E_val

def p2_step3_calibrate_threshold(E_val, y_val, config, verbose=True):
    threshold = np.percentile(E_val, 95) 
    return E_val, threshold

def create_patient_bags(df, features, energies):
    unique_patients = df['patient_id'].unique()
    bags_z, bags_e, bag_labels = [], [], []
    
    for pid in tqdm(unique_patients):
        indices = df.index[df['patient_id'] == pid].tolist()
        if len(indices) == 0: continue
        
        p_feats = features[indices] 
        p_energies = energies[indices] 
        p_label = df.loc[indices[0], 'label']
        
        bags_z.append(p_feats)
        bags_e.append(p_energies)
        bag_labels.append(p_label)
        
    return bags_z, bags_e, np.array(bag_labels)

def pad_bags(bags, max_len=None):
    if max_len is None:
        max_len = max([b.shape[0] for b in bags])
        
    feature_dim = bags[0].shape[1]
    n_samples = len(bags)
    padded_data = np.zeros((n_samples, max_len, feature_dim), dtype=np.float32)
    
    for i, bag in enumerate(bags):
        length = min(len(bag), max_len)
        padded_data[i, :length, :] = bag[:length, :]
        
    return padded_data

def build_deep_riemannian_mil_model_3(input_dim, curvature_layer):
    c_val = 0.01 

    input_features = tf.keras.Input(shape=(None, input_dim), name='bag_features')
    input_energy = tf.keras.Input(shape=(None, 1), name='bag_energy') 
    
    energy_norm = layers.BatchNormalization(name="energy_auto_scale")(input_energy)
    x_norm = layers.LayerNormalization(epsilon=1e-6, name="pre_manifold_norm")(input_features)
    x_norm = layers.Lambda(lambda x: x * 0.1, name="rescale_to_ball_radius")(x_norm)

    x_hyp = layers.Lambda(lambda v: PoincareMath.exp_map(v, c=c_val), name="euclidean_to_hyperbolic")(x_norm)
    x_hyp = HyperbolicDense(512, c=c_val, activation='relu', name="hyp_dense_1")(x_hyp)
    x_hyp = HyperbolicDense(256, c=c_val, activation='relu', name="hyp_dense_2")(x_hyp)

    _, att_weights = curvature_layer([x_norm, energy_norm])
    bag_representation_hyp = FrechetMean(c=c_val, name="frechet_mean")([x_hyp, att_weights])
    bag_representation_tan = layers.Lambda(lambda v: PoincareMath.log_map(v, c=c_val), name="hyperbolic_to_euclidean")(bag_representation_hyp)

    energy_global = layers.GlobalAveragePooling1D(name="energy_pooling")(energy_norm) 
    merged = layers.Concatenate(name="polarity_anchor")([bag_representation_tan, energy_global])
    
    x = layers.Dense(128, activation='relu', name="clf_dense_1")(merged)
    x = layers.Dropout(config.TRAIN.DROPOUT)(x) 
    x = layers.Dense(64, activation='relu', name="clf_dense_2")(x)
    output = layers.Dense(2, activation='softmax', name="output")(x)
    
    model = tf.keras.Model(inputs=[input_features, input_energy], outputs=output, name="Deep_Riemannian_MIL")
    return model

def train_robust_mil_ensemble(model_name, fold, train_df, features, riemann_tool, config, fold_threshold, n_estimators=5, max_bag_len=32):
    E_raw = riemann_tool.compute_batch_curvature(features).reshape(-1, 1)

    if fold_threshold == 0: fold_threshold = 1.0
    E_scaled = E_raw / fold_threshold
    E_scaled = np.clip(E_scaled, 0, 5.0)

    bags_z, bags_e, y_bag_labels = create_patient_bags(df=train_df, features=features, energies=E_scaled)

    X_bag_z = pad_bags(bags_z, max_len=max_bag_len)
    X_bag_e = pad_bags(bags_e, max_len=max_bag_len)
    y_bag_oh = tf.keras.utils.to_categorical(y_bag_labels, 2)

    ensemble_models = []
    
    for i in range(n_estimators):
        curvature_layer = CurvatureAttention() 
        model = build_deep_riemannian_mil_model_3(input_dim=features.shape[1], curvature_layer=curvature_layer)
        
        optimizer = tfa.optimizers.AdamW(
            learning_rate=config['TRAIN']['LR'],
            weight_decay=config['TRAIN']['WEIGHT_DECAY'],
            global_clipnorm=config['TRAIN'].get('CLIP_NORM', 1.0)
        )

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        callbacks_list = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=config.TRAIN.stop_patience, restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=config.TRAIN.reduceLR_patience, min_lr=1e-5, verbose=1)
        ]

        history = model.fit(
            x=[X_bag_z, X_bag_e],
            y=y_bag_oh,
            batch_size=config.TRAIN.BATCH_SIZE, 
            epochs=config.TRAIN.EPOCHS, 
            validation_split=config.DATA.val_ratio,
            callbacks=callbacks_list,
            verbose=2
        )  
        os.makedirs(config.MODEL_PATH, exist_ok=True)
        mygears.plot_training_metrics_without_lr(history=history, prefix=f"Fold_{fold}_Est_{i}", save_dir=config.MODEL_PATH)
       
        ensemble_models.append(model)
        save_path = os.path.join(config.MODEL_PATH, f"fold_{fold}_{model_name}_mil_estimator_{i}.h5")
        model.save(save_path)

    return ensemble_models

def split_ood_data(features, energies, limit):
    e_flat = energies.flatten()
    
    valid_mask = e_flat < limit
    ood_mask = ~valid_mask
    
    X_valid, E_valid = features[valid_mask], energies[valid_mask]
    X_ood, E_ood = features[ood_mask], energies[ood_mask]
    
    n_total, n_valid, n_ood = len(features), len(X_valid), len(X_ood)
    reject_rate = (n_ood / n_total * 100) if n_total > 0 else 0
    
    return (
        {'X': X_valid, 'E': E_valid}, 
        {'X': X_ood, 'E': E_ood}, 
        {'n_valid': n_valid, 'n_ood': n_ood, 'rate': reject_rate}
    )

def plot_curvature_manifold(X_train, y_train, E_train, X_ood_full, E_ood_full, ood_limit, save_path="spacetime_curvature_manifold.html"):
    mask_valid = E_ood_full.flatten() < ood_limit
    X_ood_valid = X_ood_full[mask_valid]
    X_ood_reject = X_ood_full[~mask_valid]

    reducer = Isomap(n_neighbors=30, n_components=2, n_jobs=-1)
    X_train_emb = reducer.fit_transform(X_train)
    X_ood_valid_emb = reducer.transform(X_ood_valid) if len(X_ood_valid) > 0 else np.empty((0, 2))
    X_ood_reject_emb = reducer.transform(X_ood_reject) if len(X_ood_reject) > 0 else np.empty((0, 2))

    x_min, x_max = X_train_emb[:, 0].min() - 2, X_train_emb[:, 0].max() + 2
    y_min, y_max = X_train_emb[:, 1].min() - 2, X_train_emb[:, 1].max() + 2
    x_grid = np.linspace(x_min, x_max, 100) 
    y_grid = np.linspace(y_min, y_max, 100)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    reg = 1e-6 * np.eye(2)
    mean_ida, cov_ida = X_train_emb[y_train == 0].mean(axis=0), np.cov(X_train_emb[y_train == 0].T) + reg
    mean_thl, cov_thl = X_train_emb[y_train == 1].mean(axis=0), np.cov(X_train_emb[y_train == 1].T) + reg
    
    pdf_mix = np.maximum(0.5 * multivariate_normal.pdf(grid_points, mean=mean_ida, cov=cov_ida) + 0.5 * multivariate_normal.pdf(grid_points, mean=mean_thl, cov=cov_thl), 1e-12)
    z_grid = -np.log(pdf_mix).reshape(xx.shape)
    
    def get_z_height(points):
        if len(points) == 0: return np.array([])
        pmix = np.maximum(0.5 * multivariate_normal.pdf(points, mean_ida, cov_ida) + 0.5 * multivariate_normal.pdf(points, mean_thl, cov_thl), 1e-12)
        return -np.log(pmix)

    z_ida, z_thl = get_z_height(X_train_emb[y_train==0]), get_z_height(X_train_emb[y_train==1])
    z_ood_valid, z_ood_reject = get_z_height(X_ood_valid_emb), get_z_height(X_ood_reject_emb)

    fig = go.Figure()
    fig.add_trace(go.Surface(x=x_grid, y=y_grid, z=z_grid, colorscale='Greys', opacity=0.8, showscale=False, name='Probability Landscape'))
    
    subset_mask = (y_train == 0)
    fig.add_trace(go.Scatter3d(x=X_train_emb[subset_mask, 0], y=X_train_emb[subset_mask, 1], z=z_ida, mode='markers',
                               marker=dict(size=4, color=E_train[subset_mask].flatten(), colorscale='Viridis', showscale=True, symbol='circle'), name='IDA (Train)'))
    
    subset_mask = (y_train == 1)
    fig.add_trace(go.Scatter3d(x=X_train_emb[subset_mask, 0], y=X_train_emb[subset_mask, 1], z=z_thl, mode='markers',
                               marker=dict(size=4, color=E_train[subset_mask].flatten(), colorscale='Viridis', showscale=False, symbol='diamond'), name='THL (Train)'))
    
    if len(X_ood_valid_emb) > 0:
        fig.add_trace(go.Scatter3d(x=X_ood_valid_emb[:, 0], y=X_ood_valid_emb[:, 1], z=z_ood_valid, mode='markers', marker=dict(size=5, color='blue', symbol='square', opacity=0.9), name=f'OOD Accepted'))
    
    if len(X_ood_reject_emb) > 0:
        fig.add_trace(go.Scatter3d(x=X_ood_reject_emb[:, 0], y=X_ood_reject_emb[:, 1], z=z_ood_reject, mode='markers', marker=dict(size=5, color='red', symbol='x', opacity=1.0), name=f'OOD Rejected'))
    
    fig.update_layout(title=f"Manifold Surface & OOD Filter (Limit: {ood_limit:.2f})", scene=dict(xaxis_title="Isomap Dim 1", yaxis_title="Isomap Dim 2", zaxis_title="Free Energy (-log P)", camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))), width=1200, height=900)
    fig.write_html(save_path)

def calculate_optimal_thresholds(y_true, y_prob, verbose=True):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    t_upper = thresholds[np.argmax(tpr + (1 - fpr) - 1)]
    
    thl_probs = y_prob[y_true == 1]
    t_lower = np.percentile(thl_probs, 10) if len(thl_probs) > 0 else 0.20
        
    t_upper = np.clip(t_upper, 0.5, 0.99)
    t_lower = np.clip(t_lower, 0.01, 0.49)
        
    return t_upper, t_lower

def calculate_robust_ood_limit(config, val_gen_list, feature_extractors_map, riemann_tools_map, fold_models_map):
    all_valid_energies = []
    FLIPPED_FOLDS = [0, 1, 3]

    for fold in range(config.DATA.N_SPLIT):
        val_gen = val_gen_list[fold]
        feature_extractor = feature_extractors_map[fold]
        riemann_tool = riemann_tools_map[fold]
        
        X_val_z = feature_extractor.predict(val_gen, verbose=2)
        y_true = val_gen.classes if hasattr(val_gen, 'classes') else np.argmax(np.concatenate([y for x, y in val_gen], axis=0), axis=1)
        E_val = riemann_tool.compute_batch_curvature(X_val_z)

        bags_z, bags_e, y_bag_labels = create_patient_bags(df=val_gen.df, features=X_val_z, energies=E_val.reshape(-1, 1))
        X_bag_z, X_bag_e = pad_bags(bags_z), pad_bags(bags_e)
        
        batch_predictions = [model.predict([X_bag_z, X_bag_e], verbose=0) for model in fold_models_map[fold]]
        raw_preds = np.mean(batch_predictions, axis=0)[:, 1] 
        corrected_preds = 1.0 - raw_preds if fold in FLIPPED_FOLDS else raw_preds
            
        patient_correct_mask = ((corrected_preds > 0.5).astype(int) == y_bag_labels)
        valid_energy_bags = [bags_e[i] for i in range(len(bags_e)) if patient_correct_mask[i]]
    
        valid_energies = np.concatenate(valid_energy_bags).flatten() if len(valid_energy_bags) > 0 else E_val.flatten()
        all_valid_energies.extend(valid_energies)
        
    ood_limit = np.percentile(all_valid_energies, 95) 
    return ood_limit, np.array(all_valid_energies)

def calibrate_ood_limit(energies_list, method='percentile', param=99):
    all_energies = np.concatenate([e.flatten() for e in energies_list]) if isinstance(energies_list, list) else energies_list.flatten()
    mean_e, std_e, max_e = np.mean(all_energies), np.std(all_energies), np.max(all_energies)
    
    if method == 'percentile':
        limit = np.percentile(all_energies, param)
    elif method == 'sigma':
        limit = mean_e + (param * std_e)
    elif method == 'max':
        limit = max_e * param
        
    return limit

def apply_riemannian_filter(features, energies, limit, rescue_top_k=0, extreme_limit=100.0, verbose=False):
    e_flat = energies.flatten()
    n_total = len(features)
    valid_mask = e_flat < limit
    n_rescued = 0
    
    if rescue_top_k > 0:
        potential_rescue_idx = np.where((e_flat >= limit) & (e_flat < extreme_limit))[0]
        if len(potential_rescue_idx) > 0:
            top_k_sub_idx = np.argsort(e_flat[potential_rescue_idx])[-rescue_top_k:]
            valid_mask[potential_rescue_idx[top_k_sub_idx]] = True
            n_rescued = len(top_k_sub_idx)

    clean_features = features[valid_mask]
    clean_energies = energies[valid_mask]
    n_valid = np.sum(valid_mask)
    
    stats = {"n_total": n_total, "n_valid": n_valid, "n_rejected": n_total - n_valid, "n_rescued": n_rescued, "reject_rate": (n_total - n_valid) / n_total if n_total > 0 else 0.0}
    return clean_features, clean_energies, valid_mask, stats

def predict_clinical_workflow(inference_gen, inference_gen_id, feature_extractor, riemann_tool, ensemble_models, config, curvature_threshold=15.0, ood_energy_limit=29.5434, rescue_top_k=10, extreme_limit=50.0, verbose=True, return_patch_data=False, enable_ood_filter=True):
    data_item = inference_gen.__getitem__(idx=inference_gen_id, get_pids=True)
    batch_patients, labels_raw, X_bag_batch = data_item[0], data_item[1], data_item[2] if len(data_item) == 4 else data_item[1]
    
    X_patches, pid = X_bag_batch[0], batch_patients[0]
    Z_patient = np.array(feature_extractor.predict(X_patches, batch_size=config['TRAIN']['BATCH_SIZE'], verbose=2)) 
    E_patient = riemann_tool.compute_batch_curvature(Z_patient).reshape(-1, 1) 

    if enable_ood_filter:
        Z_input, E_input, valid_mask, stats = apply_riemannian_filter(Z_patient, E_patient, limit=ood_energy_limit, rescue_top_k=rescue_top_k, extreme_limit=extreme_limit, verbose=verbose)
    else:
        Z_input, E_input, valid_mask = Z_patient, E_patient, np.ones(len(E_patient), dtype=bool)
        stats = {"n_total": len(Z_patient), "n_valid": len(Z_patient), "n_rejected": 0, "reject_rate": 0.0}

    n_diagnostic = len(Z_input)
    final_diagnosis, confidence, avg_prob = "Indeterminate", 0.0, np.array([0.5, 0.5])

    if n_diagnostic > 0:
        max_len = 32
        bag_z, bag_e = np.zeros((1, max_len, Z_input.shape[1])), np.zeros((1, max_len, 1))
        limit = min(n_diagnostic, max_len)
        bag_z[0, :limit, :], bag_e[0, :limit, :] = Z_input[:limit, :], E_input[:limit, :]
        
        ensemble_models = [ensemble_models] if not isinstance(ensemble_models, list) else ensemble_models
        avg_prob = np.mean([model.predict([bag_z, bag_e], verbose=0)[0] for model in ensemble_models], axis=0) 
        
        prediction_idx = np.argmax(avg_prob)
        confidence = avg_prob[prediction_idx]
        final_diagnosis = ["IDA", "THL"][prediction_idx]
        
    try:
        all_patient_paths = inference_gen.patient_groups[pid]['patch_path'].tolist()
        if len(all_patient_paths) != len(Z_patient): all_patient_paths = [f"patch_{i}.png" for i in range(len(Z_patient))]
    except (KeyError, AttributeError):
        all_patient_paths = ["Unknown_Path"] * len(Z_patient)

    explanation_paths = []
    if n_diagnostic > 0:
        top_k_idx_local = np.argsort(E_input.flatten())[-3:][::-1]
        top_indices_original = np.where(valid_mask)[0][top_k_idx_local]
        explanation_paths = [all_patient_paths[i] for i in top_indices_original]

    patch_df = None
    if return_patch_data:
        status_list = ["REJECTED (OOD)" if not valid_mask[i] else "Accepted (High Energy)" if e > curvature_threshold else "Accepted (Low Energy)" for i, e in enumerate(E_patient.flatten())]
        patch_df = pd.DataFrame({"patient_id": [pid] * len(Z_patient), "patch_path": all_patient_paths, "curvature_energy": E_patient.flatten(), "is_valid": valid_mask, "status": status_list})

    return {
        "diagnosis": final_diagnosis, "confidence": float(confidence), "raw_probs": avg_prob,
        "n_total": stats['n_total'], "n_valid": stats['n_valid'], "n_rejected": stats['n_rejected'], "reject_rate": stats['reject_rate'],
        "explanation_caption": f"Diagnosed {final_diagnosis} ({confidence:.1%}).\nSafety Filter: {stats['n_valid']} accepted, {stats['n_rejected']} rejected.",
        "explanation_images": explanation_paths, "patch_dataframe": patch_df 
    }

# ==============================================================================
# Main Execution Pipeline (Command-Line Interface)
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="R-MIL Hematopathology Pipeline")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'calibrate', 'test', 'xai'],
                        help="Execution mode: train, calibrate, test, or xai")
    parser.add_argument('--data_dir', type=str, default=None, help="Override default dataset path")
    parser.add_argument('--weights_dir', type=str, default=None, help="Override default models path")
    args = parser.parse_args()

    # 1. Initialize Base Config
    config = configsCell_mil_stage2.get_config()
    t1 = datetime.now()  

    # Override paths if provided via CLI
    if args.data_dir:
        config.DATASET = args.data_dir
    if args.weights_dir:
        config.BASEMODEL_PATH = args.weights_dir
        config.MODEL_PATH = args.weights_dir

    # 2. Reset Execution Flags based on Mode
    config.TRAIN.Enable = False
    config.TRAIN.Phase3 = False
    config.Evaluate_Val = False
    config.Evaluate_Test = False
    config.find_ood = False
    config.find_threshold = False
    config.PLOT_CURVATURE = False

    if args.mode == 'train':
        print("\n--- MODE: R-MIL ENSEMBLE TRAINING ---")
        config.TRAIN.Enable = True
        config.TRAIN.Phase3 = True
    elif args.mode == 'calibrate':
        print("\n--- MODE: THRESHOLD CALIBRATION (PHASE 1B) ---")
        config.Evaluate_Val = True
        config.find_ood = True
        config.find_threshold = True
    elif args.mode == 'test':
        print("\n--- MODE: INDEPENDENT TEST INFERENCE ---")
        config.Evaluate_Test = True
        config.Performance_Report = True 
    elif args.mode == 'xai':
        print("\n--- MODE: EXPLAINABLE AI VISUALIZATION ---")
        config.PLOT_CURVATURE = True

    # ---------------------------------------------------------
    # Prepare Data Generators
    # ---------------------------------------------------------
    model_names = config['MODEL']['NAMES']
    dataset_root = config['DATASET']
    class_labels = config['DATA']['CLASS_LABELS']
    model_name = config.MODEL.NAMES[0]
    
    found_files, patient_patch_count = mygears.create_cell_dataframe_debug(config)
    df = mygears.create_cell_dataframe_flex(config, glob_pattern="[01]/*/normalized/*-cells-256/*.png")
    all_patient_ids = df['patient_id'].unique()

    patient_labels = df.groupby('patient_id')['label'].first() 

    train_patients, test_patients = train_test_split(
        patient_labels.index,
        test_size=config.DATA.TEST_SIZE,
        stratify=patient_labels.values,
        random_state=config.DATA.SEED
    )

    train_df = df[df['patient_id'].isin(train_patients)]
    test_df = df[df['patient_id'].isin(test_patients)]

    train_patient_labels = train_df.groupby('patient_id')['label'].first()
    sss = StratifiedShuffleSplit(n_splits=config.DATA.N_SPLIT, test_size=config.DATA.val_ratio, random_state=config.DATA.SEED)

    train_df_fold_list = []
    val_df_fold_list = []

    for fold, (train_idx, val_idx) in enumerate(sss.split(train_patient_labels.index, train_patient_labels.values)):
        train_fold_patients = train_patient_labels.index[train_idx]
        val_fold_patients   = train_patient_labels.index[val_idx]

        train_df_fold_list.append(train_df[train_df['patient_id'].isin(train_fold_patients)])
        val_df_fold_list.append(train_df[train_df['patient_id'].isin(val_fold_patients)])
        
    model_builder = ModelBuilder(model_name=model_name, config=config)
    preprocessing_fn = model_builder.get_preprocessing_function()        

    train_gen_list = []
    val_gen_list = []

    for fold in range(config.DATA.N_SPLIT): 
        train_df_fold = train_df_fold_list[fold].copy()
        val_df_fold   = val_df_fold_list[fold].copy()
           
        if platform.system() == 'Linux' and config['DATA']['VERIFY']:
            train_df_fold = mygears.verify_images(train_df_fold, 'patch_path')
            val_df_fold = mygears.verify_images(val_df_fold, 'patch_path')
            
        train_log = config.BASE + '-Rieman-' + f"Fold-{fold}-" + model_name + '-' + config.MIL_MODEL + '-' + config['TRAIN']['DATETIME']        
            
        if config['SAVE']:            
            os.makedirs(train_log, exist_ok=True)
            json_file_path = os.path.join(train_log,'_config.json')  
            with open(json_file_path, 'w') as json_file:    
                json.dump(mygears.cfg_to_dict(config), json_file, indent=4)            

        train_df_fold = train_df_fold.sample(frac=1, random_state=config.DATA.SEED).reset_index(drop=True)
        train_df_fold_sampled = mygears.stratified_subsample(train_df_fold, config['TRAIN']['SAMPLE_SIZE']) if config['TRAIN']['SAMPLE'] else train_df_fold
            
        train_gen = PatchDatasetPreserve(
            config,
            df=train_df_fold_sampled,
            batch_size=config.TRAIN.BATCH_SIZE,
            preprocessing_function=preprocessing_fn,
        )
            
        val_df_fold = val_df_fold.sample(frac=1, random_state=config.DATA.SEED).reset_index(drop=True)
        val_df_fold_sampled = mygears.stratified_subsample(val_df_fold, config['TRAIN']['SAMPLE_SIZE_VAL']) if config['TRAIN']['SAMPLE'] else val_df_fold
            
        val_gen = PatchDatasetPreserve(
            config,
            df=val_df_fold_sampled,
            batch_size=config.TRAIN.BATCH_SIZE,
            preprocessing_function=preprocessing_fn,
            shuffle=False
        )
            
        train_gen_list.append(train_gen)
        val_gen_list.append(val_gen)


    # ---------------------------------------------------------
    # Execute Pipeline
    # ---------------------------------------------------------
    
    # === PHASE 1: TRAIN MODEL & MANIFOLD ===
    if config.TRAIN.Enable:
        model_mapping = {
            model_name: [ 
                os.path.join(config.BASEMODEL_PATH, 'Linux-Fold-0-ConvNeXtLarge-classweights-20251205-1028.hdf5'),
                os.path.join(config.BASEMODEL_PATH, 'Linux-Fold-1-ConvNeXtLarge-classweights-20251205-2213.hdf5'),
                os.path.join(config.BASEMODEL_PATH, 'Linux-Fold-2-ConvNeXtLarge-classweights-20251205-2213.hdf5'),
                os.path.join(config.BASEMODEL_PATH, 'Linux-Fold-3-ConvNeXtLarge-classweights-20251205-2213.hdf5'),
                os.path.join(config.BASEMODEL_PATH, 'Linux-Fold-4-ConvNeXtLarge-classweights-20251205-2213.hdf5'),
            ],
        }

        trained_models_list = mygears.load_ensemble_models(model_mapping, config)
        riemann_tools_all, feature_extractors_all, X_train_z_all, X_val_z_all, y_val_z_all, energies_val_all, thresholds_all = [], [], [], [], [], [], []
            
        for fold in range(config.DATA.N_SPLIT):
            train_gen_per_fold = train_gen_list[fold]
            val_gen_per_fold = val_gen_list[fold]
            model_per_fold = trained_models_list[fold]
            
            feature_extractor, X_train_z, y_train_z = p1_step1_extract_features(model_name=model_name, model=model_per_fold, data_gen=train_gen_per_fold, layer_index=config.TRAIN.Feature_Layer_Index)
            riemann_tool = p1_step2_fit_manifold(features=X_train_z, labels=y_train_z, n_classes=config.MODEL.NUM_CLASSES, feature_dim=X_train_z.shape[1])
            E_train = p1_step3_compute_train_energy(riemann_tool=riemann_tool, features=X_train_z, save_path=os.path.join(config.MODEL_PATH, f"fold_{fold}_{model_name}_manifold.pkl"))
            
            riemann_tools_all.append(riemann_tool)
            feature_extractors_all.append(feature_extractor)
            X_train_z_all.append(X_train_z)
        
            X_val_z, y_val_z = p2_step1_extract_val_features(model_name=model_name, feature_extractor=feature_extractor, val_gen=val_gen_per_fold)
            X_val_z_all.append(X_val_z)
            y_val_z_all.append(y_val_z)
            
            E_val = p2_step2_compute_val_energy(riemann_tool=riemann_tool, X_val_z=X_val_z, save_path=os.path.join(config.MODEL_PATH, f"fold_{fold}_{model_name}_val_manifold.pkl"))
            _, threshold = p2_step3_calibrate_threshold(E_val=E_val, y_val=y_val_z, config=config)
            
            energies_val_all.append(E_val)
            thresholds_all.append(threshold)
            
            del model_per_fold, X_train_z, E_train, X_val_z, E_val
            gc.collect()
            
            if fold == config.TRAIN.Actual_Fold: break
        
        threshold_data = {
            "per_fold": {f"fold_{i}": float(t) for i, t in enumerate(thresholds_all)},
            "statistics": {"mean": float(np.mean(thresholds_all)), "std": float(np.std(thresholds_all)), "min": float(np.min(thresholds_all)), "max": float(np.max(thresholds_all))},
            "config": {"model_name": model_name, "timestamp": config.TRAIN.DATETIME}
        }
        
        try:
            with open(os.path.join(config.MODEL_PATH, f"{model_name}_manifold_thresholds_{config.TRAIN.DATETIME}.json"), 'w') as f:
                json.dump(threshold_data, f, indent=4)
        except Exception as e:
            print(f"[Error] Failed to save thresholds: {e}")

        if config.TRAIN.Phase3:
            _, fold_thresholds = mygears.load_thresholds(os.path.join(config.MODEL_PATH, "ConvNeXtLarge_manifold_thresholds_20251217-2135.json"))
            riemann_tools_map, X_train_z_all, X_val_z_all = {}, {}, {} 
            
            for fold in range(config.DATA.N_SPLIT):
                riemann_tools_map[fold] = mygears.load_one_manifold(fold, model_name, config, val=False)
                X_train_z_all[fold] = np.load(os.path.join(config.MODEL_PATH, f"fold_{fold}_{model_name}_features.npy"))
                
                val_path = os.path.join(config.MODEL_PATH, f"fold_{fold}_{model_name}_val_features.npy")
                if os.path.exists(val_path):
                    X_val_z_all[fold] = np.load(val_path)
                else:
                    feature_extractor = mygears.load_one_extractor(fold, model_name, config)
                    X_val_z, _ = p2_step1_extract_val_features(model_name=model_name, feature_extractor=feature_extractor, val_gen=val_gen_list[fold], verbose=False)
                    np.save(val_path, X_val_z)
                    X_val_z_all[fold] = X_val_z
                    del feature_extractor
                    tf.keras.backend.clear_session()
                
            fold_aurocs, all_fold_mil_ensembles = [], {} 
            
            for fold in range(config.DATA.N_SPLIT):
                current_threshold = fold_thresholds[f"fold_{fold}"]
                riemann_tool = riemann_tools_map[fold]
                
                mil_ensemble = train_robust_mil_ensemble(
                    model_name, fold=fold, train_df=train_gen_list[fold].df, features=X_train_z_all[fold], 
                    riemann_tool=riemann_tool, config=config, fold_threshold=current_threshold, n_estimators=5
                )
                all_fold_mil_ensembles[fold] = mil_ensemble
                
                E_val_norm = np.clip(riemann_tool.compute_batch_curvature(X_val_z_all[fold]).reshape(-1, 1) / current_threshold, 0, 5.0)
                bags_val_z, bags_val_e, y_val_bag_labels = create_patient_bags(df=val_gen_list[fold].df, features=X_val_z_all[fold], energies=E_val_norm)
                
                X_bag_val_z, X_bag_val_e = pad_bags(bags_val_z, max_len=32), pad_bags(bags_val_e, max_len=32)
                avg_val_probs = np.mean([model.predict([X_bag_val_z, X_bag_val_e], verbose=2) for model in mil_ensemble], axis=0)
                
                fold_aurocs.append(roc_auc_score(y_val_bag_labels, avg_val_probs[:, 1]) if len(np.unique(y_val_bag_labels)) > 1 else 0.0)
                
                if fold == config['TRAIN']['Actual_Fold']: break


    # === PHASE 2: CALIBRATE THRESHOLDS & OOD LIMITS ===
    if config.Evaluate_Val:
        feature_extractors_map = {fold: mygears.load_one_extractor(fold, model_name, config) for fold in range(config.DATA.N_SPLIT)}
        riemann_tools_map = {fold: mygears.load_one_manifold(fold, model_name, config) for fold in range(config.DATA.N_SPLIT)}
        all_fold_ensembles = {fold: mygears.load_mil_ensemble_for_fold(fold, model_name, config, n_estimators=5) for fold in range(config.DATA.N_SPLIT)}

        if config.find_ood:        
            ood_limit, energies_list = calculate_robust_ood_limit(config, val_gen_list, feature_extractors_map, riemann_tools_map, all_fold_ensembles)
            ood_energy_limit_percentile = calibrate_ood_limit(energies_list, method='percentile', param=95)
            ood_energy_limit_sigma = calibrate_ood_limit(energies_list, method='sigma', param=95)
            ood_energy_limit_max = calibrate_ood_limit(energies_list, method='max', param=95)

        if config.find_threshold:
            fold_aurocs, all_validation_patient_predictions_list, fold_t_uppers, fold_t_lowers = [], [], [], []
            X_val_z_all, y_val_z_all, energies_val_all, thresholds_all = [], [], [], []
            
            for fold in range(config.DATA.N_SPLIT):
                train_gen_per_fold, val_gen_per_fold = train_gen_list[fold], val_gen_list[fold]
                feature_extractor, riemann_tool = feature_extractors_map[fold], riemann_tools_map[fold]

                X_val_z, y_val_z = p2_step1_extract_val_features(model_name, feature_extractor=feature_extractor, val_gen=val_gen_per_fold)
                X_val_z_all.append(X_val_z)
                y_val_z_all.append(y_val_z)
                
                E_val = p2_step2_compute_val_energy(riemann_tool=riemann_tool, X_val_z=X_val_z, save_path=os.path.join(config.MODEL_PATH, f"fold_{fold}_manifold_val.pkl"))
                _, threshold = p2_step3_calibrate_threshold(E_val=E_val, y_val=y_val_z, config=config)
                
                energies_val_all.append(E_val)
                thresholds_all.append(threshold)
                
                bags_val_z, bags_val_e, y_val_bag_labels = create_patient_bags(df=val_gen_per_fold.df, features=X_val_z, energies=E_val.reshape(-1, 1))
                X_bag_val_z, X_bag_val_e = pad_bags(bags_val_z, max_len=32), pad_bags(bags_val_e, max_len=32)
                
                avg_val_probs = np.mean([model.predict([X_bag_val_z, X_bag_val_e], verbose=2) for model in all_fold_ensembles[fold]], axis=0)
                prob_thl_val = avg_val_probs[:, 1]

                if fold in [0, 1, 3]: prob_thl_val = 1.0 - prob_thl_val
                fold_aurocs.append(roc_auc_score(y_val_bag_labels, prob_thl_val) if len(np.unique(y_val_bag_labels)) > 1 else 0.0)
                
                t_up, t_low = calculate_optimal_thresholds(y_val_bag_labels, prob_thl_val)
                fold_t_uppers.append(t_up)
                fold_t_lowers.append(t_low)
                
                unique_patients = val_gen_per_fold.df['patient_id'].unique()
                for pid, prob, true_lbl in zip(unique_patients, prob_thl_val, y_val_bag_labels):
                    all_validation_patient_predictions_list.append({
                        "val_fold": fold, "patient_id": pid, "probability_1": f"{prob:.4f}", "label": int(true_lbl), "threshold_upper_used": f"{t_up:.4f}", "threshold_lower_used": f"{t_low:.4f}"
                    })
            
            final_t_upper, final_t_lower = np.median(fold_t_uppers), np.median(fold_t_lowers)
            
            with open(os.path.join(config.SAVE_PATH, "patient_predictions_mil_val.csv"), 'w', newline='') as output_file:
                dict_writer = csv.DictWriter(output_file, fieldnames=all_validation_patient_predictions_list[0].keys())
                dict_writer.writeheader()
                dict_writer.writerows(all_validation_patient_predictions_list)


    # === PHASE 3: INDEPENDENT TEST INFERENCE ===
    if config.Evaluate_Test: 
        CLINICAL_WORKFLOW_CONFIG = {
            "MODEL_DIR": config.MODEL_PATH,
            "MODEL_NAME_BASE": "ConvNeXtLarge",
            "ACTIVE_FOLD": 4, 
            "N_ESTIMATORS": 5, 
            "OOD_ENERGY_LIMIT": 29.5434,   
            "EXTREME_CEILING": 50.0,       
            "RESCUE_TOP_K": 10,            
            "THRESHOLDS_CURVE": 15,
            "THRESHOLDS": {"CONFIDENT_THL": 0.5463, "CONFIDENT_IDA": 0.4900},
            "FLIPPED_FOLDS": [0, 1, 3] 
        }

        LOAD_STRATEGY = 'BEST_FOLD' 
        prefix = "val_fold_4_filter_ood_rescue"
        
        feature_extractors_map, riemann_tools_map, all_fold_ensembles = {}, {}, {}
        folds_to_load = [CLINICAL_WORKFLOW_CONFIG["ACTIVE_FOLD"]] if LOAD_STRATEGY == 'BEST_FOLD' else range(config.DATA.N_SPLIT)
        
        for fold in folds_to_load:
            feature_extractors_map[fold] = mygears.load_one_extractor(fold, model_name, config.BASEMODEL_PATH)
            riemann_tools_map[fold] = mygears.load_one_manifold(fold, model_name, config.MODEL_PATH)
            all_fold_ensembles[fold] = mygears.load_mil_ensemble_for_fold(fold, model_name, config.MODEL_PATH, n_estimators=5)

        fold = CLINICAL_WORKFLOW_CONFIG["ACTIVE_FOLD"]
        
        test_df = test_df.sample(frac=1, random_state=config.DATA.SEED).reset_index(drop=True)
        test_df_sampled = mygears.stratified_subsample(test_df, config.TRAIN.SAMPLE_SIZE_TEST) if config.TRAIN.SAMPLE else test_df
        inference_gen = FullBagDataset(config, df=test_df_sampled, preprocessing_function=preprocessing_fn, expect_rgba=True, bg_color=(0, 0, 0), mode='test')
        
        all_patients_patch_audit, ensemble_results_data = [], []
        
        for patient_idx in range(0, len(inference_gen)):
            p_ids, labels_raw, X_bag_batch, labels_onehot = inference_gen.__getitem__(idx=patient_idx, get_pids=True)
            current_pid, true_label = p_ids[0], int(labels_raw[0])       

            if LOAD_STRATEGY == 'BEST_FOLD':
                report = predict_clinical_workflow(
                    inference_gen, patient_idx, feature_extractors_map[fold], riemann_tools_map[fold], all_fold_ensembles[fold], config, 
                    curvature_threshold=CLINICAL_WORKFLOW_CONFIG["THRESHOLDS_CURVE"], ood_energy_limit=CLINICAL_WORKFLOW_CONFIG["OOD_ENERGY_LIMIT"], 
                    rescue_top_k=CLINICAL_WORKFLOW_CONFIG["RESCUE_TOP_K"], extreme_limit=CLINICAL_WORKFLOW_CONFIG["EXTREME_CEILING"], 
                    enable_ood_filter=True, verbose=True, return_patch_data=True
                )
                final_prob_thl, patch_audit_frame, report_images = report['raw_probs'][1], report['patch_dataframe'], report['explanation_images']

            elif LOAD_STRATEGY == 'ENSEMBLE':
                fold_probs = []
                for fold in folds_to_load:
                    report = predict_clinical_workflow(
                        inference_gen, patient_idx, feature_extractors_map[fold], riemann_tools_map[fold], all_fold_ensembles[fold], config, 
                        enable_ood_filter=False, ood_energy_limit=CLINICAL_WORKFLOW_CONFIG["OOD_ENERGY_LIMIT"], verbose=False, return_patch_data=(fold==CLINICAL_WORKFLOW_CONFIG["ACTIVE_FOLD"])
                    )
                    raw_p = report['raw_probs'][1]
                    fold_probs.append(1.0 - raw_p if fold in CLINICAL_WORKFLOW_CONFIG["FLIPPED_FOLDS"] else raw_p)
                    if report['patch_dataframe'] is not None: patch_audit_frame = report['patch_dataframe']
                final_prob_thl = np.mean(fold_probs)

            t_upper = CLINICAL_WORKFLOW_CONFIG["THRESHOLDS"]["CONFIDENT_THL"] 
            if final_prob_thl >= t_upper:
                final_diagnosis, decision_status, pred_class = "THL", "POSITIVE (High Confidence)", 1
            else:
                final_diagnosis, decision_status, pred_class = "Screen Negative (Likely IDA)", "NEGATIVE (Rule-Out)", 0
                
            stats_source = report 
            ensemble_results_data.append({
                "patient_id": current_pid, "true_label": true_label, "predicted_class": pred_class, "probability_thl": final_prob_thl, "diagnosis_text": final_diagnosis, "status": decision_status, "strategy": LOAD_STRATEGY,
                "total_patches": stats_source['n_total'], "accepted_patches": stats_source['n_valid'], "rejected_patches": stats_source['n_rejected'], "rejection_rate": f"{stats_source['reject_rate']:.4f}"
            })
            
            if patch_audit_frame is not None:
                patch_audit_frame['label'] = true_label
                try:
                    patient_sub_df = inference_gen.df[inference_gen.df['patient_id'] == current_pid]
                    patch_audit_frame['patch_path'] = patient_sub_df['patch_path'].values if len(patient_sub_df) == len(patch_audit_frame) and 'patch_path' in patient_sub_df.columns else "mismatch_length"
                except Exception as e:
                    patch_audit_frame['patch_path'] = f"error_{str(e)}"
                all_patients_patch_audit.append(patch_audit_frame)

        if ensemble_results_data:
            csv_path = os.path.join(config.SAVE_PATH, f"{prefix}_clinical_report_test_{LOAD_STRATEGY}.csv")
            try:
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=ensemble_results_data[0].keys())
                    writer.writeheader()
                    writer.writerows(ensemble_results_data)
            except Exception as e: print(f"[Error] {e}")

        if all_patients_patch_audit:
            full_audit_df = pd.concat(all_patients_patch_audit, ignore_index=True)
            full_audit_df.to_csv(os.path.join(config.SAVE_PATH, f"{prefix}_patch_audit_test_{LOAD_STRATEGY}.csv"), index=False)

        if config.Performance_Report:       
            T_LOWER, T_UPPER, T_OPT = 0.4914, 0.6505, 0.5742 
            try:
                df_perf = pd.read_csv('val_predictions_mean_QC_Std_20251220-1240.csv')
                y_true_perf, y_prob_perf = df_perf['true_class'].to_numpy(), df_perf['probability_1'].to_numpy()
                df_triage = pd.DataFrame([mygears.apply_clinical_triage(p, T_LOWER, T_UPPER, T_OPT) for p in y_prob_perf])
                df_perf = pd.concat([df_perf, df_triage.drop(columns=['probability'])], axis=1)
                
                y_pred_confident = df_perf.loc[df_perf['pred_status'] == 'Confident', 'final_pred'].to_numpy()
                y_true_confident = df_perf.loc[df_perf['pred_status'] == 'Confident', 'true_class'].to_numpy()
                
                print(classification_report(y_true=y_true_confident, y_pred=y_pred_confident, target_names=config['DATA']['CLASS_LABELS']))
                p_measures, cm = mygears.evaluate_classification_performance(y_true_confident, y_pred_confident, class_labels=config['DATA']['CLASS_LABELS'], save_dir='.', save_prefix=str('Val'))
                mygears.plot_confusion_matrices(confusionmatrix=cm, p_measures=p_measures, class_labels=config['DATA']['CLASS_LABELS'], save_dir='.', save_prefix=str('Val'))  
            except FileNotFoundError:
                print("Performance Report file not found. Skipping metrics generation.")


    # === PHASE 4: XAI VISUALIZATION ===
    if config.PLOT_CURVATURE:
        fold = 4 
        feature_extractor = mygears.load_one_extractor(fold, model_name, config.BASEMODEL_PATH)
        X_train_full = np.load(f'models/fold_{fold}_ConvNeXtLarge_features.npy')
        y_train_full = train_gen_list[fold].classes 
        
        X_train, y_train = (X_train_full[:30000], y_train_full[:30000]) if len(X_train_full) > 30000 else (X_train_full, y_train_full)
        riemann_tool = mygears.load_one_manifold(fold, 'ConvNeXtLarge', config.MODEL_PATH, val=False)
        E_train = riemann_tool.compute_batch_curvature(X_train)
        
        ood_gen = PatchDatasetPreserve(
            config, df=val_df_fold_list[fold][:9000], batch_size=config.TRAIN.BATCH_SIZE, preprocessing_function=preprocessing_fn, shuffle=True, expect_rgba=True
        )

        try:
            X_ood_full = feature_extractor.predict(ood_gen, verbose=1)
            E_ood_full = riemann_tool.compute_batch_curvature(X_ood_full)
            OOD_ENERGY_LIMIT = 29.5434    
            valid_group, ood_group, stats = split_ood_data(X_ood_full, E_ood_full, OOD_ENERGY_LIMIT)       
            X_valid, X_ood = valid_group['X'], ood_group['X']
        except Exception as e:
            X_ood_available = False

        plot_curvature_manifold(X_train, y_train, E_train, X_ood_full, E_ood_full, ood_limit=OOD_ENERGY_LIMIT, save_path="spacetime_curvature_IDA_THL_OOD_Val_fold_4_mil_1.html")

    t2 = datetime.now() - t1
    print(f'\nPipeline [{args.mode}] execution time: {t2}')
    