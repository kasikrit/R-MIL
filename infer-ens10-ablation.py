#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 18:41:47 2025

@author: kasikritdamkliang
"""
import tensorflow as tf
import os
import cv2
from datetime import datetime
from tqdm import tqdm
# import itertools
import matplotlib.pylab as plt
import numpy as np
import pandas as pd

# from tensorflow.keras import models, layers, optimizers, Model
from sklearn.metrics import confusion_matrix, classification_report
# from tensorflow.keras.utils.np_utils import to_categorical
from tensorflow.keras.utils import to_categorical
import tensorflow_addons as tfa
tqdm_callback = tfa.callbacks.TQDMProgressBar(show_epoch_progress=False)
# from livelossplot import PlotLossesKeras


# import tensorflow.keras.layers as L
import warnings
warnings.filterwarnings("ignore")

import seaborn as sns
# from imblearn.over_sampling import RandomOverSampler
# from imutils import paths

# from keras.applications import ResNet50
# from tensorflow.keras.models import Sequential, Model
# from keras.applications.resnet50 import preprocess_input

# from keras_applications.resnet import ResNet50

# from tensorflow.keras.preprocessing import image

# from tensorflow.keras.applications.resnet_v2 import (
#     ResNet101V2, ResNet50V2, preprocess_input)





# from tqdm import tqdm
# from vit_keras import vit, utils, visualize

import random
import json
from yacs.config import CfgNode as CN

# from tensorflow.keras.applications.efficientnet import (
#     EfficientNetB7, preprocess_input)


def cfg_to_dict(cfg_node):
    """
    Recursively convert a YACS CfgNode to a nested dictionary.
    """
    if isinstance(cfg_node, CN):
        return {k: v for k, v in cfg_node.items()}
    return cfg_node

def filter_hidden_files(file_paths):
    """
    Filter out paths that start with a '.' indicating they are hidden.
    This only considers the filename, not part of the path, as hidden.
    """
    return [path for path in file_paths if not os.path.basename(path).startswith('.')]

# def seed_everything(seed = seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     tf.random.set_seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     os.environ['TF_DETERMINISTIC_OPS'] = '1'

# seed_everything()

print(f"{tf.__version__ = }") #2.11.0, python = 3.9.16
#2.3.0, python = 3.6
print(tf.test.gpu_device_name())

# print(f"{keras.__version__ = }") #2.11.0, python = 3.9.16
#2.4.0, python = 3.6

# See https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
def make_pair(imgs,labels):
    pairs = []
    for img, mask in zip(imgs, labels):
        pairs.append( (img, mask) )
    
    return pairs

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    
def plot_images_with_labels(images, labels, train_log, set_label, save=False):
    """
    Plots a grid of images with their corresponding labels and saves the figure.
    
    Args:
    - images (list or np.array): List of images to be displayed.
    - labels (np.array): Array of one-hot encoded labels.
    - train_log (str): The directory or log name to save the plot.
    - set_label (str): The specific set label (e.g., "train", "validation", etc.).
    - config (dict): Configuration dictionary containing image dimensions.
    
    Returns:
    - None: Displays and saves the plot.
    """
    # Convert one-hot encoded labels to class indices
    class_indices = np.argmax(labels, axis=1)

    # Create subplots
    fig, axes = plt.subplots(3, 4, figsize=(14, 10), dpi=120)
    axes = axes.flatten()

    for img, label_idx, ax in zip(images, class_indices, axes):
        # Undo any preprocessing (like rescaling for pre-trained models)
        if np.max(img) == 1:
            img = img * 255

        # Display the image
        img_shape = (config['DATA']['W'], config['DATA']['H'], config['DATA']['C'])
        ax.imshow(img.reshape(img_shape).astype("uint8"))

        # Set the title to the corresponding class label
        ax.set_title(f"Label: {label_idx}", fontsize=18)
        ax.axis('on')

    # Set the overall title for the figure
    plot_title = f"{train_log}-{set_label}"
    plt.suptitle(plot_title, fontsize=18)

    # Show the plot
    plt.show()

    # Save the figure
    if save:
        plot_file = os.path.join(train_log, f"{plot_title}.png")
        fig.savefig(plot_file, dpi=120)
        print(f'Saved {plot_file}')


def plot_images_without_labels(images, train_log, set_label, save=False):
    """
    Plots a grid of images without displaying labels and optionally saves the figure.

    Args:
    - images (list or np.array): List of images to be displayed.
    - train_log (str): The directory or log name to save the plot.
    - set_label (str): The specific set label (e.g., "train", "validation", etc.).
    
    Returns:
    - None: Displays and optionally saves the plot.
    """
    # Create subplots
    fig, axes = plt.subplots(3, 4, figsize=(14, 10), dpi=120)
    axes = axes.flatten()

    for img, ax in zip(images, axes):
        # Undo any preprocessing (like rescaling for pre-trained models)
        if np.max(img) == 1:
            img = img * 255

        # Display the image
        img_shape = (config['DATA']['W'], config['DATA']['H'], config['DATA']['C'])
        ax.imshow(img.reshape(img_shape).astype("uint8"))

        # Remove axis labels and ticks
        ax.axis('off')

    # Set the overall title for the figure
    plot_title = f"{train_log}-{set_label}"
    plt.suptitle(plot_title, fontsize=18)

    # Show the plot
    plt.show()

    # Save the figure
    if save:
        plot_file = os.path.join(train_log, f"{plot_title}.png")
        fig.savefig(plot_file, dpi=120, bbox_inches='tight')
        print(f'Saved {plot_file}')

    
# Wrapper preprocessing function
# def vit_preprocess_with_conversion(img):
#     if isinstance(img, Image.Image):  # If the image is a PIL image (e.g., PngImageFile)
#         img = np.asarray(img)  # Convert to NumPy array
#     return vit.preprocess_inputs(img)  # Apply the ViT preprocessing after conversion
    
def get_preprocessing_function(model_name):
    # Mapping model names to their respective preprocessing functions
    preprocessing_mapping = {
        'VGG16': tf.keras.applications.vgg16.preprocess_input,
        'VGG19': tf.keras.applications.vgg19.preprocess_input,
        'EfficientNetB7': tf.keras.applications.efficientnet.preprocess_input,
        'Xception': tf.keras.applications.xception.preprocess_input,
        'InceptionResNetV2': tf.keras.applications.inception_resnet_v2.preprocess_input,
        'InceptionV3': tf.keras.applications.inception_v3.preprocess_input,
        'DenseNet121': tf.keras.applications.densenet.preprocess_input,
        'DenseNet201': tf.keras.applications.densenet.preprocess_input,
        'MobileNet': tf.keras.applications.mobilenet.preprocess_input,
        'MobileNetV2': tf.keras.applications.mobilenet_v2.preprocess_input,
        
        # EfficientNetV2 models use the same preprocessing function
        'EfficientNetV2B3': tf.keras.applications.efficientnet_v2.preprocess_input,
        'EfficientNetV2S': tf.keras.applications.efficientnet_v2.preprocess_input,
        'EfficientNetV2M': tf.keras.applications.efficientnet_v2.preprocess_input,
        'EfficientNetV2L': tf.keras.applications.efficientnet_v2.preprocess_input,
        
        # ConvNeXt models use the same preprocessing function
        'ConvNeXtTiny': tf.keras.applications.convnext.preprocess_input,
        'ConvNeXtSmall': tf.keras.applications.convnext.preprocess_input,
        'ConvNeXtBase': tf.keras.applications.convnext.preprocess_input,
        'ConvNeXtLarge': tf.keras.applications.convnext.preprocess_input,
        'ConvNeXtXLarge': tf.keras.applications.convnext.preprocess_input,
    }

    if model_name in preprocessing_mapping:
        return preprocessing_mapping[model_name]
    # elif model_name.startswith('ViT'):
    #     return _preprocess_vit
    else:
        raise ValueError(f"No preprocessing function for model {model_name}")

# def _preprocess_vit(img):
#     if isinstance(img, Image.Image):
#         img = np.asarray(img)
#     return vit.preprocess_inputs(img)
    
def print_model_summary(model):
    total_params = model.count_params()  # Total parameters (trainable + non-trainable)
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])  # Trainable parameters
    non_trainable_params = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])  # Non-trainable parameters

    # Print the model summary information
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: {non_trainable_params:,}")
 
#%       
def custom_preprocessing_function(image, model_name):
    preprocessing_function = get_preprocessing_function(model_name)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_pre = preprocessing_function(image_bgr)
   
    return image_pre


import mygears
from FullBagDataset import FullBagDataset # Ensure this is imported
from ModelBuilder import ModelBuilder
from sklearn.model_selection import train_test_split
# import configsInfer
# config = configsInfer.get_config()
import configsCell as cf
config = cf.get_config()
print(f"{config=}")

#% Inference for >= 2 models
t1 = datetime.now()
VAL = False
INFERENCE = True
#% Model names for the ensemble
model_names = config['MODEL']['NAMES']
model_name = model_names[0]
class_labels = ['IDA', 'THL']
dataset_root = config['DATASET']
print(f'{dataset_root=}')
print(f'{model_name=}')

#%%
# found_files, patient_patch_count = mygears.create_cell_dataframe_debug(config)
df = mygears.create_cell_dataframe_flex(config,
    glob_pattern = "[01]/*/normalized/*-cells-256/*.png")
# Get a unique list of all patients
all_patient_ids = df['patient_id'].unique()
# Now, df should contain data for all 20 patients
print(f"\nDataFrame created with {df['patient_id'].nunique()} unique patients.\n")
print(f"Number of segmented cells: {len(df)}")
print(df.columns)
print(df.sample(1))

#%% Get unique patients and their labels
patient_labels = df.groupby('patient_id')['label'].first()  # one label per patient

# Split patients into train and test (70:30)
train_patients, test_patients = train_test_split(
    patient_labels.index,
    test_size=config.DATA.TEST_SIZE, #0.30
    stratify=patient_labels.values,
    random_state=config.DATA.SEED
)

# Create train/test DataFrames
train_df = df[df['patient_id'].isin(train_patients)]
test_df = df[df['patient_id'].isin(test_patients)]

print(f"\nTrain patients: {train_df['patient_id'].nunique()}, Test patients: {test_df['patient_id'].nunique()}")
print(f"\n{train_patients=}")
print(f"\n{test_patients=}")

#%% Prepare MIL models and other staffs

CLINICAL_WORKFLOW_CONFIG = {
    "MODEL_DIR": config.MODEL_PATH,
    "MODEL_NAME_BASE": "ConvNeXtLarge",
    "ACTIVE_FOLD": 4,           # Best Fold
    "N_ESTIMATORS": 5,          # Estimators per fold
    
    # Safety
    "OOD_ENERGY_LIMIT": 29.5434,   # Standard Safety Gate (95th Percentile)
    "EXTREME_CEILING": 50.0,       # The Hard Outlier Cap (blocks absolute artifacts)
    "RESCUE_TOP_K": 10,            # Force-keep the 10 most salient 'outlier' cells
    
    "THRESHOLDS_CURVE": 15,
    
    # Diagnostic Thresholds (Fold 4 Specific)
    # "THRESHOLDS": {
    #     "CONFIDENT_THL": 0.5463,  
    #     "CONFIDENT_IDA": 0.4900   
    # },
    
    # Polarity Correction
    # Fold 4 is correct. Folds 0,1,3 are flipped.
    "FLIPPED_FOLDS": [0, 1, 3] 
}

LOAD_STRATEGY = 'BEST_FOLD' 

#%%
print(f"\n{'='*40}\nPreparing CLINICAL INFERENCE QC: {LOAD_STRATEGY}\n{'='*40}")

# ---------------------------------------------------------
# 2. Load Models based on Strategy
# ---------------------------------------------------------

#% Load Base Ensemble Models - ConvNextLarge
loaded_model_list = mygears.load_ensemble_models(model_names, config)
print(f"Loaded {len(loaded_model_list)} models for ensemble.")

feature_extractors_map = {}
riemann_tools_map = {}
all_fold_ensembles = {}

folds_to_load = [CLINICAL_WORKFLOW_CONFIG["ACTIVE_FOLD"]] if LOAD_STRATEGY == 'BEST_FOLD' else range(config.DATA.N_SPLIT)

print(f"Loading resources for folds: {list(folds_to_load)}...")

for fold in folds_to_load:
    print(f"\n  > Loading Fold {fold} system...")
    # A. Feature Extractor
    feature_extractors_map[fold] = mygears.load_one_extractor(fold, model_name, config.BASEMODEL_PATH)
    # B. Riemann Tool
    riemann_tools_map[fold] = mygears.load_one_manifold(fold, model_name, config.MODEL_PATH)
    # C. MIL Ensemble
    all_fold_ensembles[fold] = mygears.load_mil_ensemble_for_fold(fold, model_name, config.MODEL_PATH, n_estimators=5)

print(f"[System Ready] Loaded {len(feature_extractors_map)} pipelines.")


#%%
if config.Evaluate_Val:
    from sklearn.model_selection import StratifiedShuffleSplit
    print("\nExtract patients and their labels")
    train_patient_labels = train_df.groupby('patient_id')['label'].first()
    # Desired validation ratio
    sss = StratifiedShuffleSplit(
        n_splits=config.DATA.N_SPLIT, 
        test_size=config.DATA.val_ratio,
        random_state=config.DATA.SEED
    )
    
    train_df_fold_list = []
    val_df_fold_list = []
    for fold, (train_idx, val_idx) in enumerate(sss.split(train_patient_labels.index, train_patient_labels.values)):
        train_fold_patients = train_patient_labels.index[train_idx]
        val_fold_patients   = train_patient_labels.index[val_idx]
    
        train_df_fold = train_df[train_df['patient_id'].isin(train_fold_patients)]
        val_df_fold   = train_df[train_df['patient_id'].isin(val_fold_patients)]
        train_df_fold_list.append(train_df_fold)
        val_df_fold_list.append(val_df_fold)
        
        print(f"\nFold {fold}: Train patients={train_df_fold['patient_id'].nunique()}, "
              f"Val patients={val_df_fold['patient_id'].nunique()}")
        
        # print(f"{train_fold_patients=}")
        # print(f"{val_fold_patients=}")
        
    #%% [Step 2] Inference on Validation Folds (Threshold Calibration with Standardized QC)
    print('\n[Step 2] Inference on Validation Folds (Standardized QC via Fold 4)')  
    
    # --- 1. CONFIGURATION ---
    # Options: 'RESPECTIVE' (Case 1) or 'ENSEMBLE_OTHERS' (Case 2)
    # PREDICTION_STRATEGY = 'RESPECTIVE' 
    PREDICTION_STRATEGY = 'ENSEMBLE_OTHERS' 
    
    # --- Setup Standardized QC Tools (Always Fold 4) ---
    BEST_FOLD_ID = CLINICAL_WORKFLOW_CONFIG["ACTIVE_FOLD"] # Fold 4
    standard_qc_riemann = riemann_tools_map[BEST_FOLD_ID]
    standard_qc_extractor = feature_extractors_map[BEST_FOLD_ID] 
    
    all_val_predictions = [] # Single list for combined logging

    # Loop through each validation fold
    for fold_idx, val_df in enumerate(val_df_fold_list):        
        print(f"\n{'-'*40}")
        print(f"Processing Validation Fold {fold_idx} | Strategy: {PREDICTION_STRATEGY}")
        print(f"Data: {len(val_df)} patches, {val_df['patient_id'].nunique()} patients")
        print(f"{'-'*40}")
        
        # Determine which models to use based on strategy
        if PREDICTION_STRATEGY == 'RESPECTIVE':
            target_model_indices = [fold_idx]
        else: # ENSEMBLE_OTHERS
            target_model_indices = [idx for idx in range(len(loaded_model_list)) if idx != fold_idx]
 
        # Collect predictions from selected models
        print(f"{target_model_indices=}")
    
        # 1.1 Initialize Generator
        model_builder = ModelBuilder(model_name=model_names[0], config=config)
        preprocessing_fn = model_builder.get_preprocessing_function()
        
        # Stratified Sampling logic
        val_df_work = val_df.sample(frac=1, random_state=config.DATA.SEED).reset_index(drop=True)
        val_df_sampled = (
            mygears.stratified_subsample(val_df_work, config['TRAIN']['SAMPLE_SIZE_VAL'])
            if config['TRAIN']['SAMPLE'] else val_df_work
        )   
        
        val_bag_gen = FullBagDataset(
            config=config, df=val_df_sampled,
            preprocessing_function=preprocessing_fn,
            shuffle=False, expect_rgba=True, mode='test'
        )
     
        # 2. Inference Loop
        order = 1
        for i in tqdm(range(len(val_bag_gen)), desc=f"Predicting Fold {fold_idx}"):            
            batch_patients, labels_true, X_batch, _ = val_bag_gen.__getitem__(i, get_pids=True)
            patient_id, class_id, X_patches = batch_patients[0], labels_true[0], X_batch[0]
            n_total = len(X_patches)
            
            print(f"\nPredicting Patient {patient_id} | Class {class_id}")
            
            # ---------------------------------------------------------
            # 2A. Standardized Patch QC (Always Fold 4 Manifold)
            # ---------------------------------------------------------
            Z_for_qc = standard_qc_extractor.predict(X_patches, batch_size=config.TRAIN.BATCH_SIZE, verbose=0)
            E_for_qc = standard_qc_riemann.compute_batch_curvature(Z_for_qc).reshape(-1, 1)
            
            _, _, valid_mask, qc_stats = mygears.apply_riemannian_filter(
                features=Z_for_qc, energies=E_for_qc,
                limit=CLINICAL_WORKFLOW_CONFIG["OOD_ENERGY_LIMIT"],
                rescue_top_k=CLINICAL_WORKFLOW_CONFIG["RESCUE_TOP_K"],
                extreme_limit=CLINICAL_WORKFLOW_CONFIG["EXTREME_CEILING"],
                verbose=True
            )
            
            X_patches_qc = X_patches[valid_mask]
            n_passed = len(X_patches_qc)
        
            # ---------------------------------------------------------
            # 2B. Diagnostic Inference (Strategy 2B: Soft Voting Consensus)
            # ---------------------------------------------------------
            if n_passed > 0:
                # List to store the full probability tensor (N_patches, 2) from each model
                all_model_patch_probs = []
                
                # 1. Collect predictions from all targeted models
                for m_idx in target_model_indices:
                    current_model = loaded_model_list[m_idx]
                    # Shape: (N_passed, 2)
                    y_pred_raw = current_model.predict(X_patches_qc, 
                                                       batch_size=config.TRAIN.BATCH_SIZE, 
                                                       verbose=0)
                    all_model_patch_probs.append(y_pred_raw)
                
                # 2. Compute averaged model probabilities per patch (Soft-Voting)
                # Shape: (N_passed, 2)
                y_patch_probs_avg = np.mean(all_model_patch_probs, axis=0)
    
                # --- Aggregation Strategy A: Mean-Based ---
                # Mean probability of Class 1 across all QC-passed patches
                final_prob_mean = np.mean(y_patch_probs_avg[:, 1])
                
                # --- Strategy 2B: Argmax-Based (Hard Voting on Consensus) ---
                # 1. Determine class for each patch based on averaged model probs
                y_patch_labels = np.argmax(y_patch_probs_avg, axis=1)
                
                # 2. Count votes for Class 1 (THL)
                class_1_count = np.sum(y_patch_labels == 1)
                total_patches = len(y_patch_labels)
                
                # Final patient-level argmax probability
                P_patient_argmax = class_1_count / total_patches
                
                # Map back to your logging variable name
                final_prob_argmax = P_patient_argmax
            
                # Save Combined Results to single list
                all_val_predictions.append({
                    "val_fold": fold_idx,
                    "patient_id": patient_id,
                    "true_class": class_id,
                    "probability_1_mean": final_prob_mean,
                    "probability_1_argmax": final_prob_argmax,
                    "predicted_class": 1 if final_prob_mean >= 0.5 else 0, # Defaulting decision to mean
                    "qc_pass_rate": n_passed / n_total,
                    "qc_rescued": qc_stats['n_rescued'],
                    "strategy": PREDICTION_STRATEGY,
                    "models_used": str(target_model_indices)
                })
                print(f"\n   Patient {patient_id} | Class {class_id} | \
        QC Pass: {n_passed}/{n_total} | \
        Pass Rate: {n_passed / n_total:.4f} | \
        Mean Prob: {final_prob_mean:.4f} | \
        Argmax Prob: {final_prob_argmax:.4f}")
            else:
                print(f"\n   [!] QC REJECT: Patient {patient_id} had 0 valid patches.")
                   
            order += 1
            # for each patient        
    # --- Save Combined Results ---
    print("\nSaving validation results...")
    df_val = pd.DataFrame(all_val_predictions)
    dt = config.TRAIN.DATETIME
    csv_file_name = f"val_predictions_{PREDICTION_STRATEGY}_QC_Std_{dt}.csv"
    df_val.to_csv(csv_file_name, index=False)
    print(f"Results saved to: {csv_file_name}")    
   
    #%%
    from sklearn.metrics import roc_curve 
    def calculate_optimal_thresholds(y_true, y_prob, verbose=True):
        """
        Calculates thresholds based on the paper's methodology (Eq 26-29).
        
        T_upper: Maximizes Youden's Index (Sensitivity + Specificity - 1).
        T_lower: 10th Percentile of probabilities for the Positive Class (THL).
                 (Represents the lower bound of 'typical' THL predictions).
        """
        # 1. T_upper (Youden's Index)
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        youden_index = tpr + (1 - fpr) - 1
        best_idx = np.argmax(youden_index)
        t_upper = thresholds[best_idx]
        
        # 2. T_lower (Percentile of Positives)
        # Filter probabilities where true label is THL (1)
        thl_probs = y_prob[y_true == 1]
        if len(thl_probs) > 0:
            t_lower = np.percentile(thl_probs, 10) # 10th percentile as per Eq 28
        else:
            t_lower = 0.20 # Fallback
            
        # Safety clamp
        t_upper = np.clip(t_upper, 0.5, 0.99)
        t_lower = np.clip(t_lower, 0.01, 0.49)
        
        if verbose:
            print(f"   [Calib] T_upper (Youden): {t_upper:.4f}, T_lower (P10): {t_lower:.4f}")
            
        return t_upper, t_lower

    def compute_hybrid_thresholds(y_true, y_score, min_sens=0.95, min_spec=0.95):
        """
        Use this method
        Computes three thresholds:
        1. T_lower: Ensures Sensitivity >= min_sens (Safe Negative boundary)
        2. T_optimal: Youden's Index (Balanced boundary)
        3. T_upper: Ensures Specificity >= min_spec (Safe Positive boundary)
        """
        if len(np.unique(y_true)) < 2:
            return 0.5, 0.5, 0.5 # Fallback if fold has only one class
        
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        
        # --- 1. Optimal (Youden) ---
        youden_index = tpr - fpr
        idx_opt = np.argmax(youden_index)
        t_opt = thresholds[idx_opt]
        
        # --- 2. T_lower (High Sensitivity / Confident Negative) ---
        # We want Sensitivity (TPR) >= min_sens.
        # Thresholds are descending. We find the first index (highest score) where TPR >= min_sens.
        # This keeps the "Negative" region as small as safely possible.
        valid_sens_indices = np.where(tpr >= min_sens)[0]
        if len(valid_sens_indices) > 0:
            # Use the highest threshold that meets sensitivity requirement
            t_lower = thresholds[valid_sens_indices[0]]
        else:
            t_lower = t_opt # Fallback
            
        # --- 3. T_upper (High Specificity / Confident Positive) ---
        # We want Specificity (1-FPR) >= min_spec  => FPR <= (1 - min_spec).
        # We find the last index (lowest score) where FPR <= limit.
        # This keeps the "Positive" region as large as safely possible.
        valid_spec_indices = np.where(fpr <= (1 - min_spec))[0]
        if len(valid_spec_indices) > 0:
            # Use the lowest threshold that meets specificity requirement
            t_upper = thresholds[valid_spec_indices[-1]]
        else:
            t_upper = t_opt # Fallback
            
        return t_lower, t_opt, t_upper
    
    #%% --------------------------------------------------------- 
    # Load Data
    # ---------------------------------------------------------
    # Ensure you point to the correct file generated in the previous step
    try:
        csv_file_name = 'val_predictions_RESPECTIVE_QC_Std_20251221-1222.csv'
        df_preds = pd.read_csv(csv_file_name)
    except FileNotFoundError:
        print(f"File not found: {csv_file_name}. Please run Validation Inference first.")
        # Dummy data for demonstration if file is missing
        df_preds = pd.DataFrame() 
    
    true_col = 'true_class' 
    prob_col = 'probability_1_mean'
    # prob_col = 'probability_1_argmax'
    
    print(f"Loaded {len(df_preds)} rows.")
    print(f"Using columns: '{true_col}' (Truth) and '{prob_col}' (Score)")
    
    # ---------------------------------------------------------
    # Calculate Thresholds Per Fold
    # ---------------------------------------------------------
    fold_t_lowers = []
    fold_t_opts = []
    fold_t_uppers = []
    
    # Constraints for Hybrid Logic
    REQ_SENS = 0.95  # 95% Sensitivity required for T_lower
    REQ_SPEC = 0.95  # 95% Specificity required for T_upper
    
    folds = sorted(df_preds["val_fold"].unique())
    
    print("\n" + "="*40)
    print(f"CALIBRATION RESULTS (Sens>={REQ_SENS}, Spec>={REQ_SPEC})")
    print("="*40)
    print(f"{'Fold':<6} | {'T_lower':<10} | {'T_opt':<10} | {'T_upper':<10} | {'Status'}")
    print("-" * 40)
    
    for fold in folds:
        fold_df = df_preds[df_preds["val_fold"] == fold]
        
        y_true = fold_df[true_col].values.astype(int)
        y_score = fold_df[prob_col].values.astype(float)
        
        t_low, t_opt, t_up = compute_hybrid_thresholds(y_true, y_score, REQ_SENS, REQ_SPEC)
        
        fold_t_lowers.append(t_low)
        fold_t_opts.append(t_opt)
        fold_t_uppers.append(t_up)
        
        status = "OK" if t_low < t_up else "Overlap (Tight)"
        print(f"{fold:<6} | {t_low:.4f} | {t_opt:.4f} | {t_up:.4f} | {status}")
    
    # ---------------------------------------------------------
    # Final Output for Config
    # ---------------------------------------------------------
    # Use Median to be robust against outliers
    final_t_lower = np.median(fold_t_lowers)
    final_t_upper = np.median(fold_t_uppers)
    final_t_opt   = np.median(fold_t_opts)
    
    print("\n" + "="*40)
    print("="*40)
    print(f"fold_t_lowers = {np.round(fold_t_lowers, 4).tolist()}")
    print(f"fold_t_uppers = {np.round(fold_t_uppers, 4).tolist()}")
    print(f"fold_t_opts   = {np.round(fold_t_opts, 4).tolist()}")
    print("-" * 20)
    print(f"final_t_lower = {final_t_lower:.4f}  (Confident IDA < this)")
    print(f"final_t_opt   = {final_t_opt:.4f}    (Balanced Cutoff)")
    print(f"final_t_upper = {final_t_upper:.4f}  (Confident THL >= this)")
    print(f"Uncertainty Region: {final_t_lower:.4f} - {final_t_upper:.4f}")
    print("="*40)
  
"""
'mil_model_3'
========================================
CALIBRATION RESULTS (Sens>=0.95, Spec>=0.95)
fold_t_lowers = [0.5335, 0.5236, 0.4735, 0.3266, 0.4914]
fold_t_uppers = [0.6505, 1.535, 0.6068, 0.3346, 1.4976]
fold_t_opts   = [0.6505, 0.5341, 0.5742, 0.3308, 1.4976]
--------------------
final_t_lower = 0.4914  (Confident IDA < this)
final_t_opt   = 0.5742    (Balanced Cutoff)
final_t_upper = 0.6505  (Confident THL >= this)
Uncertainty Region: 0.4914 - 0.6505
"""

""" Respective fold mode
fold_t_lowers = [0.5175, 0.4569, 0.2851, 0.3197, 0.5805]
fold_t_uppers = [0.6713, 1.4868, 0.6088, 0.3553, 0.5852]
fold_t_opts   = [0.5175, 0.4725, 0.6088, 0.3371, 0.5818]
--------------------
final_t_lower = 0.4569  (Confident IDA < this)
final_t_opt   = 0.5175    (Balanced Cutoff)
final_t_upper = 0.6088  (Confident THL >= this)
Uncertainty Region: 0.4569 - 0.6088

"""
#%%
if config.Performance_Report:       
    
    # Determine which models to use based on strategy
    # PREDICTION_STRATEGY = 'RESPECTIVE'
    PREDICTION_STRATEGY = 'ENSEMBLE_OTHERS'
    if PREDICTION_STRATEGY == 'RESPECTIVE':
        # Respective model
        T_LOWER = 0.4569  #(Confident IDA < this)
        T_UPPER = 0.6088  #(Confident THL >= this)
        T_OPT = 0.5175    #(Balanced Cutoff)
        audit_path = 'val_predictions_RESPECTIVE_QC_Std_20251221-1222.csv'
    else: # ENSEMBLE_OTHERS
    # [Config] Applied Calibrated Hybrid Thresholds:
    #    T_upper (Confident THL): >= 0.6070
    #    T_lower (Confident IDA): <  0.4900
    #    Uncertain Region:       0.4900 - 0.6070
       
        T_LOWER = 0.5130  #(Confident IDA < this)
        T_UPPER = 0.5729  #(Confident THL >= this)
        T_OPT = 0.5538    #(Balanced Cutoff)
        
        audit_path = 'val_predictions_ENSEMBLE_OTHERS_QC_Std_20251221-1341.csv'
        # audit_path = 'Test_Ensemble/test_predictions_final_ENSEMBLE_OTHERS_20251221-2255.csv'
    df = pd.read_csv(audit_path)
    print(df.columns)
    
    save_dir='Val'
    os.makedirs(save_dir, exist_ok=True)

    # Assuming T_LOWER, T_UPPER, T_OPT are already defined
    triage_data = []
    
    for idx, row in df.iterrows():
        result = mygears.apply_clinical_triage(
            p_patient_mean=row['probability_1_mean'],
            p_patient_argmax=row['probability_1_argmax'],
            t_lower=T_LOWER,
            t_upper=T_UPPER,
            t_opt=T_OPT,
            # fallback_mode='mean',
            fallback_mode='argmax'
        )
        triage_data.append(result)
        
    # Convert results to DataFrame and join
    df_results = pd.DataFrame(triage_data)
    df = pd.concat([df, df_results], axis=1)
    # 5. Deep Analysis of the Triage Output
    n_total = len(df)
    n_confident = len(df[df['pred_status'] == "Confident"])
    n_uncertain = len(df[df['pred_status'] == "Uncertain"])
    
    print(f"{'='*40}")
    print(f"[Clinical Report] Triage Strategy: Hybrid")
    print(f"{'='*40}")
    print(f"Total Patients:  {n_total}")
    print(f"Confident Class: {n_confident} ({n_confident/n_total:.1%})")
    print(f"Uncertain (Grey Zone): {n_uncertain} ({n_uncertain/n_total:.1%})")
    print(f"{'='*40}")
   
    #% full
    y_true = df['true_class']
    y_pred = df['final_pred']
    print(f"y_true: {y_true.shape}")
    print(f"y_pred: {y_pred.shape}")
    
    # Extract probabilities only for the Confident subset 
    y_true_confident = df.loc[df['pred_status'] == 'Confident', 'true_class'].to_numpy()
    y_pred_confident = df.loc[df['pred_status'] == 'Confident', 'final_pred'].to_numpy()
    print(f"y_true_confident: {y_true_confident.shape}")
    print(f"y_pred_confident: {y_pred_confident.shape}")
   
    print('\nVal set classification_report')
    from sklearn.metrics import classification_report
    print(classification_report(
        y_true = y_true,
        y_pred = y_pred,
        target_names=config['DATA']['CLASS_LABELS'])
        )
    
    p_measures, cm = mygears.evaluate_classification_performance(
       y_true,
       y_pred,
       class_labels=config['DATA']['CLASS_LABELS'],
       save_dir=save_dir,
       save_prefix=str(f'Val-{PREDICTION_STRATEGY}'),
       )
    
    mygears.plot_confusion_matrices(
                confusionmatrix=cm,
                p_measures=p_measures,
                class_labels=config['DATA']['CLASS_LABELS'],
                save_dir=save_dir,
                save_prefix=str(f'Val-{PREDICTION_STRATEGY}')
            ) 
#%% [Step 4] Inference on Unseen Test Set (Riemannian Ablation: M4, M5, M6)

if config.Inference:
    if config.TRAIN.SAMPLE:
        test_df_sampled = mygears.stratified_subsample(
            test_df, 
            config.TRAIN.SAMPLE_SIZE_TEST,
            random_state=config.DATA.SEED
        )
    else:
        test_df_sampled = test_df

    print(f"Test DataFrame size: {len(test_df_sampled)}")
    
    # --- 1. GENERAL CONFIGURATION ---
    save_dir = 'Ablation_2'
    os.makedirs(save_dir, exist_ok=True)
    
    PREDICTION_STRATEGY = 'ENSEMBLE_OTHERS' 
    #T_LOWER, T_UPPER, T_OPT = 0.5130, 0.5729, 0.5538 #20260901
    target_model_indices = list(range(len(loaded_model_list))) # Use all 5 folds
    
    # --- Setup Standardized QC Tools (The "Supervisor") ---
    BEST_FOLD_ID = 4
    standard_qc_riemann = riemann_tools_map[BEST_FOLD_ID]
    standard_qc_extractor = feature_extractors_map[BEST_FOLD_ID] 
    
    # Initialize Generator
    model_builder = ModelBuilder(model_name=model_names[0], config=config)
    preprocessing_fn = model_builder.get_preprocessing_function()
    
    inference_gen = FullBagDataset(
        config=config, df=test_df_sampled,
        preprocessing_function=preprocessing_fn,
        shuffle=False, expect_rgba=True, mode='test'
    )

    # --- 2. DEFINE ABLATION MODES ---
    ablation_modes = [
        # {"name": "M4_BASELINE", "use_qc": False, "use_triage": False},
        # {"name": "M5_QC_FULL",  "use_qc": True,  "use_triage": False},
        {"name": "M6_QC_TRIAGE","use_qc": True,  "use_triage": True},
    ]

    # --- 3. MASTER INFERENCE LOOP ---
    for mode in ablation_modes:
        EVAL_MODE = mode["name"]
        use_qc = mode["use_qc"]
        use_triage = mode["use_triage"]
        
        t1 = datetime.now()
        print(f"\n{'='*50}")
        print(f"STARTING ABLATION RUN: {EVAL_MODE}")
        print(f"QC Gate: {use_qc} | Triage Abstention: {use_triage}")
        print(f"{'='*50}")
        
        test_results = []
        
        for i in tqdm(range(len(inference_gen)), desc=f"Inferencing {EVAL_MODE}"):
            batch_patients, labels_true, X_batch, _ = inference_gen.__getitem__(i, get_pids=True)
            patient_id, class_id, X_patches = batch_patients[0], labels_true[0], X_batch[0]
            n_total = len(X_patches)
            
            # ---------------------------------------------------------
            # STEP A: Patch Quality Control (Supervisor Gate)
            # ---------------------------------------------------------
            if use_qc:
                Z_for_qc = standard_qc_extractor.predict(X_patches, batch_size=config.TRAIN.BATCH_SIZE, verbose=0)
                E_for_qc = standard_qc_riemann.compute_batch_curvature(Z_for_qc).reshape(-1, 1)
                
                _, _, valid_mask, qc_stats = mygears.apply_riemannian_filter(
                    features=Z_for_qc,
                    energies=E_for_qc,
                    limit=CLINICAL_WORKFLOW_CONFIG["OOD_ENERGY_LIMIT"],
                    rescue_top_k=CLINICAL_WORKFLOW_CONFIG["RESCUE_TOP_K"],
                    extreme_limit=CLINICAL_WORKFLOW_CONFIG["EXTREME_CEILING"],
                    # verbose=False, # Keep logs clean during loop
                    verbose=True
                )
                X_patches_qc = X_patches[valid_mask]
                n_rescued = qc_stats['n_rescued']
            else:
                X_patches_qc = X_patches
                n_rescued = 0
                
            n_passed = len(X_patches_qc)
            
            # ---------------------------------------------------------
            # STEP B: Diagnostic Inference & Triage
            # ---------------------------------------------------------
            if n_passed > 0:
                # 1. Collect predictions from all targeted models
                all_model_patch_probs = []
                for m_idx in target_model_indices:
                    current_model = loaded_model_list[m_idx]
                    y_pred_raw = current_model.predict(X_patches_qc, batch_size=config.TRAIN.BATCH_SIZE, verbose=0)
                    all_model_patch_probs.append(y_pred_raw)
                
                # 2. Compute probabilities
                y_patch_probs_avg = np.mean(all_model_patch_probs, axis=0)
                P_patient_mean = np.mean(y_patch_probs_avg[:, 1])
                y_patch_labels = np.argmax(y_patch_probs_avg, axis=1)
                P_patient_argmax = np.sum(y_patch_labels == 1) / len(y_patch_labels)
                
                # 3. Apply Clinical Triage Logic
                if use_triage:
                    fallback_mode = 'mean' #20250901,
                    fallback_mode = 'mean' #20250907,
                    # fallback_mode = 'argmax' #20250905
                    res = mygears.apply_clinical_triage(
                        p_patient_mean=P_patient_mean, 
                        p_patient_argmax=P_patient_argmax,
                        t_lower=config.T_LOWER,
                        t_upper=config.T_UPPER,
                        t_opt=config.T_OPT,
                        fallback_mode=fallback_mode 
                    )
                    final_pred = res['final_pred']
                    pred_status = res['pred_status']
                else:
                    # triage_off: Force hard prediction based on T_OPT
                    fallback_mode = 'triage_off'
                    final_pred = 1 if P_patient_mean >= config.T_OPT else 0
                    pred_status = "Dummy_Confident" # Dammy confident to ensure it evaluates
                    
            else:
                # SAFETY NET: Total rejection by QC
                P_patient_mean = np.nan
                P_patient_argmax = np.nan
                final_pred = -1 # Flag for "Rejected"
                pred_status = "Abstain_NoValidPatches"

            # 4. Store Results
            test_results.append({
                "patient_id": patient_id,
                "true_class": class_id,
                "probability_1_mean": P_patient_mean,
                "probability_1_argmax": P_patient_argmax,
                "final_pred": final_pred,
                "pred_status": pred_status,
                "eval_mode": EVAL_MODE,
                "fallback_mode": fallback_mode,
                "patches_total": n_total,
                "patches_passed": n_passed,
                "qc_rescued": n_rescued,
                "energies": qc_stats['energies']
            })

        # --- 4. Save CSV Output ---
        test_df_final = pd.DataFrame(test_results)
        dt = config.TRAIN.DATETIME
        final_csv = os.path.join(save_dir, f"test_ablation_{EVAL_MODE}_{dt}.csv")
        test_df_final.to_csv(final_csv, index=False)
        print(f"\nSaved Ablation Results to: {final_csv}")
        print(f"Time taken for {EVAL_MODE}: {datetime.now() - t1}")
        
    #%% R2-6
    """
    from sklearn.metrics import brier_score_loss
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    from scipy.stats import linregress
    
    def expected_calibration_error(y_true, y_prob, n_bins=15):
        # Calculates the Expected Calibration Error (ECE)
        bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
        binids = np.digitize(y_prob, bins) - 1
        
        ece = 0.0
        for i in range(n_bins):
            bin_mask = (binids == i)
            if np.sum(bin_mask) > 0:
                bin_acc = np.mean(y_true[bin_mask])
                bin_conf = np.mean(y_prob[bin_mask])
                bin_weight = np.sum(bin_mask) / len(y_prob)
                ece += bin_weight * np.abs(bin_acc - bin_conf)
        return ece
    
    def calibration_slope_intercept(y_true, y_prob):
        # Calculates calibration slope and intercept using log-odds.
        # Clip to prevent log(0)
        eps = 1e-15
        y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
        
        # Convert probabilities to log-odds (logit)
        logit_prob = np.log(y_prob_clipped / (1 - y_prob_clipped))
        
        # Fit logistic regression: P(Y=1) ~ logit(prob)
        # The slope and intercept of this fit are the calibration metrics
        lr = LogisticRegression(penalty=None, solver='lbfgs')
        lr.fit(logit_prob.reshape(-1, 1), y_true)
        
        slope = lr.coef_[0][0]
        intercept = lr.intercept_[0]
        
        return slope, intercept
    
    def bootstrap_calibration_metrics(y_true, y_prob, n_bootstraps=2000):
        #Runs 2000 bootstraps to get 95% CIs for calibration metrics.
        brier_scores, ece_scores, slopes, intercepts = [], [], [], []
        n = len(y_true)
        rng = np.random.RandomState(42)
        
        for _ in range(n_bootstraps):
            idx = rng.randint(0, n, n)
            y_t_boot, y_p_boot = y_true[idx], y_prob[idx]
            
            # Skip samples that only have 1 class
            if len(np.unique(y_t_boot)) < 2:
                continue
                
            brier_scores.append(brier_score_loss(y_t_boot, y_p_boot))
            ece_scores.append(expected_calibration_error(y_t_boot, y_p_boot))
            
            s, i = calibration_slope_intercept(y_t_boot, y_p_boot)
            slopes.append(s)
            intercepts.append(i)
            
        def get_ci(scores):
            scores = np.sort(scores)
            mean_val = np.mean(scores)
            lower = scores[int(0.025 * len(scores))]
            upper = scores[int(0.975 * len(scores))]
            return f"{mean_val:.3f} [{lower:.3f}--{upper:.3f}]"
    
        return {
            "Brier": get_ci(brier_scores),
            "ECE (15-bin)": get_ci(ece_scores),
            "Calib. Slope": get_ci(slopes),
            "Calib. Intercept": get_ci(intercepts)
        }
    
    # ==========================================
    # 1. Load Data
    # ==========================================
    # Assuming M6 CSV is in the Ablation folder
    dt = config.TRAIN.DATETIME
    df = pd.read_csv('Ablation/test_ablation_M6_QC_TRIAGE_20260901-1109.csv')
    
    # STRICTLY filter to the Confident subset (Autonomous Triage Cohort) as evaluated in the manuscript
    df_conf = df[df['pred_status'] == 'Confident']
    
    y_true = df_conf['true_class'].to_numpy()
    y_prob_uncal = df_conf['probability_1_mean'].to_numpy()
    
    print(f"Evaluating Calibration on {len(y_true)} confident patients.")
    
    # ==========================================
    # 2. Fit Calibrators
    # ==========================================
    # Platt Scaling (Logistic Regression)
    platt = LogisticRegression(penalty=None, solver='lbfgs')
    platt.fit(y_prob_uncal.reshape(-1, 1), y_true)
    y_prob_platt = platt.predict_proba(y_prob_uncal.reshape(-1, 1))[:, 1]
    
    # Isotonic Regression
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(y_prob_uncal, y_true)
    y_prob_iso = iso.predict(y_prob_uncal)
    
    # ==========================================
    # 3. Bootstrap Metrics
    # ==========================================
    print("\nBootstrapping Uncalibrated...")
    metrics_uncal = bootstrap_calibration_metrics(y_true, y_prob_uncal)
    
    print("Bootstrapping Platt...")
    metrics_platt = bootstrap_calibration_metrics(y_true, y_prob_platt)
    
    print("Bootstrapping Isotonic...")
    metrics_iso = bootstrap_calibration_metrics(y_true, y_prob_iso)
    
    # ==========================================
    # 4. Construct Table
    # ==========================================
    results = [
        {"Model": "Uncalibrated Baseline"} | metrics_uncal,
        {"Model": "Platt Scaling (Logistic)"} | metrics_platt,
        {"Model": "Isotonic Regression"} | metrics_iso
    ]
    
    df_results = pd.DataFrame(results)
    print("\n=== Calibration Metrics Table ===")
    print(df_results.to_string(index=False))
    
    df_results.to_csv(f'Ablation/{dt}_calibration_metrics_table.csv', index=False)
    """    
#%%
if config.Ablation_Study:
    def compute_clinical_metrics(df_test, confident_only=False):
    
        if confident_only:
            df_eval = df_test[df_test['pred_status'] == 'Confident']
        else:
            df_eval = df_test.copy()
        
        y_true = df_eval['true_class'].values
        y_pred = df_eval['final_pred'].values
        
        TP = np.sum((y_true == 1) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        
        eps = 1e-12
        
        Accuracy = (TP + TN) / (TP + TN + FP + FN + eps)
        Sensitivity = TP / (TP + FN + eps)
        Specificity = TN / (TN + FP + eps)
        PPV = TP / (TP + FP + eps)
        
        FPR = FP / (FP + TN + eps)
        FNR = FN / (FN + TP + eps)
        
        LR_pos = float('inf') if FP == 0 else Sensitivity / (FPR + eps)
        LR_neg = FNR / (Specificity + eps)
        
        coverage = len(df_eval) / len(df_test)
        
        return {
            "Accuracy": Accuracy,
            "Sensitivity": Sensitivity,
            "Specificity": Specificity,
            "PPV": PPV,
            "LR+": LR_pos,
            "LR-": LR_neg,
            "Coverage": coverage
        }

    #%   
    import math
    from statsmodels.stats.proportion import proportion_confint
    from sklearn.metrics import confusion_matrix, roc_auc_score
    
    def compute_detailed_clinical_metrics(y_true, y_pred,
                        y_prob=None, total_n=None,
                        name="Subset", save_path=None, n_bootstraps=2000):
        """
        Computes ACC, TPR, TNR, PPV, NPV with 95% Wilson CIs, Likelihood Ratios with 95% CIs, 
        Coverage, and AUC with 95% Bootstrap CIs. Saves results to JSON.
        All metrics and CIs are formatted to 2 precision points.
        
        Args:
            y_true (array-like): Ground truth labels (0 or 1).
            y_pred (array-like): Predicted hard labels (0 or 1).
            y_prob (array-like, optional): Predicted probabilities for class 1 (for AUC).
            total_n (int, optional): The total number of patients before abstention (for Coverage).
            name (str): Identifier for the evaluated group.
            save_path (str, optional): Path to save the output JSON.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        n = len(y_true)
        coverage = (n / total_n) if total_n else 1.0
        
        if n == 0:
            return {"Group": name, "Error": "No samples evaluated"}
            
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        # Base Proportions
        metrics = {
            "ACC": (tp + tn, n),
            "Sensitivity": (tp, tp + fn),
            "Specificity": (tn, tn + fp),
            "PPV": (tp, tp + fp),
            "NPV": (tn, tn + fn),
        }
        
        results = {
            "Group": name,
            "N": int(n),
            "Coverage": f"{coverage:.1%}"
        }
        
        numeric_results = {}
        
        # Calculate values and Wilson Score Intervals (forced to 2 precision points)
        for metric, (count, total) in metrics.items():
            val = count / total if total > 0 else 0
            low, high = proportion_confint(count, total, method='wilson') if total > 0 else (0, 0)
            results[metric] = f"{val:.2f} [{low:.2f} - {high:.2f}]"
            numeric_results[metric] = val
            
        # Likelihood Ratios and 95% CIs (Simel et al., 1991)
        sens = numeric_results["Sensitivity"]
        spec = numeric_results["Specificity"]
        
        # Haldane-Anscombe correction: Add 0.5 to all cells for variance calc if any cell is 0
        if tp == 0 or fp == 0 or tn == 0 or fn == 0:
            tp_c, fp_c, tn_c, fn_c = tp + 0.5, fp + 0.5, tn + 0.5, fn + 0.5
        else:
            tp_c, fp_c, tn_c, fn_c = tp, fp, tn, fn
    
        # LR+ CI calculation
        if fp == 0:
            results["LR+"] = "undefined (0/N FP)"
        else:
            lr_plus = sens / (1 - spec) if (1 - spec) > 0 else float('inf')
            # Standard error of ln(LR+)
            se_ln_lr_plus = math.sqrt(max(0, 1/tp_c - 1/(tp_c + fn_c) + 1/fp_c - 1/(fp_c + tn_c)))
            # Use corrected LR+ to center the CI if correction was applied
            lr_plus_c = (tp_c / (tp_c + fn_c)) / (fp_c / (fp_c + tn_c))
            
            ci_low_lr_plus = lr_plus_c * math.exp(-1.96 * se_ln_lr_plus)
            ci_high_lr_plus = lr_plus_c * math.exp(1.96 * se_ln_lr_plus)
            results["LR+"] = f"{lr_plus:.2f} [{ci_low_lr_plus:.2f} - {ci_high_lr_plus:.2f}]"
            
        # LR- CI calculation
        if spec == 0:
            results["LR-"] = "undefined (Spec=0)"
        else:
            lr_minus = (1 - sens) / spec
            # Standard error of ln(LR-)
            se_ln_lr_minus = math.sqrt(max(0, 1/fn_c - 1/(tp_c + fn_c) + 1/tn_c - 1/(fp_c + tn_c)))
            # Use corrected LR- to center the CI if correction was applied
            lr_minus_c = (fn_c / (tp_c + fn_c)) / (tn_c / (fp_c + tn_c))
            
            ci_low_lr_minus = lr_minus_c * math.exp(-1.96 * se_ln_lr_minus)
            ci_high_lr_minus = lr_minus_c * math.exp(1.96 * se_ln_lr_minus)
            results["LR-"] = f"{lr_minus:.2f} [{ci_low_lr_minus:.2f} - {ci_high_lr_minus:.2f}]"
            
        # AUC & 95% Bootstrap Confidence Interval (forced to 2 precision points)
        if y_prob is not None and len(np.unique(y_true)) > 1:
            y_prob = np.array(y_prob)
            auc_val = roc_auc_score(y_true, y_prob)
            
            rng = np.random.RandomState(42) # Seed for reproducibility
            bootstrapped_scores = []
            
            for _ in range(n_bootstraps):
                indices = rng.randint(0, n, n)
                if len(np.unique(y_true[indices])) < 2:
                    continue # Skip samples that lack both classes
                score = roc_auc_score(y_true[indices], y_prob[indices])
                bootstrapped_scores.append(score)
                
            sorted_scores = np.sort(bootstrapped_scores)
            ci_lower = sorted_scores[int(0.025 * len(sorted_scores))]
            ci_upper = sorted_scores[int(0.975 * len(sorted_scores))]
            
            results["AUC"] = f"{auc_val:.2f} [{ci_lower:.2f} - {ci_upper:.2f}]"
        else:
            results["AUC"] = "N/A"
            
        # Save to JSON
        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"Metrics saved to: {save_path}")
                
        return results

    #%
    
    from statsmodels.stats.contingency_tables import mcnemar
    
    def run_mcnemar(y_true, y_pred_1, y_pred_2):
        correct_1 = (y_pred_1 == y_true)
        correct_2 = (y_pred_2 == y_true)
    
        n01 = np.sum((correct_1 == True) & (correct_2 == False))
        n10 = np.sum((correct_1 == False) & (correct_2 == True))
    
        table = [[0, n01],
                 [n10, 0]]
    
        result = mcnemar(table, exact=True)
        return result.pvalue


    #%% Filter for Confident Subset
    # df_conf = test_df_final[test_df_final['pred_status'] == 'Confident']
    # save_dir = 'Test_Ensemble'  
    # T_LOWER, T_UPPER, T_OPT = 0.5130, 0.5729, 0.5538
    # csv_file_name_baseline = 'Ablation/test_ablation_BASELINE_20260219-0102.csv'
    # csv_file_name_qcfull = 'Ablation/test_ablation_QC_FULL_20260219-0152.csv'
    #csv_file_name_qctri = 'Ablation/test_ablation_QC_TRIAGE_20260219-0250.csv'
    
    csv_file_name_baseline = 'Ablation/test_ablation_M4_BASELINE_20260901-1109.csv'
    csv_file_name_qcfull = 'Ablation/test_ablation_M5_QC_FULL_20260901-1109.csv'
    csv_file_name_qctri = 'Ablation/test_ablation_M6_QC_TRIAGE_20260901-1109.csv'
    save_dir = 'Ablation'
    
    df_baseline = pd.read_csv(csv_file_name_baseline)
    df_qcfull = pd.read_csv(csv_file_name_qcfull)
    df_qctri = pd.read_csv(csv_file_name_qctri)  
    dt = config.TRAIN.DATETIME
    
    df_m4 = df_baseline
    df_m5 = df_qcfull
    df_m6 = df_qctri[df_qctri['pred_status'] == 'Confident']
    
    #%% Compute metrics (Total N=48 used to establish the cohort coverage size)
    
    report_m4 = compute_detailed_clinical_metrics(
        y_true=df_baseline['true_class'], 
        y_pred=df_baseline['final_pred'], 
        y_prob=df_baseline['probability_1_mean'], 
        total_n=48, 
        name="M4", 
        save_path=os.path.join(save_dir, f"{dt}_M4.json"),
        n_bootstraps=2000
    )

    for k, v in report_m4.items():
        print(f"{k}: {v}")
    
    #%%
    # 1. Handle edge-case: If QC rejected all patches, force the prediction to be wrong.
    # We do this by flipping the true class (e.g., if true is 1, pred becomes 0)
    # This ensures N remains exactly 48 and Coverage remains 100%.
    df_qcfull['final_pred_safe'] = np.where(
        df_qcfull['final_pred'] == -1, 
        1 - df_qcfull['true_class'], # Forces a False Positive or False Negative
        df_qcfull['final_pred']
    )
    
    # 2. Compute Metrics for M5
    report_m5 = compute_detailed_clinical_metrics(
        y_true=df_qcfull['true_class'], 
        y_pred=df_qcfull['final_pred_safe'], # Use the safe predictions
        y_prob=df_qcfull['probability_1_mean'], 
        total_n=48, 
        name="M5", 
        save_path=os.path.join(save_dir, f"{dt}_M5.json")
    )
    
    for k, v in report_m5.items():
        print(f"{k}: {v}")
    
    #%%
    # csv_file_name_qctri = 'Ablation_2/test_ablation_M6_QC_TRIAGE_20260905-2216.csv'
    # df_qctri = pd.read_csv(csv_file_name_qctri)
    # 1. Isolate the M6 Autonomous Cohort (Filter out 'Uncertain' cases)
    # df_m6 = df_qctri[df_qctri['pred_status'] == 'Confident']

    # 2. Compute metrics for the M6 subset
    report_m6 = compute_detailed_clinical_metrics(
        y_true=df_m6['true_class']  , 
        y_pred=df_m6['final_pred'], 
        y_prob=df_m6['probability_1_mean'], 
        total_n=48, # Retain 48 to accurately compute the % Coverage
        name="M6", 
        save_path=os.path.join(save_dir, f"{dt}_M6.json")
    )
    
    # Optional: Print the results to verify
    for k, v in report_m6.items():
        print(f"{k}: {v}")
    
    #%% patient_level_rejection

    # 1. Load the ablation results
    df = pd.read_csv('test_ablation_M6_QC_TRIAGE_20260901-1109.csv')
    
    # 2. Calculate Patient-Level Rejection Rate
    df['Rejection_Rate_Pct'] = 100 * (1 - df['patches_passed'] / df['patches_total'])
    
    # 3. Filter for THL patients and sort by highest rejection rate
    thl_patients = df[df['true_class'] == 1].copy()
    top_rejected_thl = thl_patients.sort_values(by='Rejection_Rate_Pct', ascending=False).head(5)
    
    # 4. Select columns to demonstrate "patient-level consequences"
    columns_to_show = [
        'patient_id', 'patches_total', 'patches_passed', 'qc_rescued', 
        'Rejection_Rate_Pct', 'probability_1_mean', 'final_pred', 'pred_status'
    ]
    
    print("Top 5 THL Patients with the Highest Rejection Rates:")
    print(top_rejected_thl[columns_to_show].to_string(index=False))
    
    #%%
    # qc_triage = compute_detailed_clinical_metrics(
    #     df_qctri['true_class'], 
    #     df_qctri['final_pred'], 
    #     name="QC Triage"
    # )
    
    # y_true_confident = df_qctri.loc[df_qctri['pred_status'] == 'Confident', 'true_class'].to_numpy()
    # y_pred_confident = df_qctri.loc[df_qctri['pred_status'] == 'Confident', 'final_pred'].to_numpy()
    # qc_triage_conf = compute_detailed_clinical_metrics(
    #     y_true_confident, 
    #     y_pred_confident, 
    #     name="QC Triage Confident Subset"
    # )
    
    # df_ens = pd.read_csv('Test_Ensemble/test_predictions_final_ENSEMBLE_OTHERS_20251221-2255.csv')
    
    # qc_full = compute_detailed_clinical_metrics(
    #     df_ens['true_class'], 
    #     df_ens['final_pred'],  
    #     name="QC Ens Full"
    # )
    
    # qc_tri = qc_full
    
    # y_true_confident = df_ens.loc[df_ens['pred_status'] == 'Confident', 'true_class'].to_numpy()
    # y_pred_confident = df_ens.loc[df_ens['pred_status'] == 'Confident', 'final_pred'].to_numpy()
    # qc_conf = compute_detailed_clinical_metrics(
    #     y_true_confident, 
    #     y_pred_confident, 
    #     name="QC Ens Triage Confident Subset"     
    # )
    
    #%%
    
    df_qcfull = df_m6.copy()

    # 1. Calculate Patient-Level Metrics
    df_qcfull['survival_rate'] = df_qcfull['patches_passed'] / df_qcfull['patches_total']
    df_qcfull['rejection_rate'] = 1.0 - df_qcfull['survival_rate']
    
    # 2. Overall Cohort Statistics
    total_patches = df_qcfull['patches_total'].sum()
    total_passed = df_qcfull['patches_passed'].sum()
    overall_survival = total_passed / total_patches
    overall_rejection = 1.0 - overall_survival
    
    print(f"=== OVERALL QC GATE PERFORMANCE ===")
    print(f"Total Patches: {total_patches}")
    print(f"Passed Patches: {total_passed}")
    print(f"Overall Rejection Rate: {overall_rejection:.2%}\n")
    
    # 3. Class-Level Bias Analysis (Crucial for explaining Sensitivity drops)
    class_stats = df_qcfull.groupby('true_class')[['patches_total', 'patches_passed']].sum()
    class_stats['survival_rate'] = class_stats['patches_passed'] / class_stats['patches_total']
    class_stats['rejection_rate'] = 1.0 - class_stats['survival_rate']
    
    print("=== REJECTION RATE BY CLASS ===")
    print(class_stats[['patches_total', 'rejection_rate']].to_string(formatters={'rejection_rate': '{:.2%}'.format}))
    
    # 4. Identify the Most "Rejected" Patients (Top 5)
    print("\n=== TOP 5 PATIENTS WITH HIGHEST REJECTION RATES ===")
    top_rejected = df_qcfull.sort_values(by='rejection_rate', ascending=False).head(5)
    print(top_rejected[['patient_id', 'true_class', 'patches_total', 'rejection_rate']].to_string(formatters={'rejection_rate': '{:.2%}'.format}))
    
    #%%
    # y_true = df_baseline['true_class'].values
    # y_pred_1 = df_baseline['final_pred'].values
    # y_pred_2 = df_qcfull['final_pred'].values
    
    # pvalue = run_mcnemar(y_true, y_pred_1, y_pred_2)
    # print(pvalue)
    
#%% \label{tab:qc_rejected_rate}   
from scipy.stats import wilcoxon

def evaluate_ablation_pair(df_baseline, df_experimental, subset_mode="Full"):
    """
    Strictly aligns two dataframes row-by-row to compute paired statistical tests.
    
    Args:
        df_baseline: Dataframe for the baseline model (e.g., M4).
        df_experimental: Dataframe for the experimental model (e.g., M5 or M6).
        subset_mode: "Full" for N=48, or "Confident" to strictly filter both models 
                     to only evaluate the patients that survived the M6 triage gate.
    """
    # 1. Isolate the evaluation cohort based on the experimental model's output
    if subset_mode == "Confident":
        df_exp_filtered = df_experimental[df_experimental['pred_status'] == 'Confident'].copy()
        confident_patients = df_exp_filtered['patient_id'].unique()
        # Strictly filter the baseline to ONLY those exact same patients to prevent selection bias
        df_base_filtered = df_baseline[df_baseline['patient_id'].isin(confident_patients)].copy()
    else:
        df_exp_filtered = df_experimental.copy()
        df_base_filtered = df_baseline.copy()
        
        # Handle the M5 edge case where QC rejected ALL patches (silent dropping fix)
        if 'final_pred_safe' not in df_exp_filtered.columns:
            df_exp_filtered['final_pred'] = np.where(
                df_exp_filtered['final_pred'] == -1, 
                1 - df_exp_filtered['true_class'], # Force an incorrect prediction
                df_exp_filtered['final_pred']
            )

    # 2. Merge to guarantee perfect paired alignment row-by-row
    df_paired = pd.merge(
        df_base_filtered[['patient_id', 'true_class', 'final_pred', 'probability_1_mean']],
        df_exp_filtered[['patient_id', 'final_pred', 'probability_1_mean']],
        on='patient_id',
        suffixes=('_base', '_exp')
    )
    
    n_patients = len(df_paired)
    if n_patients == 0:
        return print("Error: No overlapping patients found.")

    # ==========================================
    # TEST 1: McNemar's Test (Binary Accuracy)
    # ==========================================
    df_paired['base_correct'] = df_paired['final_pred_base'] == df_paired['true_class']
    df_paired['exp_correct'] = df_paired['final_pred_exp'] == df_paired['true_class']

    both_correct = len(df_paired[df_paired['base_correct'] & df_paired['exp_correct']])
    both_wrong = len(df_paired[~df_paired['base_correct'] & ~df_paired['exp_correct']])
    exp_only = len(df_paired[~df_paired['base_correct'] & df_paired['exp_correct']]) # Rescues
    base_only = len(df_paired[df_paired['base_correct'] & ~df_paired['exp_correct']]) # Failures

    mcnemar_table = [[both_correct, exp_only], [base_only, both_wrong]]
    
    # We use exact=True because discordant pairs (rescues/failures) are typically small (< 25)
    result_mcnemar = mcnemar(mcnemar_table, exact=True)

    # ==========================================
    # TEST 2: Wilcoxon Signed-Rank Test (Probabilistic Error)
    # ==========================================
    # Calculate Absolute Error |P(y=1) - Y_true|
    df_paired['error_base'] = np.abs(df_paired['probability_1_mean_base'] - df_paired['true_class'])
    df_paired['error_exp'] = np.abs(df_paired['probability_1_mean_exp'] - df_paired['true_class'])

    # Run Wilcoxon
    stat, p_value_wilcoxon = wilcoxon(df_paired['error_base'], df_paired['error_exp'])

    # ==========================================
    # Output Formatting
    # ==========================================
    print(f"\n{'='*50}")
    print(f"EVALUATION: {subset_mode} Cohort (N={n_patients})")
    print(f"{'='*50}")
    print(f"McNemar's p-value: {result_mcnemar.pvalue:.4f}")
    print(f"Triage Rescues (Exp Correct, Base Wrong): {exp_only}")
    print(f"Triage Failures (Base Correct, Exp Wrong): {base_only}")
    print(f"---")
    print(f"Wilcoxon p-value: {p_value_wilcoxon:.4f}")
    print(f"Mean Prob Error (Base): {df_paired['error_base'].mean():.4f}")
    print(f"Mean Prob Error (Exp):  {df_paired['error_exp'].mean():.4f}")

# --- Execution ---

csv_m4 = 'Ablation/test_ablation_M4_BASELINE_20260901-1109.csv'
csv_m5 = 'Ablation/test_ablation_M5_QC_FULL_20260901-1109.csv'
csv_m6 = 'Ablation/test_ablation_M6_QC_TRIAGE_20260901-1109.csv'
df_m4 = pd.read_csv(csv_m4)
df_m5 = pd.read_csv(csv_m5)
df_m6 = pd.read_csv(csv_m6)

# Ablation 1: M4 vs M5 (Full Cohort)
evaluate_ablation_pair(df_baseline=df_m4, df_experimental=df_m5, subset_mode="Full")

# Ablation 2: M4 vs M6 (Autonomous / Confident Subset)
evaluate_ablation_pair(df_baseline=df_m4, df_experimental=df_m6, subset_mode="Confident")


#%%
config = cf.get_config()
from PatchDatasetPreserve import PatchDatasetPreserve
if config.External_Patch_Audit:
    # found_files, patient_patch_count = mygears.create_cell_dataframe_debug(config)
    df_ane = mygears.create_cell_dataframe_anerbc(config,
        glob_pattern = "*.png"
    )
    # Get a unique list of all patients
    # all_patient_ids = df['patient_id'].unique()
    # Now, df should contain data for all 20 patients
    # print(f"\nDataFrame created with {df['patient_id'].nunique()} unique patients.\n")
    print(f"Number of segmented cells: {len(df_ane)}")
    print(df_ane.columns)
    print(df_ane.sample(1))
    
    external_gen = PatchDatasetPreserve(
            config,
            df=df_ane,
            batch_size=config.TRAIN.BATCH_SIZE,
            # preprocessing_function=preprocessing_fn,
            shuffle=False,
            expect_rgba=False,
        )
    # --- Sanity check ---
    print("\nGenerator summary:")
    print(external_gen.summary())
    rand_idx = np.random.randint(0, len(external_gen)) 
    images, labels = external_gen.__getitem__(rand_idx)
    print(f"{rand_idx}, {images.dtype=}, {images.shape=}, {labels.shape=}")
    # external_gen.plot_random_batch(
    #     batch_idx=rand_idx,
    #     note=f'AneRBC-II',
    #     save_dir=f'AneRBC-II'
    # )
    
    #%% --- 0. Initialize Audit Storage ---
    patch_audit_data = [] 
    
    t1 = datetime.now()
    print(f'\n[Step 4] Patch-Level Audit on AneRBC-II | Generator: PatchDatasetPreserve')
    
    # --- 1. CONFIGURATION ---
    save_dir = f'AneRBC_II_Audit_Results'
    os.makedirs(save_dir, exist_ok=True)
    
    # --- Setup Standardized QC Tools (The Supervisor) ---
    # We use the Fold 4 Manifold as the "Gold Standard" for filtering
    BEST_FOLD_ID = 4
    standard_qc_riemann = riemann_tools_map[BEST_FOLD_ID]
    standard_qc_extractor = feature_extractors_map[BEST_FOLD_ID] 
    
    # 2. Use the prepared external_gen (PatchDatasetPreserve)
    # This generator iterates in batches (e.g., 32 patches at a time)
    print(f"Processing {len(external_gen)} batches ({len(df_ane)} total patches)...")
    
    # 3. Inference Loop (Batch-by-Batch)
    # Since shuffle=False, the order matches df_ane exactly
    for batch_idx in tqdm(range(len(external_gen)), desc="Riemannian Filtering"):
        
        # 1. Retrieve Batch Data
        # X_patches shape: (batch_size, 256, 256, 3)
        X_patches, _ = external_gen.__getitem__(batch_idx)
        
        # Identify the slice of the dataframe corresponding to this batch
        start_idx = batch_idx * config.TRAIN.BATCH_SIZE
        end_idx = min(start_idx + config.TRAIN.BATCH_SIZE, len(df_ane))
        batch_meta = df_ane.iloc[start_idx:end_idx]
        
        # ---------------------------------------------------------
        # 2A. Patch QC (Hyperbolic Curvature Calculation)
        # ---------------------------------------------------------
        # Project patches to the latent space of the Supervisor
        Z_for_qc = standard_qc_extractor.predict(X_patches, verbose=0)
        
        # Compute Curvature Energy (Riemannian Metric)
        # Higher energy = closer to the Poincaré boundary (potential OOD/Artifact)
        E_for_qc = standard_qc_riemann.compute_batch_curvature(Z_for_qc).reshape(-1, 1)
        
        # Apply Riemannian Filter
        # Note: rescue_top_k is applied per batch here
        _, _, valid_mask, _ = mygears.apply_riemannian_filter(
            features=Z_for_qc, 
            energies=E_for_qc,
            limit=CLINICAL_WORKFLOW_CONFIG["OOD_ENERGY_LIMIT"],
            rescue_top_k=CLINICAL_WORKFLOW_CONFIG["RESCUE_TOP_K"],
            extreme_limit=CLINICAL_WORKFLOW_CONFIG["EXTREME_CEILING"],
            verbose=False
        )
        
        # --- 2B. Metadata Synchronization ---
        limit_val = CLINICAL_WORKFLOW_CONFIG["OOD_ENERGY_LIMIT"]
        
        for i, (meta_idx, row) in enumerate(batch_meta.iterrows()):
            energy = float(E_for_qc[i])
            is_valid = valid_mask[i]
            
            # Status categorization for visualization
            if is_valid:
                status = "Accepted (Standard)" if energy <= limit_val else "Accepted (High Energy Rescue)"
            else:
                status = "REJECTED (OOD/Extreme)"
            
            patch_audit_data.append({
                "patient_id": row.get('patient_id', 'Unknown'),
                "label": row.get('label', -1),
                "patch_path": row['patch_path'],
                "curvature_energy": energy,
                "status": status
            })

    # --- 4. SAVE & REPORT ---
    dt_str = datetime.now().strftime("%Y%m%d-%H%M")
    patch_audit_df = pd.DataFrame(patch_audit_data)
    
    # Save the full audit CSV
    filter_result_file = f"AneRBC_II_patch_audit_{dt_str}.csv"
    patch_csv_path = os.path.join(save_dir, filter_result_file)
    patch_audit_df.to_csv(patch_csv_path, index=False)
    
    # Provide deep analysis of the filter performance on external data
    n_total = len(patch_audit_df)
    n_passed = len(patch_audit_df[patch_audit_df['status'].str.contains('Accepted')])
    
    print(f"\n{'='*40}")
    print(f"AUDIT SUMMARY: AneRBC-II")
    print(f"{'='*40}")
    print(f"Total Patches: {n_total}")
    print(f"Passed QC:     {n_passed} ({n_passed/n_total:.2%})")
    print(f"Rescued:      {len(patch_audit_df[patch_audit_df['status'] == 'Accepted (High Energy Rescue)'])}")
    print(f"Audit CSV:     {patch_csv_path}")
    print(f"{'='*40}")

    # --- 5. VISUALIZATION ---
    # Triggering your existing plotter to see the AneRBC results
    mygears.plot_patient_audit_grids(
        patch_df=patch_audit_df, 
        save_dir=os.path.join(save_dir, "Visualization_Grids"), 
        max_patches=32, 
        config=config
    )
    print("Rieman Filter Completed")
    
#%%
config = cf.get_config()
if config.Performance_Comparison:
    # Approach: Compare Ensemble (All-Models) vs. Respective (Fold 4).
    val_ens_file = 'val_predictions_ENSEMBLE_OTHERS_QC_Std_20251221-1341.csv'
    val_res_file = 'val_predictions_RESPECTIVE_QC_Std_20251221-1222.csv'

    val_ens_df = pd.read_csv(val_ens_file)
    print(val_ens_df.columns)
    print(val_ens_df.sample(3))
    val_res_df = pd.read_csv(val_res_file)
    
    #%%
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, f1_score
    
    def evaluate_ensemble_advantage(ens_df, res_df, save_path="comparison_1_roc.png"):
        """
        Comparison 1: Rigorously compares Ensemble vs. Respective strategies using 
        ROC-AUC and statistical significance testing.
        """
        # 1. Align Data by Patient ID to ensure paired comparison
        merged = pd.merge(
            ens_df[['patient_id', 'true_class', 'probability_1_mean']],
            res_df[['patient_id', 'probability_1_mean']],
            on='patient_id', suffixes=('_ens', '_res')
        )
        
        y_true = merged['true_class'].values
        y_ens = merged['probability_1_mean_ens'].values
        y_res = merged['probability_1_mean_res'].values
        
        # 2. Compute Global Metrics
        auc_ens = roc_auc_score(y_true, y_ens)
        auc_res = roc_auc_score(y_true, y_res)
        
        # P-value Approximation (Permutation Test)
        obs_diff = np.abs(auc_ens - auc_res)
        n_perm = 5000
        count = 0
        for _ in range(n_perm):
            mask = np.random.randint(0, 2, size=len(y_true)).astype(bool)
            s_a = np.where(mask, y_ens, y_res)
            s_b = np.where(mask, y_res, y_ens)
            if np.abs(roc_auc_score(y_true, s_a) - roc_auc_score(y_true, s_b)) >= obs_diff:
                count += 1
        p_val = count / n_perm
    
        # 3. Visualization
        fpr_e, tpr_e, _ = roc_curve(y_true, y_ens)
        fpr_r, tpr_r, _ = roc_curve(y_true, y_res)
        
        plt.figure(figsize=(7, 6), dpi=300)
        plt.plot(fpr_e, tpr_e, label=f'Ensemble (AUC={auc_ens:.3f})', color='#1f77b4', lw=2.5)
        plt.plot(fpr_r, tpr_r, label=f'Respective (AUC={auc_res:.3f})', color='#d62728', linestyle='--')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        plt.title("Ensemble vs. Respective Model Performance", fontsize=14)
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.legend(loc='lower right')
        plt.grid(alpha=0.2)
        plt.savefig(save_path, dpi=300)
        
        return {
            "AUC_Ensemble": auc_ens,
            "AUC_Respective": auc_res,
            "P_Value": p_val,
            "Improvement": (auc_ens - auc_res) / auc_res
        }

    results = evaluate_ensemble_advantage(val_ens_df, val_res_df)
    print(results)
    # {'AUC_Ensemble': 0.9289827779142443, 'AUC_Respective': 0.6848167253473076, 'P_Value': 0.0, 'Improvement': 0.35654218644135316}

    #%% Comparison 2: Aggregation Logic (Mean vs. Argmax)
    val_mean_file = 'val_predictions_mean_QC_Std_20251220-1240.csv'
    val_argmax_file = 'val_predictions_argmax_QC_Std_20251220-1240.csv'
    val_mean_df = pd.read_csv(val_mean_file)
    print(val_mean_df.columns)
    print(val_mean_df.sample(3))
    val_argmax_df = pd.read_csv(val_argmax_file)
    print(val_argmax_df.columns)
    print(val_argmax_df.sample(3))
    
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report
    
    def evaluate_aggregation_logic_comparison(mean_df, argmax_df, save_dir='.'):
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Alignment - Ensure patients match
        # We merge to ensure paired comparison for McNemar's
        merged = pd.merge(
            mean_df[['patient_id', 'true_class', 'predicted_class']],
            argmax_df[['patient_id', 'predicted_class']],
            on='patient_id',
            suffixes=('_mean', '_argmax')
        )
        
        y_true = merged['true_class']
        y_pred_mean = merged['predicted_class_mean']
        y_pred_argmax = merged['predicted_class_argmax']
        
        # 2. Performance Metrics
        def get_metrics(y_t, y_p):
            return {
                'Accuracy': accuracy_score(y_t, y_p),
                'Precision (Macro)': precision_score(y_t, y_p, average='macro'),
                'Recall (Macro)': recall_score(y_t, y_p, average='macro'),
                'F1 (Macro)': f1_score(y_t, y_p, average='macro'),
                'F1 (THL/Class 1)': f1_score(y_t, y_p, pos_label=1)
            }
    
        metrics_mean = get_metrics(y_true, y_pred_mean)
        metrics_argmax = get_metrics(y_true, y_pred_argmax)
        
        results_df = pd.DataFrame([metrics_mean, metrics_argmax], index=['Soft Voting (Mean)', 'Hard Voting (Argmax)'])
        
        # 3. McNemar's Test (Statistical Significance of Difference)
        # Contingency Table:
        #                 Argmax Correct | Argmax Incorrect
        # Mean Correct        a          |        b
        # Mean Incorrect      c          |        d
        
        correct_mean = (y_pred_mean == y_true)
        correct_argmax = (y_pred_argmax == y_true)
        
        a = np.sum(correct_mean & correct_argmax)
        b = np.sum(correct_mean & ~correct_argmax)
        c = np.sum(~correct_mean & correct_argmax)
        d = np.sum(~correct_mean & ~correct_argmax)
        
        contingency_table = [[a, b], [c, d]]
        # exact=True because sample size might be small
        mcnemar_result = mcnemar(contingency_table, exact=True)
        p_value = mcnemar_result.pvalue
        
        # 4. Visualization
        metrics_plot_df = results_df.reset_index().melt(id_vars='index', var_name='Metric', value_name='Score')
        
        plt.figure(figsize=(10, 6), dpi=300)
        # ax = sns.barplot(data=metrics_plot_df, x='Metric', y='Score',
        #             hue='index', palette='muted')
        color_pal = sns.color_palette("colorblind", 10).as_hex()
        ax = sns.barplot(data=metrics_plot_df, x='Metric', y='Score',
                        hue='index', palette=[color_pal[8], color_pal[9]])
        plt.title(f'Comparison 2: Aggregation Logic Performance\n(McNemar p-value: {p_value:.4f})', fontsize=14)
        plt.ylim(0, 1.1)
        plt.xlabel('Evaluation Metrics', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(title='Logic', loc='lower right')
        plt.grid(axis='y', alpha=0.3)
        
        for p in plt.gca().patches:
            if p.get_height() > 0:
                plt.gca().annotate(f'{p.get_height():.3f}', 
                                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                                   ha = 'center', va = 'center', 
                                   xytext = (0, 9), 
                                   textcoords = 'offset points',
                                   fontsize=12)
                
        plt.tight_layout()
        plot_path = os.path.join(save_dir, 'comparison_2_metrics.png')
        plt.savefig(plot_path, dpi=300)
        plt.show()
        plt.close()
        
        # Print reports
        print("\n" + "="*40)
        print("COMPARISON 2: MEAN VS ARGMAX")
        print("="*40)
        print(results_df.T)
        print(f"\nMcNemar Test p-value: {p_value:.4f}")
        if p_value < 0.05:
            print("Conclusion: There IS a statistically significant difference between the two logic types.")
        else:
            print("Conclusion: There is NO statistically significant difference (p >= 0.05).")
        
        return results_df, p_value
    
    
    # Let's try to run it on the uploaded Ensemble file for a real result:
    try:
        df_ens = pd.read_csv('val_predictions_ENSEMBLE_OTHERS_QC_Std_20251221-1341.csv')
        # Simulate the two dataframes the user mentioned
        # Mean-based DF (using T_OPT 0.5538)
        T_LOWER, T_UPPER, T_OPT = 0.5130, 0.5729, 0.5538
        mean_df_sim = df_ens.copy()
        mean_df_sim['predicted_class'] = (mean_df_sim['probability_1_mean'] >= T_OPT).astype(int)
        
        # Argmax-based DF (using 0.5 as cutoff for the ratio)
        argmax_df_sim = df_ens.copy()
        argmax_df_sim['predicted_class'] = (argmax_df_sim['probability_1_argmax'] >= 0.5).astype(int)
        
        results_df, p_value = evaluate_aggregation_logic_comparison(mean_df_sim, argmax_df_sim)
    except Exception as e:
        print(f"Demo failed: {e}")
    
    #%% Comparison 3: The "Clinical Triage" Effect (Full vs. Confident)
    T_LOWER, T_UPPER, T_OPT = 0.5130, 0.5729, 0.5538
    from sklearn.metrics import confusion_matrix
    
    def compute_detailed_clinical_metrics(y_true, y_pred, name="Subset"):
        """
        Computes ACC, TPR, TNR, PPV, NPV with 95% Wilson CIs and Likelihood Ratios.
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        n = len(y_true)
        
        # Proportions
        metrics = {
            "ACC": (tp + tn, n),
            "TPR (Sens)": (tp, tp + fn),
            "TNR (Spec)": (tn, tn + fp),
            "PPV": (tp, tp + fp),
            "NPV": (tn, tn + fn),
        }
        
        results = {}
        for metric, (count, total) in metrics.items():
            val = count / total if total > 0 else 0
            low, high = proportion_confint(count, total, method='wilson')
            results[metric] = f"{val:.2%} [{low:.3f} - {high:.3f}]"
            results[f"{metric}_val"] = val # numeric for LR
            
        # Likelihood Ratios
        sens = results["TPR (Sens)_val"]
        spec = results["TNR (Spec)_val"]
        
        lr_plus = sens / (1 - spec) if (1 - spec) > 0 else float('inf')
        lr_minus = (1 - sens) / spec if spec > 0 else 0
        
        return {
            "Group": name,
            "N": n,
            "ACC": results["ACC"],
            "Sensitivity": results["TPR (Sens)"],
            "Specificity": results["TNR (Spec)"],
            "PPV": results["PPV"],
            "NPV": results["NPV"],
            "LR+": f"{lr_plus:.2f}",
            "LR-": f"{lr_minus:.3f}"
        }
    
    #%% Determine which models to use based on strategy
    # PREDICTION_STRATEGY = 'RESPECTIVE'
    PREDICTION_STRATEGY = 'ENSEMBLE_OTHERS'
    if PREDICTION_STRATEGY == 'RESPECTIVE':
        # Respective model
        T_LOWER = 0.4569  #(Confident IDA < this)
        T_UPPER = 0.6088  #(Confident THL >= this)
        T_OPT = 0.5175    #(Balanced Cutoff)
        audit_path = 'Test_Ensemble/test_predictions_final_RESPECTIVE_20251221-2255.csv'
    else: # ENSEMBLE_OTHERS
        T_LOWER = 0.5130  #(Confident IDA < this)
        T_UPPER = 0.5729  #(Confident THL >= this)
        T_OPT = 0.5538    #(Balanced Cutoff)
        audit_path = 'Test_Ensemble/test_predictions_final_ENSEMBLE_OTHERS_20251221-2255.csv'
    
    df = pd.read_csv(audit_path)
    # print(df.columns)
    df = df.reset_index(drop=True)
    
    cols_to_drop = ['final_pred', 'diagnosis_label', 'pred_status', 'probability_used']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    save_dir='Test_Ensemble'
    os.makedirs(save_dir, exist_ok=True)

    # Assuming T_LOWER, T_UPPER, T_OPT are already defined
    triage_data = []
    
    for idx, row in df.iterrows():
        result = mygears.apply_clinical_triage(
            p_patient_mean=row['probability_1_mean'],
            p_patient_argmax=row['probability_1_argmax'],
            t_lower=T_LOWER,
            t_upper=T_UPPER,
            t_opt=T_OPT,
            # fallback_mode='mean',
            fallback_mode='argmax'
        )
        triage_data.append(result)
        
    # Convert results to DataFrame and join
    df_results = pd.DataFrame(triage_data)
    df = pd.concat([df, df_results], axis=1)
    print(df.columns)
    
    #%% 5. Deep Analysis of the Triage Output
    n_total = len(df)
    n_confident = len(df[df['pred_status'] == "Confident"])
    n_uncertain = len(df[df['pred_status'] == "Uncertain"])
    
    print(f"{'='*40}")
    print(f"[Clinical Report] Triage Strategy: Hybrid")
    print(f"{'='*40}")
    print(f"Total Patients:  {n_total}")
    print(f"Confident Class: {n_confident} ({n_confident/n_total:.1%})")
    print(f"Uncertain (Grey Zone): {n_uncertain} ({n_uncertain/n_total:.1%})")
    print(f"{'='*40}")
   
    #% full
    y_true = df['true_class']
    y_pred = df['final_pred']
    print(f"y_true: {y_true.shape}")
    print(f"y_pred: {y_pred.shape}")
    
    # Extract probabilities only for the Confident subset 
    y_true_confident = df.loc[df['pred_status'] == 'Confident', 'true_class'].to_numpy()
    y_pred_confident = df.loc[df['pred_status'] == 'Confident', 'final_pred'].to_numpy()
    print(f"y_true_confident: {y_true_confident.shape}")
    print(f"y_pred_confident: {y_pred_confident.shape}")
    
    # 1. Full Dataset (Strategy A)
    full_metrics = compute_detailed_clinical_metrics(
        df['true_class'], 
        df['final_pred'], 
        name="Full Set (Strategy A)"
    )
    
    # 2. Confident Subset (Strategy B)
    df_conf = df[df['pred_status'] == 'Confident']
    conf_metrics = compute_detailed_clinical_metrics(
        df_conf['true_class'], 
        df_conf['final_pred'], 
        name="Confident Subset (Strategy B)"
    )
    
    # 3. Create Comparison Table
    comparison_df = pd.DataFrame([full_metrics, conf_metrics])
    print("\n" + "="*40)
    print("COMPARISON 3: CLINICAL TRIAGE EFFECT (Full vs. Confident)")
    print("="*40)
    print(comparison_df.set_index("Group").T)
    
    # Save to CSV for the paper
    comparison_df.to_csv(os.path.join(save_dir, "Comparison_3_Triage_Effect.csv"), index=False)
    
    #%%  
    def plot_clinical_forest_plot(save_path="Comparison_3_ForestPlot.png"):
        # ---------------------------------------------------------
        # 1. Data Setup (From your provided results)
        # ---------------------------------------------------------
        metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV']
        
        # Strategy A: Full Set (N=48)
        full_vals = [0.7708, 0.8889, 0.6190, 0.7500, 0.8125]
        full_low  = [0.635, 0.719, 0.409, 0.579, 0.570]
        full_high = [0.867, 0.961, 0.792, 0.867, 0.934]
        
        # Strategy B: Confident Subset (N=35)
        conf_vals = [0.9429, 0.9130, 1.0000, 1.0000, 0.8571]
        conf_low  = [0.814, 0.732, 0.758, 0.845, 0.601]
        conf_high = [0.984, 0.976, 1.000, 1.000, 0.960]
    
        # Calculate error bars (Distance from point to bound)
        full_err = [np.array(full_vals) - np.array(full_low), np.array(full_high) - np.array(full_vals)]
        conf_err = [np.array(conf_vals) - np.array(conf_low), np.array(conf_high) - np.array(conf_vals)]
    
        # ---------------------------------------------------------
        # 2. Plotting Configuration (Publication Quality)
        # ---------------------------------------------------------
        plt.rcParams.update({'font.size': 14, 'font.family': 'sans-serif'})
        fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
        
        y_pos = np.arange(len(metrics))
        height = 0.35  # Offset for grouping
    
        # Plot Strategy A (Full Set)
        ax.errorbar(full_vals, y_pos + height/2, xerr=full_err, fmt='o', color='#E69F00', #orange
                    capsize=5, elinewidth=2, markeredgewidth=2, label='Full Set (Strategy A)')
    
        # Plot Strategy B (Confident Subset)
        ax.errorbar(conf_vals, y_pos - height/2, xerr=conf_err, fmt='s', color='#009E73', #green
                    capsize=5, elinewidth=2, markeredgewidth=2, label='Confident Subset (Strategy B)')
    
        # ---------------------------------------------------------
        # 3. Aesthetics & Formatting
        # ---------------------------------------------------------
        ax.set_yticks(y_pos)
        ax.set_yticklabels(metrics, fontsize=16)
        ax.set_xlabel('Probability Score', fontsize=16,)
        ax.set_title('Comparison 3: Diagnostic Performance Triage Effect', fontsize=18, pad=20)
        
        ax.set_xlim(0.35, 1.05)  # Focus on the relevant range
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5) # Random chance line
        
        ax.legend(loc='lower left', frameon=True, fontsize=14)
        ax.grid(axis='x', linestyle=':', alpha=0.6)
        
        # Add percentage labels next to points for clarity
        for i, v in enumerate(full_vals):
            ax.text(v, i + height + 0.05, f"{v:.1%}", color='#E69F00', ha='center', fontweight='bold')
        for i, v in enumerate(conf_vals):
            ax.text(v, i - height - 0.2, f"{v:.1%}", color='#009E73', ha='center', fontweight='bold')
    
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.show()
    
    # Execute plot
    plot_clinical_forest_plot()
    
    #%% Comparison 4: Inter-Fold Generalizability (Respective Folds 0-4)
    # "Full" vs. "Confident"
    fold_0_file = 'Test_Respective/0/test_predictions_final_RESPECTIVE_20251222-1107.csv'
    fold_1_file = 'Test_Respective/1/test_predictions_final_fold_1_RESPECTIVE_20251226-1631.csv'
    fold_2_file = 'Test_Respective/2/test_predictions_final_RESPECTIVE_20251222-1022.csv'
    fold_3_file = 'Test_Respective/3/test_predictions_final_fold_3_RESPECTIVE_20251226-1211.csv'    
    fold_4_file = 'Test_Respective/4/test_predictions_final_fold_4_RESPECTIVE_20251226-1234.csv'
    file_list = [fold_0_file, fold_1_file, fold_2_file,
                 fold_3_file, fold_4_file]
    
    fold_0_df = pd.read_csv(fold_0_file)
    print(fold_0_df.columns)
    

    def get_fold_metrics(df):
        """Helper to compute Accuracy, Sensitivity, and Specificity."""
        y_true = df['true_class']
        y_pred = df['final_pred']
        
        if len(y_true) == 0: return 0, 0, 0
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        acc = (tp + tn) / len(y_true)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        return acc, sens, spec
    
    def run_comparison_4_interfold(file_list, save_dir='Comparison_4_Results'):
        os.makedirs(save_dir, exist_ok=True)
        summary_data = []
        plot_data = []
    
        for i, f_path in enumerate(file_list):
            if not os.path.exists(f_path):
                print(f"Skipping missing file: {f_path}")
                continue
                
            df = pd.read_csv(f_path)
            
            # --- 1. Full Set Analysis ---
            acc_f, sens_f, spec_f = get_fold_metrics(df)
            
            # --- 2. Confident Set Analysis ---
            df_conf = df[df['pred_status'] == 'Confident']
            acc_c, sens_c, spec_c = get_fold_metrics(df_conf)
            
            # --- 3. Store for Summary Table ---
            summary_data.append({
                'Fold': i,
                'Full_Acc': acc_f, 'Full_Sens': sens_f, 'Full_Spec': spec_f,
                'Conf_Acc': acc_c, 'Conf_Sens': sens_c, 'Conf_Spec': spec_c,
                'Conf_N': len(df_conf),
                'Throughput': len(df_conf) / len(df)
            })
            
            # --- 4. Store for Boxplots ---
            plot_data.extend([
                {'Fold': i, 'Group': 'Full', 'Metric': 'Sensitivity', 'Value': sens_f},
                {'Fold': i, 'Group': 'Full', 'Metric': 'Specificity', 'Value': spec_f},
                {'Fold': i, 'Group': 'Confident', 'Metric': 'Sensitivity', 'Value': sens_c},
                {'Fold': i, 'Group': 'Confident', 'Metric': 'Specificity', 'Value': spec_c}
            ])
    
        results_df = pd.DataFrame(summary_data)
        long_df = pd.DataFrame(plot_data)
    
        # --- 5. Calculate Stability Statistics (CV) ---
        stats = {}
        for group in ['Full', 'Conf']:
            acc_col = f'{group}_Acc'
            mean_acc = results_df[acc_col].mean()
            std_acc = results_df[acc_col].std()
            cv_acc = (std_acc / mean_acc) if mean_acc > 0 else 0
            stats[group] = {'Mean_Acc': mean_acc, 'Std_Acc': std_acc, 'CV': cv_acc}
    
        # --- 6. Visualization ---
        sns.set_context("talk")
        plt.figure(figsize=(12, 7), dpi=300)
        color_pal = sns.color_palette("colorblind", 10).as_hex()
        ax = sns.boxplot(data=long_df, x='Metric', y='Value',
                         hue='Group',
                         # palette='muted'
                         palette=[color_pal[6], color_pal[8]],
                         )
        sns.stripplot(data=long_df, x='Metric', y='Value',
                      hue='Group', 
                      dodge=True,
                      alpha=0.6,
                      linewidth=1,
                      edgecolor='gray',
                      ax=ax)
        
        plt.title(f'Inter-Fold Generalizability: Full vs. Confident Cohorts\n'
                  f'Full Accuracy CV: {stats["Full"]["CV"]:.2%} | Confident Accuracy CV: {stats["Conf"]["CV"]:.2%}', 
                  fontsize=18,
                  # fontweight='bold',
                  pad=20)
        plt.ylim(-0.05, 1.55)
        plt.ylabel('Performance Score', fontsize=16)
        plt.legend(title='Triage Group',
                   loc='upper left',
                   fontsize=13
                   )
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir,
                                 'comparison_4_interfold_stability.png'))
        
        return results_df, stats, long_df

    # Execute
    fold_results, fold_stats, long_df = run_comparison_4_interfold(
        file_list,
        save_dir='.', 
        )
    print(f"{fold_results=}")
    print(f"{fold_stats=}")
    # Save to CSV for the paper
    fold_results.to_csv("Comparison_4_fold_results.csv", index=False)
    long_df.to_csv("Comparison_4_long_df.csv", index=False)
    
#%% 
if config.Inference_Report:
    # 1. Load the M6 Results
    save_dir = 'Ablation'
    dt = config.TRAIN.DATETIME
    csv_file_name = os.path.join(save_dir, f'test_ablation_M6_QC_TRIAGE_20260901-1109.csv')
    df = pd.read_csv(csv_file_name)
    
    # =================================================================
    # 2. THE CRITICAL CORRECTION FOR M6: Filter for the Autonomous Zone
    # =================================================================
    # We only evaluate patients the model was confident enough to diagnose.
    df_conf = df[df['pred_status'] == 'Confident']
    
    y_true_conf = df_conf['true_class'].to_numpy()
    y_pred_conf = df_conf['final_pred'].to_numpy()
    y_prob_conf = df_conf['probability_1_mean'].to_numpy()
    
    total_cohort_size = len(df) # Should be 48
    
    print(f"\n--- M6 Autonomous Subset ---")
    print(f"Total Patients: {total_cohort_size}")
    print(f"Confident Patients: {len(df_conf)}")
    print(f"Coverage: {len(df_conf)/total_cohort_size:.1%}")
    
    if len(df_conf) > 0:
        # ---------------------------------------------------------
        # 3. Generate the Manuscript Table Metrics (Wilson CIs, AUC CIs)
        # ---------------------------------------------------------
        report_m6 = compute_detailed_clinical_metrics(
            y_true=y_true_conf, 
            y_pred=y_pred_conf, 
            y_prob=y_prob_conf, 
            total_n=total_cohort_size, # Passes 48 to correctly calculate the Coverage %
            name="M6 (Proposed Autonomous R-MIL)", 
            save_path=os.path.join(save_dir, "M6_detailed_metrics.json")
        )
        
        # ---------------------------------------------------------
        # 4. Generate the Confusion Matrix & Plots (mygears)
        # ---------------------------------------------------------
        save_prefix = f'test_predictions_M6_{dt}'
        
        p_measures, cm = mygears.evaluate_classification_performance(
            y_true=y_true_conf,
            y_pred=y_pred_conf,
            class_labels=config['DATA']['CLASS_LABELS'],
            save_dir=save_dir,
            save_prefix=save_prefix
        )
        
        mygears.plot_confusion_matrices(
            confusionmatrix=cm,
            p_measures=p_measures,
            class_labels=config['DATA']['CLASS_LABELS'],
            save_dir=save_dir,
            save_prefix=save_prefix
        )
        
        print("\n[SUCCESS] M6 metrics, JSON, and Confusion Matrix plots generated correctly for the Autonomous Subset.")
    else:
        print("\n[!] No patients survived the triage gate. Cannot plot confusion matrix.") 
    
    #%%
    df_intervals = mygears.calculate_clinical_wilson_intervals(
        f'{save_dir}/{dt}_Test_Ensemble_Confident_measures.json')
    print(df_intervals)
    # df_intervals.to_csv(f'{save_dir}/df_intervals.csv', index=False)
    df_intervals.to_csv(f'{save_dir}/{dt}_df_intervals_confident.csv', index=False)
    
    #%% 1. Calculate Metrics (using your reusable function)
    # p_measures, cm = mygears.evaluate_classification_performance(
    #     y_true,
    #     y_pred,
    #     class_labels=config['DATA']['CLASS_LABELS'], # e.g. ['IDA', 'THL']
    #     save_dir=save_dir,   # Default to current dir
    #     save_prefix=save_dir
    # )
        
    #%% 2. Plot Confusion Matrices (using your reusable function)
    mygears.plot_confusion_matrices(
        confusionmatrix=cm,
        p_measures=p_measures,
        class_labels=config['DATA']['CLASS_LABELS'],
        save_dir=config.TRAIN.get('SAVE_PATH', '.'),
        save_prefix=data['save_prefix']
    )
        
    #%% 3. Plot ROC (Uses PROBABILITIES 0.0-1.0)
    # Note: Only possible if probability data exists (it might be empty for some subsets)
    data = df_conf.copy()
    if len(np.unique(data['y_true'])) > 1:
        auc_score, opt_thresh = plot_roc_curve(
            y_true=data['y_true'],
            y_probs=data['y_prob'],  
            save_dir=config.TRAIN.get('SAVE_PATH', '.'),
            save_prefix=data['save_prefix']
        )
    else:
        print("   [Skipping ROC] Not enough classes to plot ROC.")
        
    #%%
    # 1. Prepare One-Hot Encoded Ground Truth
    # Shape becomes (N, 2)
    y_true_onehot = to_categorical(y_true, num_classes=2)
    
    # 2. Prepare 2D Probabilities
    # data['y_prob'] is just P(THL). We need [P(IDA), P(THL)].
    prob_thl = df.loc[df['pred_status'] == 'Confident', 'probability_1_mean'].to_numpy()
    prob_ida = 1.0 - prob_thl               # Shape (N,)
    y_probs_2d = np.column_stack((prob_ida, prob_thl)) # Shape (N, 2)
    
    #%% 3. Call the function with the 2D array
    mygears.plot_multiclass_roc(
        y_true=y_true_onehot,
        y_probs=y_probs_2d,       # <--- Pass the 2D array here
        class_labels=['IDA', 'THL'], 
        save_dir=save_dir,
        save_prefix='multiclass_roc_confident_meanbased'
    )
        
    #%% 4. Standard Calibration Plot
    mygears.plot_calibration_curve(
        y_true,
        y_probs=prob_thl,  # Uses the 1D probability array
        n_bins=10,
        save_dir=save_dir,
        save_prefix='calibration_plot_confident_meanbased'
    )
        
    #%% 5. Comparison Plot (Platt vs Isotonic)
    # Check: Needs >1 unique class and sufficient samples to fit regressors
    mygears.plot_calibration_comparison(
        y_true,
        y_probs=prob_thl,
        n_bins=6,
        save_dir=save_dir,
        save_prefix='calibration_plot_compare_confident_meanbased'
    )
           
    #%%
    # 6. Compute Clinical Metrics (LR, Sensitivity, Bayes)
    print(f"\n   >> Clinical Metrics")
    
    # pass y_true and y_pred (Hard labels)
    clinical_results = mygears.compute_clinical_metrics(
        y_true, 
        y_pred, 
        pre_test_prob=0.40  # Adjust this if prevalence differs
    )
    print(clinical_results)
    
    # Optional: Save these specific metrics to a JSON file if needed
    # (The evaluate_classification_performance function already saves standard metrics, 
    # but this adds the advanced LR/Bayesian stats)
    clinical_json_path = os.path.join(
        save_dir, 
        f"confident_meanbased_clinical_metrics.json"
    )
    
    try:
        with open(clinical_json_path, 'w') as f:
            json.dump(clinical_results, f, indent=4)
        print(f"   Saved clinical metrics -> {clinical_json_path}")
    except Exception as e:
        print(f"   Could not save clinical JSON: {e}")
        
    #%%
    # 7. Compute Calibration Scores (Brier & R2)
    print(f"\n   >> Calibration Scores")
    
    # Check if we have probabilities (Case 1 and 2 always do; Case 3 might be empty)
    if len(prob_thl) > 0:
        calib_scores = mygears.compute_calibration_scores(
            y_true, 
            y_probs=prob_thl
        )
        print(calib_scores)
        # Save scores to JSON
        calib_json_path = os.path.join(
            save_dir, 
            f"{dt}_confident_meanbased_calibration_scores.json"
        )
        
        try:
            with open(calib_json_path, 'w') as f:
                json.dump(calib_scores, f, indent=4)
            print(f"   Saved calibration scores to: {calib_json_path}")
        except Exception as e:
            print(f"   Could not save calibration JSON: {e}")
            
    else:
        print("   [Skipping] No probability data available for calibration scoring.")
           


#%%












































