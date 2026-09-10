#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 18:41:47 2025

@author: kasikritdamkliang
"""
import tensorflow as tf
import os, platform
import cv2
from datetime import datetime
from tqdm import tqdm
# import itertools
import matplotlib.pylab as plt
import numpy as np
import pandas as pd

# from tensorflow.keras import models, layers, optimizers, Model
from sklearn.metrics import classification_report
# from tensorflow.keras.utils.np_utils import to_categorical
import tensorflow_addons as tfa
tqdm_callback = tfa.callbacks.TQDMProgressBar(show_epoch_progress=False)
# from livelossplot import PlotLossesKeras


# import tensorflow.keras.layers as L
import warnings
warnings.filterwarnings("ignore")

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
# import configsInfer
# config = configsInfer.get_config()
import configsCell as cf
config = cf.get_config()

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
if config.Performance_Report:       
    
    # Determine which models to use based on strategy
    PREDICTION_STRATEGY = 'RESPECTIVE'
    # PREDICTION_STRATEGY = 'ENSEMBLE_OTHERS'
    if PREDICTION_STRATEGY == 'RESPECTIVE':
        # Respective model
        T_LOWER = 0.4569  #(Confident IDA < this)
        T_UPPER = 0.6088  #(Confident THL >= this)
        T_OPT = 0.5175    #(Balanced Cutoff)
        audit_path = 'val_predictions_RESPECTIVE_QC_Std_20251221-1222.csv'
    else: # ENSEMBLE_OTHERS
        T_LOWER = 0.5130  #(Confident IDA < this)
        T_UPPER = 0.5729  #(Confident THL >= this)
        T_OPT = 0.5538    #(Balanced Cutoff)
        audit_path = 'val_predictions_ENSEMBLE_OTHERS_QC_Std_20251221-1341.csv'
    
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
   
    #% Verify Shapes
    y_true = df['true_class']
    y_pred = df['final_pred']
    
    # Extract probabilities only for the Confident subset
    y_pred_confident = df.loc[df['pred_status'] == 'Confident', 'final_pred'].to_numpy()
    y_true_confident = df.loc[df['pred_status'] == 'Confident', 'true_class'].to_numpy()
    
    # y_pred_confident = df.loc[df['pred_status'] == 'Confident', 'predicted_class'].to_numpy()
    # y_true_confident = df.loc[df['pred_status'] == 'Confident', 'true_class'].to_numpy()
    # y_true = y_true_confident
    # y_pred = y_pred_confident
    
    print(f"y_true: {y_true.shape}")
    print(f"y_pred: {y_pred.shape}")
    
    print('\nVal set classification_report')
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

#%% [Step 4] Inference on Unseen Test Set
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
    
    t1 = datetime.now()
    print('\nInference on Unseen Test Set (Standardized QC + Strategy 2B)')
    
    # --- 1. CONFIGURATION ---
    save_dir = 'Test_Respective'
    os.makedirs(save_dir, exist_ok=True)
    
    PREDICTION_STRATEGY = 'ENSEMBLE_OTHERS' 
    # PREDICTION_STRATEGY = 'RESPECTIVE'
    
    if PREDICTION_STRATEGY == 'RESPECTIVE':
        T_LOWER, T_UPPER, T_OPT = 0.4569, 0.6088, 0.5175
        target_model_indices = [4]
        # target_model_indices = [3]
        # target_model_indices = [2]
        # target_model_indices = [1]
        # target_model_indices = [0]
    else: # ENSEMBLE_OTHERS (Full Ensemble for Unseen Test)
        T_LOWER, T_UPPER, T_OPT = 0.5130, 0.5729, 0.5538
        target_model_indices = list(range(len(loaded_model_list))) # Use all models
    print(f"{target_model_indices=}") 
    
    # --- Setup Standardized QC Tools (The "Supervisor") ---
    BEST_FOLD_ID = 4
    standard_qc_riemann = riemann_tools_map[BEST_FOLD_ID]
    standard_qc_extractor = feature_extractors_map[BEST_FOLD_ID] 
    
    # 2. Initialize Generator
    model_builder = ModelBuilder(model_name=model_names[0], config=config)
    preprocessing_fn = model_builder.get_preprocessing_function()
    
    inference_gen = FullBagDataset(
        config=config, df=test_df_sampled,
        preprocessing_function=preprocessing_fn,
        shuffle=False, expect_rgba=True, mode='test'
    )
       
    # 3. Inference Loop
    print(f"\nInference for {len(inference_gen)} test patients using {len(target_model_indices)} models...")
    test_results = []
 
    order = 1
    for i in tqdm(range(len(inference_gen)), desc="Inferencing Test Patients"):
        # pass 
        print("\n\nRetrieve Patient Data")
        batch_patients, labels_true, X_batch, _ = inference_gen.__getitem__(i, get_pids=True)
        patient_id, class_id, X_patches = batch_patients[0], labels_true[0], X_batch[0]
        n_total = len(X_patches)
        print(f"Patient: {patient_id}")
        print(f"Label: {class_id}")  
        print(f"{X_patches.shape=}")
        # inference_gen.plot_random_patient_batch(
        #     idx=i,
        #     save_dir=save_dir)
        
        # ---------------------------------------------------------
        # 2A. Standardized Patch QC (Supervisor Gate)
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
            # 1. Collect predictions from all targeted models
            all_model_patch_probs = []
            for m_idx in target_model_indices:
                current_model = loaded_model_list[m_idx]
                y_pred_raw = current_model.predict(X_patches_qc, 
                                                   batch_size=config.TRAIN.BATCH_SIZE, 
                                                   verbose=0)
                all_model_patch_probs.append(y_pred_raw)
            
            # 2. Compute averaged model probabilities per patch (Soft-Voting Consensus)
            # Shape: (N_passed, 2)
            y_patch_probs_avg = np.mean(all_model_patch_probs, axis=0)

            # --- Aggregation Strategy A: Mean-Based ---
            # Represents the soft probabilistic average: P = 1/N * Σ p_i
            P_patient_mean = np.mean(y_patch_probs_avg[:, 1])

            # --- Strategy 2B: Argmax-Based (Hard Voting on Consensus) ---
            # 1. Determine class for each patch based on averaged model probs
            y_patch_labels = np.argmax(y_patch_probs_avg, axis=1)

            # 2. Count votes for Class 1 (THL)
            class_1_count = np.sum(y_patch_labels == 1)
            total_patches = len(y_patch_labels)

            # Final patient-level argmax probability (Ratio of THL votes)
            P_patient_argmax = class_1_count / total_patches
            
            # --- 3. Apply Clinical Triage Logic ---
            res = mygears.apply_clinical_triage(
                p_patient_mean=P_patient_mean, 
                p_patient_argmax=P_patient_argmax,
                t_lower=T_LOWER, 
                t_upper=T_UPPER, 
                t_opt=T_OPT,
                fallback_mode='mean' # High-sensitivity choice
            )
            
            # 4. Store Results for Clinical Audit
            test_results.append({
                "patient_id": patient_id,
                "true_class": class_id,
                "probability_1_mean": P_patient_mean,
                "probability_1_argmax": P_patient_argmax,
                "final_pred": res['final_pred'],
                "diagnosis_label": res['diagnosis_label'],
                "pred_status": res['pred_status'],
                "qc_pass_rate": n_passed / n_total,
                "qc_rescued": qc_stats['n_rescued'],
                "strategy": PREDICTION_STRATEGY
            })
            
            print(f"\n   Patient {patient_id} | Class {class_id} | \
QC Pass: {n_passed}/{n_total} | \
Pass Rate: {n_passed / n_total:.4f} | \
Mean Prob: {P_patient_mean:.4f} | \
Argmax Prob: {P_patient_argmax:.4f} \
Status: {res['pred_status']}")
        else:
            print(f"   [!] QC REJECT: Patient {patient_id} had 0 valid patches.")
            
        order += 1

    # --- 5. Save and Evaluate ---
    test_df_final = pd.DataFrame(test_results)
    dt = config.TRAIN.DATETIME
    final_csv = f"test_predictions_final_{PREDICTION_STRATEGY}_{dt}.csv"
    test_df_final.to_csv(final_csv, index=False)
    
    t2 = datetime.now() - t1
    print(f"Inference times for the test set: {t2}")
    
    #%% Filter for Confident Subset
    df_conf = test_df_final[test_df_final['pred_status'] == 'Confident']
    save_dir = 'Test_Ensemble'
    # csv_file_name = 'Test_Ensemble/test_predictions_final_ENSEMBLE_OTHERS_20251221-2255.csv'
    # df_conf = pd.read_csv(csv_file_name)
    
    y_true_conf = df_conf['true_class'].values
    y_pred_conf = df_conf['final_pred'].values
    
    print(f"\n{'='*40}")
    print(f"FINAL TEST REPORT ({PREDICTION_STRATEGY})")
    print(f"{'='*40}")
    print(f"Confident Fraction: {len(df_conf)}/{len(test_df_final)} ({len(df_conf)/len(test_df_final):.1%})")
    
    if len(y_true_conf) > 0:
        print("\nConfident Zone Performance:")
        print(classification_report(y_true_conf, y_pred_conf, 
                target_names=config['DATA']['CLASS_LABELS']))
        
        # Generate Q1 Publication plots
        p_measures, cm = mygears.evaluate_classification_performance(
            y_true_conf, y_pred_conf,
            class_labels=config['DATA']['CLASS_LABELS'],
            save_dir=save_dir,
            save_prefix=f'test_predictions_final_{PREDICTION_STRATEGY}_{dt}'
        )
    else:
        print("\n[!] No patients reached the confidence thresholds.")  
    
    
#%% 
config = cf.get_config()

if config.Inference_Audit:   
    # --- 0. Initialize Audit Storage ---
    test_results = []
    patch_audit_data = [] # To store individual patch results for the CSV
    
    t1 = datetime.now()
    print('\n[Step 4] Inference on Unseen Dataset (Auditing Enabled)')
    
    # --- 1. CONFIGURATION ---
    PREDICTION_STRATEGY = 'ENSEMBLE_OTHERS' 
    save_dir = f'Test_Set_Audit_Grids'
    os.makedirs(save_dir, exist_ok=True)
    
    # Thresholds and model setup
    if PREDICTION_STRATEGY == 'RESPECTIVE':
        T_LOWER, T_UPPER, T_OPT = 0.4569, 0.6088, 0.5175
        target_model_indices = [4]
    else: 
        T_LOWER, T_UPPER, T_OPT = 0.5130, 0.5729, 0.5538
        target_model_indices = list(range(len(loaded_model_list)))
    
    # --- Setup Standardized QC Tools ---
    BEST_FOLD_ID = 4
    standard_qc_riemann = riemann_tools_map[BEST_FOLD_ID]
    standard_qc_extractor = feature_extractors_map[BEST_FOLD_ID] 
    
    # Initialize Generator
    model_builder = ModelBuilder(model_name=model_names[0], config=config)
    preprocessing_fn = model_builder.get_preprocessing_function()
    
    inference_gen = FullBagDataset(
        config=config,
        # df=test_df,
        df=df_ane,
        # preprocessing_function=preprocessing_fn,
        shuffle=False,
        expect_rgba=True,
        mode='test',
    )
     
    # 3. Inference Loop
    print(f"Auditing...")
    
    for i in tqdm(range(len(inference_gen)), desc="Processing"):
        # 1. Retrieve Patient Data
        batch_patients, labels_true, X_batch, _ = inference_gen.__getitem__(i, get_pids=True)
        patient_id, class_id, X_patches = batch_patients[0], labels_true[0], X_batch[0]
    
        print(f"Patient: {patient_id}")
        print(f"Label: {class_id}")  
        print(f"{X_patches.shape=}")
        
        # IMPORTANT: Get original patch paths for this patient to link with curvature results
        # This assumes FullBagDataset stores the dataframe or patch list internally
        patient_patch_paths = inference_gen.df[inference_gen.df['patient_id'] == patient_id]['patch_path'].tolist()
    
        # ---------------------------------------------------------
        # 2A. Standardized Patch QC (The Supervisor)
        # ---------------------------------------------------------
        Z_for_qc = standard_qc_extractor.predict(X_patches, batch_size=config.TRAIN.BATCH_SIZE, verbose=0)
        E_for_qc = standard_qc_riemann.compute_batch_curvature(Z_for_qc).reshape(-1, 1)
        
        # Call the filter
        _, _, valid_mask, qc_stats = mygears.apply_riemannian_filter(
            features=Z_for_qc, energies=E_for_qc,
            limit=CLINICAL_WORKFLOW_CONFIG["OOD_ENERGY_LIMIT"],
            rescue_top_k=CLINICAL_WORKFLOW_CONFIG["RESCUE_TOP_K"],
            extreme_limit=CLINICAL_WORKFLOW_CONFIG["EXTREME_CEILING"],
            verbose=False
        )
        
        # --- COLLECT PATCH AUDIT DATA ---
        limit_val = CLINICAL_WORKFLOW_CONFIG["OOD_ENERGY_LIMIT"]
        
        for idx in range(len(X_patches)):
            energy = float(E_for_qc[idx])
            is_valid = valid_mask[idx]
            
            # Categorize the status for the plotter
            if is_valid:
                if energy <= limit_val:
                    status = "Accepted (Standard)"
                else:
                    status = "Accepted (High Energy Rescue)"
            else:
                status = "REJECTED (OOD/Extreme)"
    
            patch_audit_data.append({
                "patient_id": patient_id,
                "label": class_id,
                "patch_path": patient_patch_paths[idx],
                "curvature_energy": energy,
                "status": status
            })
    
    
    # --- 4. SAVE AUDIT FILES ---
    dt_str = datetime.now().strftime("%Y%m%d-%H%M")
    
    # Save Patch-Level Audit CSV (For the Plotter)
    patch_audit_df = pd.DataFrame(patch_audit_data)
    filter_result_file = f"patch_audit_test_{dt_str}.csv"
    patch_csv_path = os.path.join(save_dir, filter_result_file)
    patch_audit_df.to_csv(patch_csv_path, index=False)
        
    print(f"\nAudit CSV saved: {patch_csv_path}")
    
    # print(" --- 5. EXECUTE VISUALIZATION ---")
    # # Now we call provided function using the CSV we just created
    # mygears.plot_patient_audit_grids(
    #     patch_df=patch_audit_df, 
    #     save_dir=save_dir, 
    #     max_patches=32, 
    #     config=config
    # )
    print("Rieman Filter Completed")

#%%
config = cf.get_config()
from PatchDatasetPreserve import PatchDatasetPreserve
if config.External_Patch_Audit:
    # found_files, patient_patch_count = mygears.create_cell_dataframe_debug(config)
    df = mygears.create_cell_dataframe_anerbc(config,
        glob_pattern = "*.png"
    )
    
    print(f"Number of segmented cells: {len(df)}")
    print(df.columns)
    print(df.sample(1))
    
    model_builder = ModelBuilder(model_name=model_names[0], config=config)
    preprocessing_fn = model_builder.get_preprocessing_function()
    
    external_gen = PatchDatasetPreserve(
            config,
            df=df,
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
    if platform.system() != 'Linux':
        external_gen.plot_random_batch(
            batch_idx=rand_idx,
            note=f'AneRBC-II-{config.base_dir_list[0]}',
            save_dir=f'AneRBC-II'
        )
    
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
    print(f"Processing {len(external_gen)} batches ({len(df)} total patches)...")
    
    # 3. Inference Loop (Batch-by-Batch)
    # Since shuffle=False, the order matches df_ane exactly
    for batch_idx in tqdm(range(len(external_gen)), desc="Riemannian Filtering"):
        
        # 1. Retrieve Batch Data
        # X_patches shape: (batch_size, 256, 256, 3)
        X_patches, _ = external_gen.__getitem__(batch_idx)
        
        # Identify the slice of the dataframe corresponding to this batch
        start_idx = batch_idx * config.TRAIN.BATCH_SIZE
        end_idx = min(start_idx + config.TRAIN.BATCH_SIZE, len(df))
        batch_meta = df.iloc[start_idx:end_idx]
        
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
            verbose=True
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
    filter_result_file = f"AneRBC_II_{config.base_dir_list[0]}_patch_audit_{dt_str}.csv"
    patch_csv_path = os.path.join(save_dir, filter_result_file)
    patch_audit_df.to_csv(patch_csv_path, index=False)
    print(f'Saved -> {patch_csv_path}')
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
    # Triggering existing plotter to see the AneRBC results
    mygears.plot_patient_audit_grids(
        patch_df=patch_audit_df, 
        save_dir=os.path.join(save_dir, "Visualization_Grids"), 
        max_patches=32, 
        config=config
    )
    t2 = datetime.now() - t1
    print(f"Rieman Filter Completed: time used {t2}")
    
#%%

# 1. Load the CSV file
# file_path = 'AneRBC_II_Audit_Results/AneRBC_II_Original_images_patch_audit_20251225-2118.csv'
file_path = 'AneRBC_II_Audit_Results/AneRBC_II_preprocessing_fn_patch_audit_20251225-1711.csv'
df = pd.read_csv(file_path)

# 2. Get total number of patches
total_patches = len(df)

# 3. Analyze the 'status' column
status_counts = df['status'].value_counts()
status_percentages = (status_counts / total_patches) * 100

# 4. Group into overall 'Accepted' vs 'Rejected'
accepted_mask = df['status'].str.contains('Accepted')
rejected_mask = df['status'].str.contains('REJECTED')

total_accepted = accepted_mask.sum()
total_rejected = rejected_mask.sum()

pct_accepted = (total_accepted / total_patches) * 100
pct_rejected = (total_rejected / total_patches) * 100

# 5. Print the results to plug into the manuscript
print(file_path)
print(f"Total Patches Evaluated (N): {total_patches:,}")
print("-" * 40)
print("Detailed Breakdown:")
for status, count in status_counts.items():
    pct = (count / total_patches) * 100
    print(f"  - {status}: {count:,} ({pct:.2f}%)")
print("-" * 40)
print(f"Total Accepted (X%): {total_accepted:,} ({pct_accepted:.2f}%)")
print(f"Total Rejected (Y%): {total_rejected:,} ({pct_rejected:.2f}%)")

# Total Patches Evaluated (N): 12,000
# ----------------------------------------
# Detailed Breakdown:
#   - Accepted (Standard): 11,942 (99.52%)
#   - Accepted (High Energy Rescue): 53 (0.44%)
#   - REJECTED (OOD/Extreme): 5 (0.04%)
# ----------------------------------------
# Total Accepted (X%): 11,995 (99.96%)
# Total Rejected (Y%): 5 (0.04%)


# 'AneRBC_II_Audit_Results/AneRBC_II_preprocessing_fn_patch_audit_20251225-1711.csv'
# Total Patches Evaluated (N): 12,000
# ----------------------------------------
# Detailed Breakdown:
#   - Accepted (Standard): 11,569 (96.41%)
#   - Accepted (High Energy Rescue): 418 (3.48%)
#   - REJECTED (OOD/Extreme): 13 (0.11%)
# ----------------------------------------
# Total Accepted (X%): 11,987 (99.89%)
# Total Rejected (Y%): 13 (0.11%)

#%%












































