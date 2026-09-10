# -*- coding: utf-8 -*-

import sys
import os
import platform
import gc
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from natsort import natsorted

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from yacs.config import CfgNode as CN
import tensorflow_addons as tfa

warnings.filterwarnings("ignore")

import mygears
import configsAneTunedHP_2 as cf

# ==============================================================================
# Utility & Callbacks
# ==============================================================================

class CustomTQDMProgressBar(tfa.callbacks.TQDMProgressBar):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        lr = tf.keras.backend.get_value(self.model.optimizer.learning_rate)
        logs['learning_rate'] = lr
        super().on_epoch_end(epoch, logs)

tqdm_callback = tfa.callbacks.TQDMProgressBar(show_epoch_progress=False)

def cfg_to_dict(cfg_node):
    """Recursively convert a YACS CfgNode to a nested dictionary."""
    if isinstance(cfg_node, CN):
        return {k: v for k, v in cfg_node.items()}
    return cfg_node

def count_plot(df, title, save=False):
    fig = plt.figure(dpi=300)   
    class_order = ['0', '1']
    palette = sns.color_palette("Set2", 3)
    ax = sns.countplot(x=df['label'], palette=palette, order=class_order) 
    
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')

    plt.xlabel("Class")
    plt.title(title)   
    if save:
        plot_file = f"{title}.png"
        fig.savefig(plot_file)  
        print(f"Saved {plot_file}")
    plt.show()

def plot_lr_with_cosine_overlay(history_df, train_log, lr_schedules, steps_per_epoch, save_plots=True):
    epochs = range(1, len(history_df) + 1)
    total_steps = steps_per_epoch * len(epochs)
    iteration_range = np.arange(0, total_steps, steps_per_epoch)
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    for col in [c for c in history_df.columns if c.startswith("lr_")]:
        ax.plot(epochs, history_df[col], label=f"{col} (logged)", linestyle='-', linewidth=2)
    
    for name, schedule in lr_schedules.items():
        cosine_vals = [tf.keras.backend.get_value(schedule(step)) for step in iteration_range]
        ax.plot(epochs, cosine_vals[:len(epochs)], label=f"lr_{name} (cosine)", linestyle='--')
    
    ax.set_title(f"Learning Rate Comparison:\n{train_log}")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Learning Rate")
    ax.legend()
    plt.tight_layout()
    
    if save_plots:
        os.makedirs(train_log, exist_ok=True)
        fig.savefig(os.path.join(train_log, f"{train_log}-learning-rate-comparison.png"))
        
    plt.show()

def plot_training_metrics_without_lr(history, train_log, save_plots=True):
    epochs = range(1, len(history.history['loss']) + 1)
    acc = history.history.get('accuracy')
    val_acc = history.history.get('val_accuracy')
    loss = history.history.get('loss')
    val_loss = history.history.get('val_loss')

    fig1 = plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'orange', label='Validation accuracy')
    plt.title('Training and Validation Accuracy ' + train_log)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    if save_plots:
        acc_file_path = os.path.join(train_log, f'{train_log}-accuracy.png')
        fig1.savefig(acc_file_path)

    fig = plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'orange', label='Validation loss')
    plt.title('Training and Validation Loss ' + train_log)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    if save_plots:
        plot_file_path = os.path.join(train_log, f'{train_log}-loss.png')
        fig.savefig(plot_file_path)

def plot_manual_learning_rates_with_loss(history_df: pd.DataFrame, train_log: str, include_loss: bool = False, save_plots: bool = True):
    required_lr_cols = ['epoch', 'lr_stage2', 'lr_stage3', 'lr_head']
    required_loss_cols = ['loss', 'val_loss']

    if not all(col in history_df.columns for col in required_lr_cols):
        print("Warning: history_df missing LR columns. Skipping plot.")
        return

    if include_loss and not all(col in history_df.columns for col in required_loss_cols):
        include_loss = False

    epochs = history_df['epoch']
    fig, ax1 = plt.subplots(figsize=(12, 7), dpi=300) 

    color_lr = 'tab:blue'
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Learning Rate (Log Scale)", color=color_lr)
    ax1.set_yscale('log') 
    ax1.tick_params(axis='y', labelcolor=color_lr)

    lr_cols_to_plot = ['lr_stage2', 'lr_stage3', 'lr_head']
    lines = []
    for i, col in enumerate(lr_cols_to_plot):
        lr_values = pd.to_numeric(history_df[col], errors='coerce')
        line, = ax1.plot(epochs, lr_values, label=f"{col}", linestyle='-', linewidth=2, color=plt.cm.viridis(i/len(lr_cols_to_plot)))
        lines.append(line)

    ax1.grid(True, which="both", ls=":", alpha=0.6, axis='y') 

    if include_loss:
        ax2 = ax1.twinx()  
        color_loss = 'tab:red'
        ax2.set_ylabel('Loss', color=color_loss)
        ax2.tick_params(axis='y', labelcolor=color_loss)

        loss = pd.to_numeric(history_df['loss'], errors='coerce')
        val_loss = pd.to_numeric(history_df['val_loss'], errors='coerce')

        line_loss, = ax2.plot(epochs, loss, label='Training Loss', linestyle='--', color='lightcoral')
        line_val_loss, = ax2.plot(epochs, val_loss, label='Validation Loss', linestyle='--', color='orange')
        lines.extend([line_loss, line_val_loss])

    plot_title = f"Logged Learning Rates{' & Loss' if include_loss else ''} \nLog: {train_log}"
    fig.suptitle(plot_title) 

    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best')

    plt.grid(True, which="major", ls="-", alpha=0.5, axis='x') 
    fig.tight_layout()

    if save_plots:
        try:
            os.makedirs(train_log, exist_ok=True)
            base_filename = os.path.basename(train_log)
            suffix = "-lr-loss-manual" if include_loss else "-lr-manual"
            save_path = os.path.join(train_log, f"{base_filename}{suffix}.png")
            fig.savefig(save_path)
        except Exception as e:
            print(f"Error saving plot: {e}")

    plt.show()

def print_model_summary(model):
    total_params = model.count_params()  
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])  
    non_trainable_params = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])  

    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: {non_trainable_params:,}")

def verify_images(df, image_column):
    valid_image_paths = []
    for idx, row in df.iterrows():
        image_path = row[image_column]
        try:
            img = Image.open(image_path)
            img.verify()  
            valid_image_paths.append(image_path)
        except (IOError, SyntaxError, Image.UnidentifiedImageError) as e:
            print(f"CRITICAL ERROR: Corrupted or invalid image found: {image_path}. {e}")
            sys.exit(1)
    return df[df[image_column].isin(valid_image_paths)]

def evaluate_classification_performance(true_classes, predicted_classes, save_dir: str = None, save_prefix: str = "evaluation", average: str = "macro avg", verbose: bool = True):
    cm = confusion_matrix(true_classes, predicted_classes)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    TPR = TP / (TP + FN + 1e-10)   
    TNR = TN / (TN + FP + 1e-10)   
    PPV = TP / (TP + FP + 1e-10)   
    NPV = TN / (TN + FN + 1e-10)
    FPR = FP / (FP + TN + 1e-10)
    FNR = FN / (TP + FN + 1e-10)
    FDR = FP / (TP + FP + 1e-10)
    ACC = (TP + TN) / (TP + FP + FN + TN + 1e-10)

    sen = TPR.mean()
    spec = TNR.mean()
    class_report = classification_report(true_classes, predicted_classes, output_dict=True, zero_division=0)

    p_measures = {
        'TP': TP.tolist(), 'TN': TN.tolist(), 'FP': FP.tolist(), 'FN': FN.tolist(),
        'TPR': TPR.tolist(), 'TNR': TNR.tolist(), 'PPV': PPV.tolist(), 'NPV': NPV.tolist(),
        'FPR': FPR.tolist(), 'FNR': FNR.tolist(), 'FDR': FDR.tolist(), 'ACC': ACC.tolist(),
        'precision': class_report[average]['precision'], 'f1-score': class_report[average]['f1-score'],
        'MeanAcc': ACC.mean(), 'MeanSen': sen, 'MeanSpec': spec,
    }

    if verbose:
        print("\n=== Classification Performance Summary ===")
        print(f"Sensitivity (Mean TPR): {sen:.3f}")
        print(f"Specificity (Mean TNR): {spec:.3f}")
        print(f"Accuracy (Mean):       {p_measures['MeanAcc']:.3f}")
        print(f"Precision (Macro):     {p_measures['precision']:.3f}")
        print(f"F1-Score (Macro):      {p_measures['f1-score']:.3f}")
        print("------------------------------------------------")
        print(f"Confusion Matrix:\n{cm}")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{save_prefix}_p_measures.json")
        with open(save_path, "w") as f:
            json.dump(p_measures, f, indent=4)

    return p_measures, cm

def plot_confusion_matrices(confusionmatrix, p_measures, config, train_log, save_prefix: str = None):
    ACC_mean = np.mean(p_measures['ACC'])
    TPR_mean = np.mean(p_measures['TPR'])
    TNR_mean = np.mean(p_measures['TNR'])
    class_labels = config['DATA']['CLASS_LABELS']

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    sns.heatmap(confusionmatrix, cmap='Reds', annot=True, cbar=True, fmt="d", xticklabels=class_labels, yticklabels=class_labels, annot_kws={'size': 14, 'weight': 'bold'}, ax=ax)
    ax.set_ylabel('Actual', fontsize=14)
    ax.set_xlabel(f"Predicted\nAccuracy={ACC_mean:.2f}\nSensitivity={TPR_mean:.2f}\nSpecificity={TNR_mean:.2f}", fontsize=14)
    plt.tight_layout()
    plt.show()

    if config.SAVE:
        file_path = os.path.join(train_log, f"{save_prefix}-{train_log}-confuseMatrix.png")
        fig.savefig(file_path)

    confusionmatrix_normalized = confusionmatrix.astype('float') / confusionmatrix.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    sns.heatmap(confusionmatrix_normalized, cmap='Reds', annot=True, cbar=True, fmt=".2f", xticklabels=class_labels, yticklabels=class_labels, annot_kws={'size': 14, 'weight': 'bold'}, ax=ax)
    ax.set_ylabel('Actual', fontsize=14)
    ax.set_xlabel(f"Predicted\nAccuracy={ACC_mean:.2f}\nSensitivity={TPR_mean:.2f}\nSpecificity={TNR_mean:.2f}", fontsize=14)
    plt.tight_layout()
    plt.show()

    if config.SAVE:
        file_path = os.path.join(train_log, f"{save_prefix}-{train_log}-confuseMatrixNorm.png")
        fig.savefig(file_path)

def stratified_subsample(df, n_samples=30_000, random_state=1337):
    y = df['label']
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_samples, random_state=random_state)
    idx, _ = next(sss.split(df, y))
    return df.iloc[idx]

def predict_full_dataset(model, datagen, device: str = '/device:GPU:0', verbose: int = 1):
    print(f"\n[INFO] Starting prediction on {len(datagen.df)} samples ({datagen.__len__()} batches)")
    t_start = datetime.now()

    with tf.device(device):
        y_pred_all = model.predict(datagen, verbose=verbose)

    total_samples = len(datagen.df)
    y_pred_all = y_pred_all[:total_samples]
    y_pred_argmax = np.argmax(y_pred_all, axis=1)
    elapsed_time = datetime.now() - t_start

    print("\n[INFO] Prediction complete.")
    if len(datagen.classes) != len(y_pred_argmax):
         print(f"Warning: Length mismatch after prediction! Expected {len(datagen.classes)} samples, got {len(y_pred_argmax)}.")

    return y_pred_all, y_pred_argmax, elapsed_time

def reset_vram():
    tf.keras.backend.clear_session()
    gc.collect()

def create_anerbc_dataframe(base_dir_list, glob_pattern="**/*.png"):
    all_patch_files = []
    for base in base_dir_list:
        base_path = Path(base)
        patch_files = natsorted(list(base_path.glob(glob_pattern)))
        all_patch_files.extend(patch_files)

    if not all_patch_files:
        return pd.DataFrame()

    data_records = []
    for p in all_patch_files:
        parts = p.parts
        filename = p.name
        
        category = next((token for token in parts if token in ["Healthy_individuals", "Diseased_individuals"]), "Unknown")
        label = 0 if category == "Healthy_individuals" else 1 if category == "Anemic_individuals" else -1
        patient_id = p.parent.name

        data_records.append({
            "base_dir": parts[0],
            "label": label,
            "patient_id": patient_id,
            "slide_id": category,
            "cell_id": filename,
            "patch_path": p
        })

    return pd.DataFrame(data_records)

# ==============================================================================
# Main Execution Pipeline
# ==============================================================================

config = cf.get_config()
from ModelBuilder import ModelBuilder, MetricsTrackerCosineDecay

t1 = datetime.now()  

dataset_root = config['DATASET']
class_labels = config['DATA']['CLASS_LABELS']

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
skf = StratifiedKFold(n_splits=config.DATA.N_SPLIT,
                      shuffle=True,
                      random_state=config.DATA.SEED)

train_df_fold_list = []
val_df_fold_list = []
val_fold_patients_list = []

for fold, (train_idx, val_idx) in enumerate(skf.split(train_patient_labels.index, train_patient_labels.values)):
    train_fold_patients = train_patient_labels.index[train_idx]
    val_fold_patients   = train_patient_labels.index[val_idx]
    val_fold_patients_list.append(val_fold_patients)

    train_df_fold_list.append(train_df[train_df['patient_id'].isin(train_fold_patients)])
    val_df_fold_list.append(train_df[train_df['patient_id'].isin(val_fold_patients)])

train_patient_labels = train_df.groupby('patient_id')['label'].first()
sss = StratifiedShuffleSplit(n_splits=config.DATA.N_SPLIT, test_size=config.DATA.val_ratio, random_state=config.DATA.SEED)

train_df_fold_list = []
val_df_fold_list = []
val_fold_patients_list = []

for fold, (train_idx, val_idx) in enumerate(sss.split(train_patient_labels.index, train_patient_labels.values)):
    train_fold_patients = train_patient_labels.index[train_idx]
    val_fold_patients   = train_patient_labels.index[val_idx]
    val_fold_patients_list.append(val_fold_patients)
    
    train_df_fold_list.append(train_df[train_df['patient_id'].isin(train_fold_patients)])
    val_df_fold_list.append(train_df[train_df['patient_id'].isin(val_fold_patients)])

train_patches, val_patches = [], []
for fold in range(len(train_df_fold_list)):
    train_patches.append(len(train_df_fold_list[fold]))
    val_patches.append(len(val_df_fold_list[fold]))

folds = np.arange(len(train_patches))
fig, ax = plt.subplots(figsize=(14,8), dpi=300)
bar_width = 0.25

train_bars = plt.bar(folds - bar_width/2, train_patches, width=bar_width, label="Train Patches", color="#1f77b4")
val_bars   = plt.bar(folds + bar_width/2, val_patches, width=bar_width, label="Val Patches", color="#ff7f0e")

for bars in [train_bars, val_bars]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height, f"{height:,}", ha="center", va="bottom", fontsize=20, rotation=45)

plt.xticks(folds, [f"Fold {i}" for i in folds], fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel("Patch Count", fontsize=20)
plt.xlabel("Fold", fontsize=20)
plt.legend(fontsize=20, loc='lower right')
plt.tight_layout()

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fname = os.path.join(config.BASEPATH, f"data2025-5-fold-distribution-patches-{config.DATA.SEED}.png")
fig.savefig(fname)

builder = ModelBuilder(model_name=config.MODEL.NAMES[0], config=config)
preprocessing_fn = builder.get_preprocessing_function()       

from PatchDatasetPreserve import PatchDatasetPreserve
test_df = test_df.sample(frac=1, random_state=config.DATA.SEED).reset_index(drop=True)
test_df_sampled = stratified_subsample(test_df, config.TRAIN.SAMPLE_SIZE_TEST) if config.TRAIN.SAMPLE else test_df

test_gen = PatchDatasetPreserve(
    config,
    df=test_df_sampled,
    batch_size=config.TRAIN.BATCH_SIZE,
    preprocessing_function=preprocessing_fn,
    shuffle=False,
)  

train_gen_list = []
val_gen_list = []
class_weight_list = []

train_folds = config.TRAIN.FOLDS
for fold in train_folds: 
    train_df_fold = train_df_fold_list[fold].copy()
    val_df_fold   = val_df_fold_list[fold].copy()

    count_plot(train_df_fold, f"Train Set Fold {fold} - {len(train_df_fold)}")
    count_plot(val_df_fold, f"Validation Set Fold {fold} - {len(val_df_fold)}")   
           
    if platform.system() == 'Linux' and config['DATA']['VERIFY']:
        train_df_fold = verify_images(train_df_fold, 'patch_path')
        val_df_fold = verify_images(val_df_fold, 'patch_path')
        
    models_per_fold = []
    for model_name in config['MODEL']['NAMES']:    
        train_log = config.BASE + '-' + f"Data2025-SEED-{config.DATA.SEED}-Fold-{fold}-cellseg-{model_name}-classweights-{config['TRAIN']['DATETIME']}"
        
        if config['SAVE'] == True:            
            os.makedirs(train_log, exist_ok=True)
            cfg_dict = cfg_to_dict(config)
            json_file_path = os.path.join(train_log,'_config.json')  
            with open(json_file_path, 'w') as json_file:    
                json.dump(cfg_dict, json_file, indent=4)            
                                        
        train_df_fold = train_df_fold.sample(frac=1, random_state=config.DATA.SEED).reset_index(drop=True)
        train_df_fold_sampled = stratified_subsample(train_df_fold, config['TRAIN']['SAMPLE_SIZE']) if config['TRAIN']['SAMPLE'] else train_df_fold
        
        train_gen = PatchDatasetPreserve(
            config,
            df=train_df_fold_sampled,
            batch_size=config.TRAIN.BATCH_SIZE,
            shuffle=True,
        )

        rand_idx = np.random.randint(0, len(train_gen)) 
        if fold == 0:
            train_gen.plot_random_batch(rand_idx, note=f'Train fold {fold}', save_dir=train_log)
        
        val_df_fold = val_df_fold.sample(frac=1, random_state=config.DATA.SEED).reset_index(drop=True)
        val_df_fold_sampled = stratified_subsample(val_df_fold, config['TRAIN']['SAMPLE_SIZE_VAL']) if config['TRAIN']['SAMPLE'] else val_df_fold
        
        val_gen = PatchDatasetPreserve(
            config,
            df=val_df_fold_sampled,
            batch_size=config.TRAIN.BATCH_SIZE,
            shuffle=False,
        )

        rand_idx = np.random.randint(0, len(val_gen))                
        if fold == 0:
            val_gen.plot_random_batch(rand_idx, note=f'Val fold {fold}', save_dir=train_log)  
        
        train_gen_list.append(train_gen)
        val_gen_list.append(val_gen)
        
        labels = np.unique(train_df_fold['label'])  
        class_weights = compute_class_weight(class_weight='balanced', classes=labels, y=train_df_fold['label'])
        class_weight_dict = dict(enumerate(class_weights))
        class_weight_list.append(class_weight_dict)
        
        #%%      
        if config.TRAIN.TL:
            if model_name == 'ConvNeXtLarge':
                backup_model_best_for_TL = os.path.join(config.BASEMODEL_PATH, 'Linux-AneRBC-ConvNeXtLarge-BGR-20250327-1542.hdf5')
            
            from keras.utils import custom_object_scope
            from keras.applications.convnext import LayerScale
            with custom_object_scope({'LayerScale': LayerScale}):
                loaded_model = tf.keras.models.load_model(backup_model_best_for_TL)  
            print_model_summary(loaded_model)   
            
        #%%          
        from ModelBuilder import CustomConvNeXtWithHeadL2_Bag

        model_builder = CustomConvNeXtWithHeadL2_Bag(
            model_name=model_name,
            input_shape=(None,) + config.DATA.DIMENSION,  # <-- FIX: Prepend (None,) for the bag dimension
            num_classes=config.MODEL.NUM_CLASSES,
            dense_units=[128, 64, 32, 16, 8],
            activation=config.TRAIN.ACTIVATION,
            dropout_rate=config.TRAIN.DROPOUT,
        )
        
        model, base_model = model_builder.build_model(freeze_backbone=True, add_l2=True)
        if config.TRAIN.TL:
            model_builder.copy_weights_from(loaded_model, max_layer_index=296)
            del loaded_model
            reset_vram()
            
        print_model_summary(model)
        
        #%%        
        class GradualUnfreeze(tf.keras.callbacks.Callback):
            def __init__(self, model, unfreeze_schedule: dict, optimizer_spec=None, factor=0.7, patience=4, min_lr=1e-7, warm_restart_factor=1.2, verbose=True):
                super().__init__()
                self.model = model
                self.verbose = verbose
                self.unfreeze_schedule = dict(sorted(unfreeze_schedule.items()))
                self.current_min_index = None  
                self.last_val_loss = float('inf')
                self.wait = 0
                self.optimizer_spec = optimizer_spec
                self.factor = factor
                self.patience = patience
                self.min_lr = min_lr
                self.warm_restart_factor = warm_restart_factor
                self.last_unfreeze_epoch = -1
        
            def on_epoch_begin(self, epoch, logs=None):
                min_index = 0
                for start_epoch, idx in self.unfreeze_schedule.items():
                    if epoch >= start_epoch: min_index = idx
                    else: break
        
                if self.current_min_index != min_index:
                    self.current_min_index = min_index
                    self.last_unfreeze_epoch = epoch
        
                    for i, layer in enumerate(self.model.layers):
                        layer.trainable = (i >= min_index)
        
                    if self.optimizer_spec is not None:
                        for opt, _ in self.optimizer_spec:
                            try:
                                lr_obj = opt.learning_rate
                                if isinstance(lr_obj, tf.keras.optimizers.schedules.LearningRateSchedule):
                                    pass
                                else:
                                    lr = float(tf.keras.backend.get_value(lr_obj))
                                    new_lr = lr * self.warm_restart_factor
                                    tf.keras.backend.set_value(lr_obj, new_lr)
                            except Exception:
                                pass

            def on_epoch_end(self, epoch, logs=None):
                if logs is None or 'val_loss' not in logs or self.optimizer_spec is None:
                    return
        
                current_val_loss = logs['val_loss']
                if current_val_loss < self.last_val_loss:
                    self.last_val_loss = current_val_loss
                    self.wait = 0
                else:
                    self.wait += 1
        
                if self.wait >= self.patience:
                    self._reduce_lr_all()
                    self.wait = 0
            
            def _reduce_lr_all(self):
                if not hasattr(self, "optimizer_spec") or self.optimizer_spec is None: return
                for opt, _ in self.optimizer_spec:
                    try:
                        lr_attr = opt.learning_rate
                        if hasattr(lr_attr, "assign"):
                            old_lr = float(tf.keras.backend.get_value(lr_attr))
                            new_lr = max(old_lr * self.factor, self.min_lr)
                            tf.keras.backend.set_value(lr_attr, new_lr)
                        elif isinstance(lr_attr, (float, int)):
                            new_lr = max(lr_attr * self.factor, self.min_lr)
                            opt.learning_rate = new_lr
                    except Exception:
                        pass

        steps_per_epoch = int(len(train_df_fold_sampled) / config['TRAIN']['BATCH_SIZE'])
        stage2_layers = model.layers[53:270]   
        stage3_layers = model.layers[270:294]  
        head_layers   = model.layers[295:]     
        
        stage2_lr, stage3_lr, head_lr = config.TRAIN.base_lr
        base_lrs = {"stage2": stage2_lr, "stage3": stage3_lr, "head": head_lr}        
        mode = config.TRAIN.lr_mode
        
        if mode == "cosine":
            alpha = 0.6       
            t_mul = 2.0       
            m_mul = 0.9       
            lr_stage2 = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=base_lrs["stage2"], first_decay_steps=steps_per_epoch * 25, t_mul=t_mul, m_mul=m_mul, alpha=alpha)
            lr_stage3 = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=base_lrs["stage3"], first_decay_steps=steps_per_epoch * 15, t_mul=t_mul, m_mul=m_mul, alpha=alpha)
            lr_head = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=base_lrs["head"], first_decay_steps=steps_per_epoch * 40, t_mul=t_mul, m_mul=m_mul, alpha=alpha) 
        elif mode == "manual":
            lr_stage2 = tf.Variable(base_lrs["stage2"], trainable=False, dtype=tf.float32)
            lr_stage3 = tf.Variable(base_lrs["stage3"], trainable=False, dtype=tf.float32)
            lr_head   = tf.Variable(base_lrs["head"],   trainable=False, dtype=tf.float32)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        
        multi_optimizer_config = {
            "stage2": {"lr": lr_stage2, "layers": stage2_layers},
            "stage3": {"lr": lr_stage3, "layers": stage3_layers},
            "head":   {"lr": lr_head,   "layers": head_layers},
        }

        new_weight_decay = 1e-4 
        opt_stage2 = tf.keras.optimizers.RMSprop(learning_rate=lr_stage2, weight_decay=new_weight_decay) 
        opt_stage3 = tf.keras.optimizers.RMSprop(learning_rate=lr_stage3, weight_decay=new_weight_decay) 
        opt_head   = tf.keras.optimizers.RMSprop(learning_rate=lr_head,   weight_decay=new_weight_decay) 
                
        optimizers = tfa.optimizers.MultiOptimizer([
            (opt_stage2, stage2_layers),
            (opt_stage3, stage3_layers),
            (opt_head,   head_layers),
        ])
        
        optimizer_spec = [
            (opt_stage2, stage2_layers),
            (opt_stage3, stage3_layers),
            (opt_head,   head_layers),
        ]
            
        model.compile(
            optimizer=optimizers,
            loss="binary_crossentropy",
            metrics=["accuracy"],
            weighted_metrics=["accuracy"]
        )
        
        unfreeze_schedule = {
            0: 294,   
            10: 270,  
            25: 53,   
            80: 0     
        } 
        
        if mode == "manual":
            gradual_unfreeze_cb = GradualUnfreeze(
                model=model, unfreeze_schedule=unfreeze_schedule, optimizer_spec=optimizer_spec,
                factor=config.TRAIN.LR_FACTOR, patience=config.TRAIN.Reduce_LR_patience, min_lr=1e-7, warm_restart_factor=1.2 
            )
        else:  
            gradual_unfreeze_cb = GradualUnfreeze(
                model=model, unfreeze_schedule=unfreeze_schedule, optimizer_spec=optimizer_spec, warm_restart_factor=1.2
            )
        
        if config['TRAIN']['Enable']:
            t_train_start = datetime.now()
            backup_model_best = os.path.join(config.MODEL_PATH, f'{train_log}.hdf5')    

            early_stopping_callbacks = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',  
                patience=config['TRAIN']['stop_patience'],  
                restore_best_weights=True,
                verbose=config.verbose
            )
        
            mcp2 = ModelCheckpoint(
                filepath=backup_model_best, monitor='val_loss', verbose=config.verbose,
                save_best_only=True, mode='min', save_weights_only=True  
            )
    
            callbacks_list = [
                    mcp2, early_stopping_callbacks, tqdm_callback, gradual_unfreeze_cb,
            ]
            
            if config['SAVE']:
                csv_file_path = os.path.join(train_log, f"{train_log}.csv") 
                metrics_tracker = MetricsTrackerCosineDecay(csv_file_path, optimizer_config=multi_optimizer_config)
                callbacks_list.append(metrics_tracker)
                
            if platform.system() == 'Darwin' or platform.system() == 'Windows':
                from livelossplot import PlotLossesKeras
                callbacks_list.append(PlotLossesKeras())
            
            #%%
            with tf.device('/device:GPU:0'):
                history = model.fit(
                    train_gen,  
                    validation_data = val_gen,
                    epochs = config['TRAIN']['EPOCHS'],  
                    verbose = config.TRAIN.verbose,
                    class_weight=class_weight_dict,
                    callbacks = callbacks_list,
                    workers= 8 if platform.system() == 'Linux' else 0,  
                    use_multiprocessing=True if platform.system() == 'Linux' else False,
                    )
                
            t_train_end = datetime.now() - t_train_start
            models_per_fold.append(model)

        if config.SAVEHIST and config.TRAIN.Enable:
            model_history_df = pd.DataFrame(history.history)                      
            history_file = f'{train_log}_history_df.csv'
            os.makedirs(train_log, exist_ok=True)
            with open(os.path.join(train_log, history_file), mode='w') as f:
                model_history_df.to_csv(f)  
                
            plot_training_metrics_without_lr(history=history, train_log=train_log) 
            
            if mode == 'cosine':
                pd.set_option('display.float_format', '{:.10e}'.format)
                history_df = pd.read_csv(csv_file_path)            
                plot_lr_with_cosine_overlay(
                    history_df=history_df, train_log=train_log,
                    lr_schedules={"stage2": lr_stage2, "stage3": lr_stage3, "head": lr_head}, steps_per_epoch=steps_per_epoch
                )
                plot_manual_learning_rates_with_loss(history_df=history_df, train_log=train_log, include_loss=True)
            
            if mode == 'manual':
                pd.set_option('display.float_format', '{:.10e}'.format)
                history_df = pd.read_csv(csv_file_path)   
                plot_manual_learning_rates_with_loss(history_df=history_df, train_log=train_log, include_loss=False)
                plot_manual_learning_rates_with_loss(history_df=history_df, train_log=train_log, include_loss=True)
            
            
            if config.TRAIN.Evaluate_Val:          
               if config.PILOT:
                    if platform.system() == 'Windows':  
                        backup_model_best = "D:\THL-g\models\Linux-Fold-0-cellseg-stage2-ConvNeXtLarge-classweights-20251228-2306.hdf5"
                    # else:
                    #     backup_model_best = 'models/Linux-Fold-0-stage2-ConvNeXtLarge-bal-aug-20251023-1259.hdf5'
                    
               best_model = model_builder.build_model(freeze_backbone=True, add_l2=True)
               if 'ConvNeXt' in backup_model_best:
                   with custom_object_scope({'LayerScale': LayerScale}):
                       best_model.load_weights(backup_model_best)
               else:
                   best_model.load_weights(backup_model_best)
            
               print_model_summary(best_model)
            
               best_model.compile(optimizer=optimizers, loss='binary_crossentropy', metrics=['accuracy'], weighted_metrics=["accuracy"])
               scores = best_model.evaluate(val_gen, verbose=config.verbose)
            
               for metric, value in zip(best_model.metrics_names, scores):
                   print("mean {}: {:.2}".format(metric, value))
            
               y_pred_all, y_pred_argmax, elapsed_time = predict_full_dataset(model=best_model, datagen=val_gen, verbose=config.verbose)
               print(classification_report(y_true=val_gen.classes, y_pred=y_pred_argmax, target_names=config['DATA']['CLASS_LABELS']))
               
               p_measures, cm = evaluate_classification_performance(
                   true_classes=val_gen.classes, predicted_classes=y_pred_argmax, save_dir=train_log, save_prefix=str('Val-' + train_log), verbose=True
               )
               plot_confusion_matrices(cm, p_measures, config, train_log, save_prefix='Val')

               if config.TRAIN.Evaluate_Test:
                   scores_test = best_model.evaluate(test_gen, verbose=config.verbose)
                   for metric, value in zip(best_model.metrics_names, scores_test):
                       print("mean {}: {:.2}".format(metric, value))
                
                   y_pred_all_test, y_pred_argmax_test, elapsed_time = predict_full_dataset(model=best_model, datagen=test_gen, verbose=config.verbose)
                   if len(test_gen.classes) == len(y_pred_argmax_test):
                       print(classification_report(y_true=test_gen.classes, y_pred=y_pred_argmax_test, target_names=config['DATA']['CLASS_LABELS']))
                       p_measures_test, cm_test = evaluate_classification_performance(
                           true_classes=test_gen.classes, predicted_classes=y_pred_argmax_test, save_dir=train_log, save_prefix=str('Test-' + train_log), verbose=True
                       )
                       plot_confusion_matrices(cm_test, p_measures_test, config, train_log, save_prefix='Test')
               
               if 'best_model' in locals():
                   del best_model
               if 'best_model' in globals():
                   del best_model
               reset_vram()

        if 'model' in locals() or 'model' in globals():
            del model
            reset_vram()
            
#%%
if config.Evaluate_Test:
    from sklearn.covariance import EmpiricalCovariance
    
    def calibrate_euclidean_mahalanobis(feature_matrix):
        """
        Fits the Euclidean Mahalanobis parameters using extracted training features.
        
        Args:
            feature_matrix: Numpy array of shape (Num_Patches, Feature_Dim). 
                            For ConvNeXtLarge, Feature_Dim is 1536.
        Returns:
            mu: Mean feature vector.
            prec: Inverse covariance matrix (Precision matrix).
        """
        print("Fitting Empirical Covariance Matrix...")
        # Use sklearn to calculate covariance robustly (handles numerical instability)
        cov_estimator = EmpiricalCovariance(assume_centered=False)
        cov_estimator.fit(feature_matrix)
        
        mu = cov_estimator.location_
        prec = cov_estimator.precision_ # This is \Sigma^{-1}
        
        return mu, prec
    
    def apply_euclidean_qc_gate(bag_features, mu, prec, threshold):
        """
        Filters a bag of patch features using Euclidean Mahalanobis distance.
        
        Args:
            bag_features: tf.Tensor of shape (Num_Patches, Feature_Dim)
            mu: Numpy array (Feature_Dim,)
            prec: Numpy array (Feature_Dim, Feature_Dim)
            threshold: Float threshold (\tau_limit)
            
        Returns:
            filtered_features: tf.Tensor of patches that passed QC
        """
        # Convert numpy parameters to tensors
        mu_tensor = tf.cast(tf.constant(mu), tf.float32)
        prec_tensor = tf.cast(tf.constant(prec), tf.float32)
        
        # Center the features: (x - \mu)
        centered_features = bag_features - mu_tensor # Shape: (N, d)
        
        # Compute Mahalanobis distance: sqrt((x-\mu)^T \Sigma^{-1} (x-\mu))
        # Matrix multiplication: (N, d) @ (d, d) -> (N, d)
        left_term = tf.matmul(centered_features, prec_tensor) 
        
        # Dot product with itself for each patch: sum(left_term * centered_features, axis=1)
        squared_dist = tf.reduce_sum(left_term * centered_features, axis=1)
        mahalanobis_dist = tf.sqrt(tf.maximum(squared_dist, 1e-9)) # Shape: (N,)
        
        # Create binary mask: True if distance < threshold
        mask = mahalanobis_dist < threshold
        
        # Filter the bag
        filtered_features = tf.boolean_mask(bag_features, mask)
        
        return filtered_features
    
    #%% --- Euclidean Ensemble Configuration & Setup ---
    import csv
    
    # ---------------------------------------------------------
    # 1. Euclidean Workflow Configuration
    # ---------------------------------------------------------
    EUCLIDEAN_WORKFLOW_CONFIG = {
        "FOLDS": range(config.DATA.N_SPLIT), # Evaluate across all 5 folds
        
        # Triage Thresholds (M3) - Use exact thresholds as M6 for fair comparison
        "THRESHOLDS": {
            "CONFIDENT_THL": 0.5463,  
            "CONFIDENT_IDA": 0.4900   
        }
    }
    
    euclidean_backbones_map = {}
    euclidean_attention_map = {}
    euclidean_calib_map = {} # To store mu, prec, and tau_limit per fold
    
    #%% --- Phase 1: 5-Fold Euclidean QC Calibration ---
    print("\n" + "="*50 + "\nCALIBRATING EUCLIDEAN MAHALANOBIS GATE (5 FOLDS)\n" + "="*50)
    
    for fold in EUCLIDEAN_WORKFLOW_CONFIG["FOLDS"]:
        print(f"\n--- Calibrating Fold {fold} ---")
        
        # 1. Load the trained Euclidean components for this fold
        euclidean_backbones_map[fold] = load_euclidean_extractor(fold, config.BASEMODEL_PATH)
        euclidean_attention_map[fold] = load_attention_mil_head(fold, config.MODEL_PATH)
          
        # 2. Extract Training Features (Calculate mu & prec)
        print("  > Extracting training features...")
        train_features_list = []
        for idx, row in train_df_fold_list[fold].iterrows():
            patch = preprocessing_fn(cv2.imread(row['patch_path']))
            patch = np.expand_dims(patch, axis=0) 
            train_features_list.append(euclidean_backbones_map[fold].predict(patch, verbose=0)[0])
            
        train_matrix = np.array(train_features_list)
        mu, prec = calibrate_euclidean_mahalanobis(train_matrix)
        
        # 3. Extract Validation Features (Calibrate tau_limit)
        print("  > Extracting validation features for threshold calibration...")
        val_features_list = []
        for idx, row in val_df_fold_list[fold].iterrows():
            patch = preprocessing_fn(cv2.imread(row['patch_path']))
            patch = np.expand_dims(patch, axis=0) 
            val_features_list.append(euclidean_backbones_map[fold].predict(patch, verbose=0)[0])
            
        val_matrix = np.array(val_features_list)
        
        # Compute distances to find 95th percentile
        mu_tensor = tf.cast(tf.constant(mu), tf.float32)
        prec_tensor = tf.cast(tf.constant(prec), tf.float32)
        centered_val = val_matrix - mu_tensor
        left_term = tf.matmul(centered_val, prec_tensor)
        val_distances = tf.sqrt(tf.maximum(tf.reduce_sum(left_term * centered_val, axis=1), 1e-9)).numpy()
        
        tau_limit = np.percentile(val_distances, 95)
        print(f"  [Fold {fold} Calibrated] TAU_LIMIT: {tau_limit:.4f}")
        
        # 4. Store fold calibration data
        euclidean_calib_map[fold] = {
            'mu': mu,
            'prec': prec,
            'tau_limit': tau_limit
        }
    
    #%% --- Phase 2: Euclidean Ensemble Inference (M1, M2, M3) ---
    def predict_euclidean_workflow(patches, backbone, attention_head, mu, prec, tau_limit, apply_qc=True):
        """Processes a patient bag through a single Euclidean fold pipeline."""
        raw_features = backbone(patches, training=False) 
        
        if apply_qc:
            clean_features = apply_euclidean_qc_gate(raw_features, mu, prec, tau_limit)
        else:
            clean_features = raw_features
            
        n_raw = tf.shape(raw_features)[0].numpy()
        n_clean = tf.shape(clean_features)[0].numpy()
        
        if n_clean == 0:
            return None, 0, n_raw
            
        bag_features = tf.expand_dims(clean_features, axis=0)
        patient_prediction = attention_head(bag_features, training=False) 
        
        prob_thl = patient_prediction[0, 1].numpy()
        return prob_thl, n_clean, n_raw
    
    print(f"\n{'='*50}\nSTARTING 5-FOLD EUCLIDEAN INFERENCE (M1, M2, M3)\n{'='*50}")
    
    # Data Preparation
    from FullBagDataset import FullBagDataset
    inference_gen = FullBagDataset(
        config,
        df=test_df_sampled, # Independent Test Set
        preprocessing_function=preprocessing_fn,
        expect_rgba=True,
        bg_color=(0, 0, 0),
        mode='test',
    )
    
    euclidean_results_data = []  
    
    for patient_idx in range(len(inference_gen)):
        p_ids, labels_raw, X_bag_batch, labels_onehot = inference_gen.__getitem__(
            idx=patient_idx, get_pids=True
        )
        current_pid = p_ids[0]
        true_label = int(labels_raw[0])      
        patches = X_bag_batch[0] 
        
        print(f"\n--- Processing {patient_idx+1}/{len(inference_gen)}: {current_pid} (True: {true_label}) ---")
    
        m1_fold_probs = []
        m2_fold_probs = []
        m2_clean_counts = []
        n_raw_total = patches.shape[0]
    
        # Iterate through all 5 folds for this specific patient
        for fold in EUCLIDEAN_WORKFLOW_CONFIG["FOLDS"]:
            backbone = euclidean_backbones_map[fold]
            attention = euclidean_attention_map[fold]
            calib = euclidean_calib_map[fold]
    
            # M1: Vanilla Euclidean (No QC)
            prob_m1, _, _ = predict_euclidean_workflow(
                patches, backbone, attention, calib['mu'], calib['prec'], calib['tau_limit'], apply_qc=False
            )
            m1_fold_probs.append(prob_m1)
            
            # M2: Euclidean + QC
            prob_m2, n_clean_m2, _ = predict_euclidean_workflow(
                patches, backbone, attention, calib['mu'], calib['prec'], calib['tau_limit'], apply_qc=True
            )
            if prob_m2 is not None:
                m2_fold_probs.append(prob_m2)
                m2_clean_counts.append(n_clean_m2)
    
        # -----------------------------------------------------
        # Ensemble Aggregation (Average the fold probabilities)
        # -----------------------------------------------------
        final_prob_m1 = np.mean(m1_fold_probs)
        
        if m2_fold_probs:
            final_prob_m2 = np.mean(m2_fold_probs)
            avg_clean_m2 = int(np.mean(m2_clean_counts))
        else:
            final_prob_m2 = None
            avg_clean_m2 = 0
    
        # -----------------------------------------------------
        # Tri-State Decision Logic (M3 Abstention on M2 Ensemble)
        # -----------------------------------------------------
        t_upper = EUCLIDEAN_WORKFLOW_CONFIG["THRESHOLDS"]["CONFIDENT_THL"]
        
        if final_prob_m2 is None:
            diagnosis_m3 = "QC Failed (All patches dropped)"
            status_m3 = "ABSTAIN"
            pred_class_m3 = -1
        elif final_prob_m2 >= t_upper:
            diagnosis_m3 = "THL"
            status_m3 = "POSITIVE (High Confidence)"
            pred_class_m3 = 1
        else:
            diagnosis_m3 = "Screen Negative (Likely IDA)"
            status_m3 = "NEGATIVE (Rule-Out)"
            pred_class_m3 = 0
    
        print(f"   > M1 (No QC Ensemble) Prob: {final_prob_m1:.4f}")
        print(f"   > M2 (w/ QC Ensemble) Prob: {final_prob_m2:.4f} | Avg Retained Patches: {avg_clean_m2}/{n_raw_total}")
        print(f"   > M3 (Triage Ensemble) Result: {diagnosis_m3}")
    
        # -----------------------------------------------------
        # Logging
        # -----------------------------------------------------
        euclidean_results_data.append({
            "patient_id": current_pid,
            "true_label": true_label,
            
            # M1 Metrics
            "M1_prob_thl": final_prob_m1,
            "M1_pred_class": 1 if final_prob_m1 >= 0.5 else 0,
            
            # M2 Metrics
            "M2_prob_thl": final_prob_m2,
            "M2_pred_class": 1 if (final_prob_m2 is not None and final_prob_m2 >= 0.5) else 0,
            
            # M3 Metrics (with Triage)
            "M3_predicted_class": pred_class_m3,
            "M3_diagnosis_text": diagnosis_m3,
            "M3_status": status_m3,
            
            # QC Stats
            "total_patches": n_raw_total,
            "avg_accepted_patches_M2": avg_clean_m2,
            "avg_rejected_patches_M2": n_raw_total - avg_clean_m2 if final_prob_m2 is not None else n_raw_total,
            "rejection_rate": f"{((n_raw_total - avg_clean_m2) / n_raw_total):.4f}" if final_prob_m2 is not None else "1.0000"
        })
    
    # ---------------------------------------------------------
    # 7. Save Output
    # ---------------------------------------------------------
    if euclidean_results_data:
        csv_path = os.path.join(config.SAVE_PATH, "Euclidean_Ensemble_Ablation_M1_M2_M3_Results.csv")
        try:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=euclidean_results_data[0].keys())
                writer.writeheader()
                writer.writerows(euclidean_results_data)
            print(f"\n[IO] Euclidean Ensemble Ablation Report saved: {csv_path}")
        except Exception as e:
            print(f"[Error] {e}")
            
#%%
t2 = datetime.now() - t1
print('\nAll execution time: ', t2)

#%%
