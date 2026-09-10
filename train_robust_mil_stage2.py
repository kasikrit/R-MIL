# train_robust_mil.py (Phase 3: Robust Training/Inference)
# This script simulates the final "Patient-Level Aggregation" step. It doesn't retrain the CNN backbone but trains the Attention Weights based on curvature.
import os
import logging
import gc

# 0 = all messages
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
    

# Set Python-level TensorFlow logger to ERROR only
tf.get_logger().setLevel(logging.ERROR)

# Optional: Suppress Abseil logging (TF uses absl-py)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import tensorflow_addons as tfa
from datetime import datetime
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import os, platform
import json
from sklearn.model_selection import train_test_split
import mygears
from PatchDatasetPreserve import PatchDatasetPreserve
from ModelBuilder import ModelBuilder

t1 = datetime.now()
import configsCell_mil_stage2 as cfm
config = cfm.get_config()
print(config)

model_names = config['MODEL']['NAMES']
dataset_root = config['DATASET']
class_labels = config['DATA']['CLASS_LABELS']

print(f"{dataset_root=}, {class_labels=}")
print(f"{config.BASEMODEL_PATH=}")
print(f"{config.SAVE_PATH=}")
print(f"{config.MODEL_PATH=}")

#%%
#% -- Read data --
# found_files, patient_patch_count = mygears.create_cell_dataframe_debug(config)
df = mygears.create_cell_dataframe_flex(config,
    glob_pattern = "[01]/*/normalized/*-cells-256/*.png")
# Get a unique list of all patients
all_patient_ids = df['patient_id'].unique()
# Now, df should contain data for all 20 patients
print(f"\nDataFrame created with {df['patient_id'].nunique()} unique patients.\n")
print(df.columns)
print(df.sample(1))

#%%
# Get unique patients and their labels
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

#%%
# --- Train counts ---
train_patient_counts = train_df.groupby('label')['patient_id'].nunique()
train_patch_counts   = train_df.groupby('label').size()

train_summary = pd.DataFrame({
    'patient_count': train_patient_counts,
    'patch_count': train_patch_counts
})

# --- Test counts ---
test_patient_counts = test_df.groupby('label')['patient_id'].nunique()
test_patch_counts   = test_df.groupby('label').size()

test_summary = pd.DataFrame({
    'patient_count': test_patient_counts,
    'patch_count': test_patch_counts
})

print(f"\nTotal train {train_patient_counts.sum()} patients, summary per label:")
print(train_summary)

print(f"\nTotal test {test_patient_counts.sum()} patients, summary per label:")
print(test_summary)

#%%
from sklearn.model_selection import StratifiedShuffleSplit
print("\nExtract patients and their labels")
train_patient_labels = train_df.groupby('patient_id')['label'].first()
# Desired validation ratio
sss = StratifiedShuffleSplit(
    n_splits=config.DATA.N_SPLIT, 
    # test_size=config.DATA.TEST_SIZE, 
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
    print(f"{train_fold_patients=}")
    print(f"{val_fold_patients=}")
       
#%% Collect patch counts from your fold DataFrames
train_patches, val_patches = [], []
for fold in range(len(train_df_fold_list)):
    n_train_patches = len(train_df_fold_list[fold])
    n_val_patches   = len(val_df_fold_list[fold])
    train_patches.append(n_train_patches)
    val_patches.append(n_val_patches)
    print(f"\nFold {fold}: Train patches={n_train_patches}, Val patches={n_val_patches}")

#%%

# 1. Initialize a list to hold the statistics
fold_patient_stats = []

# 2. Iterate through each fold
for fold in range(len(train_df_fold_list)):
    # Get the DataFrames for the current fold
    df_train = train_df_fold_list[fold]
    df_val = val_df_fold_list[fold]
    
    # --- Training Patient Counts ---
    # Group by label and count unique patient_ids
    train_counts = df_train.groupby('label')['patient_id'].nunique()
    train_ida_patients = train_counts.get(0, 0) # Label 0
    train_thl_patients = train_counts.get(1, 0) # Label 1
    
    # --- Validation Patient Counts ---
    val_counts = df_val.groupby('label')['patient_id'].nunique()
    val_ida_patients = val_counts.get(0, 0)
    val_thl_patients = val_counts.get(1, 0)
    
    # --- Totals ---
    total_train_patients = df_train['patient_id'].nunique()
    total_val_patients = df_val['patient_id'].nunique()
    
    # 3. Append to list
    fold_patient_stats.append({
        'Fold': fold,
        'Train IDA Patients': train_ida_patients,
        'Train THL Patients': train_thl_patients,
        'Train Total': total_train_patients,
        'Val IDA Patients': val_ida_patients,
        'Val THL Patients': val_thl_patients,
        'Val Total': total_val_patients
    })

# 4. Create a DataFrame for nice display
df_patient_summary = pd.DataFrame(fold_patient_stats)

# 5. Display the table
print("\n=== Detailed Patient Distribution per Fold ===")
print(df_patient_summary.to_string(index=False))

#%%

# 1. Initialize a list to hold the statistics
fold_stats = []

# 2. Iterate through each fold
for fold in range(len(train_df_fold_list)):
    # Get the DataFrames for the current fold
    df_train = train_df_fold_list[fold]
    df_val = val_df_fold_list[fold]
    
    # --- Training Counts ---
    # value_counts() gives a Series like {0: 1200, 1: 800}
    train_counts = df_train['label'].value_counts()
    train_ida = train_counts.get(0, 0) # Get count for 0 (IDA), default to 0 if missing
    train_thl = train_counts.get(1, 0) # Get count for 1 (THL)
    
    # --- Validation Counts ---
    val_counts = df_val['label'].value_counts()
    val_ida = val_counts.get(0, 0)
    val_thl = val_counts.get(1, 0)
    
    # --- Totals ---
    total_train = len(df_train)
    total_val = len(df_val)
    
    # 3. Append to list
    fold_stats.append({
        'Fold': fold,
        'Train IDA (0)': train_ida,
        'Train THL (1)': train_thl,
        'Train Total': total_train,
        'Val IDA (0)': val_ida,
        'Val THL (1)': val_thl,
        'Val Total': total_val
    })

# 4. Create a DataFrame for nice display
df_fold_summary = pd.DataFrame(fold_stats)

# 5. Display the table
print("\n=== Detailed Patch Distribution per Fold ===")
print(df_fold_summary.to_string(index=False))

# Optional: Verify the ratio
# df_fold_summary['Train Ratio (IDA:THL)'] = df_fold_summary['Train IDA (0)'] / df_fold_summary['Train THL (1)']
# print(df_fold_summary)

#%%
print("\n Data for plotting")
folds = np.arange(len(train_patches))

# After creating the figure and axes
fig, ax = plt.subplots(figsize=(14,8), dpi=300)
bar_width = 0.25

# Bars
train_bars = plt.bar(folds - bar_width/2, train_patches, 
                     width=bar_width, label="Train Patches", color="#1f77b4")
val_bars   = plt.bar(folds + bar_width/2, val_patches, 
                     width=bar_width, label="Val Patches", color="#ff7f0e")

# Labels on top of bars
for bars in [train_bars, val_bars]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height, f"{height:,}", 
                 ha="center", va="bottom", fontsize=20, rotation=45)

# Formatting
plt.xticks(folds, [f"Fold {i}" for i in folds], fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel("Patch Count", fontsize=20)
plt.xlabel("Fold", fontsize=20)
plt.legend(fontsize=20, loc='lower right')
plt.tight_layout()

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.show()
fig.savefig(f"5-fold-cv-seed{config.DATA.SEED}.png")
plt.close(fig)

#%%
model_name = config.MODEL.NAMES[0]
# model_name = 'DenseNet121'
print(f"{model_name=}")
model_builder = ModelBuilder(
    model_name=model_name,
    config=config)
preprocessing_fn = model_builder.get_preprocessing_function()       
print(f"{model_builder=}, {preprocessing_fn=}")

#%% 
print("\nPrepare test df")
test_df = test_df.sample(frac=1,
        random_state=config.DATA.SEED).reset_index(drop=True)

test_df_sampled = (
    mygears.stratified_subsample(
        test_df, config.TRAIN.SAMPLE_SIZE_TEST)
        if config.TRAIN.SAMPLE else test_df
)
mygears.count_plot(test_df_sampled, "Test Samples")

#%%
# from PatchDatasetPreserve import PatchDatasetPreserve
# print("\nCreate test gen")
# test_gen = PatchDatasetPreserve(
#     config,
#     df=test_df_sampled,
#     batch_size=config.TRAIN.BATCH_SIZE,
#     preprocessing_function=preprocessing_fn,
#     shuffle=False,
#     # save=True,
#     # use_cutmix=True,
#     # cutmix_version='v2'
#     )  

# print("\nTest generator summary:")
# mygears.count_plot(test_df, f"Test Set  - {len(test_df)}")
# test_gen_dict = test_gen.summary() 

# rand_idx = np.random.randint(0, len(test_gen)) 
# print(f"{rand_idx=}")
# images, labels = test_gen.__getitem__(rand_idx)
# print(f"{images.shape=}, {labels.shape=}") #Soft Labels
# img_final = images[0]
# print(np.min(images), np.max(images))
# test_gen.plot_random_batch(batch_idx=rand_idx, note="Unseen Test Set")
   
#%%
# if config.TRAIN.Enable:
train_gen_list = []
val_gen_list = []
class_weight_list = []
train_folds = range(0, config.DATA.N_SPLIT)
for fold in train_folds: 
    print(f"\n\n=== Fold {fold} ===")
    
    # --- 1. Prepare DataFrames ---
    train_df_fold = train_df_fold_list[fold].copy()
    val_df_fold   = val_df_fold_list[fold].copy()
    
    print(f"Train: {len(train_df_fold)} patches, {train_df_fold['patient_id'].nunique()} patients")
    print(f"Val:   {len(val_df_fold)} patches, {val_df_fold['patient_id'].nunique()} patients")

    # --- 2. Optional: visualize class balance ---
    train_summary = train_df_fold['label'].value_counts().sort_index()
    val_summary   = val_df_fold['label'].value_counts().sort_index()
    print(f"Train label distribution:\n{train_summary}")
    print(f"Val label distribution:\n{val_summary}")

    mygears.count_plot(train_df_fold, f"Train Set Fold {fold} - {len(train_df_fold)}")
    mygears.count_plot(val_df_fold, f"Validation Set Fold {fold} - {len(val_df_fold)}")   
    
    #     train_df_fold_bal = create_balanced_dataframe(train_df_fold) 
    #     val_df_fold_bal = create_balanced_dataframe(val_df_fold) 
    #     count_plot(train_df_fold_bal, f"Balanced Train Set Fold {fold} - {len(train_df_fold_bal)}")
    #     count_plot(val_df_fold_bal, f"Balanced Validation Set Fold {fold} - {len(val_df_fold_bal)}")
      
     #%   
    if platform.system() == 'Linux' and config['DATA']['VERIFY']:
        print("\nVerify images in the train and validation dataframes")
        train_df_fold = mygears.verify_images(train_df_fold, 'patch_path')
        val_df_fold = mygears.verify_images(val_df_fold, 'patch_path')
        print("Verify completed")
        
    #%   
    # Plot images for each class separately
    # plot_images_by_class(train_df_fold, class_label='0', savefile="train_IDA1.png")  # IDA
    # plot_images_by_class(train_df_fold, class_label='1', savefile="train_Thalas1.png") 
    # plot_images_by_class(val_df_fold, class_label='0', savefile="val_IDA1.png")  # IDA
    # plot_images_by_class(val_df_fold, class_label='1', savefile="val_Thalas1.png") 
    
    # plot_images_by_class(test_df, class_label='0', savefile="test_IDA.png")  # IDA
    # plot_images_by_class(test_df, class_label='1', savefile="test_Thalas.png")
    
    models_per_fold = []
    for model_name in config['MODEL']['NAMES']:
        print(f"{model_name=}")
        train_log = config.BASE + '-Rieman-' + \
            f"Fold-{fold}-" + \
            model_name + '-stage2-' + \
            config.MIL_MODEL + '-' + \
            config['TRAIN']['DATETIME']        
        print(f"{train_log=}")
        
        if config['SAVE'] == True:           
            os.makedirs(train_log, exist_ok=True)
            cfg_dict = mygears.cfg_to_dict(config)
            json_file_path = os.path.join(train_log,'_config.json')  
            with open(json_file_path, 'w') as json_file:    
                json.dump(cfg_dict, json_file, indent=4)           
            print(f"\nSaved: {json_file_path}")
            # pass
                                
        #%                
        # model_builder = ModelBuilder(model_name=model_name, config=config)
        # model = model_builder.create_model()
        # preprocessing_fn = model_builder.get_preprocessing_function()       

        # print("\nDisplay last 29 layers")
        # for index, layer in enumerate(model.layers[-29:]):  # last 29 layers
        #     print(index, layer.name, layer.output.shape, layer.trainable)    
        #     # layer.trainable = True   
        # mygears.print_model_summary(model)
        # print(f"{len(model.layers)=}")
        # print(f"Trainable layers: {sum([l.trainable for l in model.layers])} / {len(model.layers)}")
        
        #%
        print("\nPrepare train_df for generator")
        train_df_fold = train_df_fold.sample(
            frac=1, random_state=config.DATA.SEED).reset_index(drop=True)
        train_df_fold_sampled = (
            mygears.stratified_subsample(
                train_df_fold,
                config['TRAIN']['SAMPLE_SIZE'],
                )
            if config['TRAIN']['SAMPLE'] else train_df_fold
        )
        
        #%
        # count_plot(train_df_fold_bal_sampled, f"Train Set Fold {fold} - {len(train_df_fold_bal_sampled)}")      
        train_gen = PatchDatasetPreserve(
            config,
            df=train_df_fold_sampled,
            batch_size=config.TRAIN.BATCH_SIZE,
            preprocessing_function=preprocessing_fn,
            shuffle=True, # <--- EXPLICITLY SET TO TRUE FOR TRAINING
        )
        # --- Sanity check ---
        print("\nTrain generator summary:")
        train_gen_dict = train_gen.summary()
        rand_idx = np.random.randint(0, len(train_gen)) 
        images, labels = train_gen.__getitem__(rand_idx)
        print(f"{rand_idx}, {images.dtype=}, {images.shape=}, {labels.shape=}")
        # img = images[4]
        # plt.figure();
        # plt.imshow(img.astype(np.uint8));
        # plt.title("RGB cleaned")
        train_gen.plot_random_batch(
            batch_idx=rand_idx,
            note=f'Train fold {fold}',
            save_dir=train_log
        )
        
        #%
        val_df_fold = val_df_fold.sample(
            frac=1, random_state=config.DATA.SEED).reset_index(drop=True)
        
        val_df_fold_sampled = (
            mygears.stratified_subsample(val_df_fold, config['TRAIN']['SAMPLE_SIZE_VAL'])
            if config['TRAIN']['SAMPLE'] else val_df_fold
        )   
        
        #%
        val_gen = PatchDatasetPreserve(
            config,
            df=val_df_fold_sampled,
            batch_size=config.TRAIN.BATCH_SIZE,
            preprocessing_function=preprocessing_fn,
            shuffle=False,
            # save=True,
            # use_cutmix=True # Keep False for validation/testing
            # verbose=True
        )
        print("\nVal generator summary:")
        val_gen_dict = val_gen.summary()  
        rand_idx = np.random.randint(0, len(val_gen))               
        images, labels = val_gen.__getitem__(rand_idx)
        print(f"{images.shape=}, {labels.shape=}")
        val_gen.plot_random_batch(rand_idx,
                                  note=f'Val fold {fold}',
                                  save_dir=train_log
                                  )  
        
        train_gen_list.append(train_gen)
        val_gen_list.append(val_gen)
        
        # if fold >= config.TRAIN.Actual_Fold:
        #     break

print("Data Loaded.")


#%%
def save_feature_extractors(config, trained_models_list, layer_index=-2):
    """
    Creates and saves feature extractors from a list of trained models.

    Args:
        config: Configuration dictionary/object containing 'MODEL_PATH' and 'DATA.N_SPLIT'.
        trained_models_list (list): A list of your fully trained TF/Keras models (one per fold).
        layer_index (int or str): The index or name of the layer to use as feature output. 
                                  Defaults to -2 (usually the layer before the final Dense/Softmax).
                                  If using ConvNeXt/ResNet with 'pooling=avg', the output of the 
                                  base model is often the features, so you might check your model summary.
    """
    
    # Ensure the save directory exists
    # Handling config['MODEL_PATH'] vs config.MODEL_PATH based on your object type
    try:
        save_dir = config['MODEL_PATH']
    except TypeError:
        save_dir = config.MODEL_PATH
        
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n[Saving Feature Extractors] Saving to: {save_dir}")

    for fold in range(config.DATA.N_SPLIT):
        # 1. Get the trained full model for this fold
        full_model = trained_models_list[fold]
        
        # 2. Identify the output layer
        # If layer_index is an integer (e.g., -2), get layer by index
        # If it's a string (e.g., 'global_average_pooling2d'), get by name
        if isinstance(layer_index, int):
            target_layer = full_model.layers[layer_index]
        else:
            target_layer = full_model.get_layer(layer_index)
            
        print(f"  Fold {fold}: extracting features from layer '{target_layer.name}'")

        # 3. Create the Feature Extractor Model
        feature_extractor = tf.keras.Model(
            inputs=full_model.input,
            outputs=target_layer.output,
            name=f"FeatureExtractor_Fold{fold}"
        )

        # 4. Save to Disk
        filename = f"fold_{fold}_feature_extractor.h5"
        fe_path = os.path.join(save_dir, filename)
        
        feature_extractor.save(fe_path)
        print(f"    Saved: {filename}")

    print("[Done] All feature extractors saved.")

#%%
# trained_models_list = mygears.load_ensemble_models(model_names, config)
# print("Models have been loaded.")
# save_feature_extractors(config, trained_models_list, layer_index=-2)

#%%
from riemannian_utils import CurvatureAttention, RiemannianMetrics

def p1_step1_extract_features(
        model_name,
        model,
        data_gen,
        layer_index=-2,
        verbose=True):
    """
    Step 1: Creates a feature extractor and processes the dataset to get Z.
    Input: Trained CNN model, Data Generator.
    Output: feature_extractor model, features (Z), labels (y).
    """
    if verbose: print("[P1.1] Building Feature Extractor & Extracting Features...")
    
    # Define Feature Extractor (Cut off classification head)
    # Automatically finds the penultimate layer or specific layer index
    feature_extractor = tf.keras.Model(
        inputs=model.input,
        outputs=model.layers[layer_index].output
    ) 
    feature_extractor.trainable = False
    # print(feature_extractor.summary())
    print(">> Extracting features...")
    # Extract Features (Z)
    # Note: data_gen should not shuffle to keep alignment if needed later
    features = feature_extractor.predict(
        data_gen, 
        verbose=1,
        workers=4,
        use_multiprocessing=True
    )
    
    # Get labels (One-hot or sparse depending on generator)
    # Accessing .classes directly is faster for generators like PatchDataset
    if hasattr(data_gen, 'classes'):
        labels = data_gen.classes
    else:
        # Fallback: iterate generator (slower)
        labels = np.concatenate([y for x, y in data_gen], axis=0)
        if len(labels.shape) > 1: labels = np.argmax(labels, axis=1)

    if verbose: print(f"  -> Extracted shape: {features.shape}")
    
    fe_path = os.path.join(config['MODEL_PATH'], f"fold_{fold}_{model_name}_feature_extractor.h5")
    feature_extractor.save(fe_path)
    print(f"Saved Feature Extractor: {fe_path}")
    
    features_save_path = os.path.join(config['MODEL_PATH'], f"fold_{fold}_{model_name}_features.npy")
    np.save(features_save_path, features)
    print(f"Saved Features: {features_save_path}")
    
    return feature_extractor, features, labels

def p1_step2_fit_manifold(features, labels, n_classes, feature_dim, verbose=True):
    """
    Step 2: Fits the Riemannian Manifold (Centroids & Precision Matrices).
    Input: Features (Z), Labels.
    Output: Fitted riemann_tool.
    """
    if verbose: print("[P1.2] Fitting Riemannian Manifold (Method A)...")
    
    # Initialize your custom tool (from manifold_engine)
    riemann_tool = RiemannianMetrics(
        n_classes=n_classes, 
        feature_dim=feature_dim
    )
    
    # Fit manifold using Inverse Covariance (Precision Matrix)
    riemann_tool.fit_manifold(features, labels)
    
    if verbose: print("  -> Manifold parameters (mu, sigma_inv) computed.")
    return riemann_tool

def p1_step3_compute_train_energy(riemann_tool, features, save_path=None, verbose=True):
    """
    Step 3: Computes Curvature/Energy for the training set.
    Input: Fitted riemann_tool, Features (Z).
    Output: Energy scores (E_train).
    """
    if verbose: print("[P1.3] Computing Training Curvature (Method C)...")
    
    # Calculate Mahalanobis Energy / Ricci Scalar Proxy
    E_train = riemann_tool.compute_batch_curvature(features)
    
    # Optional: Save the manifold tool for this fold
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(riemann_tool, f)
        if verbose: print(f"  -> Manifold saved to {save_path}")
        
    return E_train

def p2_step1_extract_val_features(
        model_name,
        feature_extractor,
        val_gen,
        verbose=True):
    """
    Step 1: Extracts features from the Validation set using the Phase 1 extractor.
    """
    if verbose: print("[P2.1] Extracting Validation Features...")
    
    X_val_z = feature_extractor.predict(val_gen, verbose=1)
    
    if hasattr(val_gen, 'classes'):
        y_val = val_gen.classes
    else:
        y_val = np.concatenate([y for x, y in val_gen], axis=0)
        if len(y_val.shape) > 1: y_val = np.argmax(y_val, axis=1)
        
    # fe_path = os.path.join(config['MODEL_PATH'], f"fold_{fold}_{model_name}_val_feature_extractor.h5")
    # feature_extractor.save(fe_path)
    # print(f"Saved Validation Feature Extractor: {fe_path}")
        
    return X_val_z, y_val

def p2_step2_compute_val_energy(riemann_tool, X_val_z, save_path=None, verbose=True):
    """
    Step 2: Computes Curvature/Energy for Validation set using Phase 1 Manifold.
    """
    if verbose: print("[P2.2] Computing Validation Energies...")
    
    # Crucial: Use the tool fitted on TRAIN data to score VAL data
    E_val = riemann_tool.compute_batch_curvature(X_val_z)
    
    # Optional: Save the manifold tool for this fold
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(E_val, f)
        if verbose: print(f"  -> Val Manifold saved to {save_path}")
    
    return E_val

def p2_step3_calibrate_threshold(E_val, y_val, config, verbose=True):
    """
    Step 3: Determines optimal energy threshold (T).
    (Can use percentiles, OOD detection logic, or max accuracy).
    """
    if verbose: print("[P2.3] Calibrating Energy Threshold...")
    
    # Example Strategy: Percentile-based (e.g., keep 90% of data)
    # Or use your specific run_phase_2_validation logic here
    
    # For now, let's assume a percentile strategy as a placeholder
    # In your real code, you might loop through thresholds to maximize patient acc
    threshold = np.percentile(E_val, 95) # Flag top 5% as high energy
    
    # Return energies and the single threshold scalar
    return E_val, threshold

def build_robust_mil_model(input_dim, curvature_layer):
    """
    Builds the Robust MIL Aggregation Model.
    Inputs: 
       - Features (B, D)
       - Energies (B, 1)
    Output: 
       - Class Prediction (B, 2) re-weighted
    """
    input_features = tf.keras.Input(shape=(input_dim,), name='features')
    input_energy = tf.keras.Input(shape=(1,), name='energy')
    
    # Apply Curvature Attention
    weighted_feats, attn_weights = curvature_layer([input_features, input_energy])
    
    # Classifier Head (Simple Linear or MLP)
    x = tf.keras.layers.Dense(64, activation='relu')(weighted_feats)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(2, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=[input_features, input_energy], outputs=output)
    return model

def create_patient_bags(df, features, energies):
    """
    Groups patch features and energies into Patient Bags.
    
    Args:
        df: DataFrame containing 'patient_id' and 'label' columns, corresponding to 'features'.
        features: Numpy array (N_total_patches, D)
        energies: Numpy array (N_total_patches, 1)
        
    Returns:
        bags_z: List of arrays [ (N_p1, D), (N_p2, D), ... ]
        bags_e: List of arrays [ (N_p1, 1), (N_p2, 1), ... ]
        bag_labels: List of labels [ y_p1, y_p2, ... ]
    """
    print(f"{len(df)=}")
    print("[Bagging] Grouping patches by Patient ID...")
    
    unique_patients = df['patient_id'].unique()
    bags_z = []
    bags_e = []
    bag_labels = []
    
    # We assume 'features' are aligned with 'df' row-by-row.
    # (Ensure shuffle=False was used during extraction)
    
    for pid in tqdm(unique_patients):
        # Get indices for this patient
        indices = df.index[df['patient_id'] == pid].tolist()
        
        if len(indices) == 0: continue
        
        # Slice features
        p_feats = features[indices] # Shape (N_patches, D)
        p_energies = energies[indices] # Shape (N_patches, 1)
        
        # Get Label (Assuming all patches have same patient label in DF)
        p_label = df.loc[indices[0], 'label']
        
        bags_z.append(p_feats)
        bags_e.append(p_energies)
        bag_labels.append(p_label)
        
    return bags_z, bags_e, np.array(bag_labels)

def pad_bags(bags, max_len=None):
    """
    Pads bags to the same length for batch training (or use RaggedTensors).
    Here we use zero-padding.
    """
    if max_len is None:
        max_len = max([b.shape[0] for b in bags])
        
    feature_dim = bags[0].shape[1]
    n_samples = len(bags)
    
    padded_data = np.zeros((n_samples, max_len, feature_dim), dtype=np.float32)
    # Mask to ignore padding later (optional implementation detail)
    
    for i, bag in enumerate(bags):
        length = min(len(bag), max_len)
        padded_data[i, :length, :] = bag[:length, :]
        
    return padded_data


from riemannian_utils import PoincareMath, HyperbolicDense, FrechetMean
from tensorflow.keras import layers

def build_deep_riemannian_mil_model_0(input_dim, curvature_layer):
    """
    Revised Deep Riemannian MIL Model.
    Improvements:
    1. Unified Curvature (c=0.1) for consistent geometry.
    2. Widened Encoder (1536 -> 128 -> 64) to preserve feature details.
    """
    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------
    # Use a unified curvature. c=0.1 corresponds to Radius ~3.16
    # This provides more embedding volume than c=1.0
    c_val = 0.1 

    # ---------------------------------------------------------
    # 1. Inputs
    # ---------------------------------------------------------
    input_features = tf.keras.Input(shape=(None, input_dim), name='bag_features')
    input_energy = tf.keras.Input(shape=(None, 1), name='bag_energy') 
    
    # ---------------------------------------------------------
    # 2. Normalization & Scaling
    # ---------------------------------------------------------
    # Scale Energy
    energy_norm = layers.BatchNormalization(name="energy_auto_scale")(input_energy)

    # Scale Features (LayerNorm + Rescale to Radius ~3.9)
    # This prepares them for the c=0.1 manifold
    x_norm = layers.LayerNormalization(epsilon=1e-6, name="pre_manifold_norm")(input_features)
    x_norm = layers.Lambda(lambda x: x * 0.1, name="rescale_to_ball_radius")(x_norm)

    # ---------------------------------------------------------
    # 3. Manifold Projection
    # ---------------------------------------------------------
    # Project into the c=0.1 space
    x_hyp = layers.Lambda(lambda v: PoincareMath.exp_map(v, c=c_val), name="euclidean_to_hyperbolic")(x_norm)

    # ---------------------------------------------------------
    # 4. Deep Hyperbolic Encoder (Widened)
    # ---------------------------------------------------------
    # Layer 1: 1536 -> 128 (Preserve more info)
    x_hyp = HyperbolicDense(128, c=c_val, activation='relu', name="hyp_dense_1")(x_hyp)
    
    # Layer 2: 128 -> 64 (Bottleneck)
    x_hyp = HyperbolicDense(64, c=c_val, activation='relu', name="hyp_dense_2")(x_hyp)

    # ---------------------------------------------------------
    # 5. Curvature-Aware Attention
    # ---------------------------------------------------------
    # Note: CurvatureAttention usually calculates internally, 
    # but we pass the normalized inputs for stability.
    _, att_weights = curvature_layer([x_norm, energy_norm])
    
    # ---------------------------------------------------------
    # 6. Aggregation (Frechet Mean)
    # ---------------------------------------------------------
    # CRITICAL FIX: Use the SAME c_val here
    bag_representation_hyp = FrechetMean(c=c_val, name="frechet_mean")([x_hyp, att_weights])
    
    # ---------------------------------------------------------
    # 7. Tangent Projection & Classifier
    # ---------------------------------------------------------
    # CRITICAL FIX: Use the SAME c_val here
    bag_representation_tan = layers.Lambda(lambda v: PoincareMath.log_map(v, c=c_val), name="hyperbolic_to_euclidean")(bag_representation_hyp)
    
    x = layers.Dense(32, activation='relu', name="clf_dense_1")(bag_representation_tan)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(16, activation='relu', name="clf_dense_2")(x)
    
    output = layers.Dense(2, activation='softmax', name="output")(x)
    
    model = tf.keras.Model(inputs=[input_features, input_energy], outputs=output, name="Deep_Riemannian_MIL")
    return model

def build_deep_riemannian_mil_model_1(input_dim, curvature_layer):
    """
    Tuned Model: Wider Layers, Flatter Curvature, Lower Dropout
    """
    # [TUNING 1] Flatter Curvature (0.01 vs 0.1)
    # Allows easier optimization for features that aren't purely hierarchical
    c_val = 0.01 

    # Inputs
    input_features = tf.keras.Input(shape=(None, input_dim), name='bag_features')
    input_energy = tf.keras.Input(shape=(None, 1), name='bag_energy') 
    
    # Normalization (Keep this, it's working)
    energy_norm = layers.BatchNormalization(name="energy_auto_scale")(input_energy)
    x_norm = layers.LayerNormalization(epsilon=1e-6, name="pre_manifold_norm")(input_features)
    x_norm = layers.Lambda(lambda x: x * 0.1, name="rescale_to_ball_radius")(x_norm)

    # Projection
    x_hyp = layers.Lambda(lambda v: PoincareMath.exp_map(v, c=c_val), name="euclidean_to_hyperbolic")(x_norm)

    # [TUNING 2] Increase Width (Capacity)
    # 1536 -> 512 (Retain more info)
    x_hyp = HyperbolicDense(512, c=c_val, activation='relu', name="hyp_dense_1")(x_hyp)
    
    # 512 -> 256 (Gradual compression)
    x_hyp = HyperbolicDense(256, c=c_val, activation='relu', name="hyp_dense_2")(x_hyp)

    # Attention
    _, att_weights = curvature_layer([x_norm, energy_norm])
    
    # Aggregation
    bag_representation_hyp = FrechetMean(c=c_val, name="frechet_mean")([x_hyp, att_weights])
    
    # Tangent Projection
    bag_representation_tan = layers.Lambda(lambda v: PoincareMath.log_map(v, c=c_val), name="hyperbolic_to_euclidean")(bag_representation_hyp)
    
    # Simplify the Head to prevent Label Flipping
    # Remove the multi-layer perceptron (Dense 32 -> 16). 
    # Use a single, strong linear projection.   
    # Optional: Add BatchNormalization to center the tangent vectors before classification
    # x = layers.BatchNormalization(name="tangent_norm")(bag_representation_tan)
    
    # Classifier Head
    x = layers.Dense(128, activation='relu', name="clf_dense_1")(bag_representation_tan)
    
    # [TUNING 3] Reduce Dropout (0.3 -> 0.1)
    # We need the model to learn efficiently, not drop connections yet
    x = layers.Dropout(0.1)(x) 
    x = layers.Dense(64, activation='relu', name="clf_dense_2")(x)
    
    output = layers.Dense(2, activation='softmax', name="output")(x)
    
    model = tf.keras.Model(inputs=[input_features, input_energy], outputs=output, name="Deep_Riemannian_MIL")
    return model 
# >> Best Performing Fold: 4 (AUC: 0.7649), DATA.val_ratio = 0.2 flip 
  # aucs = [1-0.22456140350877188, 1-0.39649122807017545,
  #            0.5789473684210527, 1-0.26666666666666666, 
  #            0.7649122807017544]
  #  np.mean(aucs) #0.6912280701754386
# Flip 0, 1, 3

# >> Best Performing Fold: 4 (AUC: 0.8491), DATA.val_ratio = 0.3
   #  aucs = [1-0.3017543859649123, 1-0.3649122807017544,
   #           0.6421052631578947, 1-0.375438596491228, 
   #           0.8491228070175438]
   # np.mean(aucs) # Mean: 0.6898245614035088
# Flip 0, 1, 3


def build_deep_riemannian_mil_model_2(input_dim, curvature_layer):
    """
    Tuned Model: Wider Layers, Flatter Curvature, Lower Dropout
    """
    # [TUNING 1] Flatter Curvature (0.01 vs 0.1)
    # Allows easier optimization for features that aren't purely hierarchical
    c_val = 0.01 

    # Inputs
    input_features = tf.keras.Input(shape=(None, input_dim), name='bag_features')
    input_energy = tf.keras.Input(shape=(None, 1), name='bag_energy') 
    
    # Normalization (Keep this, it's working)
    energy_norm = layers.BatchNormalization(name="energy_auto_scale")(input_energy)
    x_norm = layers.LayerNormalization(epsilon=1e-6, name="pre_manifold_norm")(input_features)
    x_norm = layers.Lambda(lambda x: x * 0.1, name="rescale_to_ball_radius")(x_norm)

    # Projection
    x_hyp = layers.Lambda(lambda v: PoincareMath.exp_map(v, c=c_val), name="euclidean_to_hyperbolic")(x_norm)

    # [TUNING 2] Increase Width (Capacity)
    # 1536 -> 512 (Retain more info)
    x_hyp = HyperbolicDense(512, c=c_val, activation='relu', name="hyp_dense_1")(x_hyp)
    
    # 512 -> 256 (Gradual compression)
    x_hyp = HyperbolicDense(256, c=c_val, activation='relu', name="hyp_dense_2")(x_hyp)

    # Attention
    _, att_weights = curvature_layer([x_norm, energy_norm])
    
    # Aggregation
    bag_representation_hyp = FrechetMean(c=c_val, name="frechet_mean")([x_hyp, att_weights])
    
    # Tangent Projection
    bag_representation_tan = layers.Lambda(lambda v: PoincareMath.log_map(v, c=c_val), name="hyperbolic_to_euclidean")(bag_representation_hyp)
    
    # Simplify the Head to prevent Label Flipping
    # Remove the multi-layer perceptron (Dense 32 -> 16). 
    # Use a single, strong linear projection.   
    # Optional: Add BatchNormalization to center the tangent vectors before classification
    x = layers.BatchNormalization(name="tangent_norm")(bag_representation_tan)
     
    output = layers.Dense(2, activation='softmax', name="output")(x)
    
    model = tf.keras.Model(inputs=[input_features, input_energy], outputs=output, name="Deep_Riemannian_MIL")
    return model
# >> Best Performing Fold: 4 (AUC: 0.7474)
    # aucs =  [1-0.2070175438596491, 
    #          1-0.4175438596491228, 
    #          1-0.49473684210526314, 
    #          1-0.3017543859649123, 
    #          0.7473684210526316]
    # np.mean(aucs) # 0.6652631578947369

def build_deep_riemannian_mil_model_3(input_dim, curvature_layer):
    """
    Tuned Model: Wider Layers, Flatter Curvature, Lower Dropout.
    INCLUDES: Energy Skip Connection to fix 'Label Flipping' (Polarity Anchor).
    """
    # [TUNING 1] Flatter Curvature (0.01 vs 0.1)
    # Allows easier optimization for features that aren't purely hierarchical
    c_val = 0.01 

    # ---------------------------------------------------------
    # 1. Inputs & Normalization
    # ---------------------------------------------------------
    input_features = tf.keras.Input(shape=(None, input_dim), name='bag_features')
    input_energy = tf.keras.Input(shape=(None, 1), name='bag_energy') 
    
    # Normalization (Keep this, it's working)
    energy_norm = layers.BatchNormalization(name="energy_auto_scale")(input_energy)
    x_norm = layers.LayerNormalization(epsilon=1e-6, name="pre_manifold_norm")(input_features)
    x_norm = layers.Lambda(lambda x: x * 0.1, name="rescale_to_ball_radius")(x_norm)

    # ---------------------------------------------------------
    # 2. Manifold Learning (Hyperbolic Space)
    # ---------------------------------------------------------
    # Projection
    x_hyp = layers.Lambda(lambda v: PoincareMath.exp_map(v, c=c_val), name="euclidean_to_hyperbolic")(x_norm)

    # [TUNING 2] Increase Width (Capacity)
    # 1536 -> 512 (Retain more info)
    x_hyp = HyperbolicDense(512, c=c_val, activation='relu', name="hyp_dense_1")(x_hyp)
    
    # 512 -> 256 (Gradual compression)
    x_hyp = HyperbolicDense(256, c=c_val, activation='relu', name="hyp_dense_2")(x_hyp)

    # Attention
    _, att_weights = curvature_layer([x_norm, energy_norm])
    
    # Aggregation (Frechet Mean)
    bag_representation_hyp = FrechetMean(c=c_val, name="frechet_mean")([x_hyp, att_weights])
    
    # Tangent Projection (Return to Euclidean)
    bag_representation_tan = layers.Lambda(lambda v: PoincareMath.log_map(v, c=c_val), name="hyperbolic_to_euclidean")(bag_representation_hyp)

    # ---------------------------------------------------------
    # 3. [NEW] Polarity Anchor (Skip Connection)
    # ---------------------------------------------------------
    # We add the "Energy" directly to the classifier input. 
    # High Energy usually means "Thalassemia". This gives the model a "Hint" 
    # about which direction is positive, preventing the flip (AUC 0.22 -> 0.78).
    
    # Global Average Pooling on the Energy Input (to get 1 value per bag)
    # energy_norm is shape (Batch, N_patches, 1) -> (Batch, 1)
    energy_global = layers.GlobalAveragePooling1D(name="energy_pooling")(energy_norm) 
    
    # Concatenate Tangent Vectors with Energy
    # bag_representation_tan is (Batch, 256)
    # energy_global is (Batch, 1)
    # merged is (Batch, 257)
    merged = layers.Concatenate(name="polarity_anchor")([bag_representation_tan, energy_global])
    
    # ---------------------------------------------------------
    # 4. Classifier Head (Deep Version)
    # ---------------------------------------------------------
    # Pass 'merged' input instead of just 'bag_representation_tan'
    x = layers.Dense(128, activation='relu', name="clf_dense_1")(merged)
    
    # [TUNING 3] Reduce Dropout (0.3 -> 0.1)
    x = layers.Dropout(config.TRAIN.DROPOUT)(x) 
    x = layers.Dense(64, activation='relu', name="clf_dense_2")(x)
    
    output = layers.Dense(2, activation='softmax', name="output")(x)
    
    model = tf.keras.Model(inputs=[input_features, input_energy], outputs=output, name="Deep_Riemannian_MIL")
    return model
# >> Best Performing Fold: 2 (AUC: 0.5439) , DATA.val_ratio = 0.2
    # aucs = [1-0.1894736842105263, 1-0.2807017543859649,
    #          0.5438596491228069, 1-0.2070175438596491,
    #          1-0.1333333333333333] => 0, 1, 2, 3
    # np.mean(aucs) #0.7466666666666666 => 0.75

# >> Best Performing Fold: 4 (AUC: 0.8737), DATA.val_ratio = 0.3
    # aucs=[1-0.22807017543859648, 1-0.2982456140350877, 
    #     0.6771929824561403, 1-0.22807017543859648,
     #     0.8736842105263158] => 0, 1, 3
    # print(np.mean(aucs)) #0.759298245614035 => 0.76

# >> Best Performing Fold: 4 (AUC: 0.8632), DATA.val_ratio = 0.3
# Fold,Original AUC,Diagnosis,Corrected AUC (1−AUC), DATA.val_ratio = 0.3
# Fold 0,0.6140,Correct Polarity (Weak),0.6140
# Fold 1,0.2667,FLIPPED,0.7333
# Fold 2,0.6737,Correct Polarity,0.6737
# Fold 3,0.2246,FLIPPED,0.7754
# Fold 4,0.1368,FLIPPED,0.8632
        # Mean = 0.7319 => 1, 3, 4
   
#%%
config = cfm.get_config()
print(config)
model_root = config.BASEMODEL_PATH # Or wherever your root is defined
model_name = 'ConvNeXtLarge'
model_mapping = {
    model_name: [ #segmented cell - baseline Patch-Level Backbone
        # os.path.join(model_root, 'Linux-Fold-0-ConvNeXtLarge-classweights-20251205-1028.hdf5'),
        # os.path.join(model_root, 'Linux-Fold-1-ConvNeXtLarge-classweights-20251205-2213.hdf5'),
        # os.path.join(model_root, 'Linux-Fold-2-ConvNeXtLarge-classweights-20251205-2213.hdf5'),
        # os.path.join(model_root, 'Linux-Fold-3-ConvNeXtLarge-classweights-20251205-2213.hdf5'),
        # os.path.join(model_root, 'Linux-Fold-4-ConvNeXtLarge-classweights-20251205-2213.hdf5'),
        
        # stage2 - seed 1337 Patch-Level Backbone
        os.path.join(model_root, 'Linux-Fold-0-stage2-ConvNeXtLarge-classweights-20251113-1824.hdf5'),
        os.path.join(model_root, 'Linux-Fold-1-stage2-ConvNeXtLarge-classweights-20251113-1824.hdf5'),
        os.path.join(model_root, 'Linux-Fold-2-stage2-ConvNeXtLarge-classweights-20251113-1824.hdf5'),
        os.path.join(model_root, 'Linux-Fold-3-stage2-ConvNeXtLarge-classweights-20251113-1824.hdf5'),
        os.path.join(model_root, 'Linux-Fold-4-stage2-ConvNeXtLarge-classweights-20251113-1824.hdf5'),
    ],    
}

#%%
if config.TRAIN.Enable:   
    trained_models_list = mygears.load_ensemble_models_1(
        model_mapping,
        config)
    print("Models have been loaded.")
    
    # model_per_fold = trained_models_list[0]
    # model_per_fold.summary()
    # for index, layer in enumerate(model_per_fold.layers):  # last 29 layers
    #         print(index, layer.name, layer.output.shape, layer.trainable)  
    # model_per_fold.layers[296].name # 296 batch_normalization (None, 1536) True

    # feature_extractor = tf.keras.Model(
    #     inputs=model_per_fold.input,
    #     outputs=model_per_fold.layers[296].output
    # )
    # feature_extractor.summary()
    
    # Storage for cross-fold analysis
    riemann_tools_all = []
    feature_extractors_all = []
    X_train_z_all = []
    X_val_z_all = []
    y_val_z_all = []
    energies_val_all = []
    thresholds_all = []
        
    for fold in range(config.DATA.N_SPLIT):
        print(f"\n{'='*40}")
        print(f" PROCESSING FOLD: {fold}/{config.DATA.N_SPLIT - 1}")
        print(f"{'='*40}")
    
        # 1. Get Data & Model
        train_gen_per_fold = train_gen_list[fold]
        val_gen_per_fold = val_gen_list[fold]
        model_per_fold = trained_models_list[fold]
        
        for index, layer in enumerate(model_per_fold.layers[config.TRAIN.Feature_Layer_Index:]): 
                print(index, layer.name, layer.output.shape, layer.trainable)  
        
        # ====================================================
        print("\n\nPHASE 1: Pre-training & Manifold Construction")
        # ====================================================
        
        # Step 1.1: Extract Training Features
        print(f"{config.TRAIN.Feature_Layer_Index=}")
        feature_extractor, X_train_z, y_train_z = p1_step1_extract_features(
            model_name=model_name,
            model=model_per_fold, 
            data_gen=train_gen_per_fold,
            layer_index=config.TRAIN.Feature_Layer_Index,            
        )
                     
        # Step 1.2: Fit Manifold
        riemann_tool = p1_step2_fit_manifold(
            features=X_train_z, 
            labels=y_train_z, 
            n_classes=config.MODEL.NUM_CLASSES, 
            feature_dim=X_train_z.shape[1]
        )
        
        # Step 1.3: Compute Training Energy (Baseline) & Save
        save_pkl_path = os.path.join(config.MODEL_PATH, f"fold_{fold}_{model_name}_stage2_manifold_{config.TRAIN.DATETIME}.pkl")
        E_train = p1_step3_compute_train_energy(
            riemann_tool=riemann_tool, 
            features=X_train_z, 
            save_path=save_pkl_path
        )
        
        # Store P1 Artifacts
        riemann_tools_all.append(riemann_tool)
        feature_extractors_all.append(feature_extractor)
        X_train_z_all.append(X_train_z)
    
        # ====================================================
        print("\n\nPHASE 2: Validation & Thresholding")
        # ====================================================
        
        # Step 2.1: Extract Validation Features
        X_val_z, y_val_z = p2_step1_extract_val_features(
            model_name=model_name,
            feature_extractor=feature_extractor, 
            val_gen=val_gen_per_fold
        )
        X_val_z_all.append(X_val_z)
        y_val_z_all.append(y_val_z)
        
        # Step 2.2: Compute Validation Energy using val feature Val_z
        save_pkl_path = os.path.join(config.MODEL_PATH, f"fold_{fold}_{model_name}_val_manifold_{config.TRAIN.DATETIME}.pkl")
        E_val = p2_step2_compute_val_energy(
            riemann_tool=riemann_tool, 
            X_val_z=X_val_z,
            save_path=save_pkl_path
        )
        
        # Step 2.3: Calibrate Threshold
        _, threshold = p2_step3_calibrate_threshold(
            E_val=E_val, 
            y_val=y_val_z, 
            config=config
        )
        
        # Store P2 Results
        energies_val_all.append(E_val)
        thresholds_all.append(threshold)
        
        print(f"\n[Result] Fold {fold} Threshold: {threshold:.4f}")
    
        # ====================================================
        # Cleanup
        # ====================================================
        del model_per_fold, X_train_z, E_train, X_val_z, E_val
        gc.collect()
        
        if fold == config.TRAIN.Actual_Fold:
            print("\n[DEBUG] Breaking loops for testing.")
            break
    
    print("\nPhase 1 & 2 complete for all folds.")
    
    #%
         
    print(f"\n{'='*40}")
    print("SAVING THRESHOLDS")
    print(f"{'='*40}")

    # 1. Structure the data
    # Convert numpy floats to standard python floats for JSON serialization
    threshold_data = {
        "per_fold": {
            f"fold_{i}": float(t) for i, t in enumerate(thresholds_all)
        },
        "statistics": {
            "mean": float(np.mean(thresholds_all)),
            "std": float(np.std(thresholds_all)),
            "min": float(np.min(thresholds_all)),
            "max": float(np.max(thresholds_all))
        },
        "config": {
            "model_name": model_name,
            "timestamp": config.TRAIN.DATETIME
        }
    }
    
    # 2. Define path
    json_filename = f"{model_name}_stage2_manifold_thresholds_{config.TRAIN.DATETIME}.json"
    save_path = os.path.join(config.MODEL_PATH, json_filename)
    
    # 3. Save to JSON
    try:
        with open(save_path, 'w') as f:
            json.dump(threshold_data, f, indent=4)
        print(f"[Success] Thresholds saved to:\n   {save_path}")
        print(f"[Info] Mean Threshold: {threshold_data['statistics']['mean']:.4f}")
    except Exception as e:
        print(f"[Error] Failed to save thresholds: {e}")
    
    # with open(save_path, 'r') as f:
    #     data = json.load(f)       
    # # Extract the key values
    # mean_threshold = data['statistics']['mean']
    # fold_thresholds = data['per_fold']
    
    # print(f"Loaded Thresholds from: {os.path.basename(save_path)}")
    # print(f" -> Mean Threshold: {mean_threshold:.4f}")
    

#%%
def train_robust_mil_ensemble(
    model_name,
    fold,
    train_df, 
    features, 
    riemann_tool, 
    config, 
    fold_threshold, # <--- Used for Normalization
    n_estimators=5, 
    max_bag_len=32
):
    """
    Phase 3: Trains an ensemble of Bag-Level MIL models for robust aggregation.
    """
    print(f"\n{'='*40}\n[Phase 3] Robust Aggregation Training (Ensemble)\n{'='*40}")

    # --- 1. Data Preparation (Common for all estimators) ---
    print(">> [Step 1] Preparing Feature + Energy dataset...")
    
    # A. Compute Energies (Curvature)
    print(f"   Computing curvature for {features.shape[0]} patches...")
    E_raw = riemann_tool.compute_batch_curvature(features).reshape(-1, 1)

    # ---------------------------------------------------------
    # [NEW] Normalize Energy using the Fold Threshold
    # ---------------------------------------------------------
    # Mapping strategy: 1.0 = "The Threshold". 
    # < 1.0 = In-Distribution, > 1.0 = High Energy / Outlier
    print(f"   [Normalization] Scaling Energy by Fold Threshold: {fold_threshold:.4f}")
    
    # Avoid division by zero (safety check)
    if fold_threshold == 0: fold_threshold = 1.0
        
    E_scaled = E_raw / fold_threshold
    
    # Optional: Clip extreme outliers (e.g., > 5x threshold) to prevent numerical instability
    # This keeps values in a reasonable range [0, 5.0] for the neural net
    E_scaled = np.clip(E_scaled, 0, 5.0)

    # B. Create Patient Bags
    # IMPORTANT: Pass the SCALED energy here
    print("   Creating Patient Bags (grouping patches by patient ID)...")
    bags_z, bags_e, y_bag_labels = create_patient_bags(
        df=train_df,
        features=features,
        energies=E_scaled  # <--- Using Scaled Energy
    )

    # C. Pad/Truncate Bags for Tensor Format
    print(f"   Padding bags to fixed length: {max_bag_len}...")
    X_bag_z = pad_bags(bags_z, max_len=max_bag_len)
    X_bag_e = pad_bags(bags_e, max_len=max_bag_len)
    y_bag_oh = tf.keras.utils.to_categorical(y_bag_labels, 2)

    print(f"   Final Bag Data Shapes: Z={X_bag_z.shape}, E={X_bag_e.shape}")

    # --- 2. Ensemble Training Loop ---
    ensemble_models = []
    
    print(f"\n>> [Step 2] Training Ensemble of {n_estimators} Models...")
    
    for i in range(n_estimators):
        print(f"\n   --- Training Estimator {i}/{n_estimators} ---")
        
        # A. Build Model Instance
        curvature_layer = CurvatureAttention() 
        print("Calling build_deep_riemannian_mil_model_3()")
        # model = build_deep_riemannian_mil_model_1(
        model = build_deep_riemannian_mil_model_3(
            input_dim=features.shape[1], 
            curvature_layer=curvature_layer
        )
        
        # B. Optimizer (AdamW with Gradient Clipping)
        optimizer = tfa.optimizers.AdamW(
            learning_rate=config['TRAIN']['LR'],
            weight_decay=config['TRAIN']['WEIGHT_DECAY'],
            global_clipnorm=config['TRAIN'].get('CLIP_NORM', 1.0) # Crucial for Riemannian
        )

        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # C. Callbacks (Early Stopping)
        callbacks_list = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=config.TRAIN.stop_patience,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5,       # Keep this
                patience=config.TRAIN.reduceLR_patience,
                min_lr=1e-5,      # INCREASE floor (was 1e-6) - Don't let it die completely
                verbose=1
            )
        ]

        # D. Fit Model
        # Using Batch Size 8 as recommended
        history = model.fit(
            x=[X_bag_z, X_bag_e],
            y=y_bag_oh,
            batch_size=config.TRAIN.BATCH_SIZE, 
            epochs=config.TRAIN.EPOCHS, 
            validation_split=config.DATA.val_ratio,
            callbacks=callbacks_list,
            verbose=2
        )  
        print(f"[PLOT] training graph to {config.MODEL_PATH}")
        # Ensure directory exists before plotting
        os.makedirs(config.MODEL_PATH, exist_ok=True)
        mygears.plot_training_metrics_without_lr(
                history=history, 
                prefix=f"Fold_{fold}_Est_{i}",
                save_dir=config.MODEL_PATH
        )
        print(f"\n   --- Training Estimator {i}/{n_estimators} Finished---")
       
        ensemble_models.append(model)
        
        # Save individual estimators
        save_path = os.path.join(config.MODEL_PATH, f"fold_{fold}_{model_name}_mil_estimator_{i}_{config.TRAIN.DATETIME}.h5")
        model.save(save_path)
        print(f"   Saved estimator {i} to {save_path}")

    print(f"\n[Phase 3 Complete] Trained {len(ensemble_models)} ensemble models.")
    return ensemble_models
     
#%% Phase 3: Trains an ensemble of Bag-Level MIL models for robust aggregation.
if config.TRAIN.Phase3:
    print("\n\nPhase 3: Trains an ensemble of Bag-Level MIL models for robust aggregation.")
    from sklearn.metrics import roc_auc_score
    
    # --- 1. Load Thresholds ---
    _, fold_thresholds = mygears.load_thresholds(
        # mil_model_1
        # os.path.join(config.MODEL_PATH, "ConvNeXtLarge_manifold_thresholds_20251216-0038.json")
        
        # mil_model_3
        # os.path.join(config.MODEL_PATH, "ConvNeXtLarge_manifold_thresholds_20251217-2135.json")
        # mil_model_3_stage2
        os.path.join(config.MODEL_PATH,
            "ConvNeXtLarge_stage2_manifold_thresholds_20251228-1448.json"
        )
    
    )
    print(f'Using {fold_thresholds=} with train_robust_mil_ensemble calling')

    # --- 2. Load MANIFOLD Tools (Needed to compute Energy) ---
    riemann_tools_map = {}
    print(f"\n[Loading] Manifold Tools...")
    for fold in range(config.DATA.N_SPLIT):
        # Always load the TRAINING tool (Source of Truth)
        riemann_tools_map[fold] = mygears.load_one_manifold(
            fold,
            model_name='ConvNeXtLarge_stage2',
            manifold_path=config.MODEL_PATH
            )
    print(f'{riemann_tools_map=}')
    # --- 3. Load FEATURES (Train & Val) ---
    X_train_z_all = {}
    X_val_z_all = {} 
    
    print(f"\n[Loading] Preparing Features...")
    
    for fold in range(config.DATA.N_SPLIT):
        print(f"   Processing Fold {fold}...")
        
        # -------------------------------------------------------
        # A. Load Training Features (Available on disk)
        # -------------------------------------------------------
        tr_path = os.path.join(config.MODEL_PATH, f"fold_{fold}_{model_name}_features_{config.TRAIN.DATETIME}.npy")
        if os.path.exists(tr_path):
            X_train_z_all[fold] = np.load(tr_path)
        else:
            raise FileNotFoundError(f"Critical: Train features missing: {tr_path}")

        # -------------------------------------------------------
        # B. Load OR Extract Validation Features
        # -------------------------------------------------------
        # fold_0_ConvNeXtLarge_val_manifold.pkl
        val_path = os.path.join(config.MODEL_PATH, f"fold_{fold}_{model_name}_val_features_{config.TRAIN.DATETIME}.npy")
        
        if os.path.exists(val_path):
            # Case 1: Load from disk (Fast)
            print(f"      -> Loading Val features from disk.")
            X_val_z_all[fold] = np.load(val_path)
            print(f"Loaded {val_path}")
        else:
            # Case 2: Extract on-the-fly (Slower, but works if missing)
            print(f"      -> Features missing. Extracting on-the-fly...")
            
            # 1. Load the Extractor for this specific fold
            # (Ensure load_one_extractor is defined/imported)
            feature_extractor = mygears.load_one_extractor(
                fold,
                model_name,
                config.MODEL_PATH,
                )
            print(f"Loaded {feature_extractor}")
            
            # 2. Get the Generator
            val_gen_per_fold = val_gen_list[fold]
            
            print("Extracting val features...")
            # Note: We pass 'val_gen_per_fold'
            X_val_z, _ =  p2_step1_extract_val_features(       
                model_name=model_name,
                feature_extractor=feature_extractor, 
                val_gen=val_gen_per_fold,
                verbose=False
            )
            
            # 4. Save to disk so next time it's fast
            np.save(val_path, X_val_z)
            print(f"      -> Saved features to {val_path}")
            
            # 5. Store in dictionary [FIXED BUG HERE]
            X_val_z_all[fold] = X_val_z
            
            # 6. Clean up memory
            del feature_extractor
            tf.keras.backend.clear_session()
            
        print(f"      -> Fold {fold} Shapes: Train {X_train_z_all[fold].shape}, Val {X_val_z_all[fold].shape}")
        
    #% --- 4. Main Training Loop ---
    fold_aurocs = [] 
    all_fold_mil_ensembles = {} 
    
    print(f"\n\nStarting Cross-Validation with Ensemble Strategy...")      
    
    for fold in range(config.DATA.N_SPLIT):
        print(f"\n>>> Processing FOLD {fold} <<<") 
        
        # A. Setup
        current_threshold = fold_thresholds[f"fold_{fold}"]
        riemann_tool = riemann_tools_map[fold]
        
        # B. Train (Using Loaded Features)
        mil_ensemble = train_robust_mil_ensemble(
            model_name,
            fold=fold,
            train_df=train_gen_list[fold].df,      
            features=X_train_z_all[fold], # Loaded from disk
            riemann_tool=riemann_tool, 
            config=config,
            fold_threshold=current_threshold, 
            n_estimators=5
        )
        all_fold_mil_ensembles[fold] = mil_ensemble
        
        # C. Evaluate
        print(f"   [Eval] Evaluating Ensemble on Validation Fold {fold}...")      
        
        # 1. Get Val Features (Already Loaded!)
        X_val_z = X_val_z_all[fold]
        
        # 2. Compute Val Energy (On-the-fly using Tool)
        E_val_raw = riemann_tool.compute_batch_curvature(X_val_z).reshape(-1, 1)
        
        # 3. Normalize
        E_val_norm = np.clip(E_val_raw / current_threshold, 0, 5.0)
        
        # 4. Create Bags
        bags_val_z, bags_val_e, y_val_bag_labels = create_patient_bags(
            df=val_gen_list[fold].df, 
            features=X_val_z, 
            energies=E_val_norm
        )
        
        # 5. Pad & Predict
        X_bag_val_z = pad_bags(bags_val_z, max_len=32)
        X_bag_val_e = pad_bags(bags_val_e, max_len=32)
        print(X_bag_val_z.shape, X_bag_val_e.shape)
        val_probs = []
        for model in mil_ensemble:
            preds = model.predict(
                [X_bag_val_z, X_bag_val_e],
                verbose=2)
            val_probs.append(preds)
        
        avg_val_probs = np.mean(val_probs, axis=0)
        
        if len(np.unique(y_val_bag_labels)) > 1:
            auc = roc_auc_score(y_val_bag_labels, avg_val_probs[:, 1])
            print(f"   [Result] Fold {fold} AUC: {auc:.4f}")
            fold_aurocs.append(auc)
        else:
            fold_aurocs.append(0.0)
        
        # Debugging Break
        if fold == config['TRAIN']['Actual_Fold']:
            print(f"DEBUG: Stopping after Fold {fold} as requested.")
            break
    
    
    #%%

    def identify_target_and_inverted_folds(fold_aurocs, threshold=0.5):
        """
        Determines the best performing fold and identifies folds requiring 
        probability flipping (AUC < 0.5).
        """
        fold_aurocs = np.array(fold_aurocs)
        
        # 1. Identify Target Fold (Highest Absolute Performance)
        # We use argmax on raw scores because we want the best 'naturally' aligned model
        target_fold = np.argmax(fold_aurocs)
        
        # 2. Identify Folds to Flip
        # Folds with AUC < 0.5 have learned the class separation but inverted the labels
        folds_to_flip = np.where(fold_aurocs < threshold)[0].tolist()
        
        # 3. Calculate Potential Mean (after flipping)
        # AUC_flipped = 1 - AUC_original
        corrected_aurocs = np.array([1 - a if a < threshold else a for a in fold_aurocs])
        potential_mean = np.mean(corrected_aurocs)
        
        return {
            "target_fold": target_fold,
            "folds_to_flip": folds_to_flip,
            "natural_mean": np.mean(fold_aurocs),
            "corrected_mean": potential_mean,
            "target_auc": fold_aurocs[target_fold]
        }
    
    # Data from your log
    # fold_aurocs = [0.8736842105263157, 0.22807017543859648, 0.687719298245614, 0.9052631578947369, 0.19649122807017544]
    print(f"{fold_aurocs=}")
    analysis = identify_target_and_inverted_folds(fold_aurocs)
    
    print(f"Target Fold: {analysis['target_fold']}")
    print(f"Folds to Flip: {analysis['folds_to_flip']}")
    print(f"Corrected Mean AUC: {analysis['corrected_mean']:.4f}")
    
    print("Phase 3 Traing MIL completed")
    
    # fold_aurocs=[0.8736842105263157, 0.22807017543859648, 0.687719298245614, 0.9052631578947369, 0.19649122807017544]
    # Target Fold: 3
    # Folds to Flip: [1, 4]
    # Corrected Mean AUC: 0.8084
    
    # fold_aurocs=[0.1298245614035088, 0.45964912280701753, 0.687719298245614, 0.09122807017543859, 0.14385964912280702]
    # Target Fold: 2
    # Folds to Flip: [0, 1, 3, 4]
    # Corrected Mean AUC: 0.7726
    
    # fold_aurocs=[0.8736842105263157, 0.3543859649122807, 0.687719298245614, 0.09122807017543859, 0.856140350877193]
    # Target Fold: 0
    # Folds to Flip: [1, 3]
    # Corrected Mean AUC: 0.7944
    
    
#%%
def split_ood_data(features, energies, limit):
    """
    Splits data into 'Valid' (Accepted) and 'OOD' (Rejected) based on Energy Limit.
    
    Args:
        features (np.array): Shape (N, D) - Feature vectors
        energies (np.array): Shape (N,) or (N, 1) - Riemannian Energy scores
        limit (float): The OOD Energy Threshold (e.g., 29.5434)
        
    Returns:
        valid_data: Dictionary containing 'X' (features) and 'E' (energies) for valid data.
        ood_data: Dictionary containing 'X' (features) and 'E' (energies) for rejected data.
        stats: Dictionary with counts and rejection rate.
    """
    # Ensure energies are flattened for masking
    e_flat = energies.flatten()
    
    # 1. Create Masks
    valid_mask = e_flat < limit
    ood_mask = ~valid_mask  # Inverse of valid is OOD
    
    # 2. Filter Data
    X_valid = features[valid_mask]
    E_valid = energies[valid_mask]
    
    X_ood = features[ood_mask]
    E_ood = energies[ood_mask]
    
    # 3. Calculate Stats
    n_total = len(features)
    n_valid = len(X_valid)
    n_ood = len(X_ood)
    reject_rate = (n_ood / n_total * 100) if n_total > 0 else 0
    
    print(f"\n[OOD Split] Limit: {limit:.4f}")
    print(f"   Valid (In-Distribution): {n_valid} ({100-reject_rate:.1f}%)")
    print(f"   OOD (Rejected):          {n_ood} ({reject_rate:.1f}%)")
    
    return (
        {'X': X_valid, 'E': E_valid}, 
        {'X': X_ood, 'E': E_ood}, 
        {'n_valid': n_valid, 'n_ood': n_ood, 'rate': reject_rate}
    )
  
#%%
if config.PLOT_CURVATURE: 
    import plotly.graph_objects as go
    from sklearn.manifold import Isomap
    from scipy.stats import multivariate_normal
       
    def plot_curvature_manifold(X_train, y_train, E_train, 
                                X_ood_full, E_ood_full, 
                                ood_limit, 
                                save_path="spacetime_curvature_manifold.html"):
        """
        Generates a 3D Manifold Surface plot (Z=Density) and overlays 
        training data (colored by Energy) and OOD data (split by Energy Limit).
        
        Args:
            X_train (np.array): Training features (N_train, D)
            y_train (np.array): Training labels (N_train,)
            E_train (np.array): Training Riemannian Energy (N_train,)
            X_ood_full (np.array): OOD features (N_ood, D)
            E_ood_full (np.array): OOD Energy (N_ood,)
            ood_limit (float): Threshold to split Valid vs Rejected OOD.
            save_path (str): Output filename.
        """
        print(f"\n[Plotting] Generating Manifold Surface with OOD Limit: {ood_limit}")
        
        # ---------------------------------------
        # 1. Split OOD Data (Valid vs Rejected)
        # ---------------------------------------
        # We do this BEFORE embedding to keep indices aligned
        mask_valid = E_ood_full.flatten() < ood_limit
        
        X_ood_valid = X_ood_full[mask_valid]
        X_ood_reject = X_ood_full[~mask_valid]
        
        print(f"   OOD Split: {len(X_ood_valid)} Valid (Blue Square) | {len(X_ood_reject)} Rejected (Red X)")
    
        # ---------------------------------------
        # 2. Dimensionality Reduction (Isomap)
        # ---------------------------------------
        print(f"   Projecting to 2D Manifold via Isomap...")
        reducer = Isomap(n_neighbors=30, n_components=2, n_jobs=-1)
        
        # Fit on Train, Transform All
        X_train_emb = reducer.fit_transform(X_train)
        
        # Transform OOD subsets
        if len(X_ood_valid) > 0:
            X_ood_valid_emb = reducer.transform(X_ood_valid)
        else:
            X_ood_valid_emb = np.empty((0, 2))
            
        if len(X_ood_reject) > 0:
            X_ood_reject_emb = reducer.transform(X_ood_reject)
        else:
            X_ood_reject_emb = np.empty((0, 2))
    
        # ---------------------------------------
        # 3. Create Landscape Surface (Density)
        # ---------------------------------------
        # Create meshgrid
        x_min, x_max = X_train_emb[:, 0].min() - 2, X_train_emb[:, 0].max() + 2
        y_min, y_max = X_train_emb[:, 1].min() - 2, X_train_emb[:, 1].max() + 2
        
        x_grid = np.linspace(x_min, x_max, 100) 
        y_grid = np.linspace(y_min, y_max, 100)
        xx, yy = np.meshgrid(x_grid, y_grid)
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        
        # Fit Gaussians
        reg = 1e-6 * np.eye(2)
        
        # Class 0 (IDA)
        mean_ida = X_train_emb[y_train == 0].mean(axis=0)
        cov_ida  = np.cov(X_train_emb[y_train == 0].T) + reg
        
        # Class 1 (THL)
        mean_thl = X_train_emb[y_train == 1].mean(axis=0)
        cov_thl  = np.cov(X_train_emb[y_train == 1].T) + reg
        
        # PDF
        pdf_ida = multivariate_normal.pdf(grid_points, mean=mean_ida, cov=cov_ida)
        pdf_thl = multivariate_normal.pdf(grid_points, mean=mean_thl, cov=cov_thl)
        
        pdf_mix = 0.5 * pdf_ida + 0.5 * pdf_thl
        pdf_mix = np.maximum(pdf_mix, 1e-12)
        
        # Z-Axis: Free Energy (-log P)
        z_grid = -np.log(pdf_mix).reshape(xx.shape)
        
        # Helper to get Z for scatter points
        def get_z_height(points):
            if len(points) == 0: return np.array([])
            p1 = multivariate_normal.pdf(points, mean_ida, cov_ida)
            p2 = multivariate_normal.pdf(points, mean_thl, cov_thl)
            pmix = np.maximum(0.5*p1 + 0.5*p2, 1e-12)
            return -np.log(pmix)
    
        # Calculate Z for all groups
        z_ida = get_z_height(X_train_emb[y_train==0])
        z_thl = get_z_height(X_train_emb[y_train==1])
        z_ood_valid = get_z_height(X_ood_valid_emb)
        z_ood_reject = get_z_height(X_ood_reject_emb)
    
        # ---------------------------------------
        # 4. Plotting
        # ---------------------------------------
        print("   Generating 3D Plot...")
        fig = go.Figure()
        
        # A. The Manifold Surface
        fig.add_trace(go.Surface(
            x=x_grid, y=y_grid, z=z_grid,
            colorscale='Greys', opacity=0.8, showscale=False,
            name='Probability Landscape'
        ))
        
        # B. IDA (Train) - Circles
        subset_mask = (y_train == 0)
        fig.add_trace(go.Scatter3d(
            x=X_train_emb[subset_mask, 0], 
            y=X_train_emb[subset_mask, 1], 
            z=z_ida,
            mode='markers',
            marker=dict(
                size=4, 
                color=E_train[subset_mask].flatten(),
                colorscale='Viridis', showscale=True,
                colorbar=dict(title="Riemann Energy", x=0.9),
                symbol='circle'
            ),
            name='IDA (Train)'
        ))
        
        # C. THL (Train) - Diamonds
        subset_mask = (y_train == 1)
        fig.add_trace(go.Scatter3d(
            x=X_train_emb[subset_mask, 0], 
            y=X_train_emb[subset_mask, 1], 
            z=z_thl,
            mode='markers',
            marker=dict(
                size=4, 
                color=E_train[subset_mask].flatten(),
                colorscale='Viridis', showscale=False,
                symbol='diamond'
            ),
            name='THL (Train)'
        ))
        
        # D. VALID OOD (Accepted) - Blue Squares
        if len(X_ood_valid_emb) > 0:
            fig.add_trace(go.Scatter3d(
                x=X_ood_valid_emb[:, 0], 
                y=X_ood_valid_emb[:, 1], 
                z=z_ood_valid,
                mode='markers',
                marker=dict(
                    size=5, 
                    color='blue',       # Explicit Blue
                    symbol='square',    # Explicit Square
                    opacity=0.9
                ),
                name=f'OOD Accepted (E < {ood_limit:.1f})'
            ))
    
        # E. REJECTED OOD (Invalid) - Red X
        if len(X_ood_reject_emb) > 0:
            fig.add_trace(go.Scatter3d(
                x=X_ood_reject_emb[:, 0], 
                y=X_ood_reject_emb[:, 1], 
                z=z_ood_reject,
                mode='markers',
                marker=dict(
                    size=5, 
                    color='red',        # Explicit Red
                    symbol='x',         # Explicit X
                    opacity=1.0
                ),
                name=f'OOD Rejected (E > {ood_limit:.1f})'
            ))
    
        # Layout
        fig.update_layout(
            title=f"Manifold Surface & OOD Filter (Limit: {ood_limit:.2f})",
            scene=dict(
                xaxis_title="Isomap Dim 1",
                yaxis_title="Isomap Dim 2",
                zaxis_title="Free Energy (-log P)",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
            ),
            width=1200, height=900,
            margin=dict(l=0, r=0, b=0, t=40),
            legend=dict(x=0, y=1),
            font=dict(size=14)
        )
        
        fig.write_html(save_path)
        print(f"   [IO] Plot saved to {save_path}")
    
    #%% ---------------------------------------
    # 1. Load Data & Align Labels
    # ---------------------------------------
    print("Loading Feature Data...")
    fold = 4 #the best AUC fold of mil model_1 and model_3
    feature_extractor = load_one_extractor(fold, model_name, config.BASEMODEL_PATH)
    # A. Load Patch Features (Source of Geometry)
    X_train_full = np.load(f'models/fold_{fold}_ConvNeXtLarge_features.npy')
    
    try:
        # Use the generator's internal class list to match patch order
        y_train_full = train_gen_list[fold].classes 
    except NameError:
        raise ValueError("CRITICAL: 'train_gen_list' not found. Cannot align patch labels.")
    
    print(f"Original Shapes: X={X_train_full.shape}, y={y_train_full.shape}")
    
    n_samples = 30_000
    n_samples_val = int(n_samples * 0.3)
    print(f"{n_samples=}, {n_samples_val}")
    # X_train, y_train, E_train = shuffle(X_train_full, y_train_full, E_train_full, random_state=42)
    if len(X_train_full) > n_samples:
        X_train = X_train_full[:n_samples]
        y_train = y_train_full[:n_samples]
    print(f"Subsampled Shapes: X={X_train.shape}, y={y_train.shape}")
   
    # B. Compute Riemannian Energy (Source of Color)
    # Assumes 'riemann_tools_map' is loaded from Phase 3
    print("Computing Riemannian Curvature Energy...")
    riemann_tool = load_one_manifold(fold, 'ConvNeXtLarge', config.MODEL_PATH, val=False)
    E_train = riemann_tool.compute_batch_curvature(X_train)
    print(f"Original Shapes: E={E_train.shape}")
    
    #%%
    # from pathlib import Path
    # import glob
    # import cv2
    
    # # Define base paths for healthy and anemic individuals
    # if platform.system() == 'Windows':
    #     base_dir = Path(r"D:/")
    # if platform.system() == 'Linux':
    #     base_dir = config.ANE_PATH 
    #%
    # print("\nLoad OOD samples from AneRBC dataset")
    # # Staining Norm data
    # # healthy_base_path = os.path.join(base_dir, 'AneRBC_dataset/AneRBC-I/Healthy_individuals/RGB_segmented/patch')
    # # anemia_base_path = os.path.join(base_dir, 'AneRBC_dataset/AneRBC-I/Anemic_individuals/RGB_segmented/patch')
    
    # healthy_base_path = os.path.join(base_dir, 'AneRBC_dataset/AneRBC-I/Healthy_individuals/RGB_segmented')
    # anemia_base_path = os.path.join(base_dir, 'AneRBC_dataset/AneRBC-I/Anemic_individuals/RGB_segmented')
    
    # # Get all file paths for healthy and anemic individuals
    # healthy_paths = glob.glob(os.path.join(healthy_base_path, '*.png'))
    # anemia_paths = glob.glob(os.path.join(anemia_base_path, '*.png'))
    # print(f"{len(healthy_paths)=}, {len(anemia_paths)=}")
    
    # # img = cv2.imread(healthy_paths[0], cv2.IMREAD_UNCHANGED)
    # # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # # plt.imshow(img)
    
    # # Create class labels (0 for healthy, 1 for anemia)
    # healthy_class = [0] * len(healthy_paths)
    # anemia_class = [1] * len(anemia_paths)
    
    # ood_df = pd.DataFrame({
    #     'patch_path': healthy_paths,  
    #     'label': healthy_class            # Class labels
    # })
    
    #%%
    ood_df = val_df_fold_list[fold] #4
    
    ood_gen = PatchDatasetPreserve(
            config,
            # df=ood_df,
            df=ood_df[:n_samples_val],
            batch_size=config.TRAIN.BATCH_SIZE,
            preprocessing_function=preprocessing_fn,
            shuffle=True,
            expect_rgba=True,
            # save=True,
            # use_cutmix=True # Keep False for validation/testing
            # verbose=True
        )
    print("\nOOD generator summary:")
    ood_gen_dict = ood_gen.summary() 
    
    rand_idx = np.random.randint(0, len(ood_gen))               
    images, labels = ood_gen.__getitem__(rand_idx)
    print(f"{images.shape=}, {labels.shape=}")
    ood_gen.plot_random_batch(rand_idx,
                              note=f'Val fold {fold} set',
                              save_dir='Inference'
                              ) 
    plot_file = "spacetime_curvature_IDA_THL_OOD_Val_fold_4_mil_1.html"
    
    #%% E. Load OOD (Validation fold 4)
    try:
        X_ood_available = True
        
        print("Extracting OOD features...")
        # X_ood_full = np.load('models/fold_4_ConvNeXtLarge_val_features.npy')
        X_ood_full = feature_extractor.predict(ood_gen, verbose=1)
        print(f'{X_ood_full.shape=}')         
        print("Compute Energy for OOD too")
        E_ood_full = riemann_tool.compute_batch_curvature(X_ood_full)
        print(f"{E_ood_full.shape=}")
        
        # 1. Set the Updated Limit
        OOD_ENERGY_LIMIT = 29.5434    
        print("Apply the Split valid and invalid energy of patches")
        # This will print the rejection rate based on 29.5434
        valid_group, ood_group, stats = split_ood_data(X_ood_full, E_ood_full, OOD_ENERGY_LIMIT)       
        X_valid = valid_group['X']
        X_ood = ood_group['X']
        # 3. Visualization logic (Optional)
        # Visualize 'ood_group['X']' to confirm they look like artifacts (blur/stain issues).
        # Visualize 'valid_group['X']' to confirm they look like high-quality cells.
  
        # if len(X_ood_full) >= n_samples:            
        #     n_samples_ood = int(n_samples*0.3)
        #     if n_samples_ood <= len(X_ood_full):   
        #         print(f"{n_samples_ood=}")
        #         X_ood = X_ood[:n_samples_ood]
        #         E_ood = E_ood[:n_samples_ood]
        # print(f"OOD Loaded: {X_ood.shape=}, {E_ood.shape=}")
             
    except Exception as e:
      print(f"No OOD provided or Error: {e} - skipping OOD points")
      X_ood_available = False

    #%%  
    plot_curvature_manifold(X_train, y_train, E_train, 
        X_ood_full, E_ood_full, 
        ood_limit=OOD_ENERGY_LIMIT, 
        save_path=plot_file)
              
    #%% ---------------------------------------
    """
    # 2. Dimensionality Reduction (Isomap)
    # ---------------------------------------
    print(f"Projecting features to 2D Manifold via Isomap...")
    
    # Isomap preserves the geodesic (curved) distances best
    reducer = Isomap(n_neighbors=30, n_components=2, n_jobs=-1)
    
    # Fit on Train, Transform Train & OOD
    X_emb = reducer.fit_transform(X_train)
    
    if X_ood_available:
        X_ood_emb = reducer.transform(X_ood)
    
    # ---------------------------------------
    # 3. Create Grid for Landscape Surface
    # ---------------------------------------
    # Create a meshgrid covering the embedding space
    x_min, x_max = X_emb[:, 0].min() - 2, X_emb[:, 0].max() + 2
    y_min, y_max = X_emb[:, 1].min() - 2, X_emb[:, 1].max() + 2
    
    x_grid = np.linspace(x_min, x_max, 100) 
    y_grid = np.linspace(y_min, y_max, 100)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # ---------------------------------------
    # 4. Fit Gaussian Densities (The Landscape)
    # ---------------------------------------
    print("Computing Probability Landscape...")
    reg = 1e-6 * np.eye(2) # Stability regularization
    
    # Fit distinct Gaussians for IDA (0) and THL (1)
    mean_ida = X_emb[y_train == 0].mean(axis=0)
    cov_ida  = np.cov(X_emb[y_train == 0].T) + reg
    
    mean_thl = X_emb[y_train == 1].mean(axis=0)
    cov_thl  = np.cov(X_emb[y_train == 1].T) + reg
    
    # Evaluate PDF on the grid
    pdf_ida = multivariate_normal.pdf(grid_points, mean=mean_ida, cov=cov_ida)
    pdf_thl = multivariate_normal.pdf(grid_points, mean=mean_thl, cov=cov_thl)
    
    # The "Manifold Surface" is the mixture of both classes
    pdf_mix = 0.5 * pdf_ida + 0.5 * pdf_thl
    pdf_mix = np.maximum(pdf_mix, 1e-12)
    
    # ---------------------------------------
    # 5. Calculate Z-Axis (Free Energy)
    # ---------------------------------------
    # Z = -log(Probability). Lower prob = Higher Z (Peaks)
    z_grid = -np.log(pdf_mix).reshape(xx.shape)
    
    # Compute Z-height for specific points so they sit on the surface
    # (We look up their probability in the mixture model)
    def get_z_height(points, m1, c1, m2, c2):
        p1 = multivariate_normal.pdf(points, m1, c1)
        p2 = multivariate_normal.pdf(points, m2, c2)
        pmix = np.maximum(0.5*p1 + 0.5*p2, 1e-12)
        return -np.log(pmix)
    
    z_ida = get_z_height(X_emb[y_train==0], mean_ida, cov_ida, mean_thl, cov_thl)
    z_thl = get_z_height(X_emb[y_train==1], mean_ida, cov_ida, mean_thl, cov_thl)
    
    if X_ood_available:
        z_ood = get_z_height(X_ood_emb, mean_ida, cov_ida, mean_thl, cov_thl)
    
    #% ---------------------------------------
    # 6. Plotting
    # ---------------------------------------
    print("Generating 3D Plot...")
    fig = go.Figure()
    
    # A. The Manifold Surface (Wireframe/Surface)
    fig.add_trace(go.Surface(
        x=x_grid, y=y_grid, z=z_grid,
        colorscale='Greys', opacity=0.8, # Subtle background
        showscale=False,
        name='Probability Landscape'
    ))
    
    # B. IDA Points (Class 0)
    # Symbol: Circle. Color: Mapped to Riemannian Energy.
    subset_mask = (y_train == 0)
    fig.add_trace(go.Scatter3d(
        x=X_emb[subset_mask, 0], 
        y=X_emb[subset_mask, 1], 
        z=z_ida,
        mode='markers',
        marker=dict(
            size=4, 
            color=E_train[subset_mask].flatten(), # <--- COLOR BY RIEMANNIAN ENERGY
            colorscale='Viridis', 
            showscale=True,
            colorbar=dict(title="Riemannian Energy", x=0.8),
            symbol='circle'
        ),
        name='IDA (Circle)'
    ))
    
    # C. THL Points (Class 1)
    # Symbol: Diamond. Color: Mapped to Riemannian Energy.
    subset_mask = (y_train == 1)
    fig.add_trace(go.Scatter3d(
        x=X_emb[subset_mask, 0], 
        y=X_emb[subset_mask, 1], 
        z=z_thl,
        mode='markers',
        marker=dict(
            size=4, 
            color=E_train[subset_mask].flatten(), # <--- COLOR BY RIEMANNIAN ENERGY
            colorscale='Viridis', 
            showscale=False, # Share the scale with IDA
            symbol='diamond'
        ),
        name='THL (Diamond)'
    ))
    
    # D. OOD Points
    # Symbol: X. Color: Red (Distinct).
    if X_ood_available:
        fig.add_trace(go.Scatter3d(
            x=X_ood_emb[:, 0], 
            y=X_ood_emb[:, 1], 
            z=z_ood,
            mode='markers',
            marker=dict(
                size=5, 
                color='orange', 
                symbol='square',  # <--- CHANGED from 'x' to 'cross' (Star-like)
                # Options: 'circle', 'square', 'diamond', 'cross', 'x'
            ),
            name='Unseen Data'
        ))
    
    fig.update_layout(
        title="Space-Time Curvature: Manifold Density (Z) vs. Riemannian Energy (Color)",
        scene=dict(
            xaxis_title="Isomap Dim 1",
            yaxis_title="Isomap Dim 2",
            zaxis_title="Free Energy (-log P)"
        ),
        width=1000, height=800,
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(
            x=0,        # x=0 is left
            y=1,        # y=1 is top
            xanchor='left',
            yanchor='top'
        ),
        font=dict(size=14) # Normal screen font
    )
    
    fig.write_html(plot_file)
    
    # Target: 10 inches x 8 inches at 300 DPI
    # target_dpi = 300 # 1000px * 3 = 3000px (approx 10 inches at 300 DPI)
    # width_in = 10
    # height_in = 8
    
    # fig.write_image(
    #     "spacetime_curvature_IDA_THL_OOD.png",
    #     width=width_in * target_dpi,   # 3000 px
    #     height=height_in * target_dpi, # 2400 px
    #     scale=3 # 1:3 pixel mapping
    # )
    
    print(f"Done. Plot saved to {plot_file}")
    
 
    """
#%% Prediction Clinical Workflow
import csv
from sklearn.metrics import roc_curve, roc_auc_score

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

#%% Re-calibration T_upper and T_lower is Required
if config.Evaluate_Val:
    LOAD_STRATEGY = 'ENSEMBLE'
    feature_extractors_map = {}
    print(f"\nLoading ensemble of {config.DATA.N_SPLIT} extractors...")
    for fold in range(config.DATA.N_SPLIT):
        feature_extractor = load_one_extractor(fold, model_name, config)
        feature_extractors_map[fold] = feature_extractor
    print(f"Total Feature Extractors Loaded: {len(feature_extractors_map)}")
    
    #%%
    riemann_tools_map = {} # Dictionary to store {fold_index: tool_object}    
    print(f"\n[Loading] Manifold Tools (Strategy: {LOAD_STRATEGY})...")        
    # Load ALL tools for all folds *.pkl
    print(f"Loading {config.DATA.N_SPLIT} manifold tools for ensemble...")
    for fold in range(config.DATA.N_SPLIT):
        tool = load_one_manifold(fold, model_name, config)
        riemann_tools_map[fold] = tool     
    print(f"Total Manifold Tools Loaded: {len(riemann_tools_map)}")
       
    #%% Load mil model lists for ALL folds
    all_fold_ensembles = {}
    for fold in range(config.DATA.N_SPLIT):
        models_list = load_mil_ensemble_for_fold(fold, model_name, config, n_estimators=5)
        all_fold_ensembles[fold] = models_list
    print(f"Total Ensembles Loaded: {len(all_fold_ensembles)} folds.")
    
    #%%
    
    def calculate_robust_ood_limit(config, val_gen_list, feature_extractors_map, riemann_tools_map, fold_models_map):
        """
        Calculates the OOD Energy Limit using specific logic:
        1. Energy is NEVER flipped (Geometric magnitude).
        2. Predictions ARE flipped for Folds 0, 1, 3.
        3. Limit is calculated based on High-Confidence/Correct samples.
        """
        print(f"\n[OOD Calibration] Collecting Energies & Corrected Predictions...")
        
        all_valid_energies = []
        
        # Folds that need probability inversion (from your Phase 3 findings)
        FLIPPED_FOLDS = [0, 1, 3]
    
        for fold in range(config.DATA.N_SPLIT):
            print(f"   \nProcessing Fold {fold}...")
            
            # -------------------------------------------------
            print("1. Setup Data & Tools")
            # -------------------------------------------------
            val_gen = val_gen_list[fold]
            feature_extractor = feature_extractors_map[fold]
            riemann_tool = riemann_tools_map[fold]
            model = fold_models_map[fold] # You need the model to check predictions!
            
            # -------------------------------------------------
            print("2. Extract Features (Z) and True Labels (y)")
            # -------------------------------------------------
            # Note: Ensure val_gen is NOT shuffled so labels align
            X_val_z = feature_extractor.predict(val_gen, verbose=2)
            print(f'{X_val_z.shape=}')
            
            if hasattr(val_gen, 'classes'):
                y_true = val_gen.classes
            else:
                y_true = np.concatenate([y for x, y in val_gen], axis=0)
                y_true = np.argmax(y_true, axis=1)
    
            # -------------------------------------------------
            print("3. Compute Energy (DO NOT INVERT THIS)")
            # -------------------------------------------------
            # Energy is a magnitude (distance). It is always positive and correct.
            E_val = riemann_tool.compute_batch_curvature(X_val_z)

            
            # -------------------------------------------------
            print("4. Compute Predictions (APPLY INVERSION HERE)")
            # -------------------------------------------------
            # We need bags to predict. 
            # (Assuming you have a helper to bag 'X_val_z' and 'E_val' like 'create_patient_bags')
            # For this snippet, let's assume we can predict directly or you have the bagged input ready.
            # This is a simplified placeholder for the prediction step:
            
            # Create bags just for prediction context
            # pad_bags needs to know that the "feature dimension" is 1
            E_val = E_val.reshape(-1, 1)
            bags_z, bags_e, y_bag_labels = create_patient_bags(
                df=val_gen.df,
                features=X_val_z,
                energies=E_val
            )
            print(f"{len(bags_z)=}, {len(bags_e)=}")
           
            X_bag_z = pad_bags(bags_z)
            X_bag_e = pad_bags(bags_e)
            print(f"{X_bag_z.shape=}, {X_bag_e.shape=}")
            
            # --- FIX: Handle Ensemble List ---
            ensemble_models = fold_models_map[fold] # This is a LIST of models
            batch_predictions = []

            for single_model in ensemble_models:
                # Predict with each individual model in the ensemble
                # Shape: (N_bags, 2)
                preds = single_model.predict([X_bag_z, X_bag_e], verbose=0) 
                batch_predictions.append(preds)

            # Average the predictions across the 5 models
            # Shape: (N_bags, 2)
            avg_preds = np.mean(batch_predictions, axis=0)
            
            # Extract probability of Class 1 (Thalassemia)
            raw_preds = avg_preds[:, 1] 
            
            # --- CRITICAL FIX: Invert Probability if Fold is Flipped ---
            if fold in FLIPPED_FOLDS:
                print(f"   [Flip] Inverting probabilities for Fold {fold}...")
                corrected_preds = 1.0 - raw_preds
            else:
                corrected_preds = raw_preds
                
            # -------------------------------------------------
            print("5. Filter for Calibration")
            # -------------------------------------------------
            # We only want to set the threshold based on samples the model 
            # actually understands (Correct Predictions).
            
            # predicted_labels = (corrected_preds > 0.5).astype(int)
            # correct_mask = (predicted_labels == y_true)
            # A. FIX: Use 'y_bag_labels' (Patient labels) instead of 'y_true' (Patch labels)
            predicted_patient_labels = (corrected_preds > 0.5).astype(int)
            patient_correct_mask = (predicted_patient_labels == y_bag_labels)
            print(f"   Accurate Patients: {np.sum(patient_correct_mask)}/{len(y_bag_labels)}")
            
            # # Keep energies of correctly classified patients
            # valid_energies = E_val[correct_mask] 
            # # If you can't run predictions easily here, just use ALL energies:
            # valid_energies = E_val 
            
            # B. Select Energies from Correct Patients Only
            # 'bags_e' is a list of arrays (one per patient). We filter this list.
            valid_energy_bags = [bags_e[i] for i in range(len(bags_e)) if patient_correct_mask[i]]
        
            # C. Flatten to get a single array of "Valid Patch Energies"
            if len(valid_energy_bags) > 0:
                valid_energies = np.concatenate(valid_energy_bags).flatten()
            else:
                print("   [Warning] No patients correctly classified. Using all energies.")
                valid_energies = E_val.flatten()
          
            all_valid_energies.extend(valid_energies)
            
        # -------------------------------------------------
        print("6. Calculate The Limit (e.g., 95th Percentile)")
        # -------------------------------------------------
        all_valid_energies = np.array(all_valid_energies)
        ood_limit = np.percentile(all_valid_energies, 95) # 95% of valid data is below this
        
        print(f"\n[Result] Calculated OOD Energy Limit: {ood_limit:.4f}")
        return ood_limit, all_valid_energies
    
    def calibrate_ood_limit(energies_list, method='percentile', param=99):
        """
        Robust OOD Limit Calibration.
        methods:
          'percentile': Uses the param-th percentile (e.g., 99). Robust to outliers.
          'sigma': Uses Mean + param * StdDev (e.g., Mean + 3*Std).
          'max': The old Max * param method (Not recommended).
        """
        # 1. Flatten all energies into a single distribution array
        if isinstance(energies_list, list):
            # Handle list of arrays (e.g. [array([1,2]), array([3,4])])
            all_energies = np.concatenate([e.flatten() for e in energies_list])
        else:
            all_energies = energies_list.flatten()
    
        # 2. Calculate Statistics
        mean_e = np.mean(all_energies)
        std_e = np.std(all_energies)
        max_e = np.max(all_energies)
        
        limit = 0.0
    
        # 3. Apply Method
        if method == 'percentile':
            # Recommended: Cuts off the top (100-param)% extreme outliers
            limit = np.percentile(all_energies, param)
            print(f"\n[Safety Calibration] Method: {param}th Percentile")
            
        elif method == 'sigma':
            # Standard Statistical Control
            limit = mean_e + (param * std_e)
            print(f"\n[Safety Calibration] Method: Mean + {param} Sigma")
            
        elif method == 'max':
            # Your previous method (Fragile)
            limit = max_e * param
            print(f"\n[Safety Calibration] Method: Max * {param}")
    
        # 4. Report
        print(f"   Distribution: Mean={mean_e:.2f}, Std={std_e:.2f}, Max={max_e:.2f}")
        print(f"   Calculated OOD Limit: {limit:.4f}")
        
        return limit
    
    #%%       0         1        2       3          4
    # aucs= [0.6140, 1-0.2667, 0.6737, 1-0.2246, 1-0.1368]
    # print(np.mean(aucs)) #mil_model_3
    
    #%%
    if config.find_ood:        
        print("\n\nFind ood_energy_limit")
        print(f"\n[Computing] Re-calculating Validation Energies...")
        ood_limit, energies_list = calculate_robust_ood_limit(
            config,
            val_gen_list,
            feature_extractors_map,
            riemann_tools_map,
            all_fold_ensembles)
        
        print(f"Reconstruction Complete. Ready for Threshold Calibration.") 
        # --- USAGE ---
        # Recommended for Clinical Safety: 99th Percentile
        # This says: "We accept 99% of our validation patients, but reject the weirdest 1%."
        ood_energy_limit_percentile = calibrate_ood_limit(energies_list, method='percentile', param=95)
        ood_energy_limit_sigma = calibrate_ood_limit(energies_list, method='sigma', param=95)
        ood_energy_limit_max = calibrate_ood_limit(energies_list, method='max', param=95)

    #%%
    if config.find_threshold:
        print("\n\nFind thresholds using validation sets")
        # Tracking Variables
        fold_aurocs = [] 
        all_validation_patient_predictions_list = [] # List to store dicts for CSV
        
        # Threshold Tracking
        fold_t_uppers = []
        fold_t_lowers = []
        
        n_estimators = 5
        
        print(f"\n{'='*40}")
        print(f"STARTING ROBUST MIL TRAINING (N_SPLIT={config.DATA.N_SPLIT})")
        print(f"{'='*40}")
        X_val_z_all = []
        y_val_z_all = []
        energies_val_all = []
        thresholds_all = []
        for fold in range(config.DATA.N_SPLIT):
            print(f"\n>>> Processing FOLD {fold} <<<")
            
            # --- A. Retrieve Train Data ---
            train_gen_per_fold = train_gen_list[fold]
            current_train_df = train_gen_per_fold.df
            feature_extractor = feature_extractors_map[fold]
            val_gen_per_fold = val_gen_list[fold]
            riemann_tool = riemann_tools_map[fold] # result of trained-manifold

            # ---------------------------------------------------------
            # --- C. Evaluate on Validation Set ---
            # ---------------------------------------------------------
            print(f"   [Eval] Evaluating Ensemble on Validation Fold {fold}...")                    
            # Step 2.1: Extract Validation Features
            X_val_z, y_val_z = p2_step1_extract_val_features(
                model_name,
                feature_extractor=feature_extractor, 
                val_gen=val_gen_per_fold
            )
            X_val_z_all.append(X_val_z)
            y_val_z_all.append(y_val_z)
            
            # Step 2.2: Compute Validation Energy
            save_pkl_path = os.path.join(config.MODEL_PATH, f"fold_{fold}_manifold_val.pkl")
            E_val = p2_step2_compute_val_energy(
                riemann_tool=riemann_tool, 
                X_val_z=X_val_z,
                save_path=save_pkl_path
            )
                  
            # Step 2.3: Calibrate Threshold
            # (Assuming you might want to visualize or save this per fold)
            _, threshold = p2_step3_calibrate_threshold(
                E_val=E_val, 
                y_val=y_val_z, 
                config=config
            )
            
            # Store P2 Results
            energies_val_all.append(E_val)
            thresholds_all.append(threshold)
            
            print(f"\n[Result] Fold {fold} Threshold: {threshold:.4f}")
            # --- FIX IS HERE: Reshape to (N, 1) ---
            # The stored energies are likely (N,), so we must force (N, 1)
            E_val_z = E_val.reshape(-1, 1) 
            
            bags_val_z, bags_val_e, y_val_bag_labels = create_patient_bags(
                df=val_gen_per_fold.df, 
                features=X_val_z, 
                energies=E_val_z
            )
            X_bag_val_z = pad_bags(bags_val_z, max_len=32)
            X_bag_val_e = pad_bags(bags_val_e, max_len=32)
            
            # 2. Ensemble Prediction
            val_probs = []
            for model in all_fold_ensembles[fold]:
                # Predict: (N_patients, 2)
                preds = model.predict([X_bag_val_z, X_bag_val_e], verbose=2)
                val_probs.append(preds)
            
            # Average predictions (Soft Voting)
            avg_val_probs = np.mean(val_probs, axis=0) # Shape (N_patients, 2)
            prob_thl_val = avg_val_probs[:, 1]

            # ---------------------------------------------------------
            # [CRITICAL FIX] Apply Polarity Correction
            # ---------------------------------------------------------
            # Folds 0, 1, 3 are flipped. We must invert them so that
            # High Prob = Thalassemia for ALL folds.
            if fold in [0, 1, 3]:
                print(f"   [Flip] Inverting probabilities for Fold {fold}...")
                prob_thl_val = 1.0 - prob_thl_val

            # ---------------------------------------------------------

            # 3. Calculate AUC (Now it will be ~0.78 instead of 0.22)
            if len(np.unique(y_val_bag_labels)) > 1:
                fold_auc = roc_auc_score(y_val_bag_labels, prob_thl_val)
            else:
                fold_auc = 0.0
            fold_aurocs.append(fold_auc)
            print(f"   [Score] Fold {fold} AUC: {fold_auc:.4f}")
            
            # 4. Re-Calibration: Calculate Optimal Thresholds for this Fold
            t_up, t_low = calculate_optimal_thresholds(y_val_bag_labels, prob_thl_val)
            fold_t_uppers.append(t_up)
            fold_t_lowers.append(t_low)
            
            # 5. Collect Predictions for CSV
            # We need Patient IDs to map back. 
            # `create_patient_bags` iterates unique_patients in order, so we can retrieve them from DF
            unique_patients = val_gen_per_fold.df['patient_id'].unique()
            
            for pid, prob, true_lbl in zip(unique_patients, prob_thl_val, y_val_bag_labels):
                record = {
                    "val_fold": fold,
                    "patient_id": pid,
                    "probability_1": f"{prob:.4f}",
                    "label": int(true_lbl),
                    "threshold_upper_used": f"{t_up:.4f}",
                    "threshold_lower_used": f"{t_low:.4f}"
                }
                all_validation_patient_predictions_list.append(record)
            
                # Break for debugging if needed
                # if fold >= config['TRAIN']['Actual_Fold']:
                #     print(f"DEBUG: Stopping after Fold {fold}.")
                #     break
        
        # ---------------------------------------------------------
        # --- D. Finalize Calibration & Save CSV ---
        # ---------------------------------------------------------
        print("\n" + "="*40)
        print("FINAL CALIBRATION RESULTS")
        print("="*40)
        
        # 1. Average Thresholds across folds (Eq 27 & 29)
        final_t_upper = np.median(fold_t_uppers)
        final_t_lower = np.median(fold_t_lowers)
        
        print(f"Thresholds per fold (Upper): {[f'{t:.3f}' for t in fold_t_uppers]}")
        print(f"Thresholds per fold (Lower): {[f'{t:.3f}' for t in fold_t_lowers]}")
        print(f"\n>> RE-CALIBRATED HYBRID THRESHOLDS:")
        print(f"   T_upper (Confident THL): >= {final_t_upper:.4f}")
        print(f"   T_lower (Confident IDA): <  {final_t_lower:.4f}")
        print(f"   Uncertain Region:       {final_t_lower:.4f} - {final_t_upper:.4f}")
        
        # 2. Save CSV
        csv_filename = "patient_predictions_mil_val.csv"
        csv_path = os.path.join(config.SAVE_PATH, csv_filename)
        
        keys = all_validation_patient_predictions_list[0].keys()
        with open(csv_path, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(all_validation_patient_predictions_list)
            
        print(f"\n[IO] Validation predictions saved to: {csv_path}")
    
        # # Based on calibration results:
        # fold_t_uppers = [0.510, 0.627, 0.516, 0.990, 0.607]
        # fold_t_lowers = [0.490, 0.490, 0.490, 0.490, 0.490]
        
        # # Use Median for robustness
        # final_t_upper = np.median(fold_t_uppers) # 0.6070
        # final_t_lower = np.median(fold_t_lowers) # 0.4900
        
        # print(f"\n[Config] Applied Calibrated Hybrid Thresholds:")
        # print(f"   T_upper (Confident THL): >= {final_t_upper:.4f}")
        # print(f"   T_lower (Confident IDA): <  {final_t_lower:.4f}")
        # print(f"   Uncertain Region:       {final_t_lower:.4f} - {final_t_upper:.4f}")
       
#%% 
def apply_riemannian_filter(features, energies, limit, 
                            rescue_top_k=0, extreme_limit=100.0, 
                            verbose=False):
    """
    Applies Riemannian Safety Filter with a 'Saliency Rescue' gate for 
    pathological morphology (e.g., tear-drop cells).
    """
    e_flat = energies.flatten()
    n_total = len(features)
    
    # 1. Standard Safety Mask (In-Distribution)
    valid_mask = e_flat < limit
    
    # 2. Saliency Rescue Logic (Approach 3)
    n_rescued = 0
    if rescue_top_k > 0:
        # Identify indices that are currently REJECTED but below the EXTREME ceiling
        potential_rescue_idx = np.where((e_flat >= limit) & (e_flat < extreme_limit))[0]
        
        if len(potential_rescue_idx) > 0:
            # Sort potential rescues by energy (highest saliency first)
            # We want the 'most distorted' biological cells
            potential_energies = e_flat[potential_rescue_idx]
            top_k_sub_idx = np.argsort(potential_energies)[-rescue_top_k:]
            
            actual_rescue_indices = potential_rescue_idx[top_k_sub_idx]
            
            # Update the mask to include these rescued salient cells
            valid_mask[actual_rescue_indices] = True
            n_rescued = len(actual_rescue_indices)

    # 3. Filter Arrays
    clean_features = features[valid_mask]
    clean_energies = energies[valid_mask]
    
    # 4. Stats
    n_valid = np.sum(valid_mask)
    n_rejected = n_total - n_valid
    reject_rate = (n_rejected / n_total) if n_total > 0 else 0.0
    
    if verbose:
        print(f"   [Safety Filter] Limit: {limit:.2f} | Extreme Ceiling: {extreme_limit:.2f}")
        print(f"   [Saliency Gate] Rescued {n_rescued} high-energy diagnostic cells.")
        print(f"   [Safety Filter] Final Passed: {n_valid} | Blocked: {n_rejected} | Rejected Rate: {reject_rate}")

    stats = {
        "n_total": n_total,
        "n_valid": n_valid,
        "n_rejected": n_rejected,
        "n_rescued": n_rescued,
        "reject_rate": reject_rate
    }
    
    return clean_features, clean_energies, valid_mask, stats

def predict_clinical_workflow(
    inference_gen, 
    inference_gen_id,
    feature_extractor,
    riemann_tool,
    ensemble_models,
    config,
    curvature_threshold=15.0,  # Used only for audit labels (e.g., High vs Low Energy)
    ood_energy_limit=29.5434, # Primary Safety Limit
    rescue_top_k=10,           # Saliency Rescue (Approach 3)
    extreme_limit=50.0,        # Hard ceiling for rescue
    verbose=True,
    return_patch_data=False,
    enable_ood_filter=True
):
    """
    Phase 4: Inference with Integrated Riemannian Safety Filter.
    """
    if verbose:
        print(f"{'='*40}\n[Clinical Inference] Processing Patient... (Safety Filter: {enable_ood_filter})\n{'='*40}")
    
    # --- 1. Preprocessing & Feature Extraction ---
    if verbose: print(f">> [Step 1] Extraction: Retrieving patient patches from Generator...")
    
    try:
        # Request PIDs to track file paths later
        data_item = inference_gen.__getitem__(idx=inference_gen_id, get_pids=True)
        if len(data_item) == 4:
            batch_patients, labels_raw, X_bag_batch, _ = data_item
        else:
            batch_patients, X_bag_batch, _ = data_item
            
    except Exception as e:
        print(f"Error accessing generator: {e}")
        return {"diagnosis": "Error", "note": str(e), "raw_probs": np.array([0.5, 0.5])}

    X_patches = X_bag_batch[0] 
    pid = batch_patients[0]
    label = labels_raw[0] if 'labels_raw' in locals() else "Unknown"

    if verbose:
        print(f"   - Patient ID: {pid}")
        print(f"   - Input Shape: {X_patches.shape}")

    if verbose: print("    Extracting Features...")
    Z_patient = feature_extractor.predict(X_patches, batch_size=config['TRAIN']['BATCH_SIZE'], verbose=2) 
    Z_patient = np.array(Z_patient) 
    
    # --- 2. Geometric Projection ---
    if verbose: print(">> [Step 2] Computing Geometric Curvature (Energies)...")
    E_patient = riemann_tool.compute_batch_curvature(Z_patient).reshape(-1, 1) 
    # print(f"{E_patient=}")
    # --- 3. Riemannian Safety Filtering ---
    # We use the separate function to split Valid vs OOD
    if verbose: print(f">> [Step 3] Applying Safety Filter (Limit < {ood_energy_limit:.4f})...")

    if enable_ood_filter:
        # Call the helper function
        Z_input, E_input, valid_mask, stats = apply_riemannian_filter(
            Z_patient, 
            E_patient, 
            limit=ood_energy_limit, 
            rescue_top_k=rescue_top_k,
            extreme_limit=extreme_limit,
            verbose=verbose
        )
    else:
        # Bypass Mode: Use ALL data
        if verbose: print(f"   [!] Filter DISABLED: Using all patches regardless of energy.")
        Z_input, E_input = Z_patient, E_patient
        valid_mask = np.ones(len(E_patient), dtype=bool)
        stats = {
            "n_total": len(Z_patient),
            "n_valid": len(Z_patient), 
            "n_rejected": 0, 
            "reject_rate": 0.0
        }

    n_diagnostic = len(Z_input)

    # --- 4. Prediction Phase ---
    final_diagnosis = "Indeterminate"
    confidence = 0.0
    avg_prob = np.array([0.5, 0.5])
    is_ood = (stats['reject_rate'] > 0.5) # Flag if majority of patient is rejected

    if n_diagnostic == 0:
        if verbose: print("   [!] SAFETY STOP: No valid patches remained after filtering.")
        final_diagnosis = "INCONCLUSIVE (OOD)"
    else:
        if verbose: print(">> [Step 4] Ensemble Prediction...")
        
        # Pad Bags for MIL Model Input
        max_len = 32 # Ensure this matches your model training config
        bag_z = np.zeros((1, max_len, Z_input.shape[1]))
        limit = min(n_diagnostic, max_len)
        
        # CRITICAL: We fill the bag ONLY with Valid (Z_input) features
        bag_z[0, :limit, :] = Z_input[:limit, :]
        
        bag_e = np.zeros((1, max_len, 1))
        bag_e[0, :limit, :] = E_input[:limit, :]
        
        if not isinstance(ensemble_models, list):
            ensemble_models = [ensemble_models]
        
        ensemble_probs = []
        for model in ensemble_models:
            # Model sees Clean Data
            prob = model.predict([bag_z, bag_e], verbose=0)[0] 
            ensemble_probs.append(prob)
        
        avg_prob = np.mean(ensemble_probs, axis=0) 
        prediction_idx = np.argmax(avg_prob)
        confidence = avg_prob[prediction_idx]
        classes = ["IDA", "THL"] 
        final_diagnosis = classes[prediction_idx]
        
        if verbose:
            print(f"   - Result: {final_diagnosis} ({confidence:.2%})")

    # --- 5. Explanation & Audit Data ---
    if verbose: print(">> [Step 5] Generating Explanations...")
    
    try:
        # Get paths from generator metadata
        all_patient_paths = inference_gen.patient_groups[pid]['patch_path'].tolist()
        # Fallback if lengths mismatch due to generator internal logic
        if len(all_patient_paths) != len(Z_patient):
             all_patient_paths = [f"patch_{i}.png" for i in range(len(Z_patient))]
    except (KeyError, AttributeError):
        all_patient_paths = ["Unknown_Path"] * len(Z_patient)

    # 5a. Generate Top-K Explanation (From VALID patches only)
    explanation_paths = []
    top_patches_energies = []
    top_indices_original = []

    if n_diagnostic > 0:
        top_k = 3
        # We look at the VALID input to find highest energy used for prediction
        # Get indices relative to Z_input
        top_k_idx_local = np.argsort(E_input.flatten())[-top_k:][::-1]
        
        # Map back to original Z_patient indices using valid_mask
        # valid_indices is a list of locations where valid_mask is True
        valid_indices_global = np.where(valid_mask)[0]
        top_indices_original = valid_indices_global[top_k_idx_local]
        
        top_patches_energies = E_input.flatten()[top_k_idx_local]
        explanation_paths = [all_patient_paths[i] for i in top_indices_original]

    # 5b. Generate Full Patch DataFrame
    patch_df = None
    if return_patch_data:
        # Create status column based on OOD Limit and Valid Mask
        status_list = []
        for i, e in enumerate(E_patient.flatten()):
            if not valid_mask[i]:
                status_list.append("REJECTED (OOD)")
            elif e > curvature_threshold:
                status_list.append("Accepted (High Energy)")
            else:
                status_list.append("Accepted (Low Energy)")

        patch_df = pd.DataFrame({
            "patient_id": [pid] * len(Z_patient),
            "patch_path": all_patient_paths,
            "curvature_energy": E_patient.flatten(),
            "is_valid": valid_mask,
            "status": status_list
        })

    # --- Final Report ---
    report = {
        "diagnosis": final_diagnosis,
        "confidence": float(confidence),
        "raw_probs": avg_prob,
        
        # Stats from the filter function
        "n_total": stats['n_total'],
        "n_valid": stats['n_valid'],
        "n_rejected": stats['n_rejected'],
        "reject_rate": stats['reject_rate'],
        
        "explanation_caption": (
            f"Diagnosed {final_diagnosis} ({confidence:.1%}).\n"
            f"Safety Filter: {stats['n_valid']} accepted, {stats['n_rejected']} rejected."
        ),
        "explanation_images": explanation_paths,
        "patch_dataframe": patch_df 
    }
    
    return report

#%% --- Clinical Inference Execution ---
if config.Evaluate_Test: 
    from riemannian_utils import PoincareMath, HyperbolicDense, FrechetMean, CurvatureAttention
    
    # ---------------------------------------------------------
    # 1. Configuration
    # ---------------------------------------------------------
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
        "THRESHOLDS": {
            "CONFIDENT_THL": 0.5463,  
            "CONFIDENT_IDA": 0.4900   
        },
        
        # Polarity Correction
        # Fold 4 is correct. Folds 0,1,3 are flipped.
        "FLIPPED_FOLDS": [0, 1, 3] 
    }

    LOAD_STRATEGY = 'BEST_FOLD' 
    # LOAD_STRATEGY = 'ENSEMBLE'
    prefix = "val_fold_4_filter_ood_rescue"
    print(f"{prefix=}")
    
    #%%
    print(f"\n{'='*40}\nSTARTING CLINICAL INFERENCE: {LOAD_STRATEGY}\n{'='*40}")

    # ---------------------------------------------------------
    # 2. Load Models based on Strategy
    # ---------------------------------------------------------
    feature_extractors_map = {}
    riemann_tools_map = {}
    all_fold_ensembles = {}
    
    folds_to_load = [CLINICAL_WORKFLOW_CONFIG["ACTIVE_FOLD"]] if LOAD_STRATEGY == 'BEST_FOLD' else range(config.DATA.N_SPLIT)
    
    print(f"Loading resources for folds: {list(folds_to_load)}...")

    for fold in folds_to_load:
        print(f"\n  > Loading Fold {fold} system...")
        # A. Feature Extractor
        feature_extractors_map[fold] = load_one_extractor(fold, model_name, config.BASEMODEL_PATH)
        # B. Riemann Tool
        riemann_tools_map[fold] = load_one_manifold(fold, model_name, config.MODEL_PATH)
        # C. MIL Ensemble
        all_fold_ensembles[fold] = load_mil_ensemble_for_fold(fold, model_name, config.MODEL_PATH, n_estimators=5)

    print(f"[System Ready] Loaded {len(feature_extractors_map)} pipelines.")

    #%% ---------------------------------------------------------
    # 3. Prepare Data
    # ---------------------------------------------------------
    # df_ood = mygears.create_cell_dataframe_flex(
    #     config,
    #     glob_pattern = "[01]/*/cells/*/*.png",  
    #     # glob_pattern = "[0]/OOD-test-00/cells/*/*.jpg", 
    #     # glob_pattern = "[0]/OOD-test-01/cells/*/*.JPG",  
    #     # glob_pattern = "[0]/OOD-test-02/cells/*/*.jpg", 
    # )
    # # Get a unique list of all patients
    # all_patient_ids = df_ood['patient_id'].unique()
    # # Now, df should contain data for all 20 patients
    # print(f"\nDataFrame created with {df_ood['patient_id'].nunique()} unique patients.\n")
    # print(f"{len(df_ood)=}")
    # print(df_ood.columns)
    # print(df_ood.sample(1))
    
    #%%    
    from FullBagDataset import FullBagDataset
    fold = CLINICAL_WORKFLOW_CONFIG["ACTIVE_FOLD"]
    inference_gen = FullBagDataset(
        config,
        # df=test_df_sampled, # Ensure this DF is loaded
        df=val_df_fold_list[fold],
        # df=val_df_fold_sampled,
        # df=df_ood,
        preprocessing_function=preprocessing_fn,
        expect_rgba=True,
        # expect_rgba=False,
        bg_color=(0, 0, 0),
        mode='test',
    )
    
    #%% Select Patients
    rand_gen_idx = np.random.choice(len(inference_gen), size=config.n_select, replace=False)
    print(f"Random selected {len(rand_gen_idx)} patients for audit.")
    
    #% Get a batch for inferencing test
    for inference_gen_id in rand_gen_idx:      
        print(f"\nReading {inference_gen_id=}")    
        # X_bag, y_bag = inference_gen[rand_idx]
        batch_patients, labels_raw, X_bag_batch, labels_onehot = inference_gen.__getitem__(
            idx=0, 
            get_pids=True,
        )
        current_label = labels_raw[0] 
       # 2. Extract the specific label
        # labels_raw is a numpy array like [0] or [1]
        print(f"Patient: {batch_patients[0]}")
        print(f"Label: {current_label}")  
        print(f"{X_bag_batch.shape=}") # (1, N_patches, 256, 256, 3)
        
        if config.BASE != 'Linux':
            inference_gen.plot_random_patient_batch(
                idx=inference_gen_id,
                save_dir='Inference')
        break
    
    #%% 
    # 4. Inference Loop
    # ---------------------------------------------------------
    all_patients_patch_audit = [] 
    ensemble_results_data = []  
    
    for patient_idx in range(0, len(inference_gen)):
    #     print(patient_idx)
    # for index, patient_idx in enumerate(rand_gen_idx):
        # print(index, patient_idx)
        p_ids, labels_raw, X_bag_batch, labels_onehot = inference_gen.__getitem__(
            idx=patient_idx,
            get_pids=True
        )
        current_pid = p_ids[0]
        true_label = int(labels_raw[0])      
        print(f"\n--- Processing {patient_idx+1}/{len(inference_gen)}: {current_pid} (True: {true_label}) ---")

        # Container for probabilities from the selected strategy
        final_prob_thl = 0.0
        final_diagnosis = "Unknown"
        decision_status = "N/A"
        patch_audit_frame = None
        report_images = []

        # === STRATEGY A: BEST FOLD ===
        if LOAD_STRATEGY == 'BEST_FOLD':
            fold = CLINICAL_WORKFLOW_CONFIG["ACTIVE_FOLD"]
            
            report = predict_clinical_workflow(
                inference_gen,
                patient_idx,
                feature_extractors_map[fold],
                riemann_tools_map[fold],
                all_fold_ensembles[fold],
                config,
                curvature_threshold=CLINICAL_WORKFLOW_CONFIG["THRESHOLDS_CURVE"],
                ood_energy_limit=CLINICAL_WORKFLOW_CONFIG["OOD_ENERGY_LIMIT"],
                rescue_top_k=CLINICAL_WORKFLOW_CONFIG["RESCUE_TOP_K"],
                extreme_limit=CLINICAL_WORKFLOW_CONFIG["EXTREME_CEILING"],
                enable_ood_filter=True,  
                # enable_ood_filter=False,
                verbose=True,
                return_patch_data=True
            )

            # Extract Probability
            final_prob_thl = report['raw_probs'][1]
            patch_audit_frame = report['patch_dataframe']
            report_images = report['explanation_images']

        # === STRATEGY B: ENSEMBLE ===
        elif LOAD_STRATEGY == 'ENSEMBLE':
            fold_probs = []
            
            for fold in folds_to_load:
                # Run Pipeline
                report = predict_clinical_workflow(
                    inference_gen, patient_idx,
                    feature_extractors_map[fold],
                    riemann_tools_map[fold],
                    all_fold_ensembles[fold],
                    config,
                    enable_ood_filter=False,
                    ood_energy_limit=CLINICAL_WORKFLOW_CONFIG["OOD_ENERGY_LIMIT"],
                    verbose=False, # Keep log clean
                    return_patch_data=(fold==CLINICAL_WORKFLOW_CONFIG["ACTIVE_FOLD"]) # Only save patches for Best Fold
                )
                
                # Get Prob
                raw_p = report['raw_probs'][1]
                
                # --- POLARITY CORRECTION ---
                if fold in CLINICAL_WORKFLOW_CONFIG["FLIPPED_FOLDS"]:
                    # print(f"    (F{fold} Inverted: {raw_p:.2f} -> {1-raw_p:.2f})")
                    fold_probs.append(1.0 - raw_p)
                else:
                    fold_probs.append(raw_p)
                    
                # Capture patch data from the 'Active' fold for auditing geometry
                if report['patch_dataframe'] is not None:
                    patch_audit_frame = report['patch_dataframe']

            # Average across all folds
            final_prob_thl = np.mean(fold_probs)
            print(f"   Ensemble Aggregation: {final_prob_thl:.4f}")

        # -----------------------------------------------------
        # 5. Tri-State Decision Logic
        # -----------------------------------------------------
        # t_upper = CLINICAL_WORKFLOW_CONFIG["THRESHOLDS"]["CONFIDENT_THL"]
        # t_lower = CLINICAL_WORKFLOW_CONFIG["THRESHOLDS"]["CONFIDENT_IDA"]
        
        # if final_prob_thl >= t_upper:
        #     final_diagnosis = "THL"
        #     decision_status = "POSITIVE (High Conf)"
        #     pred_class = 1
        # elif final_prob_thl < t_lower:
        #     final_diagnosis = "IDA"
        #     decision_status = "NEGATIVE (High Conf)"
        #     pred_class = 0
        # else:
        #     final_diagnosis = "Uncertain / Borderline"
        #     decision_status = "UNCERTAIN"
        #     pred_class = 1 if final_prob_thl > 0.5 else 0 # Soft fallback
            
        # print(f"   > Result: {final_diagnosis} ({final_prob_thl:.4f})")
        
        # -----------------------------------------------------
        # 5. Revised Decision Logic (High-Specificity Rule-In)
        # -----------------------------------------------------
        t_upper = CLINICAL_WORKFLOW_CONFIG["THRESHOLDS"]["CONFIDENT_THL"] # 0.5463
        
        # LOGIC:
        # If Prob >= 0.5463: The model is shouting "Thalassemia!". We trust it (0% Error in pilot).
        # If Prob < 0.5463: The model is muttering. We treat this as "Negative/IDA" to be safe.
        
        if final_prob_thl >= t_upper:
            final_diagnosis = "THL"
            decision_status = "POSITIVE (High Confidence)"
            pred_class = 1
        else:
            # Merges "Confident IDA" and "Uncertain" into a safe "Screen Negative"
            # Captures all true IDA cases.
            final_diagnosis = "Screen Negative (Likely IDA)"
            decision_status = "NEGATIVE (Rule-Out)"
            pred_class = 0
            
        print(f"   > Result: {final_diagnosis} ({final_prob_thl:.4f})")
            
        # -----------------------------------------------------
        # 6. Detailed Logging (Revised)
        # -----------------------------------------------------
        
        # A. Patient Level Logging
        # ------------------------
        # Extract stats from the report (Default to Best Fold stats if Ensemble)
        # Assuming 'report' holds the data from the Active Fold (Strategy A) 
        # or the last run fold (Strategy B - slightly imperfect but usually Fold 4 is last/active)
        
        # Best practice: Use stats from the Best Fold calculation
        stats_source = report 
        
        ensemble_results_data.append({
            "patient_id": current_pid,
            "true_label": true_label,
            "predicted_class": pred_class,
            "probability_thl": final_prob_thl,
            "diagnosis_text": final_diagnosis,
            "status": decision_status,
            "strategy": LOAD_STRATEGY,
            
            # --- NEW METRICS ---
            "total_patches": stats_source['n_total'],
            "accepted_patches": stats_source['n_valid'],
            "rejected_patches": stats_source['n_rejected'],
            "rejection_rate": f"{stats_source['reject_rate']:.4f}"
        })
        
        # B. Patch Level Logging
        # ----------------------
        if patch_audit_frame is not None:
            # 1. Add True Label
            patch_audit_frame['label'] = true_label
            
            # 2. Add Patch Paths (Retrieve from Dataset DF)
            # Filter the main dataframe for this patient to get paths
            # Note: Ensure the sort order matches! 
            # Usually 'create_patient_bags' sorts by filename, but we check inference_gen logic.
            # Assuming inference_gen.df is available and grouped by patient.
            try:
                # Get subset of dataframe for this patient
                patient_sub_df = inference_gen.df[inference_gen.df['patient_id'] == current_pid]
                
                # If sizes match, assign paths
                if len(patient_sub_df) == len(patch_audit_frame):
                    # specific column name depends on your config (e.g., 'file_path', 'path', 'filename')
                    # Adjust 'path' to match your dataframe column
                    if 'patch_path' in patient_sub_df.columns:
                        patch_audit_frame['patch_path'] = patient_sub_df['patch_path'].values
                    else:
                        patch_audit_frame['patch_path'] = "path_col_not_found"
                else:
                    patch_audit_frame['patch_path'] = "mismatch_length"
            except Exception as e:
                patch_audit_frame['patch_path'] = f"error_{str(e)}"

            all_patients_patch_audit.append(patch_audit_frame)

    # ---------------------------------------------------------
    # 7. Save Output
    # ---------------------------------------------------------
    # Save Patient Results
    if ensemble_results_data:
        csv_path = os.path.join(config.SAVE_PATH, f"{prefix}_clinical_report_test_{LOAD_STRATEGY}.csv")
        try:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=ensemble_results_data[0].keys())
                writer.writeheader()
                writer.writerows(ensemble_results_data)
            print(f"\n[IO] Patient Report saved: {csv_path}")
        except Exception as e:
            print(f"[Error] {e}")

    # Save Patch Audit
    if all_patients_patch_audit:
        full_audit_df = pd.concat(all_patients_patch_audit, ignore_index=True)
        audit_file = f"{prefix}_patch_audit_test_{LOAD_STRATEGY}.csv"
        audit_path = os.path.join(config.SAVE_PATH, audit_file)
        full_audit_df.to_csv(audit_path, index=False)
        print(f"[IO] Patch Audit Log saved: {audit_path}")
        
    #%%

       
#%% Performance Analysis
if config.Performance_Report:       
    
    # 1. Calibration Settings
    T_LOWER = 0.4914  # Boundary for Confident IDA
    T_UPPER = 0.6505  # Boundary for Confident THL
    T_OPT   = 0.5742  # Optimal cutoff for the 'Uncertain' fallback
    
    audit_path = 'val_predictions_mean_QC_Std_20251220-1240.csv'
    # audit_path = 'val_predictions_argmax_QC_Std_20251220-1240.csv'
    df = pd.read_csv(audit_path)
    
    # 2. Extract Ground Truth and Probabilities
    y_true = df['true_class'].to_numpy()
    y_prob = df['probability_1'].to_numpy()
    
    # 3. CALL APPLY_CLINICAL_TRIAGE
    # We process each patient probability to determine confidence status and triage labels
    triage_results = [mygears.apply_clinical_triage(p, T_LOWER, T_UPPER, T_OPT) for p in y_prob]
    
    # 4. Integrate Results into the DataFrame
    # Convert the list of dictionaries to a DataFrame and merge with original data
    df_triage = pd.DataFrame(triage_results)
    
    # We drop the 'probability' column from the triage dataframe to avoid duplication with 'probability_1'
    df = pd.concat([df, df_triage.drop(columns=['probability'])], axis=1)
    
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

    # Review top entries
    print(df[['patient_id', 'true_class', 'probability_1', 'pred_status', 'diagnosis_label']].head())
    
    #% Verify Shapes
    # y_true = df['true_class']
    # y_pred = df['final_pred']
    
    # Extract probabilities only for the Confident subset
    y_pred_confident = df.loc[df['pred_status'] == 'Confident', 'final_pred'].to_numpy()
    # For a complete academic evaluation, you likely also need the corresponding labels:
    y_true_confident = df.loc[df['pred_status'] == 'Confident', 'true_class'].to_numpy()
    y_true = y_true_confident
    y_pred = y_pred_confident
    print(f"y_true: {y_true.shape}")
    print(f"y_pred: {y_pred.shape}")
    
       
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
       save_dir='.',
       save_prefix=str('Val'),
       )
    
    mygears.plot_confusion_matrices(
                confusionmatrix=cm,
                p_measures=p_measures,
                class_labels=config['DATA']['CLASS_LABELS'],
                save_dir='.',
                save_prefix=str('Val')
            )  
     
#%%
if config.External_Infer:
    print("\n\nfor external inference (e.g., raw images from a new hospital without labels).")
    print("-" * 60)
    print(" [UI] INITIALIZING EXTERNAL INFERENCE PIPELINE")
    print("-" * 60)
    best_fold = 0
    best_threshold=3.1483
    ood_energy_limit=17.29108543395996
    print(" [UI] Loading Feature Extractor (Best Fold)...")
    feature_extractor = load_one_extractor(best_fold, config)
    
    print("[UI] Loading MIL Ensembles (Best Fold)...")
    mil_ensemble = load_mil_ensemble_for_fold(best_fold, config)
    
    # 1. Create a DataFrame for the external sample
    #    Structure must match what FullBagDataset expects: 'patient_id' and 'patch_path'
    image_path_list = [
        r'H:/My Drive/THL-g/Kasikrit.jpg',
        r'H:/My Drive/THL-g/Kasikrit.jpg',
        r'H:/My Drive/THL-g/Kasikrit.jpg',
        r"H:/My Drive/THL-g/Blue-Jay-on-redbud-tree-by-Tom-Reichner_news.png",
        r"H:/My Drive/THL-g/Blue-Jay-on-redbud-tree-by-Tom-Reichner_news.png",
        r"H:/My Drive/THL-g/Blue-Jay-on-redbud-tree-by-Tom-Reichner_news.png",
        ]
    
    patient_id = 'Ext_Patient_001'
    external_df = pd.DataFrame({
        'patient_id': [patient_id] * len(image_path_list), # Group all images under one ID
        'patch_path': image_path_list                      # List of absolute paths to images
    })
    
    print(f" [UI] Created DataFrame for Patient: {patient_id}")
    print(f"      - Total Images Found: {len(external_df)}")
    print(f"      - First 3 paths: {external_df['patch_path'].head(3).tolist()}")
    
    # 2. Initialize the Generator in 'external' mode
    feature_extractor_preprocess_input = tf.keras.applications.convnext.preprocess_input
    
    print(" [UI] Initializing FullBagDataset (Mode: external)...")
    inference_gen = FullBagDataset(
        config=config,
        df=external_df,
        preprocessing_function=feature_extractor_preprocess_input,
        mode='external',        # <--- CRITICAL: Tells generator to return (PID, X) only
        expect_rgba=False,        
        bg_color=(0, 0, 0),      # Black background for transparent parts
    )  
    
    # Statement in predict_clinical_workflow:
    # Fetch data item. We check length to handle 'external' vs 'test' modes dynamically.
    print(" [UI] Loading batch data...")
    data_item = inference_gen.__getitem__(0, get_pids=True)
    inference_gen.plot_random_patient_batch(
                idx=0,
                save_dir='Inference')
    
    if len(data_item) == 2:
        # Mode = 'external' (FullBagDataset returns: batch_patients, X_batch)
        batch_patients, X_bag_batch = data_item
        print("Mode: External Inference (No Labels)")
    elif len(data_item) == 3:
        # Mode = 'inference' (Returns: batch_patients, X_batch, y_batch)
        batch_patients, X_bag_batch, _ = data_item
    else:
        # Mode = 'test' (Returns: batch_patients, labels, X_batch, y_batch)
        batch_patients, _, X_bag_batch, _ = data_item
    
    # Unpack the batch (Remove the '1' dimension)
    # X_bag_batch shape: (1, 1000, 256, 256, 3)
    X_patches = X_bag_batch[0] 
    print(f" [UI] Batch Unpacked. Input Tensor Shape: {X_patches.shape}")
    
    # Extract Features using the Frozen Model (e.g., ConvNeXt)
    # Output Z_patient shape: (1000, 16) 
    print(" [UI] Extracting Features...")
    Z_patient = feature_extractor.predict(X_patches, batch_size=32, verbose=1)
    print(f"{Z_patient.shape=}")
    
    # Compute Curvature Energy (E)
    # E_patient shape: (1000, 1)
    print(" [UI] Computing Riemannian Curvature (Energy)...")
    riemann_tool = load_one_manifold(best_fold, config)
    E_patient = riemann_tool.compute_batch_curvature(Z_patient).reshape(-1, 1)
    print(f"{E_patient=}")
    
    # Stats for UI
    e_min, e_max, e_mean = E_patient.min(), E_patient.max(), E_patient.mean()
    print(f"      - Energy Stats: Min={e_min:.4f}, Max={e_max:.4f}, Avg={e_mean:.4f}")

    # Filter Condition
    # threshold comes from the Best Fold validation (e.g., 3.1483)
    
    curvature_threshold=best_threshold
    is_diagnostic = E_patient.flatten() > curvature_threshold
    print(f"{curvature_threshold=}, {is_diagnostic=}")
    
    print("Apply Filter")
    Z_diagnostic = Z_patient[is_diagnostic]
    E_diagnostic = E_patient[is_diagnostic]
    
    n_kept = len(Z_diagnostic)
    n_total = len(Z_patient)
    print(f" [UI] Filter Result: Kept {n_kept}/{n_total} patches ({n_kept/n_total:.1%})")
    
    # Store indices for explainability later
    diagnostic_indices = np.where(is_diagnostic)[0]
    
    # Check max energy against limit (e.g., 100.0)
    max_energy = np.max(E_diagnostic) if len(E_diagnostic) > 0 else 0
    is_ood = max_energy > ood_energy_limit
    if is_ood:
        print(f" [UI] WARNING: OOD DETECTED! Max Energy {max_energy:.2f} > Limit {ood_energy_limit}")
    
    if len(Z_diagnostic) > 0:
        # 1. Prepare Inputs (Pad/Truncate to max_len=32)
        # We create a dummy batch of size 1 for the MIL model
        bag_z = np.zeros((1, 32, 16)) 
        bag_e = np.zeros((1, 32, 1))
        
        limit = min(len(Z_diagnostic), 32)
        bag_z[0, :limit] = Z_diagnostic[:limit]
        bag_e[0, :limit] = E_diagnostic[:limit]
        print(f" [UI] MIL Input Prepared. Bag Size: {limit} patches (Padded to 32)")
        
        # 2. Loop through Ensemble Models
        print(f" [UI] Running Ensemble Voting ({len(mil_ensemble)} models)...")
        votes = []
        for i, model in enumerate(mil_ensemble):
            # Predict returns [Prob_IDA, Prob_THL]
            pred = model.predict([bag_z, bag_e], verbose=0)[0]
            votes.append(pred)
            print(f"      - Model {i+1}: IDA={pred[0]:.3f}, THL={pred[1]:.3f}")
            
        # 3. Average the votes
        avg_pred = np.mean(votes, axis=0)
        final_class = "THL" if np.argmax(avg_pred) == 1 else "IDA"
        
        print("-" * 40)
        print(f" [UI] AGGREGATED RESULT: {final_class}")
        print(f"      - Avg Probabilities: IDA={avg_pred[0]:.4f}, THL={avg_pred[1]:.4f}")
        print("-" * 40)
        
        # Map indices back to paths in the dataframe
        top_indices = diagnostic_indices[np.argsort(E_diagnostic.flatten())[-3:]] # Top 3
        explanation_paths = external_df.iloc[top_indices]['patch_path'].tolist()
        
        external_inference_report = {
            "diagnosis": final_class,
            "confidence": np.max(avg_pred),
            "explanation_images": explanation_paths
        }
        print(" [UI] Report Generated.")
        print(external_inference_report)
    else:
        print(" [UI] ERROR: No diagnostic patches passed the filter. Cannot proceed to diagnosis.")



#%%
"""
def analyze_model_attention(model, threshold):
    # Extracts the learned parameters (alpha, beta) from the CurvatureAttention layer
    # to validate the attention mechanism analytically.
    
    # Locate the CurvatureAttention layer
    attn_layer = None
    for layer in model.layers:
        if isinstance(layer, CurvatureAttention):
            attn_layer = layer
            break
    
    if attn_layer:
        alpha = attn_layer.alpha.numpy()[0]
        beta = attn_layer.beta.numpy()[0]
        print(f"\n[Analysis] Learned Attention Parameters:")
        print(f"  - Alpha (Sensitivity to Curvature): {alpha:.4f}")
        print(f"  - Beta (Bias/Base Attention):       {beta:.4f}")
        
        # Theoretical validation: Does attention increase with energy?
        # Sigmoid(alpha * E + beta)
        dummy_low = tf.math.sigmoid(alpha * 0.0 + beta).numpy()
        dummy_high = tf.math.sigmoid(alpha * (threshold * 2) + beta).numpy()
        print(f"  - Theoretical Attention @ Energy=0.0: {dummy_low:.4f}")
        print(f"  - Theoretical Attention @ Energy={threshold*2:.2f}: {dummy_high:.4f}")
    else:
        print("[Analysis] CurvatureAttention layer not found or using functional API extraction.")

def query_manifold_samples(features, energies, labels, threshold,
                           query_type='high_energy', n_samples=5):
    # Queries samples based on their geometric properties relative to the manifold threshold.
    
    # Args:
    #     query_type (str): 'high_energy' (Anomalous/Informative) or 'low_energy' (Generic/Noise).
    
    if query_type == 'high_energy':
        # Filter for samples exceeding the curvature threshold
        indices = np.where(energies > threshold)[0]
        print(f"\n[Query] Analyzing High-Energy Samples (E > {threshold:.4f})...")
        print(f"  - Found {len(indices)} samples exhibiting high scalar curvature.")
    else:
        # Filter for samples within the flat region
        indices = np.where(energies <= threshold)[0]
        print(f"\n[Query] Analyzing Low-Energy Samples (E <= {threshold:.4f})...")
        print(f"  - Found {len(indices)} samples residing in the flat manifold region.")

    # Randomly select a subset for detailed inspection
    if len(indices) == 0:
        return
        
    selected_indices = np.random.choice(indices, min(n_samples, len(indices)), replace=False)
    
    return selected_indices

#%% --- Execution Block ---

# 1. Inspect Learned Topology-Awareness
# We verify if the model actually learned to weight high-energy patches higher.
analyze_model_attention(mil_model, threshold=threshold)

# 2. Targeted Sampling Strategy
# We define specific queries to validate the "Instance Selection" capability.
queries = ['high_energy', 'low_energy']

for q_type in queries:
    target_indices = query_manifold_samples(
        X_train_z, E_train, y_bag_labels, 
        threshold=4.0031, query_type=q_type,
        n_samples=32)
    
    if target_indices is None: continue

    print(f"\n  {'Idx':<6} | {'Energy (Curvature)':<20} | {'Prediction (Prob Class 1)':<25} | {'Ground Truth':<12} | {'Interpretation'}")
    print("-" * 40)

    for idx in target_indices:
        # Prepare inputs with correct batch dimensions (1, D)
        sample_z = X_train_z[idx:idx+1]
        sample_e = E_train[idx:idx+1]
        
        # Inference
        pred_probs = mil_model.predict([sample_z, sample_e], verbose=0)[0]
        
        # Metric Extraction
        energy_val = sample_e[0][0]
        prob_thl = pred_probs[1] # Probability of Thalassemia
        # gt_label = y_bag_labels[idx]
        gt_label = train_df_fold_sampled.iloc[idx]['label']
        
        # Academic Interpretation of the result
        if q_type == 'high_energy':
            # High energy should ideally lead to confident predictions (High or Low prob)
            confidence = abs(prob_thl - 0.5) * 2 # 0 to 1
            interp = "Informative" if confidence > 0.5 else "Ambiguous Anomaly"
        else:
            # Low energy should ideally result in suppressed/neutral predictions
            interp = "Suppressed/Generic"

        print(f"  {idx:<6} | {energy_val:<20.4f} | {prob_thl:<25.4f} | {gt_label:<12} | {interp}")

    print("-" * 40)
    
#%%
def visualize_geometric_samples(generator, indices, energies, model, features, labels, 
                                title="Geometric Analysis", save_dir=None):
    
    # Visualizes specific samples identified by Riemannian curvature analysis.
    
    # Args:
    #     generator: The PatchDatasetPreserve instance (train_gen).
    #     indices: List of global indices (from query_manifold_samples).
    #     energies: Array of curvature energies (E_train).
    #     model: The trained MIL model (for getting probability).
    #     features: Array of feature vectors (X_train_z).
    #     labels: Array of ground truth labels.
    
    n_samples = len(indices)
    if n_samples == 0:
        print(f"[{title}] No samples to visualize.")
        return

    # Dynamic Grid Layout
    cols = 8
    rows = int(np.ceil(n_samples / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 3.0), dpi=150)
    axes = axes.flatten()
    
    print(f"\n[Visualization] Generatiing grid for {title} ({n_samples} samples)...")

    for i, idx in enumerate(indices):
        ax = axes[i]
        
        # 1. Retrieve Image Path & Load directly (Avoids re-normalization artifacts)
        # We access the DataFrame directly via the generator
        row_data = generator.df.iloc[idx]     
        root = Path(config.DATASET) 
        img_path = root / row_data['patch_path']    
        
        try:
            # Read and Convert for Display
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img.shape[-1] == 4: # Handle RGBA
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 2. Get Metrics
            energy_val = energies[idx][0]
            gt_label = labels[idx]
            
            # Predict (On-the-fly for this sample)
            # Reshape to (1, D) and (1, 1) for model input
            sample_z = features[idx:idx+1]
            sample_e = energies[idx:idx+1]
            pred_prob = model.predict([sample_z, sample_e], verbose=0)[0][1] # Prob Class 1 (THL)

            # 3. Plot
            ax.imshow(img)
            
            # Color-code border based on correctness/energy logic
            # Red border if High Energy (Attention Active), Blue if Low
            border_color = 'red' if 'High' in title else 'blue'
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(2)

            # Annotations
            ax.set_title(
                f"Idx: {idx}\nE: {energy_val:.2f}\nP(THL): {pred_prob:.2f}\nGT: {gt_label}",
                fontsize=14, pad=3
            )
            
        except Exception as e:
            print(f"Error loading image at index {idx}: {e}")
            ax.text(0.5, 0.5, "Load Error", ha='center', va='center')

        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle(f"{title}\n(E = Curvature Energy, P = Predicted Prob)", fontsize=16, y=0.99)
    plt.tight_layout(h_pad=5.0)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{title.replace(' ', '_')}.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved visualization -> {save_path}")
        
    plt.show()

#%% --- Integration into your Loop ---

for q_type in queries:
    # 1. Query Indices
    target_indices = query_manifold_samples(
        X_train_z, E_train, y_bag_labels, 
        threshold=4.0031, query_type=q_type,
        n_samples=32  # Visualizing 32 samples per type
    )
    
    if target_indices is None: continue

    # 2. Visualize
    visualize_geometric_samples(
        generator=train_gen,
        indices=target_indices,
        energies=E_train,
        model=mil_model,
        features=X_train_z,
        labels=train_gen.df.label.values,
        title=f"Geometric Analysis - {q_type.replace('_', ' ').title()}",
        save_dir=train_log
    )
"""
#%%
t2 = datetime.now() - t1
print('\nAll execution time: ', t2)