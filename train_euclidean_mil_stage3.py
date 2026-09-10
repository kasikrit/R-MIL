# train_robust_mil.py (Phase 3: Robust Training/Inference)
# This script simulates the final "Patient-Level Aggregation" step. It doesn't retrain the CNN backbone but trains the Attention Weights based on curvature.
import os
import logging

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
import platform
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from keras.utils import custom_object_scope
from keras.applications.convnext import LayerScale 
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
found_files, patient_patch_count = mygears.create_cell_dataframe_debug(config)
df = mygears.create_cell_dataframe_flex(config, glob_pattern="[01]/*/normalized/*-cells-256/*.png")
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
        train_log = config.BASE + '-EUCLIMIL-' + \
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
from riemannian_utils import RiemannianMetrics

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

def create_euclidean_patient_bags(df, features):
    """Groups patch features into Patient Bags for Euclidean MIL."""
    
    # --- CRITICAL FIX: Align DataFrame and Feature Array Lengths ---
    # If model.predict() dropped the final partial batch during extraction, 
    # we truncate the DataFrame to match the feature array exactly.
    n_features = features.shape[0]
    if len(df) != n_features:
        print(f"   [Alignment Fix] DataFrame ({len(df)}) != Features ({n_features}). Truncating trailing patches.")
        df = df.iloc[:n_features].copy()

    unique_patients = df['patient_id'].unique()
    bags_z = []
    bag_labels = []
    
    # Convert columns to numpy arrays for fast, positional indexing
    patient_ids = df['patient_id'].to_numpy()
    labels = df['label'].to_numpy()
    
    for pid in tqdm(unique_patients, desc="Bagging Euclidean Patches"):
        # Use np.where to get exact positional indices (0 to N)
        indices = np.where(patient_ids == pid)[0]
        if len(indices) == 0: continue
        
        bags_z.append(features[indices]) # Shape (N_patches, D)
        bag_labels.append(labels[indices[0]])
        
    return bags_z, np.array(bag_labels)

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

from tensorflow.keras import layers
from ModelBuilder import AttentionMIL # Make sure AttentionMIL is in ModelBuilder

def build_euclidean_mil_head(feature_dim=1536, max_len=32):
    """
    Stage 3: Euclidean Attention-MIL Head (Baseline M1)
    """
    # 1. Input is strictly the feature bag (No Energy)
    input_features = tf.keras.Input(shape=(max_len, feature_dim), name='bag_features')
    
    # 2. Apply Standard Euclidean Attention-MIL
    bag_representation = AttentionMIL(L_dim=256, name="attention_mil_pooling")(input_features)
    
    # 3. Classification Head (Matches the capacity of the Riemannian branch)
    x = layers.Dense(128, activation='relu', name="clf_dense_1")(bag_representation)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(64, activation='relu', name="clf_dense_2")(x)
    
    output = layers.Dense(2, activation='softmax', name="output")(x)
    
    model = tf.keras.Model(inputs=input_features, outputs=output, name="Euclidean_Attention_MIL")
    return model

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
def train_euclidean_mil_ensemble(fold, train_df, features, config, n_estimators=5, max_bag_len=32):
    """Trains an ensemble of Euclidean Attention-MIL models."""
    print(f"\n{'='*40}\n[Phase 3] Euclidean Attention-MIL Training\n{'='*40}")

    # 1. Create Patient Bags (No Energy Required)
    bags_z, y_bag_labels = create_euclidean_patient_bags(df=train_df, features=features)

    # 2. Pad Bags for Tensor format
    X_bag_z = pad_bags(bags_z, max_len=max_bag_len) 
    y_bag_oh = tf.keras.utils.to_categorical(y_bag_labels, 2)

    ensemble_models = []
    
    for i in range(n_estimators):
        print(f"\n   --- Training Estimator {i}/{n_estimators} ---")
        
        model = build_euclidean_mil_head(feature_dim=features.shape[1], max_len=max_bag_len)
        
        optimizer = tfa.optimizers.AdamW(
            learning_rate=config['TRAIN']['LR'],
            weight_decay=config['TRAIN']['WEIGHT_DECAY']
        )

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        callbacks_list = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=config.TRAIN.stop_patience, restore_best_weights=True, verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=config.TRAIN.reduceLR_patience, min_lr=1e-5, verbose=1
            )
        ]

        # Fit Model (Only passing X_bag_z)
        history = model.fit(
            x=X_bag_z, y=y_bag_oh,
            batch_size=config.TRAIN.BATCH_SIZE, 
            epochs=config.TRAIN.EPOCHS, 
            validation_split=config.DATA.val_ratio,
            callbacks=callbacks_list,
            verbose=2
        )  
        
        ensemble_models.append(model)
        
        # Save individual estimators 
        save_path = os.path.join(config.MODEL_PATH, f"fold_{fold}_Euclidean_Attention_MIL_estimator_{i}.h5")
        model.save(save_path)
        print(f"   Saved estimator {i} to {save_path}")

    return ensemble_models
     
#%% Phase 3: Train Euclidean Attention-MIL (Ablation Baseline)
if config.TRAIN.Phase3:
    print("\n\nPhase 3: Trains an ensemble of Euclidean Bag-Level MIL models.")
    from sklearn.metrics import roc_auc_score
    
    # NOTE: No thresholds or Riemann tools are loaded because Euclidean MIL does not use curvature.
    
    X_train_z_all = {}
    X_val_z_all = {} 
    
    print(f"\n[Loading] Preparing Euclidean Features...")
    for fold in range(config.DATA.N_SPLIT):
        print(f"   Processing Fold {fold}...")
        
        # A. Load Training Features,  fold_4_ConvNeXtLarge_val_features.npy
        tr_path = os.path.join(config.MODEL_PATH, f"fold_{fold}_{model_name}_features.npy")
        if os.path.exists(tr_path):
            X_train_z_all[fold] = np.load(tr_path)
            print(f"      -> Loaded Train features from disk. -- {tr_path}")
        else:
            raise FileNotFoundError(f"Critical: Train features missing: {tr_path}")

        # B. Load OR Extract Validation Features
        val_path = os.path.join(config.MODEL_PATH, f"fold_{fold}_{model_name}_val_features.npy")
        if os.path.exists(val_path):
            X_val_z_all[fold] = np.load(val_path)
            print(f"      -> Loaded Val features from disk. -- {val_path}")
        else:
            print(f"      -> Extracting Val features on-the-fly...")
            feature_extractor = mygears.load_one_extractor(fold, model_name, config.MODEL_PATH)
            X_val_z, _ = p2_step1_extract_val_features(model_name, feature_extractor, val_gen_list[fold], verbose=False)
            np.save(val_path, X_val_z)
            X_val_z_all[fold] = X_val_z
            tf.keras.backend.clear_session()
            
    # --- Main Training Loop ---
    fold_aurocs = [] 
    all_fold_mil_ensembles = {} 
    
    for fold in range(config.DATA.N_SPLIT):
        print(f"\n>>> Processing EUCLIDEAN FOLD {fold} <<<") 
        
        # Train Euclidean Ensemble
        mil_ensemble = train_euclidean_mil_ensemble(
            fold=fold,
            train_df=train_gen_list[fold].df,      
            features=X_train_z_all[fold], 
            config=config,
            n_estimators=5
        )
        all_fold_mil_ensembles[fold] = mil_ensemble
        
        # Evaluate on Validation
        print(f"   [Eval] Evaluating Euclidean Ensemble on Validation Fold {fold}...")      
        X_val_z = X_val_z_all[fold]
        
        bags_val_z, y_val_bag_labels = create_euclidean_patient_bags(df=val_gen_list[fold].df, features=X_val_z)
        X_bag_val_z = pad_bags(bags_val_z, max_len=32)
        
        val_probs = []
        for model in mil_ensemble:
            # Predict ONLY on X_bag_z
            preds = model.predict(X_bag_val_z, verbose=0)
            val_probs.append(preds)
        
        avg_val_probs = np.mean(val_probs, axis=0)
        
        if len(np.unique(y_val_bag_labels)) > 1:
            auc = roc_auc_score(y_val_bag_labels, avg_val_probs[:, 1])
            print(f"   [Result] Fold {fold} Euclidean AUC: {auc:.4f}")
            fold_aurocs.append(auc)
        else:
            fold_aurocs.append(0.0)
            
        
        # Debugging Break
        if fold == config['TRAIN']['Actual_Fold']:
            print(f"DEBUG: Stopping after Fold {fold} as requested.")
            break
    
    print(f"\n[Euclidean Training Complete] Mean AUC across folds: {np.mean(fold_aurocs):.4f}")
  
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
    feature_extractor = mygears.load_one_extractor(fold, model_name, config.BASEMODEL_PATH)
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
    riemann_tool = mygears.load_one_manifold(fold, 'ConvNeXtLarge', config.MODEL_PATH, val=False)
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
        fold_t_uppers = [0.510, 0.627, 0.516, 0.990, 0.607]
        fold_t_lowers = [0.490, 0.490, 0.490, 0.490, 0.490]
        
        # Use Median for robustness
        final_t_upper = np.median(fold_t_uppers) # 0.6070
        final_t_lower = np.median(fold_t_lowers) # 0.4900
        
        print(f"\n[Config] Applied Calibrated Hybrid Thresholds:")
        print(f"   T_upper (Confident THL): >= {final_t_upper:.4f}")
        print(f"   T_lower (Confident IDA): <  {final_t_lower:.4f}")
        print(f"   Uncertain Region:       {final_t_lower:.4f} - {final_t_upper:.4f}")
       
        # [Config] Applied Calibrated Hybrid Thresholds:
        #    T_upper (Confident THL): >= 0.6070
        #    T_lower (Confident IDA): <  0.4900
        #    Uncertain Region:       0.4900 - 0.6070
           
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

#%% --- Euclidean Ablation Inference Execution (M1, M2, M3) ---
if config.Evaluate_Test: 
    from sklearn.covariance import EmpiricalCovariance
    from tensorflow.keras.utils import custom_object_scope
    from ModelBuilder import AttentionMIL
    
    def build_euclidean_mil_head_dynamic(feature_dim=1536):
        """
        Builds the Stage 3 Euclidean Attention-MIL Head with a dynamic bag length.
        Expects input shape: (Batch_Size, None, Feature_Dim)
        """
        input_features = tf.keras.Input(shape=(None, feature_dim), name='bag_features')
        bag_representation = AttentionMIL(L_dim=256, name="attention_mil_pooling")(input_features)
        
        x = tf.keras.layers.Dense(128, activation='relu', name="clf_dense_1")(bag_representation)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(64, activation='relu', name="clf_dense_2")(x)
        
        output = tf.keras.layers.Dense(2, activation='softmax', name="output")(x)
        
        return tf.keras.Model(inputs=input_features, outputs=output, name="Euclidean_Attention_MIL_Dynamic")

    def calibrate_euclidean_mahalanobis(feature_matrix):
        """Fits the Euclidean Mahalanobis parameters using extracted training features."""
        cov_estimator = EmpiricalCovariance(assume_centered=False)
        cov_estimator.fit(feature_matrix)
        return cov_estimator.location_, cov_estimator.precision_

    def apply_euclidean_qc_gate(bag_features, mu, prec, threshold):
        """Filters a bag of patch features using Euclidean Mahalanobis distance."""
        mu_tensor = tf.cast(tf.constant(mu), tf.float32)
        prec_tensor = tf.cast(tf.constant(prec), tf.float32)
        centered_features = bag_features - mu_tensor 
        left_term = tf.matmul(centered_features, prec_tensor) 
        squared_dist = tf.reduce_sum(left_term * centered_features, axis=1)
        mahalanobis_dist = tf.sqrt(tf.maximum(squared_dist, 1e-9)) 
        mask = mahalanobis_dist < threshold
        return tf.boolean_mask(bag_features, mask)

    def load_euclidean_extractor(fold_idx, model_path):
        fe_filename = f"fold_{fold_idx}_ConvNeXtLarge_feature_extractor.h5"
        fe_path = os.path.join(model_path, fe_filename)
        with custom_object_scope({'LayerScale': LayerScale}):
            return load_model(fe_path)

    def load_euclidean_mil_ensemble(fold_idx, model_path, n_estimators=5):
        """
        Loads the trained Euclidean models into a dynamic-length architecture.
        """
        fold_models = []
        with custom_object_scope({'AttentionMIL': AttentionMIL}):
            for i in range(n_estimators):
                filename = f"fold_{fold_idx}_Euclidean_Attention_MIL_estimator_{i}.h5"
                filepath = os.path.join(model_path, filename)
                
                # 1. Build the dynamic model
                model = build_euclidean_mil_head_dynamic(feature_dim=1536)
                
                # 2. Inject the trained weights
                model.load_weights(filepath)
                
                fold_models.append(model)
                
        print(f"  [Loaded] Dynamic Euclidean Attention Ensemble (Fold {fold_idx})")
        return fold_models

    # ---------------------------------------------------------
    # 1. Configuration
    # ---------------------------------------------------------
    EUCLIDEAN_WORKFLOW_CONFIG = {
        "FOLDS": range(config.DATA.N_SPLIT), 
        "MAX_BAG_LEN": 32, # Ensure this matches training exactly
        "THRESHOLDS": {
            "CONFIDENT_THL": 0.5463,  
            "CONFIDENT_IDA": 0.4900   
        }
    }
    
    euclidean_calib_map = {} 
    
    # Path where your .npy files reside (Based on your terminal output)
    FEATURE_DIR = os.path.join(config.BASEPATH, 'R-MIL', 'models') 
    # Path where your Euclidean Attention models reside
    MIL_MODEL_DIR = config.MODEL_PATH 
    
    # =========================================================================
    # PHASE 1: Calibrate Euclidean QC Gate (Fast load via .npy)
    # =========================================================================
    print("\n" + "="*50 + "\nPHASE 1: CALIBRATING EUCLIDEAN QC GATE (FAST LOAD)\n" + "="*50)
    
    for fold in EUCLIDEAN_WORKFLOW_CONFIG["FOLDS"]:
        print(f"--- Calibrating Fold {fold} ---")
        
        # 1. Load Training Features (.npy)
        train_npy = os.path.join(FEATURE_DIR, f"fold_{fold}_ConvNeXtLarge_features.npy")
        X_train_z = np.load(train_npy)
        
        # Calculate mu and precision matrix
        mu, prec = calibrate_euclidean_mahalanobis(X_train_z)
        
        # 2. Load Validation Features (.npy)
        val_npy = os.path.join(FEATURE_DIR, f"fold_{fold}_ConvNeXtLarge_val_features.npy")
        X_val_z = np.load(val_npy)
        
        # 3. Calculate Mahalanobis Distances for Validation Set
        mu_tensor = tf.cast(tf.constant(mu), tf.float32)
        prec_tensor = tf.cast(tf.constant(prec), tf.float32)
        centered_val = X_val_z - mu_tensor
        left_term = tf.matmul(centered_val, prec_tensor)
        val_distances = tf.sqrt(tf.maximum(tf.reduce_sum(left_term * centered_val, axis=1), 1e-9)).numpy()
        
        # 4. Calibrate tau_limit to the 95th percentile
        tau_limit = np.percentile(val_distances, 95)
        print(f"  [Fold {fold} Calibrated] TAU_LIMIT: {tau_limit:.4f}")
        
        euclidean_calib_map[fold] = {'mu': mu, 'prec': prec, 'tau_limit': tau_limit}
      
    print(f"{euclidean_calib_map=}")
    # =========================================================================
    # PHASE 2: Clinical Inference Execution (M1, M2, M3)
    # =========================================================================
    print(f"\n{'='*50}\nPHASE 2: STARTING 5-FOLD EUCLIDEAN INFERENCE (M1, M2, M3)\n{'='*50}")

    # Load all models into RAM
    euclidean_backbones_map = {}
    euclidean_attention_ensembles = {}
    for fold in EUCLIDEAN_WORKFLOW_CONFIG["FOLDS"]:
        print(f"Loading Fold {fold} Models...")
        euclidean_backbones_map[fold] = load_euclidean_extractor(fold, FEATURE_DIR)
        euclidean_attention_ensembles[fold] = load_euclidean_mil_ensemble(fold, MIL_MODEL_DIR, n_estimators=5)

    from FullBagDataset import FullBagDataset
    inference_gen = FullBagDataset(
        config,
        df=test_df_sampled, 
        preprocessing_function=preprocessing_fn,
        expect_rgba=True,
        bg_color=(0, 0, 0),
        mode='test',
    )
    
    def predict_euclidean_ensemble_workflow(patches, backbone, attention_ensemble, mu, prec, tau_limit, max_len=32, apply_qc=True):
        """Processes a patient bag through an ensemble of 5 Euclidean models with Padding."""
        # 1. Extract features using the Fold's Backbone
        # Set batch size to avoid OOM on large bags
        raw_features = backbone.predict(patches, batch_size=config.TRAIN.BATCH_SIZE, verbose=config.verbose) 
        
        # 2. Apply Euclidean QC Gate
        if apply_qc:
            clean_features = apply_euclidean_qc_gate(raw_features, mu, prec, tau_limit)
        else:
            clean_features = raw_features
            
        n_raw = tf.shape(raw_features)[0].numpy()
        n_clean = tf.shape(clean_features)[0].numpy()
    
        if n_clean == 0:
            return None, 0, n_raw
        
            
        # 3. Evaluate the FULL bag (No padding or truncation)
        # Adds batch dimension -> Shape: (1, N_clean, 1536)
        bag_z = tf.expand_dims(clean_features, axis=0)
            
        # 4. Predict using all 5 estimators in the ensemble
        ensemble_probs = []
        for model in attention_ensemble:
            patient_prediction = model.predict(bag_z, verbose=config.verbose) 
            ensemble_probs.append(patient_prediction[0, 1]) # Prob of THL
            
        return np.mean(ensemble_probs), n_clean, n_raw

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
            attention_ensemble = euclidean_attention_ensembles[fold]
            calib = euclidean_calib_map[fold]
            max_len = EUCLIDEAN_WORKFLOW_CONFIG["MAX_BAG_LEN"]
    
            # M1: Vanilla Euclidean (No QC)
            prob_m1, _, _ = predict_euclidean_ensemble_workflow(
                patches, backbone, attention_ensemble, calib['mu'], calib['prec'], calib['tau_limit'], max_len, apply_qc=False
            )
            m1_fold_probs.append(prob_m1)
            
            # M2: Euclidean + QC
            prob_m2, n_clean_m2, _ = predict_euclidean_ensemble_workflow(
                patches, backbone, attention_ensemble, calib['mu'], calib['prec'], calib['tau_limit'], max_len, apply_qc=True
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
        t_lower = EUCLIDEAN_WORKFLOW_CONFIG["THRESHOLDS"]["CONFIDENT_IDA"]
        
        if final_prob_m2 is None:
            diagnosis_m3 = "QC Failed (All patches dropped)"
            status_m3 = "ABSTAIN"
            pred_class_m3 = -1
        elif final_prob_m2 >= t_upper:
            diagnosis_m3 = "THL"
            status_m3 = "POSITIVE (High Confidence)"
            pred_class_m3 = 1
        elif final_prob_m2 < t_lower:
            diagnosis_m3 = "IDA"
            status_m3 = "NEGATIVE (High Confidence)"
            pred_class_m3 = 0
        else:
            diagnosis_m3 = "Uncertain / Borderline"
            status_m3 = "UNCERTAIN"
            pred_class_m3 = 1 if final_prob_m2 >= 0.5 else 0 # Fallback
    
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
    # 3. Save Output
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
            
#%% Performance Analysis
# if config.Performance_Report:       
#     import pandas as pd
#     import numpy as np
#     import mygears
    
#     # 1. Calibration Settings
#     T_LOWER = 0.4914  # Boundary for Confident IDA
#     T_UPPER = 0.6505  # Boundary for Confident THL
#     T_OPT   = 0.5742  # Optimal cutoff for the 'Uncertain' fallback
    
#     audit_path = 'val_predictions_mean_QC_Std_20251220-1240.csv'
#     # audit_path = 'val_predictions_argmax_QC_Std_20251220-1240.csv'
#     df = pd.read_csv(audit_path)
    
#     # 2. Extract Ground Truth and Probabilities
#     y_true = df['true_class'].to_numpy()
#     y_prob = df['probability_1'].to_numpy()
    
#     # 3. CALL APPLY_CLINICAL_TRIAGE
#     # We process each patient probability to determine confidence status and triage labels
#     triage_results = [mygears.apply_clinical_triage(p, T_LOWER, T_UPPER, T_OPT) for p in y_prob]
    
#     # 4. Integrate Results into the DataFrame
#     # Convert the list of dictionaries to a DataFrame and merge with original data
#     df_triage = pd.DataFrame(triage_results)
    
#     # We drop the 'probability' column from the triage dataframe to avoid duplication with 'probability_1'
#     df = pd.concat([df, df_triage.drop(columns=['probability'])], axis=1)
    
#     # 5. Deep Analysis of the Triage Output
#     n_total = len(df)
#     n_confident = len(df[df['pred_status'] == "Confident"])
#     n_uncertain = len(df[df['pred_status'] == "Uncertain"])
    
#     print(f"{'='*40}")
#     print(f"[Clinical Report] Triage Strategy: Hybrid")
#     print(f"{'='*40}")
#     print(f"Total Patients:  {n_total}")
#     print(f"Confident Class: {n_confident} ({n_confident/n_total:.1%})")
#     print(f"Uncertain (Grey Zone): {n_uncertain} ({n_uncertain/n_total:.1%})")
#     print(f"{'='*40}")

#     # Review top entries
#     print(df[['patient_id', 'true_class', 'probability_1', 'pred_status', 'diagnosis_label']].head())
    
#     #% Verify Shapes
#     # y_true = df['true_class']
#     # y_pred = df['final_pred']
    
#     # Extract probabilities only for the Confident subset
#     y_pred_confident = df.loc[df['pred_status'] == 'Confident', 'final_pred'].to_numpy()
#     # For a complete academic evaluation, you likely also need the corresponding labels:
#     y_true_confident = df.loc[df['pred_status'] == 'Confident', 'true_class'].to_numpy()
#     y_true = y_true_confident
#     y_pred = y_pred_confident
#     print(f"y_true: {y_true.shape}")
#     print(f"y_pred: {y_pred.shape}")
    
       
#     print('\nVal set classification_report')
#     from sklearn.metrics import classification_report
#     print(classification_report(
#         y_true = y_true,
#         y_pred = y_pred,
#         target_names=config['DATA']['CLASS_LABELS'])
#         )
    
#     p_measures, cm = mygears.evaluate_classification_performance(
#        y_true,
#        y_pred,
#        class_labels=config['DATA']['CLASS_LABELS'],
#        save_dir='.',
#        save_prefix=str('Val'),
#        )
    
#     mygears.plot_confusion_matrices(
#                 confusionmatrix=cm,
#                 p_measures=p_measures,
#                 class_labels=config['DATA']['CLASS_LABELS'],
#                 save_dir='.',
#                 save_prefix=str('Val')
#             )  

#%%

def compute_euclidean_ablation_metrics(csv_path, pre_test_prob=0.40):
    """
    Parses the Euclidean Ensemble CSV and computes clinical metrics for M1, M2, and M3.
    Extracts metrics exclusively for the THL (Positive) class to populate the ablation matrix.
    """
    df = pd.read_csv(csv_path)
    total_cohort_size = len(df)
    
    def _calc_metrics(y_true, y_pred, is_autonomous=False):
        # Handle severe abstention cases (e.g., M3 dropping almost everything)
        if len(y_true) == 0:
            return {"Eval N": 0, "Coverage": "0.0%", "ACC": "0.000", "Sens": "0.000", 
                    "Spec": "0.000", "PPV": "0.000", "NPV": "0.000", "LR+": "0.00", "LR-": "0.00"}
            
        TP = np.sum((y_pred == 1) & (y_true == 1))
        TN = np.sum((y_pred == 0) & (y_true == 0))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        FN = np.sum((y_pred == 0) & (y_true == 1))
        
        denom_tpr = (TP + FN)
        TPR = TP / denom_tpr if denom_tpr > 0 else 0.0
        
        denom_tnr = (TN + FP)
        TNR = TN / denom_tnr if denom_tnr > 0 else 0.0
        
        PPV = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        NPV = TN / (TN + FN) if (TN + FN) > 0 else 0.0
        ACC = (TP + TN) / len(y_true)
        
        epsilon = 1e-10
        LR_plus = TPR / max(1 - TNR, epsilon)
        LR_minus = (1 - TPR) / max(TNR, epsilon)
        
        # Bayesian Post-Test Probability
        pre_test_odds = pre_test_prob / (1 - pre_test_prob)
        post_test_prob = (pre_test_odds * LR_plus) / (1 + (pre_test_odds * LR_plus))
        
        # LaTeX Formatting for divide-by-zero occurrences
        lr_plus_str = r"$\infty$" if FP == 0 and TP > 0 else f"{LR_plus:.2f}"
        lr_minus_str = r"$\infty$" if TN == 0 and FN > 0 else f"{LR_minus:.3f}"
        
        coverage = (len(y_true) / total_cohort_size) * 100 if is_autonomous else 100.0
        
        return {
            "Eval N": len(y_true),
            "Coverage": f"{coverage:.1f}\%",
            "ACC": f"{ACC:.3f}",
            "Sens": f"{TPR:.3f}",
            "Spec": f"{TNR:.3f}",
            "PPV": f"{PPV:.3f}",
            "NPV": f"{NPV:.3f}",
            "LR+": lr_plus_str,
            "LR-": lr_minus_str,
            "Post-test Prob": f"{post_test_prob:.4f}"
        }

    # Extract target columns
    y_true = df['true_label'].to_numpy()
    m1_pred = df['M1_pred_class'].to_numpy()
    m2_pred = df['M2_pred_class'].to_numpy()
    
    # Isolate M3 Confident Cohort
    df_m3 = df[df['M3_status'].str.contains('Confidence', na=False)]
    m3_true = df_m3['true_label'].to_numpy()
    m3_pred = df_m3['M3_predicted_class'].to_numpy()
    
    results = {
        "M1": _calc_metrics(y_true, m1_pred),
        "M2": _calc_metrics(y_true, m2_pred),
        "M3": _calc_metrics(m3_true, m3_pred, is_autonomous=True)
    }
    
    for model_name, metrics in results.items():
        print(f"\n--- {model_name} ---")
        for k, v in metrics.items():
            print(f"{k}: {v}")
            
    return results

# clinical_results  = compute_euclidean_ablation_metrics('7075/Euclidean_Ensemble_Ablation_M1_M2_M3_Results.csv')
# import json
# clinical_json_path = '7075/Euclidean_Ensemble_Ablation_M1_M2_M3_Results.json'
# with open(clinical_json_path, 'w') as f:
#     json.dump(clinical_results, f, indent=4)
# print(f"   Saved clinical metrics -> {clinical_json_path}")

#%%
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

#%%
csv_file = 'Ablation/Euclidean_Ensemble_Ablation_M1_M2_M3_Results.csv'
save_dir = 'Ablation'
df_eucli = pd.read_csv(csv_file)
df_eucli.columns

# ---------------------------------------------------------
# 1. Evaluate M1: Vanilla Euclidean Baseline
# ---------------------------------------------------------
print("\n--- Evaluating M1 (Vanilla Euclidean Baseline) ---")
res_m1 = compute_detailed_clinical_metrics(
    y_true=df_eucli['true_label'], 
    y_pred=df_eucli['M1_pred_class'], 
    y_prob=df_eucli['M1_prob_thl'], 
    total_n=len(df_eucli), 
    name="M1_Vanilla_Euclidean",
    save_path="metrics_M1.json"
)
print(json.dumps(res_m1, indent=4))


# ---------------------------------------------------------
# 2. Evaluate M2: Euclidean MIL + QC Gate
# ---------------------------------------------------------
print("\n--- Evaluating M2 (Euclidean MIL + QC Gate) ---")
res_m2 = compute_detailed_clinical_metrics(
    y_true=df_eucli['true_label'], 
    y_pred=df_eucli['M2_pred_class'], 
    y_prob=df_eucli['M2_prob_thl'], 
    total_n=len(df_eucli), 
    name="M2_Euclidean_QC",
    save_path="metrics_M2.json"
)
print(json.dumps(res_m2, indent=4))


# ---------------------------------------------------------
# 3. Evaluate M3: Euclidean MIL + Auto Triage
# ---------------------------------------------------------
print("\n--- Evaluating M3 (Euclidean MIL + Auto Triage) ---")
# Step A: Filter for only the confident predictions
df_m3_confident = df_eucli[df_eucli['M3_status'] != 'UNCERTAIN'].copy()

# Step B: Call the metric function
# Note: Since M3 doesn't have a separate probability column in your df, 
# we use the underlying M2 probabilities for the AUC calculation of this subset.
res_m3 = compute_detailed_clinical_metrics(
    y_true=df_m3_confident['true_label'], 
    y_pred=df_m3_confident['M3_predicted_class'], 
    y_prob=df_m3_confident['M2_prob_thl'], 
    total_n=len(df_eucli),  # Pass the FULL dataframe length here for correct Coverage %
    name="M3_Euclidean_Triage",
    save_path="metrics_M3.json"
)
print(json.dumps(res_m3, indent=4))


#%%
from statsmodels.stats.contingency_tables import mcnemar

# 1. Load your prediction CSVs
df_euclidean = pd.read_csv('Ablation/Euclidean_Ensemble_Ablation_M1_M2_M3_Results.csv')
df_riemannian = pd.read_csv('Ablation/test_predictions_final_ENSEMBLE_OTHERS_20251221-2255.csv') 

# 2. Merge dataframes to ensure perfect patient-to-patient alignment
df = pd.merge(df_euclidean, df_riemannian, on='patient_id', suffixes=('_euc', '_rie'))

# Print columns just to verify
print("Available columns after merge:\n", df.columns.tolist())

# 3. Extract Ground Truth
# Since the names didn't overlap, we just use 'true_label' (from Euclidean) 
# or 'true_class' (from Riemannian). They are identical.
y_true = df['true_label'].to_numpy()

# 4. Extract Euclidean Predictions (M1, M2, M3)
pred_m1 = df['M1_pred_class'].to_numpy()
pred_m2 = df['M2_pred_class'].to_numpy()
# M3: Treat Abstention ('UNCERTAIN') as an incorrect prediction by mapping it to a dummy class (-99)
pred_m3 = np.where(df['M3_status'] == 'UNCERTAIN', -99, df['M3_predicted_class'])

# 5. Extract Riemannian Predictions (M4, M5, M6)
# IMPORTANT: Adjust 'M4_pred_class' if your M4 (No QC) predictions are saved under a different column name in your CSV.
pred_m4 = df['M4_pred_class'].to_numpy() if 'M4_pred_class' in df.columns else df['final_pred'].to_numpy()

# M5 is the Riemannian model with QC but NO abstention (forced prediction)
pred_m5 = df['final_pred'].to_numpy()

# M6 is the Riemannian model WITH abstention (Confident cohort only)
# Your Riemannian script outputs 'Uncertain' instead of 'UNCERTAIN' for the status
pred_m6 = np.where(df['pred_status'] == 'Uncertain', -99, df['final_pred'])

# 6. Define McNemar Function
def run_mcnemar(y_true, model_a_preds, model_b_preds, name_a, name_b):
    """Calculates McNemar's test for two sets of predictions."""
    correct_a = (model_a_preds == y_true)
    correct_b = (model_b_preds == y_true)
    
    b = np.sum(correct_a & ~correct_b)
    c = np.sum(~correct_a & correct_b)
    
    table = [[np.sum(correct_a & correct_b), b],
             [c, np.sum(~correct_a & ~correct_b)]]
    
    result = mcnemar(table, exact=True)
    
    print(f"\n--- {name_a} vs {name_b} ---")
    print(f"Discordant Pairs: {name_a} better = {b}, {name_b} better = {c}")
    print(f"P-Value: {result.pvalue:.4f}")
    
    if result.pvalue < 0.05:
        print("Conclusion: Statistically Significant Difference (p < 0.05)")
    else:
        print("Conclusion: No Statistically Significant Difference")
        
    return result.pvalue

# 7. Execute Tests
print("\nEvaluating McNemar's Tests for Ablation Matrix:")
p_m1_m4 = run_mcnemar(y_true, pred_m1, pred_m4, "M1 (Euclidean Base)", "M4 (Riemannian Base)")
p_m2_m5 = run_mcnemar(y_true, pred_m2, pred_m5, "M2 (Euc + QC)", "M5 (Rie + QC)")
p_m3_m6 = run_mcnemar(y_true, pred_m3, pred_m6, "M3 (Euc + Triage)", "M6 (Rie + Triage)")

#%%
t2 = datetime.now() - t1
print('\nAll execution time: ', t2)