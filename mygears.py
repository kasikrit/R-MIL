#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 17:25:41 2025

@author: kasikritdamkliang
"""
import tensorflow as tf
from tensorflow import keras
import os, platform
import time
import cv2
from datetime import datetime
from tqdm import tqdm
import glob
# import itertools
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from PIL import Image 
import seaborn as sns 
from collections import defaultdict
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import random
import tensorflow_addons as tfa
from pathlib import Path
from natsort import natsorted
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

class CustomTQDMProgressBar(tfa.callbacks.TQDMProgressBar):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Safely extract the learning rate as a scalar for logging
        lr = tf.keras.backend.get_value(self.model.optimizer.learning_rate)
        
        # Add the learning rate to the logs for display
        logs['learning_rate'] = lr
        super().on_epoch_end(epoch, logs)

# def cfg_to_dict(cfg_node):
#     """
#     Recursively convert a YACS CfgNode to a nested dictionary.
#     """
#     if isinstance(cfg_node, CN):
#         return {k: v for k, v in cfg_node.items()}
#     return cfg_node

def cfg_to_dict(cfg_node):
    """
    Recursively convert a YACS CfgNode to a nested dictionary.
    """
    # if isinstance(cfg_node, config):
    #     return {k: v for k, v in cfg_node.items()}
    return {k: v for k, v in cfg_node.items()}

def filter_hidden_files(file_paths):
    """
    Filter out paths that start with a '.' indicating they are hidden.
    This only considers the filename, not part of the path, as hidden.
    """
    return [path for path in file_paths if not os.path.basename(path).startswith('.')]

def make_pair(imgs, labels):
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

def convertBinary(image, debug=False):
    h, w, _ = image.shape
    # Convert the RGB image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply a binary threshold
    _, binary_image = cv2.threshold(gray_image, 160, 255,
                    cv2.THRESH_BINARY)
    
    pixels = cv2.countNonZero(binary_image)
    pixel_ratio = (pixels/(h * w)) * 100
    title = 'Pixel ratio: {:.2f}%'.format(pixel_ratio)
    if debug==True:
        plt.figure()
        plt.subplot(121)
        plt.imshow(image)
        plt.subplot(122)
        plt.imshow(binary_image, cmap='gray')
        plt.title(title)
        print(title)
    
    # Convert back to 3 channels (to ensure it has RGB channels)
    binary_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    
    return binary_image
        

def preprocess_for_alexnet(image):
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # image_resized = cv2.resize(image_bgr, (227, 227))
    image_resized = image_bgr.astype(np.float32)

    # 4. Subtract ImageNet mean values for each channel (RGB)
    mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
    image_normalized = image_resized - mean

    return image_normalized

def create_balanced_dataframe(patchPaths_train, class_list_train):
    """
    Creates a balanced DataFrame by oversampling the minority classes.

    Args:
    patchPaths_train (list): List of file paths to training images.
    class_list_train (list): List of class labels corresponding to the training images.

    Returns:
    pd.DataFrame: A DataFrame containing balanced image paths and their respective class labels.
    """
    
    # Initialize RandomOverSampler
    ros = RandomOverSampler()

    # Convert class list to NumPy array for compatibility with RandomOverSampler
    y = np.array(class_list_train)
    X = np.array(patchPaths_train).reshape(-1, 1)  # Reshape to 2D array for compatibility

    # Print class distribution before resampling
    print("Class distribution before resampling:", Counter(y))

    # Perform the oversampling
    X_Ros, y_Ros = ros.fit_resample(X, y)

    # Print class distribution after resampling
    print("Class distribution after resampling:", Counter(y_Ros))

    # Create a balanced DataFrame
    balanced_df = pd.DataFrame({
        'path': X_Ros.flatten(),  # Flatten to 1D array
        'class': y_Ros            # Class labels
    })
    
    return balanced_df

def get_png_files_and_classes(dataset_root,
    patient_set,
    aug=False):
    """
    This function takes the root directory of the dataset and a set of patient data (test_set or train_set),
    and returns three items:
    1. png_list: A list of all PNG file paths.
    2. class_list: A list of corresponding class IDs for each PNG file.
    3. class_counts: A dictionary with the total number of PNG files for each class.
    
    Parameters:
    - dataset_root (str): The root directory of the dataset.
    - patient_set (list of tuples): A list where each entry is a tuple (patient_id, class_id).
    
    Returns:
    - png_list (list of str): List of all PNG file paths.
    - class_list (list of str): List of corresponding class IDs for each PNG file.
    - class_counts (dict): A dictionary with the total number of PNG files for each class.
    """
    png_list = []    # To store the paths of all PNG files
    class_list = []  # To store the corresponding class IDs
    class_counts = defaultdict(int)  # To store the count of PNG files for each class
    
    # Iterate through the patient_set to read *.png files
    for (patient_path, class_id) in patient_set:
        print(patient_path, class_id)
        # break
        # Construct the patch path
        if aug:
            patch_path = os.path.join(dataset_root,
                # class_id,
                patient_path,
                'cells',
                '*',
                '*.png')
            print(f"{patch_path=}")    
            png_files = glob.glob(patch_path)

            
            patch_path_aug = os.path.join(dataset_root,
                    # class_id,
                    patient_path,
                    'aug',
                    '*.png')
            print(f"{patch_path_aug=}")  
            # Use glob to find all png files in the constructed path
            png_files_cell = glob.glob(patch_path)
            png_files_aug = glob.glob(patch_path_aug)
            png_files = png_files_cell + png_files_aug
        else:
            patch_path = os.path.join(dataset_root,
                #class_id,
                patient_path,
                'cells',
                '*',
                '*.png')  
            print(f"{patch_path=}")    
            png_files = glob.glob(patch_path)
        
        # patient_id = patient_path.split(os.sep)[2]
        print(f"Patient {patient_path} (Class {class_id}): Found {len(png_files)} PNG files.")
        
        # Append png_files to png_list and class_list
        png_list.extend(png_files)        # Add all PNG file paths to png_list
        class_list.extend([class_id] * len(png_files))  # Add the class_id for each PNG file
        
        # Update the count of PNG files for the current class_id
        class_counts[class_id] += len(png_files)
    
    # Return both lists and the class counts
    return png_list, class_list, class_counts

#%%
def count_plot(df, title, save_dir=None, save=False):
    fig = plt.figure(dpi=300)   
    # Set the desired class order: first '0' then '1'
    class_order = ['0', '1']
    
    # Create a countplot, explicitly setting the order of classes
    palette = sns.color_palette("Set2", 3)
    ax = sns.countplot(x=df['label'],
                       palette=palette,
                       order=class_order)  # Specify the order of classes
    # Add class count labels on each bar
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')

    # Add labels and title
    plt.xlabel("Class")
    plt.title(title)   
    if save==True:
        plot_file = os.path.join(save_dir, f"{title}.png")
        fig.savefig(plot_file)  
        print(f"Saved {plot_file}")
    
    plt.show()

def plot_images_with_labels(images, 
    labels,
    train_log,
    set_label,
    # save=False,
    config,
    ):
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
    if config['SAVE']:
        plot_file = os.path.join(train_log, f"{plot_title}.png")
        fig.savefig(plot_file, dpi=120)
        print(f'Saved {plot_file}')

def plot_images_by_class(df, class_label, savefile):
    """
    Plots a grid of images for a specific class (IDA or Thalassemia).

    Args:
    - df (DataFrame): DataFrame containing image paths and labels.
    - class_label (int): Class to filter (0 = IDA, 1 = Thalassemia).
    - num_images (int): Number of images to display.

    Returns:
    - None: Displays the images in a grid.
    """
    num_images=12
    # Filter dataframe based on the selected class
    class_df = df[df['class'] == class_label]

    # Check if we have enough images
    num_images = min(num_images, len(class_df))
    if num_images == 0:
        print(f"No images found for class {class_label}")
        return

    # Sample images randomly
    # sample_df = class_df.sample(n=num_images, random_state=42).reset_index(drop=True)
    sample_df = class_df.sample(n=num_images).reset_index(drop=True)

    # Create subplots
    fig, axes = plt.subplots(3, 4, figsize=(14, 10), dpi=120)
    axes = axes.flatten()

    for i in range(num_images):
        img_path = sample_df.iloc[i]['path']  # Use .iloc[i] instead of .loc[i]
        label = sample_df.iloc[i]['class']

        # Load the image using OpenCV (convert to RGB)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Display the image
        axes[i].imshow(img)
        # axes[i].set_title(f"Label: {'IDA' if label == '0' else 'Thalassemia'}", fontsize=12)
        axes[i].axis("off")

    # Hide unused subplots if there are fewer than 12 images
    for j in range(num_images, len(axes)):
        axes[j].axis("off")

    # Set the overall title for the figure
    # plt.suptitle(f"Sample Images for {'IDA' if class_label == '0' else 'Thalassemia'}", fontsize=18)
    plt.show()
    fig.savefig(savefile)

#%
def model_mapping_function(model_name, config):
    model_root = config.BASEMODEL_PATH  # Assuming config contains the save path
    # Select appropriate model paths based on color mode
    if config['DATA']['COLOR'] == 'RGB':
        model_mapping = {
            'EfficientNetV2S': [
                os.path.join(model_root, 'Linux-Fold-0-EfficientNetV2S-rgb-bal-aug-20250302-2215.hdf5'),
                os.path.join(model_root, 'Linux-Fold-1-EfficientNetV2S-rgb-bal-aug-20250303-1001.hdf5'),
                os.path.join(model_root, 'Linux-Fold-2-EfficientNetV2S-rgb-bal-aug-20250303-1001.hdf5'),
                os.path.join(model_root, 'Linux-Fold-3-EfficientNetV2S-rgb-bal-aug-20250303-1001.hdf5'),
                os.path.join(model_root, 'Linux-Fold-4-EfficientNetV2S-rgb-bal-aug-20250303-1001.hdf5'),
            ]
        }
    if config['DATA']['COLOR'] == 'BGR': 
        model_mapping = { #cell patch
            'EfficientNetV2S': [
                os.path.join(model_root, 'Linux-Fold-0-EfficientNetV2S-bal-aug-20250110-1607.hdf5'),
                os.path.join(model_root, 'Linux-Fold-1-EfficientNetV2S-bal-aug-20250110-1607.hdf5'),
                os.path.join(model_root, 'Linux-Fold-2-EfficientNetV2S-bal-aug-20250110-1607.hdf5'),
                os.path.join(model_root, 'Linux-Fold-3-EfficientNetV2S-bal-aug-20250109-1903.hdf5'),
                os.path.join(model_root, 'Linux-Fold-4-EfficientNetV2S-bal-aug-20250110-1607.hdf5'),
            ],
            
            # 'ConvNeXtLarge': [
            #     os.path.join(model_root, 'Linux-Fold-0-stage2-ConvNeXtLarge-bal-aug-20250409-0043.hdf5'),
            #     os.path.join(model_root, 'Linux-Fold-1-stage2-ConvNeXtLarge-bal-aug-20250418-2141.hdf5'),
            #     os.path.join(model_root, 'Linux-Fold-2-stage2-ConvNeXtLarge-bal-aug-20250418-2141.hdf5'),
            #     os.path.join(model_root, 'Linux-Fold-3-stage2-ConvNeXtLarge-bal-aug-20250418-2141.hdf5'),
            #     os.path.join(model_root, 'Linux-Fold-4-stage2-ConvNeXtLarge-bal-aug-20250418-2141.hdf5'),
            # ],
            # Assuming RGBA color mode  => BGR
            'ConvNeXtLarge': [ #segmented cell - baseline
                os.path.join(model_root, 'Linux-Fold-0-ConvNeXtLarge-classweights-20251205-1028.hdf5'),
                os.path.join(model_root, 'Linux-Fold-1-ConvNeXtLarge-classweights-20251205-2213.hdf5'),
                os.path.join(model_root, 'Linux-Fold-2-ConvNeXtLarge-classweights-20251205-2213.hdf5'),
                os.path.join(model_root, 'Linux-Fold-3-ConvNeXtLarge-classweights-20251205-2213.hdf5'),
                os.path.join(model_root, 'Linux-Fold-4-ConvNeXtLarge-classweights-20251205-2213.hdf5'),
            ],
            
            'ConvNeXtLarge_cell_patch': [ # cell patch - stage2
                os.path.join(model_root, 'Linux-Fold-0-stage2-ConvNeXtLarge-classweights-20251113-1824.hdf5'),
                os.path.join(model_root, 'Linux-Fold-1-stage2-ConvNeXtLarge-classweights-20251113-1824.hdf5'),
                os.path.join(model_root, 'Linux-Fold-2-stage2-ConvNeXtLarge-classweights-20251113-1824.hdf5'),
                os.path.join(model_root, 'Linux-Fold-3-stage2-ConvNeXtLarge-classweights-20251113-1824.hdf5'),
                os.path.join(model_root, 'Linux-Fold-4-stage2-ConvNeXtLarge-classweights-20251113-1824.hdf5'),
            ],
        }

    # Check if the requested model exists in the mapping
    if model_name in model_mapping:
        return model_mapping[model_name]
    else:
        raise ValueError(f"No model mapping found for model {model_name}")

from ModelBuilder import ModelBuilder
from tensorflow.keras.models import load_model
from keras.utils import custom_object_scope
from keras.applications.convnext import LayerScale
def load_ensemble_models(model_names, config):
    print("\nLoading models")
    models = []
    for model_name in tqdm(model_names):
        model_paths = model_mapping_function(model_name, config)  # Get all model paths for the name
        for model_path in model_paths:
            print(f"\nLoading model from: {model_path}")
            model_builder = ModelBuilder(model_name=model_name, config=config)
            best_model = model_builder.create_model()
            if 'ConvNeXt' in model_path:
                with custom_object_scope({'LayerScale': LayerScale}):
                    best_model.load_weights(model_path, by_name=True, skip_mismatch=True)
                    # best_model.load_model(model_path)
            models.append(best_model)
    return models

import gc
def load_ensemble_models_1(model_mapping_dict, config):
    """
    Loads models based on a dictionary mapping architecture names to file paths.
    
    Args:
        model_mapping_dict (dict): Keys are model names (e.g., 'EfficientNetV2S'), 
                                   Values are lists of file paths.
        config (dict): Configuration dictionary for ModelBuilder.
        
    Returns:
        list: A flat list of loaded Keras models.
    """
    print("\nLoading models from mapping dictionary...")
    models = []
    
    # Iterate through the dictionary items (Architecture Name -> List of Paths)
    for model_name, model_paths in model_mapping_dict.items():
        
        print(f"\nProcessing Architecture: {model_name}")
        
        for model_path in tqdm(model_paths, desc=f"Loading {model_name} Folds"):
            print(f"  -> Loading weights from: {model_path}")
            
            # 1. Create the model skeleton using the architecture name
            model_builder = ModelBuilder(model_name=model_name, config=config)
            best_model = model_builder.create_model()
            
            # 2. Load weights (Handle ConvNeXt custom objects if necessary)
            try:
                if 'ConvNeXt' in model_name or 'ConvNeXt' in model_path:
                    # Ensure LayerScale is imported or available in context
                    with custom_object_scope({'LayerScale': LayerScale}): 
                        best_model.load_weights(model_path, by_name=True, skip_mismatch=True)
                else:
                    best_model.load_weights(model_path, by_name=True, skip_mismatch=True)
                
                models.append(best_model)
                
            except Exception as e:
                print(f"Error loading {model_path}: {e}")

            # Optional: Garbage collection to free up temporary build artifacts
            gc.collect()

    print(f"\nSuccessfully loaded {len(models)} models.")
    return models

def predict_for_model(model, images, verbose=0):
    """
    Predicts the class probabilities for a single model.
    
    Args:
    model: The model to use for prediction.
    images (numpy array): Batch of images for inference.

    Returns:
    numpy array: Class probabilities from the model.
    """
    prediction = model.predict(images, verbose=verbose)
    return prediction

def print_model_summary(model):
    total_params = model.count_params()  # Total parameters (trainable + non-trainable)
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])  # Trainable parameters
    non_trainable_params = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])  # Non-trainable parameters

    # Print the model summary information
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: {non_trainable_params:,}")


def plot_training_metrics(history, train_log, save_plots=True):
    """
    Function to plot training metrics including accuracy, validation accuracy, loss, 
    validation loss, and learning rate (LR).

    Parameters:
    - history: The history object returned by the model training.
    - train_log: A directory path where the plots should be saved.
    - save_plots: Boolean, if True, the plots will be saved as PNG files.
    """

    # Extract data from history
    epochs = range(1, len(history.history['loss']) + 1)
    acc = history.history.get('accuracy')
    val_acc = history.history.get('val_accuracy')
    loss = history.history.get('loss')
    val_loss = history.history.get('val_loss')
    lr = history.history.get('lr')

    # Accuracy and Validation Accuracy Plot
    fig1 = plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'orange', label='Validation accuracy')
    plt.title('Training and Validation Accuracy ' + train_log)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Save accuracy plot if save_plots is True
    if save_plots:
        acc_file = f'{train_log}-accuracy.png'
        acc_file_path = os.path.join(train_log, acc_file)
        fig1.savefig(acc_file_path)

    # Create subplots: 1 row, 2 columns
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), dpi=300)

    # Plot 1: Loss and Validation Loss
    axs[0].plot(epochs, loss, 'b', label='Training loss')
    axs[0].plot(epochs, val_loss, 'orange', label='Validation loss')
    axs[0].set_title('Training and Validation Loss ' + train_log)
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    # Plot 2: Learning Rate (LR)
    axs[1].plot(epochs, lr, 'g', label='Learning rate')
    axs[1].set_title('Learning Rate ' + train_log)
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Learning Rate')
    axs[1].legend()

    # Display the subplots
    plt.tight_layout()
    plt.show()

    # Save plots if save_plots is True
    if save_plots:
        plot_file = f'{train_log}-loss-lr-subplot.png'
        plot_file_path = os.path.join(train_log, plot_file)
        fig.savefig(plot_file_path)
            
def plot_training_metrics_from_df(history_df, train_log, save_plots=True):
    # Extract metrics from the DataFrame
    epochs = range(1, len(history_df) + 1)
    acc = history_df['accuracy']
    val_acc = history_df['val_accuracy']
    loss = history_df['loss']
    val_loss = history_df['val_loss']
    lr = history_df['lr']

    # Accuracy and Validation Accuracy Plot
    fig1 = plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'orange', label='Validation accuracy')
    plt.title('Training and Validation Accuracy ' + train_log)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Save accuracy plot if save_plots is True
    if save_plots:
        os.makedirs(train_log, exist_ok=True)
        acc_file = f'{train_log}-accuracy.png'
        acc_file_path = os.path.join(train_log, acc_file)
        fig1.savefig(acc_file_path)

    # Create subplots: 1 row, 2 columns
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), dpi=300)

    # Plot 1: Loss and Validation Loss
    axs[0].plot(epochs, loss, 'b', label='Training loss')
    axs[0].plot(epochs, val_loss, 'orange', label='Validation loss')
    axs[0].set_title('Training and Validation Loss ' + train_log)
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    # Plot 2: Learning Rate (LR)
    axs[1].plot(epochs, lr, 'g', label='Learning rate')
    axs[1].set_title('Learning Rate ' + train_log)
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Learning Rate')
    axs[1].legend()

    # Display the subplots
    plt.tight_layout()
    plt.show()

    # Save plots if save_plots is True
    if save_plots:
        os.makedirs(train_log, exist_ok=True)
        plot_file = f'{train_log}-loss-lr-subplot.png'
        plot_file_path = os.path.join(train_log, plot_file)
        fig.savefig(plot_file_path)
        
def plot_training_metrics_without_lr(history, prefix, save_dir=None):
    # Extract data from history
    epochs = range(1, len(history.history['loss']) + 1)
    acc = history.history.get('accuracy')
    val_acc = history.history.get('val_accuracy')
    loss = history.history.get('loss')
    val_loss = history.history.get('val_loss')

    # --- Plot 1: Accuracy ---
    fig1 = plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'orange', label='Validation accuracy')
    plt.title('Training and Validation Accuracy ' + prefix)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # [FIX] Save BEFORE plt.show()
    if save_dir is not None:
        acc_file = f'{prefix}-accuracy.png'
        acc_file_path = os.path.join(save_dir, acc_file)
        fig1.savefig(acc_file_path)
        print(f"Saved -> {acc_file_path}")

    plt.show()
    plt.close(fig1) # Good practice to close figure to free memory

    # --- Plot 2: Loss ---
    fig2 = plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'orange', label='Validation loss')
    plt.title('Training and Validation Loss ' + prefix)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # [FIX] Save BEFORE plt.show()
    if save_dir is not None:
        plot_file = f'{prefix}-loss.png'
        plot_file_path = os.path.join(save_dir, plot_file)
        fig2.savefig(plot_file_path)
        print(f"Saved -> {plot_file_path}")

    plt.show()
    plt.close(fig2)

def format_p_measures(p_measures):
    for key, value in p_measures.items():
        if key in ['TP', 'TN', 'FP', 'FN']:  # Keep these values as integers
            if isinstance(value, list):  # If value is a list, convert each item to int
                p_measures[key] = [int(v) for v in value]
            else:
                p_measures[key] = int(value)
        else:  # For other metrics, format to two decimal points
            if isinstance(value, list):  # If it's a list, round each item to 2 decimal places
                p_measures[key] = [round(v, 3) for v in value]
            elif isinstance(value, (float, int)):  # If it's a scalar, round it to 2 decimals
                p_measures[key] = round(value, 3)
    return p_measures

def verify_images(df, image_column):
    """
    Verifies the images in the DataFrame. Removes rows with corrupted or unidentifiable images.

    Parameters:
    - df: The pandas DataFrame containing image paths.
    - image_column: The name of the column in the DataFrame containing the image paths.

    Returns:
    - A DataFrame with only valid images.
    """
    valid_image_paths = []

    for idx, row in df.iterrows():
        image_path = row[image_column]
        try:
            # Try to open the image
            img = Image.open(image_path)
            img.verify()  # Verify the image (this catches image corruption)
            valid_image_paths.append(image_path)
        except (IOError, SyntaxError, Image.UnidentifiedImageError) as e:
            print(f"Skipping corrupted or invalid image: {image_path}, {e}")
    
    # Filter the DataFrame to keep only valid images
    return df[df[image_column].isin(valid_image_paths)]

     
#% Horizontal and vertical flip  
def augment_images_and_update_df(df, 
                                 #dest_dir,
                                 save=False):
    """
    Function to perform horizontal and vertical flips on images and append augmented images to the DataFrame.
    
    Args:
    - df (pd.DataFrame): DataFrame containing 'path' and 'class' columns.
    - dest_dir (str): Destination directory to save augmented images.
    
    Returns:
    - df (pd.DataFrame): Updated DataFrame with augmented image paths and their respective classes.
    """
    
    # List to store new augmented file paths and classes
    augmented_data = []

    # Loop through the DataFrame
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        file = row['path']
        class_id = str(row['class'])

        # Open the image
        try:
            img = Image.open(file)
        except FileNotFoundError:
            print(f"File not found: {file}")
            continue
        
        # Extract patient_id from the file path
        parts = file.split('/')
        patient_id = parts[3]

        # Horizontal flip
        img_h_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
       
        # Vertical flip
        img_v_flip = img.transpose(Image.FLIP_TOP_BOTTOM)
        
        # Save the flipped images
        base_dir, filename = os.path.split(file)
        filename_without_extension = os.path.splitext(filename)[0]
        
        # Destination paths for augmented images
        patient_dir = os.path.join(
            # dest_dir,
            #base_dir
            #class_id,
            #patient_id,
            base_dir.split(os.sep)[0],
            base_dir.split(os.sep)[1],
            base_dir.split(os.sep)[2],
            base_dir.split(os.sep)[3],
            'aug')
        # Horizontal flip path
        h_flip_path = os.path.join(patient_dir, f'{filename_without_extension}_h_flip.png')
        # Vertical flip path
        v_flip_path = os.path.join(patient_dir, f'{filename_without_extension}_v_flip.png')
        
        if save:
            # Ensure the destination directory exists
            os.makedirs(patient_dir, exist_ok=True)
            img_h_flip.save(h_flip_path)
            img_v_flip.save(v_flip_path)
        
        # Append the new augmented paths and classes to the list
        augmented_data.append({'path': h_flip_path, 'class': class_id})
        augmented_data.append({'path': v_flip_path, 'class': class_id})

    # Convert augmented data to DataFrame and append it to the original DataFrame
    augmented_df = pd.DataFrame(augmented_data)
    df = pd.concat([df, augmented_df], ignore_index=True)

    return df

from collections import defaultdict
from pathlib import Path
from natsort import natsorted
def create_cell_dataframe_debug(config):
    """
    Debugging version to trace where file search fails
    and count the number of patches for each patient_id.
    
    UPDATES:
    - Sorts label directories (0 then 1).
    - Sorts patient directories (alphanumeric/natural order).
    """
    all_patch_files = []
    patient_patch_count = defaultdict(int)

    for each_base_dir in config.base_dir_list:
        # base_dir = Path(each_base_dir)
        base_dir = Path(os.path.join(config.DATASET, each_base_dir))
        print(f"\n--- Searching in Base Directory: {base_dir} ---")
        
        # Level 1: label directories (0 and 1) -> SORTED
        label_dirs = natsorted([p for p in base_dir.glob('[01]') if p.is_dir()])
        print(f"Found {len(label_dirs)} label directories: {[d.name for d in label_dirs]}")

        for label_dir in label_dirs:
            # Level 2: patient directories -> SORTED
            patient_dirs = natsorted([p for p in label_dir.glob('*') if p.is_dir()])
            print(f"  Found {len(patient_dirs)} patient directories in '{label_dir.name}'")

            for patient_dir in patient_dirs:
                patient_id = patient_dir.name  # Use directory name as patient_id
                glob_pattern = "normalized/*-cells-256"
                cell_dirs = list(patient_dir.glob(glob_pattern))
                
                if not cell_dirs:
                    print(f"    -> WARNING: No '{glob_pattern}' directory found in {patient_dir}")
                else:
                    total_patches = 0
                    for cell_dir in cell_dirs:
                        # Patch files are already sorted here in your original code
                        patch_files = natsorted(list(cell_dir.glob('*.png')))
                        num_patches = len(patch_files)
                        total_patches += num_patches
                        all_patch_files.extend(patch_files)

                    patient_patch_count[patient_id] += total_patches
                    print(f"    Patient '{patient_id}': {total_patches} patches found")

    print("\n--- Summary ---")
    print(f"Total .png files found across all searches: {len(all_patch_files)}")
    print(f"Total unique patients found: {len(patient_patch_count)}")
    
    # Sort by patch count (descending)
    sorted_counts = sorted(patient_patch_count.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 10 patients by patch count:")
    for pid, count in sorted_counts[:10]:
        print(f"  {pid}: {count} patches")

    return all_patch_files, dict(patient_patch_count)

def create_cell_dataframe(config, glob_pattern):
    """
    Finds all cell patches and parses their paths to create a structured DataFrame.
    """
    all_patch_files = []
    for each_base_dir in config.base_dir_list:
        # base_dir = Path(each_base_dir)
        base_dir = Path(os.path.join(config.DATASET, each_base_dir))
        print(f"{base_dir=}")
        # Find all .png files inside subdirectories like '...-cells-256/'
        # glob_pattern = "[01]/*/normalized/*-cells-256/*.png"
        # glob_pattern = "[0]/2364-55/normalized/*-cells-256/*.png"
        patch_files = natsorted(list(base_dir.glob(glob_pattern)))
        all_patch_files.extend(patch_files)

    if not all_patch_files:
        print("No patch files found. Check your base_dir_list and paths.")
        return pd.DataFrame()

    data_records = []
    for p in all_patch_files:
        try:
            record = {
                'base_dir': p.parts[-6], # e.g., '20250111'
                'label': int(p.parts[-5]), # e.g., 1
                'patient_id': p.parts[-4], # e.g., '100392-52'
                'slide_id': p.parent.name.split('-cells-256')[0], # e.g., 'DSC10105'
                'cell_id': p.name, # e.g., 'cell-0.png'
                'patch_path': p
            }
            
            # break;
            data_records.append(record)
        except (IndexError, ValueError) as e:
            print(f"Could not parse path: {p}. Error: {e}")
    # print("Debug: ", record)
    return pd.DataFrame(data_records)

def create_cell_dataframe_flex(config, glob_pattern):
    """
    Finds all cell patches and stores paths relative to config.DATASET.
    Output path format: "20240920/0/patient_id/..." (POSIX style)
    """
    # 1. Define the Anchor (The root path to strip away)
    dataset_root = Path(config.DATASET)
    
    all_patch_files = []
    
    # 2. Collect all absolute paths first
    for each_base_dir in config.base_dir_list:
        base_dir = dataset_root / each_base_dir
        
        # print(f"Searching: {base_dir}")
        if not base_dir.exists():
            continue
            
        patch_files = natsorted(list(base_dir.glob(glob_pattern)))
        all_patch_files.extend(patch_files)

    if not all_patch_files:
        print("No patch files found. Check your base_dir_list and paths.")
        return pd.DataFrame()

    data_records = []
    for p in all_patch_files:
        try:
            # --- CRITICAL STEP: Create Relative Path ---
            # Removes "H:\My Drive\THL-g\" from the start
            rel_path_obj = p.relative_to(dataset_root)
            
            # --- Parsing based on Relative Structure ---
            # Expected Relative Path: 20240920 / 0 / 2610-52 / normalized / DSC... / cell.png
            # indices:                   0       1      2           3           4         5
            
            parts = rel_path_obj.parts
            
            record = {
                'base_dir': parts[0],      # '20240920'
                'label': int(parts[1]),    # 1
                'patient_id': parts[2],    # '2610-52'
                'slide_id': p.parent.name.split('-cells-256')[0],
                'cell_id': p.name,
                
                # --- CRITICAL STEP: Store as POSIX String ---
                # This ensures it looks like "20240920/0/..." on BOTH Windows and Linux
                'patch_path': rel_path_obj.as_posix()
            }
            
            data_records.append(record)
        except (IndexError, ValueError) as e:
            print(f"Could not parse path: {p}. Error: {e}")

    return pd.DataFrame(data_records)


def create_cell_dataframe_anerbc(config, glob_pattern="*.png"):
    """
    Minimalist loader for AneRBC-II Dataset Audit.
    Focuses only on labels and paths for Riemannian curvature calculation.
    """
    dataset_root = Path(config.DATASET)
    data_records = []
    
    # AneRBC-II Label Mapping
    label_map = {
        "Healthy_individuals": 0,
        "Anemic_individuals": 1
    }

    print(f"[AneRBC Loader] Scanning root: {dataset_root}")

    # 1. Iterate through Class Folders
    for class_folder in dataset_root.iterdir():
        if not class_folder.is_dir() or class_folder.name not in label_map:
            continue
            
        current_label = label_map[class_folder.name]
        
        # 2. Iterate through base_dir_list (e.g., ['RGB_segmented'])
        for b_dir in config.base_dir_list:
            target_path = class_folder / b_dir
            
            if not target_path.exists():
                continue
                
            print(f"   Processing {class_folder.name} -> {b_dir}...")
            
            # 3. Glob patches and sort
            patch_files = natsorted(list(target_path.glob(glob_pattern)))
            
            for p in patch_files:
                try:
                    # Get path relative to dataset_root for POSIX storage
                    rel_path = p.relative_to(dataset_root)
                    
                    record = {
                        'base_dir': b_dir,
                        'label': current_label,
                        'patch_path': rel_path.as_posix()
                    }
                    data_records.append(record)
                    
                except Exception:
                    continue

    df = pd.DataFrame(data_records)
    
    if not df.empty:
        print(f"\n[AneRBC Loader] Successfully loaded {len(df)} patches.")
        print(f"Distribution:\n{df['label'].value_counts().rename({0:'Healthy', 1:'Anemic'})}")
    else:
        print("[!] No patches found. Verify config.DATASET and config.base_dir_list.")
        
    return df
    
from sklearn.model_selection import StratifiedShuffleSplit
def stratified_subsample(df, n_samples=30_000, random_state=42):
    y = df['label']
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_samples, random_state=random_state)
    idx, _ = next(sss.split(df, y))
    return df.iloc[idx]

def identify_unique_and_duplicate_patients(df, patient_col="patient_id", fold_col="val_fold"):
    """
    Identify unique and duplicate patients in an ensemble prediction dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain patient_id and val_fold columns.
    patient_col : str
        Column name for patient ID.
    fold_col : str
        Column name for fold index.

    Returns
    -------
    results : dict
        {
            "unique_patients": list,
            "duplicate_patients": list,
            "n_unique": int,
            "n_duplicates": int,
            "duplicate_folds": pd.Series
        }
    """

    print("\n--- Identify Unique and Duplicate Patients ---")

    # 1. Unique patients
    unique_patients = sorted(df[patient_col].unique())
    print(f"Unique patients: {len(unique_patients)}")

    # 2. Count occurrences of each patient
    patient_counts = df[patient_col].value_counts()

    # Patients appearing more than once
    duplicate_patients = patient_counts[patient_counts > 1].index.tolist()
    print(f"Duplicate patients: {len(duplicate_patients)}")

    if len(duplicate_patients) > 0:
        print(duplicate_patients)

        # 3. For each duplicate, list the folds where they appeared
        duplicate_folds = (
            df.groupby(patient_col)[fold_col]
            .unique()
            .loc[duplicate_patients]
        )

        print("\nDuplicate patients and their folds:")
        print(duplicate_folds)

    else:
        duplicate_folds = pd.Series(dtype=object)
        print("No duplicated patients detected.")

    return {
        "unique_patients": unique_patients,
        "duplicate_patients": duplicate_patients,
        "n_unique": len(unique_patients),
        "n_duplicates": len(duplicate_patients),
        "duplicate_folds": duplicate_folds
    }

import json

def load_thresholds(json_path):
    """
    Loads manifold thresholds from a JSON file.
    
    Args:
        json_path (str): Path to the .json file saved in Phase 2.
        
    Returns:
        mean_threshold (float): The average threshold across all folds.
        fold_thresholds (dict): Dictionary mapping 'fold_X' -> threshold value.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"[Error] Threshold file not found at: {json_path}")
        
    print(f">> Loading Thresholds from: {os.path.basename(json_path)}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    # 1. Extract Global Mean (First return value)
    mean_threshold = data.get('statistics', {}).get('mean', 1.0)
    
    # 2. Extract Per-Fold Dictionary (Second return value)
    fold_thresholds = data.get('per_fold', {})
    
    # Validation
    if not fold_thresholds:
        raise ValueError("JSON file does not contain 'per_fold' data.")
        
    print(f"   [Loaded] Mean Threshold: {mean_threshold:.4f}")
    print(f"   [Loaded] Folds found: {list(fold_thresholds.keys())}")
    
    return mean_threshold, fold_thresholds


#%% 
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_classification_performance(y_true, y_pred, class_labels, save_dir='.', save_prefix='patient_test'):
    """
    Computes detailed classification metrics, prints summary, and saves to JSON.
    """
    # 1. Classification Report
    class_report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
    
    # 2. Confusion Matrix & Derived Metrics
    cm = confusion_matrix(y_true, y_pred)
    
    FP = cm.sum(axis=0) - np.diag(cm)  
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    
    # Metrics
    TPR = TP / (TP + FN + 1e-10) # Sensitivity / Recall
    TNR = TN / (TN + FP + 1e-10) # Specificity
    PPV = TP / (TP + FP + 1e-10) # Precision
    NPV = TN / (TN + FN + 1e-10) # Negative Predictive Value
    FPR = FP / (FP + TN + 1e-10) # Fall out
    FNR = FN / (TP + FN + 1e-10) # False Negative Rate
    FDR = FP / (TP + FP + 1e-10) # False Discovery Rate
    ACC = (TP + TN) / (TP + FP + FN + TN + 1e-10) # Overall Accuracy
    
    sen = np.nanmean(TPR)
    spec = np.nanmean(TNR)
    
    print(f"Mean Sensitivity: {sen:.4f}, Mean Specificity: {spec:.4f}")
    
    # 3. Construct Dictionary
    p_measures = {
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
        'TPR': TPR, 'TNR': TNR, 'PPV': PPV, 'NPV': NPV,
        'FPR': FPR, 'FNR': FNR, 'FDR': FDR, 'ACC': ACC,
        'precision': class_report['macro avg']['precision'],
        'f1-score' : class_report['macro avg']['f1-score'],
        'MeanAcc': np.nanmean(ACC),
        'MeanSen': sen,
        'MeanSpec': spec,            
    }
    
    # 4. Format & Sanitize for JSON (Convert arrays to lists)
    # Assumes format_p_measures is defined, otherwise does basic list conversion
    for key, value in p_measures.items():
        if isinstance(value, np.ndarray):
            p_measures[key] = value.tolist()
            
    # If you have your custom format function:
    # p_measures = format_p_measures(p_measures)

    # 5. Save to JSON
    filename = f"{save_prefix}_measures.json"
    file_path = os.path.join(save_dir, filename)
    
    with open(file_path, 'w') as file:
        json.dump(p_measures, file, indent=4)
    
    print(f"Saved metrics to: {file_path}")
    
    return p_measures, cm

def plot_confusion_matrices(confusionmatrix, p_measures, class_labels, save_dir='.', save_prefix='patient-test'):
    """
    Plots and saves both raw and normalized confusion matrices.
    """
    # Extract metrics for labels
    # Check if they are lists (from JSON formatting) or arrays
    mean_acc = p_measures['MeanAcc']
    mean_sen = p_measures['MeanSen']
    mean_spec = p_measures['MeanSpec']

    # --- Helper to plot one matrix ---
    def _plot_heatmap(cm, fmt, filename_suffix, title_suffix=""):
        fig = plt.figure(figsize=(6, 4), dpi=300)
        # 1. Capture the Axes object
        ax = sns.heatmap(cm, cmap='Reds',
                         annot=True,
                         fmt=fmt, 
                         cbar=True,
                         annot_kws={"size": 14},
                         xticklabels=class_labels,
                         yticklabels=class_labels)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=14)
        
        # 2. Configure the tick parameters
        ax.tick_params(axis='x', labelsize=12) # Set x-axis font size
        ax.tick_params(axis='y', labelsize=12) # Set y-axis font size
                
        plt.ylabel('Actual', fontsize=12)
        plt.xlabel("Predicted\n"
                   "Accuracy={:0.2f}\n"
                   "Sensitivity={:0.2f}\n"
                   "Specificity={:0.2f}\n".format(mean_acc, mean_sen, mean_spec),
                   fontsize=12)
        
        plt.tight_layout()
        plt.show()
        
        filename = os.path.join(save_dir, f"{save_prefix}-{filename_suffix}.png")
        fig.savefig(filename)
        print(f"Saved plot: {filename}")

    # 1. Raw Confusion Matrix
    _plot_heatmap(confusionmatrix,
                  fmt="d",
                  filename_suffix="confuseMatrix")

    # 2. Normalized Confusion Matrix
    cm_normalized = confusionmatrix.astype('float') / confusionmatrix.sum(axis=1)[:, np.newaxis]
    _plot_heatmap(cm_normalized,
                  fmt=".2f",
                  filename_suffix="confuseMatrixNorm")


from sklearn.metrics import roc_curve, auc

def plot_roc_curve(y_true, y_probs, save_dir='.', save_prefix='patient-test'):
    """
    Computes, plots, and saves the ROC curve and calculates the optimal threshold (Youden's Index).
    
    Args:
        y_true (array-like): Ground truth labels (0 or 1).
        y_probs (array-like): Predicted probabilities for the positive class (Class 1).
        save_dir (str): Directory to save the plot.
        save_prefix (str): Prefix for the filename.
        
    Returns:
        roc_auc (float): Area Under the Curve.
        optimal_threshold (float): Threshold maximizing (TPR - FPR).
    """
    # 1. Compute ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    
    # 2. Compute AUC
    roc_auc = auc(fpr, tpr)
    
    # 3. Find Optimal Threshold (Youden’s Index)
    youden_index = tpr - fpr
    optimal_threshold_index = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_threshold_index]
    
    print(f"AUROC: {roc_auc:.4f}")
    print(f"Optimal Threshold (Youden): {optimal_threshold:.4f}")
    
    # 4. Plotting
    fig = plt.figure(figsize=(8, 6), dpi=300, facecolor='w', edgecolor='k')
    
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.scatter(fpr[optimal_threshold_index], tpr[optimal_threshold_index], 
                color='red', label=f'Optimal Threshold = {optimal_threshold:.2f}', zorder=5)
    
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=2)
    
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR / Sensitivity)")
    plt.title("ROC Curve Analysis for THL Detection")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
    
    # 5. Save
    filename = f"{save_prefix}-AUROC.png"
    plot_path = os.path.join(save_dir, filename)
    fig.savefig(plot_path)
    plt.close()
    
    print(f"Saved ROC plot: {plot_path}")
    
    return roc_auc, optimal_threshold

from scipy.interpolate import interp1d
from itertools import cycle

def plot_multiclass_roc(y_true, y_probs, class_labels, save_dir='.', save_prefix='patient_test'):
    """
    Computes and plots ROC curves for a multi-class problem (Micro, Macro, and Per-Class).
    
    Args:
        y_true (np.array): Ground truth labels, One-Hot Encoded (Shape: [n_samples, n_classes]).
        y_probs (np.array): Predicted probabilities (Shape: [n_samples, n_classes]).
        class_labels (list): List of class names (strings).
        save_dir (str): Directory to save the plot.
        save_prefix (str): Prefix for the output filename.
    """
    
    n_classes = len(class_labels)
    
    # Dictionaries to store metrics
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # 1. Compute ROC curve and ROC area for each class
    for i in range(n_classes):
        # Note: We slice [:, i] to get the column for the specific class
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # 2. Compute Micro-average ROC curve and ROC area
    # (treats the entire element-wise prediction as a binary problem)
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # 3. Compute Macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Then interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        interp_func = interp1d(fpr[i], tpr[i], kind='linear', fill_value="extrapolate")
        mean_tpr += interp_func(all_fpr)
    
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # 4. Plotting
    fig = plt.figure(figsize=(8, 6), dpi=300, facecolor='w', edgecolor='k')
    
    # Plot Micro-average
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'micro-average ROC curve (area = {roc_auc["micro"]:0.2f})',
             color='deeppink', linestyle=':', linewidth=4)
    
    # Plot Macro-average
    plt.plot(fpr["macro"], tpr["macro"],
             label=f'macro-average ROC curve (area = {roc_auc["macro"]:0.2f})',
             color='navy', linestyle=':', linewidth=4)
    
    # Plot Each Class
    # Colors: Red (#e6194B) and Green (#3cb44b) as per your request
    colors = cycle(['#e6194B', '#3cb44b', '#4363d8', '#f58231', '#911eb4']) 
    
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {i}: {class_labels[i]} (area = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1 - Specificity', fontsize=14)
    plt.ylabel('Sensitivity', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    
    # 5. Save
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{save_prefix}-ROC-AUROC-multiclass.png"
    plot_path = os.path.join(save_dir, filename)
    fig.savefig(plot_path)
    plt.close()
    
    print(f"Saved Multi-class ROC plot: {plot_path}")

from sklearn.calibration import calibration_curve
def plot_calibration_curve(y_true, y_probs, n_bins=10, save_dir='.', save_prefix='patient-test'):
    """
    Computes, plots, and saves a standard calibration curve.

    Args:
        y_true (array-like): Ground truth labels (0 or 1).
        y_probs (array-like): Predicted probabilities for the positive class.
        n_bins (int): Number of bins to discretize the [0, 1] interval.
        save_dir (str): Directory to save the plot.
        save_prefix (str): Prefix for the output filename.
    """
    print(f"Plotting Calibration Curve for {len(y_true)} samples...")

    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=n_bins, strategy='uniform')

    # Plot
    fig = plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(prob_pred, prob_true, marker='o', linestyle='-', label='Model Calibration')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration (y=x)')
    
    plt.xlabel("Predicted Probability")
    plt.ylabel("Observed Probability")
    plt.title("Calibration Plot")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    # Save
    filename = f"{save_prefix}-calibration-plot.png"
    save_path = os.path.join(save_dir, filename)
    fig.savefig(save_path)
    plt.close()
    
    print(f"Saved calibration plot: {save_path}")

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

def plot_calibration_comparison(y_true, y_probs, n_bins=10, save_dir='.', save_prefix='patient-test'):
    """
    Applies Platt Scaling and Isotonic Regression, then plots comparison curves.

    Args:
        y_true (array-like): Ground truth labels (0 or 1).
        y_probs (array-like): Raw predicted probabilities for the positive class.
        n_bins (int): Number of bins for the calibration curve.
        save_dir (str): Directory to save the plot.
        save_prefix (str): Prefix for the output filename.
    """
    print(f"Comparing Calibration Methods for {len(y_true)} samples...")

    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    
    # Reshape for sklearn models (n_samples, 1)
    y_probs_2d = y_probs.reshape(-1, 1)

    # --- 1. Platt Scaling (Logistic Regression) ---
    platt_scaler = LogisticRegression()
    platt_scaler.fit(y_probs_2d, y_true)
    y_prob_platt = platt_scaler.predict_proba(y_probs_2d)[:, 1]

    # --- 2. Isotonic Regression ---
    iso_scaler = IsotonicRegression(out_of_bounds="clip")
    iso_scaler.fit(y_probs.flatten(), y_true)  # Isotonic expects 1D input for X
    y_prob_iso = iso_scaler.predict(y_probs.flatten())

    # --- 3. Compute Curves ---
    # Original (Baseline)
    prob_true_orig, prob_pred_orig = calibration_curve(y_true, y_probs, n_bins=n_bins, strategy='uniform')
    # Platt
    prob_true_platt, prob_pred_platt = calibration_curve(y_true, y_prob_platt, n_bins=n_bins, strategy='uniform')
    # Isotonic
    prob_true_iso, prob_pred_iso = calibration_curve(y_true, y_prob_iso, n_bins=n_bins, strategy='uniform')

    # --- 4. Plot ---
    fig = plt.figure(figsize=(8, 6), dpi=300)
    
    plt.plot(prob_pred_orig, prob_true_orig, marker='x', linestyle=':', color='black', alpha=0.5, label='Original (Uncalibrated)')
    plt.plot(prob_pred_platt, prob_true_platt, marker='o', linestyle='-', label='Platt Scaling')
    plt.plot(prob_pred_iso, prob_true_iso, marker='s', linestyle='-', label='Isotonic Regression')
    
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration (y=x)')

    plt.xlabel("Predicted Probability")
    plt.ylabel("Observed Probability")
    plt.title("Calibration Comparison: Platt vs. Isotonic")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    # Save
    filename = f"{save_prefix}-calibration-comparison.png"
    save_path = os.path.join(save_dir, filename)
    fig.savefig(save_path)
    plt.close()

    print(f"Saved comparison plot: {save_path}")

def compute_clinical_metrics(y_true, y_pred, pre_test_prob=0.40):
    """
    Computes TP, TN, FP, FN, Sensitivity, Specificity, Likelihood Ratios, 
    and Bayesian Post-Test Probabilities for both IDA (Class 0) and THL (Class 1).

    Args:
        y_true (array-like): Ground truth labels (0 or 1).
        y_pred (array-like): Predicted labels (0 or 1).
        pre_test_prob (float): Pre-test probability for Bayesian calculation.

    Returns:
        dict: A dictionary containing metrics for both classes.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # ---------------------------------------------------------
    # Helper to calculate metrics for a specific positive class
    # ---------------------------------------------------------
    def _calc_metrics(pos_class):
        # Define Positive/Negative based on the target class
        # If pos_class is 0 (IDA), then 0 is Positive, 1 is Negative.
        # If pos_class is 1 (THL), then 1 is Positive, 0 is Negative.
        
        TP = np.sum((y_pred == pos_class) & (y_true == pos_class))
        TN = np.sum((y_pred != pos_class) & (y_true != pos_class))
        FP = np.sum((y_pred == pos_class) & (y_true != pos_class))
        FN = np.sum((y_pred != pos_class) & (y_true == pos_class))
        
        # Sensitivity (TPR) & Specificity (TNR)
        denom_tpr = (TP + FN)
        TPR = TP / denom_tpr if denom_tpr > 0 else 0.0
        
        denom_tnr = (TN + FP)
        TNR = TN / denom_tnr if denom_tnr > 0 else 0.0
        
        # Likelihood Ratios (LR+ and LR-)
        epsilon = 1e-10
        LR_plus = TPR / max(1 - TNR, epsilon)
        LR_minus = (1 - TPR) / max(TNR, epsilon)
        
        # Bayesian Post-Test Probability
        pre_test_odds = pre_test_prob / (1 - pre_test_prob)
        post_test_odds = pre_test_odds * LR_plus
        post_test_prob = post_test_odds / (1 + post_test_odds)
        
        return {
            "TP": int(TP), "TN": int(TN), "FP": int(FP), "FN": int(FN),
            "Sensitivity": round(TPR, 4), 
            "Specificity": round(TNR, 4),
            "LR+": round(LR_plus, 4), 
            "LR-": round(LR_minus, 4), 
            "Post-test Probability": round(post_test_prob, 4)
        }

    # ---------------------------------------------------------
    # Compute for both classes
    # ---------------------------------------------------------
    results = {
        "IDA": _calc_metrics(pos_class=0),
        "THL": _calc_metrics(pos_class=1)
    }
    
    # Print formatted output
    for cls_name, metrics in results.items():
        print(f"\n--- {cls_name} ---")
        for k, v in metrics.items():
            print(f"{k}: {v}")
            
    return results

from sklearn.metrics import brier_score_loss, r2_score
def compute_calibration_scores(y_true, y_probs):
    """
    Computes Brier Score and R^2 Score to evaluate probability calibration.

    Args:
        y_true (array-like): Ground truth labels (0 or 1).
        y_probs (array-like): Predicted probabilities for the positive class (Class 1).

    Returns:
        dict: A dictionary containing 'Brier Score' and 'R2 Score'.
    """
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    
    print(f"Calculating scores for {len(y_true)} samples...")
    
    # Compute Brier Score (Lower is better)
    # Mean squared difference between predicted probability and actual outcome
    brier = brier_score_loss(y_true, y_probs)
    
    # Compute R^2 Score (Higher is better)
    # Proportion of variance in the dependent variable predictable from the independent variable
    r2 = r2_score(y_true, y_probs)
    
    print(f"Brier Score: {brier:.4f}")
    print(f"R^2 Score:   {r2:.4f}")
    
    return {
        "Brier Score": brier,
        "R2 Score": r2
    }


from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, brier_score_loss

def create_master_summary(performance_cases, save_dir='.', filename='master_performance_summary.csv'):
    """
    Iterates through performance_cases and compiles key metrics into a DataFrame.
    """
    summary_rows = []
    
    print("\n" + "="*40)
    print("COMPILING MASTER SUMMARY")
    print("="*40)

    for case_name, data in performance_cases.items():
        y_true = data['y_true']
        y_pred = data['y_pred']
        y_prob = data.get('y_prob', []) # Get probs if they exist
        
        # Skip if empty
        if len(y_true) == 0:
            continue
            
        # --- 1. Basic Metrics ---
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        
        # Calculate Sensitivity/Specificity safely (handling cases with missing classes)
        # Labels: 0=IDA, 1=THL
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1]) 
        tn, fp, fn, tp = cm.ravel()
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # --- 2. Probability Metrics (AUC, Brier) ---
        auc_score = np.nan
        brier_score = np.nan
        
        if len(y_prob) > 0:
            # AUC requires at least one example of each class
            if len(np.unique(y_true)) > 1:
                auc_score = roc_auc_score(y_true, y_prob)
            
            brier_score = brier_score_loss(y_true, y_prob)
            
        # --- 3. Build Row ---
        row = {
            "Case Name": case_name,
            "N_Samples": len(y_true),
            "Accuracy": round(acc, 4),
            "F1_Macro": round(f1, 4),
            "Sensitivity": round(sens, 4),
            "Specificity": round(spec, 4),
            "AUC": round(auc_score, 4) if not np.isnan(auc_score) else "N/A",
            "Brier_Score": round(brier_score, 4) if not np.isnan(brier_score) else "N/A"
        }
        summary_rows.append(row)

    # --- 4. Create DataFrame & Save ---
    summary_df = pd.DataFrame(summary_rows)
    
    # Reorder columns for readability
    cols = ["Case Name", "N_Samples", "Accuracy", "F1_Macro", "Sensitivity", "Specificity", "AUC", "Brier_Score"]
    summary_df = summary_df[cols]
    
    # Save to CSV
    save_path = os.path.join(save_dir, filename)
    summary_df.to_csv(save_path, index=False)
    
    print("\nMaster Summary Table:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved master summary to: {save_path}")
    
    return summary_df

def format_p_measures(p_measures):
    for key, value in p_measures.items():
        if key in ['TP', 'TN', 'FP', 'FN']:  # Keep these values as integers
            if isinstance(value, list):  # If value is a list, convert each item to int
                p_measures[key] = [int(v) for v in value]
            else:
                p_measures[key] = int(value)
        else:  # For other metrics, format to two decimal points
            if isinstance(value, list):  # If it's a list, round each item to 2 decimal places
                p_measures[key] = [round(v, 3) for v in value]
            elif isinstance(value, (float, int)):  # If it's a scalar, round it to 2 decimals
                p_measures[key] = round(value, 3)
    return p_measures

def hybrid_classification_rule(row, T_upper, T_lower, T_opt):
    """
    Classifies a sample based on the Mean-Based Tri-State Logic.
    
    Logic:
    1. Confident IDA: P_mean < T_lower
    2. Confident THL: P_mean >= T_upper
    3. Uncertain:     T_lower <= P_mean < T_upper (Use T_opt to force binary label)
    """
    mean_based_prob = row["probability_1"]
    
    if mean_based_prob < T_lower:
        # Confident IDA
        return 0, "Confident", "IDA"
        
    elif mean_based_prob >= T_upper:
        # Confident Thalassemia
        return 1, "Confident", "THL"
        
    else:
        # Uncertain Region -> Use T_opt for binary forcing
        final_pred = 1 if mean_based_prob >= T_opt else 0
        label = "THL" if final_pred == 1 else "IDA"
        return final_pred, "Uncertain", label
    
#%% Phase 3: Trains an ensemble of Bag-Level MIL models for robust aggregation.
# Define the Loading Logic (Function for reusability)
import joblib
def load_one_extractor(fold_idx, model_name, model_path):
    fe_filename = f"fold_{fold_idx}_{model_name}_feature_extractor.h5"
    fe_path = os.path.join(model_path, fe_filename)
    print(f"{fe_path=}")
    if os.path.exists(fe_path):
        # Assuming LayerScale is needed for ConvNeXt
        with custom_object_scope({'LayerScale': LayerScale}):
            model = load_model(fe_path)
        print(f"  [Loaded] Fold {fold_idx}: {fe_filename}")
        return model
    else:
        print(f"  [Error] File not found: {fe_path}")


def load_one_manifold(fold_idx, model_name, manifold_path, val=False):
    """
    Loads a specific fold's manifold tool.
    Args:
        val (bool): If True, loads the Validation manifold. If False, loads Training manifold.
    """
    # 1. Determine suffix based on val flag
    # Based on your logs: 
    # Train -> "fold_X_Model_manifold.pkl"
    # Val   -> "fold_X_Model_val_manifold.pkl"
    suffix = "_val_manifold.pkl" if val else "_manifold.pkl"
    dataset_type = "Validation" if val else "Training"
    
    filename = f"fold_{fold_idx}_{model_name}{suffix}"
    
    # 2. Construct primary path
    manifold_file = os.path.join(manifold_path, filename)
    print(f'{manifold_file=}')
    # 3. Try Loading
    if os.path.exists(manifold_file):
        tool = joblib.load(manifold_file)
        print(f"  [Loaded] {dataset_type} Manifold Tool for Fold {fold_idx}")
        return tool
               
    # 5. Critical Failure
    raise FileNotFoundError(
        f"CRITICAL: {dataset_type} Manifold tool not found.\n"
        f"   Checked: {manifold_file}\n"
    )

from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.models import load_model
import os

# Ensure these are imported from your utils file
from riemannian_utils import PoincareMath, HyperbolicDense, FrechetMean, CurvatureAttention
def load_mil_ensemble_for_fold(fold_idx, model_name, model_path, n_estimators=5):
    fold_models = []
    
    # Define dictionary of ALL custom layers used in the model
    # The keys must match the class names exactly
    custom_layers = {
        'CurvatureAttention': CurvatureAttention,
        'HyperbolicDense': HyperbolicDense,
        'FrechetMean': FrechetMean
    }
    
    # Open the scope with ALL custom layers
    with custom_object_scope(custom_layers):    
        for i in range(n_estimators):
            # Matches save format: "fold_0_mil_estimator_0.h5"
            filename = f"fold_{fold_idx}_{model_name}_mil_estimator_{i}.h5"
            filepath = os.path.join(model_path, filename)
            
            model = None
            
            if os.path.exists(filepath):
                print(f"  Loading: {filepath}")
                model = load_model(filepath)
            else:
                # Try SAVE_PATH fallback
                alt_path = os.path.join(config.SAVE_PATH, filename)
                if os.path.exists(alt_path):
                     print(f"  Loading (fallback): {alt_path}")
                     model = load_model(alt_path)
                else:
                     print(f"  [Warning] Missing estimator: {filepath}")
            
            if model is not None:
                fold_models.append(model)
    
    if not fold_models:
        raise FileNotFoundError(f"CRITICAL: No models loaded for Fold {fold_idx}!")
        
    print(f"  [Loaded] Fold {fold_idx+1}: {len(fold_models)} estimators")
    return fold_models

    
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
        print(f"   [Safety Filter] Limit: {limit:.4f} | Extreme Ceiling: {extreme_limit:.4f}")
        print(f"   [Saliency Gate] Rescued {n_rescued} high-energy diagnostic cells.")
        print(f"   [Safety Filter] Final Passed: {n_valid} | Blocked: {n_rejected} | Rejected Rate: {reject_rate:.4f}")

    stats = {
        "n_total": n_total,
        "n_valid": n_valid,
        "n_rejected": n_rejected,
        "n_rescued": n_rescued,
        "reject_rate": reject_rate
    }
    
    return clean_features, clean_energies, valid_mask, stats

def apply_clinical_triage(p_patient_mean, p_patient_argmax, t_lower, t_upper, t_opt, fallback_mode='mean'):
    """
    Applies a Tri-state (Hybrid) decision logic to categorize patients.
    
    Args:
        p_patient_mean (float): probability_1_mean from your dataframe.
        p_patient_argmax (float): probability_1_argmax from your dataframe.
        t_lower (float): Lower threshold for IDA confidence.
        t_upper (float): Upper threshold for THL confidence.
        t_opt (float): Optimal binary threshold (for mean-based fallback).
        fallback_mode (str): 'mean' to use t_opt, or 'argmax' to use majority patch vote.
        
    Returns:
        dict: Triage results mapping to your dataframe structure.
    """
    # T_LOWER = 0.5130  #(Confident IDA < this)
    # T_UPPER = 0.5729  #(Confident THL >= this)
    # T_OPT = 0.5538    #(Balanced Cutoff)
    # 1. Confident IDA Zone
    if p_patient_mean < t_lower:
        pred_status = "Confident"
        final_pred = 0  # IDA
        diagnosis_label = "IDA"
        
    # 2. Confident THL Zone
    elif p_patient_mean >= t_upper:
        pred_status = "Confident"
        final_pred = 1  # THL
        diagnosis_label = "THL"
        
    # 3. Grey Zone (Uncertain)
    else:
        pred_status = "Uncertain"
        
        if fallback_mode == 'mean':
            # Case A: Fallback based on optimal probability threshold
            final_pred = 1 if p_patient_mean >= t_opt else 0
        else:
            # Case B: Fallback based on majority patch voting (argmax > 0.5)
            # Since p_patient_argmax is a ratio (0.0 to 1.0), 0.5 is the vote majority
            final_pred = 1 if p_patient_argmax >= 0.5 else 0
            
        diagnosis_label = "THL" if final_pred == 1 else "IDA"

    return {
        "pred_status": pred_status,
        "final_pred": final_pred,
        "diagnosis_label": diagnosis_label,
        "probability_used": p_patient_mean if fallback_mode == 'mean' else p_patient_argmax
    }

def apply_clinical_binary(p_patient_mean, t_upper=0.6088):
    """
    Applies a binary decision logic with a high-confidence ceiling.
    Strictly categorizes patients as Confident THL or IDA.
    
    Args:
        p_patient_mean (float): probability_1_mean from ConvNeXtLarge ensemble.
        t_upper (float): The threshold for THL (e.g., 0.6088).
        
    Returns:
        dict: Binary results for clinical logging.
    """
    
    if p_patient_mean >= t_upper:
        pred_status = "Confident"
        final_pred = 1  # THL
        diagnosis_label = "THL"
    else:
        # All other cases, including those below T_UPPER, are defaulted to IDA
        pred_status = "Binary_Default"
        final_pred = 0  # IDA
        diagnosis_label = "IDA"

    return {
        "pred_status": pred_status,
        "final_pred": final_pred,
        "diagnosis_label": diagnosis_label,
        "probability": p_patient_mean
    }

# Function for Patch Visualization 
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt

def plot_patient_audit_grids(patch_df, save_dir="Patient_Audit_Plots", max_patches=32, config=None):
    """
    Generates an 8x4 grid plot for each status type per patient.
    Auto-detects 'rgba' vs 'rgb' reading mode based on file extension.
    
    Args:
        patch_df (pd.DataFrame): The dataframe from patch audit CSV.
        save_dir (str): Directory to save the generated plots.
        max_patches (int): Max patches to plot (default 32 for 8x4 grid).
        config: Configuration object containing DATASET path (optional).
    """
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Get unique patients
    patients = patch_df['patient_id'].unique()
    
    print(f"[Audit Plotter] Found {len(patients)} patients. Generating grids...")
    
    # Dataset Root (Handle config being passed or not)
    dataset_root = config.DATASET if config else ""

    for pid in patients:
        # Filter for this patient
        pat_df = patch_df[patch_df['patient_id'] == pid]
        true_label = pat_df['label'].iloc[0] if 'label' in pat_df.columns else "Unknown"
        
        # Get unique statuses (e.g., 'Accepted (High Energy)', 'REJECTED (OOD)')
        statuses = pat_df['status'].unique()
        
        for status in statuses:
            # Filter for this status
            status_df = pat_df[pat_df['status'] == status]
            n_total = len(status_df)
            
            # Sampling Logic
            if n_total > max_patches:
                plot_df = status_df.sample(n=max_patches, random_state=42)
                note = f"(Random 32 of {n_total})"
            else:
                plot_df = status_df
                note = f"(All {n_total})"
            
            # Setup Plot
            cols = 8
            rows = 4
            fig, axes = plt.subplots(rows, cols, figsize=(20, 10), dpi=150)
            fig.suptitle(f"Patient: {pid} (Label: {true_label}) | Status: {status} {note}", fontsize=16)
            
            axes_flat = axes.flatten()
            
            # Plot Images
            for idx, (i, row) in enumerate(plot_df.iterrows()):
                img_path = row['patch_path']
                energy = row['curvature_energy']
                
                ax = axes_flat[idx]
                
                # Construct Full Path
                img_path_full = os.path.join(dataset_root, img_path)
                
                # --- AUTO-DETECT COLOR MODE ---
                # Logic: If .png, assume RGBA (Transparent). If .jpg/.jpeg, assume RGB.
                ext = os.path.splitext(img_path_full)[1].lower()
                if ext == '.png':
                    read_mode = cv2.IMREAD_UNCHANGED # Preserves Alpha Channel
                    cvt_mode = cv2.COLOR_BGRA2RGBA   # Convert for Matplotlib
                else:
                    read_mode = cv2.IMREAD_COLOR     # Standard 3-channel
                    cvt_mode = cv2.COLOR_BGR2RGB
                # ------------------------------

                # Load Image
                if os.path.exists(img_path_full):
                    img = cv2.imread(img_path_full, read_mode)
                    
                    if img is not None:
                        # Convert color space for matplotlib
                        try:
                            # Handle case where file is .png but has no alpha
                            if read_mode == cv2.IMREAD_UNCHANGED and img.shape[2] == 3:
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            elif read_mode == cv2.IMREAD_UNCHANGED and img.shape[2] == 4:
                                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
                            else:
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                
                            ax.imshow(img)
                            
                        except Exception as e:
                             ax.text(0.5, 0.5, "Format Err", ha='center', va='center')
                    else:
                        ax.text(0.5, 0.5, "Corrupt", ha='center', va='center')
                else:
                    # Provide a visual hint if path is wrong (common issue)
                    ax.text(0.5, 0.5, "Not Found", ha='center', va='center', fontsize=8)
                    # print(f"File Not Found: {img_path_full}") # Optional: debug log

                # Add Energy Score as Label
                # Red for rejected/high energy, Green/Blue for accepted/low
                if 'REJECTED' in status:
                    text_color = 'red' 
                elif 'High Energy' in status:
                     text_color = 'purple' # Differentiate high energy accepted
                else:
                    text_color = 'green'

                ax.set_title(f"E={energy:.1f}", color=text_color, fontsize=10, fontweight='bold')
                # ax.axis('off')
                # --- NEW BORDER LOGIC ---
                # Instead of ax.axis('off'), we remove ticks but keep the box (spines)
                ax.set_xticks([])
                ax.set_yticks([])                  
                border_color = 'lightgray'                  
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color(border_color)
                    spine.set_linewidth(1.5) # Make it slightly thicker
                               
            # Hide unused subplots
            for j in range(len(plot_df), len(axes_flat)):
                axes_flat[j].axis('off')
            
            # Save Plot
            safe_status = status.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "-")
            filename = f"{pid}_{safe_status}.png"
            save_path = os.path.join(save_dir, filename)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
            plt.savefig(save_path)
            plt.close(fig)
            
    print(f"[Audit Plotter] Done! Check folder: {save_dir}")

# --- Usage Example ---
# import pandas as pd
# # audit_path = ''
# df = pd.read_csv(audit_path)
# plot_patient_audit_grids(df, config=config)

from statsmodels.stats.proportion import proportion_confint

def calculate_clinical_wilson_intervals(json_path, class_names=['IDA', 'THL']):
    """
    Computes 95% Wilson Score Intervals for key clinical metrics.
    Suitable for Q1 journal tables.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    metrics = ['ACC', 'TPR', 'TNR', 'PPV', 'NPV', 'FPR', 'FNR', 'FDR']
    results = []

    for i, class_name in enumerate(class_names):
        # Extract counts
        tp, tn, fp, fn = data['TP'][i], data['TN'][i], data['FP'][i], data['FN'][i]
        
        # Map metrics to (count, total)
        stats_map = {
            'ACC': (tp + tn, tp + tn + fp + fn),
            'TPR': (tp, tp + fn),
            'TNR': (tn, tn + fp),
            'PPV': (tp, tp + fp),
            'NPV': (tn, tn + fn),
            'FPR': (fp, fp + tn),
            'FNR': (fn, fn + tp),
            'FDR': (fp, tp + fp)
        }
        
        for m in metrics:
            count, n = stats_map[m]
            val = count / n if n > 0 else 0
            # Calculate Wilson Interval
            low, high = proportion_confint(count, n, alpha=0.05, method='wilson')
            
            results.append({
                'Class': class_name,
                'Metric': m,
                'Estimate': f"{val:.2f}",
                '95% CI': f"[{low:.3f} - {high:.3f}]",
                'Raw (n/N)': f"{count}/{n}"
            })
            
    return pd.DataFrame(results)

# Usage
# df_intervals = calculate_clinical_wilson_intervals('Test_Ensemble_Confident_measures.json')

#%%

"""
from TDA_Engine import TopologicalFeatureExtractor
def generate_offline_features(df, config, split_name="data"):
    # Extracts TDA features and saves the resulting DataFrame as a .pkl file.
    
    # 1. Setup TDA Engine
    # Ensure these params match your specific TDA requirements
    tda_engine = TopologicalFeatureExtractor(pi_resolution=32, bandwidth=1.0)
    
    if df.empty:
        print(f"⚠️ Warning: Input DataFrame for {split_name} is empty.")
        return df

    print(f"\n--- Processing {split_name} set ({len(df)} patches) ---")
    
    # 2. Extract Features
    tda_feat_list = []
    
    # Using tqdm for progress bar
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_path = row['patch_path'] # Ensure this column name matches your DF
        
        try:
            # Read Image
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None: 
                raise ValueError("Image not found or unreadable")
            
            # Handle RGBA -> RGB (Black Background)
            if img.shape[2] == 4:
                alpha = img[:, :, 3]
                rgb = img[:, :, :3]
                img_comp = rgb.copy()
                img_comp[alpha == 0] = 0
            else:
                img_comp = img
            
            # Ensure 256x256
            if img_comp.shape[0] != 256:
                img_comp = cv2.resize(img_comp, (256, 256))

            # Run TDA
            tda_feature = tda_engine.process_image(img_comp)
            
            # Append successful feature
            tda_feat_list.append(tda_feature)
            
        except Exception as e:
            # print(f"Error {img_path}: {e}") # Optional: Uncomment to see specific errors
            # Append None so the list length stays synced with DataFrame index
            tda_feat_list.append(None)

    # 3. Assign to DataFrame
    df['tda_feature'] = tda_feat_list
    
    # 4. Remove Failures
    # We drop rows where tda_feature is None (failed images)
    initial_count = len(df)
    df = df.dropna(subset=['tda_feature'])
    final_count = len(df)
    
    if initial_count != final_count:
        print(f"Dropped {initial_count - final_count} images due to processing errors.")

    # 5. Save to Pickle (The requested revision)
    save_filename = f"{split_name}_tda_features.pkl"
    save_path = os.path.join(config.BASEPATH, save_filename)
    
    print(f"Saving {split_name} data to: {save_path}")
    # Protocol 4 is stable and efficient for large binary data (numpy arrays)
    df.to_pickle(save_path, protocol=4)
    
    return df

"""
















