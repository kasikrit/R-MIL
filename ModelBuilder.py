import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import (VGG16, VGG19, EfficientNetB7, Xception, 
                                           InceptionResNetV2, InceptionV3, DenseNet121, 
                                           DenseNet201, MobileNet, MobileNetV2)
from tensorflow.keras import layers, models, applications
from tensorflow.keras.layers import (Dense, Dropout, BatchNormalization, 
                                     Flatten, Input, Conv2D, MaxPooling2D)
from tensorflow.keras.regularizers import l2
# from vit_keras import vit, utils, visualize
import tensorflow_addons as tfa
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import os
import cv2

class ModelBuilder:
    def __init__(self, model_name,
                config,
                # add_custom_layers=True,
                base_model_trainable=False):
        self.model_name = model_name
        self.config = config
        self.base_model = None
        self.model = None
        self.preprocessing_function = None
        self.base_model_trainable = base_model_trainable
    
    def create_model(self):
        vit_models = ['ViT_b16', 'ViT_l16', 'ViT_b32', 'ViT_l32']
        # convnext_models = ['ConvNeXtTiny', 'ConvNeXtSmall', 'ConvNeXtBase', 'ConvNeXtLarge', 'ConvNeXtXLarge']
        
        if self.model_name in vit_models:
            self.model = self._create_vit_model()
        # elif self.model_name in convnext_models:
        #     self.model = self._load_convnext_model()
        else:
            self.model = self._create_cnn_model()
        
        # print(self.model.summary())
        print(f'\nCreated: {self.model.name}')
        print(f'Default {self.base_model.trainable=}')
        
        return self.model
    
    def _create_cnn_model(self):
        eff_net_list = [
            'EfficientNetV2B3',
            'EfficientNetV2S',
            'EfficientNetV2M',
            'EfficientNetV2L',
        ]
        
        convNeX_list = [
        'ConvNeXtTiny',
        'ConvNeXtSmall',
        'ConvNeXtBase',
        'ConvNeXtLarge',
        'ConvNeXtXLarge',
        ] 
        print(f"{self.model_name=}")
        
        if self.model_name == 'AlexNet':
            base_model = self._create_alexnet_model()
            self.base_model = base_model
            return base_model
        else:
            if self.model_name in eff_net_list:
                base_model = self._load_efficientnet_model()
            elif self.model_name in convNeX_list: 
                base_model = self._load_convnext_model()
                print(type(base_model))
            elif self.model_name == 'VGG16':
                base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=self.config['DATA']['DIMENSION'])
            elif self.model_name == 'VGG19':
                base_model = tf.keras.applications.VGG19(weights='imagenet', include_top=False, input_shape=self.config['DATA']['DIMENSION'])
            elif self.model_name == 'EfficientNetB7':
                base_model = tf.keras.applications.EfficientNetB7(weights='imagenet', include_top=False, input_shape=self.config['DATA']['DIMENSION'])
            elif self.model_name == 'Xception':
                base_model = tf.keras.applications.Xception(weights='imagenet', include_top=False, input_shape=self.config['DATA']['DIMENSION'])
            elif self.model_name == 'InceptionResNetV2':
                base_model = tf.keras.applications.InceptionResNetV2(weights='imagenet', include_top=False, input_shape=self.config['DATA']['DIMENSION'])
            elif self.model_name == 'InceptionV3':
                base_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=self.config['DATA']['DIMENSION'])
            elif self.model_name == 'DenseNet121':
                base_model = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, input_shape=self.config['DATA']['DIMENSION'])
            elif self.model_name == 'DenseNet201':
                base_model = tf.keras.applications.DenseNet201(weights='imagenet', include_top=False, input_shape=self.config['DATA']['DIMENSION'])
            elif self.model_name == 'MobileNet':
                base_model = tf.keras.applications.MobileNet(weights='imagenet', include_top=False, input_shape=self.config['DATA']['DIMENSION'])
            elif self.model_name == 'MobileNetV2':
                base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=self.config['DATA']['DIMENSION'])
            elif self.model_name == 'ResNet50':
                base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=self.config['DATA']['DIMENSION'])
            elif self.model_name == 'ResNet50V2':
                base_model = tf.keras.applications.ResNet50V2(weights='imagenet', include_top=False, input_shape=self.config['DATA']['DIMENSION'])
            else:
                raise ValueError(f"Model {self.model_name} not supported")
            
            self.base_model = base_model
            return self._add_custom_layers(base_model)
    
    def _create_alexnet_model(self):
        input_shape = self.config['DATA']['DIMENSION']
        
        L2_decay = self.config['TRAIN']['LR'] / 10.0
        activation_func = self.config.get('ACTIVATION', 'relu')
    
        # Define the AlexNet architecture
        inputs = Input(shape=input_shape)
    
        # Layer 1: Convolution + MaxPooling
        x = Conv2D(96, (11, 11), strides=4, activation=activation_func, padding='valid', kernel_regularizer=l2(L2_decay))(inputs)
        x = MaxPooling2D((3, 3), strides=2)(x)
        x = BatchNormalization()(x)
    
        # Layer 2: Convolution + MaxPooling
        x = Conv2D(256, (5, 5), activation=activation_func, padding='same', kernel_regularizer=l2(L2_decay))(x)
        x = MaxPooling2D((3, 3), strides=2)(x)
        x = BatchNormalization()(x)
    
        # Layer 3, 4, 5: Three Convolution Layers
        x = Conv2D(384, (3, 3), activation=activation_func, padding='same', kernel_regularizer=l2(L2_decay))(x)
        x = Conv2D(384, (3, 3), activation=activation_func, padding='same', kernel_regularizer=l2(L2_decay))(x)
        x = Conv2D(256, (3, 3), activation=activation_func, padding='same', kernel_regularizer=l2(L2_decay))(x)
        x = MaxPooling2D((3, 3), strides=2)(x)
        x = BatchNormalization()(x)
    
        # Flatten the network
        x = Flatten()(x)
    
        # Fully Connected Layer 1
        x = Dense(4096, activation=activation_func, kernel_regularizer=l2(L2_decay))(x)
        x = Dropout(self.config['TRAIN']['DROPOUT'])(x)
    
        # Fully Connected Layer 2
        x = Dense(4096, activation=activation_func, kernel_regularizer=l2(L2_decay))(x)
        x = Dropout(self.config['TRAIN']['DROPOUT'])(x)
    
        # Fully Connected Layer 3 (Output Layer)
        outputs = Dense(self.config['MODEL']['NUM_CLASSES'], activation='softmax')(x)
    
        model = Model(inputs=inputs, outputs=outputs, name='AlexNet')
    
        print(f"Created AlexNet Model: {model.name}")
        return model

    def _create_vit_model(self):
        activation_func = tfa.activations.gelu
    
        if self.model_name == 'ViT_b16':
            base_model = vit.vit_b16(
                image_size=self.config['DATA']['IMAGE_SIZE'],
                activation=activation_func,  # GELU activation
                pretrained=True,
                include_top=False,
                pretrained_top=False,
                classes=self.config['MODEL']['NUM_CLASSES']
            )
        elif self.model_name == 'ViT_l16':
            base_model = vit.vit_l16(
                image_size=self.config['DATA']['IMAGE_SIZE'],
                activation=activation_func,  # GELU activation
                pretrained=True,
                include_top=False,
                pretrained_top=False,
                classes=self.config['MODEL']['NUM_CLASSES']
            )
        elif self.model_name == 'ViT_b32':
            base_model = vit.vit_b32(
                image_size=self.config['DATA']['IMAGE_SIZE'],
                activation=activation_func,  # GELU activation
                pretrained=True,
                include_top=False,
                pretrained_top=False,
                classes=self.config['MODEL']['NUM_CLASSES']
            )
        elif self.model_name == 'ViT_l32':
            base_model = vit.vit_l32(
                image_size=self.config['DATA']['IMAGE_SIZE'],
                activation=activation_func,  # GELU activation
                pretrained=True,
                include_top=False,
                pretrained_top=False,
                classes=self.config['MODEL']['NUM_CLASSES']
            )
    
        self.base_model = base_model
        return self._add_custom_layers(base_model)

    def _add_custom_layers(self, base_model):
        activation_func = self.config.get('ACTIVATION', 'relu')  # for ViT family models
    
        # custom_layers = Flatten()(base_model.output) # train-cell3-stage2-20250406.txt
        custom_layers = layers.GlobalAveragePooling2D()(base_model.output)
        custom_layers = BatchNormalization()(custom_layers)
        
        lr = self.config['TRAIN']['LR']
        l2_reg = self.config['TRAIN'].get('l2_reg', None)
        if l2_reg is None:
            L2_decay = lr / 10
        else:
            L2_decay = lr / l2_reg
            print(f"\nL2_decay uses {l2_reg=}")
       
        custom_layers = Dense(1024, activation=activation_func,
                        kernel_regularizer=l2(L2_decay)
                        )(custom_layers)
        custom_layers = Dropout(self.config['TRAIN']['DROPOUT'])(custom_layers)
        custom_layers = BatchNormalization()(custom_layers)
    
        custom_layers = Dense(512, activation=activation_func,
                               # kernel_regularizer=l2
                              )(custom_layers)
        custom_layers = Dropout(self.config['TRAIN']['DROPOUT'])(custom_layers)
        custom_layers = BatchNormalization()(custom_layers)
    
        custom_layers = Dense(256, activation=activation_func,
                               # kernel_regularizer=l2
                              # kernel_regularizer=l2(L2_decay)
                              )(custom_layers)
        custom_layers = Dropout(self.config['TRAIN']['DROPOUT'])(custom_layers)
        custom_layers = BatchNormalization()(custom_layers)
    
        custom_layers = Dense(64, activation=activation_func,
                               # kernel_regularizer=l2
                              )(custom_layers)
        custom_layers = Dropout(self.config['TRAIN']['DROPOUT'])(custom_layers)
        custom_layers = BatchNormalization()(custom_layers)
    
        custom_layers = Dense(32, activation=activation_func,
                               # kernel_regularizer=l2
                              )(custom_layers)
        custom_layers = Dropout(self.config['TRAIN']['DROPOUT'])(custom_layers)
        custom_layers = BatchNormalization()(custom_layers)
    
        custom_layers = Dense(16, activation=activation_func,
                               # kernel_regularizer=l2
                              )(custom_layers)
        custom_layers = Dropout(self.config['TRAIN']['DROPOUT'])(custom_layers)
    
        custom_layers = Dense(self.config['MODEL']['NUM_CLASSES'], activation='softmax')(custom_layers)
        
        model = Model(base_model.input, custom_layers, name=self.model_name)
        base_model.trainable = False
        return model

    def _load_convnext_model(self):
        # Mapping ConvNeXt model names to the respective TensorFlow Keras classes
        convnext_mapping = {
            'ConvNeXtTiny': tf.keras.applications.ConvNeXtTiny,
            'ConvNeXtSmall': tf.keras.applications.ConvNeXtSmall,
            'ConvNeXtBase': tf.keras.applications.ConvNeXtBase,
            'ConvNeXtLarge': tf.keras.applications.ConvNeXtLarge,
            'ConvNeXtXLarge': tf.keras.applications.ConvNeXtXLarge,
        }

        if self.model_name in convnext_mapping:
            ConvNeXtClass = convnext_mapping[self.model_name]
            return ConvNeXtClass(
                weights='imagenet',
                include_top=False,
                input_shape=self.config['DATA']['DIMENSION']
            )
        else:
            raise ValueError(f"ConvNeXt model {self.model_name} not supported")

    def _load_efficientnet_model(self):
        efficientnet_mapping = {
            'EfficientNetV2B3': tf.keras.applications.EfficientNetV2B3,
            'EfficientNetV2S': tf.keras.applications.EfficientNetV2S,
            'EfficientNetV2M': tf.keras.applications.EfficientNetV2M,
            'EfficientNetV2L': tf.keras.applications.EfficientNetV2L
        }
        
        if self.model_name in efficientnet_mapping:
            EfficientNetClass = efficientnet_mapping[self.model_name]
            return EfficientNetClass(
                weights='imagenet', 
                include_top=False, 
                input_shape=self.config['DATA']['DIMENSION']
            )
        else:
            raise ValueError(f"EfficientNet model {self.model_name} not supported")

    def get_preprocessing_function(self):
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
            'ResNet50': tf.keras.applications.resnet50.preprocess_input,
            'ResNet50V2': tf.keras.applications.resnet_v2.preprocess_input,
            
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
    
        if self.model_name in preprocessing_mapping:
            return preprocessing_mapping[self.model_name]
        elif self.model_name.startswith('ViT'):
            return self._preprocess_vit
        elif self.model_name == 'AlexNet':
            return self._preprocess_for_alexnet
        else:
            raise ValueError(f"No preprocessing function for model {self.model_name}")

    def _preprocess_vit(self, img):
        try:
            if isinstance(img, Image.Image):
                # img.verify()
                img = np.asarray(img)   
            # Apply ViT-specific preprocessing
            return vit.preprocess_inputs(img)
        except (IOError, UnidentifiedImageError, ValueError) as e:
            # Catch errors related to opening and reading the image
            print(f"Skipping invalid or corrupted image. Error: {e}")
            return None  # Return None for invalid images
    
    def _preprocess_for_alexnet(self, image):
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # image_resized = cv2.resize(image_bgr, (227, 227))
        image_resized = image_bgr.astype(np.float32)

        # 4. Subtract ImageNet mean values for each channel (RGB)
        mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
        image_normalized = image_resized - mean

        return image_normalized

class MetricsTracker_backup(tf.keras.callbacks.Callback):
    def __init__(self, csv_file_path):
        super(MetricsTracker_backup, self).__init__()
        self.csv_file_path = csv_file_path
        self.history = []

        # Write CSV header if not exists
        if not os.path.exists(self.csv_file_path):
            with open(self.csv_file_path, 'w') as f:
                f.write('epoch,loss,val_loss,accuracy,val_accuracy,lr_stage2,lr_stage3,lr_head\n')

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # Default fallbacks
        lr_stage2, lr_stage3, lr_head = float('nan'), float('nan'), float('nan')

        opt = self.model.optimizer

        if hasattr(opt, 'optimizers_and_layers'):  # MultiOptimizer detected
            for sub_opt, layer_group in opt.optimizers_and_layers:
                try:
                    lr = sub_opt.learning_rate
                    if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
                        lr_val = tf.keras.backend.get_value(lr(self.model.optimizer.iterations))
                    else:
                        lr_val = tf.keras.backend.get_value(lr)
                except Exception:
                    lr_val = float('nan')

                # Assign by heuristics: map layer names or order
                if 'stage2' in layer_group[0].name.lower():
                    lr_stage2 = lr_val
                elif 'stage3' in layer_group[0].name.lower():
                    lr_stage3 = lr_val
                else:
                    lr_head = lr_val
        else:
            # Standard optimizer fallback
            try:
                lr = opt.learning_rate
                if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
                    lr_val = tf.keras.backend.get_value(lr(opt.iterations))
                else:
                    lr_val = tf.keras.backend.get_value(lr)
                lr_head = lr_val
            except Exception:
                lr_head = float('nan')

        # Common metrics
        epoch_metrics = {
            'epoch': epoch + 1,
            'loss': logs.get('loss'),
            'val_loss': logs.get('val_loss'),
            'accuracy': logs.get('accuracy'),
            'val_accuracy': logs.get('val_accuracy'),
        }

        # Handle learning rate logging
        if isinstance(self.model.optimizer, tf.keras.optimizers.Optimizer):
            # Single optimizer case
            try:
                lr = self.model.optimizer.learning_rate
                if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
                    lr_val = tf.keras.backend.get_value(lr(self.model.optimizer.iterations))
                else:
                    lr_val = tf.keras.backend.get_value(lr)
                epoch_metrics['lr'] = lr_val
            except Exception:
                epoch_metrics['lr'] = float('nan')

        elif hasattr(self.model.optimizer, 'optimizers_and_layers'):
            # MultiOptimizer case
            lr_map = {}
            for sub_opt, layer_group in self.model.optimizer.optimizers_and_layers:
                try:
                    lr = sub_opt.learning_rate
                    if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
                        lr_val = tf.keras.backend.get_value(lr(self.model.optimizer.iterations))
                    else:
                        lr_val = tf.keras.backend.get_value(lr)
                except Exception:
                    lr_val = float('nan')

                # Heuristic or naming for LR keys
                group_name = layer_group[0].name.lower()
                if 'stage2' in group_name:
                    lr_map['lr_stage2'] = lr_val
                elif 'stage3' in group_name:
                    lr_map['lr_stage3'] = lr_val
                else:
                    lr_map['lr_head'] = lr_val

            print(f"[Epoch {epoch+1}] Learning rates: f{lr_map=}")

            epoch_metrics.update(lr_map)

        # Store to memory
        self.history.append(epoch_metrics)

        # Prepare dynamic header and line format
        headers = list(epoch_metrics.keys())
        values = [epoch_metrics[h] for h in headers]

        # If first epoch and file is empty, write header
        if epoch == 0 and not os.path.exists(self.csv_file_path):
            with open(self.csv_file_path, 'w') as f:
                f.write(','.join(headers) + '\n')

        # Append data
        with open(self.csv_file_path, 'a') as f:
            f.write(','.join(map(str, values)) + '\n')

    def get_dataframe(self):
        return pd.DataFrame(self.history)
    
#%%
import tensorflow as tf
class MetricsTracker(tf.keras.callbacks.Callback):
    def __init__(self, csv_file_path):
        super(MetricsTracker, self).__init__()
        self.csv_file_path = csv_file_path
        self.history = []

        # Define expected headers for manual mode
        self.headers = ['epoch', 'loss', 'val_loss', 'accuracy', 'val_accuracy',
                        'lr_stage2', 'lr_stage3', 'lr_head']

        # Write CSV header if file doesn't exist
        if not os.path.exists(self.csv_file_path):
            with open(self.csv_file_path, 'w') as f:
                f.write(','.join(self.headers) + '\n')
        else:
            # Optional: Check if existing header matches, could clear if not
            pass

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        opt = self.model.optimizer

        # Initialize LR values to NaN
        lr_stage2, lr_stage3, lr_head = float('nan'), float('nan'), float('nan')

        # --- Simplified Logic for Manual MultiOptimizer ---
        if isinstance(opt, tfa.optimizers.MultiOptimizer):
            optimizers = [sub_opt for sub_opt, _ in opt.optimizers_and_layers]

            # Assume fixed order: stage2, stage3, head
            if len(optimizers) == 3:
                try:
                    lr_stage2 = tf.keras.backend.get_value(optimizers[0].learning_rate)
                except Exception as e:
                    print(f"Warning: Could not get LR for optimizer 0 (stage2): {e}")
                try:
                    lr_stage3 = tf.keras.backend.get_value(optimizers[1].learning_rate)
                except Exception as e:
                    print(f"Warning: Could not get LR for optimizer 1 (stage3): {e}")
                try:
                    lr_head = tf.keras.backend.get_value(optimizers[2].learning_rate)
                except Exception as e:
                    print(f"Warning: Could not get LR for optimizer 2 (head): {e}")
            else:
                print(f"Warning: Expected 3 optimizers in MultiOptimizer, found {len(optimizers)}. LR tracking might be incorrect.")
        else:
            print("Warning: Optimizer is not MultiOptimizer. LR tracking for manual mode might fail.")
            # Optional: Add fallback for single optimizer if needed for other modes
            # try:
            #     lr_val = tf.keras.backend.get_value(opt.learning_rate)
            #     lr_head = lr_val # Assign to head or a generic 'lr' column
            # except Exception:
            #     pass

        # --- Collect Metrics ---
        epoch_metrics = {
            'epoch': epoch + 1,
            'loss': logs.get('loss'),
            'val_loss': logs.get('val_loss'),
            'accuracy': logs.get('accuracy'),
            'val_accuracy': logs.get('val_accuracy'),
            'lr_stage2': lr_stage2,
            'lr_stage3': lr_stage3,
            'lr_head': lr_head,
        }

        # Store to memory
        self.history.append(epoch_metrics)

        # --- Append to CSV ---
        # Ensure values are in the correct order defined by self.headers
        values = [epoch_metrics.get(h, float('nan')) for h in self.headers]

        try:
            with open(self.csv_file_path, 'a') as f:
                f.write(','.join(map(str, values)) + '\n')
        except IOError as e:
            print(f"Error writing to CSV {self.csv_file_path}: {e}")

    def get_dataframe(self):
        """Returns the history collected so far as a pandas DataFrame."""
        return pd.DataFrame(self.history)

# --- Example Usage (assuming 'model' is compiled with the MultiOptimizer) ---
# metrics_tracker = MetricsTracker('path/to/your/training_log.csv')
# model.fit(..., callbacks=[metrics_tracker, ...])
# history_df = metrics_tracker.get_dataframe()   
    
    
#%%    

# from keras.models import load_model    
from keras.layers import Layer
# import tensorflow as tf

class LayerScale(Layer):
    def __init__(self, init_values=1e-5, projection_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.init_values = init_values
        self.projection_dim = projection_dim
    
    def build(self, input_shape):
        self.gamma = self.add_weight(
            name="gamma",
            shape=(input_shape[-1],),
            initializer=tf.keras.initializers.Constant(self.init_values),
            trainable=True,
        )
    
    def call(self, inputs):
        return self.gamma * inputs
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "init_values": self.init_values,
            "projection_dim": self.projection_dim
        })
        return config

# from keras_tuner import HyperModel
# from tensorflow.keras import layers, models, applications, optimizers, regularizers

# class ConvNeXtLargeSEAutoModel(HyperModel):
#     def __init__(self, classes, IMG_WIDTH, IMG_HEIGHT):
#         self.classes = classes
#         self.IMG_WIDTH = IMG_WIDTH
#         self.IMG_HEIGHT = IMG_HEIGHT

#     def build(self, hp):
#         input_shape = (self.IMG_HEIGHT, self.IMG_WIDTH, 3)
#         inputs = layers.Input(shape=input_shape)

#         base_model = applications.ConvNeXtLarge(
#             include_top=False,
#             weights="imagenet",
#             input_tensor=inputs,
#             pooling=None
#         )
#         base_model.trainable = hp.Boolean("trainable_backbone", default=False)
#         x = base_model.output

#         activation_fn = hp.Choice("activation_fn", ["relu", "gelu", "swish"], default="relu")
#         use_batchnorm = hp.Boolean("use_batchnorm", default=True)
#         dropout_rate = hp.Float("dropout", 0.2, 0.6, step=0.1, default=0.4)
#         l2_reg = hp.Float("l2_reg", 1e-5, 1e-2, sampling="log", default=1e-4)

#         # SE Block
#         if hp.Boolean("use_se_block", default=True):
#             se_position = hp.Choice("se_block_position", ["pre_gap", "post_gap"])
#             se_ratio = hp.Choice("se_ratio", [8, 16, 32], default=16)

#             if se_position == "pre_gap":
#                 x = self.squeeze_excitation_layer(x, out_dim=x.shape[-1], ratio=se_ratio, activation=activation_fn)

#         x = layers.GlobalAveragePooling2D()(x)

#         if use_batchnorm:
#             x = layers.BatchNormalization()(x)

#         if hp.Boolean("use_se_block", default=True) and se_position == "post_gap":
#             x = self.squeeze_excitation_dense(x, units=x.shape[-1], ratio=se_ratio, activation=activation_fn)

#         # ⚠️ Fixed custom dense stack with tuning options inside
#         for units in [1024, 512, 256, 64, 32, 16]:
#             x = layers.Dense(units, activation=activation_fn,
#                              kernel_regularizer=regularizers.l2(l2_reg))(x)
#             x = layers.Dropout(dropout_rate)(x)
#             if use_batchnorm:
#                 x = layers.BatchNormalization()(x)

#         outputs = layers.Dense(self.classes, activation="softmax")(x)
#         model = models.Model(inputs, outputs)

#         optimizer_name = hp.Choice("optimizer", ["adam", "sgd", "rmsprop"], default="adam")
#         learning_rate = hp.Float("learning_rate", 1e-5, 1e-2, sampling="log", default=1e-3)

#         optimizer_map = {
#             'adam': optimizers.Adam,
#             'sgd': optimizers.SGD,
#             'rmsprop': optimizers.RMSprop
#         }
#         optimizer = optimizer_map[optimizer_name](learning_rate=learning_rate)

#         model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
#         return model

#     def squeeze_excitation_layer(self, input_layer, out_dim, ratio, activation='relu'):
#         squeeze = layers.GlobalAveragePooling2D()(input_layer)
#         excitation = layers.Dense(out_dim // ratio, activation=activation)(squeeze)
#         excitation = layers.Dense(out_dim, activation='sigmoid')(excitation)
#         excitation = layers.Reshape((1, 1, out_dim))(excitation)
#         return layers.Multiply()([input_layer, excitation])

#     def squeeze_excitation_dense(self, input_tensor, units, ratio, activation='relu'):
#         excitation = layers.Dense(units // ratio, activation=activation)(input_tensor)
#         excitation = layers.Dense(units, activation='sigmoid')(excitation)
#         return layers.Multiply()([input_tensor, excitation])

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_tuner_results(tuner, csv_path="tuner_results.csv", save_plots=True, plot_dir="plots"):
    os.makedirs(plot_dir, exist_ok=True)
    print("🔍 Extracting tuner results...")

    # Step 1: Collect results
    results = []
    for trial in tuner.oracle.get_best_trials(num_trials=len(tuner.oracle.trials)):
        trial_data = trial.hyperparameters.values.copy()
        trial_data["val_accuracy"] = trial.score if trial.score is not None else 0.0
        results.append(trial_data)

    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved results to: {csv_path}")

    # Step 2: Display top 5
    print("\n📋 Top 5 configurations by validation accuracy:")
    print(df.sort_values("val_accuracy", ascending=False).head())

    # Step 3: Plot accuracy vs each hyperparameter
    print("\n📊 Generating tuning plots...")
    categorical_columns = ["trainable_backbone", "use_se_block", "se_block_position", 
                           "se_ratio", "activation_fn", "use_batchnorm", "optimizer"]
    numeric_columns = ["dropout", "l2_reg", "learning_rate"]

    for col in df.columns:
        if col == "val_accuracy":
            continue

        plt.figure(figsize=(8, 4))
        if col in categorical_columns:
            sns.boxplot(x=col, y="val_accuracy", data=df)
        elif col in numeric_columns:
            sns.scatterplot(x=col, y="val_accuracy", data=df)
            if col == "learning_rate":
                plt.xscale("log")
        else:
            continue  # skip unexpected types

        plt.title(f"{col} vs Validation Accuracy")
        plt.tight_layout()
        if save_plots:
            filename = f"{col}_vs_accuracy.png"
            plt.savefig(os.path.join(plot_dir, filename))
        plt.show()

    print("✅ All plots saved to:", plot_dir)
    return df

# from keras.models import load_model    
from keras.layers import Layer
# import tensorflow as tf

def freeze_backbone_layers(base_model, freeze_until=140):
    """
    Freeze the first `freeze_until` layers of the backbone model.
    
    Parameters:
        base_model (tf.keras.Model): Pretrained ConvNeXt model.
        freeze_until (int): Index of the last layer to freeze.
    
    """
    total_layers = len(base_model.layers)

    for i, layer in enumerate(base_model.layers):
        layer.trainable = i >= freeze_until

    num_trainable = sum(layer.trainable for layer in base_model.layers)
    num_frozen = total_layers - num_trainable

    print(f"✔ Backbone layer freezing applied:")
    print(f"  Total layers        : {total_layers}")
    print(f"  Frozen layers       : {num_frozen} (0 to {freeze_until - 1})")
    print(f"  Trainable layers    : {num_trainable} ({freeze_until} to {total_layers - 1})")

    return base_model

class LayerScale(Layer):
    def __init__(self, init_values=1e-5, projection_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.init_values = init_values
        self.projection_dim = projection_dim
    
    def build(self, input_shape):
        self.gamma = self.add_weight(
            name="gamma",
            shape=(input_shape[-1],),
            initializer=tf.keras.initializers.Constant(self.init_values),
            trainable=True,
        )
    
    def call(self, inputs):
        return self.gamma * inputs
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "init_values": self.init_values,
            "projection_dim": self.projection_dim
        })
        return config

#%%
from tensorflow.keras.callbacks import Callback
import tensorflow as tf

class MetricsTrackerCosineDecay_backup(Callback):
    def __init__(self, csv_file_path, optimizer_config=None):
        super().__init__()
        self.csv_file_path = csv_file_path
        self.history = []
        self.optimizer_config = optimizer_config or {}

        # Define CSV header fields
        self.header_fields = ['epoch', 'loss', 'val_loss', 'accuracy', 'val_accuracy']
        self.header_fields += [f'lr_{name}' for name in self.optimizer_config]

        # Write header to file if not present
        if not os.path.exists(self.csv_file_path):
            with open(self.csv_file_path, 'w') as f:
                f.write(','.join(self.header_fields) + '\n')

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        epoch_metrics = {
            'epoch': epoch + 1,
            'loss': logs.get('loss'),
            'val_loss': logs.get('val_loss'),
            'accuracy': logs.get('accuracy'),
            'val_accuracy': logs.get('val_accuracy'),
        }

        # Extract per-group learning rates
        for name, group in self.optimizer_config.items():
            lr_schedule = group['lr']
            try:
                if isinstance(lr_schedule, tf.keras.optimizers.schedules.LearningRateSchedule):
                    lr_val = tf.keras.backend.get_value(
                        lr_schedule(self.model.optimizer.iterations)
                    )
                else:
                    lr_val = tf.keras.backend.get_value(lr_schedule)
            except Exception:
                lr_val = float('nan')

            epoch_metrics[f'lr_{name}'] = lr_val

        # Verbose debug print
        print(f"[Epoch {epoch + 1}] Learning rates:")
        for name in self.optimizer_config:
            print(f"  ↪ lr_{name}: {epoch_metrics[f'lr_{name}']:.8f}")
        print(epoch_metrics)

        # Append to in-memory history and disk log
        self.history.append(epoch_metrics)
        with open(self.csv_file_path, 'a') as f:
            line = ','.join(str(epoch_metrics.get(k, '')) for k in self.header_fields)
            f.write(line + '\n')

    def get_dataframe(self):
        return pd.DataFrame(self.history)
    
#%%
from tensorflow.keras.callbacks import Callback
import tensorflow as tf
class MetricsTrackerCosineDecay_backup2(Callback):
    def __init__(self, csv_file_path, optimizer_config=None):
        super().__init__()
        self.csv_file_path = csv_file_path
        self.history = []
        self.optimizer_config = optimizer_config or {}

        # Define CSV header fields
        self.header_fields = ['epoch', 'loss', 'val_loss', 'accuracy', 'val_accuracy']
        self.header_fields += [f'lr_{name}' for name in self.optimizer_config]

        # Create CSV with header if not exists
        if not os.path.exists(self.csv_file_path):
            with open(self.csv_file_path, 'w') as f:
                f.write(','.join(self.header_fields) + '\n')

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_step = tf.keras.backend.get_value(self.model.optimizer.iterations)

        epoch_metrics = {
            'epoch': epoch + 1,
            'loss': logs.get('loss'),
            'val_loss': logs.get('val_loss'),
            'accuracy': logs.get('accuracy'),
            'val_accuracy': logs.get('val_accuracy'),
        }

        # Extract learning rates per group
        for name, group in self.optimizer_config.items():
            lr_schedule = group['lr']
            try:
                if isinstance(lr_schedule, tf.keras.optimizers.schedules.LearningRateSchedule):
                    lr_val = tf.keras.backend.get_value(lr_schedule(current_step))
                else:
                    lr_val = tf.keras.backend.get_value(lr_schedule)
            except Exception:
                lr_val = float('nan')

            epoch_metrics[f'lr_{name}'] = lr_val

        # Optional verbose logging
        print(f"[Epoch {epoch + 1}] Learning Rates at Step {current_step}:")
        for name in self.optimizer_config:
            print(f"  ↪ lr_{name}: {epoch_metrics[f'lr_{name}']:.8f}")
        print(epoch_metrics)

        # Save to memory and disk
        self.history.append(epoch_metrics)
        with open(self.csv_file_path, 'a') as f:
            line = ','.join(str(epoch_metrics.get(k, '')) for k in self.header_fields)
            f.write(line + '\n')

    def get_dataframe(self):
        return pd.DataFrame(self.history)

#%%
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

class MetricsTrackerCosineDecay(Callback):
    """
    Tracks training and validation metrics + logs LR values from multiple optimizers
    (e.g., MultiOptimizer + CosineDecayRestarts) into a CSV file.
    """

    def __init__(self, csv_file_path, optimizer_config=None, verbose=True):
        super().__init__()
        self.csv_file_path = csv_file_path
        self.optimizer_config = optimizer_config or {}
        self.verbose = verbose

        self.history = []
        self.global_step = 0
        self.steps_per_epoch = None

        # === Define CSV header fields ===
        self.header_fields = ['epoch', 'loss', 'val_loss', 'accuracy', 'val_accuracy']
        self.header_fields += [f'lr_{name}' for name in self.optimizer_config]

        # Create CSV if not exist
        if not os.path.exists(self.csv_file_path):
            with open(self.csv_file_path, 'w') as f:
                f.write(','.join(self.header_fields) + '\n')

    # ===============================================================
    def on_train_begin(self, logs=None):
        # Estimate steps per epoch (used for CosineDecay)
        self.steps_per_epoch = self.params.get("steps", None)
        if self.verbose:
            print(f"[MetricsTracker] Steps per epoch = {self.steps_per_epoch}")

    def on_batch_end(self, batch, logs=None):
        # Maintain manual global step counter
        self.global_step += 1

    # ===============================================================
    def _get_lr_value(self, lr_schedule):
        """Safely evaluate learning rate from a tf.keras schedule or scalar."""
        try:
            # Case 1: schedule (e.g., CosineDecayRestarts)
            if isinstance(lr_schedule, tf.keras.optimizers.schedules.LearningRateSchedule):
                return float(lr_schedule(self.global_step).numpy())
            # Case 2: optimizer LR variable
            elif tf.is_tensor(lr_schedule):
                return float(tf.keras.backend.get_value(lr_schedule))
            else:
                return float(lr_schedule)
        except Exception:
            return float('nan')

    # ===============================================================
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # --- Base metrics ---
        epoch_metrics = {
            'epoch': epoch + 1,
            'loss': logs.get('loss'),
            'val_loss': logs.get('val_loss'),
            'accuracy': logs.get('accuracy'),
            'val_accuracy': logs.get('val_accuracy'),
        }
        print(f"[Epoch {epoch + 1}] Global Step = {self.global_step}")
        # --- Learning rate tracking ---
        for name, group in self.optimizer_config.items():
            lr_val = self._get_lr_value(group['lr'])
            epoch_metrics[f'lr_{name}'] = lr_val

        # --- Console summary ---
        if self.verbose:
            print(f"[Epoch {epoch + 1}] Global Step = {self.global_step}")
            for name in self.optimizer_config:
                print(f"  ↪ lr_{name}: {epoch_metrics[f'lr_{name}']:.6e}")
            print(f"  ↪ loss={epoch_metrics['loss']:.4f}, val_loss={epoch_metrics['val_loss']:.4f}")

        # --- Save to in-memory history ---
        self.history.append(epoch_metrics)

        # --- Append to CSV ---
        with open(self.csv_file_path, 'a') as f:
            line = ','.join(
                f"{epoch_metrics[k]:.10e}" if isinstance(epoch_metrics[k], float) and not pd.isna(epoch_metrics[k])
                else str(epoch_metrics[k])
                for k in self.header_fields
            )
            f.write(line + '\n')

    # ===============================================================
    def get_dataframe(self):
        """Return logged history as pandas DataFrame."""
        return pd.DataFrame(self.history)

#%%
import tensorflow as tf
from tensorflow.keras import layers, models, applications

class CustomConvNeXtWithHead:
    def __init__(self,
                 input_shape=(256, 256, 3),
                 num_classes=2,
                 dense_units=[128, 64, 32, 16],
                 activation='relu',
                 dropout_rate=0.3):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dense_units = dense_units
        self.activation = activation
        self.dropout_rate = dropout_rate

    def build_model(self, freeze_backbone=True):
        # Load base ConvNeXtLarge without top
        base_model = applications.ConvNeXtLarge(
            include_top=False,
            weights="imagenet",
            input_shape=self.input_shape,
            pooling=None
        )

        if freeze_backbone:
            for layer in base_model.layers:
                layer.trainable = False

        # Add classification head
        inputs = base_model.input
        x = base_model.output
        # x = layers.LayerNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)

        for units in self.dense_units:
            x = layers.Dense(units, activation=self.activation)(x)
            x = layers.Dropout(self.dropout_rate)(x)
            x = layers.LayerNormalization()(x)

        outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        self.model = models.Model(inputs=inputs, outputs=outputs)
        return self.model

    def copy_weights_from(self, loaded_model, max_layer_index=None):
        """
        Copy weights from another model (usually pre-trained).
        If max_layer_index is given, only up to that layer will be copied.
        """
        if not hasattr(self, 'model'):
            raise ValueError("Model not built yet. Call build_model() first.")

        print(f"Copying weights from layer 0 to {max_layer_index or len(loaded_model.layers) - 1}")
        for i in range(len(self.model.layers)):
            if max_layer_index is not None and i > max_layer_index:
                break
            try:
                self.model.layers[i].set_weights(loaded_model.layers[i].get_weights())
            except Exception as e:
                print(f"Skipped layer {i}: {self.model.layers[i].name} — {str(e)}")

import tensorflow as tf
from tensorflow.keras import layers, models, applications, regularizers
class CustomConvNeXtWithHeadL2:
    def __init__(self,
                 model_name,
                 input_shape=(256, 256, 3),
                 num_classes=2,
                 dense_units=[128, 64, 32, 16, 8],
                 activation='relu',
                 dropout_rate=0.3,
                 l2_reg=3e-5):
        self.model_name = model_name
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dense_units = dense_units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg

    def build_model(self, freeze_backbone=True, add_l2=True):
        # --- Base backbone ---
        if self.model_name == 'ConvNeXtLarge':
            base_model = applications.ConvNeXtLarge(
                include_top=False,
                weights="imagenet",
                input_shape=self.input_shape,
                pooling=None
            )
            
        if self.model_name == 'ConvNeXtTiny':
            base_model = applications.ConvNeXtTiny(
                include_top=False,
                weights="imagenet",
                input_shape=self.input_shape,
                pooling=None
            )

        if freeze_backbone:
            for layer in base_model.layers:
                layer.trainable = False

        # 6517.log
        # --- Classification head ---
        # inputs = base_model.input
        # x = base_model.output
        # x = layers.GlobalAveragePooling2D(name="gap")(x)
        
        # for units in self.dense_units:
        #     x = layers.Dense(
        #         units,
        #         activation=self.activation,
        #         kernel_regularizer=regularizers.l2(self.l2_reg) if add_l2 else None,
        #         name=f"dense_{units}"
        #     )(x)
            
        #     x = layers.Dropout(self.dropout_rate, name=f"dropout_{units}")(x)
        #     x = layers.LayerNormalization(name=f"ln_{units}")(x)

        # # --- Final classifier ---
        # if self.num_classes == 1:
        #     activation = "sigmoid"
        #     output_units = 1
        # else:
        #     activation = "softmax"
        #     output_units = self.num_classes

        # outputs = layers.Dense(
        #     output_units,
        #     activation=activation,
        #     kernel_regularizer=regularizers.l2(self.l2_reg) if add_l2 else None, #6517.log         
        #     name="classifier_output"
        # )(x)

        # --- Classification head ---
        inputs = base_model.input
        x = base_model.output
        x = layers.GlobalAveragePooling2D(name="gap")(x)

        for i, units in enumerate(self.dense_units):
            x = layers.Dense(
                units,
                activation=self.activation,
                kernel_regularizer=regularizers.l2(self.l2_reg) if add_l2 else None,
                name=f"dense_{units}_{i}"
            )(x)
            # 6571.log
            x = layers.Dropout(self.dropout_rate, name=f"dropout_{units}")(x)
            x = layers.LayerNormalization(name=f"ln_{units}")(x)
            
            # 6577.log
            # x = layers.LayerNormalization(epsilon=1e-5, name=f"ln_{units}_{i}")(x)            
            # x = layers.Dropout(rate=self.dropout_rate, name=f"dropout_{units}_{i}")(x)

        # --- Final classifier ---
        if self.num_classes == 1:
            activation = "sigmoid"
            output_units = 1
        else:
            activation = "softmax"
            output_units = self.num_classes

        outputs = layers.Dense(
            output_units,
            activation=activation,
            kernel_regularizer=regularizers.l2(self.l2_reg) if add_l2 else None,
            name="classifier_output"
        )(x)

        model = models.Model(inputs=inputs, outputs=outputs,
                             name=f"{self.model_name}_CustomHead")

        self.model = model
        return self.model

    def copy_weights_from(self, loaded_model, max_layer_index=None):
        """
        Copy weights from another pretrained model up to a given index.
        Useful when partially reusing ConvNeXt weights.
        """
        if not hasattr(self, 'model'):
            raise ValueError("Model not built yet. Call build_model() first.")

        total_layers = len(loaded_model.layers)
        print(f"Copying weights up to layer {max_layer_index or total_layers - 1}.")

        for i in range(len(self.model.layers)):
            if max_layer_index is not None and i > max_layer_index:
                break
            try:
                self.model.layers[i].set_weights(loaded_model.layers[i].get_weights())
            except Exception as e:
                print(f"Skipped layer {i}: {self.model.layers[i].name} — {str(e)}")


#%%
import tensorflow as tf
from tensorflow.keras import layers

class AttentionMIL(layers.Layer):
    """
    Standard Attention-based MIL layer for the Euclidean baseline.
    """
    def __init__(self, L_dim=256, **kwargs):
        super(AttentionMIL, self).__init__(**kwargs)
        self.L_dim = L_dim
        self.V = layers.Dense(L_dim, activation='tanh', name='attention_V')
        self.w = layers.Dense(1, activation=None, use_bias=False, name='attention_w')

    def call(self, inputs):
        # inputs shape: (Batch, Num_Patches, Features)
        attention = self.V(inputs)
        attention_scores = self.w(attention)
        attention_weights = tf.nn.softmax(attention_scores, axis=1)
        weighted_features = inputs * attention_weights
        return tf.reduce_sum(weighted_features, axis=1)

    def get_config(self):
        config = super(AttentionMIL, self).get_config()
        config.update({"L_dim": self.L_dim})
        return config
        
class CustomConvNeXtWithHeadL2_Bag:
    """
    Bag-level MIL classifier using ConvNeXt as patch encoder + Dense aggregation head.
    Designed for compatibility with PatientBagDataset.
    """

    def __init__(self,
                 input_shape=(None, 256, 256, 3),  # <-- RGBA patches from PatientBagDataset
                 num_classes=2,
                 dense_units=[128, 64, 32, 16],
                 activation='relu',
                 dropout_rate=0.3,
                 l2_reg=3e-5,
                 use_rgba_projection=True):
        
        self.input_shape = input_shape           # (K, H, W, 4)
        self.num_classes = num_classes
        self.dense_units = dense_units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.use_rgba_projection = use_rgba_projection

    # ------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------
    def build_model(self, freeze_backbone=True, add_l2=True):
        K = self.input_shape[0]
        patch_shape = self.input_shape[1:]  # (256, 256, 3)

        # 1. Base ConvNeXt (Use pooling=None to match Stage 2 structure)
        base_model = applications.ConvNeXtLarge(
            include_top=False,
            weights="imagenet",
            input_shape=patch_shape,
            pooling=None  # <-- KEEP THIS NONE for weight alignment
        )

        if freeze_backbone:
            for layer in base_model.layers:
                layer.trainable = False

        # 2. Bag Input
        bag_input = layers.Input(shape=self.input_shape, name="bag_input")

        # 3. Optional Projection
        if self.input_shape[-1] == 4 and self.use_rgba_projection:
            projected = layers.Dense(3, name="rgba_projection")(bag_input)
        else:
            projected = bag_input

        # 4. Apply Backbone to Each Patch
        # Output shape: (Batch, 32, 8, 8, 1536)
        patch_features_map = layers.TimeDistributed(
            base_model, 
            name="patch_encoder"
        )(projected) 

        # 5. Apply Spatial Pooling to Each Patch (THE FIX)
        # This flattens 8x8 spatial dims -> 1 vector per patch
        # Output shape: (Batch, 32, 1536)
        patch_features = layers.TimeDistributed(
            layers.GlobalAveragePooling2D(), 
            name="patch_spatial_pooling"
        )(patch_features_map)

        # 6. Bag Pooling (Aggregate patches)
        # Output shape: (Batch, 1536)
        # bag_feature = layers.GlobalAveragePooling1D(name="bag_pooling")(patch_features)
        #Revise 2026-08-27
        bag_feature = AttentionMIL(L_dim=256, name="attention_mil_pooling")(patch_features)

        # 7. Dense Head
        x = bag_feature
        for i, units in enumerate(self.dense_units):
            x = layers.Dense(
                units,
                activation=self.activation,
                kernel_regularizer=regularizers.l2(self.l2_reg) if add_l2 else None,
                name=f"dense_{units}_{i}"
            )(x)
            x = layers.Dropout(self.dropout_rate, name=f"dropout_{units}_{i}")(x)
            x = layers.LayerNormalization(name=f"ln_{units}_{i}")(x)

        # 8. Final Classifier
        output_units = 1 if self.num_classes == 1 else self.num_classes
        act = "sigmoid" if self.num_classes == 1 else "softmax"

        outputs = layers.Dense(
            output_units,
            activation=act,
            kernel_regularizer=regularizers.l2(self.l2_reg) if add_l2 else None,
            name="classifier_output"
        )(x)

        self.model = models.Model(inputs=bag_input, outputs=outputs, name="ConvNeXtLarge_BagModel")
        return self.model, base_model

    # ------------------------------------------------------------
    # Copy weights from pretrained patch-level model
    # ------------------------------------------------------------
    def copy_weights_from(self, loaded_model, max_layer_index=None):
        if not hasattr(self, "model"):
            raise ValueError("Call build_model() before copying weights.")
        
        target_base = self.model.get_layer("patch_encoder").layer

        for i in range(len(target_base.layers)):
            if max_layer_index is not None and i > max_layer_index:
                break
            try:
                target_base.layers[i].set_weights(
                    loaded_model.layers[i].get_weights()
                )
            except Exception:
                pass

def transfer_convnext_backbone_weights(Mk_model, Mf_model, verbose=True):
    """
    Copies ConvNeXt backbone weights from the stage-2 patch model (Mk_model)
    into the stage-3 bag-level model (Mf_model).
    
    Since Mk_model is a Functional model (flat layers), we copy the first N layers
    to the inner backbone of Mf_model.
    """
    
    # -----------------------------
    # 1. Identify target backbone inside Mf_model
    # -----------------------------
    try:
        # Mf_model -> "patch_encoder" (TimeDistributed) -> .layer (ConvNeXtLarge)
        target_backbone = Mf_model.get_layer("patch_encoder").layer
    except Exception as e:
        raise ValueError(f"Could not find backbone inside Mf_model: {e}")

    if verbose:
        print(f"\n=== Transferring ConvNeXt backbone weights ===")
        print(f"Source model (Mk): {len(Mk_model.layers)} layers")
        print(f"Target backbone (inside Mf): {len(target_backbone.layers)} layers")

    # -----------------------------
    # 2. Copy weights by index
    # -----------------------------
    # We assume the first len(target_backbone.layers) in Mk_model 
    # correspond exactly to the backbone.
    
    copied = 0
    skipped = 0
    no_weights = 0

    for i, t_layer in enumerate(target_backbone.layers):
        try:
            # Get corresponding layer from source
            # (Assumes Mk_model starts with the exact same backbone structure)
            s_layer = Mk_model.layers[i]
            
            s_weights = s_layer.get_weights()
            
            if s_weights:
                t_layer.set_weights(s_weights)
                copied += 1
            else:
                no_weights += 1 # Layer has no weights (e.g., Input, Flatten, Activation)
                
        except Exception as e:
            skipped += 1
            if verbose:
                print(f"Warning: Failed to copy layer {i} ({t_layer.name}): {e}")

    print(f"Transfer Summary:")
    print(f"  - Layers with weights copied: {copied}")
    print(f"  - Layers without weights (skipped safely): {no_weights}")
    print(f"  - Errors/Mismatches: {skipped}")
    print("  - Backbone transfer complete.\n")

    
#%%

