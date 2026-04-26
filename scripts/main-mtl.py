# =============================================================================
# AGMTL-DenseCBAM: Attention-Guided Multi-Task Learning DenseNet with CBAM
# For Robust Temporomandibular Disorder Classification
# =============================================================================
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (
    Layer, Conv2D, Dense, GlobalAveragePooling2D, Multiply, Reshape,
    Dropout, BatchNormalization, Flatten
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import Sequence
import warnings
warnings.filterwarnings("ignore")

sns.set_style('darkgrid')

# =============================================================================
# CONFIGURATION
# =============================================================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-4
USE_CBAM = True                 # Set to False to use baseline self-attention
USE_MTL = True                  # Set to True for multi-task learning with artifact head
ARTIFACT_CLASSES = 4            # None, Motion Blur, Gaussian Noise, Metal Streak
AUX_LOSS_WEIGHT = 0.3           # Weight for auxiliary task loss

# =============================================================================
# DATA LOADING
# =============================================================================
tr_gen = ImageDataGenerator(rescale=1.0/255.0, horizontal_flip=True)
tst_gen = ImageDataGenerator(rescale=1.0/255.0)

train_gen = tr_gen.flow_from_directory(
    'data/train',
    target_size=IMG_SIZE,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True,
    batch_size=BATCH_SIZE
)

val_gen = tr_gen.flow_from_directory(
    'data/validation',
    target_size=IMG_SIZE,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=False,
    batch_size=BATCH_SIZE
)

test_gen = tst_gen.flow_from_directory(
    'data/test',
    target_size=IMG_SIZE,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=False,
    batch_size=BATCH_SIZE
)

# =============================================================================
# CELL 3: MTL Data Generator Class (Only needed if USE_MTL=True)
# =============================================================================
class MTLDataGenerator(Sequence):
    """Custom generator that yields both TMD and artifact labels."""
    def __init__(self, directory, batch_size=8, img_size=(224,224), 
                 artifact_mode='clean', shuffle=True):
        self.datagen = ImageDataGenerator(rescale=1.0/255.0, horizontal_flip=True)
        self.gen = self.datagen.flow_from_directory(
            directory, target_size=img_size, batch_size=batch_size,
            class_mode='categorical', shuffle=shuffle, color_mode='rgb'
        )
        self.artifact_mode = artifact_mode
        self.batch_size = batch_size
        
    def __len__(self):
        return len(self.gen)
    
    def __getitem__(self, idx):
        X, y_tmd = self.gen[idx]
        batch_size_actual = X.shape[0]
        # Create artifact labels (all "clean" for now)
        y_artifact = np.zeros((batch_size_actual, 4))
        y_artifact[:, 0] = 1  # One-hot: [1,0,0,0] = "None/Clean"
        return X, {'tmd_output': y_tmd, 'artifact_output': y_artifact}
    
# =============================================================================
# CORRECTED CBAM IMPLEMENTATION (Keras/TensorFlow)
# Based on: Woo et al. (2018) - https://arxiv.org/abs/1807.06521
# =============================================================================

from tensorflow.keras.layers import (
    Layer, Conv2D, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D,
    Multiply, Add, Reshape, Concatenate, Activation, Lambda
)
from tensorflow.keras import backend as K

class AttentionBlock(Layer):
    def __init__(self, attention_type="cbam", reduction_ratio=16, **kwargs):
        super(AttentionBlock, self).__init__(**kwargs)
        self.attention_type = attention_type.lower()
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        filters = input_shape[-1]
        reduced_filters = max(filters // self.reduction_ratio, 1)

        if self.attention_type == "cbam":
            # Channel Attention Shared MLP
            self.channel_dense1 = Dense(reduced_filters, activation='relu', kernel_initializer='he_normal')
            self.channel_dense2 = Dense(filters, kernel_initializer='he_normal')
            self.avg_pool = GlobalAveragePooling2D()
            self.max_pool = GlobalMaxPooling2D()
            
            # Spatial Attention Conv Layer
            self.spatial_conv = Conv2D(1, (7, 7), padding='same', activation='sigmoid', kernel_initializer='he_normal')

        elif self.attention_type == "self":
            self.query_conv = Conv2D(filters // 8, (1, 1), padding="same")
            self.key_conv = Conv2D(filters // 8, (1, 1), padding="same")
            self.value_conv = Conv2D(filters, (1, 1), padding="same")

    def call(self, inputs):
        if self.attention_type == "cbam":
            # --- 1. Channel Attention ---
            # Average Pooling branch
            avg_pool = self.avg_pool(inputs)
            avg_pool = Reshape((1, 1, -1))(avg_pool)
            avg_pool = self.channel_dense1(avg_pool)
            avg_pool = self.channel_dense2(avg_pool)
            
            # Max Pooling branch
            max_pool = self.max_pool(inputs)
            max_pool = Reshape((1, 1, -1))(max_pool)
            max_pool = self.channel_dense1(max_pool)
            max_pool = self.channel_dense2(max_pool)
            
            # Combine and apply sigmoid
            channel_attention = Add()([avg_pool, max_pool])
            channel_attention = Activation('sigmoid')(channel_attention)
            x = Multiply()([inputs, channel_attention])
            
            # --- 2. Spatial Attention ---
            # Average and Max across channels
            avg_pool_spatial = Lambda(lambda z: K.mean(z, axis=-1, keepdims=True))(x)
            max_pool_spatial = Lambda(lambda z: K.max(z, axis=-1, keepdims=True))(x)
            
            # Concatenate and apply 7x7 Conv
            concat_spatial = Concatenate(axis=-1)([avg_pool_spatial, max_pool_spatial])
            spatial_attention = self.spatial_conv(concat_spatial)
            x = Multiply()([x, spatial_attention])
            
            return x

        elif self.attention_type == "self":
            batch_size, h, w, c = tf.shape(inputs)[0], inputs.shape[1], inputs.shape[2], inputs.shape[3]
            query = self.query_conv(inputs)
            key = self.key_conv(inputs)
            value = self.value_conv(inputs)

            query = tf.reshape(query, (batch_size, h*w, -1))
            key = tf.reshape(key, (batch_size, h*w, -1))
            value = tf.reshape(value, (batch_size, h*w, -1))

            attention = tf.nn.softmax(tf.matmul(query, key, transpose_b=True) / tf.sqrt(tf.cast(c, tf.float32)))
            out = tf.matmul(attention, value)
            out = tf.reshape(out, (batch_size, h, w, -1))
            return inputs + out
        
# =============================================================================
# MODEL BUILDERS
# =============================================================================

def build_single_task_model(use_cbam=True):
    """Single-task model: DenseNet201 + Attention → Normal/Subluxation"""
    base_model = tf.keras.applications.DenseNet201(
        include_top=False,
        weights='imagenet',
        input_shape=(*IMG_SIZE, 3),
        pooling=None
    )

    # Apply attention to the final convolutional block features
    x = base_model.get_layer("conv5_block32_concat").output  # Shape: (7,7,1920)

    if use_cbam:
        x = AttentionBlock(attention_type="cbam")(x)
    else:
        x = AttentionBlock(attention_type="self")(x)

    # Classifier head
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(2, activation='softmax', name='tmd_output')(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    return model


def build_mtl_model(use_cbam=True, num_artifact_classes=4):
    """Multi-task model: DenseNet201 + Attention → TMD (2) + Artifact (N)"""
    base_model = tf.keras.applications.DenseNet201(
        include_top=False,
        weights='imagenet',
        input_shape=(*IMG_SIZE, 3),
        pooling=None
    )

    x = base_model.get_layer("conv5_block32_concat").output

    if use_cbam:
        x = AttentionBlock(attention_type="cbam")(x)
    else:
        x = AttentionBlock(attention_type="self")(x)

    shared = GlobalAveragePooling2D()(x)

    # Primary head: TMD classification
    primary = Dense(1024, activation='relu', kernel_regularizer=l2(0.01))(shared)
    primary = Dropout(0.5)(primary)
    primary = BatchNormalization()(primary)
    primary = Dense(128, activation='relu')(primary)
    tmd_output = Dense(2, activation='softmax', name='tmd_output')(primary)

    # Auxiliary head: Artifact detection
    aux = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(shared)
    aux = Dropout(0.3)(aux)
    artifact_output = Dense(num_artifact_classes, activation='softmax', name='artifact_output')(aux)

    model = Model(inputs=base_model.input, outputs=[tmd_output, artifact_output])
    return model

# =============================================================================
# BUILD MODEL
# =============================================================================
tf.keras.backend.clear_session()

if USE_MTL:
    model = build_mtl_model(use_cbam=True, num_artifact_classes=4)
    losses = {'tmd_output': 'categorical_crossentropy', 
              'artifact_output': 'categorical_crossentropy'}
    loss_weights = {'tmd_output': 1.0, 'artifact_output': 0.3}
    metrics = {'tmd_output': ['accuracy']}
    # Use MTL generators
    train_data = MTLDataGenerator('data/train', batch_size=BATCH_SIZE)
    val_data = MTLDataGenerator('data/validation', batch_size=BATCH_SIZE, shuffle=False)
else:
    model = build_single_task_model(use_cbam=True)
    losses = 'categorical_crossentropy'
    loss_weights = None
    metrics = ['accuracy']
    # Use standard generators
    train_data = train_gen
    val_data = val_gen

model.compile(optimizer=Adam(1e-4), loss=losses, loss_weights=loss_weights, metrics=metrics)
model.summary()


# =============================================================================
# TRAINING
# =============================================================================
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

# For MTL, we need to provide both labels. Since current generators only give TMD labels,
# you must either:
# 1) Use a custom generator that yields (X, {'tmd_output': y_tmd, 'artifact_output': y_art})
# 2) For initial testing, use single-task mode with USE_MTL=False
# We'll assume single-task for this script.

# if USE_MTL:
#     raise NotImplementedError(
#         "Multi-task training requires a custom data generator that provides artifact labels.\n"
#         "Set USE_MTL=False for single-task CBAM training first."
#     )

history = model.fit(
    train_data,
    epochs=50,
    validation_data=val_data,
    verbose=1,
    callbacks=[early_stopping, reduce_lr]
)


# =============================================================================
# EVALUATION
# =============================================================================
results = model.evaluate(test_gen, verbose=0)
print(f'\nTest Loss: {results[0]:.4f}, Test Accuracy: {results[1]:.4f}')

y_true = test_gen.classes
y_pred_probs = model.predict(test_gen, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)

precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

cm = confusion_matrix(y_true, y_pred)
if cm.shape == (2, 2):
    TN, FP, FN, TP = cm.ravel()
    specificity = TN / (TN + FP)
else:
    specificity = None

print(f"Precision: {precision:.4f}")
print(f"Recall (Sensitivity): {recall:.4f}")
if specificity is not None:
    print(f"Specificity: {specificity:.4f}")
print(f"F1 Score: {f1:.4f}")

# =============================================================================
# SAVE HISTORY & PLOTS
# =============================================================================
hist_df = pd.DataFrame(history.history)
hist_df.to_csv('history_cbam.csv', index=False)

def plot_history(history, model_name='DenseNet201-CBAM'):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = np.arange(1, len(loss) + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, loss, label='Train Loss', linewidth=2)
    plt.plot(epochs, val_loss, label='Validation Loss', linewidth=2)
    plt.yscale('log')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss (Log Scale)', fontsize=12)
    plt.title(f'{model_name} Training Progress', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(f'{model_name}_training_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_history(history, 'DenseNet201-CBAM')

def plot_confusion_matrix(model, test_gen, model_name='Proposed Model'):
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_gen.classes
    cm = confusion_matrix(y_true, y_pred)
    class_names = ['Normal', 'SL']

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual', fontsize=14)
    plt.xlabel('Predicted', fontsize=14)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{model_name}_cm.jpg', dpi=300)
    plt.show()

plot_confusion_matrix(model, test_gen, 'DenseNet201-CBAM')

model.save('ag_densenet_cbam.h5')