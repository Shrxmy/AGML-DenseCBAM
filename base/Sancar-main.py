# Import necessary libraries and modules
import numpy as np 
import pandas as pd 
import os 
import itertools
from PIL import Image 
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set_style('darkgrid') 
import json
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix , classification_report
 
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping,ModelCheckpoint, ReduceLROnPlateau
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D,Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization,Concatenate
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score 
import warnings
warnings.filterwarnings("ignore")  
import gc

#load dataset
img_size = (224, 224)   
batch_size = 8
 
tr_gen = ImageDataGenerator(rescale=1.0/255.0,  
                            horizontal_flip=True,   
                           ) 

tst_gen = ImageDataGenerator(rescale=1.0/255.0);
 
train_gen = tr_gen.flow_from_directory('data/train',   
                                       target_size=img_size, 
                                       class_mode='categorical', 
                                       color_mode='rgb', 
                                       shuffle=True, 
                                       batch_size=batch_size 
                                      )  
 
val_gen = tr_gen.flow_from_directory('data/validation',  
                                       target_size=img_size, 
                                       class_mode='categorical', 
                                       color_mode='rgb', 
                                       shuffle=False, 
                                       batch_size=batch_size 
                                      )  
 
test_gen = tst_gen.flow_from_directory('data/test',   
                                      target_size=img_size, 
                                      class_mode='categorical', 
                                      color_mode='rgb', 
                                      shuffle=False, 
                                      batch_size=batch_size
                                     )


#define attention block function

from tensorflow.keras.layers import (
    Layer, Conv2D, Dense, GlobalAveragePooling2D, Multiply, 
    Add, Softmax, Reshape
)

class AttentionBlock(Layer): 
    def __init__(self, attention_type="cbam", reduction_ratio=16, **kwargs):
        super(AttentionBlock, self).__init__(**kwargs) 
        self.attention_type = attention_type.lower()
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        filters = input_shape[-1]
        reduced_filters = max(filters // self.reduction_ratio, 1)
 
        if self.attention_type == "se":
            self.global_avg_pool = GlobalAveragePooling2D()
            self.dense1 = Dense(reduced_filters, activation="relu")
            self.dense2 = Dense(filters, activation="sigmoid")
 
        elif self.attention_type == "self":
            self.query_conv = Conv2D(filters // 8, (1, 1), padding="same")
            self.key_conv = Conv2D(filters // 8, (1, 1), padding="same")
            self.value_conv = Conv2D(filters, (1, 1), padding="same")
 
        elif self.attention_type == "cbam":
            self.global_avg_pool = GlobalAveragePooling2D()
            self.dense1 = Dense(reduced_filters, activation="relu")
            self.dense2 = Dense(filters, activation="sigmoid") 
            self.spatial_attention = Conv2D(1, (7, 7), activation="sigmoid", padding="same")

    def call(self, inputs): 
        if self.attention_type == "se": 
            se = self.global_avg_pool(inputs)
            se = self.dense1(se)
            se = self.dense2(se)
            se = Reshape((1, 1, -1))(se)
            return Multiply()([inputs, se])
 
        elif self.attention_type == "self": 
            query = self.query_conv(inputs)
            key = self.key_conv(inputs)
            value = self.value_conv(inputs)  
            attention = Softmax(axis=-1)(tf.linalg.matmul(query, key, transpose_b=True)) 
            attention = tf.linalg.matmul(attention, value) 
            return Add()([inputs, attention])
 
        elif self.attention_type == "cbam":
            # CBAM - Önce Kanal Dikkati
            ca = self.global_avg_pool(inputs)
            ca = self.dense1(ca)
            ca = self.dense2(ca)
            ca = Reshape((1, 1, -1))(ca)
            x = Multiply()([inputs, ca]) 
            sa = self.spatial_attention(x)
            x = Multiply()([x, sa]) 
            return x
  

def build_model(model_name='resnet18' ):
# Load DenseNet201 model pre-trained on ImageNet without top layers
    base_model = tf.keras.applications.DenseNet201(include_top=False, weights='imagenet', input_shape=(224, 224, 3),pooling= None) 
    
    x = base_model.get_layer("pool3_relu").output
# Apply self-attention mechanism to enhance feature representations
    x = AttentionBlock(attention_type="self")(x) 
    x = base_model.get_layer("conv5_block32_concat").output  
     
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu',kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(2, activation='softmax')(x)   
     
# Define the final model architecture and connect input-output layers
    model = tf.keras.Model(inputs=base_model.input, outputs=x)

    return model

from tensorflow.keras.layers import Multiply 
import numpy as np 
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
 
model = build_model(model_name)
# Compile the model with optimizer, loss function, and metrics
model.compile(
    optimizer=Adam(0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.1, 
    patience=3, 
    min_lr=1e-6,
    verbose=1
)
model.summary()
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True,
    verbose=1
)
#Train the model with training and validation datasets
history = model.fit(
    train_gen,
    epochs=50, 
    validation_data=val_gen,
    verbose=1,
    callbacks=[early_stopping,reduce_lr]
)

# Evaluate the model performance on test dataset
results = model.evaluate(test_gen,verbose=0)
print(f'Test Loss: {results[0]}, Test Accuracy: {results[1]}\n')
y_true = test_gen.classes   
# Generate predictions using the trained model
y_pred_probs = model.predict(test_gen, verbose=1)  
y_pred = np.argmax(y_pred_probs, axis=1)   
y_true = np.array(y_true)
y_pred = np.array(y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted') 
# Compute the confusion matrix to analyze classification performance
cm = confusion_matrix(y_true, y_pred)
if cm.shape == (2,2):   
    TN, FP, FN, TP = cm.ravel()
    specificity = TN / (TN + FP)
else:
    specificity = None  
print(f"Precision: {precision:.4f}")
print(f"Recall (Sensitivity): {recall:.4f}")
if specificity is not None:
    print(f"Specificity: {specificity:.4f}")
print(f"F1 Score: {f1:.4f}")
    
#save history
hist_df = pd.DataFrame(history.history)
hist_df.to_csv('history.csv', index=False)

def plot_history(history, model_name, best): 
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    learning_rate = history.history['lr']
    epochs = np.arange(1, len(loss) + 1)
 
# Visualize training performance metrics such as accuracy and loss
    fig, ax1 = plt.subplots(figsize=(9, 5))
 
    ax1.plot(epochs, loss, label='Train Loss', color="#1f77b4", linewidth=2.5, linestyle='-', alpha=0.8)
    ax1.plot(epochs, val_loss, label='Validation Loss', color="#ff7f0e", linewidth=2.5, linestyle='-', alpha=0.8)
 
    ax1.scatter(best, val_loss[best-1], color='red', s=100, edgecolors='black', zorder=3, label='Best Validation Loss')
 
    ax1.set_yscale('log')
    ax1.set_xlabel('Epochs', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss (Log Scale)', fontsize=12, fontweight='bold')
    ax1.set_title(f'{model_name} Training Progress', fontsize=14, fontweight='bold')
 
    ax1.grid(which='both', linestyle=':', linewidth=0.7, alpha=0.6)
    ax1.set_facecolor('#f7f7f7')
 
    ax2 = ax1.twinx()
    ax2.plot(epochs, learning_rate, color='green', linestyle='dotted', linewidth=2, alpha=0.7, label="Learning Rate")
    ax2.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='green')
 
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', fontsize=10, frameon=True, facecolor="white")
 
    ax1.axvline(x=best, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
 
# Visualize training performance metrics such as accuracy and loss
    plt.savefig(f"{model_name}_training_plot.png", dpi=300, bbox_inches='tight', facecolor='white')
 
    plt.show()

#create training graph
plot_history(history,'DenseNet201',15)

# Import necessary libraries and modules
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.metrics import confusion_matrix
 

def create_cf_matrix(model, model_name, fontsize=14): 
# Generate predictions using the trained model
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
 
    y_true = test_gen.classes
 
# Compute the confusion matrix to analyze classification performance
    conf_matrix = confusion_matrix(y_true, y_pred_classes)
 
    class_names = ['Normal', 'SL']
 
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix,
                annot=True,
                fmt='d',
                cmap='Blues',
                annot_kws={"size": fontsize},
                xticklabels=class_names,
                yticklabels=class_names)

    plt.ylabel('Actual', fontsize=fontsize + 2)
    plt.xlabel('Predicted', fontsize=fontsize + 2) 
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    plt.title(f'Confusion Matrix - {model_name}', fontsize=fontsize + 4)
    plt.tight_layout()
# Save the trained model or results for later use
    plt.savefig(f'{model_name}.jpg', dpi=300)
    plt.show()

#create confusion matrix
create_cf_matrix(model,'Proposed Model')