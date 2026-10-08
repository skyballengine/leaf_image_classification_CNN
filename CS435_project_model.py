# -*- coding: utf-8 -*-

import os
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import tensorflow.keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
#from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt     
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple
from sklearn.utils.class_weight import compute_class_weight
import json


def define_compile_train_model():
    BASE_DIR = Path("/Users/eusebiusballentine/Desktop/Education/CS435/leafsnap_dataset/leafsnap-dataset")
    text_file_path = "/Users/eusebiusballentine/Desktop/Education/CS435/leafsnap_dataset/leafsnap-dataset/leafsnap-dataset-images.txt"
    og_df = pd.read_csv(text_file_path, sep='\t')
    df = og_df.copy()
    # df["image_path"] = df["image_path"].str.strip()
    df["image_path"] = df["image_path"].apply(lambda p: str(BASE_DIR / p))
    # missing_mask = ~df["image_path"].apply(os.path.exists)
    # print(f"Number of broken paths remaining: {missing_mask.sum()}")
    
    # if missing_mask.sum() > 0:
    #   print("Here are some paths that still fail:")
    #   print(df.loc[missing_mask, "image_path"].head(10))
    
    # df = df.drop(columns="segmented_path")
    # print(df.head())
    df['class_label_int'], unique_names = pd.factorize(df['species'])
    # print(df.head(1000))
    
    
    
    # 1. Split off the test set (e.g., 15% of total data)
    temp_df, test_df = train_test_split(
        df, 
        test_size=0.15, 
        stratify=df["class_label_int"], 
        random_state=42
    )
    
    # 2. Split remaining data into train and validation sets (e.g., 15% of remaining for val)
    train_df, val_df = train_test_split(
        temp_df, 
        test_size=0.176, 
        stratify=temp_df["class_label_int"], 
        random_state=42
    )  # 0.176 of 85% is ~15% overall
    
    # def define_compile_train_model():
    
    # configure hyperparameters
    AUTO = tf.data.AUTOTUNE
    IMAGE_SIZE = (224, 224) 
    BATCH_SIZE = 128         
    NUM_CLASSES = 184       
    EPOCHS = 1
    
    # define the data path
    DATA_DIR = "/Users/eusebiusballentine/Desktop/Education/CS435/leafsnap_dataset/leafsnap-dataset/"
    
    def load_and_preprocess(path, label):
      image = tf.io.read_file(path)
      image = tf.image.decode_jpeg(image, channels=3)
      image = tf.image.resize(image, IMAGE_SIZE)
      image = image / 255.0  # normalize to [0,1]
      return image, label
    
    # perform a perfect stratified split (80% train, 20% val)
    # train_paths, val_paths, train_labels, val_labels = train_test_split(
    #     df,
    #     labels, 
    #     test_size=0.2, 
    #     stratify=df['class_label_int'], 
    #     random_state=42
    # )
    
    train_ds = tf.data.Dataset.from_tensor_slices((train_df["image_path"].values, train_df["class_label_int"].values))
    train_ds = train_ds.map(load_and_preprocess, num_parallel_calls=AUTO).cache().shuffle(1000).batch(BATCH_SIZE).prefetch(AUTO)
    
    val_ds = tf.data.Dataset.from_tensor_slices((val_df["image_path"].values, val_df["class_label_int"].values))
    val_ds = val_ds.map(load_and_preprocess, num_parallel_calls=AUTO).cache().batch(BATCH_SIZE).prefetch(AUTO)
    
    
    # 5 conv layer model
    cnn_model = models.Sequential([
        layers.Rescaling(1./255, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
        
        ])
    
    # compile model
    cnn_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
        
        )
    
    print(cnn_model.summary())
    
    tf.keras.utils.plot_model(cnn_model, to_file='cnn.png', show_shapes=True)
    
    # Define callbacks
    # callbacks = [
    #     tf.keras.callbacks.EarlyStopping(
    #         monitor="val_loss",
    #         patience=10,  # Stop if val_loss doesn't improve for 10 epochs
    #         restore_best_weights=True,
    #     ),
    #     tf.keras.callbacks.ModelCheckpoint(
    #         filepath="best_species_model.keras",
    #         monitor="val_sparse_categorical_accuracy",
    #         save_best_only=True,
    #     ),
    # ]
    
    # train the Network
    model_training_history = cnn_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        # callbacks=callbacks
    )
    
    # save model
    cnn_model.save("leaf_classification_model.keras")
    
    # Extract accuracy metrics from the training history dictionary
    acc = model_training_history.history["sparse_categorical_accuracy"]
    val_acc = model_training_history.history["val_sparse_categorical_accuracy"]
    top5_acc = model_training_history.history["top_5_accuracy"]
    val_top5_acc = model_training_history.history["val_top_5_accuracy"]
    
    epochs_range = range(1, len(acc) + 1)
    
    # Create the plot layout
    plt.figure(figsize=(10, 6))
    
    # Plot Top-1 Accuracies (Solid lines)
    plt.plot(
        epochs_range, acc, label="Training Top-1 Accuracy", color="blue", marker="o"
    )
    plt.plot(
        epochs_range,
        val_acc,
        label="Validation Top-1 Accuracy",
        color="orange",
        marker="o",
    )
    
    # Plot Top-5 Accuracies (Dashed lines)
    plt.plot(
        epochs_range,
        top5_acc,
        label="Training Top-5 Accuracy",
        color="blue",
        linestyle="--",
        marker="x",
    )
    plt.plot(
        epochs_range,
        val_top5_acc,
        label="Validation Top-5 Accuracy",
        color="orange",
        linestyle="--",
        marker="x",
    )
    
    # Set chart attributes
    plt.title("Species Classification Accuracy Curves (184 Classes)")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy Score")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)
    
    # Display the plot
    plt.show()
    
    
    # Evaluate the model with test data
    # 1. Create the Test Dataset pipeline (Notice: NO .shuffle() step!)
    test_ds = tf.data.Dataset.from_tensor_slices(
        (test_df["image_path"].values, test_df["class_label_int"].values)
    )
    test_ds = (
        test_ds.map(load_and_preprocess, num_parallel_calls=AUTO)
        .cache()
        .batch(
            128
        )  # Match your training batch size or set to 128/256 for fast M3 parsing
        .prefetch(AUTO)
    )
    
    # 2. Evaluate the model on the unseen test dataset
    print("\n--- Evaluating Model on the Test Set ---")
    test_metrics = cnn_model.evaluate(test_ds)
    
    # 3. Print out the structured results neatly
    metric_names = cnn_model.metrics_names
    for name, value in zip(metric_names, test_metrics):
      print(f"Test {name}: {value:.4f}")
    
    
    return cnn_model, model_training_history


