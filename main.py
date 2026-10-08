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

# project modules
from CS435_project_dataset_image_check import load_and_validate_data
from CS435_project_model import define_compile_train_model

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

text_file_path = "/Users/eusebiusballentine/Desktop/Education/CS435/leafsnap_dataset/leafsnap-dataset/leafsnap-dataset-images.txt"

def main():
    print("Pipeline starting......")
    df = load_and_validate_data(text_file_path)
    if df:
        print("Data validated")
        print("Starting model training.....")
        cnn_model, model_training_history = define_compile_train_model()
        print("Finished model training")
    else:
        print("Aborted")

    
    
    
    
    

if __name__ == "__main__":
    # print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    main()