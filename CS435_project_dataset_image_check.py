# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import tensorflow.keras
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
from typing import Optional

class ImageValidationError(Exception):
    def __init__(self, message, data):
        super().__init__(message)
        self.data = data

def image_check(
        image_path: str | Path, 
        label: Optional[str] = None, 
        min_brightness: float = 0.15, 
        max_brightness: float = 1.0
        ) -> bool:
    """
    Checks an image for the proper brightness range
    Returns: a boolean
    """
    try:
        full_path = "/Users/eusebiusballentine/Desktop/Education/CS435/leafsnap_dataset/leafsnap-dataset/" + image_path
        # read and decode image file
        image_raw = tf.io.read_file(full_path)
        image = tf.image.decode_jpeg(image_raw, channels=3)
        # normalize the pixel vals
        image_normalized = tf.image.convert_image_dtype(image, tf.float32)
        # calculate average brightness acroos all the pixels
        mean_brightness = tf.reduce_mean(image_normalized)
        # check if brightness is either too low or too great
        not_too_dark = tf.math.greater(mean_brightness, min_brightness)
        not_too_bright = tf.math.less(mean_brightness, max_brightness)
        # determine if image falls in acceptable range
        good_image = tf.logical_and(not_too_bright, not_too_dark)
    
        return bool(good_image.numpy())
    
    except Exception:
        return False
    
    
def execute_image_check(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the image_check function to all image paths in the dataframe and adds a column of boolean values indicating
    whether they are of acceptable brightness
    Returns: a pandas dataframe
    """
    df['good_image'] = df['image_path'].apply(image_check, label=None, min_brightness=0.15, max_brightness=1.0)
    return df


def load_and_validate_data(path: str) -> pd.DataFrame:
    """
    Loads data from a .txt file and employs the execute_image_check function to check all images in the dataset
    Returns: a pandas dataframe
    """
    df = pd.read_csv(path, sep='\t')
    execute_image_check(df)
    try:
        result = df['good_image'].all()
        if result:
            return True
        else:
            raise ImageValidationError("Image validation failed, some images brightness is unacceptable", data=df[~df['good_image']])
    except ImageValidationError as e:
        print(f"Error Message: {e}")
        print(f"Data For Review: {e.data}")

















