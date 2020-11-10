import tensorflow as tf
import numpy as np

from tensorflow.keras import layers

def dataset_split(batch_size, img_height, img_width, data):


    train_data = tf.keras.preprocessing.image_dataset_from_directory(
        data,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    test_data = tf.keras.preprocessing.image_dataset_from_directory(
        data,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    
    class_names = train_data.class_names
    print("Split Training and Testing Set. The classifiers are : ")
    print(class_names)
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_data = train_data.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    test_data = test_data.cache().prefetch(buffer_size=AUTOTUNE)

    return train_data,test_data
